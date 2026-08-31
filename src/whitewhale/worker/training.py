"""租约隔离的 GPU 训练执行器；测试集在下载前再次拒绝。"""
from __future__ import annotations

import io
import json
import re
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import pandas as pd
from PIL import Image

from .client import ArtifactOutput, TaskLease


class TrainingWorkerApi(Protocol):
    def download_input_image(self, lease: TaskLease, image_id: str) -> bytes: ...
    def download_input_crop(self, lease: TaskLease, crop_id: str) -> bytes: ...
    def download_input_artifact(
        self, lease: TaskLease, artifact_id: str,
    ) -> bytes: ...
    def submit(
        self, lease: TaskLease, artifact: ArtifactOutput,
    ) -> str: ...
    def register_checkpoint(
        self, lease: TaskLease, artifact_id: str, *,
        stage: int, epoch: int, step: int,
    ) -> str: ...


CheckpointCallback = Callable[..., None]
TrainingExecutor = Callable[
    [dict, dict[str, Path], Path, CheckpointCallback],
    list[ArtifactOutput],
]
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


@dataclass(frozen=True)
class WorkerTrainingConfig:
    detector_base_weights: Path | None = None
    device: str = "auto"


def make_training_handler(
    api: TrainingWorkerApi,
    *,
    executor: TrainingExecutor | None = None,
    worker_config: WorkerTrainingConfig | None = None,
):
    config = worker_config or WorkerTrainingConfig()
    execute = executor or (
        lambda manifest, paths, workspace, callback:
        execute_platform_training(
            manifest, paths, workspace, callback, worker_config=config)
    )

    def handler(lease: TaskLease) -> list[ArtifactOutput]:
        if lease.task_type not in {"reid_training", "detector_training"}:
            raise ValueError(f"不是训练任务: {lease.task_type}")
        manifest = dict(lease.input_manifest)
        samples = manifest.get("samples")
        config_digest = manifest.get("config_digest")
        if not isinstance(samples, list) or not samples \
                or not isinstance(config_digest, str) \
                or len(config_digest) != 64:
            raise ValueError("训练输入 Manifest 不完整")
        if any(sample.get("split") == "test" for sample in samples):
            raise ValueError("训练 Worker 拒绝下载冻结 test 样本")
        if lease.task_type == "reid_training":
            input_ids = [str(sample.get("crop_id", ""))
                         for sample in samples]
            if len(set(input_ids)) != len(input_ids):
                raise ValueError("Re-ID 训练输入包含重复 Crop ID")
        else:
            input_ids = list(dict.fromkeys(
                str(sample.get("image_id", "")) for sample in samples))
        if any(not _SAFE_ID.fullmatch(value) for value in input_ids):
            raise ValueError("训练输入包含非法 ID")

        with tempfile.TemporaryDirectory(prefix="whitewhale-training-") as tmp:
            workspace = Path(tmp)
            inputs = workspace / "inputs"
            inputs.mkdir()
            sample_paths: dict[str, Path] = {}
            for input_id in input_ids:
                path = inputs / f"{input_id}.jpg"
                payload = (api.download_input_crop(lease, input_id)
                           if lease.task_type == "reid_training"
                           else api.download_input_image(lease, input_id))
                path.write_bytes(payload)
                sample_paths[input_id] = path

            resume = manifest.get("resume")
            if resume is not None:
                artifact_id = str(resume.get("artifact_id", ""))
                expected_sha256 = str(resume.get("sha256", ""))
                if not _SAFE_ID.fullmatch(artifact_id) \
                        or len(expected_sha256) != 64:
                    raise ValueError("恢复点 Manifest 无效")
                payload = api.download_input_artifact(lease, artifact_id)
                import hashlib
                if hashlib.sha256(payload).hexdigest() != expected_sha256:
                    raise ValueError("恢复点 SHA-256 校验失败")
                resume_path = workspace / "resume.pt"
                resume_path.write_bytes(payload)
                manifest["local_resume_path"] = str(resume_path)

            def upload_checkpoint(
                data: bytes, *, stage: int, epoch: int, step: int,
            ) -> None:
                if stage < 0 or epoch < 0 or step < 0 or not data:
                    raise ValueError("训练 Checkpoint 元数据无效")
                artifact_id = api.submit(lease, ArtifactOutput(
                    artifact_type="training_checkpoint",
                    data=data,
                    schema_version=1,
                    pipeline_config_digest=config_digest,
                    row_binding_digest=manifest.get("row_binding_digest"),
                ))
                api.register_checkpoint(
                    lease, artifact_id,
                    stage=stage, epoch=epoch, step=step)

            outputs = execute(
                manifest, sample_paths, workspace, upload_checkpoint)
            if not outputs or not any(
                    item.artifact_type == "model_weights" for item in outputs):
                raise ValueError("训练执行器没有产生 model_weights")
            return outputs

    return handler


def execute_platform_training(
    manifest: dict,
    sample_paths: dict[str, Path],
    workspace: Path,
    checkpoint_callback: CheckpointCallback,
    *,
    worker_config: WorkerTrainingConfig | None = None,
) -> list[ArtifactOutput]:
    """生产执行入口；按任务类型延迟加载 CUDA 训练依赖。"""
    task_type = manifest.get("task_type")
    if task_type == "reid_training":
        return _execute_reid(
            manifest, sample_paths, workspace, checkpoint_callback)
    if task_type == "detector_training":
        return _execute_detector(
            manifest, sample_paths, workspace, checkpoint_callback,
            worker_config or WorkerTrainingConfig())
    raise ValueError(f"未知训练任务类型: {task_type}")


def _execute_reid(
    manifest: dict,
    sample_paths: dict[str, Path],
    workspace: Path,
    checkpoint_callback: CheckpointCallback,
) -> list[ArtifactOutput]:
    from whitewhale.reid.training import run_training

    samples = manifest["samples"]
    rows = []
    identities = sorted({str(item["individual_id"]) for item in samples})
    labels = {identity: index for index, identity in enumerate(identities)}
    for item in samples:
        crop_id = str(item["crop_id"])
        identity = str(item["individual_id"])
        rows.append({
            "path": str(sample_paths[crop_id]),
            "confirmed_identity": identity,
            "identity_unit": identity,
            "label_idx": labels[identity],
            "session_id": "platform_verified",
            "series_unit": str(item["sequence_key"]),
            "encounter_key": str(item["encounter_key"]),
            "duplicate_group": str(item["duplicate_group"]),
            "frozen_split": str(item["split"]),
            "image_id": str(item["image_id"]),
        })
    frame = pd.DataFrame(rows)
    config = manifest.get("config", {})
    output = workspace / "output"
    output.mkdir()

    def on_checkpoint(
        path: Path, *, stage: int, epoch: int, step: int,
    ) -> None:
        checkpoint_callback(
            path.read_bytes(), stage=stage, epoch=epoch, step=step)

    args = SimpleNamespace(
        val_n=1,
        seed=int(manifest["seed"]),
        test_session=[],
        epochs_stage1=int(config.get("epochs_stage1", 20)),
        epochs_stage2=int(config.get("epochs_stage2", 25)),
        lr_head=float(config.get("lr_head", 1e-3)),
        lr_backbone=float(config.get("lr_backbone", 5e-6)),
        batch=int(config.get("batch_size", config.get("batch", 16))),
        hard_negative=bool(config.get("hard_negative", False)),
        init_ckpt=manifest.get("local_resume_path"),
        batches_per_epoch=int(config.get("batches_per_epoch", 40)),
        lambda_hn=float(config.get("lambda_hn", 0.2)),
        checkpoint_callback=on_checkpoint,
    )
    run_training(args, frame, output)
    weights = (output / "best.pt").read_bytes()
    report_buffer = io.BytesIO()
    with zipfile.ZipFile(
            report_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("metrics.json", "history.csv"):
            path = output / name
            if path.is_file():
                archive.writestr(name, path.read_bytes())
        archive.writestr("worker_manifest.json", json.dumps(
            manifest, sort_keys=True, separators=(",", ":")))
    common = {
        "schema_version": 1,
        "pipeline_config_digest": manifest["config_digest"],
        "row_binding_digest": manifest.get("row_binding_digest"),
    }
    return [
        ArtifactOutput(
            artifact_type="model_weights", data=weights, **common),
        ArtifactOutput(
            artifact_type="training_report",
            data=report_buffer.getvalue(), **common),
    ]


def _execute_detector(
    manifest: dict,
    sample_paths: dict[str, Path],
    workspace: Path,
    checkpoint_callback: CheckpointCallback,
    worker_config: WorkerTrainingConfig,
) -> list[ArtifactOutput]:
    from ultralytics import YOLO

    base_weights = (Path(manifest["local_resume_path"])
                    if manifest.get("local_resume_path")
                    else worker_config.detector_base_weights)
    if base_weights is None or not base_weights.is_file():
        raise ValueError("Detector 训练要求 Worker 配置本地基础权重")
    samples = manifest["samples"]
    dataset_root = workspace / "detector_dataset"
    grouped: dict[tuple[str, str], list[list[float]]] = {}
    for sample in samples:
        split = str(sample["split"])
        if split == "calibration":
            continue
        if split not in {"train", "val"}:
            raise ValueError(f"Detector 训练收到非法 Split: {split}")
        image_id = str(sample["image_id"])
        bbox = sample.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Detector 训练样本缺少 bbox")
        grouped.setdefault((split, image_id), []).append([
            float(value) for value in bbox])
    if not grouped or not {split for split, _ in grouped}.issuperset(
            {"train", "val"}):
        raise ValueError("Detector Dataset 必须包含 train 和 val")

    for (split, image_id), boxes in grouped.items():
        source = sample_paths[image_id]
        with Image.open(source) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Detector 输入图片尺寸无效")
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        target = image_dir / f"{image_id}.jpg"
        shutil.copy2(source, target)
        labels = []
        for x, y, box_width, box_height in boxes:
            if box_width <= 0 or box_height <= 0:
                raise ValueError("Detector bbox 尺寸无效")
            center_x = (x + box_width / 2) / width
            center_y = (y + box_height / 2) / height
            normalized_width = box_width / width
            normalized_height = box_height / height
            values = (center_x, center_y, normalized_width, normalized_height)
            if any(value < 0 or value > 1 for value in values):
                raise ValueError("Detector bbox 超出图片边界")
            labels.append("0 " + " ".join(f"{value:.8f}" for value in values))
        (label_dir / f"{image_id}.txt").write_text(
            "\n".join(labels) + "\n", encoding="utf-8")

    data_file = workspace / "detector_dataset.yaml"
    data_file.write_text(json.dumps({
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "dolphin"},
    }), encoding="utf-8")
    config = manifest.get("config", {})
    run_dir = workspace / "detector_run"
    model = YOLO(str(base_weights))
    model.train(
        data=str(data_file),
        project=str(run_dir), name="train", exist_ok=True,
        epochs=int(config.get("epochs", 50)),
        imgsz=int(config.get("imgsz", 1024)),
        batch=int(config.get("batch_size", config.get("batch", 8))),
        device=worker_config.device,
        seed=int(manifest["seed"]),
        pretrained=True,
    )
    weights_dir = run_dir / "train" / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    if not best.is_file():
        raise RuntimeError("YOLO 训练未产生 best.pt")
    if last.is_file():
        checkpoint_callback(
            last.read_bytes(), stage=1,
            epoch=int(config.get("epochs", 50)), step=0)
    report_buffer = io.BytesIO()
    with zipfile.ZipFile(
            report_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("results.csv", "args.yaml"):
            path = run_dir / "train" / name
            if path.is_file():
                archive.writestr(name, path.read_bytes())
        archive.writestr("worker_manifest.json", json.dumps(
            manifest, sort_keys=True, separators=(",", ":")))
    common = {
        "schema_version": 1,
        "pipeline_config_digest": manifest["config_digest"],
        "row_binding_digest": manifest.get("row_binding_digest"),
    }
    return [
        ArtifactOutput(
            artifact_type="model_weights", data=best.read_bytes(), **common),
        ArtifactOutput(
            artifact_type="training_report",
            data=report_buffer.getvalue(), **common),
    ]
