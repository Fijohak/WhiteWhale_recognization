"""租约 Worker 的检测、Embedding、HDBSCAN 批次归档处理器。"""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from whitewhale.detection.detector import detect_all_and_crop
from whitewhale.pipeline.archival import cluster_embeddings
from whitewhale.reid.embedding import extract_embeddings, make_embedder

from .client import ArtifactOutput, TaskLease


@dataclass(frozen=True)
class ArchivalWorkerConfig:
    detector_weights: Path
    reid_checkpoint: Path
    device: str = "auto"
    detector_conf: float = 0.25
    detector_imgsz: int = 1024
    pad_x: float = 0.30
    pad_up: float = 0.15
    pad_down: float = 0.60
    batch_size: int = 32


def make_batch_archival_handler(
    api,
    config: ArchivalWorkerConfig,
    *,
    detect_fn: Callable = detect_all_and_crop,
    embedder_factory: Callable = make_embedder,
    extract_fn: Callable = extract_embeddings,
    cluster_fn: Callable = cluster_embeddings,
):
    def handle(lease: TaskLease) -> ArtifactOutput:
        manifest = _validate_input_manifest(lease)
        with tempfile.TemporaryDirectory(prefix="whitewhale-archive-") as temp:
            root = Path(temp)
            images_root = root / "images"
            crops_root = root / "crops"
            images_root.mkdir()
            rows = []
            for item in manifest["images"]:
                image_id = str(item["image_id"])
                payload = api.download_input_image(lease, image_id)
                if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                    raise ValueError(f"输入图片 {image_id} SHA-256 不一致")
                suffix = Path(str(item["original_relative_path"])).suffix.lower()
                suffix = suffix if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} else ".jpg"
                local_name = f"{image_id}{suffix}"
                (images_root / local_name).write_bytes(payload)
                rows.append({
                    "image_id": image_id,
                    "relative_path": local_name,
                    "session_id": str(manifest["batch_id"]),
                })
            source = pd.DataFrame(rows)
            detected = detect_fn(
                source,
                images_root,
                crops_root,
                weights=config.detector_weights,
                conf=config.detector_conf,
                imgsz=config.detector_imgsz,
                device=config.device,
                pad_x=config.pad_x,
                pad_up=config.pad_up,
                pad_down=config.pad_down,
                preview=False,
            )
            detected = detected.copy().reset_index(drop=True)
            if "crop_index" not in detected.columns:
                detected["crop_index"] = detected.groupby(
                    "image_id", sort=False).cumcount()
            if "crop_key" not in detected.columns:
                counts = detected.groupby("image_id")["image_id"].transform("size")
                detected["crop_key"] = [
                    str(image_id) if count == 1 else
                    f"{image_id}--{int(index):03d}"
                    for image_id, index, count in zip(
                        detected["image_id"], detected["crop_index"], counts,
                        strict=True)
                ]
            observed_images = list(dict.fromkeys(
                detected["image_id"].astype(str)))
            if observed_images != [item["image_id"] for item in rows]:
                raise ValueError("检测结果行序与服务器图片清单不一致")
            embedder = embedder_factory(
                "metric-learning",
                metric_ckpt=config.reid_checkpoint,
                device=config.device,
            )
            actual_preprocess = getattr(embedder, "preprocess_id", None)
            if actual_preprocess is not None \
                    and actual_preprocess != manifest["preprocess_id"]:
                raise ValueError("Worker Re-ID 预处理协议与任务不一致")
            embedding_rows = detected.copy()
            embedding_rows["source_image_id"] = embedding_rows["image_id"]
            embedding_rows["image_id"] = embedding_rows["crop_key"]
            vectors, _ = extract_fn(
                embedding_rows,
                embedder,
                crops_dir=crops_root,
                missing="error",
                batch_size=config.batch_size,
            )
            vectors = _normalize(vectors)
            labels, probabilities = cluster_fn(
                vectors, int(manifest["min_cluster_size"]))
            output_manifest = _build_output_manifest(
                manifest, detected, labels, probabilities)
            archive_bytes = _build_archive(
                output_manifest, vectors, crops_root)
            return ArtifactOutput(
                artifact_type="batch_archival",
                data=archive_bytes,
                schema_version=1,
                pipeline_config_digest=manifest["pipeline_config_digest"],
                model_version=manifest["model_version"],
                row_binding_digest=output_manifest["row_binding_digest"],
                detector_version=manifest["detector_version"],
                preprocess_id=manifest["preprocess_id"],
            )
    return handle


def _validate_input_manifest(lease: TaskLease) -> dict:
    manifest = lease.input_manifest
    required = {
        "schema_version", "batch_id", "model_version", "detector_version",
        "preprocess_id", "pipeline_config_digest", "min_cluster_size", "images",
    }
    if not isinstance(manifest, dict) or required - set(manifest):
        raise ValueError("批次归档任务清单缺少必需字段")
    if manifest["schema_version"] != 1 or not manifest["images"]:
        raise ValueError("批次归档任务 Schema 或图片清单无效")
    if lease.required_model_version != manifest["model_version"]:
        raise ValueError("租约 Model Version 与任务清单不一致")
    if int(manifest["min_cluster_size"]) < 2:
        raise ValueError("min_cluster_size 必须至少为 2")
    image_ids = [str(item.get("image_id", "")) for item in manifest["images"]]
    if any(not value for value in image_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError("任务图片 ID 为空或重复")
    for item in manifest["images"]:
        digest = str(item.get("sha256", ""))
        if len(digest) != 64:
            raise ValueError("任务图片 SHA-256 格式错误")
    return manifest


def _normalize(vectors: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0 \
            or not np.isfinite(matrix).all():
        raise ValueError("Worker Embedding 必须是非空有限二维数组")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Worker Embedding 包含零向量")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _build_output_manifest(
    source: dict,
    detected: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    if len(detected) != len(labels) or len(labels) != len(probabilities):
        raise ValueError("聚类结果与 Crop 行数不一致")
    crop_keys = detected["crop_key"].astype(str).tolist()
    crops = [{
        "key": key,
        "image_id": str(row.image_id),
        "crop_index": int(row.crop_index),
        "bbox": [int(row.x), int(row.y), int(row.w), int(row.h)],
        "path": f"crops/{key}.jpg",
        "quality": float(row.det_conf),
        "fallback": bool(row.fallback),
    } for key, row in zip(crop_keys, detected.itertuples(), strict=True)]
    clusters = []
    for label in sorted(set(int(value) for value in labels if int(value) >= 0)):
        positions = np.flatnonzero(labels == label)
        representative = int(positions[np.argmax(probabilities[positions])])
        clusters.append({
            "label": f"cluster-{label}",
            "member_keys": [crop_keys[index] for index in positions],
            "membership_scores": [float(probabilities[index]) for index in positions],
            "representative_key": crop_keys[representative],
            "matches": [],
        })
    for position in np.flatnonzero(labels == -1):
        clusters.append({
            "label": f"noise-{int(position):06d}",
            "member_keys": [crop_keys[int(position)]],
            "membership_scores": [float(probabilities[int(position)])],
            "representative_key": crop_keys[int(position)],
            "matches": [],
        })
    digest = hashlib.sha256(json.dumps(
        crop_keys, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "batch_id": source["batch_id"],
        "model_version": source["model_version"],
        "detector_version": source["detector_version"],
        "preprocess_id": source["preprocess_id"],
        "pipeline_config_digest": source["pipeline_config_digest"],
        "row_binding_digest": digest,
        "crops": crops,
        "clusters": clusters,
    }


def _build_archive(manifest: dict, vectors: np.ndarray, crops_root: Path) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")),
        )
        vector_file = io.BytesIO()
        np.save(vector_file, vectors, allow_pickle=False)
        archive.writestr("embeddings.npy", vector_file.getvalue())
        for crop in manifest["crops"]:
            archive.write(
                crops_root / f"{crop['key']}.jpg",
                arcname=crop["path"],
            )
    return output.getvalue()
