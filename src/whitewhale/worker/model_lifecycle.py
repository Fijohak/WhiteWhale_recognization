"""固定评估与不可变 Catalog 重建的租约隔离 Worker。"""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from whitewhale.platform.catalogs import build_flat_ip_index
from whitewhale.reid.embedding import make_embedder

from .client import ArtifactOutput, TaskLease


def make_model_lifecycle_handler(
    api,
    *,
    device: str = "auto",
    batch_size: int = 32,
    embedder_factory=make_embedder,
):
    def handle(lease: TaskLease) -> ArtifactOutput:
        manifest = _validate_manifest(lease)
        with tempfile.TemporaryDirectory(prefix="whitewhale-model-") as temp:
            workspace = Path(temp)
            weight_payload = api.download_input_artifact(
                lease, manifest["weight_artifact_id"])
            if hashlib.sha256(weight_payload).hexdigest() \
                    != manifest["model_sha256"]:
                raise ValueError("模型权重 SHA-256 不一致")
            weights = workspace / "weights.pt"
            weights.write_bytes(weight_payload)
            rows = (manifest["samples"] if lease.task_type == "fixed_evaluation"
                    else manifest["observations"])
            if lease.task_type == "fixed_evaluation" \
                    and manifest["feature_dim"] is None:
                payload = _detector_evaluation(
                    api, lease, manifest, rows, weights, workspace)
                return ArtifactOutput(
                    artifact_type="evaluation_report", data=payload,
                    schema_version=int(manifest["schema_version"]),
                    pipeline_config_digest=manifest["config_digest"],
                    row_binding_digest=manifest["row_binding_digest"],
                    model_version=manifest["model_version"],
                    preprocess_id=manifest["preprocess_id"],
                )
            if manifest["feature_dim"] is None:
                raise ValueError("Catalog 重建要求 Embedding 模型特征维度")
            crop_paths: dict[str, Path] = {}
            crops = workspace / "crops"
            crops.mkdir()
            for row in rows:
                crop_id = str(row["crop_id"])
                if crop_id in crop_paths:
                    raise ValueError("模型任务包含重复 Crop")
                path = crops / f"{crop_id}.jpg"
                path.write_bytes(api.download_input_crop(lease, crop_id))
                crop_paths[crop_id] = path
            model = embedder_factory(
                manifest["model_family"], metric_ckpt=weights,
                device=device)
            vectors = _normalize(model.encode_paths(
                [crop_paths[str(row["crop_id"])] for row in rows],
                batch_size=batch_size))
            if vectors.shape[1] != int(manifest["feature_dim"]):
                raise ValueError("模型输出特征维度与 Manifest 不一致")
            if lease.task_type == "fixed_evaluation":
                baseline_vectors = None
                production = manifest.get("production_model")
                if production:
                    baseline_payload = api.download_input_artifact(
                        lease, production["weight_artifact_id"])
                    if hashlib.sha256(baseline_payload).hexdigest() \
                            != production["model_sha256"]:
                        raise ValueError("Production 权重 SHA-256 不一致")
                    baseline_path = workspace / "production-weights.pt"
                    baseline_path.write_bytes(baseline_payload)
                    baseline = embedder_factory(
                        production["model_family"],
                        metric_ckpt=baseline_path, device=device)
                    baseline_vectors = _normalize(baseline.encode_paths(
                        [crop_paths[str(row["crop_id"])] for row in rows],
                        batch_size=batch_size))
                    if baseline_vectors.shape[1] != int(
                            production["feature_dim"]):
                        raise ValueError("Production 模型特征维度不一致")
                payload = _evaluation_report(
                    manifest, rows, vectors, baseline_vectors)
                artifact_type = "evaluation_report"
            else:
                payload = _catalog_archive(manifest, rows, vectors)
                artifact_type = "catalog_rebuild"
            return ArtifactOutput(
                artifact_type=artifact_type,
                data=payload,
                schema_version=int(manifest["schema_version"]),
                pipeline_config_digest=manifest["config_digest"],
                row_binding_digest=manifest["row_binding_digest"],
                model_version=manifest["model_version"],
                preprocess_id=manifest["preprocess_id"],
            )
    return handle


def _validate_manifest(lease: TaskLease) -> dict:
    if lease.task_type not in {"fixed_evaluation", "catalog_rebuild"}:
        raise ValueError("不是模型生命周期任务")
    manifest = lease.input_manifest
    required = {
        "schema_version", "model_version", "model_family", "model_sha256",
        "weight_artifact_id", "feature_dim", "preprocess_id",
        "config_digest", "row_binding_digest",
    }
    rows_key = "samples" if lease.task_type == "fixed_evaluation" \
        else "observations"
    if not isinstance(manifest, dict) or required - set(manifest) \
            or not isinstance(manifest.get(rows_key), list) \
            or not manifest[rows_key]:
        raise ValueError("模型任务 Manifest 不完整")
    if lease.required_model_version != manifest["model_version"]:
        raise ValueError("租约与模型版本不一致")
    for key in ("model_sha256", "config_digest", "row_binding_digest"):
        if len(str(manifest[key])) != 64:
            raise ValueError(f"{key} 摘要格式无效")
    return manifest


def _normalize(vectors: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("Embedding 必须是非空有限二维矩阵")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Embedding 包含零向量")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _evaluation_report(
    manifest: dict,
    rows: list[dict],
    vectors: np.ndarray,
    baseline_vectors: np.ndarray | None,
) -> bytes:
    calibration = [index for index, row in enumerate(rows)
                   if row["split"] == "calibration"]
    test = [index for index, row in enumerate(rows) if row["split"] == "test"]
    if not calibration or len(test) < 2:
        raise ValueError("固定评估需要 calibration 和至少两个 test 样本")
    genuine: list[float] = []
    impostor: list[float] = []
    for position, left in enumerate(calibration):
        for right in calibration[position + 1:]:
            score = float(vectors[left] @ vectors[right])
            target = (genuine if rows[left]["individual_id"]
                      == rows[right]["individual_id"] else impostor)
            target.append(score)
    if not genuine or not impostor:
        raise ValueError("calibration 必须同时包含同体与异体配对")
    accept = _best_threshold(genuine, impostor)
    uncertain = min(accept, float(np.quantile(impostor, 0.95)))

    metrics = _retrieval_metrics(rows, vectors, test)
    comparison = dict(manifest["production_comparison"])
    if baseline_vectors is not None:
        baseline_metrics = _retrieval_metrics(rows, baseline_vectors, test)
        comparison.update({
            "baseline_rank1": baseline_metrics["rank1"],
            "baseline_map": baseline_metrics["map"],
            "candidate_rank1": metrics["rank1"],
            "candidate_map": metrics["map"],
            "rank1_delta": metrics["rank1"] - baseline_metrics["rank1"],
            "map_delta": metrics["map"] - baseline_metrics["map"],
        })
    else:
        comparison.update({
            "baseline_rank1": None,
            "baseline_map": None,
            "candidate_rank1": metrics["rank1"],
            "candidate_map": metrics["map"],
            "rank1_delta": None,
            "map_delta": None,
        })
    report = {
        "schema_version": 1,
        "evaluation_run_id": manifest["evaluation_run_id"],
        "model_version_id": manifest["model_version_id"],
        "model_sha256": manifest["model_sha256"],
        "dataset_version_id": manifest["dataset_version_id"],
        "row_binding_digest": manifest["row_binding_digest"],
        "metrics": metrics,
        "calibrated_thresholds": {
            "accept": accept,
            "uncertain": uncertain,
        },
        "production_comparison": comparison,
    }
    return json.dumps(report, sort_keys=True, separators=(",", ":")).encode()


def _retrieval_metrics(
    rows: list[dict], vectors: np.ndarray, test: list[int],
) -> dict[str, float]:
    hits = 0
    reciprocal_ranks: list[float] = []
    evaluable = 0
    for query in test:
        candidates = [index for index in test if index != query and
                      rows[index]["sequence_key"] != rows[query]["sequence_key"]]
        positives = {index for index in candidates
                     if rows[index]["individual_id"]
                     == rows[query]["individual_id"]}
        if not positives:
            continue
        ranked = sorted(candidates, key=lambda index:
                        float(vectors[query] @ vectors[index]), reverse=True)
        evaluable += 1
        hits += int(ranked[0] in positives)
        reciprocal_ranks.append(1.0 / min(
            rank for rank, index in enumerate(ranked, 1)
            if index in positives))
    if evaluable == 0:
        raise ValueError("test 没有可跨 Sequence 评估的同体查询")
    return {
        "rank1": hits / evaluable,
        "map": float(np.mean(reciprocal_ranks)),
        "evaluable_queries": float(evaluable),
    }


def _best_threshold(genuine: list[float], impostor: list[float]) -> float:
    candidates = sorted(set(genuine + impostor))
    return max(candidates, key=lambda threshold: (
        sum(score >= threshold for score in genuine) / len(genuine)
        - sum(score >= threshold for score in impostor) / len(impostor),
        threshold,
    ))


def _detector_evaluation(
    api,
    lease: TaskLease,
    manifest: dict,
    rows: list[dict],
    weights: Path,
    workspace: Path,
) -> bytes:
    from ultralytics import YOLO

    images_dir = workspace / "images"
    images_dir.mkdir()
    image_paths: dict[str, Path] = {}
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        image_id = str(row["image_id"])
        grouped.setdefault(image_id, []).append(row)
        if image_id not in image_paths:
            payload = api.download_input_image(lease, image_id)
            if hashlib.sha256(payload).hexdigest() != row["image_sha256"]:
                raise ValueError("Detector 评估图片 SHA-256 不一致")
            path = images_dir / f"{image_id}.jpg"
            path.write_bytes(payload)
            image_paths[image_id] = path
    model = YOLO(str(weights))
    predictions = _predict_detector(model, image_paths)
    calibration_ids = {str(row["image_id"]) for row in rows
                       if row["split"] == "calibration"}
    test_ids = {str(row["image_id"]) for row in rows
                if row["split"] == "test"}
    accept = _calibrate_detector_threshold(
        grouped, predictions, calibration_ids)
    metrics = _detector_metrics(grouped, predictions, test_ids, accept)
    comparison = dict(manifest["production_comparison"])
    production = manifest.get("production_model")
    if production:
        payload = api.download_input_artifact(
            lease, production["weight_artifact_id"])
        if hashlib.sha256(payload).hexdigest() != production["model_sha256"]:
            raise ValueError("Production Detector 权重 SHA-256 不一致")
        baseline_path = workspace / "production-detector.pt"
        baseline_path.write_bytes(payload)
        baseline_predictions = _predict_detector(
            YOLO(str(baseline_path)), image_paths)
        baseline = _detector_metrics(
            grouped, baseline_predictions, test_ids, accept)
        comparison.update({
            "baseline_f1": baseline["f1"],
            "candidate_f1": metrics["f1"],
            "f1_delta": metrics["f1"] - baseline["f1"],
        })
    else:
        comparison.update({
            "baseline_f1": None,
            "candidate_f1": metrics["f1"],
            "f1_delta": None,
        })
    return json.dumps({
        "schema_version": 1,
        "evaluation_run_id": manifest["evaluation_run_id"],
        "model_version_id": manifest["model_version_id"],
        "model_sha256": manifest["model_sha256"],
        "dataset_version_id": manifest["dataset_version_id"],
        "row_binding_digest": manifest["row_binding_digest"],
        "metrics": metrics,
        "calibrated_thresholds": {
            "accept": accept,
            "uncertain": max(0.0, accept * 0.8),
        },
        "production_comparison": comparison,
    }, sort_keys=True, separators=(",", ":")).encode()


def _predict_detector(model, image_paths: dict[str, Path]) -> dict[str, list[tuple]]:
    ordered = list(image_paths)
    results = model.predict(
        source=[str(image_paths[value]) for value in ordered],
        conf=0.001, verbose=False)
    output: dict[str, list[tuple]] = {}
    for image_id, result in zip(ordered, results, strict=True):
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confidence = result.boxes.conf.detach().cpu().numpy()
        output[image_id] = [
            (tuple(float(value) for value in box), float(score))
            for box, score in zip(xyxy, confidence, strict=True)
        ]
    return output


def _ground_truth(rows: list[dict]) -> list[tuple[float, float, float, float]]:
    boxes = []
    for row in rows:
        x, y, width, height = (float(value) for value in row["bbox"])
        boxes.append((x, y, x + width, y + height))
    return boxes


def _iou(left: tuple, right: tuple) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _calibrate_detector_threshold(
    grouped: dict[str, list[dict]],
    predictions: dict[str, list[tuple]],
    image_ids: set[str],
) -> float:
    scores = sorted({score for image_id in image_ids
                     for _, score in predictions[image_id]})
    if not scores:
        raise ValueError("calibration 没有 Detector 候选框")
    return max(scores, key=lambda threshold: (
        _detector_metrics(grouped, predictions, image_ids, threshold)["f1"],
        threshold,
    ))


def _detector_metrics(
    grouped: dict[str, list[dict]],
    predictions: dict[str, list[tuple]],
    image_ids: set[str],
    threshold: float,
) -> dict[str, float]:
    true_positive = false_positive = false_negative = 0
    for image_id in image_ids:
        ground_truth = _ground_truth(grouped[image_id])
        unmatched = set(range(len(ground_truth)))
        candidates = sorted(
            (item for item in predictions[image_id] if item[1] >= threshold),
            key=lambda item: item[1], reverse=True)
        for box, _ in candidates:
            matches = [(index, _iou(box, ground_truth[index]))
                       for index in unmatched]
            best = max(matches, key=lambda item: item[1], default=None)
            if best is not None and best[1] >= 0.5:
                true_positive += 1
                unmatched.remove(best[0])
            else:
                false_positive += 1
        false_negative += len(unmatched)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision_iou50": precision,
        "recall_iou50": recall,
        "f1": f1,
    }


def _catalog_archive(manifest: dict, rows: list[dict], vectors: np.ndarray) -> bytes:
    observation_ids = [str(row["observation_id"]) for row in rows]
    archive_manifest = {
        "schema_version": int(manifest["schema_version"]),
        "model_version_id": manifest["model_version_id"],
        "model_version": manifest["model_version"],
        "model_sha256": manifest["model_sha256"],
        "feature_dim": int(manifest["feature_dim"]),
        "preprocess_id": manifest["preprocess_id"],
        "observation_ids": observation_ids,
        "row_binding_digest": manifest["row_binding_digest"],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        embeddings = io.BytesIO()
        np.save(embeddings, vectors, allow_pickle=False)
        archive.writestr("embeddings.npy", embeddings.getvalue())
        archive.writestr("index.faiss", build_flat_ip_index(vectors))
        archive.writestr("manifest.json", json.dumps(
            archive_manifest, sort_keys=True, separators=(",", ":")))
    return buffer.getvalue()
