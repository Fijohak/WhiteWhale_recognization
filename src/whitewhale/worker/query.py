"""租约 Worker 的单图或批量查询检测与 Embedding 处理器。"""
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
from whitewhale.reid.embedding import extract_embeddings, make_embedder

from .client import ArtifactOutput, TaskLease


@dataclass(frozen=True)
class QueryWorkerConfig:
    detector_weights: Path
    reid_checkpoint: Path
    device: str = "auto"
    detector_conf: float = 0.25
    detector_imgsz: int = 1024
    pad_x: float = 0.30
    pad_up: float = 0.15
    pad_down: float = 0.60
    batch_size: int = 32


def make_query_handler(
    api,
    config: QueryWorkerConfig,
    *,
    detect_fn: Callable = detect_all_and_crop,
    embedder_factory: Callable = make_embedder,
    extract_fn: Callable = extract_embeddings,
):
    def handle(lease: TaskLease) -> ArtifactOutput:
        manifest = _validate_manifest(lease)
        with tempfile.TemporaryDirectory(prefix="whitewhale-query-") as temp:
            root = Path(temp)
            images_root = root / "images"
            crops_root = root / "crops"
            images_root.mkdir()
            rows = []
            for item in manifest["images"]:
                image_id = str(item["query_image_id"])
                payload = api.download_query_image(lease, image_id)
                if len(payload) != int(item["size_bytes"]) \
                        or hashlib.sha256(payload).hexdigest() != item["sha256"]:
                    raise ValueError(f"输入图片 {image_id} SHA-256 或大小不一致")
                suffix = Path(str(item["original_relative_path"])).suffix.lower()
                if suffix not in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
                }:
                    suffix = ".jpg"
                local_name = f"{image_id}{suffix}"
                (images_root / local_name).write_bytes(payload)
                rows.append({
                    "image_id": image_id,
                    "relative_path": local_name,
                    "session_id": str(manifest["query_request_id"]),
                })

            detected = detect_fn(
                pd.DataFrame(rows), images_root, crops_root,
                weights=config.detector_weights,
                conf=config.detector_conf,
                imgsz=config.detector_imgsz,
                device=config.device,
                pad_x=config.pad_x,
                pad_up=config.pad_up,
                pad_down=config.pad_down,
                preview=False,
            ).copy().reset_index(drop=True)
            if detected.empty:
                vectors = np.empty(
                    (0, int(manifest["feature_dim"])), dtype=np.float32)
            else:
                valid_image_ids = {item["image_id"] for item in rows}
                if not set(detected["image_id"].astype(str)).issubset(
                        valid_image_ids):
                    raise ValueError("检测结果包含查询清单之外的图片")
                if "crop_index" not in detected.columns:
                    detected["crop_index"] = detected.groupby(
                        "image_id", sort=False).cumcount()
                if "crop_key" not in detected.columns:
                    counts = detected.groupby(
                        "image_id")["image_id"].transform("size")
                    detected["crop_key"] = [
                        str(image_id) if count == 1 else
                        f"{image_id}--{int(index):03d}"
                        for image_id, index, count in zip(
                            detected["image_id"], detected["crop_index"],
                            counts, strict=True)
                    ]
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
                    embedding_rows, embedder, crops_dir=crops_root,
                    missing="error", batch_size=config.batch_size)
                vectors = _normalize(vectors, int(manifest["feature_dim"]))

            output_manifest = _build_output_manifest(manifest, detected)
            return ArtifactOutput(
                artifact_type="query_embedding",
                data=_build_archive(output_manifest, vectors),
                schema_version=1,
                pipeline_config_digest=manifest["pipeline_config_digest"],
                model_version=manifest["model_version"],
                row_binding_digest=manifest["row_binding_digest"],
                detector_version=manifest["detector_version"],
                preprocess_id=manifest["preprocess_id"],
            )

    return handle


def _validate_manifest(lease: TaskLease) -> dict:
    manifest = lease.input_manifest
    required = {
        "schema_version", "query_request_id", "catalog_id", "model_version",
        "detector_version", "preprocess_id", "feature_dim", "top_k",
        "pipeline_config_digest", "row_binding_digest", "images",
    }
    if not isinstance(manifest, dict) or required - set(manifest):
        raise ValueError("查询任务清单缺少必需字段")
    if manifest["schema_version"] != 1 or not manifest["images"]:
        raise ValueError("查询任务 Schema 或图片清单无效")
    if lease.required_model_version != manifest["model_version"]:
        raise ValueError("租约 Model Version 与查询清单不一致")
    if int(manifest["feature_dim"]) <= 0 or int(manifest["top_k"]) <= 0:
        raise ValueError("查询向量维数或 Top-K 无效")
    image_ids = [
        str(item.get("query_image_id", "")) for item in manifest["images"]]
    if any(not value for value in image_ids) \
            or len(image_ids) != len(set(image_ids)):
        raise ValueError("查询图片 ID 为空或重复")
    for field in ("pipeline_config_digest", "row_binding_digest"):
        if len(str(manifest[field])) != 64:
            raise ValueError(f"{field} 格式错误")
    for item in manifest["images"]:
        if len(str(item.get("sha256", ""))) != 64 \
                or int(item.get("size_bytes", -1)) < 0:
            raise ValueError("查询图片摘要或大小无效")
    return manifest


def _normalize(vectors: np.ndarray, feature_dim: int) -> np.ndarray:
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 \
            or matrix.shape[1] != feature_dim \
            or not np.isfinite(matrix).all():
        raise ValueError("查询 Embedding 形状、维数或数值无效")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("查询 Embedding 包含零向量")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _build_output_manifest(source: dict, detected: pd.DataFrame) -> dict:
    detections = [] if detected.empty else [{
        "query_image_id": str(row.image_id),
        "crop_index": int(row.crop_index),
        "bbox": [int(row.x), int(row.y), int(row.w), int(row.h)],
        "quality": float(row.det_conf),
        "fallback": bool(row.fallback),
    } for row in detected.itertuples()]
    return {
        "schema_version": 1,
        "query_request_id": source["query_request_id"],
        "catalog_id": source["catalog_id"],
        "model_version": source["model_version"],
        "detector_version": source["detector_version"],
        "preprocess_id": source["preprocess_id"],
        "pipeline_config_digest": source["pipeline_config_digest"],
        "row_binding_digest": source["row_binding_digest"],
        "detections": detections,
    }


def _build_archive(manifest: dict, vectors: np.ndarray) -> bytes:
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
    return output.getvalue()
