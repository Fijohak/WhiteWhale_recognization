"""从服务器 Batch 事实生成不可伪造的 GPU 批次归档任务。"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .jobs import JobQueueService
from .models import Batch, Image
from .states import BatchStage


@dataclass(frozen=True)
class ArchivalDispatchRequest:
    model_version: str
    detector_version: str
    preprocess_id: str
    pipeline_config: dict
    required_vram_mb: int = 4096
    max_attempts: int = 3


class ArchivalDispatchService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        jobs: JobQueueService,
    ) -> None:
        self._sessions = sessions
        self._jobs = jobs

    def dispatch(
        self,
        batch_id: uuid.UUID,
        request: ArchivalDispatchRequest,
    ) -> uuid.UUID:
        self._validate(request)
        canonical_config = json.dumps(
            request.pipeline_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        pipeline_digest = hashlib.sha256(
            canonical_config.encode("utf-8")).hexdigest()
        with self._sessions() as db:
            batch = db.get(Batch, batch_id)
            if batch is None or batch.stage != BatchStage.REGISTERED:
                raise ValueError("只有 registered Batch 可派发归档任务")
            images = list(db.scalars(
                select(Image)
                .where(Image.batch_id == batch_id)
                .order_by(Image.original_relative_path, Image.id)
            ))
            if not images:
                raise ValueError("Batch 没有可归档图片")
            manifest = {
                "schema_version": 1,
                "batch_id": str(batch.id),
                "batch_manifest_sha256": batch.manifest_sha256,
                "model_version": request.model_version,
                "detector_version": request.detector_version,
                "preprocess_id": request.preprocess_id,
                "pipeline_config_digest": pipeline_digest,
                "min_cluster_size": int(request.pipeline_config.get(
                    "min_cluster_size", 3)),
                "pipeline_config": request.pipeline_config,
                "images": [{
                    "image_id": str(image.id),
                    "original_relative_path": image.original_relative_path,
                    "sha256": image.source_sha256,
                    "size_bytes": image.size_bytes,
                } for image in images],
            }
        return self._jobs.create(
            task_type="batch_archival",
            required_vram_mb=request.required_vram_mb,
            required_model_version=request.model_version,
            max_attempts=request.max_attempts,
            idempotency_key=(
                f"archive:{batch_id}:{request.model_version}:"
                f"{pipeline_digest[:16]}"
            ),
            input_manifest=manifest,
            batch_id=batch_id,
        )

    @staticmethod
    def _validate(request: ArchivalDispatchRequest) -> None:
        for label, value in (
            ("Model Version", request.model_version),
            ("Detector Version", request.detector_version),
            ("Preprocess ID", request.preprocess_id),
        ):
            if not value.strip() or len(value) > 128:
                raise ValueError(f"{label} 长度必须为 1–128")
        if request.required_vram_mb <= 0 or request.max_attempts <= 0:
            raise ValueError("归档任务资源或重试次数无效")
        if not isinstance(request.pipeline_config, dict):
            raise ValueError("Pipeline Config 必须是对象")
        minimum = request.pipeline_config.get("min_cluster_size", 3)
        if not isinstance(minimum, int) or minimum < 2:
            raise ValueError("min_cluster_size 必须是至少 2 的整数")
