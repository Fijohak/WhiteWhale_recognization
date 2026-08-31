"""可续传查询上传、GPU Job 调度与固定 Catalog 结果投影。"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .catalogs import CatalogService
from .jobs import JobQueueService
from .models import (
    Artifact, ArtifactManifest, CatalogVersion, ConfirmedIndividual, Crop, Job,
    JobAttempt, ModelVersion, Observation, QueryImage, QueryRequest, UploadFile,
    UploadSession,
)
from .states import JobState
from .storage import StorageLayout


class QueryService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        storage: StorageLayout,
        jobs: JobQueueService,
        catalogs: CatalogService,
    ) -> None:
        self._sessions = sessions
        self._storage = storage
        self._jobs = jobs
        self._catalogs = catalogs

    def dispatch_upload(
        self,
        upload_session_id: uuid.UUID,
        *,
        owner_user_id: uuid.UUID,
        k: int,
        detector_version: str,
        required_vram_mb: int,
    ) -> uuid.UUID:
        if not isinstance(owner_user_id, uuid.UUID):
            raise ValueError("查询所有者无效")
        if k <= 0 or k > 100 or required_vram_mb <= 0:
            raise ValueError("查询 Top-K 或显存要求无效")
        if not detector_version.strip() or len(detector_version) > 128:
            raise ValueError("Detector Version 无效")
        active = self._catalogs.active_version(validate_index=True)
        with self._sessions() as db:
            existing = db.scalar(select(QueryRequest).where(
                QueryRequest.upload_session_id == upload_session_id))
            if existing is not None:
                if existing.owner_user_id != owner_user_id:
                    raise ValueError("查询上传所有者不匹配")
                return existing.id
            upload = db.get(UploadSession, upload_session_id)
            if upload is None or upload.owner_user_id != owner_user_id:
                raise ValueError("查询上传所有者不匹配")
            if upload.state != "complete":
                raise ValueError("查询上传尚未完成")
            files = list(db.scalars(select(UploadFile).where(
                UploadFile.upload_session_id == upload_session_id).order_by(
                    UploadFile.relative_path, UploadFile.id)))
            if not files or any(
                    item.state != "complete" or not item.completed_path
                    for item in files):
                raise ValueError("查询上传文件不完整")
            model = db.scalar(select(ModelVersion).where(
                ModelVersion.version == active.model_version))
            preprocess_id = model.preprocess_id if model else "legacy"

        query_id = uuid.uuid4()
        rows: list[QueryImage] = []
        digest_rows = []
        copied: list[Path] = []
        try:
            for index, item in enumerate(files):
                source = self._storage.resolve(
                    "staging", item.completed_path or "")
                suffix = Path(item.relative_path).suffix.lower() or ".bin"
                relative = Path("queries") / str(query_id) / \
                    f"{index:06d}{suffix}"
                target = self._storage.resolve("raw", relative)
                self._atomic_copy(source, target)
                copied.append(target)
                if self._sha256(target) != item.sha256 \
                        or target.stat().st_size != item.size_bytes:
                    raise ValueError("查询图片副本摘要或大小不一致")
                row = QueryImage(
                    query_request_id=query_id, row_index=index,
                    original_relative_path=item.relative_path,
                    source_path=relative.as_posix(), sha256=item.sha256,
                    size_bytes=item.size_bytes)
                rows.append(row)
                digest_rows.append({
                    "relative_path": item.relative_path,
                    "sha256": item.sha256, "size_bytes": item.size_bytes})
            input_digest = hashlib.sha256(json.dumps(
                digest_rows, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            request = QueryRequest(
                id=query_id, owner_user_id=owner_user_id,
                upload_session_id=upload_session_id,
                catalog_id=active.catalog_id,
                model_version=active.model_version,
                detector_version=detector_version,
                preprocess_id=preprocess_id,
                top_k=k, input_digest=input_digest)
            with self._sessions.begin() as db:
                db.add(request)
                db.flush()
                db.add_all(rows)
                db.flush()
                query_images = list(db.scalars(select(QueryImage).where(
                    QueryImage.query_request_id == query_id).order_by(
                        QueryImage.row_index)))
            pipeline_config = {
                "catalog_id": str(active.catalog_id),
                "model_version": active.model_version,
                "detector_version": detector_version,
                "preprocess_id": preprocess_id,
            }
            pipeline_digest = hashlib.sha256(json.dumps(
                pipeline_config, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            row_binding = hashlib.sha256(json.dumps(
                [str(item.id) for item in query_images],
                separators=(",", ":")).encode()).hexdigest()
            manifest = {
                "schema_version": 1,
                "query_request_id": str(query_id),
                "catalog_id": str(active.catalog_id),
                "model_version": active.model_version,
                "detector_version": detector_version,
                "preprocess_id": preprocess_id,
                "feature_dim": active.feature_dim,
                "top_k": k,
                "pipeline_config_digest": pipeline_digest,
                "row_binding_digest": row_binding,
                "images": [{
                    "query_image_id": str(item.id),
                    "original_relative_path": item.original_relative_path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                } for item in query_images],
            }
            job_id = self._jobs.create(
                task_type="query_inference",
                required_vram_mb=required_vram_mb,
                required_model_version=active.model_version,
                max_attempts=3,
                idempotency_key=f"query:{query_id}:{pipeline_digest[:16]}",
                input_manifest=manifest)
            with self._sessions.begin() as db:
                stored = db.get(QueryRequest, query_id, with_for_update=True)
                if stored is None:
                    raise RuntimeError("查询请求登记丢失")
                stored.job_id = job_id
            return query_id
        except BaseException:
            with self._sessions.begin() as db:
                stored = db.get(QueryRequest, query_id)
                if stored is not None and stored.job_id is None:
                    db.delete(stored)
            for path in copied:
                path.unlink(missing_ok=True)
            raise

    def query_image_file(
        self, job_id: uuid.UUID, query_image_id: uuid.UUID,
    ) -> Path:
        with self._sessions() as db:
            job = db.get(Job, job_id)
            image = db.get(QueryImage, query_image_id)
            if job is None or image is None \
                    or job.task_type != "query_inference" \
                    or str(query_image_id) not in {
                        item.get("query_image_id")
                        for item in job.input_manifest.get("images", [])
                    }:
                raise ValueError("查询图片不属于租约输入")
            path = self._storage.resolve("raw", image.source_path)
        if not path.is_file() or self._sha256(path) != image.sha256:
            raise ValueError("查询图片文件缺失或摘要不一致")
        return path

    def result(
        self,
        query_request_id: uuid.UUID,
        *,
        requester_user_id: uuid.UUID,
        is_admin: bool,
    ) -> dict:
        with self._sessions() as db:
            request = db.get(QueryRequest, query_request_id)
            if request is None:
                raise ValueError("查询请求不存在")
            if request.owner_user_id != requester_user_id and not is_admin:
                raise ValueError("不能查看其他用户的查询")
            if request.result_json:
                return request.result_json
            job = db.get(Job, request.job_id) if request.job_id else None
            if job is None:
                raise ValueError("查询 Job 不存在")
            if job.state != JobState.SUCCEEDED:
                error = None
                if job.state == JobState.FAILED:
                    attempt = db.scalar(select(JobAttempt).where(
                        JobAttempt.job_id == job.id).order_by(
                            JobAttempt.attempt_number.desc()).limit(1))
                    error = attempt.error_detail if attempt else None
                return {
                    "query_request_id": str(request.id),
                    "status": "failed" if job.state == JobState.FAILED
                    else "running" if job.state != JobState.QUEUED else "queued",
                    "job_id": str(job.id), "error": error,
                }
            artifact = db.scalar(select(Artifact).where(
                Artifact.job_id == job.id,
                Artifact.artifact_type == "query_embedding").order_by(
                    Artifact.created_at.desc()).limit(1))
            if artifact is None:
                raise ValueError("成功的查询 Job 缺少 query_embedding Artifact")
            artifact_path = self._storage.resolve(
                "artifacts", artifact.relative_path)
            artifact_manifest = db.get(ArtifactManifest, artifact.id)
            catalog = db.get(CatalogVersion, request.catalog_id)
            query_images = list(db.scalars(select(QueryImage).where(
                QueryImage.query_request_id == request.id).order_by(
                    QueryImage.row_index)))
            query_image_ids = {item.id for item in query_images}
            expected = {
                "catalog_id": request.catalog_id,
                "model_version": request.model_version,
                "detector_version": request.detector_version,
                "preprocess_id": request.preprocess_id,
                "top_k": request.top_k,
                "query_request_id": request.id,
                "feature_dim": catalog.feature_dim if catalog else None,
                "calibration_status": (
                    catalog.calibration_status if catalog else "unknown"),
                "pipeline_config_digest": job.input_manifest.get(
                    "pipeline_config_digest"),
                "row_binding_digest": job.input_manifest.get(
                    "row_binding_digest"),
            }
            if artifact_manifest is None:
                raise ValueError("查询 Artifact 缺少 Manifest")
            if artifact_manifest.schema_version != 1 \
                    or artifact_manifest.model_version != expected["model_version"] \
                    or artifact_manifest.detector_version \
                    != expected["detector_version"] \
                    or artifact_manifest.preprocess_id \
                    != expected["preprocess_id"] \
                    or artifact_manifest.pipeline_config_digest \
                    != expected["pipeline_config_digest"] \
                    or artifact_manifest.row_binding_digest \
                    != expected["row_binding_digest"]:
                raise ValueError("查询 Artifact Header 与任务绑定不一致")
        payload = artifact_path.read_bytes()
        if len(payload) != artifact.size_bytes \
                or hashlib.sha256(payload).hexdigest() != artifact.sha256:
            raise ValueError("查询 Artifact 摘要或大小不一致")
        report, vectors = self._read_embedding_archive(payload)
        if report.get("schema_version") != 1 \
                or report.get("query_request_id") \
                != str(expected["query_request_id"]) \
                or report.get("catalog_id") != str(expected["catalog_id"]) \
                or report.get("model_version") != expected["model_version"] \
                or report.get("detector_version") \
                != expected["detector_version"] \
                or report.get("preprocess_id") != expected["preprocess_id"] \
                or report.get("pipeline_config_digest") \
                != expected["pipeline_config_digest"] \
                or report.get("row_binding_digest") \
                != expected["row_binding_digest"]:
            raise ValueError("查询 Artifact 与请求绑定不一致")
        raw_detections = report.get("detections")
        if not isinstance(raw_detections, list) \
                or vectors.ndim != 2 \
                or vectors.shape != (
                    len(raw_detections), expected["feature_dim"]) \
                or not np.isfinite(vectors).all():
            raise ValueError("查询 Artifact 检测行与 Embedding 形状不一致")
        if len(vectors) and np.any(np.linalg.norm(vectors, axis=1) <= 0):
            raise ValueError("查询 Artifact 包含零向量")
        detections = []
        for item, vector in zip(raw_detections, vectors, strict=True):
            if not isinstance(item, dict):
                raise ValueError("查询 Artifact 检测行无效")
            try:
                query_image_id = uuid.UUID(str(item.get("query_image_id")))
            except (TypeError, ValueError) as exc:
                raise ValueError("查询 Artifact 图片 ID 无效") from exc
            if query_image_id not in query_image_ids:
                raise ValueError("查询 Artifact 引用了清单之外的图片")
            matches = self._catalogs.search_at(
                expected["catalog_id"], vector, k=expected["top_k"])
            detections.append({
                "query_image_id": str(query_image_id),
                "crop_index": item.get("crop_index"),
                "bbox": item.get("bbox"),
                "quality": item.get("quality"),
                "matches": [self._match_body(match) for match in matches],
            })
        result = {
            "query_request_id": str(expected["query_request_id"]),
            "status": "succeeded",
            "catalog_id": str(expected["catalog_id"]),
            "model_version": expected["model_version"],
            "calibration_status": expected["calibration_status"],
            "human_review_status": "candidate_only",
            "images": [{
                "query_image_id": str(item.id),
                "original_relative_path": item.original_relative_path,
            } for item in query_images],
            "detections": detections,
        }
        with self._sessions.begin() as db:
            stored = db.get(
                QueryRequest, expected["query_request_id"],
                with_for_update=True)
            if stored is not None:
                stored.state = "succeeded"
                stored.result_json = result
                stored.completed_at = datetime.now(UTC)
        return result

    @staticmethod
    def _read_embedding_archive(payload: bytes) -> tuple[dict, np.ndarray]:
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                if set(archive.namelist()) != {
                        "manifest.json", "embeddings.npy"}:
                    raise ValueError("查询 Artifact 文件清单无效")
                report = json.loads(archive.read("manifest.json"))
                vectors = np.load(
                    io.BytesIO(archive.read("embeddings.npy")),
                    allow_pickle=False)
        except (zipfile.BadZipFile, KeyError, UnicodeDecodeError,
                json.JSONDecodeError, ValueError, EOFError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("查询 Artifact"):
                raise
            raise ValueError("查询 Artifact 不是合法 ZIP/NPY 产物") from exc
        if not isinstance(report, dict):
            raise ValueError("查询 Artifact Manifest 必须是对象")
        return report, np.ascontiguousarray(vectors, dtype=np.float32)

    def _match_body(self, match) -> dict:
        with self._sessions() as db:
            row = db.execute(
                select(ConfirmedIndividual, Observation, Crop)
                .join(Observation, Observation.individual_id
                      == ConfirmedIndividual.id)
                .join(Crop, Crop.id == Observation.crop_id)
                .where(Observation.id == match.observation_id)
            ).one()
            individual, observation, crop = row
            sides = set(db.scalars(select(Observation.side).where(
                Observation.individual_id == individual.id,
                Observation.state == "active",
            )))
            return {
                "individual_id": str(individual.id),
                "individual_name": individual.display_name,
                "observation_id": str(observation.id),
                "representative_media_url": f"/api/media/crops/{crop.id}",
                "score": match.score,
                "support_frames": match.support_frames,
                "side": observation.side,
                "cross_side": "left" in sides and "right" in sides,
                "quality": observation.quality,
                "catalog_id": str(match.catalog_id),
                "model_version": match.model_version,
                "calibration_status": match.calibration_status,
            }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _atomic_copy(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as value:
                shutil.copyfileobj(value, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
