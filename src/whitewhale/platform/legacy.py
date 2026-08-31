"""现有 Manifest、审核表、权重和 Gallery 的只读登记。"""
from __future__ import annotations

import hashlib
import csv
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .catalogs import CatalogEntry, CatalogService, build_flat_ip_index
from .models import (
    Artifact,
    ArtifactManifest,
    Batch,
    CatalogVersion,
    ConfirmedIndividual,
    Crop,
    DatasetVersion,
    IdentityEvent,
    Image,
    IndividualAlias,
    Job,
    JobAttempt,
    LegacyArtifact,
    ModelPromotionEvent,
    ModelVersion,
    Observation,
    ProductionModelPointer,
    ReviewTask,
    TrainingRun,
    User,
    WorkerDevice,
)
from .states import BatchStage, JobState
from .storage import StorageLayout


@dataclass(frozen=True)
class LegacyArtifactSpec:
    artifact_kind: str
    source_path: Path
    calibration_status: str = "not_applicable"
    metadata: dict | None = None


@dataclass(frozen=True)
class LegacyReleaseImportResult:
    model_version_id: uuid.UUID
    catalog_id: uuid.UUID
    observation_count: int
    individual_count: int
    copied_raw_files: int
    copied_crop_files: int


class LegacyImportService:
    ALLOWED_KINDS = {
        "dataset_manifest", "review_csv", "model_weights",
        "gallery_embeddings", "gallery_metadata", "artifact_manifest",
    }

    def __init__(self, sessions: sessionmaker[Session], storage: StorageLayout):
        self._sessions = sessions
        self._storage = storage

    def inspect(self, spec: LegacyArtifactSpec) -> dict:
        source = spec.source_path.resolve()
        if spec.artifact_kind not in self.ALLOWED_KINDS:
            raise ValueError("Legacy Artifact 类型无效")
        if not source.is_file():
            raise ValueError(f"Legacy 文件不存在: {source}")
        if spec.calibration_status not in {
                "not_applicable", "provisional_unvalidated", "calibrated"}:
            raise ValueError("Legacy 校准状态无效")
        if spec.artifact_kind in {"model_weights", "gallery_embeddings"} \
                and spec.calibration_status == "calibrated" \
                and not (spec.metadata or {}).get("calibration_report_sha256"):
            raise ValueError("Legacy 模型/图库不能无报告声明已校准")
        return {
            "artifact_kind": spec.artifact_kind,
            "source_name": source.name,
            "source_path": str(source),
            "sha256": self._sha256(source),
            "size_bytes": source.stat().st_size,
            "calibration_status": spec.calibration_status,
            "metadata": spec.metadata or {},
        }

    def register(self, spec: LegacyArtifactSpec) -> uuid.UUID:
        inspected = self.inspect(spec)
        with self._sessions() as db:
            existing = db.scalar(select(LegacyArtifact).where(
                LegacyArtifact.artifact_kind == inspected["artifact_kind"],
                LegacyArtifact.sha256 == inspected["sha256"],
            ))
            if existing is not None:
                return existing.id
        artifact_id = uuid.uuid4()
        relative = Path("legacy") / str(artifact_id) / inspected["source_name"]
        target = self._storage.resolve("artifacts", relative)
        self._atomic_copy(Path(inspected["source_path"]), target)
        try:
            with self._sessions.begin() as db:
                db.add(LegacyArtifact(
                    id=artifact_id,
                    artifact_kind=inspected["artifact_kind"],
                    source_name=inspected["source_name"],
                    relative_path=relative.as_posix(),
                    sha256=inspected["sha256"],
                    size_bytes=inspected["size_bytes"],
                    calibration_status=inspected["calibration_status"],
                    metadata_json=inspected["metadata"],
                ))
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return artifact_id

    def import_initial_reid_release(
        self,
        *,
        model_version: str,
        weights_path: Path,
        embeddings_path: Path,
        metadata_path: Path,
        config_path: Path,
        raw_source_root: Path,
        crop_source_root: Path,
    ) -> LegacyReleaseImportResult:
        """把显式的批次内 r4 Gallery 投影为初始 Model/Catalog。

        身份只从 ``session/quality/group`` 三段路径生成；同名 group 不跨
        session 合并。旧浮点化 ID 仅作为 alias 保存。
        """
        if not model_version.strip():
            raise ValueError("Legacy Model Version 不能为空")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with metadata_path.open(newline="", encoding="utf-8-sig") as source:
            rows = list(csv.DictReader(source))
        vectors = np.load(embeddings_path, allow_pickle=False)
        feature_dim = int(config.get("feat_dim", 0))
        if not rows or vectors.shape != (len(rows), feature_dim) \
                or int(config.get("n", -1)) != len(rows) \
                or not np.isfinite(vectors).all():
            raise ValueError("Legacy Gallery 的 Meta、Embedding 与 Config 不一致")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("Legacy Gallery 包含零向量")
        vectors = np.ascontiguousarray(vectors / norms, dtype=np.float32)
        if len({row.get("image_id", "") for row in rows}) != len(rows):
            raise ValueError("Legacy Gallery image_id 为空或重复")

        registered = {}
        for key, kind, path, calibration in (
            ("weights", "model_weights", weights_path,
             "provisional_unvalidated"),
            ("embeddings", "gallery_embeddings", embeddings_path,
             "provisional_unvalidated"),
            ("metadata", "gallery_metadata", metadata_path,
             "not_applicable"),
            ("config", "artifact_manifest", config_path,
             "provisional_unvalidated"),
        ):
            registered[key] = self.register(LegacyArtifactSpec(
                artifact_kind=kind,
                source_path=path,
                calibration_status=calibration,
                metadata={"model_version": model_version},
            ))

        with self._sessions() as db:
            existing_model = db.scalar(select(ModelVersion).where(
                ModelVersion.version == model_version))
            existing_catalog = db.scalar(select(CatalogVersion).where(
                CatalogVersion.model_version == model_version,
                CatalogVersion.status == "active"))
            if existing_model is not None and existing_catalog is not None:
                return LegacyReleaseImportResult(
                    existing_model.id, existing_catalog.id,
                    existing_catalog.row_count, 0, 0, 0)
            if existing_model is not None or existing_catalog is not None:
                raise ValueError("Legacy Model/Catalog 只完成了一半，拒绝猜测修复")

        prepared = []
        copied_raw = copied_crops = 0
        raw_root = raw_source_root.resolve()
        crop_root = crop_source_root.resolve()
        for row in rows:
            parsed = self._parse_gallery_row(row)
            raw_source = (raw_root / parsed["relative_path"]).resolve()
            try:
                raw_source.relative_to(raw_root)
            except ValueError as exc:
                raise ValueError("Legacy 原图路径越界") from exc
            crop_source = crop_root / f'{parsed["image_key"]}.jpg'
            if not raw_source.is_file() or not crop_source.is_file():
                raise ValueError(
                    f'Legacy 原图或 Crop 缺失: {parsed["image_key"]}')
            raw_sha = self._sha256(raw_source)
            raw_relative = Path("legacy") / raw_sha[:2] / \
                f"{raw_sha}{raw_source.suffix.lower()}"
            crop_sha = self._sha256(crop_source)
            crop_relative = Path("legacy-crops") / crop_sha[:2] / \
                f"{crop_sha}.jpg"
            copied_raw += int(self._copy_if_missing(
                raw_source, self._storage.resolve("raw", raw_relative), raw_sha))
            copied_crops += int(self._copy_if_missing(
                crop_source,
                self._storage.resolve("artifacts", crop_relative), crop_sha))
            prepared.append({
                **parsed,
                "raw_relative": raw_relative.as_posix(),
                "raw_sha256": raw_sha,
                "raw_size": raw_source.stat().st_size,
                "crop_relative": crop_relative.as_posix(),
            })

        with self._sessions() as db:
            legacy_weight = db.get(LegacyArtifact, registered["weights"])
            config_artifact = db.get(LegacyArtifact, registered["config"])
            assert legacy_weight is not None and config_artifact is not None
            artifact_relative = legacy_weight.relative_path
            weight_sha = legacy_weight.sha256
            weight_size = legacy_weight.size_bytes
            config_sha = config_artifact.sha256
        model_relative = Path("metric-learning") / model_version / "weights.pt"
        self._copy_if_missing(
            weights_path.resolve(),
            self._storage.resolve("models", model_relative), weight_sha)

        model_id = uuid.uuid4()
        observation_ids: list[uuid.UUID] = []
        individual_ids: dict[str, uuid.UUID] = {}
        now = datetime.now(UTC)
        with self._sessions.begin() as db:
            system_user = db.scalar(select(User).where(
                User.username == "legacy-import-system"))
            if system_user is None:
                system_user = User(
                    username="legacy-import-system",
                    password_hash="!disabled-system-account",
                    is_active=False)
                db.add(system_user)
                db.flush()
            worker = db.scalar(select(WorkerDevice).where(
                WorkerDevice.name == "legacy-importer"))
            if worker is None:
                worker = WorkerDevice(
                    name="legacy-importer", gpu_model="not-applicable",
                    vram_mb=1, cuda_version="not-applicable",
                    worker_version="legacy-v1",
                    capabilities=["legacy_import"],
                    model_versions=[model_version], capacity=1,
                    is_active=False)
                db.add(worker)
                db.flush()

            lineage_digest = hashlib.sha256(json.dumps({
                "model_version": model_version,
                "weights_sha256": weight_sha,
                "config_sha256": config_sha,
                "gallery_rows": len(rows),
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            lineage = DatasetVersion(
                name=f"{model_version}-legacy-lineage",
                protocol="known_identity_update", status="frozen",
                membership_digest=lineage_digest,
                rights_snapshot={
                    "scope": "project_internal",
                    "lineage_status": "legacy_training_membership_unavailable",
                    "gallery_is_not_training_set": True,
                },
                created_by_user_id=system_user.id)
            db.add(lineage)
            job = Job(
                task_type="reid_training", state=JobState.SUCCEEDED,
                required_vram_mb=1, max_attempts=1,
                idempotency_key=f"legacy-model-import:{model_version}",
                input_manifest={
                    "schema_version": 1, "legacy_import": True,
                    "dataset_lineage_digest": lineage_digest,
                })
            db.add(job)
            db.flush()
            attempt = JobAttempt(
                job_id=job.id, attempt_number=1,
                worker_device_id=worker.id, outcome=JobState.SUCCEEDED.value,
                started_at=now, finished_at=now)
            db.add(attempt)
            db.flush()
            artifact = Artifact(
                job_id=job.id, attempt_id=attempt.id,
                artifact_type="model_weights",
                relative_path=artifact_relative,
                sha256=weight_sha, size_bytes=weight_size,
                producer_device_id=worker.id)
            db.add(artifact)
            db.flush()
            db.add(ArtifactManifest(
                artifact_id=artifact.id, schema_version=1,
                model_version=model_version,
                preprocess_id=str(config.get("preprocess", "unknown")),
                pipeline_config_digest=config_sha,
                detail={
                    "origin": "legacy_import",
                    "calibration_status": "provisional_unvalidated",
                    "legacy_artifact_ids": {
                        key: str(value) for key, value in registered.items()},
                }))
            run = TrainingRun(
                job_id=job.id, dataset_version_id=lineage.id,
                task_type="reid_training", model_family="metric-learning",
                config={"origin": "legacy_import", "lineage_incomplete": True},
                seed=0, required_vram_mb=1, max_runtime_seconds=1,
                checkpoint_interval_steps=1, state="completed")
            db.add(run)
            db.flush()
            db.add(ModelVersion(
                id=model_id, training_run_id=run.id,
                weight_artifact_id=artifact.id,
                model_family="metric-learning", version=model_version,
                status="candidate", sha256=weight_sha,
                weight_path=model_relative.as_posix(),
                feature_dim=feature_dim,
                preprocess_id=str(config.get("preprocess", "unknown")),
                checkpoint_source="legacy_import",
                license="project_internal",
                compatible_detector_version="legacy-yolo-v2",
                compatible_crop_config=str(config.get("crop", "unknown")),
                compatible_index_schema=1,
                calibrated_thresholds={}))

            batches: dict[str, Batch] = {}
            tasks: dict[str, ReviewTask] = {}
            for item in prepared:
                session = item["session_id"]
                if session not in batches:
                    session_rows = [value["relative_path"] for value in prepared
                                    if value["session_id"] == session]
                    batch = Batch(
                        name=f"legacy:{session}", stage=BatchStage.ARCHIVED,
                        manifest_sha256=hashlib.sha256(json.dumps(
                            sorted(session_rows), separators=(",", ":")
                        ).encode()).hexdigest(),
                        source_format="legacy_gallery",
                        metadata_json={
                            "session_id": session,
                            "identity_scope": "batch_local",
                        })
                    db.add(batch)
                    db.flush()
                    batches[session] = batch
                identity_key = item["identity_key"]
                if identity_key not in individual_ids:
                    individual = ConfirmedIndividual(
                        display_name=f"LEGACY:{identity_key}",
                        flags=["legacy_batch_local_identity",
                               "provider_confirmed"])
                    db.add(individual)
                    db.flush()
                    individual_ids[identity_key] = individual.id
                    task = ReviewTask(
                        task_type="identity_match",
                        subject_type="legacy_provider_identity",
                        subject_id=individual.id, status="resolved",
                        required_reviewers=1,
                        policy_version="legacy_provider_confirmation_v1")
                    db.add(task)
                    db.flush()
                    tasks[identity_key] = task
                    db.add_all([
                        IndividualAlias(
                            individual_id=individual.id,
                            alias=identity_key,
                            source="legacy_source_group"),
                        IndividualAlias(
                            individual_id=individual.id,
                            alias=item["old_identity"],
                            source="legacy_float_id"),
                        IdentityEvent(
                            individual_id=individual.id,
                            review_task_id=task.id,
                            event_type="legacy_provider_confirmed",
                            payload={
                                "identity_scope": "batch_local",
                                "automatic_cross_batch_merge": False,
                            }),
                    ])
                image = Image(
                    batch_id=batches[session].id,
                    source_path=item["raw_relative"],
                    original_relative_path=item["relative_path"],
                    source_sha256=item["raw_sha256"],
                    size_bytes=item["raw_size"],
                    quality_band=item["quality_band"],
                    exif_json={"origin": "legacy_import"})
                db.add(image)
                db.flush()
                crop = Crop(
                    image_id=image.id, crop_index=0,
                    x=item["x"], y=item["y"],
                    width=item["width"], height=item["height"],
                    detector_version="legacy-yolo-v2",
                    artifact_path=item["crop_relative"])
                db.add(crop)
                db.flush()
                observation = Observation(
                    individual_id=individual_ids[identity_key],
                    crop_id=crop.id, side="unknown",
                    quality=self._quality_score(item["quality_band"]),
                    source_review_task_id=tasks[identity_key].id)
                db.add(observation)
                db.flush()
                observation_ids.append(observation.id)

        entries = [CatalogEntry(observation_id, vector)
                   for observation_id, vector in zip(
                       observation_ids, vectors, strict=True)]
        catalogs = CatalogService(self._sessions, self._storage)
        catalog_id = catalogs.stage(
            entries, model_version=model_version,
            calibration_status="provisional_unvalidated",
            index_bytes=build_flat_ip_index(vectors))
        catalogs.activate(catalog_id)
        with self._sessions.begin() as db:
            model = db.get(ModelVersion, model_id, with_for_update=True)
            assert model is not None
            model.status = "production"
            db.add(ProductionModelPointer(
                model_family="metric-learning", model_version_id=model.id))
            db.add(ModelPromotionEvent(
                model_version_id=model.id,
                event_type="legacy_imported_production",
                actor_user_id=db.scalar(select(User.id).where(
                    User.username == "legacy-import-system")),
                catalog_id=catalog_id,
                payload={
                    "calibration_status": "provisional_unvalidated",
                    "migration_exception": True,
                }))
        return LegacyReleaseImportResult(
            model_id, catalog_id, len(observation_ids),
            len(individual_ids), copied_raw, copied_crops)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _parse_gallery_row(row: dict[str, str]) -> dict:
        required = {
            "image_id", "relative_path", "session_id", "x", "y", "w", "h",
            "confirmed_identity",
        }
        if required - set(row) or any(not str(row.get(key, "")).strip()
                                      for key in required):
            raise ValueError("Legacy Gallery Meta 缺少必需字段")
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts \
                or len(relative.parts) < 4:
            raise ValueError("Legacy Gallery relative_path 无效")
        session, quality_band, group = relative.parts[:3]
        group_match = re.fullmatch(r"(?P<number>\d+)(?:\s*[（(].*[）)])?", group)
        if session != row["session_id"] or group_match is None:
            raise ValueError(
                "Legacy 身份必须来自 session/quality/numeric-group；"
                "group 只允许附带括号备注")
        group_number = group_match.group("number")
        x, y, width, height = (int(float(row[key]))
                               for key in ("x", "y", "w", "h"))
        if min(x, y) < 0 or width <= 0 or height <= 0:
            raise ValueError("Legacy bbox 无效")
        return {
            "image_key": row["image_id"],
            "relative_path": relative.as_posix(),
            "session_id": session,
            "quality_band": quality_band,
            "identity_key": f"{session}_{group_number}",
            "old_identity": row["confirmed_identity"],
            "x": x, "y": y, "width": width, "height": height,
        }

    @staticmethod
    def _quality_score(value: str) -> float | None:
        if value == "80 and above":
            return 0.85
        if "-" in value:
            try:
                low, high = (int(item) for item in value.split("-", 1))
                return (low + high) / 200
            except ValueError:
                return None
        return None

    def _copy_if_missing(
        self, source: Path, target: Path, expected_sha256: str,
    ) -> bool:
        if target.is_file():
            if self._sha256(target) != expected_sha256:
                raise ValueError(f"既有 Legacy 副本摘要冲突: {target}")
            return False
        self._atomic_copy(source, target)
        if self._sha256(target) != expected_sha256:
            target.unlink(missing_ok=True)
            raise ValueError("Legacy 副本写入后摘要不一致")
        return True

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
