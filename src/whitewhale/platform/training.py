"""不可变 Dataset、泄漏门禁、GPU 训练/评估任务与模型上线控制。"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .catalogs import CatalogEntry, CatalogService
from .jobs import JobQueueService
from .models import (
    ActiveCatalogPointer,
    Artifact,
    CatalogVersion,
    ConfirmedIndividual,
    Crop,
    DatasetMembership,
    DatasetSplit,
    DatasetVersion,
    EvaluationResult,
    EvaluationRun,
    Image,
    Job,
    ModelPromotionEvent,
    ModelVersion,
    Observation,
    ProductionModelPointer,
    Role,
    TrainingCheckpoint,
    TrainingRun,
    UserRole,
)
from .states import JobState
from .storage import StorageLayout


_ALLOWED_LABEL_SOURCES = {
    "provider_confirmed",
    "project_verified",
    "high_trust_pseudo_label",
}
_SPLITS = {"train", "val", "calibration", "test"}


@dataclass(frozen=True)
class DatasetSampleSpec:
    observation_id: uuid.UUID
    label_source: str
    split: str
    sequence_key: str
    encounter_key: str
    duplicate_group: str
    data_license: str


@dataclass(frozen=True)
class ModelManifest:
    model_family: str
    version: str
    sha256: str
    feature_dim: int | None
    preprocess_id: str
    checkpoint_source: str
    license: str
    compatible_detector_version: str | None
    compatible_crop_config: str
    compatible_index_schema: int


class DatasetService:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def freeze(
        self,
        *,
        name: str,
        protocol: str,
        samples: list[DatasetSampleSpec],
        created_by_user_id: uuid.UUID,
        rights_snapshot: dict,
        catalog_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        if not name.strip() or protocol not in {
                "known_identity_update", "open_set_unknown"}:
            raise ValueError("Dataset 名称或协议无效")
        if not samples:
            raise ValueError("Dataset 不能为空")
        observation_ids = [item.observation_id for item in samples]
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("Dataset 包含重复 Observation")
        if {item.split for item in samples} != _SPLITS:
            raise ValueError("Dataset 必须同时冻结 train/val/calibration/test")
        for item in samples:
            if item.label_source not in _ALLOWED_LABEL_SOURCES:
                raise ValueError(
                    f"标签来源不能用于监督训练: {item.label_source}")
            if not all((item.sequence_key.strip(), item.encounter_key.strip(),
                        item.duplicate_group.strip(), item.data_license.strip())):
                raise ValueError("Dataset 分组键和数据授权不能为空")
        self._ensure_group_isolation(samples, "Sequence", "sequence_key")
        self._ensure_group_isolation(samples, "Encounter", "encounter_key")
        self._ensure_group_isolation(
            samples, "近重复图片", "duplicate_group")

        with self._sessions.begin() as db:
            rows = list(db.execute(
                select(Observation, Crop, Image, ConfirmedIndividual)
                .join(Crop, Crop.id == Observation.crop_id)
                .join(Image, Image.id == Crop.image_id)
                .join(
                    ConfirmedIndividual,
                    ConfirmedIndividual.id == Observation.individual_id,
                )
                .where(Observation.id.in_(observation_ids))
            ))
            if len(rows) != len(samples):
                raise ValueError("Dataset 引用了不存在的 Observation")
            by_observation = {row[0].id: row for row in rows}
            if any(observation.state != "active" or individual.state != "active"
                   for observation, _, _, individual in rows):
                raise ValueError("Dataset 只能冻结 active Observation 和身份")
            image_splits: dict[uuid.UUID, set[str]] = {}
            canonical = []
            for item in samples:
                observation, crop, image, individual = by_observation[
                    item.observation_id]
                image_splits.setdefault(image.id, set()).add(item.split)
                canonical.append({
                    "observation_id": str(observation.id),
                    "image_id": str(image.id),
                    "crop_id": str(crop.id),
                    "individual_id": str(individual.id),
                    "label_source": item.label_source,
                    "split": item.split,
                    "sequence_key": item.sequence_key,
                    "encounter_key": item.encounter_key,
                    "duplicate_group": item.duplicate_group,
                    "data_license": item.data_license,
                })
            if any(len(splits) > 1 for splits in image_splits.values()):
                raise ValueError("同一 Image 派生 Crop 不能跨 Split")
            canonical.sort(key=lambda value: value["observation_id"])
            digest = hashlib.sha256(json.dumps(
                canonical, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
            existing = db.scalar(select(DatasetVersion).where(
                DatasetVersion.membership_digest == digest))
            if existing is not None:
                return existing.id
            dataset = DatasetVersion(
                name=name.strip(), protocol=protocol,
                catalog_id=catalog_id, membership_digest=digest,
                rights_snapshot=rights_snapshot,
                created_by_user_id=created_by_user_id,
            )
            db.add(dataset)
            db.flush()
            for value in canonical:
                observation_id = uuid.UUID(value["observation_id"])
                db.add(DatasetMembership(
                    dataset_version_id=dataset.id,
                    observation_id=observation_id,
                    image_id=uuid.UUID(value["image_id"]),
                    crop_id=uuid.UUID(value["crop_id"]),
                    individual_id=uuid.UUID(value["individual_id"]),
                    label_source=value["label_source"],
                    sequence_key=value["sequence_key"],
                    encounter_key=value["encounter_key"],
                    duplicate_group=value["duplicate_group"],
                    data_license=value["data_license"],
                ))
                db.add(DatasetSplit(
                    dataset_version_id=dataset.id,
                    observation_id=observation_id,
                    split=value["split"],
                ))
            return dataset.id

    @staticmethod
    def _ensure_group_isolation(
        samples: list[DatasetSampleSpec], label: str, field: str,
    ) -> None:
        groups: dict[str, set[str]] = {}
        for sample in samples:
            groups.setdefault(getattr(sample, field), set()).add(sample.split)
        if any(len(splits) > 1 for splits in groups.values()):
            raise ValueError(f"{label} 不能跨 Split")


class TrainingLifecycleService:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        jobs: JobQueueService,
        storage: StorageLayout,
    ) -> None:
        self._sessions = sessions
        self._jobs = jobs
        self._storage = storage

    def dispatch_training(
        self,
        dataset_version_id: uuid.UUID,
        *,
        task_type: str,
        model_family: str,
        base_model_id: uuid.UUID | None,
        config: dict,
        seed: int,
        required_vram_mb: int,
        max_runtime_seconds: int,
        checkpoint_interval_steps: int,
        resume_checkpoint_id: uuid.UUID | None = None,
    ) -> tuple[uuid.UUID, uuid.UUID]:
        if task_type not in {"detector_training", "reid_training"}:
            raise ValueError("训练任务类型无效")
        if not model_family.strip() or any(value <= 0 for value in (
                required_vram_mb, max_runtime_seconds,
                checkpoint_interval_steps)):
            raise ValueError("训练模型族或资源限制无效")
        run_id = uuid.uuid4()
        samples = self._dataset_samples(
            dataset_version_id, {"train", "val", "calibration"})
        resume = None
        if resume_checkpoint_id is not None:
            with self._sessions() as db:
                checkpoint = db.get(TrainingCheckpoint, resume_checkpoint_id)
                if checkpoint is None or not checkpoint.verified:
                    raise ValueError("恢复点不存在或尚未验证")
                prior = db.get(TrainingRun, checkpoint.training_run_id)
                if prior is None \
                        or prior.dataset_version_id != dataset_version_id \
                        or prior.model_family != model_family \
                        or prior.task_type != task_type \
                        or prior.config != config:
                    raise ValueError(
                        "恢复点与 Dataset、任务、Model Family 或配置不兼容")
                resume = {
                    "checkpoint_id": str(checkpoint.id),
                    "artifact_id": str(checkpoint.artifact_id),
                    "sha256": checkpoint.sha256,
                }
        config_digest = hashlib.sha256(json.dumps(
            config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        manifest = {
            "schema_version": 1,
            "training_run_id": str(run_id),
            "dataset_version_id": str(dataset_version_id),
            "task_type": task_type,
            "model_family": model_family,
            "base_model_id": str(base_model_id) if base_model_id else None,
            "config": config,
            "config_digest": config_digest,
            "seed": seed,
            "max_runtime_seconds": max_runtime_seconds,
            "checkpoint_interval_steps": checkpoint_interval_steps,
            "resume": resume,
            "samples": samples,
        }
        manifest["row_binding_digest"] = hashlib.sha256(json.dumps(
            [item["observation_id"] for item in samples],
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        job_id = self._jobs.create(
            task_type=task_type,
            required_vram_mb=required_vram_mb,
            required_model_version=None,
            max_attempts=3,
            idempotency_key=(
                f"train:{dataset_version_id}:{model_family}:{task_type}:"
                f"{seed}:{config_digest[:16]}:{resume_checkpoint_id or 'fresh'}"
            ),
            input_manifest=manifest,
        )
        with self._sessions.begin() as db:
            existing = db.scalar(select(TrainingRun).where(
                TrainingRun.job_id == job_id))
            if existing is not None:
                return existing.id, job_id
            db.add(TrainingRun(
                id=run_id, job_id=job_id,
                dataset_version_id=dataset_version_id,
                task_type=task_type, model_family=model_family,
                base_model_version_id=base_model_id,
                config=config, seed=seed,
                required_vram_mb=required_vram_mb,
                max_runtime_seconds=max_runtime_seconds,
                checkpoint_interval_steps=checkpoint_interval_steps,
                resume_checkpoint_id=resume_checkpoint_id,
            ))
        return run_id, job_id

    def register_checkpoint(
        self,
        training_run_id: uuid.UUID,
        artifact_id: uuid.UUID,
        *,
        stage: int,
        epoch: int,
        step: int,
        metadata: dict | None = None,
    ) -> uuid.UUID:
        if stage < 0 or epoch < 0 or step < 0:
            raise ValueError("Checkpoint stage/epoch/step 不能为负数")
        with self._sessions.begin() as db:
            run = db.get(TrainingRun, training_run_id, with_for_update=True)
            artifact = db.get(Artifact, artifact_id)
            if run is None or artifact is None \
                    or artifact.job_id != run.job_id \
                    or artifact.artifact_type != "training_checkpoint":
                raise ValueError("Checkpoint Artifact 与 Training Run 不匹配")
            self._verify_artifact_file(artifact)
            existing = db.scalar(select(TrainingCheckpoint).where(
                TrainingCheckpoint.artifact_id == artifact_id))
            if existing is not None:
                return existing.id
            checkpoint = TrainingCheckpoint(
                training_run_id=run.id, artifact_id=artifact.id,
                stage=stage, epoch=epoch, step=step, sha256=artifact.sha256,
                verified=True, metadata_json=metadata or {},
            )
            db.add(checkpoint)
            db.flush()
            return checkpoint.id

    def register_model(
        self,
        training_run_id: uuid.UUID,
        artifact_id: uuid.UUID,
        manifest: ModelManifest,
    ) -> uuid.UUID:
        self._validate_model_manifest(manifest)
        with self._sessions.begin() as db:
            run = db.get(TrainingRun, training_run_id, with_for_update=True)
            artifact = db.get(Artifact, artifact_id)
            if run is None or artifact is None \
                    or artifact.job_id != run.job_id \
                    or artifact.artifact_type != "model_weights" \
                    or db.get(Job, run.job_id).state != JobState.SUCCEEDED:
                raise ValueError("权重 Artifact 不是已成功 Training Run 的产物")
            self._verify_artifact_file(artifact)
            if artifact.sha256 != manifest.sha256:
                raise ValueError("Model Manifest 权重 SHA-256 不一致")
            existing = db.scalar(select(ModelVersion).where(
                ModelVersion.training_run_id == training_run_id))
            if existing is not None:
                if existing.sha256 != manifest.sha256:
                    raise ValueError("Training Run 已绑定不同模型权重")
                return existing.id
            relative = Path(manifest.model_family) / manifest.version / \
                "weights.bin"
            source = self._storage.resolve("artifacts", artifact.relative_path)
            target = self._storage.resolve("models", relative)
            self._atomic_copy(source, target)
            model = ModelVersion(
                training_run_id=run.id, weight_artifact_id=artifact.id,
                model_family=manifest.model_family, version=manifest.version,
                sha256=manifest.sha256, weight_path=relative.as_posix(),
                feature_dim=manifest.feature_dim,
                preprocess_id=manifest.preprocess_id,
                checkpoint_source=manifest.checkpoint_source,
                license=manifest.license,
                compatible_detector_version=
                manifest.compatible_detector_version,
                compatible_crop_config=manifest.compatible_crop_config,
                compatible_index_schema=manifest.compatible_index_schema,
            )
            db.add(model)
            run.state = "completed"
            db.flush()
            return model.id

    def dispatch_evaluation(
        self,
        model_version_id: uuid.UUID,
        dataset_version_id: uuid.UUID,
        *,
        required_vram_mb: int,
    ) -> tuple[uuid.UUID, uuid.UUID]:
        if required_vram_mb <= 0:
            raise ValueError("评估显存要求无效")
        with self._sessions() as db:
            model = db.get(ModelVersion, model_version_id)
            dataset = db.get(DatasetVersion, dataset_version_id)
            if model is None or model.status != "candidate" \
                    or dataset is None or dataset.status != "frozen":
                raise ValueError("评估要求 candidate Model 和 frozen Dataset")
            model_version = model.version
            weight_artifact_id = model.weight_artifact_id
            model_sha256 = model.sha256
            model_family = model.model_family
            feature_dim = model.feature_dim
            preprocess_id = model.preprocess_id
            protocol = dataset.protocol
            production = db.get(ProductionModelPointer, model.model_family)
            production_model = (None if production is None else
                                db.get(ModelVersion,
                                       production.model_version_id))
        evaluation_id = uuid.uuid4()
        samples = self._dataset_samples(
            dataset_version_id, {"calibration", "test"})
        row_binding_digest = hashlib.sha256(json.dumps(
            [item["observation_id"] for item in samples],
            separators=(",", ":"),
        ).encode()).hexdigest()
        config_digest = hashlib.sha256(json.dumps({
            "protocol": protocol,
            "model_sha256": model_sha256,
            "preprocess_id": preprocess_id,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        job_id = self._jobs.create(
            task_type="fixed_evaluation",
            required_vram_mb=required_vram_mb,
            required_model_version=model_version,
            max_attempts=3,
            idempotency_key=f"evaluate:{model_version_id}:{dataset_version_id}",
            input_manifest={
                "schema_version": 1,
                "evaluation_run_id": str(evaluation_id),
                "model_version_id": str(model_version_id),
                "model_version": model_version,
                "model_family": model_family,
                "weight_artifact_id": str(weight_artifact_id),
                "model_sha256": model_sha256,
                "feature_dim": feature_dim,
                "preprocess_id": preprocess_id,
                "dataset_version_id": str(dataset_version_id),
                "protocol": protocol,
                "config_digest": config_digest,
                "row_binding_digest": row_binding_digest,
                "production_comparison": {
                    "baseline_model_version": (
                        None if production_model is None
                        else production_model.version),
                    "candidate_model_version": model_version,
                    "comparison_protocol": "same_fixed_test",
                },
                "production_model": (None if production_model is None else {
                    "model_version": production_model.version,
                    "model_family": production_model.model_family,
                    "weight_artifact_id": str(
                        production_model.weight_artifact_id),
                    "model_sha256": production_model.sha256,
                    "feature_dim": production_model.feature_dim,
                }),
                "samples": samples,
            },
        )
        with self._sessions.begin() as db:
            existing = db.scalar(select(EvaluationRun).where(
                EvaluationRun.job_id == job_id))
            if existing is not None:
                return existing.id, job_id
            db.add(EvaluationRun(
                id=evaluation_id, job_id=job_id,
                model_version_id=model_version_id,
                dataset_version_id=dataset_version_id,
                protocol=protocol,
            ))
        return evaluation_id, job_id

    def record_evaluation(
        self,
        evaluation_run_id: uuid.UUID,
        *,
        report_artifact_id: uuid.UUID,
        metrics: dict[str, float],
        production_comparison: dict,
        calibrated_thresholds: dict[str, float],
    ) -> None:
        if not metrics or not production_comparison \
                or not calibrated_thresholds:
            raise ValueError("评估报告、生产比较和阈值标定均不可缺失")
        if any(not math.isfinite(float(value)) for value in metrics.values()):
            raise ValueError("评估指标包含 NaN/Inf")
        thresholds = {key: float(value)
                      for key, value in calibrated_thresholds.items()}
        if any(not math.isfinite(value) or value < -1 or value > 1
               for value in thresholds.values()):
            raise ValueError("标定阈值超出合法相似度范围")
        with self._sessions.begin() as db:
            run = db.get(EvaluationRun, evaluation_run_id,
                         with_for_update=True)
            artifact = db.get(Artifact, report_artifact_id)
            if run is None or artifact is None \
                    or artifact.job_id != run.job_id \
                    or artifact.artifact_type != "evaluation_report" \
                    or db.get(Job, run.job_id).state != JobState.SUCCEEDED:
                raise ValueError("评估报告 Artifact 与 Evaluation Run 不匹配")
            self._verify_artifact_file(artifact)
            report = self._read_json_artifact(artifact)
            expected = {
                "evaluation_run_id": str(run.id),
                "model_version_id": str(run.model_version_id),
                "dataset_version_id": str(run.dataset_version_id),
            }
            if any(str(report.get(key)) != value
                   for key, value in expected.items()):
                raise ValueError("评估报告与 Evaluation Run 行绑定不一致")
            report_metrics = report.get("metrics")
            report_comparison = report.get("production_comparison")
            report_thresholds = report.get("calibrated_thresholds")
            if report_metrics != metrics \
                    or report_comparison != production_comparison \
                    or report_thresholds != calibrated_thresholds:
                raise ValueError("评估指标必须来自 Worker 报告，不接受客户端改写")
            if run.status == "completed":
                if run.report_artifact_id != report_artifact_id:
                    raise ValueError("Evaluation Run 已绑定不同报告")
                return
            for name, value in metrics.items():
                db.add(EvaluationResult(
                    evaluation_run_id=run.id,
                    metric_name=name, split="test", value=float(value),
                    payload={},
                ))
            run.report_artifact_id = report_artifact_id
            run.comparison = production_comparison
            run.calibrated_thresholds = thresholds
            run.status = "completed"
            model = db.get(ModelVersion, run.model_version_id,
                           with_for_update=True)
            model.calibrated_thresholds = thresholds

    def request_promotion(
        self,
        model_version_id: uuid.UUID,
        reviewer_user_id: uuid.UUID,
    ) -> uuid.UUID | None:
        self._require_reviewer(reviewer_user_id)
        with self._sessions.begin() as db:
            model = db.get(ModelVersion, model_version_id,
                           with_for_update=True)
            if model is None or model.status not in {
                    "candidate", "promotion_pending"}:
                raise ValueError("模型不处于可上线候选状态")
            self._verify_model_file(model)
            evaluation = db.scalar(
                select(EvaluationRun)
                .where(
                    EvaluationRun.model_version_id == model.id,
                    EvaluationRun.status == "completed",
                    EvaluationRun.report_artifact_id.is_not(None),
                )
                .order_by(EvaluationRun.created_at.desc())
            )
            if evaluation is None or not evaluation.comparison \
                    or not evaluation.calibrated_thresholds:
                raise ValueError("模型尚未完成固定测试、生产比较和阈值标定评估")
            if model.feature_dim is None:
                pointer = db.get(
                    ProductionModelPointer, model.model_family,
                    with_for_update=True)
                previous_id = None if pointer is None \
                    else pointer.model_version_id
                if pointer is None:
                    db.add(ProductionModelPointer(
                        model_family=model.model_family,
                        model_version_id=model.id))
                else:
                    previous = db.get(
                        ModelVersion, pointer.model_version_id,
                        with_for_update=True)
                    if previous is not None and previous.id != model.id:
                        previous.status = "retired"
                    pointer.model_version_id = model.id
                model.status = "production"
                db.add(ModelPromotionEvent(
                    model_version_id=model.id,
                    event_type="promoted_to_production",
                    actor_user_id=reviewer_user_id,
                    payload={
                        "previous_model_version_id": (
                            str(previous_id) if previous_id else None),
                        "catalog_rebuild_required": False,
                    },
                ))
                return None
            if model.status == "promotion_pending":
                event = db.scalar(select(ModelPromotionEvent).where(
                    ModelPromotionEvent.model_version_id == model.id,
                    ModelPromotionEvent.event_type == "promotion_gates_passed",
                ).order_by(ModelPromotionEvent.created_at.desc()))
                if event is None:
                    raise ValueError("上线待处理模型缺少门禁事件")
                return uuid.UUID(event.payload["catalog_rebuild_job_id"])
            observations = list(db.execute(
                select(Observation.id, Observation.crop_id)
                .join(
                    ConfirmedIndividual,
                    ConfirmedIndividual.id == Observation.individual_id,
                )
                .where(
                    Observation.state == "active",
                    ConfirmedIndividual.state == "active",
                )
                .order_by(Observation.id)
            ))
            rebuild_manifest = {
                "schema_version": model.compatible_index_schema,
                "model_version_id": str(model.id),
                "model_version": model.version,
                "model_family": model.model_family,
                "model_sha256": model.sha256,
                "weight_artifact_id": str(model.weight_artifact_id),
                "feature_dim": model.feature_dim,
                "preprocess_id": model.preprocess_id,
                "crop_config": model.compatible_crop_config,
                "observations": [{
                    "observation_id": str(observation_id),
                    "crop_id": str(crop_id),
                } for observation_id, crop_id in observations],
            }
            rebuild_manifest["row_binding_digest"] = hashlib.sha256(
                json.dumps([
                    item["observation_id"]
                    for item in rebuild_manifest["observations"]
                ], separators=(",", ":")).encode()).hexdigest()
            rebuild_manifest["config_digest"] = hashlib.sha256(json.dumps({
                "model_sha256": model.sha256,
                "feature_dim": model.feature_dim,
                "preprocess_id": model.preprocess_id,
                "crop_config": model.compatible_crop_config,
                "schema_version": model.compatible_index_schema,
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        job_id = self._jobs.create(
            task_type="catalog_rebuild",
            required_vram_mb=4096,
            required_model_version=model.version,
            max_attempts=3,
            idempotency_key=f"catalog-rebuild:{model.id}",
            input_manifest=rebuild_manifest,
        )
        with self._sessions.begin() as db:
            model = db.get(ModelVersion, model_version_id,
                           with_for_update=True)
            model.status = "promotion_pending"
            db.add(ModelPromotionEvent(
                model_version_id=model.id,
                event_type="promotion_gates_passed",
                actor_user_id=reviewer_user_id,
                payload={"catalog_rebuild_job_id": str(job_id)},
            ))
        return job_id

    def ingest_catalog_rebuild(
        self,
        model_version_id: uuid.UUID,
        artifact_id: uuid.UUID,
        reviewer_user_id: uuid.UUID,
    ) -> uuid.UUID:
        """校验 Worker 重建物、原子激活 Catalog，再完成模型指针切换。"""
        self._require_reviewer(reviewer_user_id)
        with self._sessions() as db:
            model = db.get(ModelVersion, model_version_id)
            artifact = db.get(Artifact, artifact_id)
            event = db.scalar(select(ModelPromotionEvent).where(
                ModelPromotionEvent.model_version_id == model_version_id,
                ModelPromotionEvent.event_type == "promotion_gates_passed",
            ).order_by(ModelPromotionEvent.created_at.desc()))
            if model is None or model.status != "promotion_pending" \
                    or artifact is None or event is None \
                    or artifact.artifact_type != "catalog_rebuild" \
                    or artifact.job_id != uuid.UUID(
                        event.payload["catalog_rebuild_job_id"]) \
                    or db.get(Job, artifact.job_id).state != JobState.SUCCEEDED:
                raise ValueError("Catalog 重建 Artifact 与上线申请不匹配")
            self._verify_artifact_file(artifact)
            artifact_path = self._storage.resolve(
                "artifacts", artifact.relative_path)
            expected_model = model.version
            expected_sha = model.sha256
            expected_dim = model.feature_dim
            expected_preprocess = model.preprocess_id
        try:
            with zipfile.ZipFile(artifact_path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                vectors = np.load(io.BytesIO(
                    archive.read("embeddings.npy")), allow_pickle=False)
                index_bytes = archive.read("index.faiss")
        except (KeyError, ValueError, zipfile.BadZipFile) as exc:
            raise ValueError("Catalog 重建 Artifact 结构无效") from exc
        observation_ids = [uuid.UUID(value)
                           for value in manifest.get("observation_ids", [])]
        binding = hashlib.sha256(json.dumps(
            [str(value) for value in observation_ids],
            separators=(",", ":")).encode()).hexdigest()
        if manifest.get("model_version") != expected_model \
                or manifest.get("model_sha256") != expected_sha \
                or manifest.get("feature_dim") != expected_dim \
                or manifest.get("preprocess_id") != expected_preprocess \
                or manifest.get("row_binding_digest") != binding \
                or vectors.shape != (len(observation_ids), expected_dim):
            raise ValueError("Catalog 重建 Artifact 协议或行绑定不一致")
        entries = [CatalogEntry(observation_id, vector)
                   for observation_id, vector in zip(
                       observation_ids, vectors, strict=True)]
        catalogs = CatalogService(self._sessions, self._storage)
        catalog_id = catalogs.stage(
            entries,
            model_version=expected_model,
            calibration_status="calibrated",
            index_bytes=index_bytes,
        )
        catalogs.activate(catalog_id)
        self.complete_promotion(
            model_version_id, catalog_id, reviewer_user_id)
        return catalog_id

    def complete_promotion(
        self,
        model_version_id: uuid.UUID,
        catalog_id: uuid.UUID,
        reviewer_user_id: uuid.UUID,
    ) -> None:
        self._require_reviewer(reviewer_user_id)
        with self._sessions.begin() as db:
            model = db.get(ModelVersion, model_version_id,
                           with_for_update=True)
            catalog = db.get(CatalogVersion, catalog_id)
            active_catalog_id = db.scalar(select(
                ActiveCatalogPointer.catalog_id).where(
                    ActiveCatalogPointer.singleton_id == 1))
            if model is None or model.status != "promotion_pending" \
                    or catalog is None or catalog.status != "active" \
                    or active_catalog_id != catalog_id \
                    or catalog.model_version != model.version \
                    or catalog.feature_dim != model.feature_dim:
                raise ValueError(
                    "Production Model 要求已激活的兼容 Catalog: "
                    f"model_status={None if model is None else model.status}, "
                    f"catalog_status={None if catalog is None else catalog.status}, "
                    f"active_catalog_id={active_catalog_id}, "
                    f"catalog_model={None if catalog is None else catalog.model_version}, "
                    f"model_version={None if model is None else model.version}, "
                    f"catalog_dim={None if catalog is None else catalog.feature_dim}, "
                    f"model_dim={None if model is None else model.feature_dim}"
                )
            pointer = db.get(ProductionModelPointer, model.model_family,
                             with_for_update=True)
            previous_id = None if pointer is None else pointer.model_version_id
            if pointer is None:
                pointer = ProductionModelPointer(
                    model_family=model.model_family,
                    model_version_id=model.id,
                )
                db.add(pointer)
            else:
                previous = db.get(ModelVersion, pointer.model_version_id,
                                  with_for_update=True)
                if previous is not None and previous.id != model.id:
                    previous.status = "retired"
                pointer.model_version_id = model.id
            model.status = "production"
            db.add(ModelPromotionEvent(
                model_version_id=model.id,
                event_type="promoted_to_production",
                actor_user_id=reviewer_user_id,
                catalog_id=catalog_id,
                payload={"previous_model_version_id": str(previous_id)
                         if previous_id else None},
            ))

    def rollback_production(
        self,
        model_version_id: uuid.UUID,
        reviewer_user_id: uuid.UUID,
    ) -> None:
        """将曾经上线的 retired 模型切回 Production。"""
        self._require_reviewer(reviewer_user_id)
        with self._sessions.begin() as db:
            target = db.get(
                ModelVersion, model_version_id, with_for_update=True)
            if target is None or target.status != "retired":
                raise ValueError("模型回滚目标必须是 retired Model Version")
            prior_promotion = db.scalar(select(ModelPromotionEvent.id).where(
                ModelPromotionEvent.model_version_id == target.id,
                ModelPromotionEvent.event_type.in_((
                    "promoted_to_production", "rollback_to_production")),
            ).limit(1))
            if prior_promotion is None:
                raise ValueError("模型从未通过 Production 上线门禁")
            self._verify_model_file(target)
            pointer = db.get(
                ProductionModelPointer, target.model_family,
                with_for_update=True)
            if pointer is None or pointer.model_version_id == target.id:
                raise ValueError("模型族没有可回滚的其他 Production 版本")
            current = db.get(
                ModelVersion, pointer.model_version_id,
                with_for_update=True)
            if current is None or current.status != "production" \
                    or current.model_family != target.model_family:
                raise ValueError("当前 Production Model 指针无效")

            catalog_id = None
            if target.feature_dim is not None:
                catalog_id = db.scalar(select(
                    ActiveCatalogPointer.catalog_id).where(
                        ActiveCatalogPointer.singleton_id == 1))
                catalog = db.get(CatalogVersion, catalog_id) \
                    if catalog_id else None
                if catalog is None or catalog.status != "active" \
                        or catalog.model_version != target.version \
                        or catalog.feature_dim != target.feature_dim:
                    raise ValueError(
                        "Re-ID 模型回滚前必须先激活同版本、同维数 Catalog")

            previous_id = current.id
            current.status = "retired"
            target.status = "production"
            pointer.model_version_id = target.id
            db.add(ModelPromotionEvent(
                model_version_id=target.id,
                event_type="rollback_to_production",
                actor_user_id=reviewer_user_id,
                catalog_id=catalog_id,
                payload={
                    "previous_model_version_id": str(previous_id),
                    "rollback": True,
                },
            ))

    def _dataset_samples(
        self,
        dataset_version_id: uuid.UUID,
        allowed_splits: set[str],
    ) -> list[dict]:
        with self._sessions() as db:
            dataset = db.get(DatasetVersion, dataset_version_id)
            if dataset is None or dataset.status != "frozen":
                raise ValueError("Dataset Version 不存在或未冻结")
            rows = list(db.execute(
                select(DatasetMembership, DatasetSplit, Crop, Image)
                .join(
                    DatasetSplit,
                    (DatasetSplit.dataset_version_id
                     == DatasetMembership.dataset_version_id)
                    & (DatasetSplit.observation_id
                       == DatasetMembership.observation_id),
                )
                .join(Crop, Crop.id == DatasetMembership.crop_id)
                .join(Image, Image.id == DatasetMembership.image_id)
                .where(
                    DatasetMembership.dataset_version_id
                    == dataset_version_id,
                    DatasetSplit.split.in_(allowed_splits),
                )
                .order_by(DatasetMembership.observation_id)
            ))
            samples = [{
                "observation_id": str(membership.observation_id),
                "image_id": str(membership.image_id),
                "crop_id": str(membership.crop_id),
                "individual_id": str(membership.individual_id),
                "label_source": membership.label_source,
                "sequence_key": membership.sequence_key,
                "encounter_key": membership.encounter_key,
                "duplicate_group": membership.duplicate_group,
                "split": split.split,
                "batch_id": str(image.batch_id),
                "image_sha256": image.source_sha256,
                "bbox": [crop.x, crop.y, crop.width, crop.height],
            } for membership, split, crop, image in rows]
        if not samples:
            raise ValueError("Dataset 指定 Split 没有样本")
        return samples

    def _require_reviewer(self, user_id: uuid.UUID) -> None:
        with self._sessions() as db:
            eligible = db.scalar(
                select(UserRole.user_id)
                .join(Role, Role.id == UserRole.role_id)
                .where(UserRole.user_id == user_id, Role.name == "reviewer")
            )
        if eligible is None:
            raise ValueError("模型上线必须由 reviewer 发起")

    def _verify_artifact_file(self, artifact: Artifact) -> None:
        path = self._storage.resolve("artifacts", artifact.relative_path)
        if not path.is_file() or path.stat().st_size != artifact.size_bytes \
                or self._sha256(path) != artifact.sha256:
            raise ValueError("Artifact 文件大小或 SHA-256 校验失败")

    def _verify_model_file(self, model: ModelVersion) -> None:
        path = self._storage.resolve("models", model.weight_path)
        if not path.is_file() or self._sha256(path) != model.sha256:
            raise ValueError("模型权重文件 SHA-256 校验失败")

    def _read_json_artifact(self, artifact: Artifact) -> dict:
        path = self._storage.resolve("artifacts", artifact.relative_path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("评估报告不是合法 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("评估报告根节点必须是对象")
        return value

    @staticmethod
    def _validate_model_manifest(manifest: ModelManifest) -> None:
        values = (
            manifest.model_family, manifest.version, manifest.sha256,
            manifest.preprocess_id, manifest.checkpoint_source,
            manifest.license, manifest.compatible_crop_config,
        )
        if any(not value.strip() for value in values) \
                or len(manifest.sha256) != 64 \
                or manifest.compatible_index_schema <= 0 \
                or (manifest.feature_dim is not None
                    and manifest.feature_dim <= 0):
            raise ValueError("Model Manifest 字段无效")

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
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent)
        os.close(descriptor)
        temp = Path(temp_name)
        try:
            shutil.copyfile(source, temp)
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
