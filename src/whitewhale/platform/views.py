"""归档网页使用的只读投影；所有最终状态仍由领域服务写入。"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session, aliased, sessionmaker

from .models import (
    Batch,
    Artifact,
    ArtifactManifest,
    CandidateCluster,
    CandidateClusterMember,
    ConfirmedIndividual,
    CooccurrenceEvent,
    CooccurrenceMember,
    Crop,
    DatasetMembership,
    DatasetSplit,
    DatasetVersion,
    EvaluationRun,
    EvaluationResult,
    Image,
    IdentityChangeProposal,
    IndividualAlias,
    MatchCandidate,
    ModelVersion,
    ModelPromotionEvent,
    Observation,
    ProductionModelPointer,
    RelationshipEvidence,
    RelationshipHypothesis,
    ReviewerRoster,
    ReviewTask,
    TrainingRun,
    Job,
    JobAttempt,
    JobEvent,
    JobLease,
    Role,
    User,
    UserRole,
    WorkerDevice,
    WorkerHeartbeat,
)


class ArchiveReadService:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def batches(self) -> list[dict]:
        with self._sessions() as db:
            rows = list(db.execute(
                select(
                    Batch,
                    func.count(func.distinct(Image.id)),
                    func.count(func.distinct(Crop.id)),
                    func.count(func.distinct(CandidateCluster.id)),
                )
                .outerjoin(Image, Image.batch_id == Batch.id)
                .outerjoin(Crop, Crop.image_id == Image.id)
                .outerjoin(
                    CandidateCluster, CandidateCluster.batch_id == Batch.id)
                .group_by(Batch.id)
                .order_by(Batch.created_at.desc())
            ))
            return [{
                "batch_id": str(batch.id),
                "name": batch.name,
                "stage": batch.stage.value,
                "source_format": batch.source_format,
                "image_count": image_count,
                "crop_count": crop_count,
                "cluster_count": cluster_count,
                "created_at": batch.created_at.isoformat(),
            } for batch, image_count, crop_count, cluster_count in rows]

    def dashboard(self) -> dict:
        online_after = datetime.now(UTC) - timedelta(minutes=5)
        with self._sessions() as db:
            batch_counts = {
                stage.value: count for stage, count in db.execute(
                    select(Batch.stage, func.count()).group_by(Batch.stage))
            }
            job_counts = {
                state.value: count for state, count in db.execute(
                    select(Job.state, func.count()).group_by(Job.state))
            }
            worker_total = db.scalar(
                select(func.count()).select_from(WorkerDevice)) or 0
            online_workers = db.scalar(select(func.count(func.distinct(
                WorkerHeartbeat.device_id))).where(
                    WorkerHeartbeat.reported_at >= online_after)) or 0
            pending_reviews = db.scalar(select(func.count()).select_from(
                ReviewTask).where(ReviewTask.status == "open")) or 0
            failed_jobs = sum(job_counts.get(state, 0) for state in (
                "failed", "cancelled", "lease_expired"))
            queued_jobs = job_counts.get("queued", 0)
            return {
                "batch_counts": batch_counts,
                "job_counts": job_counts,
                "worker_counts": {
                    "total": worker_total, "online": online_workers},
                "pending_reviews": pending_reviews,
                "failed_jobs": failed_jobs,
                "queued_jobs": queued_jobs,
            }

    def jobs(self, *, limit: int = 200) -> list[dict]:
        if limit <= 0 or limit > 1000:
            raise ValueError("Job 查询 limit 必须为 1–1000")
        with self._sessions() as db:
            jobs = list(db.scalars(
                select(Job).order_by(Job.created_at.desc()).limit(limit)))
            result = []
            for job in jobs:
                attempts = list(db.scalars(select(JobAttempt).where(
                    JobAttempt.job_id == job.id).order_by(
                        JobAttempt.attempt_number)))
                artifact_count = db.scalar(select(func.count()).select_from(
                    Artifact).where(Artifact.job_id == job.id)) or 0
                lease = db.scalar(select(JobLease).where(
                    JobLease.job_id == job.id))
                latest = attempts[-1] if attempts else None
                result.append({
                    "job_id": str(job.id),
                    "batch_id": str(job.batch_id) if job.batch_id else None,
                    "task_type": job.task_type,
                    "state": job.state.value,
                    "priority": job.priority,
                    "required_vram_mb": job.required_vram_mb,
                    "required_model_version": job.required_model_version,
                    "attempt_count": len(attempts),
                    "artifact_count": artifact_count,
                    "current_worker_id": str(lease.device_id)
                    if lease else None,
                    "lease_expires_at": lease.lease_expires_at.isoformat()
                    if lease else None,
                    "last_error": latest.error_detail if latest else None,
                    "created_at": job.created_at.isoformat(),
                })
            return result

    def job(self, job_id: uuid.UUID) -> dict:
        with self._sessions() as db:
            job = db.get(Job, job_id)
            if job is None:
                raise ValueError("Job 不存在")
            attempts = list(db.scalars(select(JobAttempt).where(
                JobAttempt.job_id == job_id).order_by(
                    JobAttempt.attempt_number)))
            artifacts = list(db.execute(
                select(Artifact, ArtifactManifest)
                .outerjoin(
                    ArtifactManifest,
                    ArtifactManifest.artifact_id == Artifact.id)
                .where(Artifact.job_id == job_id)
                .order_by(Artifact.created_at, Artifact.id)
            ))
            events = list(db.scalars(select(JobEvent).where(
                JobEvent.job_id == job_id).order_by(
                    JobEvent.created_at, JobEvent.id)))
            return {
                "job_id": str(job.id),
                "batch_id": str(job.batch_id) if job.batch_id else None,
                "task_type": job.task_type,
                "state": job.state.value,
                "priority": job.priority,
                "required_vram_mb": job.required_vram_mb,
                "required_model_version": job.required_model_version,
                "max_attempts": job.max_attempts,
                "idempotency_key": job.idempotency_key,
                "input_manifest": job.input_manifest,
                "created_at": job.created_at.isoformat(),
                "attempts": [{
                    "attempt_id": str(attempt.id),
                    "attempt_number": attempt.attempt_number,
                    "worker_device_id": str(attempt.worker_device_id)
                    if attempt.worker_device_id else None,
                    "started_at": attempt.started_at.isoformat()
                    if attempt.started_at else None,
                    "finished_at": attempt.finished_at.isoformat()
                    if attempt.finished_at else None,
                    "outcome": attempt.outcome,
                    "error_detail": attempt.error_detail,
                } for attempt in attempts],
                "artifacts": [{
                    "artifact_id": str(artifact.id),
                    "attempt_id": str(artifact.attempt_id),
                    "artifact_type": artifact.artifact_type,
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                    "producer_device_id": str(artifact.producer_device_id),
                    "schema_version": manifest.schema_version
                    if manifest else None,
                    "model_version": manifest.model_version
                    if manifest else None,
                    "detector_version": manifest.detector_version
                    if manifest else None,
                    "preprocess_id": manifest.preprocess_id
                    if manifest else None,
                    "pipeline_config_digest":
                    manifest.pipeline_config_digest if manifest else None,
                    "row_binding_digest": manifest.row_binding_digest
                    if manifest else None,
                    "created_at": artifact.created_at.isoformat(),
                } for artifact, manifest in artifacts],
                "events": [{
                    "event_id": event.id,
                    "attempt_id": str(event.attempt_id)
                    if event.attempt_id else None,
                    "event_type": event.event_type,
                    "payload": event.payload,
                    "created_at": event.created_at.isoformat(),
                } for event in events],
            }

    def workers(self) -> list[dict]:
        online_after = datetime.now(UTC) - timedelta(minutes=5)
        with self._sessions() as db:
            devices = list(db.scalars(select(WorkerDevice).order_by(
                WorkerDevice.name)))
            result = []
            for device in devices:
                heartbeat = db.scalar(select(WorkerHeartbeat).where(
                    WorkerHeartbeat.device_id == device.id).order_by(
                        WorkerHeartbeat.reported_at.desc()).limit(1))
                lease = db.scalar(select(JobLease).where(
                    JobLease.device_id == device.id).order_by(
                        JobLease.leased_at.desc()).limit(1))
                attempts = db.scalar(select(func.count()).select_from(
                    JobAttempt).where(
                        JobAttempt.worker_device_id == device.id)) or 0
                result.append({
                    "device_id": str(device.id),
                    "name": device.name,
                    "gpu_model": device.gpu_model,
                    "vram_mb": device.vram_mb,
                    "cuda_version": device.cuda_version,
                    "worker_version": device.worker_version,
                    "capabilities": device.capabilities,
                    "model_versions": device.model_versions,
                    "capacity": device.capacity,
                    "is_active": device.is_active,
                    "is_online": bool(
                        device.is_active and heartbeat is not None
                        and heartbeat.reported_at >= online_after),
                    "last_heartbeat_at": heartbeat.reported_at.isoformat()
                    if heartbeat else None,
                    "available_capacity": heartbeat.available_capacity
                    if heartbeat else None,
                    "current_job_id": str(lease.job_id) if lease else None,
                    "attempt_count": attempts,
                })
            return result

    def users(self) -> list[dict]:
        with self._sessions() as db:
            users = list(db.scalars(select(User).order_by(User.username)))
            return [{
                "user_id": str(user.id),
                "username": user.username,
                "is_active": user.is_active,
                "roles": sorted(db.scalars(
                    select(Role.name)
                    .join(UserRole, UserRole.role_id == Role.id)
                    .where(UserRole.user_id == user.id))),
                "created_at": user.created_at.isoformat(),
            } for user in users]

    def batch(self, batch_id: uuid.UUID) -> dict:
        with self._sessions() as db:
            batch = db.get(Batch, batch_id)
            if batch is None:
                raise ValueError("批次不存在")
            images = db.scalar(select(func.count()).select_from(Image).where(
                Image.batch_id == batch_id)) or 0
            crops = db.scalar(select(func.count()).select_from(Crop).join(
                Image, Image.id == Crop.image_id).where(
                    Image.batch_id == batch_id)) or 0
            jobs = list(db.scalars(select(Job).where(
                Job.batch_id == batch_id).order_by(Job.created_at)))
            return {
                "batch_id": str(batch.id),
                "name": batch.name,
                "stage": batch.stage.value,
                "source_format": batch.source_format,
                "manifest_sha256": batch.manifest_sha256,
                "metadata": batch.metadata_json,
                "image_count": images,
                "crop_count": crops,
                "jobs": [{
                    "job_id": str(job.id),
                    "task_type": job.task_type,
                    "state": job.state.value,
                    "created_at": job.created_at.isoformat(),
                } for job in jobs],
                "created_at": batch.created_at.isoformat(),
            }

    def review_inbox(self, reviewer_id: uuid.UUID) -> list[dict]:
        with self._sessions() as db:
            tasks = list(db.scalars(
                select(ReviewTask)
                .join(ReviewerRoster, ReviewerRoster.task_id == ReviewTask.id)
                .where(
                    ReviewerRoster.reviewer_user_id == reviewer_id,
                    ReviewTask.status == "open",
                )
                .order_by(ReviewTask.created_at, ReviewTask.id)
            ))
            return [{
                "task_id": str(task.id),
                "task_type": task.task_type,
                "subject_type": task.subject_type,
                "subject_id": str(task.subject_id),
                "status": task.status,
                "created_at": task.created_at.isoformat(),
            } for task in tasks]

    def candidate(self, cluster_id: uuid.UUID) -> dict:
        with self._sessions() as db:
            cluster = db.get(CandidateCluster, cluster_id)
            if cluster is None:
                raise ValueError("候选簇不存在")
            crops = list(db.execute(
                select(Crop, CandidateClusterMember)
                .join(
                    CandidateClusterMember,
                    CandidateClusterMember.crop_id == Crop.id,
                )
                .where(CandidateClusterMember.cluster_id == cluster_id)
                .order_by(Crop.id)
            ))
            matches = list(db.scalars(
                select(MatchCandidate)
                .where(MatchCandidate.cluster_id == cluster_id)
                .order_by(MatchCandidate.rank)
            ))
            return {
                "cluster_id": str(cluster.id),
                "batch_id": str(cluster.batch_id),
                "label": cluster.label,
                "state": cluster.state,
                "metadata": cluster.metadata_json,
                "crops": [{
                    "crop_id": str(crop.id),
                    "image_id": str(crop.image_id),
                    "media_url": f"/api/media/crops/{crop.id}",
                    "membership_score": member.membership_score,
                    "is_excluded": member.is_excluded,
                } for crop, member in crops],
                "matches": [{
                    "individual_id": str(match.individual_id),
                    "rank": match.rank,
                    "score": match.score,
                    "support_frames": match.support_frames,
                    "model_version": match.model_version,
                } for match in matches],
            }

    def cooccurrence(self, event_id: uuid.UUID) -> dict:
        with self._sessions() as db:
            event = db.get(CooccurrenceEvent, event_id)
            if event is None:
                raise ValueError("共现事件不存在")
            rows = list(db.execute(
                select(CooccurrenceMember, Crop, ConfirmedIndividual)
                .join(Crop, Crop.id == CooccurrenceMember.crop_id)
                .outerjoin(
                    ConfirmedIndividual,
                    ConfirmedIndividual.id == CooccurrenceMember.individual_id,
                )
                .where(CooccurrenceMember.event_id == event_id)
                .order_by(Crop.crop_index, Crop.id)
            ))
            return {
                "event_id": str(event.id),
                "image_id": str(event.image_id),
                "image_media_url": f"/api/media/images/{event.image_id}",
                "status": event.status,
                "source": event.source,
                "crops": [{
                    "crop_id": str(crop.id),
                    "crop_index": crop.crop_index,
                    "media_url": f"/api/media/crops/{crop.id}",
                    "membership_status": member.membership_status,
                    "individual_id": str(individual.id) if individual else None,
                    "individual_name": individual.display_name
                    if individual else None,
                } for member, crop, individual in rows],
            }

    def relationships(self) -> list[dict]:
        low = aliased(ConfirmedIndividual)
        high = aliased(ConfirmedIndividual)
        with self._sessions() as db:
            rows = list(db.execute(
                select(
                    RelationshipHypothesis,
                    low.display_name,
                    high.display_name,
                    func.count(RelationshipEvidence.id),
                )
                .join(low, low.id == RelationshipHypothesis.individual_low_id)
                .join(high, high.id == RelationshipHypothesis.individual_high_id)
                .outerjoin(
                    RelationshipEvidence,
                    RelationshipEvidence.hypothesis_id
                    == RelationshipHypothesis.id,
                )
                .group_by(RelationshipHypothesis.id, low.id, high.id)
                .order_by(RelationshipHypothesis.created_at.desc())
            ))
            return [{
                "hypothesis_id": str(hypothesis.id),
                "individual_low_id": str(hypothesis.individual_low_id),
                "individual_low_name": low_name,
                "individual_high_id": str(hypothesis.individual_high_id),
                "individual_high_name": high_name,
                "relationship_type": hypothesis.relationship_type,
                "status": hypothesis.status,
                "evidence_count": evidence_count,
                "created_at": hypothesis.created_at.isoformat(),
            } for hypothesis, low_name, high_name, evidence_count in rows]

    def identity_change(self, proposal_id: uuid.UUID) -> dict:
        with self._sessions() as db:
            proposal = db.get(IdentityChangeProposal, proposal_id)
            if proposal is None:
                raise ValueError("身份更正方案不存在")
            return {
                "proposal_id": str(proposal.id),
                "change_type": proposal.change_type,
                "status": proposal.status,
                "plan": proposal.plan,
                "plan_digest": proposal.plan_digest,
                "created_by_user_id": str(proposal.created_by_user_id),
                "created_at": proposal.created_at.isoformat(),
                "applied_at": proposal.applied_at.isoformat()
                if proposal.applied_at else None,
            }

    def datasets(self) -> list[dict]:
        with self._sessions() as db:
            datasets = list(db.scalars(select(DatasetVersion).order_by(
                DatasetVersion.created_at.desc())))
            result = []
            for dataset in datasets:
                split_counts = {
                    split: count for split, count in db.execute(
                        select(DatasetSplit.split, func.count())
                        .where(DatasetSplit.dataset_version_id == dataset.id)
                        .group_by(DatasetSplit.split)
                    )
                }
                result.append({
                    "dataset_version_id": str(dataset.id),
                    "name": dataset.name,
                    "protocol": dataset.protocol,
                    "status": dataset.status,
                    "membership_digest": dataset.membership_digest,
                    "sample_count": sum(split_counts.values()),
                    "split_counts": split_counts,
                    "created_at": dataset.created_at.isoformat(),
                })
            return result

    def training_runs(self) -> list[dict]:
        with self._sessions() as db:
            rows = list(db.execute(
                select(TrainingRun, Job)
                .join(Job, Job.id == TrainingRun.job_id)
                .order_by(TrainingRun.created_at.desc())
            ))
            return [{
                "training_run_id": str(run.id),
                "job_id": str(run.job_id),
                "dataset_version_id": str(run.dataset_version_id),
                "task_type": run.task_type,
                "model_family": run.model_family,
                "state": run.state,
                "job_state": job.state.value,
                "seed": run.seed,
                "created_at": run.created_at.isoformat(),
            } for run, job in rows]

    def models(self) -> list[dict]:
        with self._sessions() as db:
            rows = list(db.execute(
                select(ModelVersion, func.count(EvaluationRun.id))
                .outerjoin(
                    EvaluationRun,
                    and_(
                        EvaluationRun.model_version_id == ModelVersion.id,
                        EvaluationRun.status == "completed",
                    ),
                )
                .group_by(ModelVersion.id)
                .order_by(ModelVersion.created_at.desc())
            ))
            return [{
                "model_version_id": str(model.id),
                "model_family": model.model_family,
                "version": model.version,
                "status": model.status,
                "sha256": model.sha256,
                "feature_dim": model.feature_dim,
                "preprocess_id": model.preprocess_id,
                "calibrated_thresholds": model.calibrated_thresholds,
                "completed_evaluations": count,
                "created_at": model.created_at.isoformat(),
            } for model, count in rows]

    def model(self, model_version_id: uuid.UUID) -> dict:
        with self._sessions() as db:
            model = db.get(ModelVersion, model_version_id)
            if model is None:
                raise ValueError("Model Version 不存在")
            pointer = db.get(ProductionModelPointer, model.model_family)
            evaluations = list(db.execute(
                select(EvaluationRun, Job)
                .join(Job, Job.id == EvaluationRun.job_id)
                .where(EvaluationRun.model_version_id == model.id)
                .order_by(EvaluationRun.created_at.desc())
            ))
            evaluation_bodies = []
            for evaluation, job in evaluations:
                metrics = {
                    result.metric_name: result.value
                    for result in db.scalars(select(EvaluationResult).where(
                        EvaluationResult.evaluation_run_id == evaluation.id)
                        .order_by(EvaluationResult.metric_name))
                }
                evaluation_bodies.append({
                    "evaluation_run_id": str(evaluation.id),
                    "job_id": str(evaluation.job_id),
                    "dataset_version_id": str(evaluation.dataset_version_id),
                    "protocol": evaluation.protocol,
                    "status": evaluation.status,
                    "job_state": job.state.value,
                    "report_artifact_id": str(evaluation.report_artifact_id)
                    if evaluation.report_artifact_id else None,
                    "metrics": metrics,
                    "comparison": evaluation.comparison,
                    "calibrated_thresholds":
                    evaluation.calibrated_thresholds,
                    "created_at": evaluation.created_at.isoformat(),
                })
            events = list(db.scalars(select(ModelPromotionEvent).where(
                ModelPromotionEvent.model_version_id == model.id).order_by(
                    ModelPromotionEvent.created_at)))
            return {
                "model_version_id": str(model.id),
                "training_run_id": str(model.training_run_id),
                "weight_artifact_id": str(model.weight_artifact_id),
                "model_family": model.model_family,
                "version": model.version,
                "status": model.status,
                "is_production": bool(
                    pointer and pointer.model_version_id == model.id),
                "sha256": model.sha256,
                "feature_dim": model.feature_dim,
                "preprocess_id": model.preprocess_id,
                "checkpoint_source": model.checkpoint_source,
                "license": model.license,
                "compatible_detector_version":
                model.compatible_detector_version,
                "compatible_crop_config": model.compatible_crop_config,
                "compatible_index_schema": model.compatible_index_schema,
                "calibrated_thresholds": model.calibrated_thresholds,
                "evaluations": evaluation_bodies,
                "promotion_events": [{
                    "event_id": str(event.id),
                    "event_type": event.event_type,
                    "actor_user_id": str(event.actor_user_id),
                    "catalog_id": str(event.catalog_id)
                    if event.catalog_id else None,
                    "payload": event.payload,
                    "created_at": event.created_at.isoformat(),
                } for event in events],
                "created_at": model.created_at.isoformat(),
            }

    def individuals(self) -> list[dict]:
        with self._sessions() as db:
            rows = list(db.execute(
                select(ConfirmedIndividual, func.count(Observation.id))
                .outerjoin(
                    Observation,
                    and_(
                        Observation.individual_id == ConfirmedIndividual.id,
                        Observation.state == "active",
                    ),
                )
                .group_by(ConfirmedIndividual.id)
                .order_by(ConfirmedIndividual.display_name)
            ))
            return [{
                "individual_id": str(individual.id),
                "display_name": individual.display_name,
                "state": individual.state,
                "flags": individual.flags,
                "observation_count": count,
            } for individual, count in rows]

    def individual(self, individual_id: uuid.UUID) -> dict:
        with self._sessions() as db:
            individual = db.get(ConfirmedIndividual, individual_id)
            if individual is None:
                raise ValueError("正式个体不存在")
            aliases = list(db.scalars(select(IndividualAlias).where(
                IndividualAlias.individual_id == individual_id)))
            observations = list(db.execute(
                select(Observation, Crop)
                .join(Crop, Crop.id == Observation.crop_id)
                .where(Observation.individual_id == individual_id)
                .order_by(Observation.created_at.desc())
            ))
            return {
                "individual_id": str(individual.id),
                "display_name": individual.display_name,
                "state": individual.state,
                "flags": individual.flags,
                "aliases": [{"alias": item.alias, "source": item.source}
                            for item in aliases],
                "observations": [{
                    "observation_id": str(observation.id),
                    "crop_id": str(crop.id),
                    "media_url": f"/api/media/crops/{crop.id}",
                    "side": observation.side,
                    "quality": observation.quality,
                    "state": observation.state,
                    "observed_at": observation.observed_at.isoformat()
                    if observation.observed_at else None,
                } for observation, crop in observations],
            }
