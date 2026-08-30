"""Worker 产物 staging、完整性校验和幂等任务完成。"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .jobs import InvalidLease
from .models import (
    Artifact,
    ArtifactManifest,
    Job,
    JobAttempt,
    JobEvent,
    JobLease,
)
from .states import JobState, transition_job_state
from .storage import StorageLayout


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_ARTIFACT_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")


class ArtifactValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_type: str
    sha256: str
    size_bytes: int
    schema_version: int
    pipeline_config_digest: str
    row_binding_digest: str | None = None
    model_version: str | None = None
    detector_version: str | None = None
    preprocess_id: str | None = None


class WorkerResultService:
    def __init__(self, sessions: sessionmaker[Session], storage: StorageLayout):
        self._sessions = sessions
        self._storage = storage

    def start(self, job_id: uuid.UUID, lease_token: str) -> None:
        now = datetime.now(UTC)
        with self._sessions.begin() as db:
            job, lease, _ = self._current_lease(
                db, job_id, lease_token, now=now)
            job.state = transition_job_state(job.state, JobState.RUNNING)
            db.add(JobEvent(
                job_id=job.id,
                attempt_id=lease.attempt_id,
                event_type="running",
                payload={"device_id": str(lease.device_id)},
            ))

    def submit(
        self,
        job_id: uuid.UUID,
        lease_token: str,
        spec: ArtifactSpec,
        data: bytes,
    ) -> uuid.UUID:
        self._validate_spec(spec, data)
        now = datetime.now(UTC)
        with self._sessions.begin() as db:
            job, lease, _ = self._current_lease(
                db, job_id, lease_token, now=now)
            if job.state not in {JobState.RUNNING, JobState.UPLOADING}:
                raise ArtifactValidationError(
                    f"任务状态 {job.state.value} 不接受产物")
            if (job.required_model_version is not None
                    and spec.model_version != job.required_model_version):
                raise ArtifactValidationError("产物模型版本与任务要求不一致")
            expected_binding = job.input_manifest.get("row_binding_digest")
            if (expected_binding is not None
                    and spec.row_binding_digest != expected_binding):
                raise ArtifactValidationError("产物输入行绑定摘要不一致")

            existing = db.scalar(select(Artifact).where(
                Artifact.job_id == job.id,
                Artifact.artifact_type == spec.artifact_type,
                Artifact.sha256 == spec.sha256.lower(),
            ))
            if existing is not None:
                return existing.id

            relative = (
                Path(str(job.id)) / str(lease.attempt_id)
                / f"{spec.artifact_type}-{spec.sha256.lower()}.bin"
            )
            target = self._storage.resolve("artifacts", relative)
            self._atomic_write(target, data)
            artifact = Artifact(
                job_id=job.id,
                attempt_id=lease.attempt_id,
                artifact_type=spec.artifact_type,
                relative_path=relative.as_posix(),
                sha256=spec.sha256.lower(),
                size_bytes=spec.size_bytes,
                producer_device_id=lease.device_id,
            )
            db.add(artifact)
            db.flush()
            db.add(ArtifactManifest(
                artifact_id=artifact.id,
                schema_version=spec.schema_version,
                row_binding_digest=spec.row_binding_digest,
                model_version=spec.model_version,
                detector_version=spec.detector_version,
                preprocess_id=spec.preprocess_id,
                pipeline_config_digest=spec.pipeline_config_digest.lower(),
                detail={},
            ))
            if job.state == JobState.RUNNING:
                job.state = transition_job_state(job.state, JobState.UPLOADING)
            db.add(JobEvent(
                job_id=job.id,
                attempt_id=lease.attempt_id,
                event_type="artifact_uploaded",
                payload={
                    "artifact_id": str(artifact.id),
                    "artifact_type": spec.artifact_type,
                    "sha256": spec.sha256.lower(),
                },
            ))
            return artifact.id

    def complete(self, job_id: uuid.UUID, lease_token: str) -> uuid.UUID:
        supplied_digest = self._token_digest(lease_token)
        now = datetime.now(UTC)
        with self._sessions.begin() as db:
            job = db.get(Job, job_id, with_for_update=True)
            if job is None:
                raise InvalidLease("任务不存在")
            if job.state == JobState.SUCCEEDED:
                attempt = db.scalar(
                    select(JobAttempt)
                    .where(JobAttempt.job_id == job_id)
                    .order_by(JobAttempt.attempt_number.desc())
                    .limit(1)
                )
                if attempt is None or attempt.lease_token_digest is None \
                        or not hmac.compare_digest(
                            attempt.lease_token_digest, supplied_digest):
                    raise InvalidLease("租约无效、已过期或已被重新分配")
                artifact = db.scalar(
                    select(Artifact)
                    .where(Artifact.attempt_id == attempt.id)
                    .order_by(Artifact.created_at, Artifact.id)
                    .limit(1)
                )
                if artifact is None:
                    raise ArtifactValidationError("成功任务缺少产物")
                return artifact.id

            job, lease, attempt = self._current_lease(
                db, job_id, lease_token, now=now, locked_job=job)
            if job.state == JobState.RUNNING:
                job.state = transition_job_state(job.state, JobState.UPLOADING)
            if job.state != JobState.UPLOADING:
                raise ArtifactValidationError(
                    f"任务状态 {job.state.value} 不能完成")

            artifacts = list(db.scalars(
                select(Artifact)
                .where(Artifact.attempt_id == attempt.id)
                .order_by(Artifact.created_at, Artifact.id)
            ))
            if not artifacts:
                raise ArtifactValidationError("任务没有可验证产物")
            job.state = transition_job_state(job.state, JobState.VALIDATING)
            for artifact in artifacts:
                self._verify_stored_artifact(artifact)
            job.state = transition_job_state(job.state, JobState.SUCCEEDED)
            attempt.outcome = JobState.SUCCEEDED.value
            attempt.finished_at = now
            db.add(JobEvent(
                job_id=job.id,
                attempt_id=attempt.id,
                event_type="succeeded",
                payload={
                    "artifact_ids": [str(artifact.id) for artifact in artifacts],
                },
            ))
            db.delete(lease)
            return artifacts[0].id

    def fail(self, job_id: uuid.UUID, lease_token: str, detail: str) -> None:
        now = datetime.now(UTC)
        with self._sessions.begin() as db:
            job, lease, attempt = self._current_lease(
                db, job_id, lease_token, now=now)
            if job.state == JobState.LEASED:
                job.state = transition_job_state(job.state, JobState.RUNNING)
            if job.state not in {
                JobState.RUNNING, JobState.UPLOADING, JobState.VALIDATING,
            }:
                raise ArtifactValidationError(
                    f"任务状态 {job.state.value} 不能报告失败")
            job.state = transition_job_state(job.state, JobState.FAILED)
            attempt.outcome = JobState.FAILED.value
            attempt.error_detail = detail[:8000]
            attempt.finished_at = now
            db.add(JobEvent(
                job_id=job.id,
                attempt_id=attempt.id,
                event_type="failed",
                payload={"detail": detail[:2000]},
            ))
            db.delete(lease)

    def _current_lease(
        self,
        db: Session,
        job_id: uuid.UUID,
        lease_token: str,
        *,
        now: datetime,
        locked_job: Job | None = None,
    ) -> tuple[Job, JobLease, JobAttempt]:
        lease = db.scalar(
            select(JobLease)
            .where(JobLease.job_id == job_id)
            .with_for_update()
        )
        supplied_digest = self._token_digest(lease_token)
        if lease is None or lease.lease_expires_at <= now \
                or not hmac.compare_digest(
                    lease.lease_token_digest, supplied_digest):
            raise InvalidLease("租约无效、已过期或已被重新分配")
        job = locked_job or db.get(Job, job_id, with_for_update=True)
        attempt = db.get(JobAttempt, lease.attempt_id)
        if job is None or attempt is None:
            raise RuntimeError("租约引用了不存在的 Job/Attempt")
        return job, lease, attempt

    def _verify_stored_artifact(self, artifact: Artifact) -> None:
        path = self._storage.resolve("artifacts", artifact.relative_path)
        digest = hashlib.sha256()
        total = 0
        try:
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    total += len(chunk)
        except FileNotFoundError as exc:
            raise ArtifactValidationError("产物文件不存在") from exc
        if total != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
            raise ArtifactValidationError("已保存产物的 SHA-256 或大小不一致")

    @staticmethod
    def _validate_spec(spec: ArtifactSpec, data: bytes) -> None:
        if not _ARTIFACT_TYPE_PATTERN.fullmatch(spec.artifact_type):
            raise ArtifactValidationError("非法 Artifact 类型")
        if spec.size_bytes < 0 or spec.size_bytes != len(data):
            raise ArtifactValidationError("Artifact 大小不一致")
        for label, value in (
            ("SHA-256", spec.sha256),
            ("Pipeline Config 摘要", spec.pipeline_config_digest),
        ):
            if not _SHA256_PATTERN.fullmatch(value):
                raise ArtifactValidationError(f"{label} 格式错误")
        if spec.row_binding_digest is not None \
                and not _SHA256_PATTERN.fullmatch(spec.row_binding_digest):
            raise ArtifactValidationError("行绑定摘要格式错误")
        if spec.schema_version <= 0:
            raise ArtifactValidationError("Schema Version 必须大于 0")
        if hashlib.sha256(data).hexdigest() != spec.sha256.lower():
            raise ArtifactValidationError("Artifact SHA-256 校验失败")

    @staticmethod
    def _token_digest(value: str) -> bytes:
        return hashlib.sha256(value.encode("utf-8")).digest()

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".artifact-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_name, target)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
