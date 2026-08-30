"""PostgreSQL-backed GPU Job 租约服务。"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    Job,
    JobAttempt,
    JobEvent,
    JobLease,
    WorkerDevice,
    WorkerHeartbeat,
)
from .states import JobState, transition_job_state


class InvalidLease(ValueError):
    """设备令牌与当前有效租约不匹配。"""


class JobConflict(ValueError):
    pass


@dataclass(frozen=True)
class LeaseGrant:
    job_id: uuid.UUID
    attempt_id: uuid.UUID
    attempt_number: int
    lease_token: str
    lease_expires_at: datetime


@dataclass(frozen=True)
class JobSnapshot:
    job_id: uuid.UUID
    task_type: str
    state: JobState
    input_manifest: dict
    required_model_version: str | None


class JobQueueService:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def create(
        self,
        *,
        task_type: str,
        required_vram_mb: int,
        required_model_version: str | None,
        max_attempts: int,
        idempotency_key: str,
        input_manifest: dict,
        priority: int = 0,
        batch_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        if not task_type or len(task_type) > 128:
            raise ValueError("任务类型长度必须为 1–128")
        if required_vram_mb < 0 or max_attempts <= 0:
            raise ValueError("任务资源或重试次数无效")
        if not idempotency_key or len(idempotency_key) > 128:
            raise ValueError("幂等键长度必须为 1–128")
        with self._sessions.begin() as db:
            existing = db.scalar(
                select(Job)
                .where(Job.idempotency_key == idempotency_key)
                .with_for_update()
            )
            if existing is not None:
                expected = (
                    task_type,
                    required_vram_mb,
                    required_model_version,
                    max_attempts,
                    input_manifest,
                    priority,
                    batch_id,
                )
                actual = (
                    existing.task_type,
                    existing.required_vram_mb,
                    existing.required_model_version,
                    existing.max_attempts,
                    existing.input_manifest,
                    existing.priority,
                    existing.batch_id,
                )
                if actual != expected:
                    raise JobConflict("幂等键已被不同任务参数占用")
                return existing.id
            job = Job(
                task_type=task_type,
                required_vram_mb=required_vram_mb,
                required_model_version=required_model_version,
                max_attempts=max_attempts,
                idempotency_key=idempotency_key,
                input_manifest=input_manifest,
                priority=priority,
                batch_id=batch_id,
            )
            db.add(job)
            db.flush()
            db.add(JobEvent(
                job_id=job.id,
                event_type="queued",
                payload={"idempotency_key": idempotency_key},
            ))
            return job.id

    def get(self, job_id: uuid.UUID) -> JobSnapshot:
        with self._sessions() as db:
            job = db.get(Job, job_id)
            if job is None:
                raise ValueError("任务不存在")
            return JobSnapshot(
                job_id=job.id,
                task_type=job.task_type,
                state=job.state,
                input_manifest=job.input_manifest,
                required_model_version=job.required_model_version,
            )

    def assert_lease_device(
        self, job_id: uuid.UUID, device_id: uuid.UUID,
    ) -> None:
        with self._sessions() as db:
            lease_device = db.scalar(select(JobLease.device_id).where(
                JobLease.job_id == job_id))
            if lease_device != device_id:
                raise InvalidLease("任务租约不属于当前 Worker")

    def assert_latest_attempt_device(
        self, job_id: uuid.UUID, device_id: uuid.UUID,
    ) -> None:
        with self._sessions() as db:
            attempt_device = db.scalar(
                select(JobAttempt.worker_device_id)
                .where(JobAttempt.job_id == job_id)
                .order_by(JobAttempt.attempt_number.desc())
                .limit(1)
            )
            if attempt_device != device_id:
                raise InvalidLease("任务 Attempt 不属于当前 Worker")

    def record_worker_heartbeat(
        self,
        device_id: uuid.UUID,
        *,
        available_capacity: int,
        detail: dict | None = None,
    ) -> None:
        if available_capacity < 0:
            raise ValueError("可用容量不能为负数")
        with self._sessions.begin() as db:
            db.add(WorkerHeartbeat(
                device_id=device_id,
                available_capacity=available_capacity,
                detail=detail or {},
            ))


class LeaseService:
    def __init__(self, sessions: sessionmaker[Session], *,
                 lease_duration: timedelta = timedelta(minutes=5)):
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration 必须为正数")
        self._sessions = sessions
        self._lease_duration = lease_duration

    def lease_next(self, device_id: uuid.UUID) -> LeaseGrant | None:
        """为设备原子领取一个兼容任务；无可用任务时返回 None。"""
        now = datetime.now(UTC)
        with self._sessions.begin() as session:
            device = session.scalar(
                select(WorkerDevice)
                .where(WorkerDevice.id == device_id)
                .with_for_update()
            )
            if device is None or not device.is_active:
                return None

            active_leases = session.scalar(
                select(func.count())
                .select_from(JobLease)
                .where(
                    JobLease.device_id == device_id,
                    JobLease.lease_expires_at > now,
                )
            ) or 0
            if active_leases >= device.capacity:
                return None

            filters = [
                Job.state == JobState.QUEUED,
                Job.task_type.in_(tuple(device.capabilities)),
                Job.required_vram_mb <= device.vram_mb,
            ]
            if device.model_versions:
                filters.append(or_(
                    Job.required_model_version.is_(None),
                    Job.required_model_version.in_(tuple(device.model_versions)),
                ))
            else:
                filters.append(Job.required_model_version.is_(None))

            job = session.scalar(
                select(Job)
                .where(*filters)
                .order_by(Job.priority.desc(), Job.created_at, Job.id)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if job is None:
                return None

            previous_attempts = session.scalar(
                select(func.count())
                .select_from(JobAttempt)
                .where(JobAttempt.job_id == job.id)
            ) or 0
            attempt_number = previous_attempts + 1
            if attempt_number > job.max_attempts:
                return None

            attempt = JobAttempt(
                job_id=job.id,
                attempt_number=attempt_number,
                worker_device_id=device.id,
                started_at=now,
            )
            session.add(attempt)
            session.flush()

            token = secrets.token_urlsafe(32)
            token_digest = hashlib.sha256(token.encode("utf-8")).digest()
            attempt.lease_token_digest = token_digest
            expires_at = now + self._lease_duration
            session.add(JobLease(
                job_id=job.id,
                attempt_id=attempt.id,
                device_id=device.id,
                lease_token_digest=token_digest,
                leased_at=now,
                lease_expires_at=expires_at,
                last_heartbeat_at=now,
            ))
            job.state = transition_job_state(job.state, JobState.LEASED)
            session.add(JobEvent(
                job_id=job.id,
                attempt_id=attempt.id,
                event_type="leased",
                payload={
                    "device_id": str(device.id),
                    "attempt_number": attempt_number,
                    "lease_expires_at": expires_at.isoformat(),
                },
            ))

            return LeaseGrant(
                job_id=job.id,
                attempt_id=attempt.id,
                attempt_number=attempt_number,
                lease_token=token,
                lease_expires_at=expires_at,
            )

    def requeue_expired(self, *,
                        now: datetime | None = None) -> list[uuid.UUID]:
        """关闭已过期 Attempt 并把可重试 Job 放回队列。"""
        current_time = now or datetime.now(UTC)
        requeued: list[uuid.UUID] = []
        with self._sessions.begin() as session:
            leases = session.scalars(
                select(JobLease)
                .where(JobLease.lease_expires_at <= current_time)
                .order_by(JobLease.lease_expires_at, JobLease.id)
                .with_for_update(skip_locked=True)
            ).all()
            for lease in leases:
                job = session.get(Job, lease.job_id)
                attempt = session.get(JobAttempt, lease.attempt_id)
                if job is None or attempt is None:
                    raise RuntimeError("租约引用了不存在的 Job/Attempt")

                job.state = transition_job_state(job.state, JobState.LEASE_EXPIRED)
                attempt.outcome = JobState.LEASE_EXPIRED.value
                attempt.finished_at = current_time
                session.add(JobEvent(
                    job_id=job.id,
                    attempt_id=attempt.id,
                    event_type=JobState.LEASE_EXPIRED.value,
                    payload={"device_id": str(lease.device_id)},
                ))
                session.delete(lease)

                if attempt.attempt_number < job.max_attempts:
                    job.state = transition_job_state(
                        job.state, JobState.QUEUED)
                    session.add(JobEvent(
                        job_id=job.id,
                        attempt_id=attempt.id,
                        event_type="requeued",
                        payload={"next_attempt": attempt.attempt_number + 1},
                    ))
                    requeued.append(job.id)
                else:
                    job.state = JobState.FAILED
                    session.add(JobEvent(
                        job_id=job.id,
                        attempt_id=attempt.id,
                        event_type="attempts_exhausted",
                        payload={"max_attempts": job.max_attempts},
                    ))
        return requeued

    def heartbeat(self, job_id: uuid.UUID, lease_token: str, *,
                  now: datetime | None = None) -> datetime:
        """验证当前租约并续期；旧 token、过期 token 均立即拒绝。"""
        current_time = now or datetime.now(UTC)
        supplied_digest = hashlib.sha256(
            lease_token.encode("utf-8")).digest()
        with self._sessions.begin() as session:
            lease = session.scalar(
                select(JobLease)
                .where(JobLease.job_id == job_id)
                .with_for_update()
            )
            if lease is None \
                    or lease.lease_expires_at <= current_time \
                    or not hmac.compare_digest(
                        lease.lease_token_digest, supplied_digest):
                raise InvalidLease("租约无效、已过期或已被重新分配")

            base = max(current_time, lease.lease_expires_at)
            lease.last_heartbeat_at = current_time
            lease.lease_expires_at = base + self._lease_duration
            session.add(JobEvent(
                job_id=job_id,
                attempt_id=lease.attempt_id,
                event_type="heartbeat",
                payload={"lease_expires_at": lease.lease_expires_at.isoformat()},
            ))
            return lease.lease_expires_at

    def validate(
        self,
        job_id: uuid.UUID,
        lease_token: str,
        *,
        device_id: uuid.UUID | None = None,
        now: datetime | None = None,
    ) -> None:
        current_time = now or datetime.now(UTC)
        supplied_digest = hashlib.sha256(
            lease_token.encode("utf-8")).digest()
        with self._sessions() as session:
            lease = session.scalar(select(JobLease).where(
                JobLease.job_id == job_id))
            if lease is None \
                    or lease.lease_expires_at <= current_time \
                    or (device_id is not None and lease.device_id != device_id) \
                    or not hmac.compare_digest(
                        lease.lease_token_digest, supplied_digest):
                raise InvalidLease("租约无效、已过期或已被重新分配")
