"""服务端状态机；客户端只能请求动作，不能直接写最终状态。"""
from __future__ import annotations

from enum import Enum


class BatchStage(str, Enum):
    REGISTERED = "registered"
    CANDIDATE_READY = "candidate_ready"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    CATALOG_STAGED = "catalog_staged"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class JobState(str, Enum):
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    UPLOADING = "uploading"
    VALIDATING = "validating"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LEASE_EXPIRED = "lease_expired"


_BATCH_ORDER = tuple(BatchStage)
_JOB_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset({JobState.LEASED, JobState.CANCELLED}),
    JobState.LEASED: frozenset({
        JobState.RUNNING, JobState.LEASE_EXPIRED, JobState.CANCELLED,
    }),
    JobState.RUNNING: frozenset({
        JobState.UPLOADING, JobState.FAILED,
        JobState.CANCELLED, JobState.LEASE_EXPIRED,
    }),
    JobState.UPLOADING: frozenset({
        JobState.VALIDATING, JobState.FAILED,
        JobState.CANCELLED, JobState.LEASE_EXPIRED,
    }),
    JobState.VALIDATING: frozenset({JobState.SUCCEEDED, JobState.FAILED}),
    JobState.SUCCEEDED: frozenset(),
    JobState.FAILED: frozenset({JobState.QUEUED, JobState.CANCELLED}),
    JobState.CANCELLED: frozenset(),
    JobState.LEASE_EXPIRED: frozenset({JobState.QUEUED, JobState.CANCELLED}),
}


def advance_batch_stage(current: BatchStage, target: BatchStage) -> BatchStage:
    """批次只允许幂等写入或前进到下一个成功业务阶段。"""
    current = BatchStage(current)
    target = BatchStage(target)
    if current == target:
        return target
    current_index = _BATCH_ORDER.index(current)
    if current_index + 1 < len(_BATCH_ORDER) \
            and _BATCH_ORDER[current_index + 1] == target:
        return target
    raise ValueError(f"非法批次阶段变更: {current.value} -> {target.value}")


def transition_job_state(current: JobState, target: JobState) -> JobState:
    """验证任务状态转换；同状态提交视为幂等。"""
    current = JobState(current)
    target = JobState(target)
    if current == target:
        return target
    if target in _JOB_TRANSITIONS[current]:
        return target
    raise ValueError(f"非法任务状态变更: {current.value} -> {target.value}")

