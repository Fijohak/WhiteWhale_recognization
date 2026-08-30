"""盲审任务、追加式原始票和服务端共识投影。"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    ReviewConflict,
    ReviewConsensus,
    ReviewEvent,
    ReviewerRoster,
    ReviewTask,
    Role,
    UserRole,
)
from .review_policy import (
    ReviewDecision,
    ReviewVote,
    decide_cluster_purity,
    decide_identity_match,
)


_POLICIES = {
    "cluster_purity": (1, decide_cluster_purity),
    "identity_match": (3, decide_identity_match),
}
_POLICY_VERSION = "review-policy-v1"


@dataclass(frozen=True)
class ReviewView:
    task_id: uuid.UUID
    task_type: str
    subject_type: str
    subject_id: uuid.UUID
    status: str
    own_votes: tuple[ReviewVote, ...]
    consensus: ReviewDecision | None


class ReviewService:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def create_task(
        self,
        *,
        task_type: str,
        subject_type: str,
        subject_id: uuid.UUID,
        reviewer_ids: list[uuid.UUID],
    ) -> uuid.UUID:
        policy = _POLICIES.get(task_type)
        if policy is None:
            raise ValueError(f"未知审核任务类型: {task_type}")
        required_reviewers = policy[0]
        unique_reviewers = tuple(dict.fromkeys(reviewer_ids))
        if len(unique_reviewers) != required_reviewers:
            raise ValueError(
                f"{task_type} 固定需要 {required_reviewers} 名审核人")

        with self._sessions.begin() as db:
            eligible = set(db.scalars(
                select(UserRole.user_id)
                .join(Role, Role.id == UserRole.role_id)
                .where(
                    Role.name == "reviewer",
                    UserRole.user_id.in_(unique_reviewers),
                )
            ))
            if eligible != set(unique_reviewers):
                raise ValueError("审核名单包含没有 reviewer 角色的用户")
            task = ReviewTask(
                task_type=task_type,
                subject_type=subject_type,
                subject_id=subject_id,
                required_reviewers=required_reviewers,
                policy_version=_POLICY_VERSION,
            )
            db.add(task)
            db.flush()
            db.add_all([
                ReviewerRoster(task_id=task.id, reviewer_user_id=reviewer_id)
                for reviewer_id in unique_reviewers
            ])
            return task.id

    def submit_vote(
        self,
        task_id: uuid.UUID,
        reviewer_id: uuid.UUID,
        vote: ReviewVote,
    ) -> ReviewDecision:
        with self._sessions.begin() as db:
            task = db.get(ReviewTask, task_id, with_for_update=True)
            if task is None:
                raise ValueError("审核任务不存在")
            if task.status != "open":
                raise ValueError("审核任务已经结束")
            roster = db.get(ReviewerRoster, (task_id, reviewer_id))
            if roster is None:
                raise ValueError("用户不在该任务审核名单中")
            revision = (db.scalar(
                select(func.max(ReviewEvent.revision)).where(
                    ReviewEvent.task_id == task_id,
                    ReviewEvent.reviewer_user_id == reviewer_id,
                )) or 0) + 1
            db.add(ReviewEvent(
                task_id=task_id,
                reviewer_user_id=reviewer_id,
                revision=revision,
                decision=self._serialize_vote(vote),
            ))
            db.flush()
            decision = self._calculate(db, task)
            if decision.status == "pending":
                return decision

            task.status = decision.status
            db.add(ReviewConsensus(
                task_id=task.id,
                status=decision.status,
                conclusion=decision.conclusion,
                individual_id=decision.individual_id,
                flags=sorted(decision.flags),
            ))
            if decision.status == "conflict":
                db.add(ReviewConflict(
                    task_id=task.id,
                    reason="review_votes_conflict",
                    detail={"flags": sorted(decision.flags)},
                ))
            return decision

    def view_for_reviewer(
        self,
        task_id: uuid.UUID,
        reviewer_id: uuid.UUID,
    ) -> ReviewView:
        with self._sessions() as db:
            task = db.get(ReviewTask, task_id)
            if task is None:
                raise ValueError("审核任务不存在")
            if db.get(ReviewerRoster, (task_id, reviewer_id)) is None:
                raise ValueError("用户不在该任务审核名单中")
            own_events = list(db.scalars(
                select(ReviewEvent)
                .where(
                    ReviewEvent.task_id == task_id,
                    ReviewEvent.reviewer_user_id == reviewer_id,
                )
                .order_by(ReviewEvent.revision)
            ))
            stored = db.get(ReviewConsensus, task_id)
            consensus = None if stored is None else ReviewDecision(
                status=stored.status,
                conclusion=stored.conclusion,
                individual_id=stored.individual_id,
                flags=frozenset(stored.flags),
            )
            return ReviewView(
                task_id=task.id,
                task_type=task.task_type,
                subject_type=task.subject_type,
                subject_id=task.subject_id,
                status=task.status,
                own_votes=tuple(
                    self._deserialize_vote(event.decision)
                    for event in own_events
                ),
                consensus=consensus,
            )

    @staticmethod
    def _calculate(db: Session, task: ReviewTask) -> ReviewDecision:
        events = list(db.scalars(
            select(ReviewEvent)
            .where(ReviewEvent.task_id == task.id)
            .order_by(ReviewEvent.id)
        ))
        latest: dict[uuid.UUID, ReviewEvent] = {}
        for event in events:
            latest[event.reviewer_user_id] = event
        if len(latest) < task.required_reviewers:
            return ReviewDecision("pending")
        votes = [
            ReviewService._deserialize_vote(event.decision)
            for event in latest.values()
        ]
        return _POLICIES[task.task_type][1](votes)

    @staticmethod
    def _serialize_vote(vote: ReviewVote) -> dict[str, str | None]:
        return {
            "choice": vote.choice,
            "individual_id": str(vote.individual_id)
            if vote.individual_id else None,
        }

    @staticmethod
    def _deserialize_vote(payload: dict) -> ReviewVote:
        individual_id = payload.get("individual_id")
        return ReviewVote(
            choice=payload["choice"],
            individual_id=uuid.UUID(individual_id) if individual_id else None,
        )
