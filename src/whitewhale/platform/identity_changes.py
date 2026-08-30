"""经 3/3 盲审应用身份合并、拆分和照片撤回，不删除历史记录。"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .cooccurrence import CooccurrenceService
from .models import (
    ConfirmedIndividual,
    CooccurrenceMember,
    IdentityChangeEvent,
    IdentityChangeProposal,
    IndividualAlias,
    Observation,
    ReviewerRoster,
    RelationshipEvidence,
    RelationshipEvent,
    RelationshipHypothesis,
    ReviewConsensus,
    ReviewTask,
    Role,
    UserRole,
)


_TASK_TYPES = {
    "merge": "identity_merge",
    "split": "identity_split",
    "withdrawal": "observation_withdrawal",
}


@dataclass(frozen=True)
class IdentityChangeResult:
    proposal_id: uuid.UUID
    status: str
    created_individual_ids: tuple[uuid.UUID, ...] = ()
    affected_observation_ids: tuple[uuid.UUID, ...] = ()


class IdentityChangeService:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def create_merge(
        self,
        source_individual_ids: list[uuid.UUID],
        *,
        target_individual_id: uuid.UUID,
        reviewer_ids: list[uuid.UUID],
        actor_user_id: uuid.UUID,
    ) -> tuple[uuid.UUID, uuid.UUID]:
        sources = sorted(set(source_individual_ids), key=str)
        if len(sources) < 2 or target_individual_id not in sources:
            raise ValueError("合并方案至少包含两个身份且目标必须在来源中")
        plan = {
            "source_individual_ids": [str(value) for value in sources],
            "target_individual_id": str(target_individual_id),
        }
        with self._sessions.begin() as db:
            individuals = list(db.scalars(select(ConfirmedIndividual).where(
                ConfirmedIndividual.id.in_(sources))))
            if len(individuals) != len(sources) \
                    or any(item.state != "active" for item in individuals):
                raise ValueError("合并方案包含不存在或非 active 的正式身份")
            return self._create(db, "merge", plan, reviewer_ids, actor_user_id)

    def create_split(
        self,
        source_individual_id: uuid.UUID,
        *,
        assignments: dict[uuid.UUID, str],
        reviewer_ids: list[uuid.UUID],
        actor_user_id: uuid.UUID,
    ) -> tuple[uuid.UUID, uuid.UUID]:
        if len(set(assignments.values())) < 2 \
                or any(not value.strip() for value in assignments.values()):
            raise ValueError("拆分方案至少需要两个非空分组")
        with self._sessions.begin() as db:
            source = db.get(ConfirmedIndividual, source_individual_id)
            current_ids = set(db.scalars(select(Observation.id).where(
                Observation.individual_id == source_individual_id,
                Observation.state == "active",
            )))
            if source is None or source.state != "active" \
                    or current_ids != set(assignments):
                raise ValueError("拆分方案必须完整覆盖当前 active Observation")
            plan = {
                "source_individual_id": str(source_individual_id),
                "assignments": [{
                    "observation_id": str(observation_id),
                    "group": assignments[observation_id].strip(),
                } for observation_id in sorted(assignments, key=str)],
            }
            return self._create(db, "split", plan, reviewer_ids, actor_user_id)

    def create_withdrawal(
        self,
        observation_ids: list[uuid.UUID],
        *,
        reviewer_ids: list[uuid.UUID],
        actor_user_id: uuid.UUID,
    ) -> tuple[uuid.UUID, uuid.UUID]:
        values = sorted(set(observation_ids), key=str)
        if not values or len(values) != len(observation_ids):
            raise ValueError("撤回方案必须包含不重复的 Observation")
        with self._sessions.begin() as db:
            rows = list(db.scalars(select(Observation).where(
                Observation.id.in_(values))))
            if len(rows) != len(values) \
                    or any(item.state != "active" for item in rows):
                raise ValueError("撤回方案包含不存在或已撤回的 Observation")
            plan = {"observation_ids": [str(value) for value in values]}
            return self._create(
                db, "withdrawal", plan, reviewer_ids, actor_user_id)

    def apply_review(self, task_id: uuid.UUID) -> IdentityChangeResult:
        affected_crops: list[uuid.UUID] = []
        with self._sessions.begin() as db:
            task = db.get(ReviewTask, task_id, with_for_update=True)
            proposal = None if task is None else db.get(
                IdentityChangeProposal, task.subject_id, with_for_update=True)
            consensus = None if task is None else db.get(
                ReviewConsensus, task.id)
            if task is None or proposal is None or consensus is None \
                    or task.subject_type != "identity_change_proposal" \
                    or task.task_type != _TASK_TYPES.get(proposal.change_type) \
                    or task.status not in {"resolved", "conflict"}:
                raise ValueError("身份更正审核尚未形成结论")
            existing = db.scalar(select(IdentityChangeEvent).where(
                IdentityChangeEvent.proposal_id == proposal.id))
            if existing is not None:
                return self._result(existing, proposal.status)
            if task.status == "conflict":
                proposal.status = "disputed"
                event = self._event(
                    proposal, task, "identity_change_disputed", {})
                db.add(event)
                db.flush()
                return self._result(event, proposal.status)
            if consensus.conclusion == "identity_change_rejected":
                proposal.status = "rejected"
                event = self._event(
                    proposal, task, "identity_change_rejected", {})
                db.add(event)
                db.flush()
                return self._result(event, proposal.status)
            if consensus.conclusion != "identity_change_approved":
                raise ValueError("身份更正共识不可应用")

            if proposal.change_type == "merge":
                payload, affected_crops = self._apply_merge(db, proposal.plan)
            elif proposal.change_type == "split":
                payload, affected_crops = self._apply_split(db, proposal.plan)
            else:
                payload, affected_crops = self._apply_withdrawal(
                    db, proposal.plan)
            self._invalidate_relationships(
                db, affected_crops, proposal.id)
            proposal.status = "applied"
            proposal.applied_at = datetime.now(timezone.utc)
            event = self._event(
                proposal, task, f"identity_{proposal.change_type}_applied",
                payload,
            )
            db.add(event)
            db.flush()
            result = self._result(event, proposal.status)
        CooccurrenceService(self._sessions).project_ready_for_crops(
            affected_crops)
        return result

    @staticmethod
    def _create(
        db: Session,
        change_type: str,
        plan: dict,
        reviewer_ids: list[uuid.UUID],
        actor_user_id: uuid.UUID,
    ) -> tuple[uuid.UUID, uuid.UUID]:
        reviewers = list(dict.fromkeys(reviewer_ids))
        if len(reviewers) != 3:
            raise ValueError("身份更正固定需要 3 名审核人")
        eligible = set(db.scalars(
            select(UserRole.user_id)
            .join(Role, Role.id == UserRole.role_id)
            .where(Role.name == "reviewer", UserRole.user_id.in_(reviewers))
        ))
        if eligible != set(reviewers):
            raise ValueError("审核名单包含没有 reviewer 角色的用户")
        canonical = json.dumps(
            plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
        proposal = IdentityChangeProposal(
            change_type=change_type,
            plan=plan,
            plan_digest=hashlib.sha256(canonical).hexdigest(),
            created_by_user_id=actor_user_id,
        )
        db.add(proposal)
        db.flush()
        task = ReviewTask(
            task_type=_TASK_TYPES[change_type],
            subject_type="identity_change_proposal",
            subject_id=proposal.id,
            required_reviewers=3,
            policy_version="review-policy-v1",
        )
        db.add(task)
        db.flush()
        db.add_all([ReviewerRoster(
            task_id=task.id, reviewer_user_id=reviewer_id)
            for reviewer_id in reviewers])
        return proposal.id, task.id

    @staticmethod
    def _apply_merge(
        db: Session, plan: dict,
    ) -> tuple[dict, list[uuid.UUID]]:
        source_ids = [uuid.UUID(value)
                      for value in plan["source_individual_ids"]]
        target_id = uuid.UUID(plan["target_individual_id"])
        individuals = list(db.scalars(
            select(ConfirmedIndividual)
            .where(ConfirmedIndividual.id.in_(source_ids))
            .with_for_update()
        ))
        if len(individuals) != len(source_ids) \
                or any(item.state != "active" for item in individuals):
            raise ValueError("合并方案已过期，身份状态已变化")
        observations = list(db.scalars(
            select(Observation)
            .where(Observation.individual_id.in_(source_ids))
            .with_for_update()
        ))
        target = next(item for item in individuals if item.id == target_id)
        for observation in observations:
            observation.individual_id = target_id
        for individual in individuals:
            if individual.id == target_id:
                continue
            individual.state = "merged"
            alias = db.scalar(select(IndividualAlias).where(
                IndividualAlias.individual_id == target_id,
                IndividualAlias.alias == individual.display_name,
                IndividualAlias.source == "identity_merge",
            ))
            if alias is None:
                db.add(IndividualAlias(
                    individual_id=target_id,
                    alias=individual.display_name,
                    source="identity_merge",
                ))
        db.flush()
        affected_crops = [item.crop_id for item in observations]
        IdentityChangeService._sync_cooccurrence_members(db, affected_crops)
        return {
            "target_individual_id": str(target.id),
            "superseded_individual_ids": [str(item.id) for item in individuals
                                           if item.id != target.id],
            "affected_observation_ids": [str(item.id) for item in observations],
            "created_individual_ids": [],
        }, affected_crops

    @staticmethod
    def _apply_split(
        db: Session, plan: dict,
    ) -> tuple[dict, list[uuid.UUID]]:
        source_id = uuid.UUID(plan["source_individual_id"])
        source = db.get(ConfirmedIndividual, source_id, with_for_update=True)
        assignments = {
            uuid.UUID(item["observation_id"]): item["group"]
            for item in plan["assignments"]
        }
        observations = list(db.scalars(
            select(Observation)
            .where(
                Observation.individual_id == source_id,
                Observation.state == "active",
            )
            .with_for_update()
        ))
        if source is None or source.state != "active" \
                or {item.id for item in observations} != set(assignments):
            raise ValueError("拆分方案已过期，Observation 集合已变化")
        by_group: dict[str, list[Observation]] = {}
        for observation in observations:
            by_group.setdefault(assignments[observation.id], []).append(
                observation)
        created: list[ConfirmedIndividual] = []
        for group in sorted(by_group):
            individual_id = uuid.uuid4()
            individual = ConfirmedIndividual(
                id=individual_id,
                display_name=f"WW-{individual_id.hex[:8].upper()}",
                flags=["identity_split"],
            )
            db.add(individual)
            created.append(individual)
            for observation in by_group[group]:
                observation.individual_id = individual_id
        source.state = "split"
        db.flush()
        affected_crops = [item.crop_id for item in observations]
        IdentityChangeService._sync_cooccurrence_members(db, affected_crops)
        return {
            "source_individual_id": str(source_id),
            "created_individual_ids": [str(item.id) for item in created],
            "affected_observation_ids": [str(item.id) for item in observations],
            "groups": {
                group: [str(item.id) for item in values]
                for group, values in by_group.items()
            },
        }, affected_crops

    @staticmethod
    def _apply_withdrawal(
        db: Session, plan: dict,
    ) -> tuple[dict, list[uuid.UUID]]:
        observation_ids = [uuid.UUID(value)
                           for value in plan["observation_ids"]]
        observations = list(db.scalars(
            select(Observation)
            .where(Observation.id.in_(observation_ids))
            .with_for_update()
        ))
        if len(observations) != len(observation_ids) \
                or any(item.state != "active" for item in observations):
            raise ValueError("撤回方案已过期，Observation 状态已变化")
        for observation in observations:
            observation.state = "withdrawn"
        db.flush()
        affected_crops = [item.crop_id for item in observations]
        IdentityChangeService._sync_cooccurrence_members(db, affected_crops)
        return {
            "created_individual_ids": [],
            "affected_observation_ids": [str(item.id) for item in observations],
        }, affected_crops

    @staticmethod
    def _sync_cooccurrence_members(
        db: Session, crop_ids: list[uuid.UUID],
    ) -> None:
        if not crop_ids:
            return
        active = {
            crop_id: individual_id
            for crop_id, individual_id in db.execute(select(
                Observation.crop_id, Observation.individual_id).where(
                    Observation.crop_id.in_(crop_ids),
                    Observation.state == "active",
                ))
        }
        members = list(db.scalars(select(CooccurrenceMember).where(
            CooccurrenceMember.crop_id.in_(crop_ids))))
        for member in members:
            member.individual_id = active.get(member.crop_id)

    @staticmethod
    def _invalidate_relationships(
        db: Session,
        crop_ids: list[uuid.UUID],
        proposal_id: uuid.UUID,
    ) -> None:
        if not crop_ids:
            return
        event_ids = list(db.scalars(select(
            CooccurrenceMember.event_id).where(
                CooccurrenceMember.crop_id.in_(crop_ids)).distinct()))
        if not event_ids:
            return
        hypotheses = list(db.scalars(
            select(RelationshipHypothesis)
            .join(
                RelationshipEvidence,
                RelationshipEvidence.hypothesis_id
                == RelationshipHypothesis.id,
            )
            .where(RelationshipEvidence.cooccurrence_event_id.in_(event_ids))
            .distinct()
        ))
        for hypothesis in hypotheses:
            if hypothesis.status == "rejected":
                continue
            hypothesis.status = "evidence_insufficient"
            db.add(RelationshipEvent(
                hypothesis_id=hypothesis.id,
                event_type="identity_change_invalidated_evidence",
                payload={"identity_change_proposal_id": str(proposal_id)},
            ))

    @staticmethod
    def _event(
        proposal: IdentityChangeProposal,
        task: ReviewTask,
        event_type: str,
        payload: dict,
    ) -> IdentityChangeEvent:
        return IdentityChangeEvent(
            proposal_id=proposal.id,
            review_task_id=task.id,
            event_type=event_type,
            actor_user_id=proposal.created_by_user_id,
            payload=payload,
        )

    @staticmethod
    def _result(
        event: IdentityChangeEvent, status: str,
    ) -> IdentityChangeResult:
        return IdentityChangeResult(
            proposal_id=event.proposal_id,
            status=status,
            created_individual_ids=tuple(
                uuid.UUID(value)
                for value in event.payload.get("created_individual_ids", [])),
            affected_observation_ids=tuple(
                uuid.UUID(value)
                for value in event.payload.get("affected_observation_ids", [])),
        )
