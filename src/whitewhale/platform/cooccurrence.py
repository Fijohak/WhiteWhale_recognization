"""N 目标共现审核与只允许 suspected 的关系证据投影。"""
from __future__ import annotations

import itertools
import uuid

from sqlalchemy import or_, select, text
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    Collection,
    CollectionMembership,
    CooccurrenceEvent,
    CooccurrenceMember,
    Crop,
    Image,
    Observation,
    RelationshipEvidence,
    RelationshipEvent,
    RelationshipHypothesis,
    ReviewConsensus,
    ReviewerRoster,
    ReviewTask,
    Role,
    UserRole,
)


class CooccurrenceService:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def create_event(
        self,
        image_id: uuid.UUID,
        crop_ids: list[uuid.UUID],
        *,
        reviewer_ids: list[uuid.UUID],
        provenance_artifact_id: uuid.UUID | None = None,
    ) -> tuple[uuid.UUID, uuid.UUID]:
        unique_crops = list(dict.fromkeys(crop_ids))
        if len(unique_crops) < 2 or len(unique_crops) != len(crop_ids):
            raise ValueError("共现事件必须包含至少两个不同 Crop")
        unique_reviewers = list(dict.fromkeys(reviewer_ids))
        if len(unique_reviewers) != 3:
            raise ValueError("multi_target 固定需要 3 名审核人")
        with self._sessions.begin() as db:
            existing = db.scalar(select(CooccurrenceEvent).where(
                CooccurrenceEvent.image_id == image_id).with_for_update())
            if existing is not None:
                task = db.scalar(select(ReviewTask).where(
                    ReviewTask.task_type == "multi_target",
                    ReviewTask.subject_id == existing.id))
                if task is None:
                    raise ValueError("共现事件缺少对应审核任务")
                return existing.id, task.id
            image = db.get(Image, image_id)
            crops = list(db.scalars(select(Crop).where(
                Crop.id.in_(unique_crops))))
            if image is None or len(crops) != len(unique_crops) \
                    or any(crop.image_id != image_id for crop in crops):
                raise ValueError("共现 Crop 不完整或不属于同一 Image")
            self._require_reviewers(db, unique_reviewers)
            event = CooccurrenceEvent(
                image_id=image_id,
                source="nn_relationship" if image.relation_note == "nn_relationship"
                else "multi_detection",
                provenance_artifact_id=provenance_artifact_id,
            )
            db.add(event)
            db.flush()
            task = ReviewTask(
                task_type="multi_target",
                subject_type="cooccurrence_event",
                subject_id=event.id,
                required_reviewers=3,
                policy_version="review-policy-v1",
            )
            db.add(task)
            db.flush()
            db.add_all([
                ReviewerRoster(task_id=task.id, reviewer_user_id=reviewer_id)
                for reviewer_id in unique_reviewers
            ])
            db.add_all([
                CooccurrenceMember(event_id=event.id, crop_id=crop_id)
                for crop_id in unique_crops
            ])
            collection = self._relationship_collection(db)
            image_membership = db.scalar(select(CollectionMembership).where(
                CollectionMembership.collection_id == collection.id,
                CollectionMembership.image_id == image_id,
            ))
            if image_membership is None:
                db.add(CollectionMembership(
                    collection_id=collection.id,
                    image_id=image_id,
                    assignment_source=event.source,
                    membership_status="review_pending",
                ))
            else:
                image_membership.membership_status = "review_pending"
            for crop_id in unique_crops:
                membership = db.scalar(select(CollectionMembership).where(
                    CollectionMembership.collection_id == collection.id,
                    CollectionMembership.crop_id == crop_id,
                ))
                if membership is None:
                    db.add(CollectionMembership(
                        collection_id=collection.id,
                        crop_id=crop_id,
                        assignment_source="multi_detection",
                        membership_status="review_pending",
                    ))
                else:
                    membership.membership_status = "review_pending"
            return event.id, task.id

    def apply_review(self, task_id: uuid.UUID) -> uuid.UUID:
        with self._sessions.begin() as db:
            task = db.get(ReviewTask, task_id, with_for_update=True)
            consensus = db.get(ReviewConsensus, task_id)
            if task is None or consensus is None \
                    or task.task_type != "multi_target" \
                    or task.subject_type != "cooccurrence_event" \
                    or task.status not in {"resolved", "conflict"}:
                raise ValueError("多目标审核尚未形成结论")
            event = db.get(
                CooccurrenceEvent, task.subject_id, with_for_update=True)
            if event is None:
                raise ValueError("共现事件不存在")
            members = list(db.scalars(select(CooccurrenceMember).where(
                CooccurrenceMember.event_id == event.id)))
            if consensus.conclusion == "multi_target_confirmed":
                event.status = "confirmed"
                membership_status = "confirmed_member"
                for member in members:
                    member.membership_status = "confirmed_member"
                    member.source_review_task_id = task_id
            elif consensus.conclusion == "multi_target_rejected":
                event.status = "rejected"
                membership_status = "rejected"
                for member in members:
                    member.membership_status = "rejected"
                    member.source_review_task_id = task_id
            else:
                event.status = "disputed"
                membership_status = None
            if membership_status is not None:
                collection = self._relationship_collection(db)
                memberships = list(db.scalars(select(CollectionMembership).where(
                    CollectionMembership.collection_id == collection.id,
                    or_(
                        CollectionMembership.image_id == event.image_id,
                        CollectionMembership.crop_id.in_([
                            member.crop_id for member in members]),
                    ),
                )))
                for membership in memberships:
                    membership.membership_status = membership_status
            return event.id

    def project_relationships(
        self, event_id: uuid.UUID,
    ) -> list[uuid.UUID]:
        with self._sessions.begin() as db:
            event = db.get(CooccurrenceEvent, event_id, with_for_update=True)
            if event is None or event.status != "confirmed":
                raise ValueError("只有已确认多目标事件可形成关系证据")
            members = list(db.scalars(select(CooccurrenceMember).where(
                CooccurrenceMember.event_id == event_id)))
            observation_map = {
                crop_id: individual_id
                for crop_id, individual_id in db.execute(
                    select(Observation.crop_id, Observation.individual_id)
                    .where(Observation.crop_id.in_([
                        member.crop_id for member in members]))
                )
            }
            if len(observation_map) != len(members):
                raise ValueError("共现成员尚未全部确认正式身份")
            individual_ids = sorted(
                set(observation_map.values()), key=str)
            if len(individual_ids) < 2:
                raise ValueError("共现事件不足两个不同正式个体")
            hypothesis_ids = []
            for low_id, high_id in itertools.combinations(individual_ids, 2):
                hypothesis = db.scalar(select(RelationshipHypothesis).where(
                    RelationshipHypothesis.individual_low_id == low_id,
                    RelationshipHypothesis.individual_high_id == high_id,
                    RelationshipHypothesis.relationship_type == "co_occurrence",
                ))
                if hypothesis is None:
                    hypothesis = RelationshipHypothesis(
                        individual_low_id=low_id,
                        individual_high_id=high_id,
                        relationship_type="co_occurrence",
                        status="suspected",
                    )
                    db.add(hypothesis)
                    db.flush()
                    db.add(RelationshipEvent(
                        hypothesis_id=hypothesis.id,
                        event_type="suspected_from_cooccurrence",
                        payload={"cooccurrence_event_id": str(event_id)},
                    ))
                evidence = db.scalar(select(RelationshipEvidence).where(
                    RelationshipEvidence.hypothesis_id == hypothesis.id,
                    RelationshipEvidence.cooccurrence_event_id == event_id,
                ))
                if evidence is None:
                    db.add(RelationshipEvidence(
                        hypothesis_id=hypothesis.id,
                        cooccurrence_event_id=event_id,
                        evidence_type="same_frame",
                        payload={"image_id": str(event.image_id)},
                    ))
                hypothesis_ids.append(hypothesis.id)
            for member in members:
                member.individual_id = observation_map[member.crop_id]
            return hypothesis_ids

    def project_ready_for_crops(
        self, crop_ids: list[uuid.UUID],
    ) -> list[uuid.UUID]:
        """在身份投影后，幂等补齐已具备完整正式身份的共现关系。"""
        if not crop_ids:
            return []
        with self._sessions() as db:
            event_ids = list(db.scalars(
                select(CooccurrenceEvent.id)
                .join(
                    CooccurrenceMember,
                    CooccurrenceMember.event_id == CooccurrenceEvent.id,
                )
                .where(
                    CooccurrenceEvent.status == "confirmed",
                    CooccurrenceMember.crop_id.in_(crop_ids),
                )
                .distinct()
            ))
            ready: list[uuid.UUID] = []
            for event_id in event_ids:
                member_crop_ids = list(db.scalars(select(
                    CooccurrenceMember.crop_id).where(
                        CooccurrenceMember.event_id == event_id)))
                identities = list(db.scalars(select(
                    Observation.individual_id).where(
                        Observation.crop_id.in_(member_crop_ids))))
                if len(identities) == len(member_crop_ids) \
                        and len(set(identities)) >= 2:
                    ready.append(event_id)
        projected: dict[uuid.UUID, None] = {}
        for event_id in ready:
            for hypothesis_id in self.project_relationships(event_id):
                projected[hypothesis_id] = None
        return list(projected)

    @staticmethod
    def _require_reviewers(
        db: Session, reviewer_ids: list[uuid.UUID],
    ) -> None:
        eligible = set(db.scalars(
            select(UserRole.user_id)
            .join(Role, Role.id == UserRole.role_id)
            .where(
                Role.name == "reviewer",
                UserRole.user_id.in_(reviewer_ids),
            )
        ))
        if eligible != set(reviewer_ids):
            raise ValueError("审核名单包含没有 reviewer 角色的用户")

    @staticmethod
    def _relationship_collection(db: Session) -> Collection:
        db.execute(text("SELECT pg_advisory_xact_lock(91524003)"))
        collection = db.scalar(select(Collection).where(
            Collection.system_key == "nn_relationship"))
        if collection is None:
            collection = Collection(
                system_key="nn_relationship",
                name="疑似关系样本",
                kind="relationship_candidate",
                description="同框或原始目录候选；不表示已确认亲缘。",
            )
            db.add(collection)
            db.flush()
        return collection
