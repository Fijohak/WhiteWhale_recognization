"""把已解决的身份审核共识原子投影为正式个体与 Observation。"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    CandidateCluster,
    CandidateClusterMember,
    ConfirmedIndividual,
    IdentityEvent,
    Observation,
    ReviewConsensus,
    ReviewTask,
)


class IdentityService:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def apply_review(self, task_id: uuid.UUID) -> uuid.UUID:
        with self._sessions.begin() as db:
            existing_event = db.scalar(select(IdentityEvent).where(
                IdentityEvent.review_task_id == task_id))
            if existing_event is not None:
                return existing_event.individual_id

            task = db.get(ReviewTask, task_id, with_for_update=True)
            consensus = db.get(ReviewConsensus, task_id)
            if task is None or consensus is None \
                    or task.status != "resolved" \
                    or consensus.status != "resolved":
                raise ValueError("审核尚未形成可应用共识")
            if task.task_type != "identity_match" \
                    or task.subject_type != "candidate_cluster":
                raise ValueError("审核任务不是候选簇身份匹配")
            cluster = db.get(CandidateCluster, task.subject_id, with_for_update=True)
            if cluster is None:
                raise ValueError("候选簇不存在")

            if consensus.conclusion == "confirm_new":
                individual_id = uuid.uuid4()
                individual = ConfirmedIndividual(
                    id=individual_id,
                    display_name=f"WW-{individual_id.hex[:8].upper()}",
                    flags=sorted(consensus.flags),
                )
                db.add(individual)
            elif consensus.conclusion == "confirm_existing" \
                    and consensus.individual_id is not None:
                individual = db.get(
                    ConfirmedIndividual,
                    consensus.individual_id,
                    with_for_update=True,
                )
                if individual is None or individual.state != "active":
                    raise ValueError("目标正式个体不存在或已停用")
                individual_id = individual.id
            else:
                raise ValueError("共识不能创建或关联正式身份")

            crop_ids = list(db.scalars(
                select(CandidateClusterMember.crop_id).where(
                    CandidateClusterMember.cluster_id == cluster.id,
                    CandidateClusterMember.is_excluded.is_(False),
                )
            ))
            if not crop_ids:
                raise ValueError("候选簇没有可用 Crop")
            db.add_all([
                Observation(
                    individual_id=individual_id,
                    crop_id=crop_id,
                    source_review_task_id=task.id,
                )
                for crop_id in crop_ids
            ])
            db.add(IdentityEvent(
                individual_id=individual_id,
                review_task_id=task.id,
                event_type=consensus.conclusion,
                payload={
                    "candidate_cluster_id": str(cluster.id),
                    "flags": consensus.flags,
                    "crop_ids": [str(crop_id) for crop_id in crop_ids],
                },
            ))
            cluster.state = "identity_confirmed"
            return individual_id
