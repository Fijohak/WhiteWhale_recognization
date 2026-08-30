"""归档网页使用的只读投影；所有最终状态仍由领域服务写入。"""
from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    Batch,
    CandidateCluster,
    CandidateClusterMember,
    ConfirmedIndividual,
    Crop,
    Image,
    IndividualAlias,
    MatchCandidate,
    Observation,
    ReviewerRoster,
    ReviewTask,
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

    def individuals(self) -> list[dict]:
        with self._sessions() as db:
            rows = list(db.execute(
                select(ConfirmedIndividual, func.count(Observation.id))
                .outerjoin(
                    Observation,
                    Observation.individual_id == ConfirmedIndividual.id,
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
                    "observed_at": observation.observed_at.isoformat()
                    if observation.observed_at else None,
                } for observation, crop in observations],
            }
