"""只有已解决审核共识才能创建正式身份和 Observation。"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.identities import IdentityService  # noqa: E402
from whitewhale.platform.models import (  # noqa: E402
    Base, Batch, CandidateCluster, CandidateClusterMember,
    ConfirmedIndividual, Crop, Image, Observation, ReviewConsensus,
    ReviewTask,
)


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestIdentityService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
        cls.sessions = sessionmaker(cls.engine, expire_on_commit=False)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        with self.sessions.begin() as session:
            batch = Batch(name="batch", manifest_sha256="a" * 64,
                          source_format="generic")
            session.add(batch)
            session.flush()
            image = Image(
                batch_id=batch.id, source_path="x/a.jpg",
                original_relative_path="a.jpg", source_sha256="b" * 64,
                size_bytes=1,
            )
            session.add(image)
            session.flush()
            crop = Crop(
                image_id=image.id, crop_index=0, x=0, y=0,
                width=10, height=10, detector_version="det-v1",
                artifact_path="crop/a.jpg",
            )
            session.add(crop)
            session.flush()
            cluster = CandidateCluster(
                batch_id=batch.id, label="cluster-0", algorithm="hdbscan",
                algorithm_version="1", representative_crop_id=crop.id,
            )
            session.add(cluster)
            session.flush()
            session.add(CandidateClusterMember(
                cluster_id=cluster.id, crop_id=crop.id))
            task = ReviewTask(
                task_type="identity_match", subject_type="candidate_cluster",
                subject_id=cluster.id, status="resolved", required_reviewers=3,
                policy_version="review-policy-v1",
            )
            session.add(task)
            session.flush()
            session.add(ReviewConsensus(
                task_id=task.id, status="resolved", conclusion="confirm_new",
                flags=["possible_duplicate"],
            ))
            self.task_id = task.id

    def test_confirm_new_consensus_is_applied_once_with_risk_flags(self):
        service = IdentityService(self.sessions)
        individual_id = service.apply_review(self.task_id)
        self.assertEqual(service.apply_review(self.task_id), individual_id)
        with self.sessions() as session:
            individual = session.get(ConfirmedIndividual, individual_id)
            self.assertEqual(individual.flags, ["possible_duplicate"])
            self.assertEqual(session.scalar(
                select(func.count()).select_from(Observation)), 1)


if __name__ == "__main__":
    unittest.main()
