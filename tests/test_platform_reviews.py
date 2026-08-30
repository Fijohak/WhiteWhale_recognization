"""真实 PostgreSQL 上的盲审、追加票和服务端共识。"""
from __future__ import annotations

import os
import sys
import unittest
import uuid
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.models import (  # noqa: E402
    Base,
    Batch,
    ReviewEvent,
    Role,
    User,
    UserRole,
)
from whitewhale.platform.review_policy import ReviewVote  # noqa: E402
from whitewhale.platform.reviews import ReviewService  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestReviewService(unittest.TestCase):
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
            role = Role(name="reviewer")
            session.add(role)
            reviewers = [
                User(username=f"reviewer-{index}", password_hash="test")
                for index in range(3)
            ]
            session.add_all(reviewers)
            session.flush()
            session.add_all([
                UserRole(user_id=user.id, role_id=role.id)
                for user in reviewers
            ])
            batch = Batch(
                name="20140419 02",
                manifest_sha256="a" * 64,
                source_format="idolphin",
            )
            session.add(batch)
            session.flush()
            self.reviewer_ids = [user.id for user in reviewers]
            self.batch_id = batch.id

    def test_votes_are_blind_append_only_and_resolved_by_fixed_policy(self):
        service = ReviewService(self.sessions)
        subject_id = uuid.uuid4()
        task_id = service.create_task(
            task_type="identity_match",
            subject_type="candidate_cluster",
            subject_id=subject_id,
            reviewer_ids=self.reviewer_ids,
        )
        existing_id = uuid.uuid4()

        service.submit_vote(
            task_id, self.reviewer_ids[0],
            ReviewVote(choice="existing", individual_id=existing_id))
        view = service.view_for_reviewer(task_id, self.reviewer_ids[1])
        self.assertEqual(view.status, "open")
        self.assertEqual(view.own_votes, ())
        self.assertIsNone(view.consensus)

        service.submit_vote(
            task_id, self.reviewer_ids[1],
            ReviewVote(choice="existing", individual_id=existing_id))
        service.submit_vote(
            task_id, self.reviewer_ids[2],
            ReviewVote(choice="existing", individual_id=existing_id))
        view = service.view_for_reviewer(task_id, self.reviewer_ids[0])
        self.assertEqual(view.status, "resolved")
        self.assertEqual(view.consensus.conclusion, "confirm_existing")
        self.assertEqual(view.consensus.individual_id, existing_id)

        with self.sessions() as session:
            self.assertEqual(session.scalar(
                select(func.count()).select_from(ReviewEvent)), 3)

    def test_roster_size_and_reviewer_role_cannot_be_lowered_by_client(self):
        service = ReviewService(self.sessions)
        with self.assertRaisesRegex(ValueError, "3 名"):
            service.create_task(
                task_type="identity_match",
                subject_type="candidate_cluster",
                subject_id=uuid.uuid4(),
                reviewer_ids=self.reviewer_ids[:2],
            )
        with self.assertRaisesRegex(ValueError, "1 名"):
            service.create_task(
                task_type="cluster_purity",
                subject_type="candidate_cluster",
                subject_id=uuid.uuid4(),
                reviewer_ids=self.reviewer_ids,
            )


if __name__ == "__main__":
    unittest.main()
