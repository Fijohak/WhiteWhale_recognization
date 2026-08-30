"""身份合并、拆分和照片撤回必须 3/3 通过且不删除历史。"""
from __future__ import annotations

import os
import sys
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.identity_changes import IdentityChangeService  # noqa: E402
from whitewhale.platform.app import PlatformServices, create_app  # noqa: E402
from whitewhale.platform.auth import AuthService  # noqa: E402
from whitewhale.platform.models import (  # noqa: E402
    Base, Batch, ConfirmedIndividual, Crop, IdentityChangeEvent,
    IdentityChangeProposal, Image, Observation, ReviewTask,
)
from whitewhale.platform.review_policy import ReviewVote  # noqa: E402
from whitewhale.platform.reviews import ReviewService  # noqa: E402
from whitewhale.platform.states import BatchStage  # noqa: E402
from whitewhale.platform.views import ArchiveReadService  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestIdentityChanges(unittest.TestCase):
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
        self.auth = AuthService(self.sessions)
        admin = self.auth.bootstrap_admin(
            "identity-admin", "correct horse battery staple")
        reviewers = [self.auth.create_user(
            f"r-{index}", f"reviewer secure password {index}",
            roles={"reviewer"},
        ) for index in range(3)]
        with self.sessions.begin() as db:
            batch = Batch(
                name="identity-correction", manifest_sha256="a" * 64,
                source_format="generic", stage=BatchStage.PUBLISHED,
            )
            db.add(batch)
            db.flush()
            image = Image(
                batch_id=batch.id, source_path="correction.jpg",
                original_relative_path="correction.jpg",
                source_sha256="b" * 64, size_bytes=4,
            )
            db.add(image)
            db.flush()
            crops = [Crop(
                image_id=image.id, crop_index=index, x=index * 10, y=0,
                width=10, height=10, detector_version="det-v2",
                artifact_path=f"crop/{index}.jpg",
            ) for index in range(4)]
            db.add_all(crops)
            individuals = [ConfirmedIndividual(display_name=f"WW-{name}")
                           for name in ("A", "B", "C")]
            db.add_all(individuals)
            db.flush()
            source_task = ReviewTask(
                task_type="identity_match", subject_type="candidate_cluster",
                subject_id=uuid.uuid4(), required_reviewers=3,
                policy_version="review-policy-v1", status="resolved",
            )
            db.add(source_task)
            db.flush()
            observations = [
                Observation(
                    individual_id=individuals[0].id, crop_id=crops[0].id,
                    source_review_task_id=source_task.id),
                Observation(
                    individual_id=individuals[0].id, crop_id=crops[1].id,
                    source_review_task_id=source_task.id),
                Observation(
                    individual_id=individuals[1].id, crop_id=crops[2].id,
                    source_review_task_id=source_task.id),
                Observation(
                    individual_id=individuals[2].id, crop_id=crops[3].id,
                    source_review_task_id=source_task.id),
            ]
            db.add_all(observations)
            db.flush()
            self.reviewer_ids = [user.id for user in reviewers]
            self.individual_ids = [item.id for item in individuals]
            self.observation_ids = [item.id for item in observations]
            self.actor_id = admin.id

    def _approve(self, task_id: uuid.UUID) -> None:
        reviews = ReviewService(self.sessions)
        for reviewer_id in self.reviewer_ids:
            reviews.submit_vote(
                task_id, reviewer_id, ReviewVote("approve_change"))

    def test_merge_requires_three_of_three_and_preserves_source_history(self):
        service = IdentityChangeService(self.sessions)
        proposal_id, task_id = service.create_merge(
            self.individual_ids[:2],
            target_individual_id=self.individual_ids[0],
            reviewer_ids=self.reviewer_ids,
            actor_user_id=self.actor_id,
        )
        reviews = ReviewService(self.sessions)
        reviews.submit_vote(
            task_id, self.reviewer_ids[0], ReviewVote("approve_change"))
        reviews.submit_vote(
            task_id, self.reviewer_ids[1], ReviewVote("approve_change"))
        reviews.submit_vote(
            task_id, self.reviewer_ids[2], ReviewVote("uncertain"))
        service.apply_review(task_id)
        with self.sessions() as db:
            self.assertEqual(
                db.get(IdentityChangeProposal, proposal_id).status,
                "disputed",
            )
            self.assertEqual(
                db.get(ConfirmedIndividual, self.individual_ids[1]).state,
                "active",
            )

        proposal_id, task_id = service.create_merge(
            self.individual_ids[:2],
            target_individual_id=self.individual_ids[0],
            reviewer_ids=self.reviewer_ids,
            actor_user_id=self.actor_id,
        )
        self._approve(task_id)
        service.apply_review(task_id)
        service.apply_review(task_id)
        with self.sessions() as db:
            self.assertEqual(
                db.get(IdentityChangeProposal, proposal_id).status, "applied")
            self.assertEqual(
                db.get(ConfirmedIndividual, self.individual_ids[1]).state,
                "merged",
            )
            mapped = set(db.scalars(select(Observation.individual_id).where(
                Observation.id.in_(self.observation_ids[:3]))))
            self.assertEqual(mapped, {self.individual_ids[0]})
            self.assertEqual(db.scalar(select(func.count()).select_from(
                IdentityChangeEvent).where(
                    IdentityChangeEvent.proposal_id == proposal_id)), 1)

    def test_split_creates_new_identities_and_keeps_original_uuid(self):
        service = IdentityChangeService(self.sessions)
        proposal_id, task_id = service.create_split(
            self.individual_ids[0],
            assignments={
                self.observation_ids[0]: "left",
                self.observation_ids[1]: "right",
            },
            reviewer_ids=self.reviewer_ids,
            actor_user_id=self.actor_id,
        )
        self._approve(task_id)
        result = service.apply_review(task_id)
        self.assertEqual(len(result.created_individual_ids), 2)
        with self.sessions() as db:
            self.assertEqual(
                db.get(ConfirmedIndividual, self.individual_ids[0]).state,
                "split",
            )
            mapped = set(db.scalars(select(Observation.individual_id).where(
                Observation.id.in_(self.observation_ids[:2]))))
            self.assertEqual(mapped, set(result.created_individual_ids))
            self.assertIsNotNone(
                db.get(IdentityChangeProposal, proposal_id).applied_at)

    def test_withdrawal_marks_observation_but_does_not_delete_it(self):
        service = IdentityChangeService(self.sessions)
        proposal_id, task_id = service.create_withdrawal(
            [self.observation_ids[2]],
            reviewer_ids=self.reviewer_ids,
            actor_user_id=self.actor_id,
        )
        self._approve(task_id)
        service.apply_review(task_id)
        with self.sessions() as db:
            observation = db.get(Observation, self.observation_ids[2])
            self.assertEqual(observation.state, "withdrawn")
            self.assertEqual(db.scalar(select(func.count()).select_from(
                Observation).where(
                    Observation.id == self.observation_ids[2])), 1)
            self.assertEqual(
                db.get(IdentityChangeProposal, proposal_id).status, "applied")
        views = ArchiveReadService(self.sessions)
        summary = next(item for item in views.individuals()
                       if item["individual_id"] == str(self.individual_ids[1]))
        self.assertEqual(summary["observation_count"], 0)
        detail = views.individual(self.individual_ids[1])
        self.assertEqual(detail["observations"][0]["state"], "withdrawn")

    def test_authenticated_api_creates_and_reads_immutable_proposal(self):
        service = IdentityChangeService(self.sessions)
        app = create_app(services=PlatformServices(
            auth=self.auth,
            uploads=None,
            imports=None,
            reviews=ReviewService(self.sessions),
            views=ArchiveReadService(self.sessions),
            identity_changes=service,
        ))
        with TestClient(app, base_url="https://testserver") as client:
            login = client.post("/api/auth/login", json={
                "username": "identity-admin",
                "password": "correct horse battery staple",
            })
            self.assertEqual(login.status_code, 200, login.text)
            response = client.post(
                "/api/identity-changes/withdrawal",
                headers={"X-CSRF-Token": login.json()["csrf_token"]},
                json={
                    "observation_ids": [str(self.observation_ids[2])],
                    "reviewer_ids": [str(value)
                                     for value in self.reviewer_ids],
                },
            )
            self.assertEqual(response.status_code, 201, response.text)
            detail = client.get(
                f"/api/identity-changes/{response.json()['proposal_id']}")
            self.assertEqual(detail.status_code, 200, detail.text)
            self.assertEqual(detail.json()["change_type"], "withdrawal")
            self.assertEqual(
                detail.json()["plan"]["observation_ids"],
                [str(self.observation_ids[2])],
            )

        anonymous = TestClient(app, base_url="https://testserver")
        try:
            self.assertEqual(anonymous.get(
                f"/api/identity-changes/{response.json()['proposal_id']}"
            ).status_code, 401)
        finally:
            anonymous.close()


if __name__ == "__main__":
    unittest.main()
