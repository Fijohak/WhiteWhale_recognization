"""N 目标共现和疑似关系不能污染正式身份或变成已确认亲缘。"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.cooccurrence import CooccurrenceService  # noqa: E402
from whitewhale.platform.identity_changes import IdentityChangeService  # noqa: E402
from whitewhale.platform.models import (  # noqa: E402
    Base, Batch, Collection, CollectionMembership, ConfirmedIndividual,
    CooccurrenceEvent, CooccurrenceMember, Crop, Image, Observation,
    RelationshipEvidence, RelationshipHypothesis, ReviewConsensus,
    ReviewTask, Role, User, UserRole,
)
from whitewhale.platform.review_policy import ReviewVote  # noqa: E402
from whitewhale.platform.reviews import ReviewService  # noqa: E402
from whitewhale.platform.views import ArchiveReadService  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestCooccurrence(unittest.TestCase):
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
        with self.sessions.begin() as db:
            role = Role(name="reviewer")
            db.add(role)
            users = [User(username=f"r-{i}", password_hash="test")
                     for i in range(3)]
            db.add_all(users)
            db.flush()
            db.add_all([UserRole(user_id=user.id, role_id=role.id)
                        for user in users])
            batch = Batch(name="multi", manifest_sha256="a" * 64,
                          source_format="generic")
            db.add(batch)
            db.flush()
            image = Image(
                batch_id=batch.id, source_path="x.jpg",
                original_relative_path="x.jpg", source_sha256="b" * 64,
                size_bytes=1, relation_note="nn_relationship")
            db.add(image)
            db.flush()
            crops = [Crop(
                image_id=image.id, crop_index=i, x=i * 10, y=0,
                width=10, height=10, detector_version="det-v2",
                artifact_path=f"c/{i}.jpg") for i in range(3)]
            db.add_all(crops)
            db.flush()
            self.image_id = image.id
            self.crop_ids = [crop.id for crop in crops]
            self.reviewers = [user.id for user in users]

    def test_three_crops_create_one_event_and_two_of_three_confirm_members(self):
        service = CooccurrenceService(self.sessions)
        event_id, task_id = service.create_event(
            self.image_id, self.crop_ids, reviewer_ids=self.reviewers)
        reviews = ReviewService(self.sessions)
        reviews.submit_vote(
            task_id, self.reviewers[0], ReviewVote("confirm_multi_target"))
        reviews.submit_vote(
            task_id, self.reviewers[1], ReviewVote("confirm_multi_target"))
        decision = reviews.submit_vote(
            task_id, self.reviewers[2], ReviewVote("uncertain"))
        self.assertEqual(decision.conclusion, "multi_target_confirmed")
        service.apply_review(task_id)
        with self.sessions() as db:
            self.assertEqual(db.get(CooccurrenceEvent, event_id).status,
                             "confirmed")
            members = list(db.scalars(select(CooccurrenceMember).where(
                CooccurrenceMember.event_id == event_id)))
            self.assertEqual(len(members), 3)
            self.assertTrue(all(
                member.membership_status == "confirmed_member"
                for member in members))
            collection = db.scalar(select(Collection).where(
                Collection.system_key == "nn_relationship"))
            statuses = list(db.scalars(select(
                CollectionMembership.membership_status).where(
                    CollectionMembership.collection_id == collection.id)))
            self.assertEqual(statuses, [
                "confirmed_member", "confirmed_member", "confirmed_member",
                "confirmed_member",
            ])

    def test_confirmed_members_project_only_suspected_unordered_pairs(self):
        service = CooccurrenceService(self.sessions)
        event_id, task_id = service.create_event(
            self.image_id, self.crop_ids, reviewer_ids=self.reviewers)
        with self.sessions.begin() as db:
            task = db.get(ReviewTask, task_id)
            task.status = "resolved"
            db.add(ReviewConsensus(
                task_id=task_id, status="resolved",
                conclusion="multi_target_confirmed"))
        service.apply_review(task_id)
        with self.sessions.begin() as db:
            individuals = [ConfirmedIndividual(display_name=f"WW-{i}")
                           for i in range(3)]
            db.add_all(individuals)
            db.flush()
            for crop_id, individual in zip(
                    self.crop_ids[:2], individuals[:2], strict=True):
                db.add(Observation(
                    individual_id=individual.id, crop_id=crop_id,
                    source_review_task_id=task_id))
        self.assertEqual(service.project_ready_for_crops(self.crop_ids[:2]), [])
        with self.sessions.begin() as db:
            withdrawn_observation = Observation(
                individual_id=individuals[2].id, crop_id=self.crop_ids[2],
                source_review_task_id=task_id)
            db.add(withdrawn_observation)
            db.flush()
            withdrawn_observation_id = withdrawn_observation.id
        hypotheses = service.project_ready_for_crops([self.crop_ids[2]])
        self.assertEqual(len(hypotheses), 3)
        with self.sessions() as db:
            rows = list(db.scalars(select(RelationshipHypothesis)))
            self.assertTrue(all(row.relationship_type == "co_occurrence"
                                for row in rows))
            self.assertTrue(all(row.status == "suspected" for row in rows))
            self.assertFalse(any(
                row.relationship_type == "confirmed_kinship" for row in rows))

        views = ArchiveReadService(self.sessions)
        detail = views.cooccurrence(event_id)
        self.assertEqual(detail["status"], "confirmed")
        self.assertEqual(len(detail["crops"]), 3)
        self.assertTrue(all(crop["individual_id"] for crop in detail["crops"]))
        relationships = views.relationships()
        self.assertEqual(len(relationships), 3)
        self.assertTrue(all(item["status"] == "suspected"
                            for item in relationships))
        self.assertTrue(all(item["evidence_count"] == 1
                            for item in relationships))
        with self.sessions() as db:
            self.assertEqual(db.scalar(select(func.count()).select_from(
                RelationshipEvidence)), 3)

        changes = IdentityChangeService(self.sessions)
        _, withdrawal_task_id = changes.create_withdrawal(
            [withdrawn_observation_id],
            reviewer_ids=self.reviewers,
            actor_user_id=self.reviewers[0],
        )
        reviews = ReviewService(self.sessions)
        for reviewer_id in self.reviewers:
            reviews.submit_vote(
                withdrawal_task_id, reviewer_id,
                ReviewVote("approve_change"))
        changes.apply_review(withdrawal_task_id)
        with self.sessions() as db:
            statuses = set(db.scalars(select(
                RelationshipHypothesis.status)))
            self.assertEqual(statuses, {"evidence_insufficient"})


if __name__ == "__main__":
    unittest.main()
