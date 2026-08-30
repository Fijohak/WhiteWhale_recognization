"""M1 控制面基础契约：状态、数据库、文件库与健康检查。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.app import Readiness, create_app  # noqa: E402
from whitewhale.platform.models import Base  # noqa: E402
from whitewhale.platform.states import (  # noqa: E402
    BatchStage,
    JobState,
    advance_batch_stage,
    transition_job_state,
)
from whitewhale.platform.storage import StorageLayout  # noqa: E402


class TestPlatformStates(unittest.TestCase):
    def test_batch_stage_only_advances_through_successful_business_stages(self):
        self.assertEqual(
            advance_batch_stage(BatchStage.REGISTERED,
                                BatchStage.CANDIDATE_READY),
            BatchStage.CANDIDATE_READY,
        )
        self.assertEqual(
            advance_batch_stage(BatchStage.CANDIDATE_READY,
                                BatchStage.CANDIDATE_READY),
            BatchStage.CANDIDATE_READY,
        )
        with self.assertRaisesRegex(ValueError, "批次阶段"):
            advance_batch_stage(BatchStage.REGISTERED, BatchStage.APPROVED)
        with self.assertRaisesRegex(ValueError, "批次阶段"):
            advance_batch_stage(BatchStage.APPROVED,
                                BatchStage.UNDER_REVIEW)

    def test_job_state_rejects_skips_and_allows_retry_from_lease_expired(self):
        self.assertEqual(
            transition_job_state(JobState.QUEUED, JobState.LEASED),
            JobState.LEASED,
        )
        self.assertEqual(
            transition_job_state(JobState.LEASE_EXPIRED, JobState.QUEUED),
            JobState.QUEUED,
        )
        with self.assertRaisesRegex(ValueError, "任务状态"):
            transition_job_state(JobState.QUEUED, JobState.SUCCEEDED)


class TestPlatformDatabaseMetadata(unittest.TestCase):
    def test_m1_tables_are_present_with_separate_image_and_crop_entities(self):
        expected = {
            "users", "roles", "user_roles",
            "user_sessions",
            "worker_devices", "worker_tokens", "worker_registration_codes",
            "worker_heartbeats",
            "batches", "source_groups", "images", "crops",
            "upload_sessions", "upload_files", "upload_parts",
            "jobs", "job_attempts", "job_leases", "job_events",
            "artifacts", "artifact_manifests",
            "candidate_clusters", "candidate_cluster_members",
            "candidate_events", "review_tasks", "review_events",
            "reviewer_rosters", "review_consensus", "review_conflicts",
            "confirmed_individuals", "individual_aliases", "observations",
            "identity_events",
            "identity_change_proposals", "identity_change_events",
            "catalog_versions", "catalog_memberships",
            "active_catalog_pointer", "catalog_events",
            "crop_embeddings", "match_candidates",
            "collections", "collection_memberships",
            "cooccurrence_events", "cooccurrence_members",
            "relationship_hypotheses", "relationship_evidence",
            "relationship_events",
        }
        self.assertTrue(expected.issubset(Base.metadata.tables))

        images = Base.metadata.tables["images"]
        crops = Base.metadata.tables["crops"]
        self.assertIn("source_sha256", images.c)
        self.assertIn("image_id", crops.c)
        self.assertIn("crop_index", crops.c)
        self.assertTrue(any(
            set(constraint.columns.keys()) == {"image_id", "crop_index"}
            for constraint in crops.constraints
            if hasattr(constraint, "columns")
        ))


class TestStorageLayout(unittest.TestCase):
    def test_initializes_bounded_host_directories_and_rejects_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = StorageLayout(Path(tmp_dir))
            layout.initialize()

            self.assertEqual(
                {path.name for path in Path(tmp_dir).iterdir()},
                {"raw", "working", "artifacts", "models",
                 "catalog_versions", "exports", "staging"},
            )
            self.assertEqual(
                layout.resolve("raw", "batch/image.jpg"),
                Path(tmp_dir).resolve() / "raw" / "batch" / "image.jpg",
            )
            for unsafe in (
                "../secret", "batch/../secret", "/etc/passwd", "C:secret.jpg",
            ):
                with self.subTest(path=unsafe), \
                        self.assertRaisesRegex(ValueError, "路径越界"):
                    layout.resolve("raw", unsafe)
            with self.assertRaisesRegex(ValueError, "未知文件分区"):
                layout.resolve("other", "file.jpg")


class TestPlatformHealth(unittest.TestCase):
    def test_health_is_liveness_while_ready_reports_dependency_failure(self):
        app = create_app(readiness_probe=lambda: Readiness(
            database=False,
            storage=True,
            migrations=True,
            active_catalog=True,
            detail="database unavailable",
        ))
        client = TestClient(app)

        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})

        ready = client.get("/ready")
        self.assertEqual(ready.status_code, 503)
        self.assertEqual(ready.json()["status"], "not_ready")
        self.assertEqual(ready.json()["checks"]["database"], False)
        self.assertEqual(ready.json()["detail"], "database unavailable")


if __name__ == "__main__":
    unittest.main()
