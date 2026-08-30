"""需要真实 PostgreSQL 的 M1 并发契约。"""
from __future__ import annotations

import os
import sys
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Barrier

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.jobs import InvalidLease, LeaseService  # noqa: E402
from whitewhale.platform.models import (  # noqa: E402
    Base,
    Job,
    JobLease,
    WorkerDevice,
)
from whitewhale.platform.states import JobState  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestPostgresLeaseConcurrency(unittest.TestCase):
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

    def test_two_workers_cannot_lease_the_same_job(self):
        device_ids = [uuid.uuid4(), uuid.uuid4()]
        with self.sessions.begin() as session:
            session.add_all([
                WorkerDevice(
                    id=device_id,
                    name=f"worker-{index}",
                    gpu_model="RTX 4060 Laptop",
                    vram_mb=8192,
                    cuda_version="12.6",
                    worker_version="0.1.0",
                    capabilities=["embedding"],
                    model_versions=["reid-r4"],
                    capacity=1,
                )
                for index, device_id in enumerate(device_ids)
            ])
            job = Job(
                task_type="embedding",
                required_vram_mb=4096,
                required_model_version="reid-r4",
                idempotency_key="batch-1:embedding:r4",
            )
            session.add(job)
            session.flush()
            job_id = job.id

        barrier = Barrier(2)

        def lease(device_id):
            service = LeaseService(self.sessions)
            barrier.wait(timeout=5)
            return service.lease_next(device_id)

        with ThreadPoolExecutor(max_workers=2) as pool:
            grants = list(pool.map(lease, device_ids))

        successful = [grant for grant in grants if grant is not None]
        self.assertEqual(len(successful), 1)
        self.assertEqual(successful[0].job_id, job_id)
        self.assertTrue(successful[0].lease_token)
        self.assertEqual(successful[0].attempt_number, 1)

        with self.sessions() as session:
            stored_job = session.get(Job, job_id)
            self.assertEqual(stored_job.state, JobState.LEASED)
            self.assertEqual(
                session.scalar(select(func.count()).select_from(JobLease)), 1)

    def test_expired_lease_requeues_and_rejects_old_worker_token(self):
        device_ids = [uuid.uuid4(), uuid.uuid4()]
        with self.sessions.begin() as session:
            session.add_all([
                WorkerDevice(
                    id=device_id,
                    name=f"expiry-worker-{index}",
                    gpu_model="RTX 4060 Laptop",
                    vram_mb=8192,
                    cuda_version="12.6",
                    worker_version="0.1.0",
                    capabilities=["embedding"],
                    model_versions=["reid-r4"],
                    capacity=1,
                )
                for index, device_id in enumerate(device_ids)
            ])
            job = Job(
                task_type="embedding",
                required_vram_mb=4096,
                required_model_version="reid-r4",
                idempotency_key="batch-2:embedding:r4",
            )
            session.add(job)
            session.flush()
            job_id = job.id

        service = LeaseService(
            self.sessions, lease_duration=timedelta(minutes=1))
        first = service.lease_next(device_ids[0])
        self.assertIsNotNone(first)

        expired = service.requeue_expired(
            now=first.lease_expires_at + timedelta(seconds=1))
        self.assertEqual(expired, [job_id])

        second = service.lease_next(device_ids[1])
        self.assertIsNotNone(second)
        self.assertEqual(second.job_id, job_id)
        self.assertEqual(second.attempt_number, 2)
        self.assertNotEqual(second.lease_token, first.lease_token)

        with self.assertRaisesRegex(InvalidLease, "租约无效"):
            service.heartbeat(job_id, first.lease_token)
        refreshed = service.heartbeat(job_id, second.lease_token)
        self.assertGreater(refreshed, second.lease_expires_at)

    def test_alembic_upgrade_head_creates_the_m1_schema(self):
        with self.engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))

        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
        command.upgrade(config, "head")

        tables = set(inspect(self.engine).get_table_names())
        self.assertTrue({
            "alembic_version",
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
        }.issubset(tables))
        command.check(config)


if __name__ == "__main__":
    unittest.main()
