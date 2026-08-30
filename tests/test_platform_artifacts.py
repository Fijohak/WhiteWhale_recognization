"""Worker 产物回传、完整性校验与幂等完成契约。"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.artifacts import (  # noqa: E402
    ArtifactSpec,
    ArtifactValidationError,
    WorkerResultService,
)
from whitewhale.platform.jobs import LeaseService  # noqa: E402
from whitewhale.platform.models import (  # noqa: E402
    Artifact,
    Base,
    Job,
    JobLease,
    WorkerDevice,
)
from whitewhale.platform.states import JobState  # noqa: E402
from whitewhale.platform.storage import StorageLayout  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestWorkerArtifactCompletion(unittest.TestCase):
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
        self.device_id = uuid.uuid4()
        with self.sessions.begin() as session:
            session.add(WorkerDevice(
                id=self.device_id,
                name="artifact-worker",
                gpu_model="RTX 4060 Laptop",
                vram_mb=8192,
                cuda_version="12.6",
                worker_version="0.1.0",
                capabilities=["test_echo"],
                model_versions=["test-model-v1"],
                capacity=1,
            ))
            job = Job(
                task_type="test_echo",
                required_vram_mb=512,
                required_model_version="test-model-v1",
                idempotency_key="artifact-test-v1",
                input_manifest={"items": ["image-1"]},
            )
            session.add(job)
            session.flush()
            self.job_id = job.id

    def test_verified_artifact_completion_is_idempotent(self):
        grant = LeaseService(self.sessions).lease_next(self.device_id)
        self.assertIsNotNone(grant)
        payload = b'{"image-1": "ok"}'

        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = StorageLayout(tmp_dir)
            layout.initialize()
            service = WorkerResultService(self.sessions, layout)
            service.start(self.job_id, grant.lease_token)
            artifact_id = service.submit(
                self.job_id,
                grant.lease_token,
                ArtifactSpec(
                    artifact_type="test_result",
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size_bytes=len(payload),
                    schema_version=1,
                    pipeline_config_digest="a" * 64,
                    row_binding_digest="b" * 64,
                    model_version="test-model-v1",
                ),
                payload,
            )
            self.assertEqual(
                service.complete(self.job_id, grant.lease_token), artifact_id)
            self.assertEqual(
                service.complete(self.job_id, grant.lease_token), artifact_id)

            with self.sessions() as session:
                self.assertEqual(session.get(Job, self.job_id).state,
                                 JobState.SUCCEEDED)
                self.assertEqual(session.scalar(
                    select(func.count()).select_from(Artifact)), 1)
                self.assertIsNone(session.scalar(select(JobLease)))

    def test_wrong_hash_never_creates_an_artifact(self):
        grant = LeaseService(self.sessions).lease_next(self.device_id)
        self.assertIsNotNone(grant)
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = StorageLayout(tmp_dir)
            layout.initialize()
            service = WorkerResultService(self.sessions, layout)
            service.start(self.job_id, grant.lease_token)
            with self.assertRaisesRegex(ArtifactValidationError, "SHA-256"):
                service.submit(
                    self.job_id,
                    grant.lease_token,
                    ArtifactSpec(
                        artifact_type="test_result",
                        sha256="0" * 64,
                        size_bytes=2,
                        schema_version=1,
                        pipeline_config_digest="a" * 64,
                        model_version="test-model-v1",
                    ),
                    b"ok",
                )

            with self.sessions() as session:
                self.assertEqual(session.scalar(
                    select(func.count()).select_from(Artifact)), 0)


if __name__ == "__main__":
    unittest.main()
