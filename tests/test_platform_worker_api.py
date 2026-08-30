"""HTTP Worker 控制面端到端契约。"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.app import PlatformServices, create_app  # noqa: E402
from whitewhale.platform.artifacts import WorkerResultService  # noqa: E402
from whitewhale.platform.auth import AuthService  # noqa: E402
from whitewhale.platform.imports import BatchImportService  # noqa: E402
from whitewhale.platform.jobs import JobQueueService, LeaseService  # noqa: E402
from whitewhale.platform.models import Base  # noqa: E402
from whitewhale.platform.storage import StorageLayout  # noqa: E402
from whitewhale.platform.uploads import UploadService  # noqa: E402
from whitewhale.platform.worker_auth import WorkerAuthService  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestWorkerApi(unittest.TestCase):
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
        self.temp_dir = tempfile.TemporaryDirectory()
        layout = StorageLayout(self.temp_dir.name)
        layout.initialize()
        services = PlatformServices(
            auth=AuthService(self.sessions),
            uploads=UploadService(self.sessions, layout),
            imports=BatchImportService(self.sessions, layout),
            worker_auth=WorkerAuthService(self.sessions),
            jobs=JobQueueService(self.sessions),
            leases=LeaseService(self.sessions),
            results=WorkerResultService(self.sessions, layout),
        )
        self.client = TestClient(
            create_app(services=services), base_url="https://testserver")
        self.client.post("/api/auth/bootstrap", json={
            "username": "owner",
            "password": "correct horse battery staple",
        })
        login = self.client.post("/api/auth/login", json={
            "username": "owner",
            "password": "correct horse battery staple",
        })
        self.csrf = login.json()["csrf_token"]

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_worker_registers_leases_and_completes_a_validated_job(self):
        human_headers = {"X-CSRF-Token": self.csrf}
        response = self.client.post(
            "/api/workers/registration-codes", headers=human_headers)
        self.assertEqual(response.status_code, 201, response.text)
        code = response.json()["registration_code"]

        response = self.client.post("/api/workers/register", json={
            "registration_code": code,
            "name": "laptop-4060-a",
            "gpu_model": "RTX 4060 Laptop",
            "vram_mb": 8192,
            "cuda_version": "12.6",
            "worker_version": "0.1.0",
            "capabilities": ["test_echo"],
            "model_versions": ["test-model-v1"],
            "capacity": 1,
        })
        self.assertEqual(response.status_code, 201, response.text)
        worker_token = response.json()["device_token"]
        worker_headers = {"Authorization": f"Bearer {worker_token}"}

        response = self.client.post("/api/jobs", headers=human_headers, json={
            "task_type": "test_echo",
            "required_vram_mb": 512,
            "required_model_version": "test-model-v1",
            "max_attempts": 3,
            "idempotency_key": "worker-api-e2e-v1",
            "input_manifest": {"items": ["image-1"]},
        })
        self.assertEqual(response.status_code, 201, response.text)
        job_id = response.json()["job_id"]

        response = self.client.post("/api/tasks/lease", headers=worker_headers)
        self.assertEqual(response.status_code, 200, response.text)
        lease = response.json()
        self.assertEqual(lease["job_id"], job_id)
        self.assertEqual(lease["input_manifest"], {"items": ["image-1"]})
        lease_headers = {
            **worker_headers,
            "X-Lease-Token": lease["lease_token"],
        }
        response = self.client.get(
            f"/api/tasks/{job_id}/inputs", headers=lease_headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["input_manifest"],
                         {"items": ["image-1"]})
        self.assertEqual(self.client.post(
            f"/api/tasks/{job_id}/start", headers=lease_headers
        ).status_code, 204)

        payload = b'{"image-1":"ok"}'
        digest = hashlib.sha256(payload).hexdigest()
        response = self.client.post(
            f"/api/tasks/{job_id}/artifacts",
            content=payload,
            headers={
                **lease_headers,
                "Content-Type": "application/octet-stream",
                "X-Artifact-Type": "test_result",
                "X-Content-SHA256": digest,
                "X-Artifact-Size": str(len(payload)),
                "X-Schema-Version": "1",
                "X-Pipeline-Config-Digest": "a" * 64,
                "X-Model-Version": "test-model-v1",
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        artifact_id = response.json()["artifact_id"]
        response = self.client.post(
            f"/api/tasks/{job_id}/complete", headers=lease_headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["artifact_id"], artifact_id)
        response = self.client.post(
            f"/api/tasks/{job_id}/complete", headers=lease_headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["artifact_id"], artifact_id)


if __name__ == "__main__":
    unittest.main()
