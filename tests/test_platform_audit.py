"""不可变安全审计及其人用 API 契约。"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image as PillowImage
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.app import PlatformServices, create_app  # noqa: E402
from whitewhale.platform.audit import AuditService  # noqa: E402
from whitewhale.platform.auth import AuthService  # noqa: E402
from whitewhale.platform.imports import BatchImportService  # noqa: E402
from whitewhale.platform.artifacts import WorkerResultService  # noqa: E402
from whitewhale.platform.jobs import JobQueueService, LeaseService  # noqa: E402
from whitewhale.platform.media import MediaService  # noqa: E402
from whitewhale.platform.models import AuditEvent, Base, Batch, Image  # noqa: E402
from whitewhale.platform.storage import StorageLayout  # noqa: E402
from whitewhale.platform.uploads import UploadService  # noqa: E402
from whitewhale.platform.worker_auth import WorkerAuthService  # noqa: E402
from whitewhale.platform.views import ArchiveReadService  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


def _jpeg_bytes() -> bytes:
    output = io.BytesIO()
    PillowImage.new("RGB", (4, 3), "white").save(output, format="JPEG")
    return output.getvalue()


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestPlatformAudit(unittest.TestCase):
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
        self.temp = tempfile.TemporaryDirectory()
        self.layout = StorageLayout(self.temp.name)
        self.layout.initialize()
        self.auth = AuthService(self.sessions)
        self.audit = AuditService(self.sessions)
        services = PlatformServices(
            auth=self.auth,
            uploads=UploadService(self.sessions, self.layout),
            imports=BatchImportService(self.sessions, self.layout),
            media=MediaService(self.sessions, self.layout),
            worker_auth=WorkerAuthService(self.sessions),
            jobs=JobQueueService(self.sessions),
            leases=LeaseService(self.sessions),
            results=WorkerResultService(self.sessions, self.layout),
            views=ArchiveReadService(self.sessions),
            audit=self.audit,
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
        self.temp.cleanup()

    def test_login_media_exif_and_worker_revoke_are_audited(self):
        payload = _jpeg_bytes()
        with self.sessions.begin() as db:
            batch = Batch(
                name="audit", manifest_sha256="a" * 64,
                source_format="generic")
            db.add(batch)
            db.flush()
            image = Image(
                batch_id=batch.id,
                source_path="audit/image.jpg",
                original_relative_path="image.jpg",
                source_sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
                exif_json={"captured_on": "2014-04-19"},
            )
            db.add(image)
            db.flush()
            image_id = image.id
        target = self.layout.resolve("raw", "audit/image.jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

        self.assertEqual(
            self.client.get(f"/api/media/images/{image_id}").status_code, 200)
        exif = self.client.get(f"/api/images/{image_id}/exif")
        self.assertEqual(exif.status_code, 200, exif.text)
        self.assertEqual(exif.json()["captured_on"], "2014-04-19")

        headers = {"X-CSRF-Token": self.csrf}
        code = self.client.post(
            "/api/workers/registration-codes", headers=headers).json()[
                "registration_code"]
        worker = self.client.post("/api/workers/register", json={
            "registration_code": code,
            "name": "audit-worker",
            "gpu_model": "RTX 4060 Laptop",
            "vram_mb": 8192,
            "cuda_version": "12.6",
            "worker_version": "0.1.0",
            "capabilities": ["batch_archival"],
            "model_versions": ["r4"],
            "capacity": 1,
        }).json()
        revoked = self.client.post(
            f"/api/workers/{worker['device_id']}/revoke", headers=headers)
        self.assertEqual(revoked.status_code, 200, revoked.text)
        workers = self.client.get("/api/workers")
        self.assertEqual(workers.status_code, 200, workers.text)
        self.assertEqual(workers.json()[0]["name"], "audit-worker")
        self.assertFalse(workers.json()[0]["is_active"])
        dashboard = self.client.get("/api/dashboard")
        self.assertEqual(dashboard.status_code, 200, dashboard.text)
        self.assertEqual(dashboard.json()["worker_counts"]["total"], 1)

        response = self.client.get("/api/system/audit")
        self.assertEqual(response.status_code, 200, response.text)
        event_types = {item["event_type"] for item in response.json()}
        self.assertTrue({
            "login_succeeded", "media_downloaded", "exif_read",
            "worker_registered", "worker_token_revoked",
        }.issubset(event_types))
        with self.sessions() as db:
            self.assertGreaterEqual(len(list(db.scalars(select(AuditEvent)))), 5)

    def test_failed_login_is_recorded_without_password(self):
        response = self.client.post("/api/auth/login", json={
            "username": "owner", "password": "definitely wrong password",
        })
        self.assertEqual(response.status_code, 401)
        event = self.audit.recent(limit=1)[0]
        self.assertEqual(event["event_type"], "login_failed")
        self.assertNotIn("password", str(event["detail"]).lower())


if __name__ == "__main__":
    unittest.main()
