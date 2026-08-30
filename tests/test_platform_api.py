"""M1 浏览器 API：登录、CSRF、分片上传和 Batch 导入。"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image as PillowImage
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.app import PlatformServices, create_app  # noqa: E402
from whitewhale.platform.auth import AuthService  # noqa: E402
from whitewhale.platform.cooccurrence import CooccurrenceService  # noqa: E402
from whitewhale.platform.imports import BatchImportService  # noqa: E402
from whitewhale.platform.media import MediaService  # noqa: E402
from whitewhale.platform.models import Base, Batch, Crop, Image  # noqa: E402
from whitewhale.platform.reviews import ReviewService  # noqa: E402
from whitewhale.platform.storage import StorageLayout  # noqa: E402
from whitewhale.platform.uploads import UploadService  # noqa: E402
from whitewhale.platform.views import ArchiveReadService  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


def _jpeg_bytes() -> bytes:
    output = io.BytesIO()
    PillowImage.new("RGB", (4, 3), "white").save(output, format="JPEG")
    return output.getvalue()


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestPlatformBrowserApi(unittest.TestCase):
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
        self.layout = StorageLayout(self.temp_dir.name)
        self.layout.initialize()
        self.auth = AuthService(self.sessions)
        self.app = create_app(services=PlatformServices(
            auth=self.auth,
            uploads=UploadService(self.sessions, self.layout, chunk_size=1024),
            imports=BatchImportService(self.sessions, self.layout),
            media=MediaService(self.sessions, self.layout),
            reviews=ReviewService(self.sessions),
            views=ArchiveReadService(self.sessions),
            cooccurrence=CooccurrenceService(self.sessions),
        ))
        self.client = TestClient(self.app, base_url="https://testserver")

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def _login(self) -> str:
        response = self.client.post("/api/auth/bootstrap", json={
            "username": "owner",
            "password": "correct horse battery staple",
        })
        self.assertEqual(response.status_code, 201, response.text)
        response = self.client.post("/api/auth/login", json={
            "username": "owner",
            "password": "correct horse battery staple",
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        self.assertIn("Secure", response.headers["set-cookie"])
        return response.json()["csrf_token"]

    def test_csrf_protected_folder_upload_imports_a_batch(self):
        csrf = self._login()
        payload = _jpeg_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        manifest = {
            "batch_name": "20140419 02",
            "source_format": "idolphin",
            "files": [{
                "relative_path": "70-79/05/a.jpg",
                "size_bytes": len(payload),
                "sha256": digest,
            }],
        }
        denied = self.client.post("/api/uploads", json=manifest)
        self.assertEqual(denied.status_code, 403)

        headers = {"X-CSRF-Token": csrf}
        response = self.client.post(
            "/api/uploads", json=manifest, headers=headers)
        self.assertEqual(response.status_code, 201, response.text)
        grant = response.json()
        session_id = grant["session_id"]
        file_id = grant["files"][0]["file_id"]

        response = self.client.put(
            f"/api/uploads/{session_id}/files/{file_id}/parts/0",
            content=payload,
            headers={**headers, "X-Content-SHA256": digest,
                     "Content-Type": "application/octet-stream"},
        )
        self.assertEqual(response.status_code, 204, response.text)
        response = self.client.post(
            f"/api/uploads/{session_id}/files/{file_id}/complete",
            headers=headers,
        )
        self.assertEqual(response.status_code, 200, response.text)
        response = self.client.post(
            f"/api/uploads/{session_id}/complete", headers=headers)
        self.assertEqual(response.status_code, 200, response.text)
        response = self.client.post(
            f"/api/uploads/{session_id}/import", headers=headers)
        self.assertEqual(response.status_code, 201, response.text)

        with self.sessions() as session:
            self.assertEqual(session.scalar(
                select(func.count()).select_from(Batch)), 1)
            self.assertEqual(session.scalar(
                select(func.count()).select_from(Image)), 1)
            image_id = session.scalar(select(Image.id))

        response = self.client.get(f"/api/media/images/{image_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, payload)
        batches = self.client.get("/api/batches")
        self.assertEqual(batches.status_code, 200, batches.text)
        self.assertEqual(batches.json()[0]["name"], "20140419 02")
        self.assertEqual(batches.json()[0]["image_count"], 1)
        anonymous = TestClient(self.app, base_url="https://testserver")
        try:
            self.assertEqual(
                anonymous.get(f"/api/media/images/{image_id}").status_code,
                401,
            )
        finally:
            anonymous.close()

    def test_auth_me_and_logout_revoke_the_cookie_session(self):
        csrf = self._login()
        response = self.client.get("/api/auth/me")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["roles"], ["admin"])
        response = self.client.post(
            "/api/auth/logout", headers={"X-CSRF-Token": csrf})
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_cooccurrence_read_api_is_authenticated_and_type_specific(self):
        csrf = self._login()
        reviewers = [
            self.auth.create_user(
                f"reviewer-{index}", f"strong reviewer password {index}",
                roles={"reviewer"},
            ).id
            for index in range(3)
        ]
        with self.sessions.begin() as db:
            batch = Batch(name="multi", manifest_sha256="a" * 64,
                          source_format="generic")
            db.add(batch)
            db.flush()
            image = Image(
                batch_id=batch.id, source_path="multi.jpg",
                original_relative_path="multi.jpg",
                source_sha256="b" * 64, size_bytes=1,
            )
            db.add(image)
            db.flush()
            crops = [Crop(
                image_id=image.id, crop_index=index, x=index * 10, y=0,
                width=10, height=10, detector_version="det-v2",
                artifact_path=f"crop/{index}.jpg",
            ) for index in range(2)]
            db.add_all(crops)
            db.flush()
            image_id = image.id
            crop_ids = [crop.id for crop in crops]

        response = self.client.post(
            "/api/cooccurrences",
            headers={"X-CSRF-Token": csrf},
            json={
                "image_id": str(image_id),
                "crop_ids": [str(crop_id) for crop_id in crop_ids],
                "reviewer_ids": [str(reviewer_id) for reviewer_id in reviewers],
                "provenance_artifact_id": None,
            },
        )
        self.assertEqual(response.status_code, 201, response.text)
        event_id = uuid.UUID(response.json()["event_id"])
        detail = self.client.get(f"/api/cooccurrences/{event_id}")
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(detail.json()["status"], "review_pending")
        self.assertEqual(len(detail.json()["crops"]), 2)
        relationships = self.client.get("/api/relationships")
        self.assertEqual(relationships.status_code, 200, relationships.text)
        self.assertEqual(relationships.json(), [])

        anonymous = TestClient(self.app, base_url="https://testserver")
        try:
            self.assertEqual(
                anonymous.get(f"/api/cooccurrences/{event_id}").status_code,
                401,
            )
        finally:
            anonymous.close()


if __name__ == "__main__":
    unittest.main()
