"""服务器只从正式 Batch 事实生成 GPU 归档任务清单。"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.archival_dispatch import (  # noqa: E402
    ArchivalDispatchRequest,
    ArchivalDispatchService,
)
from whitewhale.platform.jobs import JobQueueService  # noqa: E402
from whitewhale.platform.app import PlatformServices, create_app  # noqa: E402
from whitewhale.platform.auth import AuthService  # noqa: E402
from whitewhale.platform.imports import BatchImportService  # noqa: E402
from whitewhale.platform.models import Base, Batch, Image  # noqa: E402
from whitewhale.platform.storage import StorageLayout  # noqa: E402
from whitewhale.platform.uploads import UploadService  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestArchivalDispatch(unittest.TestCase):
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
            batch = Batch(
                name="dispatch", manifest_sha256="a" * 64,
                source_format="generic")
            db.add(batch)
            db.flush()
            images = [Image(
                batch_id=batch.id,
                source_path=f"raw/{index}.jpg",
                original_relative_path=f"survey/{index}.jpg",
                source_sha256=f"{index + 1:064x}",
                size_bytes=10,
            ) for index in range(2)]
            db.add_all(images)
            db.flush()
            self.batch_id = batch.id
            self.image_ids = [str(image.id) for image in images]

    def test_dispatch_manifest_is_server_derived_and_idempotent(self):
        jobs = JobQueueService(self.sessions)
        service = ArchivalDispatchService(self.sessions, jobs)
        request = ArchivalDispatchRequest(
            model_version="reid-r4",
            detector_version="det-v1",
            preprocess_id="crop-v1",
            pipeline_config={"min_cluster_size": 3, "det_conf": 0.25},
            required_vram_mb=4096,
        )
        first = service.dispatch(self.batch_id, request)
        second = service.dispatch(self.batch_id, request)
        self.assertEqual(first, second)
        snapshot = jobs.get(first)
        self.assertEqual(snapshot.task_type, "batch_archival")
        self.assertEqual(snapshot.required_model_version, "reid-r4")
        self.assertEqual(
            [item["image_id"] for item in snapshot.input_manifest["images"]],
            self.image_ids,
        )
        self.assertEqual(snapshot.input_manifest["min_cluster_size"], 3)
        self.assertNotIn("source_path", snapshot.input_manifest["images"][0])

    def test_operator_api_dispatches_only_with_csrf(self):
        jobs = JobQueueService(self.sessions)
        dispatch = ArchivalDispatchService(self.sessions, jobs)
        with tempfile.TemporaryDirectory() as temporary:
            layout = StorageLayout(temporary)
            layout.initialize()
            app = create_app(services=PlatformServices(
                auth=AuthService(self.sessions),
                uploads=UploadService(self.sessions, layout),
                imports=BatchImportService(self.sessions, layout),
                jobs=jobs,
                archival_dispatch=dispatch,
            ))
            with TestClient(app, base_url="https://testserver") as client:
                client.post("/api/auth/bootstrap", json={
                    "username": "owner",
                    "password": "correct horse battery staple",
                })
                login = client.post("/api/auth/login", json={
                    "username": "owner",
                    "password": "correct horse battery staple",
                })
                body = {
                    "model_version": "reid-r4",
                    "detector_version": "det-v1",
                    "preprocess_id": "crop-v1",
                    "pipeline_config": {"min_cluster_size": 3},
                    "required_vram_mb": 4096,
                }
                denied = client.post(
                    f"/api/batches/{self.batch_id}/archive-jobs", json=body)
                self.assertEqual(denied.status_code, 403)
                response = client.post(
                    f"/api/batches/{self.batch_id}/archive-jobs",
                    json=body,
                    headers={"X-CSRF-Token": login.json()["csrf_token"]},
                )
                self.assertEqual(response.status_code, 201, response.text)


if __name__ == "__main__":
    unittest.main()
