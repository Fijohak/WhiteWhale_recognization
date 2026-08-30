"""不可变 Faiss Catalog 的验证、原子激活、查询和回滚。"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import faiss
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.catalogs import (  # noqa: E402
    CatalogEntry,
    CatalogService,
    CatalogValidationError,
    build_flat_ip_index,
)
from whitewhale.platform.app import PlatformServices, create_app  # noqa: E402
from whitewhale.platform.auth import AuthService  # noqa: E402
from whitewhale.platform.imports import BatchImportService  # noqa: E402
from whitewhale.platform.models import (  # noqa: E402
    ActiveCatalogPointer, Base, Batch, CatalogVersion, ConfirmedIndividual,
    Crop, Image, Observation, ReviewTask,
)
from whitewhale.platform.storage import StorageLayout  # noqa: E402
from whitewhale.platform.uploads import UploadService  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestCatalogLifecycle(unittest.TestCase):
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
            batch = Batch(name="batch", manifest_sha256="a" * 64,
                          source_format="generic")
            task = ReviewTask(
                task_type="identity_match", subject_type="candidate_cluster",
                subject_id=uuid.uuid4(), status="resolved",
                required_reviewers=3, policy_version="v1")
            db.add_all([batch, task])
            db.flush()
            individuals = [
                ConfirmedIndividual(display_name=f"WW-TEST-{index}")
                for index in range(2)
            ]
            db.add_all(individuals)
            db.flush()
            observations = []
            for index, individual in enumerate(
                    [individuals[0], individuals[0], individuals[1]]):
                image = Image(
                    batch_id=batch.id, source_path=f"b/{index}.jpg",
                    original_relative_path=f"{index}.jpg",
                    source_sha256=f"{index + 1:064x}", size_bytes=1)
                db.add(image)
                db.flush()
                crop = Crop(
                    image_id=image.id, crop_index=0, x=0, y=0,
                    width=10, height=10, detector_version="det-v1",
                    artifact_path=f"c/{index}.jpg")
                db.add(crop)
                db.flush()
                observation = Observation(
                    individual_id=individual.id, crop_id=crop.id,
                    source_review_task_id=task.id)
                db.add(observation)
                observations.append(observation)
            db.flush()
            self.observation_ids = [item.id for item in observations]
            self.individual_ids = [item.id for item in individuals]

    def test_activation_failure_preserves_active_and_valid_versions_roll_back(self):
        vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
                             dtype=np.float32)
        entries = [CatalogEntry(observation_id, vector) for observation_id, vector
                   in zip(self.observation_ids, vectors, strict=True)]
        index_bytes = build_flat_ip_index(vectors)
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = StorageLayout(tmp_dir)
            layout.initialize()
            service = CatalogService(self.sessions, layout)

            first = service.stage(
                entries, model_version="reid-r4",
                calibration_status="provisional_unvalidated",
                index_bytes=index_bytes)
            service.activate(first)
            self.assertEqual(service.active_catalog_id(), first)

            second = service.stage(
                entries, model_version="reid-r4",
                calibration_status="provisional_unvalidated",
                index_bytes=index_bytes)
            with self.sessions() as db:
                path = layout.resolve(
                    "catalog_versions", db.get(CatalogVersion, second).index_path)
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(CatalogValidationError, "SHA-256"):
                service.activate(second)
            self.assertEqual(service.active_catalog_id(), first)

            path.write_bytes(index_bytes)
            service.activate(second)
            self.assertEqual(service.active_catalog_id(), second)
            service.activate(first)
            self.assertEqual(service.active_catalog_id(), first)
            with self.sessions() as db:
                self.assertEqual(db.scalar(select(func.count()).select_from(
                    ActiveCatalogPointer)), 1)
                self.assertEqual(db.get(CatalogVersion, second).status, "retired")

    def test_search_aggregates_observation_hits_by_individual(self):
        vectors = np.asarray([[1.0, 0.0], [0.98, 0.02], [0.6, 0.8]],
                             dtype=np.float32)
        entries = [CatalogEntry(observation_id, vector) for observation_id, vector
                   in zip(self.observation_ids, vectors, strict=True)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = StorageLayout(tmp_dir)
            layout.initialize()
            service = CatalogService(self.sessions, layout)
            catalog_id = service.stage(
                entries, model_version="reid-r4",
                calibration_status="provisional_unvalidated",
                index_bytes=build_flat_ip_index(vectors))
            service.activate(catalog_id)
            results = service.search(np.asarray([1.0, 0.0], dtype=np.float32), k=2)
            self.assertEqual([item.individual_id for item in results],
                             self.individual_ids)
            self.assertEqual(results[0].support_frames, 2)
            self.assertEqual(results[0].calibration_status,
                             "provisional_unvalidated")

            with self.sessions.begin() as db:
                db.get(Observation, self.observation_ids[0]).individual_id = \
                    self.individual_ids[1]
                db.get(Observation, self.observation_ids[1]).state = "withdrawn"
            remapped = service.search(
                np.asarray([1.0, 0.0], dtype=np.float32), k=2)
            self.assertEqual(
                [item.individual_id for item in remapped],
                [self.individual_ids[1]],
            )
            self.assertEqual(remapped[0].support_frames, 2)
            with self.assertRaisesRegex(
                    CatalogValidationError, "active Observation"):
                service.stage(
                    entries,
                    model_version="reid-r4",
                    calibration_status="provisional_unvalidated",
                    index_bytes=build_flat_ip_index(vectors),
                )

            with self.sessions() as db:
                path = layout.resolve(
                    "catalog_versions",
                    db.get(CatalogVersion, catalog_id).index_path,
                )
            path.write_bytes(b"corrupt-after-activation")
            with self.assertRaisesRegex(CatalogValidationError, "SHA-256"):
                service.search(np.asarray([1.0, 0.0], dtype=np.float32))

    def test_stage_rejects_non_flat_index_even_when_metric_and_rows_match(self):
        vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
                             dtype=np.float32)
        normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        wrapped = faiss.IndexIDMap(faiss.IndexFlatIP(2))
        wrapped.add_with_ids(
            normalized,
            np.arange(len(normalized), dtype=np.int64),
        )
        entries = [CatalogEntry(observation_id, vector)
                   for observation_id, vector
                   in zip(self.observation_ids, vectors, strict=True)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = StorageLayout(tmp_dir)
            layout.initialize()
            service = CatalogService(self.sessions, layout)
            with self.assertRaisesRegex(CatalogValidationError, "类型"):
                service.stage(
                    entries,
                    model_version="reid-r4",
                    calibration_status="provisional_unvalidated",
                    index_bytes=faiss.serialize_index(wrapped).tobytes(),
                )

    def test_authenticated_catalog_api_activates_and_queries_with_provenance(self):
        vectors = np.asarray([[1.0, 0.0], [0.98, 0.02], [0.6, 0.8]],
                             dtype=np.float32)
        entries = [CatalogEntry(observation_id, vector)
                   for observation_id, vector
                   in zip(self.observation_ids, vectors, strict=True)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            layout = StorageLayout(tmp_dir)
            layout.initialize()
            catalogs = CatalogService(self.sessions, layout)
            catalog_id = catalogs.stage(
                entries,
                model_version="reid-r4",
                calibration_status="provisional_unvalidated",
                index_bytes=build_flat_ip_index(vectors),
            )
            app = create_app(services=PlatformServices(
                auth=AuthService(self.sessions),
                uploads=UploadService(self.sessions, layout),
                imports=BatchImportService(self.sessions, layout),
                catalogs=catalogs,
            ))
            with TestClient(app, base_url="https://testserver") as client:
                self.assertEqual(client.get("/api/catalogs").status_code, 401)
                created = client.post("/api/auth/bootstrap", json={
                    "username": "catalog-admin",
                    "password": "correct horse battery staple",
                })
                self.assertEqual(created.status_code, 201, created.text)
                login = client.post("/api/auth/login", json={
                    "username": "catalog-admin",
                    "password": "correct horse battery staple",
                })
                self.assertEqual(login.status_code, 200, login.text)
                admin_csrf = login.json()["csrf_token"]
                reviewer_created = client.post(
                    "/api/users",
                    json={
                        "username": "catalog-reviewer",
                        "password": "another correct horse battery staple",
                        "roles": ["reviewer"],
                    },
                    headers={"X-CSRF-Token": admin_csrf},
                )
                self.assertEqual(
                    reviewer_created.status_code, 201, reviewer_created.text)
                login = client.post("/api/auth/login", json={
                    "username": "catalog-reviewer",
                    "password": "another correct horse battery staple",
                })
                self.assertEqual(login.status_code, 200, login.text)
                csrf = login.json()["csrf_token"]

                listed = client.get("/api/catalogs")
                self.assertEqual(listed.status_code, 200, listed.text)
                self.assertEqual(listed.json()[0]["catalog_id"], str(catalog_id))
                self.assertEqual(listed.json()[0]["status"], "staged")
                denied = client.post(f"/api/catalogs/{catalog_id}/activate")
                self.assertEqual(denied.status_code, 403)
                activated = client.post(
                    f"/api/catalogs/{catalog_id}/activate",
                    headers={"X-CSRF-Token": csrf},
                )
                self.assertEqual(activated.status_code, 200, activated.text)

                active = client.get("/api/catalogs/active")
                self.assertEqual(active.status_code, 200, active.text)
                self.assertEqual(active.json()["model_version"], "reid-r4")
                result = client.post(
                    "/api/query/embedding",
                    json={"embedding": [1.0, 0.0], "k": 2},
                    headers={"X-CSRF-Token": csrf},
                )
                self.assertEqual(result.status_code, 200, result.text)
                body = result.json()
                self.assertEqual(body["catalog_id"], str(catalog_id))
                self.assertEqual(body["model_version"], "reid-r4")
                self.assertEqual(
                    body["calibration_status"],
                    "provisional_unvalidated",
                )
                self.assertEqual(len(body["matches"]), 2)
                self.assertEqual(body["matches"][0]["support_frames"], 2)


if __name__ == "__main__":
    unittest.main()
