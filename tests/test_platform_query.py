"""分片上传 → GPU Query Job → 固定 Catalog Top-K 的闭环。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.auth import AuthService  # noqa: E402
from whitewhale.platform.catalogs import (  # noqa: E402
    CatalogEntry, CatalogService, build_flat_ip_index,
)
from whitewhale.platform.jobs import JobQueueService  # noqa: E402
from whitewhale.platform.models import (  # noqa: E402
    Artifact, ArtifactManifest, Base, Batch, ConfirmedIndividual, Crop,
    Image, Job, JobAttempt, Observation, QueryImage, QueryRequest,
    ReviewTask, WorkerDevice,
)
from whitewhale.platform.query import QueryService  # noqa: E402
from whitewhale.platform.states import JobState  # noqa: E402
from whitewhale.platform.storage import StorageLayout  # noqa: E402
from whitewhale.platform.uploads import UploadFileSpec, UploadService  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestQueryService(unittest.TestCase):
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
        self.storage = StorageLayout(self.temp.name)
        self.storage.initialize()
        self.owner = AuthService(self.sessions).bootstrap_admin(
            "query-owner", "correct horse battery staple")
        with self.sessions.begin() as db:
            batch = Batch(name="history", manifest_sha256="a" * 64,
                          source_format="generic")
            task = ReviewTask(
                task_type="identity_match", subject_type="candidate_cluster",
                subject_id=self.owner.id, status="resolved",
                required_reviewers=3, policy_version="v1")
            individual = ConfirmedIndividual(display_name="DOLPHIN-001")
            db.add_all([batch, task, individual])
            db.flush()
            image = Image(
                batch_id=batch.id, source_path="history/one.jpg",
                original_relative_path="one.jpg", source_sha256="b" * 64,
                size_bytes=1)
            db.add(image)
            db.flush()
            crop = Crop(
                image_id=image.id, crop_index=0, x=0, y=0,
                width=10, height=10, detector_version="det-v1",
                artifact_path="history/one.jpg")
            db.add(crop)
            db.flush()
            observation = Observation(
                individual_id=individual.id, crop_id=crop.id,
                source_review_task_id=task.id, side="left", quality=0.9)
            db.add(observation)
            db.flush()
            self.observation_id = observation.id
            worker = WorkerDevice(
                name="query-worker", gpu_model="RTX 4060 Laptop",
                vram_mb=8192, cuda_version="12.6", worker_version="v1",
                capabilities=["query_inference"], model_versions=["r4"])
            db.add(worker)
            db.flush()
            self.worker_id = worker.id
        crop_file = self.storage.resolve("artifacts", "history/one.jpg")
        crop_file.parent.mkdir(parents=True, exist_ok=True)
        crop_file.write_bytes(b"crop")
        catalogs = CatalogService(self.sessions, self.storage)
        vector = np.asarray([[1.0, 0.0]], dtype=np.float32)
        catalog_id = catalogs.stage(
            [CatalogEntry(self.observation_id, vector[0])],
            model_version="r4", calibration_status="provisional_unvalidated",
            index_bytes=build_flat_ip_index(vector))
        catalogs.activate(catalog_id)
        self.catalog_id = catalog_id
        self.uploads = UploadService(self.sessions, self.storage, chunk_size=1024)

    def tearDown(self):
        self.temp.cleanup()

    def _completed_upload(self) -> tuple[object, bytes]:
        payload = b"query-image"
        grant = self.uploads.create_session(
            owner_user_id=self.owner.id, batch_name="query",
            source_format="generic", files=[UploadFileSpec(
                relative_path="folder/query.jpg", size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest())])
        file_id = grant.files[0].file_id
        self.uploads.put_part(
            grant.session_id, file_id, 0, payload,
            hashlib.sha256(payload).hexdigest())
        self.uploads.complete_file(grant.session_id, file_id)
        self.uploads.complete_session(grant.session_id)
        return grant.session_id, payload

    def test_dispatches_bound_query_and_projects_verified_top_k(self):
        upload_id, payload = self._completed_upload()
        service = QueryService(
            self.sessions, self.storage, JobQueueService(self.sessions),
            CatalogService(self.sessions, self.storage))
        query_id = service.dispatch_upload(
            upload_id, owner_user_id=self.owner.id, k=3,
            detector_version="det-v1", required_vram_mb=4096)
        with self.sessions() as db:
            query = db.get(QueryRequest, query_id)
            image = db.query(QueryImage).filter_by(
                query_request_id=query_id).one()
            job = db.get(Job, query.job_id)
            self.assertEqual(job.task_type, "query_inference")
            self.assertEqual(job.input_manifest["catalog_id"],
                             str(self.catalog_id))
            self.assertEqual(job.input_manifest["images"][0]["sha256"],
                             hashlib.sha256(payload).hexdigest())
            self.assertEqual(
                self.storage.resolve("raw", image.source_path).read_bytes(),
                payload)
            job_id = job.id
        report = json.dumps({
            "schema_version": 1,
            "query_request_id": str(query_id),
            "catalog_id": str(self.catalog_id),
            "model_version": "r4",
            "preprocess_id": "legacy",
            "detections": [{
                "query_image_id": str(image.id), "crop_index": 0,
                "bbox": [1, 2, 10, 8], "quality": 0.95,
                "embedding": [1.0, 0.0],
            }],
        }, separators=(",", ":")).encode()
        digest = hashlib.sha256(report).hexdigest()
        relative = Path(str(job_id)) / "query-result.json"
        target = self.storage.resolve("artifacts", relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(report)
        with self.sessions.begin() as db:
            job = db.get(Job, job_id)
            job.state = JobState.SUCCEEDED
            attempt = JobAttempt(
                job_id=job_id, attempt_number=1,
                worker_device_id=self.worker_id,
                outcome=JobState.SUCCEEDED.value)
            db.add(attempt)
            db.flush()
            artifact = Artifact(
                job_id=job_id, attempt_id=attempt.id,
                artifact_type="query_embedding", relative_path=relative.as_posix(),
                sha256=digest, size_bytes=len(report),
                producer_device_id=self.worker_id)
            db.add(artifact)
            db.flush()
            db.add(ArtifactManifest(
                artifact_id=artifact.id, schema_version=1,
                model_version="r4", preprocess_id="legacy",
                pipeline_config_digest=job.input_manifest[
                    "pipeline_config_digest"], detail={}))
        result = service.result(query_id, requester_user_id=self.owner.id,
                                is_admin=False)
        self.assertEqual(result["status"], "succeeded")
        match = result["detections"][0]["matches"][0]
        self.assertEqual(match["individual_name"], "DOLPHIN-001")
        self.assertAlmostEqual(match["score"], 1.0)
        self.assertEqual(match["side"], "left")
        self.assertEqual(result["catalog_id"], str(self.catalog_id))

    def test_query_upload_cannot_be_dispatched_by_another_user(self):
        upload_id, _ = self._completed_upload()
        service = QueryService(
            self.sessions, self.storage, JobQueueService(self.sessions),
            CatalogService(self.sessions, self.storage))
        with self.assertRaisesRegex(ValueError, "所有者"):
            service.dispatch_upload(
                upload_id, owner_user_id=os.urandom(16).hex(), k=3,
                detector_version="det-v1", required_vram_mb=4096)


if __name__ == "__main__":
    unittest.main()
