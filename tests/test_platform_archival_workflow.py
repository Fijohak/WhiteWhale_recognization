"""M2 从 Worker 产物到正式可回滚 Catalog 的归档闭环。"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import uuid
import zipfile
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.archival_workflow import (  # noqa: E402
    ArchivalWorkflowService,
    archival_row_binding_digest,
)
from whitewhale.platform.catalogs import CatalogService  # noqa: E402
from whitewhale.platform.models import (  # noqa: E402
    Artifact,
    ArtifactManifest,
    Base,
    Batch,
    CandidateCluster,
    ConfirmedIndividual,
    Crop,
    CropEmbedding,
    Image,
    Job,
    JobAttempt,
    MatchCandidate,
    Observation,
    ReviewTask,
    Role,
    User,
    UserRole,
    WorkerDevice,
)
from whitewhale.platform.review_policy import ReviewVote  # noqa: E402
from whitewhale.platform.reviews import ReviewService  # noqa: E402
from whitewhale.platform.states import BatchStage, JobState  # noqa: E402
from whitewhale.platform.storage import StorageLayout  # noqa: E402


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


def _result_archive(manifest: dict, vectors: np.ndarray) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        )
        vector_file = io.BytesIO()
        np.save(vector_file, vectors, allow_pickle=False)
        archive.writestr("embeddings.npy", vector_file.getvalue())
        for crop in manifest["crops"]:
            archive.writestr(crop["path"], b"jpeg-" + crop["key"].encode())
    return output.getvalue()


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestArchivalWorkflow(unittest.TestCase):
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
        with self.sessions.begin() as db:
            reviewer_role = Role(name="reviewer")
            db.add(reviewer_role)
            reviewers = [
                User(username=f"reviewer-{index}", password_hash="test")
                for index in range(3)
            ]
            db.add_all(reviewers)
            db.flush()
            db.add_all([
                UserRole(user_id=user.id, role_id=reviewer_role.id)
                for user in reviewers
            ])
            batch = Batch(
                name="new-batch",
                manifest_sha256="a" * 64,
                source_format="generic",
            )
            db.add(batch)
            db.flush()
            images = [
                Image(
                    batch_id=batch.id,
                    source_path=f"raw/{index}.jpg",
                    original_relative_path=f"{index}.jpg",
                    source_sha256=f"{index + 1:064x}",
                    size_bytes=10,
                )
                for index in range(2)
            ]
            db.add_all(images)
            worker = WorkerDevice(
                name="worker-4060",
                gpu_model="RTX 4060 Laptop",
                vram_mb=8192,
                cuda_version="12.6",
                worker_version="0.1.0",
                capabilities=["batch_archival"],
                model_versions=["reid-r4"],
            )
            db.add(worker)
            db.flush()
            job = Job(
                batch_id=batch.id,
                task_type="batch_archival",
                state=JobState.SUCCEEDED,
                required_model_version="reid-r4",
                idempotency_key="archive-new-batch-r4",
                input_manifest={},
            )
            db.add(job)
            db.flush()
            attempt = JobAttempt(
                job_id=job.id,
                attempt_number=1,
                worker_device_id=worker.id,
                outcome=JobState.SUCCEEDED.value,
            )
            db.add(attempt)
            db.flush()
            self.batch_id = batch.id
            self.image_ids = [image.id for image in images]
            self.reviewer_ids = [user.id for user in reviewers]
            self.job_id = job.id
            self.attempt_id = attempt.id
            self.worker_id = worker.id

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_worker_archive_requires_two_reviews_before_catalog_publication(self):
        crop_keys = ["crop-0", "crop-1"]
        row_digest = archival_row_binding_digest(crop_keys)
        pipeline_digest = hashlib.sha256(b"pipeline-v1").hexdigest()
        vectors = np.asarray([[1.0, 0.0], [0.98, 0.02]], dtype=np.float32)
        manifest = {
            "schema_version": 1,
            "batch_id": str(self.batch_id),
            "model_version": "reid-r4",
            "detector_version": "det-v1",
            "preprocess_id": "crop-v1",
            "pipeline_config_digest": pipeline_digest,
            "row_binding_digest": row_digest,
            "crops": [{
                "key": crop_keys[index],
                "image_id": str(image_id),
                "crop_index": 0,
                "bbox": [1, 2, 20, 10],
                "path": f"crops/{index}.jpg",
                "quality": 0.9,
            } for index, image_id in enumerate(self.image_ids)],
            "clusters": [{
                "label": "cluster-0",
                "member_keys": crop_keys,
                "membership_scores": [0.92, 0.88],
                "representative_key": crop_keys[0],
                "matches": [],
            }],
        }
        payload = _result_archive(manifest, vectors)
        digest = hashlib.sha256(payload).hexdigest()
        relative = Path(str(self.job_id)) / str(self.attempt_id) / (
            f"batch_archival-{digest}.bin"
        )
        self.layout.resolve("artifacts", relative).parent.mkdir(
            parents=True, exist_ok=True)
        self.layout.resolve("artifacts", relative).write_bytes(payload)
        with self.sessions.begin() as db:
            artifact = Artifact(
                job_id=self.job_id,
                attempt_id=self.attempt_id,
                artifact_type="batch_archival",
                relative_path=relative.as_posix(),
                sha256=digest,
                size_bytes=len(payload),
                producer_device_id=self.worker_id,
            )
            db.add(artifact)
            db.flush()
            db.add(ArtifactManifest(
                artifact_id=artifact.id,
                schema_version=1,
                row_binding_digest=row_digest,
                model_version="reid-r4",
                detector_version="det-v1",
                preprocess_id="crop-v1",
                pipeline_config_digest=pipeline_digest,
            ))
            artifact_id = artifact.id

        catalogs = CatalogService(self.sessions, self.layout)
        workflow = ArchivalWorkflowService(
            self.sessions, self.layout, catalogs=catalogs)
        cluster_ids = workflow.ingest_artifact(
            artifact_id, purity_reviewer_id=self.reviewer_ids[0])
        self.assertEqual(len(cluster_ids), 1)
        with self.sessions() as db:
            self.assertEqual(db.get(Batch, self.batch_id).stage,
                             BatchStage.CANDIDATE_READY)
            self.assertEqual(db.scalar(
                select(func.count()).select_from(Crop)), 2)
            self.assertEqual(db.scalar(
                select(func.count()).select_from(CropEmbedding)), 2)
            self.assertEqual(db.scalar(
                select(func.count()).select_from(ConfirmedIndividual)), 0)
            purity_task_id = db.scalar(select(ReviewTask.id).where(
                ReviewTask.task_type == "cluster_purity"))

        reviews = ReviewService(self.sessions)
        reviews.submit_vote(
            purity_task_id,
            self.reviewer_ids[0],
            ReviewVote(choice="confirm_cluster"),
        )
        identity_task_id = workflow.advance_after_purity(
            cluster_ids[0], identity_reviewer_ids=self.reviewer_ids)
        with self.sessions() as db:
            self.assertEqual(db.get(Batch, self.batch_id).stage,
                             BatchStage.UNDER_REVIEW)
            self.assertEqual(db.scalar(
                select(func.count()).select_from(MatchCandidate)), 0)

        for reviewer_id in self.reviewer_ids:
            reviews.submit_vote(
                identity_task_id, reviewer_id, ReviewVote(choice="new"))
        individual_id = workflow.apply_identity_review(identity_task_id)
        with self.sessions() as db:
            self.assertEqual(db.get(Batch, self.batch_id).stage,
                             BatchStage.APPROVED)
            self.assertEqual(db.scalar(
                select(func.count()).select_from(Observation)), 2)
            self.assertEqual(
                db.get(CandidateCluster, cluster_ids[0]).state,
                "identity_confirmed",
            )

        catalog_id = workflow.stage_catalog(
            self.batch_id, model_version="reid-r4")
        with self.sessions() as db:
            self.assertEqual(db.get(Batch, self.batch_id).stage,
                             BatchStage.CATALOG_STAGED)
        workflow.publish_catalog(self.batch_id, catalog_id)
        with self.sessions() as db:
            self.assertEqual(db.get(Batch, self.batch_id).stage,
                             BatchStage.PUBLISHED)
        results = catalogs.search(np.asarray([1.0, 0.0], dtype=np.float32))
        self.assertEqual(results[0].individual_id, individual_id)
        self.assertEqual(results[0].support_frames, 2)

    def test_ingest_rejects_manifest_row_binding_mismatch_without_projection(self):
        manifest = {
            "schema_version": 1,
            "batch_id": str(self.batch_id),
            "model_version": "reid-r4",
            "detector_version": "det-v1",
            "preprocess_id": "crop-v1",
            "pipeline_config_digest": "b" * 64,
            "row_binding_digest": "0" * 64,
            "crops": [{
                "key": "crop-0",
                "image_id": str(self.image_ids[0]),
                "crop_index": 0,
                "bbox": [0, 0, 5, 5],
                "path": "crops/0.jpg",
            }],
            "clusters": [],
        }
        payload = _result_archive(
            manifest, np.asarray([[1.0, 0.0]], dtype=np.float32))
        digest = hashlib.sha256(payload).hexdigest()
        relative = Path(str(self.job_id)) / str(self.attempt_id) / "bad.bin"
        path = self.layout.resolve("artifacts", relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        with self.sessions.begin() as db:
            artifact = Artifact(
                job_id=self.job_id,
                attempt_id=self.attempt_id,
                artifact_type="batch_archival",
                relative_path=relative.as_posix(),
                sha256=digest,
                size_bytes=len(payload),
                producer_device_id=self.worker_id,
            )
            db.add(artifact)
            db.flush()
            db.add(ArtifactManifest(
                artifact_id=artifact.id,
                schema_version=1,
                row_binding_digest="0" * 64,
                model_version="reid-r4",
                detector_version="det-v1",
                preprocess_id="crop-v1",
                pipeline_config_digest="b" * 64,
            ))
            artifact_id = artifact.id
        service = ArchivalWorkflowService(
            self.sessions,
            self.layout,
            catalogs=CatalogService(self.sessions, self.layout),
        )
        with self.assertRaisesRegex(ValueError, "行绑定"):
            service.ingest_artifact(
                artifact_id, purity_reviewer_id=self.reviewer_ids[0])
        with self.sessions() as db:
            self.assertEqual(db.scalar(
                select(func.count()).select_from(Crop)), 0)


if __name__ == "__main__":
    unittest.main()
