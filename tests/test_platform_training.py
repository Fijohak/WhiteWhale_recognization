"""M4 Dataset/Split、训练派发和模型上线门禁。"""
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
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.platform.auth import AuthService  # noqa: E402
from whitewhale.platform.catalogs import build_flat_ip_index  # noqa: E402
from whitewhale.platform.jobs import JobQueueService  # noqa: E402
from whitewhale.platform.media import MediaNotFound, MediaService  # noqa: E402
from whitewhale.platform.models import (  # noqa: E402
    ActiveCatalogPointer, Artifact, Base, Batch, CatalogVersion,
    ConfirmedIndividual, Crop, DatasetVersion, Image, Job, JobAttempt,
    ModelVersion, Observation, ProductionModelPointer, ReviewTask,
    WorkerDevice,
)
from whitewhale.platform.states import JobState  # noqa: E402
from whitewhale.platform.storage import StorageLayout  # noqa: E402
from whitewhale.platform.training import (  # noqa: E402
    DatasetSampleSpec, DatasetService, ModelManifest,
    TrainingLifecycleService,
)


TEST_DATABASE_URL = os.getenv("WHITEWHALE_TEST_DATABASE_URL")


@unittest.skipUnless(TEST_DATABASE_URL, "需要 WHITEWHALE_TEST_DATABASE_URL")
class TestTrainingLifecycle(unittest.TestCase):
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
        self.storage = StorageLayout(self.temp_dir.name)
        self.storage.initialize()
        auth = AuthService(self.sessions)
        self.admin = auth.bootstrap_admin(
            "train-admin", "correct horse battery staple")
        self.reviewer = auth.create_user(
            "train-reviewer", "another correct horse battery staple",
            roles={"reviewer"})
        with self.sessions.begin() as db:
            batch = Batch(name="training", manifest_sha256="a" * 64,
                          source_format="generic")
            db.add(batch)
            task = ReviewTask(
                task_type="identity_match", subject_type="candidate_cluster",
                subject_id=uuid.uuid4(), status="resolved",
                required_reviewers=3, policy_version="v1")
            db.add(task)
            db.flush()
            individuals = [ConfirmedIndividual(display_name=f"TRAIN-{index}")
                           for index in range(4)]
            db.add_all(individuals)
            db.flush()
            observations = []
            for index, individual in enumerate(individuals):
                image = Image(
                    batch_id=batch.id, source_path=f"raw/{index}.jpg",
                    original_relative_path=f"{index}.jpg",
                    source_sha256=f"{index + 1:064x}", size_bytes=1)
                db.add(image)
                db.flush()
                crop = Crop(
                    image_id=image.id, crop_index=0, x=0, y=0,
                    width=10, height=10, detector_version="det-v1",
                    artifact_path=f"training/{index}.jpg")
                db.add(crop)
                db.flush()
                observation = Observation(
                    individual_id=individual.id, crop_id=crop.id,
                    source_review_task_id=task.id)
                db.add(observation)
                observations.append(observation)
            db.flush()
            worker = WorkerDevice(
                name="train-4060", gpu_model="RTX 4060 Laptop",
                vram_mb=8192, cuda_version="12.6", worker_version="0.1.0",
                capabilities=["reid_training", "fixed_evaluation",
                              "catalog_rebuild"],
                model_versions=[],
            )
            db.add(worker)
            db.flush()
            self.observation_ids = [item.id for item in observations]
            self.crop_ids = [item.crop_id for item in observations]
            self.worker_id = worker.id
        for index in range(4):
            path = self.storage.resolve("artifacts", f"training/{index}.jpg")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"crop-{index}".encode())
        self.samples = [DatasetSampleSpec(
            observation_id=observation_id,
            label_source="project_verified",
            split=split,
            sequence_key=f"sequence-{index}",
            encounter_key=f"encounter-{index}",
            duplicate_group=f"duplicate-{index}",
            data_license="project_authorized",
        ) for index, (observation_id, split) in enumerate(zip(
            self.observation_ids,
            ["train", "val", "calibration", "test"],
            strict=True,
        ))]

    def tearDown(self):
        self.temp_dir.cleanup()

    def _freeze_dataset(self) -> uuid.UUID:
        return DatasetService(self.sessions).freeze(
            name="strict-v1",
            protocol="known_identity_update",
            samples=self.samples,
            created_by_user_id=self.admin.id,
            rights_snapshot={"scope": "project_internal"},
        )

    def _artifact_for_job(
        self, job_id: uuid.UUID, artifact_type: str, payload: bytes,
    ) -> uuid.UUID:
        digest = hashlib.sha256(payload).hexdigest()
        relative = Path(str(job_id)) / f"{artifact_type}-{digest}.bin"
        path = self.storage.resolve("artifacts", relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
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
                artifact_type=artifact_type,
                relative_path=relative.as_posix(), sha256=digest,
                size_bytes=len(payload), producer_device_id=self.worker_id)
            db.add(artifact)
            db.flush()
            return artifact.id

    def test_freeze_rejects_leakage_and_training_manifest_excludes_test(self):
        leaking = list(self.samples)
        leaking[-1] = DatasetSampleSpec(
            observation_id=leaking[-1].observation_id,
            label_source=leaking[-1].label_source,
            split="test",
            sequence_key=leaking[0].sequence_key,
            encounter_key=leaking[-1].encounter_key,
            duplicate_group=leaking[-1].duplicate_group,
            data_license=leaking[-1].data_license,
        )
        with self.assertRaisesRegex(ValueError, "Sequence"):
            DatasetService(self.sessions).freeze(
                name="leaking", protocol="known_identity_update",
                samples=leaking, created_by_user_id=self.admin.id,
                rights_snapshot={"scope": "project_internal"})

        dataset_id = self._freeze_dataset()
        lifecycle = TrainingLifecycleService(
            self.sessions, JobQueueService(self.sessions), self.storage)
        run_id, job_id = lifecycle.dispatch_training(
            dataset_id,
            task_type="reid_training",
            model_family="metric-learning",
            base_model_id=None,
            config={"epochs": 2, "batch_size": 8},
            seed=7,
            required_vram_mb=4096,
            max_runtime_seconds=3600,
            checkpoint_interval_steps=100,
        )
        with self.sessions() as db:
            dataset = db.get(DatasetVersion, dataset_id)
            self.assertEqual(dataset.status, "frozen")
            manifest = db.get(Job, job_id).input_manifest
            self.assertEqual(
                {sample["split"] for sample in manifest["samples"]},
                {"train", "val", "calibration"},
            )
            self.assertNotIn(
                str(self.observation_ids[-1]),
                {sample["observation_id"]
                 for sample in manifest["samples"]},
            )
            self.assertEqual(str(run_id), manifest["training_run_id"])
        media = MediaService(self.sessions, self.storage)
        self.assertEqual(
            media.leased_crop(job_id, self.crop_ids[0]).path.read_bytes(),
            b"crop-0",
        )
        with self.assertRaisesRegex(MediaNotFound, "输入清单"):
            media.leased_crop(job_id, self.crop_ids[-1])

    def test_model_cannot_be_production_before_evaluation_and_catalog_gate(self):
        dataset_id = self._freeze_dataset()
        lifecycle = TrainingLifecycleService(
            self.sessions, JobQueueService(self.sessions), self.storage)
        run_id, training_job_id = lifecycle.dispatch_training(
            dataset_id,
            task_type="reid_training",
            model_family="metric-learning",
            base_model_id=None,
            config={"epochs": 1}, seed=11,
            required_vram_mb=4096,
            max_runtime_seconds=3600,
            checkpoint_interval_steps=50,
        )
        weights = b"verified-model-weights"
        weight_artifact_id = self._artifact_for_job(
            training_job_id, "model_weights", weights)
        model_id = lifecycle.register_model(
            run_id,
            weight_artifact_id,
            ModelManifest(
                model_family="metric-learning",
                version="reid-r5-candidate",
                sha256=hashlib.sha256(weights).hexdigest(),
                feature_dim=256,
                preprocess_id="crop-v2",
                checkpoint_source="training_run",
                license="project_internal",
                compatible_detector_version="det-v2",
                compatible_crop_config="crop-config-v2",
                compatible_index_schema=1,
            ),
        )
        with self.assertRaisesRegex(ValueError, "评估"):
            lifecycle.request_promotion(model_id, self.reviewer.id)

        evaluation_id, evaluation_job_id = lifecycle.dispatch_evaluation(
            model_id, dataset_id, required_vram_mb=4096)
        with self.sessions() as db:
            evaluation_manifest = db.get(Job, evaluation_job_id).input_manifest
            self.assertEqual(
                {sample["split"] for sample in evaluation_manifest["samples"]},
                {"calibration", "test"},
            )
        metrics = {"rank1": 0.82, "map": 0.74}
        comparison = {
            "baseline_model_version": None,
            "candidate_model_version": "reid-r5-candidate",
            "comparison_protocol": "same_fixed_test",
        }
        thresholds = {"accept": 0.76, "uncertain": 0.61}
        report_payload = json.dumps({
            "evaluation_run_id": str(evaluation_id),
            "model_version_id": str(model_id),
            "dataset_version_id": str(dataset_id),
            "metrics": metrics,
            "production_comparison": comparison,
            "calibrated_thresholds": thresholds,
        }).encode()
        report_id = self._artifact_for_job(
            evaluation_job_id, "evaluation_report", report_payload)
        lifecycle.record_evaluation(
            evaluation_id,
            report_artifact_id=report_id,
            metrics=metrics,
            production_comparison=comparison,
            calibrated_thresholds=thresholds,
        )
        rebuild_job_id = lifecycle.request_promotion(
            model_id, self.reviewer.id)
        with self.sessions() as db:
            self.assertEqual(db.get(ModelVersion, model_id).status,
                             "promotion_pending")
            self.assertEqual(
                db.get(Job, rebuild_job_id).task_type, "catalog_rebuild")
            rebuild_manifest = db.get(Job, rebuild_job_id).input_manifest
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((4, 256)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            vector_bytes = io.BytesIO()
            np.save(vector_bytes, vectors, allow_pickle=False)
            archive.writestr("embeddings.npy", vector_bytes.getvalue())
            archive.writestr("index.faiss", build_flat_ip_index(vectors))
            archive.writestr("manifest.json", json.dumps({
                "schema_version": 1,
                "model_version_id": str(model_id),
                "model_version": "reid-r5-candidate",
                "model_sha256": hashlib.sha256(weights).hexdigest(),
                "feature_dim": 256,
                "preprocess_id": "crop-v2",
                "observation_ids": [item["observation_id"]
                                    for item in rebuild_manifest["observations"]],
                "row_binding_digest": rebuild_manifest["row_binding_digest"],
            }))
        rebuild_artifact_id = self._artifact_for_job(
            rebuild_job_id, "catalog_rebuild", buffer.getvalue())
        catalog_id = lifecycle.ingest_catalog_rebuild(
            model_id, rebuild_artifact_id, self.reviewer.id)
        with self.sessions() as db:
            model = db.get(ModelVersion, model_id)
            self.assertEqual(model.status, "production")
            pointer = db.get(ProductionModelPointer, "metric-learning")
            self.assertEqual(pointer.model_version_id, model_id)

    def test_detector_model_promotes_without_embedding_catalog(self):
        dataset_id = self._freeze_dataset()
        lifecycle = TrainingLifecycleService(
            self.sessions, JobQueueService(self.sessions), self.storage)
        run_id, training_job_id = lifecycle.dispatch_training(
            dataset_id, task_type="detector_training",
            model_family="dolphin-detector", base_model_id=None,
            config={"epochs": 1}, seed=13, required_vram_mb=4096,
            max_runtime_seconds=3600, checkpoint_interval_steps=50)
        weights = b"detector-weights"
        artifact_id = self._artifact_for_job(
            training_job_id, "model_weights", weights)
        model_id = lifecycle.register_model(
            run_id, artifact_id, ModelManifest(
                model_family="dolphin-detector", version="detector-v2",
                sha256=hashlib.sha256(weights).hexdigest(), feature_dim=None,
                preprocess_id="raw-image-v1",
                checkpoint_source="training_run", license="project_internal",
                compatible_detector_version=None,
                compatible_crop_config="crop-config-v2",
                compatible_index_schema=1))
        evaluation_id, evaluation_job_id = lifecycle.dispatch_evaluation(
            model_id, dataset_id, required_vram_mb=4096)
        metrics = {"precision_iou50": 0.9, "recall_iou50": 0.8,
                   "f1": 0.847}
        comparison = {"baseline_model_version": None,
                      "candidate_f1": 0.847, "f1_delta": None}
        thresholds = {"accept": 0.55, "uncertain": 0.44}
        report = json.dumps({
            "evaluation_run_id": str(evaluation_id),
            "model_version_id": str(model_id),
            "dataset_version_id": str(dataset_id),
            "metrics": metrics,
            "production_comparison": comparison,
            "calibrated_thresholds": thresholds,
        }).encode()
        report_id = self._artifact_for_job(
            evaluation_job_id, "evaluation_report", report)
        lifecycle.record_evaluation(
            evaluation_id, report_artifact_id=report_id, metrics=metrics,
            production_comparison=comparison,
            calibrated_thresholds=thresholds)
        self.assertIsNone(lifecycle.request_promotion(
            model_id, self.reviewer.id))
        with self.sessions() as db:
            self.assertEqual(db.get(ModelVersion, model_id).status,
                             "production")
            pointer = db.get(ProductionModelPointer, "dolphin-detector")
            self.assertEqual(pointer.model_version_id, model_id)


if __name__ == "__main__":
    unittest.main()
