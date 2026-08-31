"""GPU 训练 Worker 只消费租约内样本并可周期上传 Checkpoint。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
import hashlib

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.worker.client import ArtifactOutput, TaskLease  # noqa: E402
from whitewhale.worker.training import make_training_handler  # noqa: E402
from whitewhale.reid.training import split_frozen_dataset  # noqa: E402


class _Api:
    def __init__(self):
        self.downloaded = []
        self.submitted = []
        self.checkpoints = []

    def download_input_crop(self, lease, crop_id):
        self.downloaded.append(crop_id)
        return f"jpeg-{crop_id}".encode()

    def download_input_artifact(self, lease, artifact_id):
        self.downloaded.append(artifact_id)
        return b"resume-checkpoint"

    def download_input_image(self, lease, image_id):
        self.downloaded.append(image_id)
        return f"image-{image_id}".encode()

    def submit(self, lease, artifact):
        self.submitted.append(artifact)
        return f"artifact-{len(self.submitted)}"

    def register_checkpoint(
        self, lease, artifact_id, *, stage, epoch, step,
    ):
        self.checkpoints.append((artifact_id, stage, epoch, step))
        return f"checkpoint-{len(self.checkpoints)}"


class TestTrainingWorker(unittest.TestCase):
    def test_frozen_split_adapter_preserves_server_assignments(self):
        frame = pd.DataFrame([
            {"frozen_split": "train", "series_unit": "s-train",
             "encounter_key": "e-train", "duplicate_group": "d-train"},
            {"frozen_split": "val", "series_unit": "s-val",
             "encounter_key": "e-val", "duplicate_group": "d-val"},
            {"frozen_split": "calibration", "series_unit": "s-cal",
             "encounter_key": "e-cal", "duplicate_group": "d-cal"},
        ])
        train, val = split_frozen_dataset(frame)
        self.assertEqual(train["series_unit"].tolist(), ["s-train"])
        self.assertEqual(val["series_unit"].tolist(), ["s-val"])

        leaking = frame.copy()
        leaking.loc[1, "series_unit"] = "s-train"
        with self.assertRaisesRegex(ValueError, "series_unit"):
            split_frozen_dataset(leaking)

    def test_handler_downloads_only_manifest_crops_and_streams_checkpoint(self):
        api = _Api()

        def executor(manifest, sample_paths, workspace, checkpoint_callback):
            self.assertEqual(set(sample_paths), {"crop-a", "crop-b"})
            self.assertTrue(all(path.is_file()
                                for path in sample_paths.values()))
            checkpoint_callback(b"checkpoint", stage=1, epoch=1, step=10)
            return [ArtifactOutput(
                artifact_type="model_weights", data=b"weights",
                schema_version=1,
                pipeline_config_digest=manifest["config_digest"],
            )]

        lease = TaskLease(
            job_id="job-train", lease_token="lease",
            task_type="reid_training", required_model_version=None,
            input_manifest={
                "schema_version": 1,
                "config_digest": "a" * 64,
                "samples": [
                    {"crop_id": "crop-a", "split": "train"},
                    {"crop_id": "crop-b", "split": "val"},
                ],
            },
        )
        outputs = make_training_handler(api, executor=executor)(lease)
        self.assertEqual(api.downloaded, ["crop-a", "crop-b"])
        self.assertEqual(api.submitted[0].artifact_type,
                         "training_checkpoint")
        self.assertEqual(api.checkpoints, [("artifact-1", 1, 1, 10)])
        self.assertEqual(outputs[0].artifact_type, "model_weights")

    def test_training_handler_rejects_test_samples_defensively(self):
        api = _Api()
        lease = TaskLease(
            job_id="job-leak", lease_token="lease",
            task_type="reid_training", required_model_version=None,
            input_manifest={
                "schema_version": 1,
                "config_digest": "a" * 64,
                "samples": [{"crop_id": "test-crop", "split": "test"}],
            },
        )
        with self.assertRaisesRegex(ValueError, "test"):
            make_training_handler(
                api, executor=lambda *args: [])(lease)
        self.assertEqual(api.downloaded, [])

    def test_detector_handler_downloads_unique_source_images(self):
        api = _Api()

        def executor(manifest, sample_paths, workspace, checkpoint_callback):
            self.assertEqual(set(sample_paths), {"image-a", "image-b"})
            return [ArtifactOutput(
                artifact_type="model_weights", data=b"weights",
                schema_version=1,
                pipeline_config_digest=manifest["config_digest"],
            )]

        lease = TaskLease(
            job_id="job-detector", lease_token="lease",
            task_type="detector_training", required_model_version=None,
            input_manifest={
                "schema_version": 1,
                "config_digest": "a" * 64,
                "samples": [
                    {"crop_id": "crop-a", "image_id": "image-a",
                     "split": "train"},
                    {"crop_id": "crop-b", "image_id": "image-a",
                     "split": "train"},
                    {"crop_id": "crop-c", "image_id": "image-b",
                     "split": "val"},
                ],
            },
        )
        make_training_handler(api, executor=executor)(lease)
        self.assertEqual(api.downloaded, ["image-a", "image-b"])

    def test_verified_resume_is_downloaded_and_passed_to_executor(self):
        api = _Api()

        def executor(manifest, sample_paths, workspace, checkpoint_callback):
            resume_path = Path(manifest["local_resume_path"])
            self.assertEqual(resume_path.read_bytes(), b"resume-checkpoint")
            return [ArtifactOutput(
                artifact_type="model_weights", data=b"weights",
                schema_version=1,
                pipeline_config_digest=manifest["config_digest"],
            )]

        lease = TaskLease(
            job_id="job-resume", lease_token="lease",
            task_type="reid_training", required_model_version=None,
            input_manifest={
                "schema_version": 1,
                "config_digest": "a" * 64,
                "samples": [{"crop_id": "crop-a", "split": "train"}],
                "resume": {
                    "artifact_id": "checkpoint-a",
                    "sha256": hashlib.sha256(
                        b"resume-checkpoint").hexdigest(),
                },
            },
        )
        make_training_handler(api, executor=executor)(lease)
        self.assertEqual(api.downloaded, ["crop-a", "checkpoint-a"])


if __name__ == "__main__":
    unittest.main()
