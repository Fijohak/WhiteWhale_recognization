"""固定评估与 Catalog Worker 的租约输入和产物协议。"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.worker.client import TaskLease  # noqa: E402
from whitewhale.worker.model_lifecycle import (  # noqa: E402
    _calibrate_detector_threshold,
    _detector_metrics,
    make_model_lifecycle_handler,
)


class _Api:
    def __init__(self):
        self.downloaded = []

    def download_input_artifact(self, lease, artifact_id):
        self.downloaded.append(("artifact", artifact_id))
        return b"weights"

    def download_input_crop(self, lease, crop_id):
        self.downloaded.append(("crop", crop_id))
        return b"jpeg"


class _Embedder:
    def __init__(self, vectors):
        self._vectors = vectors

    def encode_paths(self, paths, batch_size=32):
        return self._vectors[:len(paths)]


class TestModelLifecycleWorker(unittest.TestCase):
    def test_detector_threshold_is_calibrated_on_frozen_boxes(self):
        grouped = {
            "image-a": [{"bbox": [0, 0, 10, 10]}],
            "image-b": [{"bbox": [20, 20, 10, 10]}],
        }
        predictions = {
            "image-a": [((0, 0, 10, 10), 0.9), ((30, 30, 40, 40), 0.2)],
            "image-b": [((20, 20, 30, 30), 0.8)],
        }
        threshold = _calibrate_detector_threshold(
            grouped, predictions, {"image-a", "image-b"})
        self.assertEqual(threshold, 0.8)
        self.assertEqual(_detector_metrics(
            grouped, predictions, {"image-a", "image-b"}, threshold)["f1"],
            1.0)

    def test_fixed_evaluation_calibrates_and_reports_fixed_test(self):
        api = _Api()
        vectors = np.array([
            [1, 0], [1, 0], [0, 1], [0, 1],
            [1, 0], [1, 0], [0, 1], [0, 1],
        ], dtype=np.float32)
        samples = []
        for index in range(8):
            samples.append({
                "crop_id": f"crop-{index}",
                "individual_id": "a" if index % 4 < 2 else "b",
                "sequence_key": f"sequence-{index}",
                "split": "calibration" if index < 4 else "test",
            })
        digest = hashlib.sha256(b"weights").hexdigest()
        lease = TaskLease(
            job_id="evaluation", lease_token="token",
            task_type="fixed_evaluation",
            required_model_version="reid-v2",
            input_manifest={
                "schema_version": 1,
                "evaluation_run_id": "evaluation-run",
                "model_version_id": "model-id",
                "model_version": "reid-v2",
                "model_family": "metric-learning",
                "model_sha256": digest,
                "weight_artifact_id": "weights-id",
                "feature_dim": 2,
                "preprocess_id": "crop-v2",
                "config_digest": "a" * 64,
                "row_binding_digest": "b" * 64,
                "dataset_version_id": "dataset-id",
                "production_model": None,
                "production_comparison": {
                    "baseline_model_version": None,
                    "candidate_model_version": "reid-v2",
                    "comparison_protocol": "same_fixed_test",
                },
                "samples": samples,
            },
        )
        output = make_model_lifecycle_handler(
            api, embedder_factory=lambda *args, **kwargs: _Embedder(vectors),
        )(lease)
        report = json.loads(output.data)
        self.assertEqual(output.artifact_type, "evaluation_report")
        self.assertEqual(report["metrics"]["rank1"], 1.0)
        self.assertEqual(report["calibrated_thresholds"]["accept"], 1.0)
        self.assertEqual(len(api.downloaded), 9)

    def test_catalog_rebuild_binds_rows_and_contains_flat_index(self):
        api = _Api()
        vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
        lease = TaskLease(
            job_id="catalog", lease_token="token",
            task_type="catalog_rebuild",
            required_model_version="reid-v2",
            input_manifest={
                "schema_version": 1,
                "model_version_id": "model-id",
                "model_version": "reid-v2",
                "model_family": "metric-learning",
                "model_sha256": hashlib.sha256(b"weights").hexdigest(),
                "weight_artifact_id": "weights-id",
                "feature_dim": 2,
                "preprocess_id": "crop-v2",
                "config_digest": "a" * 64,
                "row_binding_digest": "b" * 64,
                "observations": [
                    {"observation_id": "observation-a", "crop_id": "crop-a"},
                    {"observation_id": "observation-b", "crop_id": "crop-b"},
                ],
            },
        )
        output = make_model_lifecycle_handler(
            api, embedder_factory=lambda *args, **kwargs: _Embedder(vectors),
        )(lease)
        self.assertEqual(output.artifact_type, "catalog_rebuild")
        with zipfile.ZipFile(io.BytesIO(output.data)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            stored = np.load(io.BytesIO(
                archive.read("embeddings.npy")), allow_pickle=False)
            self.assertTrue(archive.read("index.faiss"))
        self.assertEqual(
            manifest["observation_ids"],
            ["observation-a", "observation-b"],
        )
        np.testing.assert_allclose(stored, vectors)


if __name__ == "__main__":
    unittest.main()
