"""GPU Worker 查询处理器的下载、检测与紧凑向量产物协议。"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.worker.client import TaskLease  # noqa: E402
from whitewhale.worker.query import (  # noqa: E402
    QueryWorkerConfig,
    make_query_handler,
)


class _InputApi:
    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.downloaded: list[str] = []

    def download_query_image(self, lease: TaskLease, image_id: str) -> bytes:
        self.downloaded.append(image_id)
        return self.payloads[image_id]


class TestWorkerQueryHandler(unittest.TestCase):
    def test_emits_zip_manifest_and_npy_for_multiple_targets(self):
        image_ids = [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
        ]
        payloads = {value: f"image-{value}".encode() for value in image_ids}
        lease = TaskLease(
            job_id="job-query", lease_token="secret",
            task_type="query_inference", required_model_version="reid-r4",
            input_manifest={
                "schema_version": 1,
                "query_request_id":
                    "10000000-0000-0000-0000-000000000001",
                "catalog_id": "20000000-0000-0000-0000-000000000001",
                "model_version": "reid-r4",
                "detector_version": "det-v1",
                "preprocess_id": "crop-v1",
                "feature_dim": 2,
                "top_k": 5,
                "pipeline_config_digest": "a" * 64,
                "row_binding_digest": "b" * 64,
                "images": [{
                    "query_image_id": image_id,
                    "original_relative_path": f"survey/{index}.jpg",
                    "sha256": hashlib.sha256(payloads[image_id]).hexdigest(),
                    "size_bytes": len(payloads[image_id]),
                } for index, image_id in enumerate(image_ids)],
            },
        )
        api = _InputApi(payloads)

        def fake_detect(frame: pd.DataFrame, images_root: Path,
                        crops_root: Path, **kwargs) -> pd.DataFrame:
            del images_root, kwargs
            crops_root.mkdir(parents=True)
            rows = []
            for index, source in frame.iterrows():
                target_count = 2 if index == 0 else 1
                for crop_index in range(target_count):
                    crop_key = (str(source["image_id"]) if target_count == 1
                                else f"{source['image_id']}--{crop_index:03d}")
                    (crops_root / f"{crop_key}.jpg").write_bytes(b"crop")
                    rows.append({
                        "image_id": source["image_id"],
                        "relative_path": source["relative_path"],
                        "crop_index": crop_index,
                        "crop_key": crop_key,
                        "x": 1 + crop_index, "y": 2,
                        "w": 20, "h": 10,
                        "det_conf": 0.9 - crop_index * 0.1,
                        "fallback": False,
                    })
            return pd.DataFrame(rows)

        vectors = np.asarray(
            [[3.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float32)
        handler = make_query_handler(
            api,
            QueryWorkerConfig(Path("det.pt"), Path("reid.pt")),
            detect_fn=fake_detect,
            embedder_factory=lambda *args, **kwargs: type(
                "Embedder", (), {"preprocess_id": "crop-v1"})(),
            extract_fn=lambda *args, **kwargs: (vectors, pd.DataFrame()),
        )

        artifact = handler(lease)

        self.assertEqual(api.downloaded, image_ids)
        self.assertEqual(artifact.artifact_type, "query_embedding")
        self.assertEqual(artifact.row_binding_digest, "b" * 64)
        with zipfile.ZipFile(io.BytesIO(artifact.data)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            embeddings = np.load(
                io.BytesIO(archive.read("embeddings.npy")),
                allow_pickle=False)
        self.assertEqual(embeddings.shape, (3, 2))
        np.testing.assert_allclose(
            np.linalg.norm(embeddings, axis=1), np.ones(3), atol=1e-6)
        self.assertEqual(len(manifest["detections"]), 3)
        self.assertNotIn("embedding", manifest["detections"][0])
        self.assertEqual(
            [row["query_image_id"] for row in manifest["detections"]],
            [image_ids[0], image_ids[0], image_ids[1]],
        )

    def test_rejects_download_hash_mismatch_before_detection(self):
        image_id = "00000000-0000-0000-0000-000000000001"
        lease = TaskLease(
            job_id="job-query", lease_token="secret",
            task_type="query_inference", required_model_version="reid-r4",
            input_manifest={
                "schema_version": 1,
                "query_request_id":
                    "10000000-0000-0000-0000-000000000001",
                "catalog_id": "20000000-0000-0000-0000-000000000001",
                "model_version": "reid-r4", "detector_version": "det-v1",
                "preprocess_id": "crop-v1", "feature_dim": 2, "top_k": 5,
                "pipeline_config_digest": "a" * 64,
                "row_binding_digest": "b" * 64,
                "images": [{
                    "query_image_id": image_id,
                    "original_relative_path": "query.jpg",
                    "sha256": "0" * 64,
                    "size_bytes": 9,
                }],
            },
        )
        handler = make_query_handler(
            _InputApi({image_id: b"different"}),
            QueryWorkerConfig(Path("det.pt"), Path("reid.pt")),
            detect_fn=lambda *args, **kwargs: self.fail("不应运行检测"),
        )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            handler(lease)


if __name__ == "__main__":
    unittest.main()
