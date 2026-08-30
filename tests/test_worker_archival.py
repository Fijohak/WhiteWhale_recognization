"""GPU Worker 批次归档处理器的下载、算法编排与产物协议。"""
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

from whitewhale.worker.archival import (  # noqa: E402
    ArchivalWorkerConfig,
    make_batch_archival_handler,
)
from whitewhale.worker.client import TaskLease  # noqa: E402


class _InputApi:
    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.downloaded: list[str] = []

    def download_input_image(self, lease: TaskLease, image_id: str) -> bytes:
        self.downloaded.append(image_id)
        return self.payloads[image_id]


class TestWorkerArchivalHandler(unittest.TestCase):
    def test_builds_valid_archive_and_keeps_each_hdbscan_noise_row_separate(self):
        image_ids = [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
            "00000000-0000-0000-0000-000000000003",
        ]
        payloads = {value: f"image-{value}".encode() for value in image_ids}
        pipeline_digest = hashlib.sha256(b"pipeline").hexdigest()
        lease = TaskLease(
            job_id="job-1",
            lease_token="secret",
            task_type="batch_archival",
            required_model_version="reid-r4",
            input_manifest={
                "schema_version": 1,
                "batch_id": "10000000-0000-0000-0000-000000000001",
                "model_version": "reid-r4",
                "detector_version": "det-v1",
                "preprocess_id": "crop-v1",
                "pipeline_config_digest": pipeline_digest,
                "min_cluster_size": 2,
                "images": [{
                    "image_id": image_id,
                    "original_relative_path": f"survey/{index}.jpg",
                    "sha256": hashlib.sha256(payloads[image_id]).hexdigest(),
                } for index, image_id in enumerate(image_ids)],
            },
        )
        api = _InputApi(payloads)

        def fake_detect(frame: pd.DataFrame, images_root: Path, out_dir: Path,
                        **kwargs) -> pd.DataFrame:
            del images_root, kwargs
            out_dir.mkdir(parents=True)
            rows = []
            for _, row in frame.iterrows():
                (out_dir / f"{row['image_id']}.jpg").write_bytes(b"crop")
                rows.append({
                    "image_id": row["image_id"],
                    "relative_path": row["relative_path"],
                    "x": 1, "y": 2, "w": 20, "h": 10,
                    "det_conf": 0.9, "fallback": False,
                })
            return pd.DataFrame(rows)

        vectors = np.asarray(
            [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)

        def fake_extract(*args, **kwargs):
            del args, kwargs
            return vectors, pd.DataFrame()

        handler = make_batch_archival_handler(
            api,
            ArchivalWorkerConfig(
                detector_weights=Path("det.pt"),
                reid_checkpoint=Path("reid.pt"),
            ),
            detect_fn=fake_detect,
            embedder_factory=lambda *args, **kwargs: object(),
            extract_fn=fake_extract,
            cluster_fn=lambda matrix, minimum: (
                np.asarray([0, 0, -1]), np.asarray([0.9, 0.8, 0.0])),
        )
        artifact = handler(lease)

        self.assertEqual(api.downloaded, image_ids)
        self.assertEqual(artifact.artifact_type, "batch_archival")
        self.assertEqual(artifact.model_version, "reid-r4")
        with zipfile.ZipFile(io.BytesIO(artifact.data)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            archived_vectors = np.load(
                io.BytesIO(archive.read("embeddings.npy")), allow_pickle=False)
            self.assertEqual(archived_vectors.shape, (3, 2))
            self.assertEqual(
                [cluster["label"] for cluster in manifest["clusters"]],
                ["cluster-0", "noise-000002"],
            )
            self.assertEqual(
                manifest["clusters"][1]["member_keys"], [image_ids[2]])
            self.assertIn(f"crops/{image_ids[0]}.jpg", archive.namelist())

    def test_download_hash_mismatch_stops_before_algorithms_run(self):
        image_id = "00000000-0000-0000-0000-000000000001"
        lease = TaskLease(
            job_id="job-2", lease_token="secret",
            task_type="batch_archival", required_model_version="reid-r4",
            input_manifest={
                "schema_version": 1,
                "batch_id": "10000000-0000-0000-0000-000000000001",
                "model_version": "reid-r4", "detector_version": "det-v1",
                "preprocess_id": "crop-v1",
                "pipeline_config_digest": "a" * 64,
                "min_cluster_size": 2,
                "images": [{
                    "image_id": image_id,
                    "original_relative_path": "x.jpg",
                    "sha256": "0" * 64,
                }],
            },
        )
        handler = make_batch_archival_handler(
            _InputApi({image_id: b"different"}),
            ArchivalWorkerConfig(Path("det.pt"), Path("reid.pt")),
            detect_fn=lambda *args, **kwargs: self.fail("不应运行检测"),
        )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            handler(lease)


if __name__ == "__main__":
    unittest.main()
