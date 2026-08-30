"""r4 版本化检索产物重建入口的清单语义与安全边界测试。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rebuild_r4_artifacts import _build_subset, prepare_sources  # noqa: E402


class TestRebuildR4Sources(unittest.TestCase):
    """同串必须先由完整 manifest 派生，身份只能来自确认库。"""

    def test_gallery_series_uses_loose_bridge_frame_from_full_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            manifest = tmp / "manifest.csv"
            pilot = tmp / "pilot.csv"
            pd.DataFrame({
                "image_id": ["first", "bridge", "last", "pool2"],
                "relative_path": ["first.jpg", "bridge.jpg", "last.jpg", "pool2.jpg"],
                "filename": [
                    "0001_20140417_SZi_01_RAY_0058.JPG",
                    "0002_20140417_SZi_01_RAY_0060.JPG",
                    "0003_20140417_SZi_01_RAY_0062.JPG",
                    "0004_20140417_OTHER_0100.JPG",
                ],
                "session_id": ["s1", "s1", "s1", "s2"],
                "label_status": ["labeled", "loose_known", "labeled", "loose_known"],
            }).to_csv(manifest, index=False)
            pd.DataFrame({
                "image_id": ["first", "last"],
                "confirmed_identity": ["NA", "NA"],
            }).to_csv(pilot, index=False)

            gallery, pool = prepare_sources(manifest, pilot, ["s1"])

        self.assertEqual(gallery["confirmed_identity"].tolist(), ["NA", "NA"])
        self.assertEqual(gallery["series_id"].nunique(), 1)
        self.assertEqual(set(pool["image_id"]), {"bridge", "pool2"})
        self.assertEqual(gallery["relative_path"].tolist(), [
            "s1/first.jpg", "s1/last.jpg"])

    def test_pilot_must_exactly_cover_labeled_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            manifest = tmp / "manifest.csv"
            pilot = tmp / "pilot.csv"
            pd.DataFrame({
                "image_id": ["known"],
                "relative_path": ["known.jpg"],
                "filename": ["known.jpg"],
                "session_id": ["s1"],
                "label_status": ["labeled"],
            }).to_csv(manifest, index=False)
            pd.DataFrame({
                "image_id": ["other"],
                "confirmed_identity": ["id"],
            }).to_csv(pilot, index=False)

            with self.assertRaisesRegex(ValueError, "labeled 图片不一致"):
                prepare_sources(manifest, pilot, ["s1"])

    def test_built_subset_config_binds_crop_jpeg_bytes(self):
        source = pd.DataFrame({
            "image_id": ["a", "b"],
            "relative_path": ["s/a.jpg", "s/b.jpg"],
            "session_id": ["s", "s"],
        })

        def fake_detect(rows, _images_root, crops_dir, *_args, **_kwargs):
            crops_dir.mkdir()
            rows.to_csv(crops_dir / "crops_manifest.csv", index=False)
            for image_id in rows["image_id"]:
                (crops_dir / f"{image_id}.jpg").write_bytes(
                    f"jpeg-{image_id}".encode("ascii"))
            return rows

        model = Mock(name="model")
        model.name = "mock-model"
        model.preprocess_id = "mock-preprocess"
        with tempfile.TemporaryDirectory() as tmp_dir, patch(
                "rebuild_r4_artifacts.detect_and_crop",
                side_effect=fake_detect), patch(
                "rebuild_r4_artifacts.extract_embeddings"), patch(
                "rebuild_r4_artifacts.yolo_crop_provenance",
                return_value={"crop": "yolo"}), patch(
                "rebuild_r4_artifacts.load_verified_embedding_artifact",
                return_value=(
                    np.zeros((2, 2), dtype=np.float32), source.copy(),
                    {"artifact_schema_version": 2,
                     "provenance_level": "generated_with_row_binding"})), patch(
                "rebuild_r4_artifacts.require_generated_artifact_provenance"):
            _, _, config = _build_subset(
                "pool", source, Path(tmp_dir),
                images_root=Path("images"),
                detector_weights=Path("detector.pt"),
                checkpoint=Path("model.pt"),
                detector_cfg={"conf": 0.25, "imgsz": 1024},
                crop_cfg={"pad_x": 0.3, "pad_up": 0.15, "pad_down": 0.6},
                model=model,
            )

        self.assertEqual(
            config["crop_bundle_digest_algorithm"],
            "sha256-v1:ordered-image-id+file-sha256")
        self.assertEqual(len(config["crop_bundle_sha256"]), 64)
        self.assertEqual(len(config["crop_manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
