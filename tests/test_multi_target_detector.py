"""平台多目标检测为同一 Image 生成 N 个独立 Crop。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.detection.detector import detect_all_and_crop  # noqa: E402


class _FakeYolo:
    def __init__(self, weights: str):
        self.weights = weights

    def predict(self, *args, **kwargs):
        del args, kwargs
        boxes = [
            SimpleNamespace(
                xyxy=[np.asarray([10, 10, 30, 30], dtype=np.float32)],
                conf=[0.9]),
            SimpleNamespace(
                xyxy=[np.asarray([55, 20, 75, 45], dtype=np.float32)],
                conf=[0.8]),
            SimpleNamespace(
                xyxy=[np.asarray([30, 50, 50, 75], dtype=np.float32)],
                conf=[0.7]),
        ]
        return [SimpleNamespace(boxes=boxes)]


class TestMultiTargetDetector(unittest.TestCase):
    def test_one_image_with_three_boxes_writes_three_independent_crops(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (100, 100), "white").save(root / "input.jpg")
            image_id = "00000000-0000-0000-0000-000000000001"
            result = detect_all_and_crop(
                pd.DataFrame([{
                    "image_id": image_id,
                    "relative_path": "input.jpg",
                    "session_id": "batch",
                }]),
                root,
                root / "crops",
                root / "detector.pt",
                model_factory=_FakeYolo,
            )
            self.assertEqual(len(result), 3)
            self.assertEqual(result["crop_index"].tolist(), [0, 1, 2])
            self.assertEqual(result["image_id"].nunique(), 1)
            self.assertEqual(result["crop_key"].nunique(), 3)
            self.assertTrue(all(
                (root / "crops" / f"{key}.jpg").is_file()
                for key in result["crop_key"]))


if __name__ == "__main__":
    unittest.main()
