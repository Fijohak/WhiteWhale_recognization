"""
接口层冒烟测试：验证 Retrieval + Evaluation 逻辑正确性。

不依赖图片与模型（构造已知 embedding 验证指标）。
运行：python -m unittest tests.test_reid_interfaces -v
"""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from pub_reid.dataset.base import DatasetAdapter, ReIDData  # noqa: E402
from whitewhale.data.manifest import compute_sha256  # noqa: E402
from whitewhale.reid.embedding import (  # noqa: E402
    DEVICE,
    extract_embeddings,
    load_verified_embedding_artifact,
    make_embedder,
    read_metadata_csv,
    require_compatible_embedding_configs,
    require_generated_artifact_provenance,
)
from whitewhale.reid.evaluation import mean_average_precision, recall_at_k, split_query_gallery  # noqa: E402
from whitewhale.reid.retrieval import cosine_topk  # noqa: E402
from whitewhale.reid.training import (  # noqa: E402
    DolphinDatasetHn,
    WithinSessionIdentitySampler,
    _isolate_test_sessions,
    _seed_training,
    _validated_init_state,
    load_confirmed,
    retrieval_r1_from_features,
    run_training,
    session_aware_cross_entropy,
    split_by_individual,
    train_one_epoch_hn,
    triplet_hn_loss,
)


def _yolo_compat_config() -> dict:
    """构造一套完整的可比 YOLO 特征配置。"""
    return {
        "model": "megadescriptor-metric-learning-r4",
        "crop": "yolo",
        "preprocess": "Resize256+CenterCrop224",
        "checkpoint_sha256": "model-hash",
        "crop_schema_version": 1,
        "detector_checkpoint_sha256": "detector-hash",
        "detector_conf": 0.25,
        "detector_imgsz": 1024,
        "detector_pad_x": 0.30,
        "detector_pad_up": 0.15,
        "detector_pad_down": 0.60,
        "detector_fallback_policy": "center_square_min_side_0.45",
    }


class TestEmbeddingArtifacts(unittest.TestCase):
    """特征产物必须记录可核对的文件与源码溯源信息。"""

    def test_extract_writes_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "session" / "sample.jpg"
            image_path.parent.mkdir()
            Image.new("RGB", (16, 16), (10, 20, 30)).save(image_path)
            manifest = pd.DataFrame({
                "image_id": ["sample"],
                "relative_path": ["session/sample.jpg"],
                "confirmed_identity": [""],
            })
            confirmed = pd.DataFrame({
                "image_id": ["sample"],
                "confirmed_identity": ["session_1"],
                "individual_id": ["session_1"],
            })
            out_path = root / "embeddings.npy"

            extract_embeddings(
                manifest, make_embedder("mock"), images_root=root,
                out_path=out_path, merge_from=confirmed,
            )

            meta_path = root / "embeddings_meta.csv"
            config = json.loads((root / "embeddings_config.json").read_text("utf-8"))
            meta = pd.read_csv(meta_path)
            self.assertEqual(meta.loc[0, "confirmed_identity"], "session_1")
            self.assertEqual(meta.loc[0, "individual_id"], "session_1")
            self.assertEqual(meta.loc[0, "session_id"], "session")
            self.assertEqual(config["artifact_schema_version"], 2)
            self.assertEqual(config["provenance_level"],
                             "generated_with_row_binding")
            self.assertIn("ordered_image_ids_sha256", config)
            self.assertEqual(config["embedding_sha256"], compute_sha256(out_path))
            self.assertEqual(config["meta_sha256"], compute_sha256(meta_path))
            self.assertEqual(config["embedding_file"], out_path.name)
            self.assertEqual(config["meta_file"], meta_path.name)
            self.assertIn("created_at_utc", config)
            self.assertIn("python", config["runtime"])

    def test_cross_checkpoint_artifacts_are_rejected(self):
        common = _yolo_compat_config()
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            require_compatible_embedding_configs(
                {**common, "checkpoint_sha256": "aaa"},
                {**common, "checkpoint_sha256": "bbb"},
            )

    def test_schema1_and_backfilled_provenance_are_rejected(self):
        common = {
            "created_at_utc": "2026-08-29T00:00:00+00:00",
            "row_binding": "embedding_row_i_to_meta_image_id_i",
            "ordered_image_ids_sha256": "row-hash",
        }
        invalid = (
            {
                **common,
                "artifact_schema_version": 1,
                "provenance_level": "generated_with_row_binding",
            },
            {
                **common,
                "artifact_schema_version": 2,
                "provenance_level": "legacy_backfilled_unverified_row_alignment",
                "provenance_backfilled": True,
            },
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                require_generated_artifact_provenance(config)

    def test_each_yolo_crop_field_must_match(self):
        left = _yolo_compat_config()
        mismatches = {
            "crop_schema_version": 2,
            "detector_checkpoint_sha256": "other-detector",
            "detector_conf": 0.30,
            "detector_imgsz": 640,
            "detector_pad_x": 0.20,
            "detector_pad_up": 0.10,
            "detector_pad_down": 0.50,
            "detector_fallback_policy": "whole_image",
        }
        for key, value in mismatches.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                require_compatible_embedding_configs(
                    left, {**left, key: value})

    def test_each_yolo_crop_field_is_required(self):
        left = _yolo_compat_config()
        fields = [
            "crop_schema_version", "detector_checkpoint_sha256",
            "detector_conf", "detector_imgsz", "detector_pad_x",
            "detector_pad_up", "detector_pad_down",
            "detector_fallback_policy",
        ]
        for key in fields:
            right = left.copy()
            right.pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                require_compatible_embedding_configs(left, right)

    def test_missing_preprocess_is_not_silently_accepted(self):
        left = {"model": "test", "crop": "whole", "preprocess": "v1"}
        right = {"model": "test", "crop": "whole"}
        with self.assertRaisesRegex(ValueError, "preprocess"):
            require_compatible_embedding_configs(left, right)

    def test_missing_feature_must_be_an_entire_nan_row(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            emb_path = root / "features.npy"
            meta_path = root / "features_meta.csv"
            config_path = root / "features_config.json"
            pd.DataFrame({"image_id": ["missing"]}).to_csv(meta_path, index=False)

            def write_artifact(values):
                np.save(emb_path, np.asarray(values, dtype=np.float32))
                config_path.write_text(json.dumps({
                    "embedding_file": emb_path.name,
                    "meta_file": meta_path.name,
                    "embedding_sha256": compute_sha256(emb_path),
                    "meta_sha256": compute_sha256(meta_path),
                    "n": 1,
                    "feat_dim": 2,
                }), encoding="utf-8")

            write_artifact([[np.nan, np.nan]])
            load_verified_embedding_artifact(
                emb_path, meta_path, allow_nonfinite=True)

            write_artifact([[np.nan, 1.0]])
            with self.assertRaisesRegex(ValueError, "部分损坏"):
                load_verified_embedding_artifact(
                    emb_path, meta_path, allow_nonfinite=True)

    def test_loader_preserves_opaque_identity_strings(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            emb_path = root / "features.npy"
            meta_path = root / "features_meta.csv"
            np.save(emb_path, np.eye(4, dtype=np.float32))
            pd.DataFrame({
                "image_id": ["image_1", "image_2", "image_3", "image_4"],
                "confirmed_identity": ["0005", "NA", "N/A", "null"],
            }).to_csv(meta_path, index=False)
            (root / "features_config.json").write_text(json.dumps({
                "embedding_file": emb_path.name,
                "meta_file": meta_path.name,
                "embedding_sha256": compute_sha256(emb_path),
                "meta_sha256": compute_sha256(meta_path),
                "n": 4,
                "feat_dim": 4,
            }), encoding="utf-8")

            _, meta, _ = load_verified_embedding_artifact(emb_path, meta_path)

            self.assertEqual(
                meta["confirmed_identity"].tolist(),
                ["0005", "NA", "N/A", "null"],
            )
            self.assertEqual(
                read_metadata_csv(meta_path)["confirmed_identity"].tolist(),
                ["0005", "NA", "N/A", "null"],
            )

    def test_artifact_loader_rejects_blank_string_image_id(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            emb_path = root / "features.npy"
            meta_path = root / "features_meta.csv"
            np.save(emb_path, np.asarray([[1.0, 0.0]], dtype=np.float32))
            pd.DataFrame({"image_id": [""]}).to_csv(meta_path, index=False)
            (root / "features_config.json").write_text(json.dumps({
                "embedding_file": emb_path.name,
                "meta_file": meta_path.name,
                "embedding_sha256": compute_sha256(emb_path),
                "meta_sha256": compute_sha256(meta_path),
                "n": 1,
                "feat_dim": 2,
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "image_id 为空或重复"):
                load_verified_embedding_artifact(emb_path, meta_path)

    def test_missing_crop_with_existing_meta_columns_returns_nan(self):
        """merge 填充掩码不得覆盖 missing 参数并让缺图分支崩溃。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest = pd.DataFrame({
                "image_id": ["missing_image"],
                "session_id": [""],
                "relative_path": ["session/missing.jpg"],
            })
            merge = pd.DataFrame({
                "image_id": ["missing_image"],
                "session_id": ["session"],
                "confirmed_identity": ["0005"],
            })

            embeddings, meta = extract_embeddings(
                manifest, make_embedder("mock"), crops_dir=root / "crops",
                merge_from=merge, missing="nan")

            self.assertTrue(np.isnan(embeddings).all())
            self.assertEqual(meta.loc[0, "session_id"], "session")


class TestHardNegativeTraining(unittest.TestCase):
    """HN 必须使用批内确认负样本，并选择最远正样本/最近负样本。"""

    @staticmethod
    def _membership(labels, series):
        membership = torch.zeros(
            max(series) + 1, max(labels) + 1, dtype=torch.bool)
        for label, series_id in zip(labels, series):
            membership[series_id, label] = True
        return membership

    def test_sampler_accepts_full_session_strings(self):
        labels = [0, 0, 1, 1, 2, 2, 3, 3]
        sessions = ["20140806 01"] * 4 + ["20140419 02"] * 4
        series = ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2"]
        sampler = WithinSessionIdentitySampler(
            labels, sessions, series, batch_size=4, batches_per_epoch=4, seed=7)
        for batch in sampler:
            indices = batch.tolist()
            batch_sessions = {sessions[index] for index in indices}
            batch_labels = [labels[index] for index in indices]
            self.assertEqual(len(batch_sessions), 1)
            self.assertEqual(len(set(batch_labels)), 2)
            self.assertTrue(all(batch_labels.count(label) >= 2
                                for label in set(batch_labels)))
            for label in set(batch_labels):
                label_series = {series[index] for index in indices
                                if labels[index] == label}
                self.assertGreaterEqual(len(label_series), 2)

    def test_sampler_mixes_single_and_multi_series_without_zero_anchor(self):
        """单串身份可参与 CE，但每批必须带入可跨串的身份。"""
        labels = [0, 0, 1, 1, 2, 2]
        sessions = ["session"] * 6
        series = [0, 1, 0, 0, 1, 1]
        sampler = WithinSessionIdentitySampler(
            labels, sessions, series, batch_size=4, batches_per_epoch=20, seed=42)

        for batch in sampler:
            indices = batch.tolist()
            batch_labels = torch.tensor([labels[index] for index in indices])
            batch_sessions = torch.tensor([0] * len(indices))
            batch_series = torch.tensor([series[index] for index in indices])
            feats = torch.eye(len(indices), dtype=torch.float32)
            _, valid_anchors = triplet_hn_loss(
                feats, batch_labels, batch_sessions, batch_series,
                self._membership(labels, series),
                return_details=True)

            self.assertIn(0, batch_labels.tolist())
            self.assertGreater(valid_anchors, 0)

    def test_sampler_rejects_all_single_series_identities(self):
        with self.assertRaisesRegex(ValueError, "跨串正样本"):
            WithinSessionIdentitySampler(
                labels=[0, 0, 1, 1], sessions=[0, 0, 0, 0],
                series=[0, 0, 1, 1], batch_size=4,
                batches_per_epoch=1, seed=7)

    def test_sampler_uses_global_series_membership_for_valid_anchors(self):
        """两个身份在所有串共现时，sampler 必须带入第三个安全负类。"""
        labels = [0, 0, 1, 1, 2, 2]
        sessions = [0] * 6
        series = [0, 1, 0, 1, 2, 2]
        membership = self._membership(labels, series)
        sampler = WithinSessionIdentitySampler(
            labels, sessions, series, batch_size=4,
            batches_per_epoch=20, seed=11)

        for batch in sampler:
            indices = batch.tolist()
            batch_labels = torch.tensor([labels[index] for index in indices])
            batch_series = torch.tensor([series[index] for index in indices])
            _, valid_anchors = triplet_hn_loss(
                torch.eye(len(indices)), batch_labels,
                torch.zeros(len(indices), dtype=torch.long), batch_series,
                membership, return_details=True)
            self.assertIn(2, batch_labels.tolist())
            self.assertGreater(valid_anchors, 0)

    def test_triplet_uses_batch_hard_extremes(self):
        feats = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
            [-1.0, 0.0],
        ])
        labels = torch.tensor([0, 0, 1, 1])
        sessions = torch.tensor([0, 0, 0, 0])
        series = torch.tensor([0, 1, 2, 3])

        loss = triplet_hn_loss(
            feats, labels, sessions, series,
            self._membership(labels.tolist(), series.tolist()), margin=0.3)

        self.assertAlmostEqual(float(loss), 1.25, places=5)

    def test_triplet_excludes_same_series_positives_and_negatives(self):
        feats = torch.tensor([
            [1.0, 0.0], [0.0, 1.0],
            [1.0, 0.0], [1.0, 0.0],
        ])
        labels = torch.tensor([0, 0, 1, 1])
        sessions = torch.tensor([0, 0, 0, 0])
        series = torch.tensor([0, 1, 0, 0])

        loss = triplet_hn_loss(
            feats, labels, sessions, series,
            self._membership(labels.tolist(), series.tolist()), margin=0.3)

        # 仅 index=1 同时具有跨串正样本和跨串负样本；index=0 的近负样本
        # 都与其同串而被剔除，label=1 也没有跨串正样本。
        self.assertAlmostEqual(float(loss), 0.3, places=5)

    def test_triplet_excludes_other_series_photo_of_cooccurring_identity(self):
        """负类曾在 anchor 串共现时，其另一串照片也不能绕过剔除约束。"""
        feats = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [1.0, 0.0],
        ])
        labels = torch.tensor([0, 0, 1])
        sessions = torch.zeros(3, dtype=torch.long)
        series = torch.tensor([0, 1, 2])
        # label=1 在完整数据的串 0 中出现过，但该照片不必恰好进入当前 batch。
        membership = torch.tensor([
            [True, True],
            [True, False],
            [False, True],
        ])

        loss, valid_anchors = triplet_hn_loss(
            feats, labels, sessions, series, membership,
            margin=0.3, return_details=True)

        # index=0 的 anchor 串 0 出现过 label=1，因此 label=1 在串 2 的照片
        # 同样被排除；只剩 index=1 可使用 label=1 作为负类。
        self.assertEqual(valid_anchors, 1)
        self.assertAlmostEqual(float(loss), 0.3, places=5)

    def test_hn_preload_keeps_aspect_ratio_and_short_edge(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            horizontal = root / "horizontal.jpg"
            vertical = root / "vertical.jpg"
            Image.new("RGB", (800, 400), (1, 2, 3)).save(horizontal)
            Image.new("RGB", (400, 800), (4, 5, 6)).save(vertical)
            frame = pd.DataFrame({
                "path": [str(horizontal), str(vertical)],
                "label_idx": [0, 1],
                "sess": [0, 0],
                "series_idx": [0, 1],
            })
            dataset = DolphinDatasetHn(
                frame, transform=lambda image: image.size, cache_size=256)

            dataset.preload()

            self.assertEqual(dataset._cache[0].size, (512, 256))
            self.assertEqual(dataset._cache[1].size, (256, 512))
            self.assertEqual(dataset[0][0], (512, 256))

    def test_arcface_ce_ignores_cross_session_classes(self):
        logits = torch.tensor([
            [2.0, 0.0, 100.0, 100.0],
            [100.0, 100.0, 2.0, 0.0],
        ])
        labels = torch.tensor([0, 2])
        sessions = torch.tensor([0, 1])
        series = torch.tensor([0, 2])
        class_sessions = torch.tensor([0, 0, 1, 1])
        class_series_membership = torch.eye(4, dtype=torch.bool)

        loss = session_aware_cross_entropy(
            logits, labels, sessions, class_sessions, series,
            class_series_membership)

        expected = torch.nn.functional.cross_entropy(
            torch.tensor([[2.0, 0.0], [2.0, 0.0]]), torch.tensor([0, 0]))
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_arcface_ce_ignores_other_identity_prototype_from_same_series(self):
        logits = torch.tensor([[2.0, 100.0, 0.0]])
        labels = torch.tensor([0])
        sessions = torch.tensor([0])
        series = torch.tensor([0])
        class_sessions = torch.tensor([0, 0, 0])
        class_series_membership = torch.tensor([
            [True, True, False],
            [False, False, True],
        ])

        loss = session_aware_cross_entropy(
            logits, labels, sessions, class_sessions, series,
            class_series_membership)

        expected = torch.nn.functional.cross_entropy(
            torch.tensor([[2.0, 0.0]]), torch.tensor([0]))
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_init_checkpoint_must_cover_complete_backbone(self):
        current = {
            "backbone.a": torch.zeros(2, 2),
            "backbone.b": torch.zeros(2),
            "head.W": torch.zeros(2, 3),
        }
        with self.assertRaisesRegex(ValueError, "完整覆盖"):
            _validated_init_state({"state": {"backbone.a": torch.zeros(2, 2)}},
                                  current)
        with self.assertRaisesRegex(ValueError, "state"):
            _validated_init_state({}, current)

        state, matched = _validated_init_state({"state": {
            "backbone.a": torch.ones(2, 2),
            "backbone.b": torch.ones(2),
            "head.W": torch.ones(2, 4),  # 类别数不同，允许跳过 head
        }}, current)
        self.assertEqual(matched, 2)
        self.assertEqual(set(state), {"backbone.a", "backbone.b"})

    def test_training_seed_resets_python_numpy_and_torch(self):
        _seed_training(19)
        first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        _seed_training(19)
        second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

        self.assertEqual(first, second)

    def test_test_sessions_are_fully_isolated_before_train_val_split(self):
        frame = pd.DataFrame({
            "image_id": ["a", "b", "c", "d"],
            "session_id": ["train", "test", "test", "train"],
            "series_unit": ["a1", "b1", "c1", "d1"],
            "confirmed_identity": ["0", "1", "2", "3"],
        })
        remaining, test = _isolate_test_sessions(frame, ["test"])

        self.assertEqual(remaining["image_id"].tolist(), ["a", "d"])
        self.assertEqual(test["image_id"].tolist(), ["b", "c"])
        self.assertEqual(test["session_id"].unique().tolist(), ["test"])

    def test_training_checkpoint_selection_and_epoch_progress(self):
        """stage1 不得冒充检索提升；逐轮状态必须足够用于中断诊断。"""
        from whitewhale.reid import training as training_module

        class ToyModel(torch.nn.Module):
            def __init__(self, backbone, n_classes):
                super().__init__()
                self.backbone = backbone
                self.head = torch.nn.Linear(1, n_classes, bias=False)

        train = pd.DataFrame({
            "image_id": ["a1", "a2", "b1", "b2"],
            "path": ["a1.jpg", "a2.jpg", "b1.jpg", "b2.jpg"],
            "confirmed_identity": ["a", "a", "b", "b"],
            "session_id": ["s1"] * 4,
            "series_unit": ["a1", "a2", "b1", "b2"],
            "label_idx": [0, 0, 1, 1],
        })
        val = pd.DataFrame({
            "image_id": ["c1", "c2", "d1", "d2"],
            "path": ["c1.jpg", "c2.jpg", "d1.jpg", "d2.jpg"],
            "confirmed_identity": ["c", "c", "d", "d"],
            "session_id": ["s1"] * 4,
            "series_id": ["c1", "c2", "d1", "d2"],
            "series_unit": ["c1", "c2", "d1", "d2"],
            "label_idx": [2, 2, 3, 3],
        })
        args = SimpleNamespace(
            val_n=2, seed=7, epochs_stage1=2, epochs_stage2=2,
            lr_head=1e-3, lr_backbone=1e-5, batch=4,
            hard_negative=False, init_ckpt=None, batches_per_epoch=3,
            lambda_hn=0.2, test_session=[],
        )
        source = pd.DataFrame()

        real_save_checkpoint = training_module._save_checkpoint
        checkpoint_calls = []

        def capture_checkpoint(payload, path):
            checkpoint_calls.append(
                (path.name, payload["stage"], payload["epoch"]))
            real_save_checkpoint(payload, path)

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(training_module, "split_by_individual",
                             return_value=(train, val)), \
                patch.object(training_module, "make_transforms",
                             return_value=None), \
                patch.object(training_module, "DataLoader",
                             side_effect=lambda *_args, **_kwargs: object()), \
                patch.object(training_module, "make_backbone",
                             return_value=torch.nn.Linear(1, 1, bias=False)), \
                patch.object(training_module, "ReIDModel", ToyModel), \
                patch.object(training_module, "train_one_epoch",
                             side_effect=[(1.0, 0.0)] * 4), \
                patch.object(training_module, "eval_retrieval",
                             side_effect=[(0.5, 4, 0), 0.5, 0.6]), \
                patch.object(training_module, "_save_checkpoint",
                             side_effect=capture_checkpoint):
            out_dir = Path(tmp_dir)
            run_training(args, source, out_dir)

            best = torch.load(out_dir / "best.pt", map_location="cpu")
            stage1 = torch.load(out_dir / "best_stage1.pt", map_location="cpu")
            last = torch.load(out_dir / "last.pt", map_location="cpu")
            history = pd.read_csv(out_dir / "history.csv")
            metrics = json.loads(
                (out_dir / "metrics.json").read_text(encoding="utf-8"))

            self.assertEqual(
                [call for call in checkpoint_calls if call[0] == "best.pt"],
                [("best.pt", 0, 0), ("best.pt", 2, 2)],
            )
            self.assertEqual((best["stage"], best["epoch"]), (2, 2))
            self.assertEqual(best["selection"], "strict_val_improvement")
            self.assertEqual((stage1["stage"], stage1["epoch"]), (1, 2))
            self.assertEqual(stage1["selection"], "final_epoch_unconditional")
            self.assertEqual((last["stage"], last["epoch"]), (2, 2))
            self.assertIn("state", last)
            self.assertIn("optimizer", last)
            self.assertEqual(last["history_rows"], 4)
            self.assertEqual(history["stage"].tolist(), [1, 1, 2, 2])
            self.assertEqual(history.loc[history["stage"] == 1, "val_r1"].tolist(),
                             [0.5, 0.5])
            self.assertEqual(metrics["best_stage"], 2)
            self.assertEqual(metrics["best_epoch"], 2)
            self.assertEqual(
                metrics["stage1_selection"]["policy"],
                "final_epoch_unconditional",
            )
            self.assertFalse(metrics["stage1_selection"]["eligible_for_best_pt"])
            self.assertEqual(
                metrics["triplet_valid_anchor_observability"][
                    "primary_total_scope"],
                "stage2_lambda_hn_only")
            self.assertIsNone(metrics["test_evaluation"])
            self.assertFalse((out_dir / "history.csv.tmp").exists())
            self.assertFalse((out_dir / "last.pt.tmp").exists())

    def test_test_session_is_evaluated_without_affecting_selection(self):
        from whitewhale.reid import training as training_module

        class ToyModel(torch.nn.Module):
            def __init__(self, backbone, n_classes):
                super().__init__()
                self.backbone = backbone
                self.head = torch.nn.Linear(1, n_classes, bias=False)

        def make_rows(prefix, identities, session):
            count = len(identities)
            return pd.DataFrame({
                "image_id": [f"{prefix}{index}" for index in range(count)],
                "path": [f"{prefix}{index}.jpg" for index in range(count)],
                "confirmed_identity": identities,
                "session_id": [session] * count,
                "series_id": [f"{prefix}{index}" for index in range(count)],
                "series_unit": [f"{prefix}{index}" for index in range(count)],
                "label_idx": list(range(count)),
            })

        train = make_rows("tr", ["a", "a", "b", "b"], "train")
        val = make_rows("va", ["c", "c", "d", "d"], "train")
        candidates = pd.concat([train, val], ignore_index=True)
        test = make_rows("te", ["e", "e", "f", "f"], "held-out")
        source = pd.concat([candidates, test], ignore_index=True)
        args = SimpleNamespace(
            val_n=2, seed=7, epochs_stage1=0, epochs_stage2=1,
            lr_head=1e-3, lr_backbone=1e-5, batch=4,
            hard_negative=False, init_ckpt=None, batches_per_epoch=3,
            lambda_hn=0.2, test_session=["held-out"],
        )

        def split_without_test(frame, *_args):
            self.assertNotIn("held-out", frame["session_id"].tolist())
            return train, val

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(training_module, "split_by_individual",
                             side_effect=split_without_test), \
                patch.object(training_module, "make_transforms", return_value=None), \
                patch.object(training_module, "DataLoader",
                             side_effect=lambda *_args, **_kwargs: object()), \
                patch.object(training_module, "make_backbone",
                             return_value=torch.nn.Linear(1, 1, bias=False)), \
                patch.object(training_module, "ReIDModel", ToyModel), \
                patch.object(training_module, "train_one_epoch",
                             return_value=(1.0, 0.0)), \
                patch.object(training_module, "eval_retrieval", side_effect=[
                    (0.5, 4, 0),  # val pretrained baseline
                    (0.4, 2, 2),  # test pretrained baseline（不参与选择）
                    0.6,          # stage2 val，决定 best.pt
                    (0.7, 3, 1),  # 训练完成后 best.pt 的唯一 test 评估
                ]) as evaluate:
            out_dir = Path(tmp_dir)
            run_training(args, source, out_dir)
            metrics = json.loads(
                (out_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(evaluate.call_count, 4)
        self.assertEqual(metrics["best_val_r1"], 0.6)
        self.assertTrue(metrics["test_evaluation"]["test_not_used_for_selection"])
        self.assertEqual(metrics["test_evaluation"]["sessions"], ["held-out"])
        self.assertEqual(metrics["test_evaluation"]["n_rows"], 4)
        self.assertEqual(metrics["test_evaluation"]["n_identity_units"], 2)
        self.assertEqual(metrics["test_evaluation"]["pretrained_baseline_r1"], 0.4)
        self.assertEqual(metrics["test_evaluation"]["best_checkpoint_r1"], 0.7)
        self.assertEqual(metrics["test_evaluation"]["best_evaluable_queries"], 3)
        self.assertEqual(metrics["test_evaluation"]["best_skipped_queries"], 1)

    def test_stage1_frozen_backbone_remains_in_eval_mode(self):
        """冻结参数之外还要冻结 BatchNorm/Dropout 训练态，保证 val 不漂移。"""
        class RecordingBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.forward_training_modes = []

            def forward(self, inputs):
                self.forward_training_modes.append(self.training)
                return self.linear(inputs)

        class ToyHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, features, _labels):
                return self.linear(features)

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = RecordingBackbone()
                self.head = ToyHead()

            def encode(self, inputs):
                return torch.nn.functional.normalize(self.backbone(inputs), dim=1)

        model = ToyModel().to(DEVICE)
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        optimizer = torch.optim.SGD(model.head.parameters(), lr=0.01)
        batch = (
            torch.tensor([[1.0, 0.0], [0.9, 0.1],
                          [0.0, 1.0], [0.1, 0.9]]),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([0, 1, 2, 3]),
        )

        train_one_epoch_hn(
            model, [batch], optimizer, lambda_hn=0.0,
            class_sessions=torch.tensor([0, 0], device=DEVICE),
            class_series_membership=torch.tensor([
                [True, False], [True, False],
                [False, True], [False, True],
            ], device=DEVICE),
        )

        self.assertEqual(model.backbone.forward_training_modes, [False])

    def test_training_loader_rejects_subset_series_reannotation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pilot = root / "pilot.csv"
            pd.DataFrame({
                "image_id": ["i1"],
                "relative_path": ["i1.jpg"],
                "session_id": ["s1"],
                "confirmed_identity": ["id1"],
            }).to_csv(pilot, index=False)

            with self.assertRaisesRegex(ValueError, "完整 dataset manifest"):
                load_confirmed(pilot, root)

    def test_ce_and_triplet_share_one_backbone_forward(self):
        """每个 HN 批次只能跑一次 backbone，CE/triplet 共用同一特征。"""
        class CountingBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.calls = 0

            def forward(self, inputs):
                self.calls += 1
                return self.linear(inputs)

        class ToyHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, features, _labels):
                return self.linear(features)

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = CountingBackbone()
                self.head = ToyHead()

            def encode(self, inputs):
                return torch.nn.functional.normalize(self.backbone(inputs), dim=1)

        model = ToyModel().to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = [
            (torch.tensor([1.0, 0.0]), torch.tensor(0), torch.tensor(0), torch.tensor(0)),
            (torch.tensor([0.9, 0.1]), torch.tensor(0), torch.tensor(0), torch.tensor(1)),
            (torch.tensor([0.0, 1.0]), torch.tensor(1), torch.tensor(0), torch.tensor(2)),
            (torch.tensor([0.1, 0.9]), torch.tensor(1), torch.tensor(0), torch.tensor(3)),
        ]

        _, _, stats = train_one_epoch_hn(
            model, [tuple(torch.stack(items) for items in zip(*batch))],
            optimizer, lambda_hn=0.2,
            class_sessions=torch.tensor([0, 0], device=DEVICE),
            class_series_membership=torch.tensor([
                [True, False], [True, False],
                [False, True], [False, True],
            ], device=DEVICE),
            return_stats=True)

        self.assertEqual(model.backbone.calls, 1)
        self.assertEqual(stats["valid_anchors_per_batch"], [4])
        self.assertEqual(stats["zero_valid_batches"], 0)

    def test_hn_epoch_rejects_zero_valid_anchor_batch(self):
        """即使绕过 sampler，训练也不得静默接受零合法 anchor。"""
        class ToyHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, features, _labels):
                return self.linear(features)

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(2, 2)
                self.head = ToyHead()

            def encode(self, inputs):
                return torch.nn.functional.normalize(self.backbone(inputs), dim=1)

        model = ToyModel().to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = (
            torch.eye(4, 2),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 0, 0, 0]),
            torch.tensor([0, 0, 1, 1]),
        )

        with self.assertRaisesRegex(RuntimeError, "合法 anchor"):
            train_one_epoch_hn(
                model, [batch], optimizer, lambda_hn=0.2,
                class_sessions=torch.tensor([0, 0], device=DEVICE),
                class_series_membership=torch.tensor([
                    [True, False], [False, True],
                ], device=DEVICE))

    def test_validation_holdout_is_stable_across_id_display_migration(self):
        frame = pd.DataFrame({
            "image_id": [f"img{i}" for i in range(8)],
            "confirmed_identity": ["old_a"] * 4 + ["old_b"] * 4,
            "series_unit": ["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2"],
        })
        renamed = frame.copy()
        renamed["confirmed_identity"] = ["new_01"] * 4 + ["new_02"] * 4

        _, val_old = split_by_individual(frame, val_n=1, seed=5)
        _, val_new = split_by_individual(renamed, val_n=1, seed=5)

        self.assertEqual(set(val_old["image_id"]), set(val_new["image_id"]))

    def test_training_split_purges_series_shared_with_validation(self):
        frame = pd.DataFrame({
            "image_id": [f"img{i}" for i in range(8)],
            "session_id": ["s1"] * 8,
            "confirmed_identity": ["a"] * 4 + ["b"] * 4,
            "series_unit": [
                "shared", "shared", "a2", "a2",
                "shared", "shared", "b2", "b2",
            ],
        })

        train, val = split_by_individual(frame, val_n=1, seed=7)

        self.assertFalse(set(train["series_unit"]) & set(val["series_unit"]))
        self.assertEqual(train.attrs["purged_shared_series_rows"], 2)

    def test_validation_does_not_use_cross_session_unverified_negatives(self):
        feats = np.asarray([
            [1.0, 0.0], [0.9, 0.1],
            [0.0, 1.0], [0.1, 0.9],
            [1.0, 0.0], [0.9, -0.1],
            [-1.0, 0.0], [-0.9, 0.1],
        ], dtype=np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
        labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])

        score = retrieval_r1_from_features(
            feats,
            labels,
            series_ids=[f"series_{i}" for i in range(8)],
            session_ids=["s1"] * 4 + ["s2"] * 4,
        )

        self.assertEqual(score, 1.0)

    def test_validation_skips_queries_without_within_session_negative(self):
        feats = np.asarray([
            [1.0, 0.0], [0.9, 0.1],
            [0.0, 1.0], [0.1, 0.9],
        ], dtype=np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)

        with self.assertRaisesRegex(ValueError, "异体负样本"):
            retrieval_r1_from_features(
                feats,
                np.asarray([0, 0, 1, 1]),
                series_ids=["a1", "a2", "b1", "b2"],
                session_ids=["s1", "s1", "s2", "s2"],
            )


class TestDatasetAdapter(unittest.TestCase):
    """DatasetAdapter 统一 schema 与 namespace 约定。"""

    def test_normalize_fills_columns(self):
        class FakeAdapter(DatasetAdapter):
            name = "fake"
            has_identity = True

            def load(self, data_root=None):  # pragma: no cover
                df = pd.DataFrame({"image_path": ["a.jpg"], "identity": ["fake__x"]})
                return self._normalize(df)

        data = FakeAdapter().load()
        self.assertEqual(data.n_images, 1)
        self.assertEqual(data.n_identities, 1)
        # 缺失列补 None
        for c in ["species", "source_dataset", "encounter_id",
                  "date", "viewpoint", "split"]:
            self.assertTrue((data.df[c].isna()).all())

    def test_identity_namespace(self):
        """identity 必须 namespace（跨源 ID 防冲突）。"""
        class FakeAdapter(DatasetAdapter):
            name = "fake"
            has_identity = True

            def load(self, data_root=None):  # pragma: no cover
                df = pd.DataFrame({"identity": ["fake__1", "fake__2"]})
                return self._normalize(df)

        data = FakeAdapter().load()
        self.assertEqual(data.n_identities, 2)
        self.assertTrue(all(i.startswith("fake__") for i in data.df["identity"]))


class TestCosineTopk(unittest.TestCase):
    def test_basic(self):
        """已知 3 个点，q0 最近邻应为 g0（同身份）。"""
        q = np.array([[1.0, 2.0]])
        g = np.array([[1.0, 2.0], [2.0, 1.0], [5.0, 5.0]])
        scores, idx = cosine_topk(q, g, k=3)
        self.assertEqual(idx[0][0], 0)
        self.assertTrue(scores[0][0] > scores[0][1] > scores[0][2])
        self.assertAlmostEqual(scores[0][0], 1.0, places=5)

    def test_k_limit(self):
        q = np.ones((2, 4))
        g = np.ones((3, 4))
        scores, idx = cosine_topk(q, g, k=10)  # k > gallery 数量
        self.assertEqual(scores.shape, (2, 3))
        self.assertEqual(idx.shape, (2, 3))


class TestMetrics(unittest.TestCase):
    def test_recall_at_k(self):
        g_ids = np.array(["g_0", "g_1", "g_2", "g_3", "g_4"], dtype=object)
        scores = np.array([
            [0.9, 0.5, 0.4, 0.3, 0.2],
            [0.4, 0.3, 0.2, 0.1, 0.0],
            [0.6, 0.5, 0.4, 0.3, 0.2],
        ])
        idx = np.array([
            [0, 1, 2, 3, 4],
            [2, 3, 4, 0, 1],   # g_1 在第 5 位
            [1, 2, 3, 4, 0],   # 候选全为 g_1/g_2/g_3/g_4/g_1，无 g_4
        ])
        q_ids = np.array(["g_0", "g_1", "g_5"], dtype=object)  # g_5 不在 gallery
        r = recall_at_k(scores, idx, q_ids, g_ids, k_list=(1, 5))
        self.assertAlmostEqual(r[1], 1 / 3)   # 仅 q0 命中
        self.assertAlmostEqual(r[5], 2 / 3)   # q0 + q1

    def test_map(self):
        """q0 候选 [g0, g1, g0] → AP = (1/1 + 2/3)/2"""
        scores = np.array([[1.0, 0.5, 0.4]])
        idx = np.array([[0, 1, 0]])
        q_ids = np.array(["g_0"], dtype=object)
        g_ids = np.array(["g_0", "g_1", "g_0"], dtype=object)
        ap = mean_average_precision(scores, idx, q_ids, g_ids)
        self.assertAlmostEqual(ap, (1 / 1 + 2 / 3) / 2)

    def test_gt_sets_ignores_same_identity(self):
        """官方匹配对模式：同身份的其他图不算正样本（防自匹配虚高）。"""
        scores = np.array([[1.0, 0.5, 0.4], [0.9, 0.8, 0.7]])
        idx = np.array([[0, 1, 2], [0, 1, 2]])
        # q0 与 g0/g1 同身份，但官方正样本只有 g0
        # q1 官方正样本只有 g2（g1 同身份但不是正样本）
        gt_sets = [{0}, {2}]
        r = recall_at_k(scores, idx, None, None, k_list=(1, 2, 3), gt_sets=gt_sets)
        self.assertEqual(r[1], 0.5)   # 仅 q0 命中（q1 的 g2 在第 3 位）
        self.assertEqual(r[3], 1.0)
        ap = mean_average_precision(scores, idx, None, None, gt_sets=gt_sets)
        # q0: AP=1/1；q1: AP=1/3 → mean=(1+1/3)/2
        self.assertAlmostEqual(ap, (1.0 + 1 / 3) / 2)

    def test_gt_sets_unknown_db_id_ignored(self):
        """正样本指向 gallery 外（db_index 缺失）时安全忽略。"""
        scores = np.array([[0.9, 0.8]])
        idx = np.array([[0, 1]])
        # gt 指向索引 5（gallery 只有 2 个）
        r = recall_at_k(scores, idx, None, None, k_list=(1, 2), gt_sets=[{5}])
        self.assertEqual(r[1], 0.0)
        self.assertEqual(r[2], 0.0)
        ap = mean_average_precision(scores, idx, None, None, gt_sets=[{5}])
        self.assertEqual(ap, 0.0)


class TestSplitQueryGallery(unittest.TestCase):
    """query/gallery 划分防泄漏（同 identity 不共用同图）。"""

    def test_split_never_share_image(self):
        df = pd.DataFrame({"identity": ["a"] * 4 + ["b"] * 2 + ["c"] * 1})
        df["image_path"] = [f"img{i}.jpg" for i in range(len(df))]
        q, g = split_query_gallery(df, identity_col="identity")
        # 每身份至少 2 张才出 query；c 只有 1 张 → 只进 gallery（整图入库）
        self.assertEqual(sorted(q["identity"].unique()), ["a", "b"])
        # 同身份 query 与 gallery 不共用同图（q/g 索引不同，按位置比较）
        for iid in ["a", "b"]:
            qi = set(q[q["identity"] == iid].index)
            gi = set(g[g["identity"].to_numpy() == iid].index)
            self.assertEqual(len(qi), 1)
            self.assertTrue(qi.isdisjoint(gi))
        self.assertEqual(len(g), 5)  # a:3 + b:1 + c:1 进 gallery

    def test_complete_series_never_crosses_split(self):
        df = pd.DataFrame({
            "identity": ["a"] * 5 + ["b"] * 2,
            "series": ["a1", "a1", "a2", "a2", "a3", "b1", "b1"],
        })
        q, g = split_query_gallery(
            df, identity_col="identity", series_col="series", seed=7)

        self.assertEqual(set(q["identity"]), {"a"})
        self.assertTrue(set(q["series"]).isdisjoint(set(g["series"])))
        self.assertTrue((g[g["identity"] == "b"]["series"] == "b1").all())

    def test_shared_multi_identity_series_is_a_global_split_unit(self):
        """真实口径：一个拍摄串含多个身份时仍不得跨 q/g。"""
        df = pd.DataFrame({
            "image_id": ["a_shared", "a_other", "b_shared", "b_other"],
            "identity": ["a", "a", "b", "b"],
            "series": ["shared", "a2", "shared", "b2"],
        })

        q, g = split_query_gallery(
            df, identity_col="identity", series_col="series", seed=42)

        self.assertFalse(q.empty)
        self.assertTrue(set(q["series"]).isdisjoint(set(g["series"])))
        self.assertTrue(set(q["identity"]).issubset(set(g["identity"])))
        for series in df["series"].unique():
            in_query = set(df[df["series"] == series].index) & set(q.index)
            in_gallery = set(df[df["series"] == series].index) & set(g.index)
            self.assertFalse(in_query and in_gallery)

    def test_identity_display_rename_does_not_change_split(self):
        df = pd.DataFrame({
            "image_id": [f"img{i}" for i in range(8)],
            "identity": ["legacy_a"] * 4 + ["legacy_b"] * 4,
            "series": ["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2"],
        })
        renamed = df.copy()
        renamed["identity"] = ["canonical_01"] * 4 + ["canonical_02"] * 4

        q_old, g_old = split_query_gallery(
            df, identity_col="identity", series_col="series", seed=11)
        q_new, g_new = split_query_gallery(
            renamed, identity_col="identity", series_col="series", seed=11)

        self.assertEqual(set(q_old["image_id"]), set(q_new["image_id"]))
        self.assertEqual(set(g_old["image_id"]), set(g_new["image_id"]))


class TestContactSheets(unittest.TestCase):
    """人工审核拼图：候选簇分组 + 噪声单独提示。"""

    def _clusters_df(self):
        return pd.DataFrame({
            "image_id": [f"img{i}" for i in range(9)],
            "relative_path": [f"01/0{i}/a.jpg" for i in range(9)],
            "filename": [f"a{i}.jpg" for i in range(9)],
            "session_id": ["1"] * 9,
            "group_id": ["1.0"] * 9,
            "sequence_guess": ["s"] * 9,
            "cluster": [0] * 3 + [1] * 3 + [-1] * 3,
            "cluster_probability": [0.9] * 9,
        })

    def test_cluster_mode_outputs(self):
        """每个候选簇一张拼图 + 噪声一张（-1 不强制并入任何簇）。"""
        from whitewhale.review.contact_sheets import build_cluster_contact_sheets

        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "clusters.csv"
            self._clusters_df().to_csv(csv, index=False)
            out = Path(tmp) / "sheets"
            build_cluster_contact_sheets(csv, out, Path("src_dataset"), mock=True)
            files = sorted(p.name for p in out.glob("*.jpg"))
            self.assertEqual(files, ["cluster_000.jpg", "cluster_001.jpg", "noise.jpg"])
            # 噪声与候选簇分属不同文件（-1 合法噪声，不强制分配）
            self.assertNotIn("cluster_002.jpg", files)

    def test_load_review_paths_traceable(self):
        """审核数据集必须能从 relative_path 还原绝对路径（可追溯原图）。"""
        from whitewhale.data.dataset import load_review_dataset

        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "clusters.csv"
            self._clusters_df().to_csv(csv, index=False)
            root = Path("src_dataset")
            df = load_review_dataset(csv, root)
            self.assertEqual(len(df), 9)
            # 绝对路径 = 图片根(规范化) + 相对路径（Windows 分隔符兼容）
            self.assertTrue(
                df["source_path"].iloc[0].startswith(str(root.resolve()))
            )
            self.assertIn("01/00/a.jpg", df["source_path"].iloc[0].replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
