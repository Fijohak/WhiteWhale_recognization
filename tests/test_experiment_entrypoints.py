"""
实验脚本入口回归测试。

验证仓库根解析、命令行冒烟，以及散图归档在空评估样本下仍能产出结构化结果。
"""
import importlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXPERIMENTS))

from experiments.artifact_utils import load_aligned_embeddings  # noqa: E402
from whitewhale.data.manifest import compute_sha256  # noqa: E402


class TestExperimentArtifactAlignment(unittest.TestCase):
    """实验必须以特征 meta 为行序，不能假设当前清单长度和顺序不变。"""

    def test_aligns_source_rows_to_embedding_meta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            emb_path = root / "features.npy"
            np.save(emb_path, np.asarray([[3.0], [1.0]], dtype=np.float32))
            pd.DataFrame({"image_id": ["c", "a"]}).to_csv(
                root / "features_meta.csv", index=False)
            source = root / "pilot.csv"
            pd.DataFrame({"image_id": ["a", "b", "c"],
                          "individual_id": ["A", "B", "C"]}).to_csv(
                              source, index=False)

            rows, embeddings = load_aligned_embeddings(emb_path, source)

            self.assertEqual(rows["image_id"].tolist(), ["c", "a"])
            self.assertEqual(rows["individual_id"].tolist(), ["C", "A"])
            self.assertEqual(embeddings[:, 0].tolist(), [3.0, 1.0])

    def test_rejects_meta_id_missing_from_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            emb_path = root / "features.npy"
            np.save(emb_path, np.asarray([[1.0]], dtype=np.float32))
            pd.DataFrame({"image_id": ["missing"]}).to_csv(
                root / "features_meta.csv", index=False)
            source = root / "pilot.csv"
            pd.DataFrame({"image_id": ["a"]}).to_csv(source, index=False)

            with self.assertRaisesRegex(ValueError, "不在源清单"):
                load_aligned_embeddings(emb_path, source)


class TestExperimentScientificSemantics(unittest.TestCase):
    """历史实验不得自匹配，也不得把跨批次未对齐身份当成可靠负例。"""

    def test_cluster_gallery_excludes_query_and_complete_series(self):
        module = importlib.import_module("eval_cluster_retrieval")
        meta = pd.DataFrame({
            "session_id": ["A", "A", "A", "A"],
            "filename": [
                "0001_20140417_SZi_01_RAY_0100.JPG",
                "0002_20140417_SZi_01_RAY_0102.JPG",
                "0003_20140417_SZi_01_RAY_0110.JPG",
                "RES20001.JPG",
            ],
        })
        module.annotate_series(meta)

        kept = module.exclude_query_and_same_series(
            meta, np.asarray([0]), np.asarray([0, 1, 2, 3]))

        self.assertEqual(kept.tolist(), [2, 3])
        self.assertTrue(set(kept).isdisjoint({0}))

        query, gallery = module.split_identity_by_series(
            meta, np.arange(len(meta)), np.random.default_rng(7))
        self.assertTrue(set(query).isdisjoint(set(gallery)))
        self.assertEqual(set(query) | set(gallery), set(range(len(meta))))
        query_series = set(meta.loc[query, "series_id"]) - {""}
        gallery_series = set(meta.loc[gallery, "series_id"]) - {""}
        self.assertTrue(query_series.isdisjoint(gallery_series))

    def test_cross_batch_preview_is_proxy_not_open_set_calibration(self):
        module = importlib.import_module("eval_openset_preview")
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pilot = root / "pilot.csv"
            feats = root / "features.npy"
            out = root / "report"
            out.mkdir()
            (out / "unknown_detail.csv").write_text(
                "legacy,unknown\n1,1\n", encoding="utf-8")
            rows = []
            vectors = []
            basis = np.eye(4, dtype=np.float32)
            row_no = 0
            for session, identities in (
                    ("20140806 01", ("A", "B")),
                    ("20140806 03", ("C", "D"))):
                for identity_no, identity in enumerate(identities):
                    for sequence in ("SZi", "HBi"):
                        rows.append({
                            "image_id": f"img-{row_no}",
                            "session_id": session,
                            "individual_id": f"{session}_{identity}",
                            "filename": (
                                f"{row_no:04d}_20140417_{sequence}_01_RAY_0001.JPG"),
                        })
                        vectors.append(basis[(identity_no + (0 if session.endswith("01") else 2))])
                        row_no += 1
            pd.DataFrame(rows).to_csv(pilot, index=False)
            np.save(feats, np.asarray(vectors, dtype=np.float32))
            pd.DataFrame({"image_id": [row["image_id"] for row in rows]}).to_csv(
                root / "features_meta.csv", index=False)

            argv = [str(EXPERIMENTS / "eval_openset_preview.py"),
                    "--pilot", str(pilot), "--feats", str(feats),
                    "--out", str(out)]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(module, "_plot"):
                module.run()

            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            self.assertNotIn("_recommended_threshold", metrics)
            self.assertEqual(
                metrics["_meta"]["calibration_status"],
                "cross_batch_unverified_proxy_only")
            self.assertIn("cross_batch_unverified", metrics["dir_1_q3"])
            curve = pd.read_csv(out / "threshold_curve.csv")
            self.assertIn("cross_batch_unverified_accept_rate", curve.columns)
            self.assertNotIn("open_set_fa", curve.columns)
            self.assertTrue((out / "cross_batch_unverified_detail.csv").exists())
            self.assertFalse((out / "unknown_detail.csv").exists())
            self.assertEqual(
                len(list((out / "legacy_pre_alignment").glob("unknown_detail*.csv"))),
                1)

    def test_openset_different_identity_excludes_same_series(self):
        module = importlib.import_module("eval_openset_preview")
        meta = pd.DataFrame({
            "session_id": ["A", "A", "A"],
            "filename": [
                "0001_20140417_SZi_01_RAY_0100.JPG",
                "0002_20140417_SZi_01_RAY_0102.JPG",
                "0003_20140417_SZi_01_RAY_0110.JPG",
            ],
        })
        module.annotate_series(meta)
        kept = module.exclude_query_and_same_series(
            meta, 0, np.asarray([0, 1, 2]))
        self.assertEqual(kept.tolist(), [2])

    def test_viewer_image_belongs_to_reported_top_identity(self):
        module = importlib.import_module("eval51_view_html")
        embeddings = np.asarray([
            [0.8, 0.6],   # A 的最佳图
            [1.0, 0.0],   # 无标签图，原实现会错误展示它
            [0.2, 0.98],  # B
        ], dtype=float)
        chosen = module.best_gallery_image_for_identity(
            np.asarray([1.0, 0.0]), embeddings,
            np.asarray([0, 1, 2]), np.asarray(["A", "", "B"]), "A")
        self.assertEqual(chosen, 0)

    def test_local_cross_series_protocol_uses_gallery_local_gt_positions(self):
        module = importlib.import_module("local_reid_benchmark")
        info = pd.DataFrame({
            "individual_id": ["A", "A", "A", "B"],
            "series_id": ["series-1", "series-1", "series-2", "series-3"],
        })
        q_idx, galleries, gt_sets = module.protocol_leave_one_out(
            np.eye(4), info, cross_sequence=True)

        query_pos = q_idx.index(0)
        self.assertEqual(galleries[query_pos], [2, 3])
        self.assertEqual(gt_sets[query_pos], {0})
        for gallery, gt in zip(galleries, gt_sets):
            self.assertTrue(all(0 <= position < len(gallery) for position in gt))

    def test_local_evaluate_handles_ragged_gallery_and_full_map_denominator(self):
        module = importlib.import_module("local_reid_benchmark")
        embeddings = np.asarray([
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.9, 0.4358899],
        ], dtype=float)
        metrics, scores, indices = module.evaluate(
            embeddings,
            q_idx=[0, 1],
            g_idx=[[1], [0, 2, 3]],
            gt_sets=[{0}, {0, 1}],
            k=1,
        )
        self.assertEqual([len(row) for row in scores], [1, 1])
        self.assertEqual([len(row) for row in indices], [1, 1])
        self.assertAlmostEqual(metrics["mAP"], (1.0 + (1.0 + 2 / 3) / 2) / 2)

    def test_local_benchmark_main_consumes_ragged_evaluate_rows(self):
        module = importlib.import_module("local_reid_benchmark")
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            embeddings_path = root / "embeddings.npy"
            meta_path = root / "embeddings_meta.csv"
            pilot_path = root / "pilot.csv"
            out = root / "report"
            rows = []
            for identity_no, identity in enumerate(("A", "B")):
                for photo_no, sequence in enumerate(("SZi", "SZi", "HBi")):
                    image_id = f"{identity}-{photo_no}"
                    rows.append({
                        "image_id": image_id,
                        "session_id": "20140806 01",
                        "individual_id": f"20140806 01_{identity}",
                        "filename": (
                            f"{identity_no}{photo_no:03d}_20140417_{sequence}_01_"
                            f"RAY_{100 + photo_no:04d}.JPG"),
                        "width": 100 + photo_no,
                        "height": 100,
                    })
            vectors = np.asarray([
                [1.0, 0.0], [0.9, 0.1], [0.8, 0.2],
                [0.0, 1.0], [0.1, 0.9], [0.2, 0.8],
            ], dtype=np.float32)
            np.save(embeddings_path, vectors)
            pd.DataFrame({"image_id": [row["image_id"] for row in rows]}).to_csv(
                meta_path, index=False)
            pd.DataFrame(rows).to_csv(pilot_path, index=False)
            argv = [str(EXPERIMENTS / "local_reid_benchmark.py"),
                    "--embeddings", str(embeddings_path),
                    "--meta", str(meta_path), "--pilot", str(pilot_path),
                    "--k", "1", "--out", str(out)]
            with mock.patch.object(sys, "argv", argv):
                module.main()
            self.assertTrue((out / "metrics.json").exists())

    def test_eval51_data_roots_are_repo_anchored(self):
        for module_name in ("eval51_extract_all", "eval51_view_html"):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(
                    module.resolve_data_root({"data_root": "src_dataset"}),
                    ROOT / "src_dataset")

    def test_eval51_reuses_only_source_bound_unchanged_crops(self):
        module = importlib.import_module("eval51_extract_all")
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            crops = root / "crops"
            crops.mkdir()
            image_id = "IMG_test"
            crop_path = crops / f"{image_id}.jpg"
            Image.new("RGB", (4, 3), "white").save(crop_path)

            crop_row = {
                "image_id": image_id,
                "relative_path": "session/a.jpg",
                "session_id": "session",
                "x": 1,
                "y": 2,
                "w": 4,
                "h": 3,
                "det_conf": 0.5,
                "fallback": False,
            }
            pd.DataFrame([crop_row]).to_csv(
                crops / "crops_manifest.csv", index=False)
            expected = pd.DataFrame([{**crop_row,
                "confirmed_identity": "007",
                "individual_id": "007",
                "series_id": "series-1",
                "sequence_key": "seq",
                "frame": 1,
            }])

            embeddings = root / "source.npy"
            meta = root / "source_meta.csv"
            np.save(embeddings, np.asarray([[1.0, 0.0]], dtype=np.float32))
            expected.to_csv(meta, index=False)
            ordered_digest = hashlib.sha256(image_id.encode("utf-8")).hexdigest()
            crop_config = {
                "crop": "yolo",
                "crop_schema_version": 1,
                "detector_checkpoint_sha256": "detector-hash",
                "detector_fallback_policy": "center_square_min_side_0.45",
                "detector_conf": 0.25,
                "detector_imgsz": 1024,
                "detector_pad_x": 0.3,
                "detector_pad_up": 0.15,
                "detector_pad_down": 0.6,
            }
            config = {
                **crop_config,
                "model": "source-model",
                "feat_dim": 2,
                "n": 1,
                "artifact_schema_version": 2,
                "created_at_utc": "2099-01-01T00:00:00+00:00",
                "embedding_file": embeddings.name,
                "embedding_sha256": compute_sha256(embeddings),
                "meta_file": meta.name,
                "meta_sha256": compute_sha256(meta),
                "provenance_level": "generated_with_row_binding",
                "row_binding": "embedding_row_i_to_meta_image_id_i",
                "ordered_image_ids_sha256": ordered_digest,
            }
            (root / "source_config.json").write_text(
                json.dumps(config), encoding="utf-8")

            rows, provenance = module.validate_reusable_crops(
                crops, expected, embeddings, crop_config)

            self.assertEqual(rows["image_id"].tolist(), [image_id])
            self.assertEqual(
                provenance["crop_reuse_mode"],
                "validated_existing_directory")
            self.assertEqual(
                provenance["crop_manifest_sha256"],
                compute_sha256(crops / "crops_manifest.csv"))
            self.assertEqual(len(provenance["crop_bundle_sha256"]), 64)

            changed = expected.copy()
            changed.loc[0, "individual_id"] = "008"
            with self.assertRaisesRegex(ValueError, "评测标签与分串"):
                module.validate_reusable_crops(
                    crops, changed, embeddings, crop_config)


class TestExperimentEntrypoints(unittest.TestCase):
    """直接位于 experiments 下的脚本应以仓库根作为默认路径基准。"""

    MODULES = (
        "confidence_check",
        "eval_cluster_retrieval",
        "eval_openset_preview",
        "eval_pool_archival",
        "local_reid_benchmark",
        "pub_reid_benchmark",
    )

    def test_repo_root_is_current_repository(self):
        for module_name in self.MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(module.REPO_ROOT, ROOT)
        confidence = importlib.import_module("confidence_check")
        self.assertEqual(confidence.base, ROOT / "outputs")

    def test_cli_help_smoke(self):
        scripts = [
            "eval_cluster_retrieval.py",
            "eval_openset_preview.py",
            "eval_pool_archival.py",
            "local_reid_benchmark.py",
            "pub_reid_benchmark.py",
        ]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        for script in scripts:
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(EXPERIMENTS / script), "--help"],
                    cwd=Path(tempfile.gettempdir()),
                    env=env,
                    capture_output=True,
                    check=False,
                )
                stderr = result.stderr.decode("utf-8", errors="replace")
                self.assertEqual(result.returncode, 0, stderr)


class TestConfidenceCheckSemantics(unittest.TestCase):
    """置信度体检按 session 判跨群，并安全处理空配对。"""

    def test_cross_batch_is_unverified_not_negative(self):
        module = importlib.import_module("confidence_check")
        rows = pd.DataFrame({
            "image_id": ["a", "b"],
            "confirmed_identity": ["A", "B"],
            "session_id": ["20140806 01", "20140806 03"],
            "source_group": ["05", "05"],
        })
        pairs = module.pair_stats(np.eye(2), rows)
        self.assertEqual(pairs.loc[0, "kind"], "cross_batch_unverified")
        table = module.threshold_table(pairs)
        self.assertTrue((table["n_above"] == 0).all())

    def test_same_tail_on_different_dates_is_still_unverified(self):
        module = importlib.import_module("confidence_check")
        rows = pd.DataFrame({
            "image_id": ["a", "b"],
            "confirmed_identity": ["A", "B"],
            "session_id": ["20140806 01", "20140418 01"],
            "source_group": ["05", "05"],
        })
        pairs = module.pair_stats(np.eye(2), rows)
        self.assertEqual(pairs.loc[0, "kind"], "cross_batch_unverified")

    def test_empty_pair_table_has_stable_schema(self):
        module = importlib.import_module("confidence_check")
        empty = pd.DataFrame(columns=[
            "image_id", "confirmed_identity", "session_id", "source_group"])
        pairs = module.pair_stats(np.empty((0, 2)), empty)
        self.assertEqual(pairs.columns.tolist(), ["i", "j", "kind", "sim"])
        self.assertEqual(len(module.threshold_table(pairs)), 7)
        self.assertTrue(np.isnan(module.precision_at_k(np.empty((0, 2)), empty, 1)))


class TestPoolArchivalEmptySamples(unittest.TestCase):
    """散图归档评估应显式处理无弱正样本、无同批 gallery 的情况。"""

    def test_empty_samples_write_null_metrics_and_csv_headers(self):
        module = importlib.import_module("eval_pool_archival")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            loose_path = tmp / "loose.csv"
            pilot_path = tmp / "pilot.csv"
            center_crops = tmp / "center_crops"
            yolo_crops = tmp / "yolo_crops"
            feat_dir = tmp / "features"
            out_dir = tmp / "nested" / "reports"

            center_crops.mkdir()
            yolo_crops.mkdir()
            (center_crops / "pool_1.jpg").touch()
            pd.DataFrame({
                "image_id": ["pool_1"],
                "relative_path": ["1/not_ray.jpg"],
            }).to_csv(loose_path, index=False)
            pd.DataFrame({
                "image_id": ["pilot_1"],
                "session_id": [2],
            }).to_csv(pilot_path, index=False)
            pd.DataFrame({
                "image_id": ["pool_1"],
                "fallback": [False],
            }).to_csv(yolo_crops / "crops_manifest.csv", index=False)

            branch_ids = {
                "center": "pool_1",
                "yolo": "pool_1",
                "pilot_full": "pilot_1",
            }
            for branch, image_id in branch_ids.items():
                (feat_dir / branch).mkdir(parents=True)
                np.save(feat_dir / branch / "embeddings.npy",
                        np.array([[1.0, 0.0]], dtype=np.float32))
                pd.DataFrame({"image_id": [image_id]}).to_csv(
                    feat_dir / branch / "embeddings_meta.csv", index=False
                )

            argv = [
                str(EXPERIMENTS / "eval_pool_archival.py"),
                "--loose", str(loose_path),
                "--pilot", str(pilot_path),
                "--center-crops", str(center_crops),
                "--yolo-crops", str(yolo_crops),
                "--feat-dir", str(feat_dir),
                "--out", str(out_dir),
                "--reuse-feats",
            ]
            with mock.patch.object(sys, "argv", argv):
                module.run()

            metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
            for branch in ("center", "yolo"):
                self.assertEqual(metrics[f"A_{branch}"]["n_evaluable_pairs"], 0)
                self.assertIsNone(metrics[f"A_{branch}"]["pair_hit@1"])
                self.assertEqual(metrics[f"B_{branch}"]["n"], 0)
                self.assertIsNone(metrics[f"B_{branch}"]["top1_mean"])

            detail = pd.read_csv(out_dir / "B_top1_detail.csv")
            self.assertEqual(
                detail.columns.tolist(), ["image_id", "branch", "session", "top1"]
            )

            # 旧式 01 前缀应唯一映射到完整 session；缺特征图片应计数并跳过。
            pd.DataFrame({
                "image_id": ["pool_1", "pool_missing"],
                "relative_path": ["01/not_ray.jpg", "01/also_not_ray.jpg"],
            }).to_csv(loose_path, index=False)
            pd.DataFrame({
                "image_id": ["pilot_1"],
                "session_id": ["20140806 01"],
            }).to_csv(pilot_path, index=False)
            mapped_out = tmp / "mapped" / "reports"
            mapped_argv = list(argv)
            mapped_argv[mapped_argv.index(str(out_dir))] = str(mapped_out)
            with mock.patch.object(sys, "argv", mapped_argv):
                module.run()

            mapped = json.loads(
                (mapped_out / "metrics.json").read_text(encoding="utf-8")
            )
            for branch in ("center", "yolo"):
                self.assertEqual(mapped[f"B_{branch}"]["n"], 1)
                self.assertEqual(mapped[f"B_{branch}"]["n_missing_features"], 1)
                self.assertAlmostEqual(mapped[f"B_{branch}"]["top1_mean"], 1.0)

    def test_top1_summary_nonempty(self):
        module = importlib.import_module("eval_pool_archival")
        summary = module.summarize_top1([0.4, 0.6, 0.8])
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["top1_mean"], 0.6)
        self.assertAlmostEqual(summary["ge_0.6"], 2 / 3)
        self.assertAlmostEqual(summary["lt_0.55"], 1 / 3)

    def test_legacy_session_code_requires_unique_match(self):
        module = importlib.import_module("eval_pool_archival")
        self.assertEqual(
            module.resolve_session_id("01", {"20140806 01", "20140806 03"}),
            "20140806 01",
        )
        with self.assertRaisesRegex(ValueError, "对应多个"):
            module.resolve_session_id("01", {"20140806 01", "20140418 01"})

    def test_weak_pairs_require_complete_series_identity(self):
        module = importlib.import_module("eval_pool_archival")
        frame = pd.DataFrame({
            "session_id": ["A", "A", "B", "A", "A"],
            "filename": [
                "0001_20140417_SZi_01_RAY_0100.JPG",
                "0002_20140417_SZi_01_RAY_0102.JPG",
                "0003_20140417_SZi_01_RAY_0101.JPG",
                "0004_20140417_HBi_01_RAY_0101.JPG",
                "0005_20140417_SZi_01_RAY_0104.JPG",
            ],
        }, index=["a", "b", "c", "d", "e"])
        module.annotate_series(frame)

        self.assertEqual(module.build_adjacent_pairs(frame, 2),
                         [("a", "b"), ("b", "e")])


if __name__ == "__main__":
    unittest.main()
