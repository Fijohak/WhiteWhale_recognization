"""
正式管线入口与推理配置回归测试。

覆盖 CLI 帮助、r4 特征/meta 配套、首次输出目录创建、跨时间配置复用，
以及检测设备 ``auto`` 到 Ultralytics 自动选择语义的转换。
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from whitewhale.config import load_config  # noqa: E402
from whitewhale.data.image_store import (  # noqa: E402
    _THUMB_CACHE,
    ImageStore,
    validate_safe_image_ids,
)
from whitewhale.data.manifest import compute_sha256  # noqa: E402
from whitewhale.detection.detector import (  # noqa: E402
    detect_and_crop, resolve_yolo_device)
from whitewhale.pipeline.assign_pool import (  # noqa: E402
    assign_pool, attach_series_manifest, build_parser, ensure_series_metadata,
    eval_within_group, load_confirmed_reviews, load_gallery, load_pool,
    load_series_manifest, main as assign_pool_main, resolve_reviewed_identity)
from whitewhale.pipeline import cross_time  # noqa: E402
from whitewhale.pipeline.archival import (  # noqa: E402
    _fit_hdbscan,
    _scoped_gallery,
    _score_feature_by_identity,
    _validate_crop_artifact,
    cluster_embeddings,
    cluster_embeddings_by_session,
    crop_bundle_provenance,
    load_gallery as load_archival_gallery,
    match_single_feature,
    run as run_archival_pipeline,
)
from whitewhale.reid.embedding import extract_embeddings  # noqa: E402
import contact_sheets as contact_sheets_entry  # noqa: E402
import evaluate as evaluate_entry  # noqa: E402
import launch_query as launch_query_entry  # noqa: E402
import launch_review as launch_review_entry  # noqa: E402
import prepare_data as prepare_data_entry  # noqa: E402
import run_pipeline as run_pipeline_entry  # noqa: E402
from train_reid import guard_embedding_output, guard_training_output  # noqa: E402


def _generated_artifact_config() -> dict:
    """构造正式入口可接受的最小生成期行绑定 provenance。"""
    return {
        "artifact_schema_version": 2,
        "provenance_level": "generated_with_row_binding",
        "created_at_utc": "2026-08-29T00:00:00+00:00",
        "row_binding": "embedding_row_i_to_meta_image_id_i",
        "ordered_image_ids_sha256": "test-row-binding",
    }


class TestFormalEvaluationProtocol(unittest.TestCase):
    """正式评估不得把跨批次未对齐身份当作确认负例。"""

    def test_pair_diagnostics_exclude_cross_session_pairs(self):
        emb = np.asarray([
            [1.0, 0.0], [0.9, 0.1], [0.0, 1.0],
            [-1.0, 0.0], [-0.9, 0.1],
        ], dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        meta = pd.DataFrame({
            "confirmed_identity": ["a", "a", "b", "c", "c"],
            "session_id": ["s1", "s1", "s1", "s2", "s2"],
            "series_id": ["a1", "a2", "b1", "c1", "c2"],
        })

        result = evaluate_entry.eval_pairs(emb, meta)

        self.assertEqual(result["n_excluded_cross_session_unverified"], 6)
        self.assertEqual(result["n_same"], 2)
        self.assertEqual(result["n_cross"], 2)

    def test_loader_maps_series_from_full_manifest_with_missing_bridge_frame(self):
        """embedding 子集缺中间帧时，series 仍来自完整 manifest。"""
        emb = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        meta = pd.DataFrame({
            "image_id": ["first", "last"],
            "session_id": ["s1", "s1"],
            "relative_path": ["s1/first.jpg", "s1/last.jpg"],
        })
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.csv"
            pd.DataFrame({
                "image_id": ["first", "bridge", "last"],
                "session_id": ["s1"] * 3,
                "filename": [
                    "0001_20140417_SZi_01_RAY_0058.JPG",
                    "0002_20140417_SZi_01_RAY_0060.JPG",
                    "0003_20140417_SZi_01_RAY_0062.JPG",
                ],
            }).to_csv(manifest_path, index=False)

            with patch.object(
                    evaluate_entry, "load_verified_embedding_artifact",
                    return_value=(emb, meta, _generated_artifact_config())):
                _, loaded = evaluate_entry._load(
                    Path("emb.npy"), Path("meta.csv"), manifest_path)

        self.assertEqual(loaded["series_id"].nunique(), 1)
        self.assertEqual(loaded["series_unit"].nunique(), 1)

    def test_loader_rejects_subset_reannotation_without_manifest(self):
        emb = np.asarray([[1.0, 0.0]], dtype=np.float32)
        meta = pd.DataFrame({
            "image_id": ["first"],
            "session_id": ["s1"],
            "relative_path": ["s1/first.jpg"],
        })
        with patch.object(
                evaluate_entry, "load_verified_embedding_artifact",
                return_value=(emb, meta, _generated_artifact_config())):
            with self.assertRaisesRegex(ValueError, "完整 manifest"):
                evaluate_entry._load(Path("emb.npy"), Path("meta.csv"), None)

    def test_loader_rejects_backfilled_artifact_unless_diagnostic_opt_in(self):
        emb = np.asarray([[1.0, 0.0]], dtype=np.float32)
        meta = pd.DataFrame({
            "image_id": ["first"],
            "session_id": ["s1"],
            "relative_path": ["s1/first.jpg"],
            "series_id": ["stable-series"],
        })
        legacy_config = {
            "artifact_schema_version": 1,
            "provenance_level": "legacy_backfilled_unverified_row_alignment",
            "provenance_backfilled": True,
        }
        with patch.object(
                evaluate_entry, "load_verified_embedding_artifact",
                return_value=(emb, meta, legacy_config)):
            with self.assertRaisesRegex(ValueError, "artifact_schema_version"):
                evaluate_entry._load(Path("emb.npy"), Path("meta.csv"), None)
            _, loaded = evaluate_entry._load(
                Path("emb.npy"), Path("meta.csv"), None,
                allow_legacy_diagnostic=True)

        self.assertEqual(loaded["series_unit"].tolist(), ["stable-series"])

    def test_retrieval_gallery_is_limited_to_same_session(self):
        emb = np.asarray([
            [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [0.0, 1.0],
            [1.0, 0.0], [1.0, 0.0],
            [-1.0, 0.0], [-1.0, 0.0],
        ], dtype=np.float32)
        meta = pd.DataFrame({
            "image_id": [f"i{i}" for i in range(8)],
            "confirmed_identity": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "session_id": ["s1"] * 4 + ["s2"] * 4,
            "series_unit": [f"series_{i}" for i in range(8)],
        })

        result = evaluate_entry.eval_retrieval(emb, meta, seed=3)

        self.assertEqual(result["protocol"], "within_session_cross_series")
        self.assertEqual(result["gallery_per_query_min"], 2)
        self.assertEqual(result["gallery_per_query_max"], 2)
        self.assertEqual(result["r1"], 1.0)

    def test_retrieval_defensively_masks_same_session_series(self):
        """即使上游误传重叠串，逐 query 候选也必须再剔除。"""
        emb = np.asarray([
            [1.0, 0.0],       # query A
            [0.8, 0.6],       # 跨串正样本 A
            [1.0, 0.0],       # 同串泄漏负样本 B
            [0.0, 1.0],       # 跨串负样本 B
        ], dtype=np.float32)
        meta = pd.DataFrame({
            "image_id": ["q", "a_pos", "b_leak", "b_neg"],
            "confirmed_identity": ["A", "A", "B", "B"],
            "session_id": ["s1"] * 4,
            "series_unit": ["shared", "a2", "shared", "b2"],
        })

        def force_leaky_split(frame, **_kwargs):
            return frame[frame["image_id"] == "q"], frame[frame["image_id"] != "q"]

        with patch.object(
                evaluate_entry, "split_query_gallery", side_effect=force_leaky_split):
            result = evaluate_entry.eval_retrieval(emb, meta, seed=42)

        self.assertEqual(result["n_split_overlap_series"], 1)
        self.assertEqual(result["n_excluded_same_series_gallery"], 1)
        self.assertEqual(result["gallery_per_query_min"], 2)
        self.assertEqual(result["r1"], 1.0)

    def test_retrieval_skips_queries_without_positive_or_negative(self):
        emb = np.asarray([
            [1.0, 0.0], [0.9, 0.1], [0.0, 1.0],
            [1.0, 0.0], [0.9, 0.1],
            [1.0, 0.0], [0.0, 1.0],
        ], dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        meta = pd.DataFrame({
            "image_id": ["qa", "ap", "bn", "qc", "cp", "qd", "en"],
            "confirmed_identity": ["A", "A", "B", "C", "C", "D", "E"],
            "session_id": ["s1", "s1", "s1", "s2", "s2", "s3", "s3"],
            "series_unit": ["aq", "ap", "bp", "cq", "cp", "dq", "ep"],
        })

        def force_split(frame, **_kwargs):
            query = frame[frame["image_id"].isin(["qa", "qc", "qd"])]
            gallery = frame[~frame["image_id"].isin(["qa", "qc", "qd"])]
            return query, gallery

        with patch.object(
                evaluate_entry, "split_query_gallery", side_effect=force_split):
            result = evaluate_entry.eval_retrieval(emb, meta, seed=42)

        self.assertEqual(result["n_query_split"], 3)
        self.assertEqual(result["n_query"], 1)
        self.assertEqual(result["n_query_skipped"], 2)
        self.assertEqual(result["n_query_skipped_no_positive"], 1)
        self.assertEqual(result["n_query_skipped_no_negative"], 1)


class TestCliEntrypoints(unittest.TestCase):
    """正式 wrapper 的 --help 必须可运行，不能被裸百分号破坏。"""

    def test_help_smoke(self):
        scripts = {
            "scripts/run_pipeline.py": b"--det-device",
            "scripts/launch_review.py": b"--clusters",
            "scripts/prepare_data.py": b"build-pilot",
            "scripts/contact_sheets.py": b"--images-root",
            "scripts/launch_query.py": b"--det-device",
            "scripts/assign_pool.py": b"--gallery-meta",
            "scripts/run_cross_time_batch.py": b"--sessions",
            "scripts/train_reid.py": b"--test-session",
        }
        for relative_path, expected in scripts.items():
            with self.subTest(script=relative_path):
                result = subprocess.run(
                    [sys.executable, str(ROOT / relative_path), "--help"],
                    cwd=ROOT, capture_output=True, check=False)
                self.assertEqual(
                    result.returncode, 0,
                    (result.stdout + result.stderr).decode(errors="replace"))
                self.assertIn(expected, result.stdout)

    def test_launch_review_requires_explicit_clusters(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "launch_review.py")],
                cwd=tmp_dir, capture_output=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"--clusters", result.stderr)

    def test_config_relative_paths_are_repo_rooted(self):
        relative = Path("outputs") / "example.csv"
        for entry in (
                run_pipeline_entry, launch_review_entry,
                prepare_data_entry, contact_sheets_entry):
            with self.subTest(entry=entry.__name__):
                self.assertEqual(entry._repo_path(relative), ROOT / relative)
                self.assertEqual(entry._repo_path(ROOT / relative), ROOT / relative)

    def test_multi_session_manifest_requires_explicit_selection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = Path(tmp_dir) / "manifest.csv"
            pd.DataFrame({
                "image_id": ["a", "b"],
                "relative_path": ["s1/a.jpg", "s2/b.jpg"],
                "session_id": ["s1", "s2"],
            }).to_csv(manifest, index=False)

            with self.assertRaisesRegex(ValueError, "必须用 --session"):
                run_pipeline_entry.select_manifest_session(manifest, None)
            selected, inferred = run_pipeline_entry.select_manifest_session(
                manifest, "s2")

        self.assertEqual(inferred, "s2")
        self.assertEqual(selected["image_id"].tolist(), ["b"])

    def test_run_pipeline_rejects_multi_session_before_processing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = Path(tmp_dir) / "manifest.csv"
            pd.DataFrame({
                "image_id": ["a", "b"],
                "relative_path": ["s1/a.jpg", "s2/b.jpg"],
                "session_id": ["s1", "s2"],
            }).to_csv(manifest, index=False)
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "run_pipeline.py"),
                "--input-manifest", str(manifest),
                "--batch-name", "unsafe",
            ], cwd=tmp_dir, capture_output=True, check=False)

        self.assertEqual(result.returncode, 2)
        self.assertIn(b"--session", result.stderr)

    def test_run_pipeline_writes_only_selected_session(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            manifest = tmp / "manifest.csv"
            pd.DataFrame({
                "image_id": ["a", "b"],
                "relative_path": ["s1/a.jpg", "s2/b.jpg"],
                "session_id": ["s1", "s2"],
            }).to_csv(manifest, index=False)
            argv = [
                "run_pipeline.py", "--input-manifest", str(manifest),
                "--session", "s2", "--batch-name", "s2",
                "--out", str(tmp / "out"),
                "--images-root", "relative_images",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                    run_pipeline_entry, "run") as run_archival:
                run_pipeline_entry.main()

            args = run_archival.call_args.args[0]
            selected = args.input_manifest_data

        self.assertEqual(selected["image_id"].tolist(), ["b"])
        self.assertEqual(args.images_root, ROOT / "relative_images")
        self.assertEqual(args.out, (tmp / "out" / "s2").resolve())
        self.assertFalse(args.out.exists())

    def test_manifest_requires_nonempty_session_column(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            missing = tmp / "missing.csv"
            blank = tmp / "blank.csv"
            pd.DataFrame({
                "image_id": ["a"], "relative_path": ["a.jpg"],
            }).to_csv(missing, index=False)
            pd.DataFrame({
                "image_id": ["a"], "relative_path": ["a.jpg"],
                "session_id": [""],
            }).to_csv(blank, index=False)

            with self.assertRaisesRegex(ValueError, "缺少 session_id"):
                run_pipeline_entry.select_manifest_session(missing, None)
            with self.assertRaisesRegex(ValueError, "全为空"):
                run_pipeline_entry.select_manifest_session(blank, None)

    def test_existing_formal_batch_is_rejected_before_archival(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "out"
            (root / "existing").mkdir(parents=True)
            argv = [
                "run_pipeline.py", "--pool", "--batch-name", "existing",
                "--out", str(root),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                    run_pipeline_entry, "run") as run_archival, self.assertRaises(
                        SystemExit):
                run_pipeline_entry.main()

        run_archival.assert_not_called()

    def test_run_pipeline_pool_defaults_follow_pipeline_config(self):
        cfg = load_config("pipeline")
        output_root = ROOT / cfg["output_root"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            argv = [
                "run_pipeline.py", "--pool", "--batch-name", "pool",
                "--out", str(Path(tmp_dir) / "out"),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                    run_pipeline_entry, "run") as run_archival:
                run_pipeline_entry.main()

        args = run_archival.call_args.args[0]
        self.assertEqual(
            args.gallery_embeddings,
            output_root / cfg["query"]["embeddings"])
        self.assertEqual(
            args.gallery_meta, output_root / cfg["query"]["meta"])
        self.assertEqual(
            args.pool_embeddings,
            output_root / cfg["pool"]["embeddings"])
        self.assertEqual(args.pool_meta, output_root / cfg["pool"]["meta"])
        self.assertEqual(args.pool_crops, output_root / cfg["pool"]["crops"])

    def test_launch_query_defaults_follow_pipeline_config(self):
        cfg = load_config("pipeline")
        output_root = ROOT / cfg["output_root"]
        app = Mock()
        app.state.n_gallery = 1
        app.state.model_name = "test"
        with patch.object(sys, "argv", ["launch_query.py"]), patch.object(
                launch_query_entry, "build_app", return_value=app
        ) as build_app, patch("uvicorn.run"):
            launch_query_entry.main()

        args = build_app.call_args.args[0]
        self.assertEqual(
            args.embeddings, output_root / cfg["query"]["embeddings"])
        self.assertEqual(args.meta, output_root / cfg["query"]["meta"])

    def test_training_refuses_accidental_artifact_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "r4"
            out.mkdir()
            (out / "best.pt").touch()
            with self.assertRaisesRegex(SystemExit, "拒绝覆盖"):
                guard_training_output(out, overwrite=False)
            guard_training_output(out, overwrite=True)

            embeddings = Path(tmp_dir) / "embeddings.npy"
            embeddings.touch()
            with self.assertRaisesRegex(SystemExit, "特征输出已有产物"):
                guard_embedding_output(embeddings, overwrite=False)
            guard_embedding_output(embeddings, overwrite=True)


class TestEntrypointPathSafety(unittest.TestCase):
    """Windows 路径与文件 basename 均须在计算前拒绝危险输入。"""

    def test_batch_name_rejects_windows_and_traversal_forms(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir) / "outputs"
            for name in (
                    "C:", "C:batch", "..", "../x", r"..\x", "CON",
                    "con.txt", "CON .txt", "LPT1", "batch.", "batch ",
                    "/absolute"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    run_pipeline_entry.resolve_batch_output(out_root, name)

            resolved = run_pipeline_entry.resolve_batch_output(
                out_root, "20140419 02")

        self.assertEqual(resolved.parent, out_root.resolve())

    def test_manifest_rejects_unsafe_empty_and_duplicate_image_ids(self):
        invalid_sets = [
            ["ok", ""],
            ["ok", "../x"],
            ["ok", "/absolute"],
            ["ok", "C:drive-relative"],
            ["ok", "CON"],
            ["same", "same"],
            ["same", "SAME"],
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            for position, image_ids in enumerate(invalid_sets):
                manifest = Path(tmp_dir) / f"invalid_{position}.csv"
                pd.DataFrame({
                    "image_id": image_ids,
                    "relative_path": ["s/a.jpg", "s/b.jpg"],
                    "session_id": ["s", "s"],
                }).to_csv(manifest, index=False)
                with self.subTest(image_ids=image_ids), self.assertRaises(ValueError):
                    run_pipeline_entry.select_manifest_session(manifest, "s")

        with self.assertRaisesRegex(ValueError, "不能为空"):
            validate_safe_image_ids([np.nan])

    def test_detector_and_embedding_validate_image_id_at_actual_entry(self):
        frame = pd.DataFrame({
            "image_id": ["../escape"],
            "relative_path": ["missing.jpg"],
        })
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            with self.assertRaisesRegex(ValueError, "不安全的 image_id"):
                detect_and_crop(frame, tmp, tmp / "crops", tmp / "detector.pt")

            model = Mock(feat_dim=2, name="mock")
            with self.assertRaisesRegex(ValueError, "不安全的 image_id"):
                extract_embeddings(frame, model, images_root=tmp)

    def test_image_store_rejects_prefix_sibling_and_drive_relative_escape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            root = tmp / "data"
            sibling = tmp / "database"
            root.mkdir()
            sibling.mkdir()
            (sibling / "secret.jpg").touch()
            store = ImageStore(root)

            with self.assertRaisesRegex(ValueError, "路径越界"):
                store.resolve("../database/secret.jpg")
            with self.assertRaisesRegex(ValueError, "路径越界"):
                store.resolve("C:secret.jpg")

    def test_thumbnail_cache_is_namespaced_by_resolved_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            roots = [tmp / "a", tmp / "b"]
            colors = [(255, 0, 0), (0, 0, 255)]
            for root, color in zip(roots, colors):
                root.mkdir()
                Image.new("RGB", (20, 20), color).save(root / "same.jpg")
            _THUMB_CACHE.clear()

            first = ImageStore(roots[0]).thumbnail("same.jpg")
            second = ImageStore(roots[1]).thumbnail("same.jpg")

            self.assertNotEqual(first, second)
            self.assertEqual(len(_THUMB_CACHE), 2)

    def test_detector_missing_source_fails_before_model_or_output(self):
        frame = pd.DataFrame({
            "image_id": ["safe_id"],
            "relative_path": ["missing.jpg"],
        })
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            out = tmp / "crops"
            with patch("ultralytics.YOLO") as yolo, self.assertRaisesRegex(
                    FileNotFoundError, "检测前终止"):
                detect_and_crop(frame, tmp, out, tmp / "detector.pt")

            yolo.assert_not_called()
            self.assertFalse(out.exists())


class TestAssignPoolEntrypoint(unittest.TestCase):
    """散图划分默认输入必须同为 r4，并可写入首次创建的输出目录。"""

    def test_default_gallery_meta_matches_r4_embeddings(self):
        base = Path("somewhere") / "outputs"
        cfg = load_config("pipeline")
        args = build_parser(base).parse_args([])
        self.assertEqual(args.gallery_embeddings,
                         base / cfg["query"]["embeddings"])
        self.assertEqual(args.gallery_meta, base / cfg["query"]["meta"])
        self.assertEqual(args.pool_embeddings,
                         base / cfg["pool"]["embeddings"])
        self.assertEqual(args.pool_meta, base / cfg["pool"]["meta"])
        self.assertEqual(args.manifest.name, "dataset_manifest.csv")

    def test_loaders_reject_subset_reannotation_without_manifest(self):
        with self.assertRaisesRegex(ValueError, "完整 manifest"):
            load_pool(Path("pool.npy"), Path("pool_meta.csv"))
        with self.assertRaisesRegex(ValueError, "完整 manifest"):
            load_gallery(Path("gallery.npy"), Path("gallery_meta.csv"))

    def test_wrapper_creates_new_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            pool_emb = tmp / "pool.npy"
            pool_meta = tmp / "pool_meta.csv"
            gallery_emb = tmp / "gallery.npy"
            gallery_meta = tmp / "gallery_meta.csv"
            manifest = tmp / "dataset_manifest.csv"
            out = tmp / "new" / "assignment"

            np.save(pool_emb, np.array([[1.0, 0.0]], dtype=np.float32))
            pd.DataFrame({
                "image_id": ["q1"],
                "relative_path": ["01/q1.jpg"],
                "session_id": ["01"],
            }).to_csv(pool_meta, index=False)
            np.save(gallery_emb, np.array(
                [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            pd.DataFrame({
                "image_id": ["g1", "g2"],
                "relative_path": ["01/g1.jpg", "01/g2.jpg"],
                "session_id": ["01", "01"],
                "confirmed_identity": [1.0, 2.0],
            }).to_csv(gallery_meta, index=False)
            pd.DataFrame({
                "image_id": ["q1", "g1", "g2"],
                "relative_path": ["q1.jpg", "g1.jpg", "g2.jpg"],
                "session_id": ["01", "01", "01"],
            }).to_csv(manifest, index=False)
            for emb_path, meta_path, n in (
                (pool_emb, pool_meta, 1),
                (gallery_emb, gallery_meta, 2),
            ):
                ordered_ids = pd.read_csv(
                    meta_path, dtype=str, keep_default_na=False)["image_id"]
                emb_path.with_name(f"{emb_path.stem}_config.json").write_text(
                    json.dumps({
                        "embedding_file": emb_path.name,
                        "meta_file": meta_path.name,
                        "embedding_sha256": compute_sha256(emb_path),
                        "meta_sha256": compute_sha256(meta_path),
                        "n": n,
                        "feat_dim": 2,
                        "model": "test-model",
                        "crop": "yolo",
                        "preprocess": "test-normalize",
                        "crop_schema_version": 1,
                        "detector_checkpoint_sha256": "detector-hash",
                        "detector_conf": 0.25,
                        "detector_imgsz": 1024,
                        "detector_pad_x": 0.30,
                        "detector_pad_up": 0.15,
                        "detector_pad_down": 0.60,
                        "detector_fallback_policy": "center_square_min_side_0.45",
                        "artifact_schema_version": 2,
                        "provenance_level": "generated_with_row_binding",
                        "created_at_utc": "2026-08-29T00:00:00+00:00",
                        "row_binding": "embedding_row_i_to_meta_image_id_i",
                        "ordered_image_ids_sha256": hashlib.sha256(
                            "\n".join(ordered_ids).encode("utf-8")
                        ).hexdigest(),
                    }),
                    encoding="utf-8",
                )

            result = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "assign_pool.py"),
                "--pool-embeddings", str(pool_emb),
                "--pool-meta", str(pool_meta),
                "--gallery-embeddings", str(gallery_emb),
                "--gallery-meta", str(gallery_meta),
                "--manifest", str(manifest),
                "--reviews", str(tmp / "missing_reviews.csv"),
                "--out", str(out),
            ], cwd=ROOT, capture_output=True, check=False)

            self.assertEqual(
                result.returncode, 0,
                (result.stdout + result.stderr).decode(errors="replace"))
            self.assertTrue((out / "pool_candidates_before.csv").exists())
            self.assertTrue((out / "pool_candidates.csv").exists())
            self.assertTrue((out / "series_manifest_provenance.json").exists())

    def test_invalid_input_does_not_create_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            out = tmp / "must_not_exist"
            argv = [
                "assign_pool.py", "--eval",
                "--manifest", str(tmp / "missing_manifest.csv"),
                "--out", str(out),
            ]

            with patch.object(sys, "argv", argv), self.assertRaises(FileNotFoundError):
                assign_pool_main()

            self.assertFalse(out.exists())

    def test_empty_pool_keeps_output_schema(self):
        pool = pd.DataFrame(columns=[
            "image_id", "group", "relative_path", "_emb"])
        gallery = pd.DataFrame(columns=[
            "image_id", "group", "confirmed_identity", "emb"])
        pool = ensure_series_metadata(pool)
        gallery = ensure_series_metadata(gallery)

        result = assign_pool(pool, gallery, topk=5, threshold=0.5)

        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), [
            "image_id", "group", "relative_path", "series_id", "series_unit",
            "top1", "top1_score", "top1_image_id", "candidates",
            "candidate_gallery_n", "excluded_same_series_n", "status"])

    def test_no_gallery_row_remains_traceable(self):
        pool = pd.DataFrame([{
            "image_id": "q1", "group": "01", "relative_path": "01/q1.jpg",
            "_emb": np.array([1.0, 0.0]),
        }])
        gallery = pd.DataFrame(columns=[
            "image_id", "group", "confirmed_identity", "emb"])
        pool = ensure_series_metadata(pool)
        gallery = ensure_series_metadata(gallery)

        result = assign_pool(pool, gallery, topk=5, threshold=0.5)

        self.assertEqual(result.loc[0, "status"], "no_gallery")
        self.assertEqual(result.loc[0, "relative_path"], "01/q1.jpg")

    def test_topk_aggregates_images_by_confirmed_identity(self):
        """照片排名 A、A、B 时，个体 Top-2 必须是 A、B。"""
        pool = pd.DataFrame([{
            "image_id": "q1", "group": "01", "relative_path": "01/q1.jpg",
            "_emb": np.array([1.0, 0.0]),
        }])
        gallery = pd.DataFrame([
            {"image_id": "a_best", "group": "01", "confirmed_identity": "A",
             "emb": np.array([1.0, 0.0])},
            {"image_id": "a_second", "group": "01", "confirmed_identity": "A",
             "emb": np.array([0.9, np.sqrt(1.0 - 0.9**2)])},
            {"image_id": "b_best", "group": "01", "confirmed_identity": "B",
             "emb": np.array([0.8, 0.6])},
        ])
        pool = ensure_series_metadata(pool)
        gallery = ensure_series_metadata(gallery)

        result = assign_pool(pool, gallery, topk=2, threshold=0.5)

        self.assertEqual(result.loc[0, "candidates"], "A@1.000; B@0.800")
        self.assertEqual(result.loc[0, "top1"], "A")
        self.assertAlmostEqual(result.loc[0, "top1_score"], 1.0)
        self.assertEqual(result.loc[0, "top1_image_id"], "a_best")

    def test_numeric_looking_sessions_remain_distinct(self):
        pool = ensure_series_metadata(pd.DataFrame([{
            "image_id": "q", "session_id": "01", "relative_path": "q.jpg",
            "_emb": np.array([1.0, 0.0]),
        }]))
        gallery = ensure_series_metadata(pd.DataFrame([
            {"image_id": "same", "session_id": "01", "relative_path": "same.jpg",
             "confirmed_identity": "A", "emb": np.array([0.8, 0.6])},
            {"image_id": "other", "session_id": "1.0", "relative_path": "other.jpg",
             "confirmed_identity": "B", "emb": np.array([1.0, 0.0])},
        ]))

        result = assign_pool(pool, gallery, topk=2, threshold=0.5)

        self.assertEqual(pool.loc[0, "group"], "01")
        self.assertEqual(gallery["group"].tolist(), ["01", "1.0"])
        self.assertEqual(result.loc[0, "top1"], "A")
        self.assertEqual(result.loc[0, "candidate_gallery_n"], 1)

    def test_full_manifest_bridge_excludes_near_duplicate_across_artifacts(self):
        """不在特征子集的 0060 必须把 pool 0058 与 gallery 0062 桥接成同串。"""
        pool = pd.DataFrame([{
            "image_id": "q58", "session_id": "s1",
            "relative_path": "s1/loose/0100_20140417_SZi_01_RAY_0058.JPG",
            "_emb": np.array([1.0, 0.0]),
        }])
        gallery = pd.DataFrame([
            {"image_id": "a59", "session_id": "s1",
             "relative_path": "s1/A/0101_20140417_SZi_01_RAY_0059.JPG",
             "confirmed_identity": "A", "emb": np.array([1.0, 0.0])},
            {"image_id": "a62", "session_id": "s1",
             "relative_path": "s1/A/0102_20140417_SZi_01_RAY_0062.JPG",
             "confirmed_identity": "A", "emb": np.array([0.99, 0.01])},
            {"image_id": "b1", "session_id": "s1",
             "relative_path": "s1/B/0200_20140417_HBi_01_RAY_0001.JPG",
             "confirmed_identity": "B", "emb": np.array([0.8, 0.6])},
            {"image_id": "a_other", "session_id": "s1",
             "relative_path": "s1/A/0300_20140418_SZi_01_RAY_0001.JPG",
             "confirmed_identity": "A", "emb": np.array([0.7, 0.714])},
        ])
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "dataset_manifest.csv"
            pd.DataFrame({
                "image_id": ["q58", "a59", "bridge60", "a62", "b1", "a_other"],
                "session_id": ["s1"] * 6,
                "relative_path": [
                    "0100_20140417_SZi_01_RAY_0058.JPG",
                    "0101_20140417_SZi_01_RAY_0059.JPG",
                    "0999_20140417_SZi_01_RAY_0060.JPG",
                    "0102_20140417_SZi_01_RAY_0062.JPG",
                    "0200_20140417_HBi_01_RAY_0001.JPG",
                    "0300_20140418_SZi_01_RAY_0001.JPG",
                ],
            }).to_csv(manifest_path, index=False)
            series_index = load_series_manifest(manifest_path)
            annotated_pool = attach_series_manifest(pool, series_index, "pool")
            annotated_gallery = attach_series_manifest(
                gallery, series_index, "gallery")

        self.assertTrue(annotated_pool.loc[0, "series_id"])
        self.assertEqual(
            annotated_pool.loc[0, "series_unit"],
            annotated_gallery.loc[0, "series_unit"])
        self.assertEqual(
            annotated_pool.loc[0, "series_unit"],
            annotated_gallery.loc[1, "series_unit"])

        result = assign_pool(
            annotated_pool, annotated_gallery, topk=2, threshold=0.5)

        self.assertEqual(result.loc[0, "top1"], "B")
        self.assertEqual(result.loc[0, "top1_image_id"], "b1")
        self.assertEqual(result.loc[0, "candidates"], "B@0.800; A@0.700")
        self.assertEqual(result.loc[0, "candidate_gallery_n"], 2)
        self.assertEqual(result.loc[0, "excluded_same_series_n"], 2)

    def test_series_manifest_requires_unique_ids_and_full_coverage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            duplicate = tmp / "duplicate.csv"
            pd.DataFrame({
                "image_id": ["same", "same"],
                "session_id": ["s1", "s1"],
                "relative_path": ["a.jpg", "b.jpg"],
            }).to_csv(duplicate, index=False)
            with self.assertRaisesRegex(ValueError, "为空或重复"):
                load_series_manifest(duplicate)

            manifest = tmp / "manifest.csv"
            pd.DataFrame({
                "image_id": ["known"],
                "session_id": ["s1"],
                "relative_path": ["known.jpg"],
            }).to_csv(manifest, index=False)
            series_index = load_series_manifest(manifest)
            self.assertEqual(len(series_index.attrs["provenance"]["manifest_sha256"]), 64)
            with self.assertRaisesRegex(ValueError, "未覆盖 pool"):
                attach_series_manifest(pd.DataFrame({
                    "image_id": ["missing"], "session_id": ["s1"],
                }), series_index, "pool")

    def test_only_same_series_gallery_has_explicit_status(self):
        pool = pd.DataFrame([{
            "image_id": "q58", "session_id": "s1",
            "relative_path": "s1/0100_20140417_SZi_01_RAY_0058.JPG",
            "_emb": np.array([1.0, 0.0]),
        }])
        gallery = pd.DataFrame([{
            "image_id": "a59", "session_id": "s1",
            "relative_path": "s1/0101_20140417_SZi_01_RAY_0059.JPG",
            "confirmed_identity": "A", "emb": np.array([1.0, 0.0]),
        }])
        pool["group"] = "s1"
        pool["series_id"] = "s1|20140417_SZi_01_RAY#0058-0059"
        pool["series_unit"] = pool["series_id"]
        gallery["group"] = "s1"
        gallery["series_id"] = pool.loc[0, "series_id"]
        gallery["series_unit"] = pool.loc[0, "series_unit"]

        result = assign_pool(pool, gallery, topk=5, threshold=0.5)

        self.assertEqual(result.loc[0, "status"], "no_cross_series_candidate")
        self.assertEqual(result.loc[0, "candidate_gallery_n"], 0)
        self.assertEqual(result.loc[0, "excluded_same_series_n"], 1)
        self.assertEqual(result.loc[0, "top1"], "")

    def test_eval_ignores_cross_session_false_negative_and_reports_skips(self):
        gallery = pd.DataFrame([
            {"image_id": "a1", "session_id": "s1",
             "relative_path": "s1/A/0001_20140417_SZi_01_RAY_0001.JPG",
             "confirmed_identity": "A", "emb": np.array([1.0, 0.0])},
            {"image_id": "a2", "session_id": "s1",
             "relative_path": "s1/A/0002_20140417_HBi_01_RAY_0001.JPG",
             "confirmed_identity": "A", "emb": np.array([0.8, 0.6])},
            {"image_id": "b1", "session_id": "s1",
             "relative_path": "s1/B/0003_20140417_SCi_01_RAY_0001.JPG",
             "confirmed_identity": "B", "emb": np.array([0.0, 1.0])},
            # 跨 session 的 C 与 a1 完全相同；不得作为 a1 的负例挤掉 A。
            {"image_id": "c1", "session_id": "s2",
             "relative_path": "s2/C/0004_20140417_SZi_01_RAY_0001.JPG",
             "confirmed_identity": "C", "emb": np.array([1.0, 0.0])},
            {"image_id": "d1", "session_id": "s2",
             "relative_path": "s2/D/0005_20140417_HBi_01_RAY_0001.JPG",
             "confirmed_identity": "D", "emb": np.array([0.0, 1.0])},
        ])
        gallery = ensure_series_metadata(gallery)

        result = eval_within_group(gallery)

        self.assertEqual(result["protocol"], "within_session_cross_series")
        self.assertEqual(result["within_group"]["s1"], 1.0)
        self.assertEqual(result["n_total"], 5)
        self.assertEqual(result["n_evaluated"], 2)
        self.assertEqual(result["n_skipped_total"], 3)
        self.assertEqual(result["n_skipped_no_cross_series_positive"], 3)
        self.assertNotIn("all_gallery_r1", result)
        self.assertEqual(
            result["cross_session_all_gallery"]["status"], "not_reported")

    def test_namespaced_identity_is_opaque_string(self):
        pool = pd.DataFrame([{
            "image_id": "q1", "group": "20140806 01",
            "relative_path": "20140806 01/q1.jpg",
            "_emb": np.array([1.0, 0.0]),
        }])
        gallery = pd.DataFrame([{
            "image_id": "g1", "group": "20140806 01",
            "confirmed_identity": "20140806 01_5.0",
            "emb": np.array([1.0, 0.0]),
        }])
        pool = ensure_series_metadata(pool)
        gallery = ensure_series_metadata(gallery)

        result = assign_pool(pool, gallery, topk=5, threshold=0.5)

        self.assertEqual(result.loc[0, "top1"], "20140806 01_5.0")
        self.assertIn("20140806 01_5.0@1.000", result.loc[0, "candidates"])
        self.assertEqual(
            resolve_reviewed_identity("5.0", "20140806 01", gallery),
            "5.0",
        )

        canonical_gallery = gallery.copy()
        canonical_gallery["confirmed_identity"] = "20140806 01_05"
        migration = {
            "20140806 01_5.0": "20140806 01_05",
        }
        self.assertEqual(
            resolve_reviewed_identity(
                "20140806 01_5.0", "20140806 01", canonical_gallery,
                migration),
            "20140806 01_05",
        )
        self.assertEqual(
            resolve_reviewed_identity(
                "5.0", "20140806 01", canonical_gallery, migration),
            "20140806 01_05",
        )
        # 数字外观不是转换授权：前导零、float 与科学计数法都保持原串。
        self.assertEqual(
            resolve_reviewed_identity("0005", "20140806 01", canonical_gallery),
            "0005")
        self.assertEqual(
            resolve_reviewed_identity("1e3", "20140806 01", canonical_gallery),
            "1e3")

    def test_reviews_preserve_leading_zero_and_literal_na(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reviews = Path(tmp_dir) / "reviews.csv"
            pd.DataFrame({
                "image_id": ["0001", "0002", "0003", "0004", "0005", "0006"],
                "review_status": ["confirmed", " Confirmed ", "confirmed",
                                  "confirmed", "confirmed", "rejected"],
                "reviewed_identity": ["0005", "NA", "N/A", "null", "", "01"],
            }).to_csv(reviews, index=False)

            confirmed = load_confirmed_reviews(reviews)

        self.assertEqual(
            confirmed["image_id"].tolist(), ["0001", "0002", "0003", "0004"])
        self.assertEqual(
            confirmed["reviewed_identity"].tolist(),
            ["0005", "NA", "N/A", "null"])

    def test_duplicate_confirmed_review_image_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reviews = Path(tmp_dir) / "reviews.csv"
            pd.DataFrame({
                "image_id": ["same", "same"],
                "review_status": ["confirmed", "confirmed"],
                "reviewed_identity": ["A", "B"],
            }).to_csv(reviews, index=False)

            with self.assertRaisesRegex(ValueError, "image_id 必须唯一"):
                load_confirmed_reviews(reviews)


class TestDetectionDevice(unittest.TestCase):
    """auto 在项目边界转换为 None，由 Ultralytics 自动选 GPU/CPU。"""

    def test_resolve_yolo_device(self):
        self.assertIsNone(resolve_yolo_device("auto"))
        self.assertIsNone(resolve_yolo_device(" AUTO "))
        self.assertIsNone(resolve_yolo_device(None))
        self.assertEqual(resolve_yolo_device("cpu"), "cpu")
        self.assertEqual(resolve_yolo_device("0"), "0")

    def test_detect_and_crop_passes_auto_as_none(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            images = tmp / "images"
            images.mkdir()
            Image.new("RGB", (20, 20), (128, 128, 128)).save(images / "q.jpg")
            frame = pd.DataFrame({
                "image_id": ["q"],
                "relative_path": ["q.jpg"],
                "session_id": ["s"],
            })
            model = Mock()
            model.predict.return_value = []

            with patch("ultralytics.YOLO", return_value=model):
                detect_and_crop(
                    frame, images, tmp / "crops", tmp / "weights.pt",
                    device="auto", preview=False)

            self.assertIsNone(model.predict.call_args.kwargs["device"])


class TestArchivalInvalidFeature(unittest.TestCase):
    """缺图/NaN 特征只能进入无效状态，不能随机指派候选。"""

    def test_nan_feature_has_no_candidate(self):
        top1, score, status = match_single_feature(
            np.array([np.nan, np.nan]),
            np.array([[1.0, 0.0]]),
            np.array(["known"]),
            threshold=0.5,
        )
        self.assertEqual(top1, "")
        self.assertTrue(np.isnan(score))
        self.assertEqual(status, "invalid_feature")

    def test_same_raw_identity_in_different_sessions_is_not_merged(self):
        feature = np.asarray([1.0, 0.0])
        gallery = np.asarray([
            [0.8, 0.6],
            [1.0, 0.0],
        ])
        keys = [("s1", "A"), ("s2", "A")]

        scores = _score_feature_by_identity(feature, gallery, keys)
        top1, score, status = match_single_feature(
            feature, gallery, keys, threshold=0.5)

        self.assertEqual(set(scores), {("s1", "A"), ("s2", "A")})
        self.assertEqual(top1, "s2::A")
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(status, "noise_match_candidate")

    def test_single_valid_embedding_degrades_to_noise(self):
        labels, probabilities = cluster_embeddings(
            np.asarray([[1.0, 0.0]], dtype=np.float32), min_cluster_size=3)

        self.assertEqual(labels.tolist(), [-1])
        self.assertEqual(probabilities.tolist(), [0.0])

    def test_session_smaller_than_min_cluster_size_degrades_to_noise(self):
        with patch("whitewhale.pipeline.archival._fit_hdbscan") as fit:
            labels, probabilities = cluster_embeddings(
                np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
                min_cluster_size=3,
            )

        self.assertEqual(labels.tolist(), [-1, -1])
        self.assertEqual(probabilities.tolist(), [0.0, 0.0])
        fit.assert_not_called()

    def test_subcluster_uses_loaded_hdbscan_module(self):
        fake_clusterer = Mock()
        fake_clusterer.fit_predict.return_value = np.asarray([0, 0])
        fake_clusterer.probabilities_ = np.asarray([0.8, 0.9])
        fake_module = Mock()
        fake_module.HDBSCAN.return_value = fake_clusterer

        with patch.dict("sys.modules", {"hdbscan": fake_module}):
            labels, probabilities = _fit_hdbscan(
                np.asarray([[1.0, 0.0], [0.9, 0.1]]),
                min_cluster_size=2,
                min_samples=1,
            )

        self.assertEqual(labels.tolist(), [0, 0])
        self.assertEqual(probabilities.tolist(), [0.8, 0.9])
        fake_module.HDBSCAN.assert_called_once_with(
            min_cluster_size=2, min_samples=1)


class TestArchivalTransactions(unittest.TestCase):
    """正式归档只能发布全新、完整的批次目录。"""

    @staticmethod
    def _args(out: Path) -> SimpleNamespace:
        return SimpleNamespace(
            out=out,
            pool=True,
            pool_embeddings=Path("pool.npy"),
            pool_meta=Path("pool_meta.csv"),
            pool_crops=Path("pool_crops"),
            gallery_embeddings=Path("gallery.npy"),
            gallery_meta=Path("gallery_meta.csv"),
            min_cluster_size=3,
            subcluster_min_size=4,
            topk=3,
            threshold_cluster=0.58,
            threshold_image=0.50,
            sheets=False,
            max_sheets=10,
            images_root=Path("images"),
        )

    @staticmethod
    def _gallery(config: dict):
        return (
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            [("s", "known")],
            pd.DataFrame({
                "image_id": ["g"], "relative_path": ["g.jpg"],
                "session_id": ["s"], "series_id": ["gallery-series"],
                "confirmed_identity": ["known"],
            }),
            config,
        )

    @staticmethod
    def _pool(config: dict, session: str = "s"):
        return (
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            pd.DataFrame({
                "image_id": ["q"], "relative_path": ["q.jpg"],
                "session_id": [session], "series_id": ["query-series"],
            }),
            config,
        )

    @staticmethod
    def _write_pool_crops(crops: Path, *, image_id: str = "q",
                          relative_path: str = "q.jpg",
                          session: str = "s",
                          config: dict | None = None) -> dict:
        """写入一份与 mock pool meta 严格绑定的 crop 产物。"""
        crops.mkdir()
        pd.DataFrame({
            "image_id": [image_id],
            "relative_path": [relative_path],
            "session_id": [session],
        }).to_csv(crops / "crops_manifest.csv", index=False)
        (crops / f"{image_id}.jpg").write_bytes(b"jpeg")
        provenance = crop_bundle_provenance(pd.DataFrame({
            "image_id": [image_id],
            "relative_path": [relative_path],
            "session_id": [session],
        }), crops)
        if config is not None:
            config.update(provenance)
        return provenance

    def test_pool_clustering_is_isolated_by_session(self):
        embeddings = np.asarray([
            [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9],
        ], dtype=np.float32)
        sessions = pd.Series(["s1", "s1", "s2", "s2"])
        with patch(
                "whitewhale.pipeline.archival.cluster_embeddings",
                side_effect=[
                    (np.asarray([0, 0]), np.asarray([0.8, 0.7])),
                    (np.asarray([0, 0]), np.asarray([0.9, 0.6])),
                ]) as cluster:
            labels, probabilities = cluster_embeddings_by_session(
                embeddings, sessions, min_cluster_size=2)

        self.assertEqual(labels.tolist(), [0, 0, 1, 1])
        self.assertEqual(probabilities.tolist(), [0.8, 0.7, 0.9, 0.6])
        self.assertEqual(cluster.call_count, 2)

    def test_pool_cluster_gallery_scope_excludes_union_of_nonempty_series(self):
        query = pd.DataFrame({
            "image_id": ["q1", "q2"],
            "session_id": ["s1", "s1"],
            "series_id": ["series-a", "series-b"],
        })
        gallery_info = pd.DataFrame({
            "image_id": ["ga", "gb", "gc", "other"],
            "session_id": ["s1", "s1", "s1", "s2"],
            "series_id": ["series-a", "series-b", "series-c", "series-c"],
        })
        gallery_emb = np.eye(4, dtype=np.float32)
        gallery_ind = [("s1", "A"), ("s1", "B"), ("s1", "C"), ("s2", "D")]

        scoped_emb, scoped_ind, excluded, empty_status = _scoped_gallery(
            query, gallery_emb, gallery_ind, gallery_info, pool_mode=True)

        self.assertEqual(scoped_emb.shape, (1, 4))
        self.assertEqual(scoped_ind, [("s1", "C")])
        self.assertEqual(excluded, 2)
        self.assertIsNone(empty_status)

    def test_pool_no_gallery_has_no_cross_session_score_and_preserves_order(self):
        config = _generated_artifact_config()
        pool_meta = pd.DataFrame({
            "image_id": ["q-s2", "q-s1"],
            "relative_path": ["q-s2.jpg", "q-s1.jpg"],
            "session_id": ["s2", "s1"],
            "series_id": ["other", "shared"],
        })
        pool_emb = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        gallery = (
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            [("s1", "same-series"), ("s1", "different-series")],
            pd.DataFrame({
                "image_id": ["g-same", "g-different"],
                "relative_path": ["g-same.jpg", "g-different.jpg"],
                "session_id": ["s1", "s1"],
                "series_id": ["shared", "different"],
                "confirmed_identity": ["same-series", "different-series"],
            }),
            config,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            crops = root / "pool_crops"
            crops.mkdir()
            pool_meta[["image_id", "relative_path", "session_id"]].to_csv(
                crops / "crops_manifest.csv", index=False)
            for image_id in pool_meta["image_id"]:
                (crops / f"{image_id}.jpg").write_bytes(image_id.encode("ascii"))
            config.update(crop_bundle_provenance(pool_meta, crops))
            args = self._args(root / "batch")
            args.pool_crops = crops
            with patch(
                    "whitewhale.pipeline.archival.load_gallery",
                    return_value=gallery), patch(
                    "whitewhale.pipeline.archival.load_verified_embedding_artifact",
                    return_value=(pool_emb, pool_meta, config)), patch(
                    "whitewhale.pipeline.archival.require_compatible_embedding_configs"), patch(
                    "whitewhale.pipeline.archival.cluster_embeddings_by_session",
                    return_value=(np.asarray([-1, -1]), np.asarray([0.0, 0.0]))):
                run_archival_pipeline(args)

            result = pd.read_csv(args.out / "clusters.csv", keep_default_na=False)

        self.assertEqual(result["image_id"].tolist(), ["q-s2", "q-s1"])
        self.assertEqual(result.loc[0, "status"], "no_gallery")
        self.assertEqual(result.loc[0, "top1"], "")
        self.assertEqual(result.loc[0, "top1_score"], "")
        self.assertEqual(result.loc[1, "top1"], "s1::different-series")
        self.assertEqual(result.loc[1, "excluded_same_series_n"], 1)

    def test_existing_output_is_rejected_without_touching_contents(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "batch"
            out.mkdir()
            marker = out / "marker.txt"
            marker.write_text("keep", encoding="utf-8")
            with patch(
                    "whitewhale.pipeline.archival.load_gallery") as load_gallery_mock:
                with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                    run_archival_pipeline(self._args(out))

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            load_gallery_mock.assert_not_called()

    def test_empty_gallery_meta_is_rejected(self):
        empty = pd.DataFrame(columns=[
            "image_id", "relative_path", "session_id", "confirmed_identity"])
        with patch(
                "whitewhale.pipeline.archival.load_verified_embedding_artifact",
                return_value=(np.zeros((0, 2)), empty,
                              _generated_artifact_config())):
            with self.assertRaisesRegex(ValueError, "gallery meta 为空"):
                load_archival_gallery(Path("gallery.npy"), Path("gallery.csv"))

    def test_incompatible_artifacts_create_no_formal_batch(self):
        config = _generated_artifact_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "batch"
            with patch(
                    "whitewhale.pipeline.archival.load_gallery",
                    return_value=self._gallery(config)), patch(
                    "whitewhale.pipeline.archival.load_verified_embedding_artifact",
                    return_value=self._pool(config)), patch(
                    "whitewhale.pipeline.archival.require_compatible_embedding_configs",
                    side_effect=ValueError("配置不兼容")), self.assertRaisesRegex(
                        ValueError, "不兼容"):
                run_archival_pipeline(self._args(out))

            self.assertFalse(out.exists())
            self.assertEqual(list(Path(tmp_dir).glob(".batch.staging-*")), [])

    def test_empty_session_fails_without_formal_batch(self):
        config = _generated_artifact_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "batch"
            with patch(
                    "whitewhale.pipeline.archival.load_gallery",
                    return_value=self._gallery(config)), patch(
                    "whitewhale.pipeline.archival.load_verified_embedding_artifact",
                    return_value=self._pool(config, session="")), patch(
                    "whitewhale.pipeline.archival.require_compatible_embedding_configs"):
                with self.assertRaisesRegex(ValueError, "空 session_id"):
                    run_archival_pipeline(self._args(out))

            self.assertFalse(out.exists())
            self.assertEqual(
                len(list(Path(tmp_dir).glob(".batch.staging-*"))), 0)

    def test_success_publishes_new_directory_without_stale_cluster_matches(self):
        config = _generated_artifact_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            previous = root / "previous"
            previous.mkdir()
            marker = previous / "marker.txt"
            marker.write_text("keep", encoding="utf-8")
            out = root / "batch"
            args = self._args(out)
            args.pool_crops = root / "pool_crops"
            self._write_pool_crops(args.pool_crops, config=config)
            with patch(
                    "whitewhale.pipeline.archival.load_gallery",
                    return_value=self._gallery(config)), patch(
                    "whitewhale.pipeline.archival.load_verified_embedding_artifact",
                    return_value=self._pool(config)), patch(
                    "whitewhale.pipeline.archival.require_compatible_embedding_configs"):
                run_archival_pipeline(args)

            self.assertTrue((out / "clusters.csv").is_file())
            self.assertFalse((out / "cluster_matches.csv").exists())
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertEqual(list(root.glob(".batch.staging-*")), [])

    def test_pool_crops_require_exact_row_binding_and_safe_basename(self):
        meta = pd.DataFrame({
            "image_id": ["q"], "relative_path": ["q.jpg"],
            "session_id": ["s"],
        })
        with tempfile.TemporaryDirectory() as tmp_dir:
            crops = Path(tmp_dir) / "crops"
            config = self._write_pool_crops(crops)
            _validate_crop_artifact(meta, crops, "pool", config)

            (crops / "q.jpg").write_bytes(b"tampered-jpeg")
            with self.assertRaisesRegex(ValueError, "crop 内容摘要不一致"):
                _validate_crop_artifact(meta, crops, "pool", config)
            (crops / "q.jpg").write_bytes(b"jpeg")

            missing_digest = dict(config)
            missing_digest.pop("crop_bundle_sha256")
            with self.assertRaisesRegex(ValueError, "缺少可信 crop 摘要字段"):
                _validate_crop_artifact(meta, crops, "pool", missing_digest)

            crop_manifest = pd.read_csv(crops / "crops_manifest.csv")
            crop_manifest.loc[0, "image_id"] = "other"
            crop_manifest.to_csv(crops / "crops_manifest.csv", index=False)
            with self.assertRaisesRegex(ValueError, "行绑定不一致"):
                _validate_crop_artifact(meta, crops, "pool", config)

        unsafe = meta.assign(image_id="../q")
        with tempfile.TemporaryDirectory() as tmp_dir, self.assertRaisesRegex(
                ValueError, "不安全的 image_id"):
            _validate_crop_artifact(unsafe, Path(tmp_dir), "pool", {})

    def test_missing_pool_crop_refuses_before_formal_batch(self):
        config = _generated_artifact_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out = root / "batch"
            args = self._args(out)
            args.pool_crops = root / "pool_crops"
            args.pool_crops.mkdir()
            pd.DataFrame({
                "image_id": ["q"], "relative_path": ["q.jpg"],
                "session_id": ["s"],
            }).to_csv(args.pool_crops / "crops_manifest.csv", index=False)
            with patch(
                    "whitewhale.pipeline.archival.load_gallery",
                    return_value=self._gallery(config)), patch(
                    "whitewhale.pipeline.archival.load_verified_embedding_artifact",
                    return_value=self._pool(config)), patch(
                    "whitewhale.pipeline.archival.require_compatible_embedding_configs"), \
                    self.assertRaisesRegex(ValueError, "crops 内容集"):
                run_archival_pipeline(args)

            self.assertFalse(out.exists())

    def test_empty_meta_relative_escape_and_invalid_parameters_are_rejected(self):
        config = _generated_artifact_config()
        cases = (
            (self._pool(config)[0], self._pool(config)[1].iloc[0:0], config),
            (self._pool(config)[0], self._pool(config)[1].assign(
                relative_path="../q.jpg"), config),
        )
        for index, pool in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp_dir:
                out = Path(tmp_dir) / "batch"
                with patch(
                        "whitewhale.pipeline.archival.load_gallery",
                        return_value=self._gallery(config)), patch(
                        "whitewhale.pipeline.archival.load_verified_embedding_artifact",
                        return_value=pool), patch(
                        "whitewhale.pipeline.archival.require_compatible_embedding_configs"), \
                        self.assertRaises(ValueError):
                    run_archival_pipeline(self._args(out))
                self.assertFalse(out.exists())

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._args(Path(tmp_dir) / "batch")
            args.topk = 0
            with patch(
                    "whitewhale.pipeline.archival.load_gallery") as gallery_loader, \
                    self.assertRaisesRegex(ValueError, "topk"):
                run_archival_pipeline(args)
            gallery_loader.assert_not_called()


class TestCrossTimeConfig(unittest.TestCase):
    """跨时间入口复用 pipeline.yaml，不再维护第二套推理参数。"""

    def test_batch_args_follow_pipeline_config(self):
        cfg = load_config("pipeline")

        def publish_batch(args):
            args.out.mkdir(parents=True)
            (args.out / "representatives").mkdir()
            args.input_manifest_data.to_csv(
                args.out / args.input_manifest_snapshot, index=False)
            pd.DataFrame({
                "image_id": ["q"], "relative_path": ["session/q.jpg"],
                "session_id": ["session"],
            }).to_csv(
                args.out / "clusters.csv", index=False)
            (args.out / "summary.json").write_text(
                json.dumps({"n_images": 1}), encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"
            out_root = output_root / "cross_time"
            manifest = pd.DataFrame({
                "image_id": ["q"], "relative_path": ["q.jpg"],
                "session_id": ["session"], "label_status": ["labeled"],
            })
            with patch.multiple(
                    cross_time, OUTPUT_ROOT=output_root, OUT_ROOT=out_root), patch.object(
                        cross_time, "run_archival", side_effect=publish_batch) as run_archival:
                cross_time.run_batch("session", manifest)

        args = run_archival.call_args.args[0]
        self.assertEqual(args.det_device, cfg["detector"]["device"])
        self.assertEqual(args.det_conf, cfg["detector"]["conf"])
        self.assertEqual(args.det_imgsz, cfg["detector"]["imgsz"])
        self.assertEqual(
            args.threshold_cluster,
            cfg["retrieval"]["threshold_cluster"])
        self.assertEqual(
            args.min_cluster_size,
            cfg["clustering"]["min_cluster_size"])
        self.assertEqual(args.det_pad_x, cfg["crop"]["pad_x"])
        self.assertEqual(args.sheets, cfg["cross_time"]["sheets"])
        output_root = ROOT / cfg["output_root"]
        self.assertEqual(
            cross_time.GALLERY_CROPS,
            output_root / cfg["cross_time"]["gallery_crops"])
        self.assertEqual(
            args.gallery_embeddings,
            output_root / cfg["query"]["embeddings"])
        self.assertEqual(
            args.gallery_meta, output_root / cfg["query"]["meta"])

    def test_active_gallery_validation_is_read_only(self):
        config = _generated_artifact_config()
        info = pd.DataFrame({
            "image_id": [f"g{index}" for index, _ in enumerate(
                cross_time.GALLERY_SESSIONS)],
            "session_id": cross_time.GALLERY_SESSIONS,
        })
        with patch.object(
                cross_time, "load_gallery",
                return_value=(np.zeros((1, 2)), [("s", "i")], info, config)
        ) as load_gallery_mock, patch.object(
                cross_time, "_validate_crop_artifact") as validate_crops, patch.object(
                cross_time, "_expected_runtime_embedding_config",
                return_value=config), patch.object(
                cross_time, "require_compatible_embedding_configs") as compatible:
            self.assertIs(cross_time.validate_active_gallery(), config)

        load_gallery_mock.assert_called_once_with(
            cross_time.GAL_NPY, cross_time.GAL_META)
        validate_crops.assert_called_once_with(
            info, cross_time.GALLERY_CROPS, "gallery", config)
        compatible.assert_called_once()

    def test_active_gallery_requires_exact_sessions_and_excludes_targets(self):
        config = _generated_artifact_config()
        bad_info = pd.DataFrame({
            "image_id": ["g"], "session_id": [cross_time.GALLERY_SESSIONS[0]],
        })
        with patch.object(
                cross_time, "load_gallery",
                return_value=(np.zeros((1, 2)), [("s", "i")], bad_info, config)), \
                self.assertRaisesRegex(ValueError, "session 集合"):
            cross_time.validate_active_gallery()

        good_info = pd.DataFrame({
            "image_id": [f"g{index}" for index, _ in enumerate(
                cross_time.GALLERY_SESSIONS)],
            "session_id": cross_time.GALLERY_SESSIONS,
        })
        with patch.object(
                cross_time, "load_gallery",
                return_value=(np.zeros((2, 2)), [("s", "i")], good_info, config)), \
                self.assertRaisesRegex(ValueError, "目标 session"):
            cross_time.validate_active_gallery(
                [cross_time.GALLERY_SESSIONS[0]])

    def test_active_gallery_rejects_runtime_model_or_crop_mismatch(self):
        expected = {
            "model": "megadescriptor-metric-learning-r4",
            "crop": "yolo",
            "preprocess": "Resize256+CenterCrop224",
            "checkpoint_sha256": "checkpoint-current",
            "crop_schema_version": 1,
            "detector_checkpoint_sha256": "detector-current",
            "detector_fallback_policy": "center",
            "detector_conf": 0.25,
            "detector_imgsz": 1024,
            "detector_pad_x": 0.3,
            "detector_pad_up": 0.15,
            "detector_pad_down": 0.6,
        }
        stale = dict(expected, checkpoint_sha256="checkpoint-stale")
        info = pd.DataFrame({
            "image_id": [f"g{index}" for index, _ in enumerate(
                cross_time.GALLERY_SESSIONS)],
            "session_id": cross_time.GALLERY_SESSIONS,
        })
        with patch.object(
                cross_time, "load_gallery",
                return_value=(np.zeros((2, 2)), [("s", "i")], info, stale)), \
                patch.object(cross_time, "_validate_crop_artifact"), patch.object(
                    cross_time, "_expected_runtime_embedding_config",
                    return_value=expected), self.assertRaisesRegex(
                    ValueError, "权重"):
            cross_time.validate_active_gallery()

    @staticmethod
    def _manifest_with_target() -> pd.DataFrame:
        sessions = [*cross_time.GALLERY_SESSIONS, "20140419 02"]
        return pd.DataFrame({
            "image_id": [f"i{index}" for index in range(len(sessions))],
            "relative_path": [f"p{index}.jpg" for index in range(len(sessions))],
            "session_id": sessions,
            "label_status": ["labeled"] * len(sessions),
        })

    def test_session_must_exist_and_be_windows_safe_leaf(self):
        manifest = self._manifest_with_target()
        self.assertEqual(
            cross_time.validate_sessions(["20140419 02"], manifest),
            ["20140419 02"],
        )
        for session in ("", "../x", r"..\x", "C:drive", "CON", "missing"):
            with self.subTest(session=session), self.assertRaises(ValueError):
                cross_time.validate_sessions([session], manifest)
        malformed = manifest.copy()
        malformed.loc[malformed["session_id"] == "20140419 02", "session_id"] = (
            " 20140419 02 ")
        with self.assertRaises(ValueError):
            cross_time.validate_sessions(["20140419 02"], malformed)

    def test_session_manifest_and_output_remain_contained(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"
            out_root = output_root / "cross_time"
            with patch.multiple(
                    cross_time, OUTPUT_ROOT=output_root, OUT_ROOT=out_root):
                manifest_path, output_path = cross_time._session_paths("20140419 02")

        self.assertEqual(manifest_path.parent, (out_root / "manifests").resolve())
        self.assertEqual(output_path.parent, out_root.resolve())

    def test_active_gallery_rebuild_api_always_refuses_overwrite(self):
        for kwargs in ({}, {"overwrite": True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                    RuntimeError, "rebuild_r4_artifacts.py --out"):
                cross_time.build_gallery(**kwargs)

    def test_invalid_target_is_rejected_before_gallery_build(self):
        manifest = self._manifest_with_target()
        for session in ("../x", "missing"):
            argv = ["run_cross_time_batch.py", "--sessions", session]
            with self.subTest(session=session), patch.object(
                    sys, "argv", argv), patch.object(
                        cross_time, "read_metadata_csv", return_value=manifest), patch.object(
                        cross_time, "validate_active_gallery") as validate_gallery, self.assertRaises(
                        SystemExit):
                cross_time.main()
            validate_gallery.assert_not_called()

    def test_empty_session_selection_is_rejected_before_gallery_build(self):
        argv = ["run_cross_time_batch.py", "--sessions"]
        with patch.object(sys, "argv", argv), patch.object(
                cross_time, "validate_active_gallery") as validate_gallery, self.assertRaises(SystemExit):
            cross_time.main()
        validate_gallery.assert_not_called()

    def test_rebuild_gallery_is_rejected_with_versioned_command_hint(self):
        argv = ["run_cross_time_batch.py", "--only-gallery", "--rebuild-gallery"]
        with patch.object(sys, "argv", argv), patch.object(
                cross_time, "validate_active_gallery") as validate_gallery:
            with self.assertRaisesRegex(SystemExit, "2"), patch(
                    "sys.stderr") as stderr:
                cross_time.main()

        validate_gallery.assert_not_called()
        stderr.write.assert_called()
        self.assertIn(
            "scripts/rebuild_r4_artifacts.py --out",
            stderr.write.call_args.args[0])

    def test_only_gallery_only_validates_active_artifact(self):
        argv = ["run_cross_time_batch.py", "--only-gallery"]
        with patch.object(sys, "argv", argv), patch.object(
                cross_time, "validate_active_gallery") as validate_gallery, patch.object(
                cross_time, "read_metadata_csv") as read_manifest:
            cross_time.main()

        validate_gallery.assert_called_once_with()
        read_manifest.assert_not_called()

    def test_existing_session_result_is_rejected_before_manifest_write(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"
            out_root = output_root / "cross_time"
            session = "20140419 02"
            (out_root / session).mkdir(parents=True)
            manifest = pd.DataFrame({
                "image_id": ["q"], "relative_path": ["q.jpg"],
                "session_id": [session], "label_status": ["labeled"],
            })
            with patch.multiple(
                    cross_time, OUTPUT_ROOT=output_root, OUT_ROOT=out_root), patch.object(
                        cross_time, "run_archival") as run_archival, self.assertRaisesRegex(
                            FileExistsError, "拒绝覆盖"):
                cross_time.run_batch(session, manifest)

            self.assertFalse((out_root / "manifests" / f"{session}.csv").exists())
            run_archival.assert_not_called()

    def test_failed_batch_does_not_replace_active_query_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"
            out_root = output_root / "cross_time"
            session = "20140419 02"
            active = out_root / "manifests" / f"{session}.csv"
            active.parent.mkdir(parents=True)
            active.write_text("old-active-manifest", encoding="utf-8")
            manifest = pd.DataFrame({
                "image_id": ["q"], "relative_path": ["q.jpg"],
                "session_id": [session], "label_status": ["labeled"],
            })
            with patch.multiple(
                    cross_time, OUTPUT_ROOT=output_root, OUT_ROOT=out_root), patch.object(
                        cross_time, "run_archival", side_effect=RuntimeError("boom")), self.assertRaisesRegex(
                            RuntimeError, "boom"):
                cross_time.run_batch(session, manifest)

            self.assertEqual(active.read_text(encoding="utf-8"), "old-active-manifest")
            self.assertFalse((out_root / session).exists())
            self.assertEqual(
                list(active.parent.glob(f".{session}.staging-*.csv")), [])

    def test_manifest_publish_failure_leaves_recoverable_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"
            out_root = output_root / "cross_time"
            session = "20140419 02"
            active = out_root / "manifests" / f"{session}.csv"
            active.parent.mkdir(parents=True)
            active.write_text("old-active-manifest", encoding="utf-8")
            manifest = pd.DataFrame({
                "image_id": ["q"], "relative_path": ["q.jpg"],
                "session_id": [session], "label_status": ["labeled"],
            })

            def publish_batch(args):
                args.out.mkdir(parents=True)
                (args.out / "representatives").mkdir()
                args.input_manifest_data.to_csv(
                    args.out / args.input_manifest_snapshot, index=False)
                pd.DataFrame({
                    "image_id": ["q"],
                    "relative_path": [f"{session}/q.jpg"],
                    "session_id": [session],
                }).to_csv(
                    args.out / "clusters.csv", index=False)
                (args.out / "summary.json").write_text(
                    json.dumps({"n_images": 1}), encoding="utf-8")

            with patch.multiple(
                    cross_time, OUTPUT_ROOT=output_root, OUT_ROOT=out_root), patch.object(
                        cross_time, "run_archival", side_effect=publish_batch), patch.object(
                        cross_time.os, "replace", side_effect=OSError("publish failed")), \
                    self.assertRaisesRegex(OSError, "publish failed"):
                cross_time.run_batch(session, manifest)

            self.assertEqual(active.read_text(encoding="utf-8"), "old-active-manifest")
            self.assertTrue((out_root / session).is_dir())
            self.assertFalse(
                (out_root / session / cross_time.COMMIT_MARKER_NAME).exists())

            with patch.multiple(
                    cross_time, OUTPUT_ROOT=output_root, OUT_ROOT=out_root), patch.object(
                        cross_time, "run_archival") as run_archival:
                cross_time.run_batch(session, manifest)

            run_archival.assert_not_called()
            recovered = cross_time.read_metadata_csv(active)
            self.assertEqual(recovered["image_id"].tolist(), ["q"])
            self.assertTrue(
                (out_root / session / cross_time.COMMIT_MARKER_NAME).is_file())

    def test_crash_after_manifest_publish_is_completed_on_next_start(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"
            out_root = output_root / "cross_time"
            session = "20140419 02"
            manifest = pd.DataFrame({
                "image_id": ["q"], "relative_path": ["q.jpg"],
                "session_id": [session], "label_status": ["labeled"],
            })

            def publish_batch(args):
                args.out.mkdir(parents=True)
                (args.out / "representatives").mkdir()
                args.input_manifest_data.to_csv(
                    args.out / args.input_manifest_snapshot, index=False)
                pd.DataFrame({
                    "image_id": ["q"],
                    "relative_path": [f"{session}/q.jpg"],
                    "session_id": [session],
                }).to_csv(
                    args.out / "clusters.csv", index=False)
                (args.out / "summary.json").write_text(
                    json.dumps({"n_images": 1}), encoding="utf-8")

            with patch.multiple(
                    cross_time, OUTPUT_ROOT=output_root, OUT_ROOT=out_root), patch.object(
                        cross_time, "run_archival", side_effect=publish_batch), patch.object(
                        cross_time, "_write_commit_marker",
                        side_effect=OSError("crash before marker")), self.assertRaisesRegex(
                        OSError, "crash before marker"):
                cross_time.run_batch(session, manifest)

            active = out_root / "manifests" / f"{session}.csv"
            self.assertTrue(active.is_file())
            self.assertFalse(
                (out_root / session / cross_time.COMMIT_MARKER_NAME).exists())

            with patch.multiple(
                    cross_time, OUTPUT_ROOT=output_root, OUT_ROOT=out_root), patch.object(
                        cross_time, "run_archival") as run_archival:
                cross_time.run_batch(session, manifest)

            run_archival.assert_not_called()
            self.assertTrue(
                (out_root / session / cross_time.COMMIT_MARKER_NAME).is_file())

    def test_commit_marker_binds_representative_bytes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"
            out_root = output_root / "cross_time"
            session = "20140419 02"
            manifest = pd.DataFrame({
                "image_id": ["q"], "relative_path": ["q.jpg"],
                "session_id": [session], "label_status": ["labeled"],
            })

            def publish_batch(args):
                args.out.mkdir(parents=True)
                representatives = args.out / "representatives"
                representatives.mkdir()
                (representatives / "cluster_000_sub0.jpg").write_bytes(b"original")
                args.input_manifest_data.to_csv(
                    args.out / args.input_manifest_snapshot, index=False)
                pd.DataFrame({
                    "image_id": ["q"],
                    "relative_path": [f"{session}/q.jpg"],
                    "session_id": [session],
                }).to_csv(args.out / "clusters.csv", index=False)
                (args.out / "summary.json").write_text(
                    json.dumps({"n_images": 1}), encoding="utf-8")

            with patch.multiple(
                    cross_time, OUTPUT_ROOT=output_root, OUT_ROOT=out_root), patch.object(
                        cross_time, "run_archival", side_effect=publish_batch) as run_archival:
                cross_time.run_batch(session, manifest)
                marker = json.loads((
                    out_root / session / cross_time.COMMIT_MARKER_NAME
                ).read_text(encoding="utf-8"))
                self.assertEqual(
                    marker["schema_version"],
                    cross_time.BATCH_COMMIT_SCHEMA_VERSION)
                self.assertIn("representatives", marker["output_digests"])

                (out_root / session / "representatives" /
                 "cluster_000_sub0.jpg").write_bytes(b"tampered")
                cross_time.run_batch(session, manifest)

            self.assertEqual(run_archival.call_count, 2)
            quarantined = list(out_root.glob(f".{session}.incomplete-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(
                (quarantined[0] / "representatives" /
                 "cluster_000_sub0.jpg").read_bytes(), b"tampered")
            self.assertTrue(
                (out_root / session / cross_time.COMMIT_MARKER_NAME).is_file())

    def test_marker_write_failure_isolates_tmp_without_deleting_it(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch = Path(tmp_dir) / "batch"
            batch.mkdir()
            with patch.object(
                    cross_time.os, "replace", side_effect=OSError("marker failed")), \
                    self.assertRaisesRegex(OSError, "marker failed"):
                cross_time._write_commit_marker(
                    "s", batch, "manifest-hash", {"output": "hash"})

            self.assertFalse(
                (batch / cross_time.COMMIT_MARKER_NAME).exists())
            isolated = list(
                batch.parent.glob(".orphan-marker-write-failed-*"))
            self.assertEqual(len(isolated), 1)
            self.assertIn("manifest-hash", isolated[0].read_text(encoding="utf-8"))

    def test_restart_isolates_publish_staging_and_marker_tmp(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_root = Path(tmp_dir) / "cross_time"
            manifests = out_root / "manifests"
            manifests.mkdir(parents=True)
            session = "20140419 02"
            active = manifests / f"{session}.csv"
            batch = out_root / session
            batch.mkdir()
            leftovers = [
                manifests / f".{session}.staging-old.csv",
                manifests / f".{session}.publish-old.csv",
                out_root / f".{session}.staging-old",
                batch / f".{cross_time.COMMIT_MARKER_NAME}.tmp-old",
            ]
            leftovers[0].write_text("staging", encoding="utf-8")
            leftovers[1].write_text("publish", encoding="utf-8")
            leftovers[2].mkdir()
            leftovers[3].write_text("marker", encoding="utf-8")

            isolated = cross_time._isolate_restart_leftovers(
                session, active, batch)

            self.assertEqual(len(isolated), 5)
            for path in leftovers:
                self.assertFalse(path.exists())
            self.assertFalse(batch.exists())
            self.assertGreaterEqual(
                len(list(out_root.glob(".orphan-*"))), 2)

    def test_legacy_batch_without_snapshot_or_marker_is_preserved_read_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "outputs"
            out_root = output_root / "cross_time"
            session = "20140419 02"
            broken = out_root / session
            broken.mkdir(parents=True)
            (broken / "summary.json").write_text("{}", encoding="utf-8")
            (broken / "clusters.csv").write_text("broken", encoding="utf-8")
            manifest = pd.DataFrame({
                "image_id": ["q"], "relative_path": ["q.jpg"],
                "session_id": [session], "label_status": ["labeled"],
            })

            with patch.multiple(
                    cross_time, OUTPUT_ROOT=output_root, OUT_ROOT=out_root), patch.object(
                        cross_time, "run_archival") as run_archival, self.assertRaisesRegex(
                            FileExistsError, "旧版结果.*只读结果保留"):
                cross_time.run_batch(session, manifest)

            quarantined = list(out_root.glob(f".{session}.incomplete-*"))
            self.assertEqual(len(quarantined), 0)
            self.assertEqual(
                (broken / "summary.json").read_text(encoding="utf-8"), "{}")
            self.assertEqual(
                (broken / "clusters.csv").read_text(encoding="utf-8"), "broken")
            run_archival.assert_not_called()

    def test_non_object_commit_marker_is_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            session = "20140419 02"
            batch = root / session
            batch.mkdir()
            (batch / "representatives").mkdir()
            pd.DataFrame({
                "image_id": ["q"],
                "relative_path": [f"{session}/q.jpg"],
                "session_id": [session],
            }).to_csv(batch / cross_time.BATCH_MANIFEST_NAME, index=False)
            pd.DataFrame({
                "image_id": ["q"],
                "relative_path": [f"{session}/q.jpg"],
                "session_id": [session],
            }).to_csv(batch / "clusters.csv", index=False)
            (batch / "summary.json").write_text(
                json.dumps({"n_images": 1}), encoding="utf-8")
            (batch / cross_time.COMMIT_MARKER_NAME).write_text(
                "[]", encoding="utf-8")

            recovered = cross_time._recover_existing_batch(
                session, root / "manifests" / f"{session}.csv", batch)

            self.assertFalse(recovered)
            self.assertFalse(batch.exists())
            quarantined = list(root.glob(f".{session}.incomplete-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(
                (quarantined[0] / cross_time.COMMIT_MARKER_NAME).read_text(
                    encoding="utf-8"),
                "[]",
            )

    def test_cross_time_identifier_reader_preserves_leading_zeroes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "identifiers.csv"
            pd.DataFrame({
                "image_id": ["001"],
                "session_id": ["01"],
                "individual_id": ["05"],
                "confirmed_identity": ["007"],
            }).to_csv(csv_path, index=False)
            loaded = cross_time.read_metadata_csv(csv_path)

        self.assertEqual(loaded.loc[0, "image_id"], "001")
        self.assertEqual(loaded.loc[0, "session_id"], "01")
        self.assertEqual(loaded.loc[0, "individual_id"], "05")
        self.assertEqual(loaded.loc[0, "confirmed_identity"], "007")


if __name__ == "__main__":
    unittest.main()
