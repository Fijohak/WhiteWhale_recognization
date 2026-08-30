"""
人工审核多人投票数据流测试。

覆盖旧版单人 CSV 兼容、不同审核人互不覆盖、保守投票裁决以及
页面只加载当前审核人判断的盲审约束。
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.review.app import (  # noqa: E402
    _resolve_history_path,
    build_app,
    export_confirmed,
    load_annotation_records,
    load_annotations,
    load_embeddings,
    load_photos,
    normalize_reviewer_id,
    normalize_reviewer_roster,
    save_annotations,
    summarize_annotations,
)
from whitewhale.data.manifest import compute_sha256  # noqa: E402
from whitewhale.review import app as review_app_module  # noqa: E402


REVIEWER_ROSTER = ("alice", "bob", "carol")


def _photos() -> pd.DataFrame:
    """构造不依赖真实图片的最小审核清单。"""
    return pd.DataFrame({
        "image_id": ["img1", "img2", "img3"],
        "relative_path": ["a.jpg", "b.jpg", "c.jpg"],
        "cluster": [1, 1, 2],
        "subcluster": [0, 0, 0],
        "individual_id": ["", "", ""],
        "session_id": ["s1", "s1", "s1"],
    })


def _write_embedding_artifact(root: Path, image_ids: list[str], *,
                              digest_ids: list[str] | None = None,
                              values: np.ndarray | None = None):
    """写入最小生成期 embedding 三件套。"""
    embeddings = root / "features.npy"
    meta = root / "features_meta.csv"
    config = root / "features_config.json"
    values = (np.eye(len(image_ids), dtype=np.float32) if values is None
              else np.asarray(values, dtype=np.float32))
    if values.ndim != 2 or len(values) != len(image_ids):
        raise ValueError("测试特征 shape 与 image_ids 不一致")
    np.save(embeddings, values)
    pd.DataFrame({"image_id": image_ids}).to_csv(meta, index=False)
    ordered = digest_ids if digest_ids is not None else image_ids
    config.write_text(json.dumps({
        "embedding_file": embeddings.name,
        "meta_file": meta.name,
        "embedding_sha256": compute_sha256(embeddings),
        "meta_sha256": compute_sha256(meta),
        "n": len(image_ids),
        "feat_dim": values.shape[1],
        "artifact_schema_version": 2,
        "provenance_level": "generated_with_row_binding",
        "created_at_utc": "2026-08-29T00:00:00+00:00",
        "row_binding": "embedding_row_i_to_meta_image_id_i",
        "ordered_image_ids_sha256": hashlib.sha256(
            "\n".join(ordered).encode("utf-8")).hexdigest(),
    }), encoding="utf-8")
    return embeddings, meta


class TestPhotoManifestLoading(unittest.TestCase):
    """审核清单必须保留字符串标识并拒绝模糊主键。"""

    def test_string_ids_and_literal_na_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clusters.csv"
            path.write_text(
                "image_id,relative_path\n0001,NA\n", encoding="utf-8")

            photos = load_photos(path)

            self.assertEqual(photos.loc[0, "image_id"], "0001")
            self.assertEqual(photos.loc[0, "relative_path"], "NA")

    def test_missing_required_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clusters.csv"
            path.write_text("image_id\nimg1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "relative_path"):
                load_photos(path)

    def test_duplicate_or_blank_image_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate = root / "duplicate.csv"
            duplicate.write_text(
                "image_id,relative_path\nimg1,a.jpg\nIMG1,b.jpg\n",
                encoding="utf-8")
            blank = root / "blank.csv"
            blank.write_text(
                "image_id,relative_path\n ,a.jpg\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "重复"):
                load_photos(duplicate)
            with self.assertRaisesRegex(ValueError, "空值"):
                load_photos(blank)


class TestAnnotationStorage(unittest.TestCase):
    """每位审核人独立保存一票，不丢失其他人的判断。"""

    def test_legacy_csv_preserves_leading_zero_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            path.write_text("image_id,label\nimg1,0001\n", encoding="utf-8")

            self.assertEqual(load_annotations(path), {"img1": "0001"})
            records = load_annotation_records(path)
            self.assertEqual(records.loc[0, "reviewer"], "")
            self.assertEqual(records.loc[0, "reviewed_at"], "")
            save_annotations(path, _photos(), {"img1": "0001"}, reviewer="")
            self.assertEqual(load_annotation_records(path).loc[0, "reviewed_at"], "")

    def test_reviewers_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            photos = _photos()
            save_annotations(
                path, photos, {"img1": "CI-001"}, reviewer="alice",
                reviewer_roster=REVIEWER_ROSTER)
            save_annotations(path, photos, {"img1": "reject", "img2": "CI-002"},
                             reviewer="bob", reviewer_roster=REVIEWER_ROSTER)

            self.assertEqual(load_annotations(path, "alice"), {"img1": "CI-001"})
            self.assertEqual(load_annotations(path, "bob"),
                             {"img1": "reject", "img2": "CI-002"})
            records = load_annotation_records(path)
            self.assertEqual(len(records), 3)
            self.assertTrue((records["reviewed_at"] != "").all())

            # Alice 清空自己的票后，Bob 的两票仍完整保留。
            save_annotations(
                path, photos, {}, reviewer="alice",
                reviewer_roster=REVIEWER_ROSTER)
            self.assertEqual(load_annotations(path, "alice"), {})
            self.assertEqual(len(load_annotation_records(path)), 2)

    def test_existing_unlocked_lock_file_is_reused_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            lock_path = path.with_name(path.name + ".lock")
            lock_path.write_text("abandoned", encoding="utf-8")

            save_annotations(
                path, _photos(), {"img1": "CI-001"}, reviewer="alice",
                reviewer_roster=REVIEWER_ROSTER)

            self.assertTrue(lock_path.exists())
            self.assertEqual(load_annotations(path, "alice"), {"img1": "CI-001"})

    def test_advisory_lock_serializes_concurrent_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            lock_path = path.with_name(path.name + ".lock")
            first_entered = threading.Event()
            release_first = threading.Event()
            second_entered = threading.Event()
            errors = []

            def first_worker():
                try:
                    with review_app_module._annotation_file_lock(path):
                        first_entered.set()
                        if not release_first.wait(timeout=3):
                            raise TimeoutError("test did not release first lock")
                except Exception as exc:  # pragma: no cover - 仅收集线程异常
                    errors.append(exc)

            def second_worker():
                try:
                    with review_app_module._annotation_file_lock(path):
                        second_entered.set()
                except Exception as exc:  # pragma: no cover - 仅收集线程异常
                    errors.append(exc)

            first = threading.Thread(target=first_worker)
            second = threading.Thread(target=second_worker)
            first.start()
            self.assertTrue(first_entered.wait(timeout=2))
            second.start()
            self.assertFalse(second_entered.wait(timeout=0.15))
            release_first.set()
            first.join(timeout=3)
            second.join(timeout=3)

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(second_entered.is_set())
            self.assertTrue(lock_path.exists())

    def test_advisory_lock_timeout_does_not_break_active_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            lock_path = path.with_name(path.name + ".lock")
            entered = threading.Event()
            release = threading.Event()

            def holder():
                with review_app_module._annotation_file_lock(path):
                    entered.set()
                    release.wait(timeout=3)

            thread = threading.Thread(target=holder)
            thread.start()
            self.assertTrue(entered.wait(timeout=2))
            try:
                with mock.patch.object(
                        review_app_module,
                        "ANNOTATION_LOCK_TIMEOUT_SECONDS", 0.1):
                    with self.assertRaisesRegex(TimeoutError, "写入锁超时"):
                        with review_app_module._annotation_file_lock(path):
                            pass
                self.assertTrue(lock_path.exists())
            finally:
                release.set()
                thread.join(timeout=3)
            self.assertFalse(thread.is_alive())

    def test_reviewer_roster_is_fixed_in_adjacent_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            photos = _photos()
            save_annotations(
                path, photos, {"img1": "CI-001"}, reviewer="alice",
                reviewer_roster=REVIEWER_ROSTER)

            sidecar = path.with_name(path.name + ".reviewer_roster.json")
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["canonical_reviewer_roster"],
                ["alice", "bob", "carol"],
            )
            self.assertEqual(len(payload["fingerprint_sha256"]), 64)

            # 配置顺序不影响同一名单；成员变化则在覆盖原始票前失败。
            save_annotations(
                path, photos, {"img1": "CI-001"}, reviewer="bob",
                reviewer_roster=("carol", "alice", "bob"))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "已固定名单不匹配"):
                save_annotations(
                    path, photos, {"img1": "changed"}, reviewer="alice",
                    reviewer_roster=("alice", "bob", "dave"))
            self.assertEqual(path.read_bytes(), before)

    def test_corrupt_reviewer_roster_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            save_annotations(
                path, _photos(), {"img1": "CI-001"}, reviewer="alice",
                reviewer_roster=REVIEWER_ROSTER)
            sidecar = path.with_name(path.name + ".reviewer_roster.json")
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            payload["fingerprint_sha256"] = "0" * 64
            sidecar.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "内容或指纹无效"):
                save_annotations(
                    path, _photos(), {"img1": "changed"}, reviewer="alice",
                    reviewer_roster=REVIEWER_ROSTER)

    def test_missing_roster_sidecar_with_named_votes_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            save_annotations(
                path, _photos(), {"img1": "CI-001"}, reviewer="alice",
                reviewer_roster=REVIEWER_ROSTER)
            sidecar = path.with_name(path.name + ".reviewer_roster.json")
            sidecar.unlink()
            before = path.read_bytes()

            for roster in (
                    REVIEWER_ROSTER, ("alice", "bob", "dave")):
                with self.subTest(roster=roster):
                    with self.assertRaisesRegex(
                            ValueError, "sidecar 缺失.*命名审核票"):
                        save_annotations(
                            path, _photos(), {"img1": "changed"},
                            reviewer="alice", reviewer_roster=roster)

            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(sidecar.exists())

    def test_anonymous_legacy_votes_allow_first_roster_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            path.write_text(
                "image_id,label\nimg1,0001\n", encoding="utf-8")

            save_annotations(
                path, _photos(), {"img2": "CI-002"}, reviewer="alice",
                reviewer_roster=REVIEWER_ROSTER)

            sidecar = path.with_name(path.name + ".reviewer_roster.json")
            self.assertTrue(sidecar.exists())
            records = load_annotation_records(path)
            self.assertEqual(set(records["reviewer"]), {"", "alice"})

    def test_anonymous_write_cannot_bypass_roster_integrity(self):
        for sidecar_state in ("valid", "missing", "corrupt"):
            with self.subTest(sidecar_state=sidecar_state):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "annotations.csv"
                    save_annotations(
                        path, _photos(), {"img1": "CI-001"},
                        reviewer="alice", reviewer_roster=REVIEWER_ROSTER)
                    sidecar = path.with_name(
                        path.name + ".reviewer_roster.json")
                    if sidecar_state == "missing":
                        sidecar.unlink()
                    elif sidecar_state == "corrupt":
                        sidecar.write_text("{broken", encoding="utf-8")
                    before = path.read_bytes()

                    with self.assertRaisesRegex(
                            ValueError, "匿名 reviewer 写入仅用于"):
                        save_annotations(
                            path, _photos(), {"img2": "legacy-vote"},
                            reviewer="")

                    self.assertEqual(path.read_bytes(), before)

    def test_shared_annotation_file_preserves_other_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            first = _photos().iloc[[0]].copy()
            second = _photos().iloc[[1]].copy()

            save_annotations(
                path, first, {"img1": "CI-001"}, reviewer="Alice",
                reviewer_roster=REVIEWER_ROSTER)
            save_annotations(
                path, second, {"img2": "CI-002"}, reviewer="ALICE",
                reviewer_roster=REVIEWER_ROSTER)

            self.assertEqual(
                load_annotations(path, "alice"),
                {"img1": "CI-001", "img2": "CI-002"})
            self.assertEqual(
                load_annotations(path, "alice", image_ids=["img2"]),
                {"img2": "CI-002"})

    def test_same_reviewer_stale_sessions_save_only_changed_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            photos = _photos()
            initial = {"img1": "old-1", "img2": "old-2"}
            save_annotations(
                path, photos, initial, reviewer="alice",
                reviewer_roster=REVIEWER_ROSTER)
            session_a = dict(initial)
            session_b = dict(initial)
            session_a["img1"] = "new-1"
            session_b["img2"] = "new-2"

            save_annotations(
                path, photos, session_a, reviewer="alice",
                reviewer_roster=REVIEWER_ROSTER,
                replace_image_ids=["img1"])
            save_annotations(
                path, photos, session_b, reviewer="alice",
                reviewer_roster=REVIEWER_ROSTER,
                replace_image_ids=["img2"])

            self.assertEqual(
                load_annotations(path, "alice"),
                {"img1": "new-1", "img2": "new-2"})


class TestVoteSummary(unittest.TestCase):
    """裁决默认只接受至少三人的完全一致票。"""

    def test_reviewer_aliases_count_as_one_canonical_identity(self):
        records = pd.DataFrame([
            {"image_id": "img1", "label": "0001", "reviewer": "alice"},
            {"image_id": "img1", "label": "0001", "reviewer": " Alice "},
            {"image_id": "img1", "label": "0001", "reviewer": "ＡＬＩＣＥ"},
            {"image_id": "img1", "label": "0001", "reviewer": "BOB"},
            {"image_id": "img1", "label": "0001", "reviewer": "carol"},
        ])

        row = summarize_annotations(
            records, reviewer_roster=REVIEWER_ROSTER).iloc[0]

        self.assertEqual(normalize_reviewer_id(" ＡＬＩＣＥ "), "alice")
        self.assertEqual(row["n_reviewers"], 3)
        self.assertEqual(row["adjudication_status"], "agreed")
        self.assertEqual(
            json.loads(row["eligible_reviewers"]),
            ["alice", "bob", "carol"])

    def test_duplicate_roster_alias_and_unregistered_vote_are_rejected_or_excluded(self):
        with self.assertRaisesRegex(ValueError, "重复"):
            normalize_reviewer_roster(["alice", "Alice", "carol"])

        records = pd.DataFrame([
            {"image_id": "img1", "label": "0001", "reviewer": "alice"},
            {"image_id": "img1", "label": "0001", "reviewer": "bob"},
            {"image_id": "img1", "label": "0001", "reviewer": "mallory"},
        ])
        row = summarize_annotations(
            records, reviewer_roster=REVIEWER_ROSTER).iloc[0]

        self.assertEqual(row["n_reviewers"], 2)
        self.assertEqual(row["adjudication_status"], "pending")
        self.assertEqual(json.loads(row["ineligible_reviewers"]), ["mallory"])

    def test_reserved_status_variants_are_canonicalized(self):
        records = pd.DataFrame([
            {"image_id": "img1", "label": "REJECT", "reviewer": "alice"},
            {"image_id": "img1", "label": "ＲＥＪＥＣＴ", "reviewer": "bob"},
            {"image_id": "img1", "label": "reject", "reviewer": "carol"},
        ])

        row = summarize_annotations(
            records, reviewer_roster=REVIEWER_ROSTER).iloc[0]

        self.assertEqual(row["adjudication_status"], "agreed")
        self.assertEqual(row["adjudicated_label"], "reject")

    def test_agreed_pending_and_conflict_are_auditable(self):
        records = pd.DataFrame([
            {"image_id": "agreed", "label": "0001", "reviewer": "alice"},
            {"image_id": "agreed", "label": "0001", "reviewer": "bob"},
            {"image_id": "agreed", "label": "0001", "reviewer": "carol"},
            {"image_id": "pending", "label": "0002", "reviewer": "alice"},
            {"image_id": "conflict", "label": "0003", "reviewer": "alice"},
            {"image_id": "conflict", "label": "reject", "reviewer": "bob"},
        ])

        summary = summarize_annotations(
            records, reviewer_roster=REVIEWER_ROSTER).set_index("image_id")
        self.assertEqual(summary.loc["agreed", "adjudication_status"], "agreed")
        self.assertEqual(summary.loc["agreed", "adjudicated_label"], "0001")
        self.assertEqual(summary.loc["pending", "adjudication_status"], "pending")
        self.assertEqual(summary.loc["pending", "adjudicated_label"], "uncertain")
        self.assertEqual(summary.loc["conflict", "adjudication_status"], "conflict")
        self.assertEqual(summary.loc["agreed", "min_reviewers_required"], 3)
        votes = json.loads(summary.loc["conflict", "reviewer_votes"])
        self.assertEqual(votes, {"alice": "0003", "bob": "reject"})

    def test_legacy_vote_cannot_double_count_named_reviewer(self):
        """匿名旧票可能就是 Alice 本人，不能与 Alice 合成两人确认。"""
        records = pd.DataFrame([
            {"image_id": "img1", "label": "0001", "reviewer": ""},
            {"image_id": "img1", "label": "0001", "reviewer": "alice"},
        ])

        row = summarize_annotations(
            records, reviewer_roster=REVIEWER_ROSTER).iloc[0]
        self.assertEqual(row["adjudication_status"], "pending")
        self.assertEqual(row["n_reviewers"], 1)
        self.assertEqual(json.loads(row["vote_counts"]), {"0001": 2})
        self.assertEqual(json.loads(row["eligible_vote_counts"]), {"0001": 1})

    def test_min_reviewer_requirement_is_configurable(self):
        records = pd.DataFrame([
            {"image_id": "img1", "label": "0001", "reviewer": "alice"},
            {"image_id": "img1", "label": "0001", "reviewer": "bob"},
        ])

        row = summarize_annotations(
            records, min_reviewers=3,
            reviewer_roster=REVIEWER_ROSTER).iloc[0]
        self.assertEqual(row["adjudication_status"], "pending")
        self.assertEqual(row["min_reviewers_required"], 3)
        with self.assertRaises(ValueError):
            summarize_annotations(
                records, min_reviewers=2,
                reviewer_roster=REVIEWER_ROSTER)
        with self.assertRaises(ValueError):
            summarize_annotations(
                records, min_reviewers=1,
                reviewer_roster=REVIEWER_ROSTER)


class TestReviewAppIsolation(unittest.TestCase):
    """应用端只展示当前审核人的票，避免被他人结论引导。"""

    def test_state_and_annotate_are_reviewer_scoped(self):
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations.csv"
            photos = _photos()
            save_annotations(
                annotations, photos, {"img1": "CI-001"}, "alice",
                reviewer_roster=REVIEWER_ROSTER)
            args = argparse.Namespace(
                annotations=annotations,
                images_root=root,
                reviewer="bob",
                reviewer_roster=REVIEWER_ROSTER,
                min_reviewers=3,
                history_lookup=None,
                history_quality=None,
                batch_embeddings=None,
                embeddings=None,
                embeddings_meta=None,
            )
            client = TestClient(build_app(args, photos=photos))

            state = client.get("/api/state").json()
            self.assertEqual(state["reviewer"], "bob")
            self.assertEqual(state["n_reviewed"], 0)
            labels = {p["image_id"]: p["label"] for p in state["photos"]}
            self.assertEqual(labels["img1"], "")

            response = client.post("/api/annotate", json={
                "image_id": "img1", "action": "confirm", "identity": "CI-002",
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(load_annotations(annotations, "alice")["img1"], "CI-001")
            self.assertEqual(load_annotations(annotations, "bob")["img1"], "CI-002")

            reserved = client.post("/api/annotate", json={
                "image_id": "img2", "action": "confirm", "identity": "Reject",
            })
            self.assertEqual(reserved.status_code, 400)
            self.assertNotIn("img2", load_annotations(annotations, "bob"))

    def test_anonymous_web_review_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                annotations=root / "annotations.csv", images_root=root,
                reviewer="", history_lookup=None, history_quality=None,
                reviewer_roster=REVIEWER_ROSTER, min_reviewers=3,
                batch_embeddings=None, embeddings=None, embeddings_meta=None,
            )
            with self.assertRaisesRegex(ValueError, "reviewer"):
                build_app(args, photos=_photos())

    def test_unregistered_web_reviewer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                annotations=root / "annotations.csv", images_root=root,
                reviewer="mallory", reviewer_roster=REVIEWER_ROSTER,
                min_reviewers=3, history_lookup=None, history_quality=None,
                batch_embeddings=None, embeddings=None, embeddings_meta=None,
            )

            with self.assertRaisesRegex(ValueError, "不在 reviewer roster"):
                build_app(args, photos=_photos())

    def test_photo_path_is_validated_by_image_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photos = _photos()
            photos.loc[0, "relative_path"] = "../outside.jpg"
            args = argparse.Namespace(
                annotations=root / "annotations.csv", images_root=root,
                reviewer="alice", reviewer_roster=REVIEWER_ROSTER,
                min_reviewers=3, history_lookup=None, history_quality=None,
                batch_embeddings=None, embeddings=None, embeddings_meta=None,
            )

            with self.assertRaisesRegex(ValueError, "路径越界"):
                build_app(args, photos=photos)

    def test_failed_single_and_batch_save_do_not_change_memory_state(self):
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                annotations=root / "annotations.csv", images_root=root,
                reviewer="alice", reviewer_roster=REVIEWER_ROSTER,
                min_reviewers=3, history_lookup=None, history_quality=None,
                batch_embeddings=None, embeddings=None, embeddings_meta=None,
            )
            client = TestClient(build_app(args, photos=_photos()))

            with mock.patch.object(
                    review_app_module, "save_annotations",
                    side_effect=OSError("disk failed")):
                with self.assertRaisesRegex(OSError, "disk failed"):
                    client.post("/api/annotate", json={
                        "image_id": "img1", "action": "confirm",
                        "identity": "CI-001",
                    })
                with self.assertRaisesRegex(OSError, "disk failed"):
                    client.post("/api/annotate_batch", json={
                        "image_ids": ["img1", "img2"],
                        "action": "reject",
                    })

            state = client.get("/api/state").json()
            labels = {row["image_id"]: row["label"] for row in state["photos"]}
            self.assertEqual(labels, {"img1": "", "img2": "", "img3": ""})
            self.assertEqual(state["n_reviewed"], 0)

    def test_history_path_cannot_escape_lookup_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "history"
            sibling = Path(tmp) / "history_evil"
            root.mkdir()
            sibling.mkdir()

            self.assertIsNone(_resolve_history_path(
                root, "..", sibling.name, "x.jpg"))
            self.assertEqual(
                _resolve_history_path(root, "known", "x.jpg"),
                (root / "known" / "x.jpg").resolve())


class TestSimilarityArtifacts(unittest.TestCase):
    """相似度提示只接受生成期行绑定且按 image_id 对齐的产物。"""

    def test_verified_artifact_loads_with_string_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings, meta = _write_embedding_artifact(
                root, ["0001", "NA"])

            values, index = load_embeddings(embeddings, meta)

            self.assertEqual(values.shape, (2, 2))
            self.assertEqual(index, {"0001": 0, "NA": 1})

    def test_reordered_meta_with_stale_row_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings, meta = _write_embedding_artifact(
                root, ["img2", "img1"], digest_ids=["img1", "img2"])

            with self.assertRaisesRegex(ValueError, "行序摘要"):
                load_embeddings(embeddings, meta)

    def test_legacy_backfilled_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings, meta = _write_embedding_artifact(root, ["img1"])
            config_path = root / "features_config.json"
            config = json.loads(config_path.read_text("utf-8"))
            config["artifact_schema_version"] = 1
            config["provenance_level"] = (
                "legacy_backfilled_unverified_row_alignment")
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "artifact_schema_version"):
                load_embeddings(embeddings, meta)

    def test_unknown_row_binding_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings, meta = _write_embedding_artifact(root, ["img1"])
            config_path = root / "features_config.json"
            config = json.loads(config_path.read_text("utf-8"))
            config["row_binding"] = "some_other_binding"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "row_binding"):
                load_embeddings(embeddings, meta)

    def test_review_manifest_must_be_covered_by_similarity_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings, meta = _write_embedding_artifact(
                root, ["img1", "img2"])
            args = argparse.Namespace(
                annotations=root / "annotations.csv", images_root=root,
                reviewer="alice", reviewer_roster=REVIEWER_ROSTER,
                min_reviewers=3, history_lookup=None, history_quality=None,
                batch_embeddings=None, embeddings=embeddings,
                embeddings_meta=meta,
            )

            with self.assertRaisesRegex(ValueError, "缺少审核图片"):
                build_app(args, photos=_photos())

    def test_batch_similarity_uses_meta_image_id_not_manifest_row_order(self):
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            embeddings, _ = _write_embedding_artifact(
                root,
                ["img3", "img1", "img2"],
                values=np.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]),
            )
            photos = _photos()
            photos["cluster"] = 1
            args = argparse.Namespace(
                annotations=root / "annotations.csv", images_root=root,
                reviewer="alice", reviewer_roster=REVIEWER_ROSTER,
                min_reviewers=3, history_lookup=None, history_quality=None,
                batch_embeddings=embeddings, embeddings=None,
                embeddings_meta=None,
            )

            state = TestClient(build_app(args, photos=photos)).get(
                "/api/state").json()
            by_id = {row["image_id"]: row for row in state["photos"]}

            self.assertEqual(by_id["img1"]["in_sim"], 0.667)
            self.assertEqual(by_id["img2"]["in_sim"], 0.667)
            self.assertEqual(by_id["img3"]["in_sim"], 0.333)


class TestConfirmedExport(unittest.TestCase):
    """只有 roster 共识可导出，且累计库不能丢失旧批次。"""

    def _args(self, root: Path, reviewer: str = "") -> argparse.Namespace:
        return argparse.Namespace(
            clusters=root / "clusters.csv",
            annotations=root / "annotations.csv",
            images_root=root,
            out=root / "confirmed.csv",
            summary_out=root / "summary.csv",
            reviewer=reviewer,
            min_reviewers=3,
            reviewer_roster=REVIEWER_ROSTER,
        )

    def test_legacy_and_consensus_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photos = _photos()
            photos.to_csv(root / "clusters.csv", index=False)
            args = self._args(root)

            # 旧版匿名票仅供追溯，不能单票覆盖正式 confirmed。
            args.out.write_text("existing-confirmed", encoding="utf-8")
            args.summary_out.write_text("existing-summary", encoding="utf-8")
            args.annotations.write_text(
                "image_id,label\nimg1,0001\nimg2,uncertain\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                export_confirmed(args)
            self.assertEqual(args.out.read_text("utf-8"), "existing-confirmed")
            self.assertEqual(args.summary_out.read_text("utf-8"), "existing-summary")

            # 后续有效导出应保留其他批次的累计确认与汇总。
            pd.DataFrame([{
                "image_id": "old-batch-image",
                "confirmed_identity": "old-id",
                "status": "confirmed",
                "reviewer": "consensus",
            }]).to_csv(args.out, index=False)
            pd.DataFrame([{
                "image_id": "old-batch-image",
                "adjudication_status": "agreed",
            }]).to_csv(args.summary_out, index=False)

            # 命名审核人触发多人模式：一致票导出，冲突票不导出。
            args.annotations.unlink()
            save_annotations(args.annotations, photos,
                             {"img1": "0001", "img2": "0002"}, "alice",
                             reviewer_roster=REVIEWER_ROSTER)
            save_annotations(args.annotations, photos,
                             {"img1": "0001", "img2": "reject"}, "bob",
                             reviewer_roster=REVIEWER_ROSTER)
            save_annotations(args.annotations, photos,
                             {"img1": "0001", "img2": "0002"}, "carol",
                             reviewer_roster=REVIEWER_ROSTER)
            export_confirmed(args)
            confirmed = pd.read_csv(args.out, dtype=str, keep_default_na=False)
            self.assertEqual(
                confirmed["image_id"].tolist(), ["img1", "old-batch-image"])
            current = confirmed.set_index("image_id").loc["img1"]
            self.assertEqual(current["reviewer"], "consensus")
            self.assertEqual(
                json.loads(current["consensus_reviewers"]),
                ["alice", "bob", "carol"])
            summary = pd.read_csv(args.summary_out, dtype=str, keep_default_na=False)
            self.assertEqual(len(summary), 3)
            self.assertIn("old-batch-image", set(summary["image_id"]))

            # 指定个人名字也不能把 Alice 单票覆盖到正式 confirmed 输出。
            args.reviewer = "alice"
            export_confirmed(args)
            guarded = pd.read_csv(args.out, dtype=str, keep_default_na=False)
            self.assertEqual(
                guarded["image_id"].tolist(), ["img1", "old-batch-image"])
            self.assertEqual(
                guarded.set_index("image_id").loc["img1", "reviewer"],
                "consensus")

    def test_multi_file_replace_failure_restores_both_old_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "confirmed.csv"
            second = root / "summary.csv"
            first.write_text("old-confirmed", encoding="utf-8")
            second.write_text("old-summary", encoding="utf-8")
            real_replace = os.replace
            injected = False

            def fail_second_install(src, dst):
                nonlocal injected
                if (not injected and Path(dst) == second
                        and Path(src).suffix == ".tmp"):
                    injected = True
                    raise OSError("injected replace failure")
                return real_replace(src, dst)

            with mock.patch.object(
                    review_app_module.os, "replace",
                    side_effect=fail_second_install):
                with self.assertRaisesRegex(OSError, "injected"):
                    review_app_module._write_csv_transaction([
                        (first, pd.DataFrame({"value": ["new-confirmed"]})),
                        (second, pd.DataFrame({"value": ["new-summary"]})),
                    ])

            self.assertEqual(first.read_text("utf-8"), "old-confirmed")
            self.assertEqual(second.read_text("utf-8"), "old-summary")
            leftovers = [path.name for path in root.iterdir()
                         if path.suffix in {".tmp", ".bak"}]
            self.assertEqual(leftovers, [])

    def test_export_scopes_shared_annotations_to_current_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = _photos().iloc[[0]].copy()
            other = _photos().iloc[[1]].copy()
            current.to_csv(root / "clusters.csv", index=False)
            args = self._args(root)
            for reviewer in REVIEWER_ROSTER:
                save_annotations(
                    args.annotations, current, {"img1": "0001"}, reviewer,
                    reviewer_roster=REVIEWER_ROSTER)
            save_annotations(
                args.annotations, other, {"img2": "0002"}, "alice",
                reviewer_roster=REVIEWER_ROSTER)

            export_confirmed(args)

            confirmed = pd.read_csv(args.out, dtype=str, keep_default_na=False)
            self.assertEqual(confirmed["image_id"].tolist(), ["img1"])
            records = load_annotation_records(args.annotations)
            self.assertEqual(set(records["image_id"]), {"img1", "img2"})

    def test_re_adjudication_removes_stale_confirmed_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photos = _photos().iloc[:2].copy()
            photos.to_csv(root / "clusters.csv", index=False)
            args = self._args(root)
            for reviewer in REVIEWER_ROSTER:
                save_annotations(
                    args.annotations, photos,
                    {"img1": "0001", "img2": "0002"}, reviewer,
                    reviewer_roster=REVIEWER_ROSTER)
            export_confirmed(args)
            self.assertEqual(
                set(pd.read_csv(args.out, dtype=str)["image_id"]),
                {"img1", "img2"})

            save_annotations(
                args.annotations, photos,
                {"img1": "0001", "img2": "reject"}, "bob",
                reviewer_roster=REVIEWER_ROSTER)
            export_confirmed(args)

            confirmed = pd.read_csv(args.out, dtype=str, keep_default_na=False)
            summary = pd.read_csv(
                args.summary_out, dtype=str, keep_default_na=False
            ).set_index("image_id")
            self.assertEqual(confirmed["image_id"].tolist(), ["img1"])
            self.assertEqual(summary.loc["img2", "adjudication_status"], "conflict")

    def test_all_current_confirmed_rows_can_be_removed_transactionally(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photos = _photos().iloc[[0]].copy()
            photos.to_csv(root / "clusters.csv", index=False)
            args = self._args(root)
            for reviewer in REVIEWER_ROSTER:
                save_annotations(
                    args.annotations, photos, {"img1": "0001"}, reviewer,
                    reviewer_roster=REVIEWER_ROSTER)
            export_confirmed(args)
            self.assertEqual(
                pd.read_csv(args.out, dtype=str)["image_id"].tolist(), ["img1"])

            # 本批重新裁决为排除后，确认库允许成为只有表头的合法空表；
            # 汇总仍与确认库在同一事务中更新，保留失效原因。
            for reviewer in REVIEWER_ROSTER:
                save_annotations(
                    args.annotations, photos, {"img1": "reject"}, reviewer,
                    reviewer_roster=REVIEWER_ROSTER)
            export_confirmed(args)

            confirmed = pd.read_csv(args.out, dtype=str, keep_default_na=False)
            summary = pd.read_csv(
                args.summary_out, dtype=str, keep_default_na=False
            ).set_index("image_id")
            self.assertTrue(confirmed.empty)
            self.assertIn("confirmed_identity", confirmed.columns)
            self.assertEqual(summary.loc["img1", "adjudication_status"], "agreed")
            self.assertEqual(summary.loc["img1", "adjudicated_label"], "reject")

    def test_concurrent_batch_exports_are_serialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations.csv"
            batch1 = _photos().iloc[[0]].copy()
            batch2 = _photos().iloc[[1]].copy()
            clusters1 = root / "batch1.csv"
            clusters2 = root / "batch2.csv"
            batch1.to_csv(clusters1, index=False)
            batch2.to_csv(clusters2, index=False)
            for reviewer in REVIEWER_ROSTER:
                save_annotations(
                    annotations, batch1, {"img1": "0001"}, reviewer,
                    reviewer_roster=REVIEWER_ROSTER)
                save_annotations(
                    annotations, batch2, {"img2": "0002"}, reviewer,
                    reviewer_roster=REVIEWER_ROSTER)

            args1 = self._args(root)
            args2 = self._args(root)
            args1.clusters = clusters1
            args2.clusters = clusters2
            real_write = review_app_module._write_csv_transaction
            first_entered = threading.Event()
            release_first = threading.Event()
            call_lock = threading.Lock()
            call_count = 0
            errors = []

            def delayed_write(outputs):
                nonlocal call_count
                with call_lock:
                    call_count += 1
                    is_first = call_count == 1
                if is_first:
                    first_entered.set()
                    if not release_first.wait(timeout=5):
                        raise TimeoutError("test did not release first export")
                return real_write(outputs)

            def worker(args):
                try:
                    export_confirmed(args)
                except Exception as exc:  # pragma: no cover - 仅用于线程回传
                    errors.append(exc)

            with mock.patch.object(
                    review_app_module, "_write_csv_transaction",
                    side_effect=delayed_write):
                first = threading.Thread(target=worker, args=(args1,))
                second = threading.Thread(target=worker, args=(args2,))
                first.start()
                self.assertTrue(first_entered.wait(timeout=5))
                second.start()
                time.sleep(0.2)
                with call_lock:
                    self.assertEqual(call_count, 1)
                release_first.set()
                first.join(timeout=5)
                second.join(timeout=5)

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(errors, [])
            confirmed = pd.read_csv(args1.out, dtype=str, keep_default_na=False)
            summary = pd.read_csv(
                args1.summary_out, dtype=str, keep_default_na=False)
            self.assertEqual(set(confirmed["image_id"]), {"img1", "img2"})
            self.assertEqual(set(summary["image_id"]), {"img1", "img2"})

    def test_export_reads_votes_only_after_acquiring_output_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photos = _photos().iloc[[0]].copy()
            photos.to_csv(root / "clusters.csv", index=False)
            args = self._args(root)
            for reviewer in REVIEWER_ROSTER:
                save_annotations(
                    args.annotations, photos, {"img1": "old-id"}, reviewer,
                    reviewer_roster=REVIEWER_ROSTER)

            real_output_locks = review_app_module._output_file_locks
            reached_lock = threading.Event()
            errors = []

            @contextmanager
            def observed_output_locks(paths):
                reached_lock.set()
                with real_output_locks(paths):
                    yield

            def worker():
                try:
                    export_confirmed(args)
                except Exception as exc:  # pragma: no cover - 仅用于线程回传
                    errors.append(exc)

            # 先占住累计输出锁。导出线程到达锁后再更新票；正确实现应在
            # 获得锁之后才读取票，因此最终只能提交 new-id。
            with real_output_locks([args.out, args.summary_out]):
                with mock.patch.object(
                        review_app_module, "_output_file_locks",
                        side_effect=observed_output_locks):
                    thread = threading.Thread(target=worker)
                    thread.start()
                    self.assertTrue(reached_lock.wait(timeout=5))
                    for reviewer in REVIEWER_ROSTER:
                        save_annotations(
                            args.annotations, photos, {"img1": "new-id"},
                            reviewer, reviewer_roster=REVIEWER_ROSTER)

            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            confirmed = pd.read_csv(args.out, dtype=str, keep_default_na=False)
            self.assertEqual(confirmed.loc[0, "confirmed_identity"], "new-id")


if __name__ == "__main__":
    unittest.main()
