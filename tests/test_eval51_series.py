"""
E5.1 连拍串评估回归测试。
验证同串判定同时约束 session、文件名序列键和连续帧段。
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.build_missed_viewer import build_cards  # noqa: E402
from experiments.diagnose_missed import (  # noqa: E402
    _as_bool,
    score_with_representatives,
)
from experiments.eval51_common import (  # noqa: E402
    annotate_series,
    evaluate_probe_clusters,
    exclude_same_series,
    split_probe_gallery,
    split_probe_gallery_series,
    summarize_probe_clusters,
)


class TestEval51Series(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame({
            "session_id": ["A", "A", "B", "A", "A", "A", "A", "A"],
            "filename": [
                "0001_20140417_SZi_01_RAY_0100.JPG",  # query
                "0002_20140417_SZi_01_RAY_0102.JPG",  # 同 session、同键、连续
                "0003_20140417_SZi_01_RAY_0101.JPG",  # 不同 session
                "0004_20140417_HBi_01_RAY_0101.JPG",  # 不同序列键
                "0005_20140417_SZi_01_RAY_0105.JPG",  # 由 0104 桥接，传递同串
                "0006_20140417_SZi_01_RAY_0104.JPG",  # 0102→0104→0105
                "RES20001.JPG",                       # 无连拍信息
                "0007_20140417_SZi_01_RAY_0110.JPG",  # 间隔大于 2，新串
            ],
        })
        annotate_series(self.meta)

    def test_series_id_respects_full_sequence_constraints(self):
        self.assertEqual(self.meta.loc[0, "series_id"],
                         self.meta.loc[1, "series_id"])
        self.assertNotEqual(self.meta.loc[0, "series_id"],
                            self.meta.loc[2, "series_id"])
        self.assertNotEqual(self.meta.loc[0, "series_id"],
                            self.meta.loc[3, "series_id"])
        self.assertEqual(self.meta.loc[0, "series_id"],
                         self.meta.loc[4, "series_id"])
        self.assertEqual(self.meta.loc[4, "series_id"],
                         self.meta.loc[5, "series_id"])
        self.assertEqual(self.meta.loc[6, "series_id"], "")
        self.assertNotEqual(self.meta.loc[0, "series_id"],
                            self.meta.loc[7, "series_id"])

    def test_exclusion_removes_cross_session_and_same_complete_series(self):
        gallery = np.arange(1, len(self.meta))
        identities = np.asarray([f"id-{i}" for i in gallery])

        kept, kept_identities = exclude_same_series(
            self.meta, 0, gallery, identities)

        self.assertEqual(kept.tolist(), [3, 6, 7])
        self.assertEqual(kept_identities.tolist(),
                         ["id-3", "id-6", "id-7"])

    def test_eval_only_scores_same_session_and_reports_skipped_queries(self):
        meta = pd.DataFrame({
            "session_id": ["A", "A", "A", "B", "A"],
            "ind": ["target", "target", "distractor", "other", "missing"],
            "series_id": ["a-q", "a-g", "a-d", "b-x", "a-missing"],
        })
        embeddings = np.asarray([
            [1.0, 0.0],       # target query
            [0.6, 0.8],       # 同 session 的正确身份，rank 2
            [0.7, 0.714],     # 同 session 干扰身份，rank 1
            [1.0, 0.0],       # 跨 session 高相似项，必须完全排除
            [0.0, 1.0],       # gallery 无正确身份，必须 skipped
        ], dtype=float)
        query = {
            "target": np.asarray([0]),
            "missing": np.asarray([4]),
        }
        gallery = np.asarray([1, 2, 3])
        identities = np.asarray(["target", "distractor", "other"])

        records = evaluate_probe_clusters(
            meta, embeddings, query, gallery, identities,
            reciprocal_rank_k=10)
        target = records[records["individual"] == "target"].iloc[0]
        missing = records[records["individual"] == "missing"].iloc[0]
        summary = summarize_probe_clusters(records, reciprocal_rank_k=10)

        self.assertEqual(target["cluster_rank"], 2)
        self.assertAlmostEqual(target["cluster_rr_at_k"], 0.5)
        self.assertEqual(target["n_gallery_images_effective_max"], 2)
        self.assertEqual(target["n_candidate_identities_effective_max"], 2)
        self.assertEqual(missing["n_query"], 0)
        self.assertEqual(
            missing["skipped_reason"],
            "no_same_session_cross_series_positive_in_gallery")
        self.assertEqual(summary["n_probe_clusters_total"], 2)
        self.assertEqual(summary["n_probe_clusters"], 1)
        self.assertEqual(summary["n_query_images_total"], 2)
        self.assertEqual(summary["n_query_images"], 1)
        self.assertEqual(summary["n_query_images_skipped"], 1)
        self.assertAlmostEqual(summary["cluster_MRR@10"], 0.5)
        self.assertNotIn("cluster_mAP", summary)

    def test_single_r1_is_weighted_by_evaluable_query_images(self):
        records = pd.DataFrame([
            {
                "n_query_total": 4,
                "n_query": 3,
                "n_query_skipped": 1,
                "n_gallery_images_effective_min": 2,
                "n_gallery_images_effective_max": 3,
                "n_gallery_images_effective_sum": 8,
                "n_candidate_identities_effective_min": 2,
                "n_candidate_identities_effective_max": 2,
                "n_candidate_identities_effective_sum": 6,
                "single_r1": 2 / 3,
                "cluster_r1": 1,
                "cluster_r5": 1,
                "cluster_rr_at_k": 1.0,
            },
            {
                "n_query_total": 1,
                "n_query": 1,
                "n_query_skipped": 0,
                "n_gallery_images_effective_min": 2,
                "n_gallery_images_effective_max": 2,
                "n_gallery_images_effective_sum": 2,
                "n_candidate_identities_effective_min": 2,
                "n_candidate_identities_effective_max": 2,
                "n_candidate_identities_effective_sum": 2,
                "single_r1": 0.0,
                "cluster_r1": 0,
                "cluster_r5": 1,
                "cluster_rr_at_k": 0.5,
            },
        ])

        summary = summarize_probe_clusters(records, reciprocal_rank_k=10)

        self.assertEqual(summary["n_query_images_total"], 5)
        self.assertEqual(summary["n_query_images"], 4)
        self.assertEqual(summary["n_query_images_skipped"], 1)
        self.assertAlmostEqual(summary["single_R@1"], 0.5)
        self.assertAlmostEqual(summary["cluster_R@1"], 0.5)

    def test_cluster_rank_beyond_cutoff_has_zero_reciprocal_rank(self):
        meta = pd.DataFrame({
            "session_id": ["A"] * 4,
            "ind": ["target", "target", "d1", "d2"],
            "series_id": ["q", "g", "d1", "d2"],
        })
        embeddings = np.asarray([
            [1.0, 0.0],
            [0.1, 0.995],
            [0.9, 0.436],
            [0.8, 0.6],
        ], dtype=float)

        records = evaluate_probe_clusters(
            meta,
            embeddings,
            {"target": np.asarray([0])},
            np.asarray([1, 2, 3]),
            np.asarray(["target", "d1", "d2"]),
            reciprocal_rank_k=2,
        )

        self.assertEqual(records.loc[0, "cluster_rank"], 3)
        self.assertEqual(records.loc[0, "cluster_rr_at_k"], 0.0)

    def test_split_rejects_identity_reused_across_sessions(self):
        meta = pd.DataFrame({
            "image_id": ["a", "b", "c", "d"],
            "session_id": ["A", "A", "B", "B"],
            "ind": ["local-1", "local-1", "local-1", "local-1"],
            "series_id": ["a1", "a2", "b1", "b2"],
        })
        with self.assertRaisesRegex(ValueError, "多个 session"):
            split_probe_gallery(meta, np.eye(4))
        with self.assertRaisesRegex(ValueError, "多个 session"):
            split_probe_gallery_series(meta, np.eye(4))

    def test_diagnostic_representative_is_actual_argmax_image(self):
        embeddings = np.asarray([
            [0.2, 0.0],
            [0.9, 0.0],
            [0.4, 0.0],
        ], dtype=float)
        gallery = np.asarray([0, 1, 2])
        identities = np.asarray(["same", "same", "other"])

        scores, representatives = score_with_representatives(
            np.asarray([1.0, 0.0]), embeddings, gallery, identities)

        self.assertAlmostEqual(scores["same"], 0.9)
        self.assertEqual(representatives["same"], 1)
        self.assertAlmostEqual(scores["other"], 0.4)
        self.assertEqual(representatives["other"], 2)

    def test_diagnostic_boolean_parser_does_not_treat_false_string_as_true(self):
        self.assertFalse(_as_bool("False"))
        self.assertTrue(_as_bool("true"))
        with self.assertRaisesRegex(ValueError, "无法解析"):
            _as_bool("maybe")

    def test_missed_viewer_keeps_one_card_per_query_and_requires_evidence(self):
        row = {
            "individual": "A_1", "q_no": 1, "q_path": "A/q.jpg",
            "q_image_id": "q1", "q_det_conf": 0.8, "q_fallback": False,
            "q_series_id": "A::seq::0", "same_cos_max": 0.2,
            "same_gallery_n": 2, "evidence_tile": "A_1__q01.png",
            "same_image_id": "s1", "same_path": "A/s.jpg",
            "top1_ind": "A_2", "top1_cos": 0.7,
            "top1_image_id": "t1", "top1_path": "A/t1.jpg",
            "top2_ind": "A_3", "top2_cos": 0.6,
            "top2_image_id": "t2", "top2_path": "A/t2.jpg",
            "top3_ind": "A_4", "top3_cos": 0.5,
            "top3_image_id": "t3", "top3_path": "A/t3.jpg",
        }
        cards = build_cards(pd.DataFrame([row, {**row, "q_no": 2,
                                                "q_image_id": "q2"}]))
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0]["query"]["q_image_id"], "q1")
        self.assertEqual(cards[1]["query"]["q_image_id"], "q2")

        with self.assertRaisesRegex(ValueError, "same_path"):
            build_cards(pd.DataFrame([{k: v for k, v in row.items()
                                       if k != "same_path"}]))

    def test_id_display_migration_does_not_change_eval51_split(self):
        meta = pd.DataFrame({
            "image_id": [f"img{i}" for i in range(8)],
            "session_id": ["A"] * 4 + ["B"] * 4,
            "ind": ["old_a"] * 4 + ["old_b"] * 4,
            "series_id": ["a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2"],
        })
        renamed = meta.copy()
        renamed["ind"] = ["new_01"] * 4 + ["new_02"] * 4
        emb = np.eye(8)

        for splitter in (split_probe_gallery, split_probe_gallery_series):
            with self.subTest(splitter=splitter.__name__):
                old = splitter(meta, emb, seed=7)
                new = splitter(renamed, emb, seed=7)
                old_query = {meta.loc[index, "image_id"]
                             for indices in old[0].values() for index in indices}
                new_query = {renamed.loc[index, "image_id"]
                             for indices in new[0].values() for index in indices}
                self.assertEqual(old_query, new_query)
                self.assertEqual(
                    {meta.loc[index, "image_id"] for index in old[1]},
                    {renamed.loc[index, "image_id"] for index in new[1]},
                )


if __name__ == "__main__":
    unittest.main()
