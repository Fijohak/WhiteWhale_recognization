"""
接口层冒烟测试：验证 Retrieval + Evaluation 逻辑正确性。

不依赖图片与模型（构造已知 embedding 验证指标）。
运行：python -m unittest tests.test_reid_interfaces -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reid.dataset.base import DatasetAdapter, ReIDData  # noqa: E402
from src.reid.evaluation.metrics import mean_average_precision, recall_at_k  # noqa: E402
from src.reid.retrieval.cosine import cosine_topk  # noqa: E402


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
        from scripts.pub_reid_benchmark import split_query_gallery

        df = pd.DataFrame({"identity": ["a"] * 4 + ["b"] * 2 + ["c"] * 1})
        df["image_path"] = [f"img{i}.jpg" for i in range(len(df))]
        q, g = split_query_gallery(df, identity_col="identity")
        # 每身份至少 2 张才参与；c 只有 1 张 → 不参与
        self.assertEqual(sorted(q["identity"].unique()), ["a", "b"])
        # 同身份 query 与 gallery 不共用同图
        for iid in ["a", "b"]:
            qi = set(q[q["identity"] == iid].index)
            gi = set(g[g["identity"] == iid].index)
            self.assertEqual(len(qi), 1)
            self.assertTrue(qi.isdisjoint(gi))
        self.assertEqual(len(g), 4)  # 4+2-2=4 张进 gallery


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
        from scripts.contact_sheets import build_cluster_contact_sheets

        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "clusters.csv"
            self._clusters_df().to_csv(csv, index=False)
            out = Path(tmp) / "sheets"
            build_cluster_contact_sheets(csv, out, Path("I:/"), mock=True)
            files = sorted(p.name for p in out.glob("*.jpg"))
            self.assertEqual(files, ["cluster_000.jpg", "cluster_001.jpg", "noise.jpg"])
            # 噪声与候选簇分属不同文件（-1 合法噪声，不强制分配）
            self.assertNotIn("cluster_002.jpg", files)

    def test_load_review_paths_traceable(self):
        """审核数据集必须能从 relative_path 还原绝对路径（可追溯原图）。"""
        from scripts.fiftyone_review import load_review_dataset

        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "clusters.csv"
            self._clusters_df().to_csv(csv, index=False)
            root = Path("I:/")
            df = load_review_dataset(csv, root)
            self.assertEqual(len(df), 9)
            # 绝对路径 = 图片根 + 相对路径（Windows 分隔符兼容）
            self.assertTrue(
                df["source_path"].iloc[0].startswith(str(root))
            )
            self.assertIn("01/00/a.jpg", df["source_path"].iloc[0].replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
