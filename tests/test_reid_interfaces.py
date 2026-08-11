"""
接口层冒烟测试：验证 Retrieval + Evaluation 逻辑正确性。

不依赖图片与模型（构造已知 embedding 验证指标）。
运行：python -m unittest tests.test_reid_interfaces -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reid.evaluation.metrics import mean_average_precision, recall_at_k  # noqa: E402
from src.reid.retrieval.cosine import cosine_topk  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
