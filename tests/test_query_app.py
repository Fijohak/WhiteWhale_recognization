"""
查询客户端（scripts/query_app.py）接口测试。

用 mock gallery + mock embedder 覆盖：
- 三态判定（known / unknown / 阈值边界）；
- 模型匹配防护（check_model_match：同模型通过，跨模型拒绝）；
- 图片端点（200 / 404）、非法图片（400）、首页 HTML。

不加载真实模型（通过 build_app 的 embedder 注入 mock），运行快且隔离。
运行：python -m unittest tests.test_query_app -v
"""
import argparse
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.query_app import build_app, check_model_match  # noqa: E402


class FakeEmbedder:
    """mock 特征提取器：encode 返回当前探针向量（已 L2 归一化）。

    probe 在每次请求前由 _query 注入，模拟"同一张图经真实模型编码"
    得到不同向量的效果（HTTP 层无法传向量，只能换探针）。
    """

    name = "fake"
    feat_dim = 64

    def __init__(self, probe: np.ndarray):
        self.probe = probe

    def encode(self, images):  # pragma: no cover - 测试替身
        return self.probe


def make_gallery(tmp: Path, n: int = 8, dim: int = 64,
                 model_cfg="hf-hub:BVRA/MegaDescriptor-T-224"):
    """构造 mock gallery：n 个随机单位向量 + meta/pilot/config + 真实小图。

    返回 (emb, args)。dim 取 64：高维随机向量两两相似度集中在 0 附近，
    unknown 探针（-emb[0]）与所有 gallery 行的余弦值大概率 < 阈值。
    """
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(tmp / "embeddings.npy", emb)
    pd.DataFrame({"image_id": [f"g{i}" for i in range(n)]}).to_csv(
        tmp / "embeddings_meta.csv", index=False)

    # 每张图生成一张真实小图，供图片端点验证
    img_dir = tmp / "imgs"
    img_dir.mkdir(exist_ok=True)
    rows = []
    for i in range(n):
        p = img_dir / f"g{i}.jpg"
        Image.new("RGB", (8, 8), (120, 120, 120)).save(p, "JPEG")
        rows.append({
            "image_id": f"g{i}",
            "relative_path": f"imgs/g{i}.jpg",
            "source_group": f"SG{i}",
            "session_id": "01",
            "quality_band": "80",
            "cluster": i % 3,
        })
    pd.DataFrame(rows).to_csv(tmp / "pilot_set.csv", index=False)
    (tmp / "embedding_config.json").write_text(
        json.dumps({"model": model_cfg}), encoding="utf-8")

    args = argparse.Namespace(
        embeddings=tmp / "embeddings.npy",
        meta=tmp / "embeddings_meta.csv",
        pilot=tmp / "pilot_set.csv",
        images_root=tmp,
        model="megadescriptor",
        dinov2_weight=None,
        k=3, threshold=0.45,
    )
    return emb, args


class TestModelMatchGuard(unittest.TestCase):
    """查询模型必须与 gallery 特征同模型（维度相同也不可比）。"""

    def test_mega_with_mega_ok(self):
        with tempfile.TemporaryDirectory() as t:
            _, args = make_gallery(Path(t))
            self.assertEqual(check_model_match(args, 4), "megadescriptor")

    def test_dinov2_with_mega_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            _, args = make_gallery(Path(t))
            args.model = "dinov2"
            with self.assertRaises(SystemExit) as cm:
                check_model_match(args, 4)
            self.assertIn("模型不匹配", str(cm.exception))

    def test_dinov2_with_dinov2_ok(self):
        with tempfile.TemporaryDirectory() as t:
            _, args = make_gallery(Path(t), model_cfg="vit_base_patch14_dinov2.lvd142m")
            args.model = "dinov2"
            self.assertEqual(check_model_match(args, 4), "dinov2")

    def test_no_config_rejected(self):
        """无 embedding_config.json 时不盲目比较，直接拒绝。"""
        with tempfile.TemporaryDirectory() as t:
            _, args = make_gallery(Path(t))
            (Path(t) / "embedding_config.json").unlink()
            with self.assertRaises(SystemExit):
                check_model_match(args, 4)

    def test_unknown_model_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            _, args = make_gallery(Path(t))
            args.model = "resnet"
            with self.assertRaises(SystemExit):
                check_model_match(args, 4)


class TestQueryApp(unittest.TestCase):
    """三态判定 + 端点行为（TestClient 走 HTTP，等价真实请求）。"""

    def setUp(self):
        from fastapi.testclient import TestClient

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.emb, self.args = make_gallery(root)
        self.embedder = FakeEmbedder(self.emb)
        self.client = TestClient(build_app(self.args, embedder=self.embedder))

    def tearDown(self):
        self.tmp.cleanup()

    def _query(self, probe: np.ndarray):
        """注入探针向量并上传一张真实小图（探针决定检索结果）。"""
        self.embedder.probe = probe
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (10, 10, 10)).save(buf, "JPEG")
        return self.client.post(
            "/api/query", files={"file": ("q.jpg", buf.getvalue(), "image/jpeg")},
        )

    @staticmethod
    def _probe_with_max_cos(target: np.ndarray, cos: float) -> np.ndarray:
        """构造与 target 精确余弦相似度为 cos 的单位向量（Gram-Schmidt）。

        probe = cos*target + sqrt(1-cos²)*v，v ⊥ target 单位向量，
        使 cos(probe, target) = cos 精确可控，其余 gallery 行分数更低。
        """
        target = target / np.linalg.norm(target)
        dim = target.shape[0]
        v = None
        for j in range(dim):
            seed = np.eye(dim)[j]
            u = seed - (seed @ target) * target
            if np.linalg.norm(u) > 1e-6:
                v = u / np.linalg.norm(u)
                break
        assert v is not None, "target 与所有基向量平行"
        probe = cos * target + np.sqrt(1.0 - cos**2) * v
        return probe.astype(np.float32).reshape(1, -1)

    def test_known_branch(self):
        """库内图（自匹配 1.0 ≥ 0.45）→ known，Top-1 为自身。"""
        r = self._query(self.emb[2:3])
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["status"], "known")
        self.assertGreaterEqual(d["max_score"], 0.45)
        self.assertEqual(d["candidates"][0]["image_id"], "g2")
        self.assertAlmostEqual(d["candidates"][0]["score"], 1.0, places=5)

    def test_unknown_branch(self):
        """库外特征（负方向，分数 < 阈值）→ unknown，仍返回 Top-K 供参考。"""
        probe = -self.emb[:1]  # 与 g0 的 cos = -1，其余行随机
        # 前提守护：探针与所有 gallery 行余弦值必须低于阈值（否则此测试无意义）
        max_cos = float(np.max(probe @ self.emb.T))
        self.assertLess(max_cos, 0.45,
                        "测试前提不成立：随机库外探针与某行相似度过高")
        r = self._query(probe)
        d = r.json()
        self.assertEqual(d["status"], "unknown")
        self.assertLess(d["max_score"], 0.45)
        self.assertEqual(len(d["candidates"]), 3)
        self.assertIn("threshold", d)

    def test_threshold_sensitivity(self):
        """阈值两侧行为：0.46 → known，0.44 → unknown（阈值 0.45）。"""
        target = self.emb[0]
        r_hi = self._query(self._probe_with_max_cos(target, 0.46))
        self.assertEqual(r_hi.json()["status"], "known")
        r_lo = self._query(self._probe_with_max_cos(target, 0.44))
        self.assertEqual(r_lo.json()["status"], "unknown")
        self.assertAlmostEqual(r_hi.json()["max_score"], 0.46, places=4)
        self.assertAlmostEqual(r_lo.json()["max_score"], 0.44, places=4)

    def test_candidate_traceability(self):
        """候选保留 image_id / source_group / session / quality / cluster 追溯。"""
        r = self._query(self.emb[0:1])
        c = r.json()["candidates"][0]
        for key in ["image_id", "source_group", "session_id", "quality_band",
                    "cluster", "score"]:
            self.assertIn(key, c)

    def test_gallery_size_endpoint(self):
        d = self.client.get("/api/gallery_size").json()
        self.assertEqual(d["n"], 8)
        self.assertEqual(d["model"], "fake")

    def test_image_endpoint_ok(self):
        r = self.client.get("/api/image/g0")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/jpeg")

    def test_image_endpoint_404(self):
        r = self.client.get("/api/image/not_exists")
        self.assertEqual(r.status_code, 404)

    def test_invalid_image_400(self):
        """上传非图片内容 → 400 且带错误信息。"""
        r = self.client.post(
            "/api/query", files={"file": ("q.txt", b"not an image", "text/plain")})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.json())

    def test_index_html(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("中华白海豚个体识别", r.text)


if __name__ == "__main__":
    unittest.main()
