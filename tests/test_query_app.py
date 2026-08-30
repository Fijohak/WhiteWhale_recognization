"""
查询客户端（src/whitewhale/query.py）接口测试。

用 mock gallery + mock embedder 覆盖：
- 二态提示（known / unknown / 阈值边界）；
- 模型匹配防护（check_model_match：同模型通过，跨模型拒绝）；
- 图片端点（200 / 404）、非法图片（400）、首页 HTML。

不加载真实模型（通过 build_app 的 embedder 注入 mock），运行快且隔离。
运行：python -m unittest tests.test_query_app -v
"""
import argparse
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.query import build_app, resolve_model  # noqa: E402
from whitewhale.data.manifest import compute_sha256  # noqa: E402
from whitewhale.detection.detector import yolo_crop_provenance  # noqa: E402


class FakeEmbedder:
    """mock 特征提取器：encode 返回当前探针向量（已 L2 归一化）。

    probe 在每次请求前由 _query 注入，模拟"同一张图经真实模型编码"
    得到不同向量的效果（HTTP 层无法传向量，只能换探针）。
    """

    name = "fake"
    feat_dim = 64

    def __init__(self, probe: np.ndarray):
        self.probe = probe
        self.last_image_size = None

    def encode(self, images):  # pragma: no cover - 测试替身
        self.last_image_size = images[0].size if images else None
        return self.probe


class FakeDetector:
    """记录 YOLO 调用参数并模拟未检出，验证查询侧设备参数转换。"""

    def __init__(self):
        self.kwargs = None

    def predict(self, image, **kwargs):  # pragma: no cover - 测试替身
        self.kwargs = kwargs
        return []


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
            "confirmed_identity": f"01_ID{i}",
            "quality_band": "80",
            "cluster": i % 3,
        })
    pd.DataFrame(rows).to_csv(tmp / "pilot_set.csv", index=False)
    metric_ckpt = None
    config = {
        "model": model_cfg,
        "crop": "whole",
        "preprocess": (
            "Resize256+CenterCrop224" if "metric-learning" in model_cfg
            else "Resize518" if "dinov2" in model_cfg else "Resize224"
        ),
        "embedding_file": "embeddings.npy",
        "meta_file": "embeddings_meta.csv",
        "embedding_sha256": compute_sha256(tmp / "embeddings.npy"),
        "meta_sha256": compute_sha256(tmp / "embeddings_meta.csv"),
        "n": n,
        "feat_dim": dim,
        "artifact_schema_version": 2,
        "created_at_utc": "2026-08-29T00:00:00+00:00",
        "provenance_level": "generated_with_row_binding",
        "row_binding": "embedding_row_i_to_meta_image_id_i",
        "ordered_image_ids_sha256": hashlib.sha256(
            "\n".join(f"g{i}" for i in range(n)).encode("utf-8")
        ).hexdigest(),
    }
    if "metric-learning" in model_cfg:
        version = model_cfg.rsplit("-", 1)[-1]
        metric_ckpt = tmp / version / "best.pt"
        metric_ckpt.parent.mkdir(parents=True, exist_ok=True)
        metric_ckpt.write_bytes(b"mock metric checkpoint")
        config["checkpoint_sha256"] = compute_sha256(metric_ckpt)
    detector_weights = tmp / "detector.pt"
    detector_weights.write_bytes(b"mock detector")
    (tmp / "embedding_config.json").write_text(
        json.dumps(config), encoding="utf-8")

    args = argparse.Namespace(
        embeddings=tmp / "embeddings.npy",
        meta=tmp / "embeddings_meta.csv",
        pilot=tmp / "pilot_set.csv",
        images_root=tmp,
        model=None,
        dinov2_weight=None,
        metric_ckpt=metric_ckpt,
        k=3, threshold=0.45,
        detect=False,  # 测试关检测裁剪（不加载 YOLO）
        det_weights=detector_weights,
        det_conf=0.25,
        det_imgsz=1024,
        det_device="auto",
        det_pad_x=0.30,
        det_pad_up=0.15,
        det_pad_down=0.60,
    )
    return emb, args


class TestModelMatchGuard(unittest.TestCase):
    """查询模型必须与 gallery 特征同模型（维度相同也不可比）。"""

    def test_auto_mega_with_mega_ok(self):
        """默认自动匹配：gallery 是 megadescriptor → 查询用 megadescriptor。"""
        with tempfile.TemporaryDirectory() as t:
            _, args = make_gallery(Path(t))
            self.assertEqual(resolve_model(args, 4), "megadescriptor")

    def test_auto_metric_with_metric_ok(self):
        """gallery 是微调特征 → 自动用微调查询模型。"""
        with tempfile.TemporaryDirectory() as t:
            _, args = make_gallery(Path(t),
                                   model_cfg="megadescriptor-metric-learning-r1")
            self.assertEqual(resolve_model(args, 4), "metric-learning")

    def test_metric_checkpoint_version_mismatch_rejected(self):
        """即使同属 metric-learning，不同版本权重也必须拒绝。"""
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            _, args = make_gallery(
                root, model_cfg="megadescriptor-metric-learning-r1")
            args.metric_ckpt = root / "r2" / "best.pt"
            args.metric_ckpt.parent.mkdir(parents=True)
            args.metric_ckpt.write_bytes(b"another checkpoint")
            with self.assertRaises(SystemExit):
                resolve_model(args, 4)

    def test_dinov2_with_mega_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            _, args = make_gallery(Path(t))
            args.model = "dinov2"
            with self.assertRaises(SystemExit) as cm:
                resolve_model(args, 4)
            self.assertIn("模型不匹配", str(cm.exception))

    def test_metric_with_mega_rejected(self):
        """微调查询模型不能配预训练 gallery（分布不同）。"""
        with tempfile.TemporaryDirectory() as t:
            _, args = make_gallery(Path(t))
            args.model = "metric-learning"
            with self.assertRaises(SystemExit):
                resolve_model(args, 4)

    def test_dinov2_with_dinov2_ok(self):
        with tempfile.TemporaryDirectory() as t:
            _, args = make_gallery(Path(t), model_cfg="vit_base_patch14_dinov2.lvd142m")
            args.model = "dinov2"
            self.assertEqual(resolve_model(args, 4), "dinov2")

    def test_no_config_rejected(self):
        """无 embedding_config.json 时不盲目比较，直接拒绝。"""
        with tempfile.TemporaryDirectory() as t:
            _, args = make_gallery(Path(t))
            (Path(t) / "embedding_config.json").unlink()
            with self.assertRaises(SystemExit):
                resolve_model(args, 4)

    def test_unknown_model_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            _, args = make_gallery(Path(t))
            args.model = "resnet"
            with self.assertRaises(SystemExit):
                resolve_model(args, 4)

    def test_dinov2_local_weight_hash_must_match(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            _, args = make_gallery(
                root, model_cfg="vit_base_patch14_dinov2.lvd142m")
            expected = root / "expected.pth"
            expected.write_bytes(b"expected")
            actual = root / "actual.pth"
            actual.write_bytes(b"actual")
            config_path = root / "embedding_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["checkpoint_sha256"] = compute_sha256(expected)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            args.dinov2_weight = str(actual)

            with self.assertRaisesRegex(SystemExit, "SHA-256"):
                build_app(args, embedder=FakeEmbedder(np.zeros((1, 64))))


class TestQueryApp(unittest.TestCase):
    """二态提示 + 端点行为（TestClient 走 HTTP，等价真实请求）。"""

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
        self.assertEqual(d["calibration_status"], "provisional_unvalidated")
        self.assertIn("未", d["threshold_warning"])
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
        """候选保留个体 ID 以及最佳匹配图片的原有追溯字段。"""
        r = self._query(self.emb[0:1])
        c = r.json()["candidates"][0]
        for key in ["confirmed_identity", "image_id", "source_group",
                    "session_id", "quality_band", "cluster", "score"]:
            self.assertIn(key, c)

    def test_topk_aggregates_images_by_confirmed_identity(self):
        """同体 A 的多张高分图不得挤掉第二个候选身份 B。"""
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            emb, args = make_gallery(root, n=3)
            query = np.zeros(64, dtype=np.float32)
            query[0] = 1.0
            emb[0] = query
            emb[1, :] = 0.0
            emb[1, :2] = [0.9, np.sqrt(1.0 - 0.9**2)]
            emb[2, :] = 0.0
            emb[2, :2] = [0.8, 0.6]
            np.save(args.embeddings, emb)
            config_path = root / "embedding_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["embedding_sha256"] = compute_sha256(args.embeddings)
            config_path.write_text(json.dumps(config), encoding="utf-8")

            pilot = pd.read_csv(args.pilot)
            pilot["confirmed_identity"] = ["01_A", "01_A", "01_B"]
            pilot.to_csv(args.pilot, index=False)
            args.k = 2
            client = TestClient(build_app(args, embedder=FakeEmbedder(query[None])))

            buf = io.BytesIO()
            Image.new("RGB", (8, 8), (10, 10, 10)).save(buf, "JPEG")
            response = client.post(
                "/api/query",
                files={"file": ("q.jpg", buf.getvalue(), "image/jpeg")},
            )
            candidates = response.json()["candidates"]

            self.assertEqual(
                [candidate["confirmed_identity"] for candidate in candidates],
                ["01_A", "01_B"],
            )
            self.assertEqual(candidates[0]["image_id"], "g0")
            self.assertAlmostEqual(candidates[0]["score"], 1.0, places=6)

    def test_same_raw_identity_in_different_sessions_stays_distinct(self):
        """批次内同号不代表跨 session 已对齐，不得聚合为一个候选。"""
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            emb, args = make_gallery(root, n=2)
            query = np.zeros(64, dtype=np.float32)
            query[0] = 1.0
            emb[0] = query
            emb[1, :] = 0.0
            emb[1, :2] = [0.8, 0.6]
            np.save(args.embeddings, emb)
            config_path = root / "embedding_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["embedding_sha256"] = compute_sha256(args.embeddings)
            config_path.write_text(json.dumps(config), encoding="utf-8")
            pilot = pd.read_csv(args.pilot, dtype=str)
            pilot["session_id"] = ["s1", "s2"]
            pilot["confirmed_identity"] = ["A", "A"]
            pilot.to_csv(args.pilot, index=False)
            args.k = 2
            client = TestClient(build_app(
                args, embedder=FakeEmbedder(query[None])))

            buf = io.BytesIO()
            Image.new("RGB", (8, 8), (10, 10, 10)).save(buf, "JPEG")
            response = client.post(
                "/api/query",
                files={"file": ("q.jpg", buf.getvalue(), "image/jpeg")},
            )
            candidates = response.json()["candidates"]

            self.assertEqual(
                [(value["session_id"], value["confirmed_identity"])
                 for value in candidates],
                [("s1", "A"), ("s2", "A")],
            )
            self.assertEqual(
                [value["identity_display"] for value in candidates],
                ["s1::A", "s2::A"],
            )

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

    def test_detector_auto_device(self):
        """查询检测的 auto 必须交给 Ultralytics 自动选设备，而非字面量。"""
        from fastapi.testclient import TestClient

        config_path = Path(self.tmp.name) / "embedding_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(yolo_crop_provenance(
            self.args.det_weights, self.args.det_conf, self.args.det_imgsz,
            self.args.det_pad_x, self.args.det_pad_up, self.args.det_pad_down))
        config_path.write_text(json.dumps(config), encoding="utf-8")
        self.args.detect = True
        detector = FakeDetector()
        client = TestClient(build_app(
            self.args, embedder=self.embedder, detector=detector))

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (10, 10, 10)).save(buf, "JPEG")
        response = client.post(
            "/api/query",
            files={"file": ("q.jpg", buf.getvalue(), "image/jpeg")})

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(detector.kwargs["device"])
        self.assertEqual(self.embedder.last_image_size, (3, 3))

    def test_crop_mode_mismatch_is_rejected(self):
        self.args.detect = True
        with self.assertRaisesRegex(SystemExit, "裁剪"):
            build_app(self.args, embedder=self.embedder, detector=FakeDetector())

    def test_non_positive_k_is_rejected(self):
        self.args.k = 0
        with self.assertRaisesRegex(ValueError, "k"):
            build_app(self.args, embedder=self.embedder)

    def test_index_html(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("中华白海豚个体识别", r.text)


class TestGalleryIntegrity(unittest.TestCase):
    """查询端必须拒绝被篡改或错配的 embedding 三件套。"""

    def test_same_length_but_tampered_meta_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            emb, args = make_gallery(root)
            meta_path = root / "embeddings_meta.csv"
            meta = pd.read_csv(meta_path)
            meta.loc[0, "image_id"] = "unrelated"
            meta.to_csv(meta_path, index=False)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                build_app(args, embedder=FakeEmbedder(emb))


if __name__ == "__main__":
    unittest.main()
