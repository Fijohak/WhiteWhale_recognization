"""
统一配置加载测试：配置名解析（"pipeline"/"reid"/"detector"）与默认值。

回归背景：load_config("pipeline") 曾把配置名当文件路径解析 → yaml 从未
实际加载（全靠 argparse fallback 兜底）；修复后配置名必须解析到默认路径。

运行：python -m pytest tests/test_config.py -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from whitewhale.config import DEFAULT_CONFIGS, load_config  # noqa: E402


class TestLoadConfig(unittest.TestCase):
    def test_by_name_resolves_default_path(self):
        """配置名 "pipeline" → 默认 yaml，关键阈值真实加载。"""
        cfg = load_config("pipeline")
        self.assertEqual(cfg["retrieval"]["threshold_cluster"], 0.58)
        self.assertEqual(cfg["retrieval"]["threshold_image"], 0.50)
        self.assertEqual(cfg["clustering"]["min_cluster_size"], 3)

    def test_by_kind_resolves_default_path(self):
        """kind 参数同样解析（与配置名等价）。"""
        cfg = load_config(None, "reid")
        self.assertEqual(cfg["val_n"], 6)          # 与旧脚本默认值一致
        self.assertEqual(cfg["epochs_stage1"], 20)
        self.assertEqual(cfg["lr_backbone"], 5e-6)

    def test_missing_path_returns_empty(self):
        """文件不存在 → 空 dict（脚本继续用 argparse 默认值，不崩溃）。"""
        self.assertEqual(load_config(Path("不存在.yaml")), {})
        self.assertEqual(load_config("不存在的配置名"), {})

    def test_all_default_configs_exist(self):
        for name, p in DEFAULT_CONFIGS.items():
            self.assertTrue(p.exists(), f"{name} 配置缺失: {p}")


if __name__ == "__main__":
    unittest.main()
