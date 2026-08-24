"""
统一配置加载。

正式 pipeline 的路径、阈值、模型参数集中在 configs/*.yaml；
CLI 参数（argparse）在 yaml 之上覆盖。默认值与重构前的脚本默认值一致。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# 仓库根目录（src/whitewhale/config.py → 向上三级）
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIGS = {
    "pipeline": REPO_ROOT / "configs" / "pipeline.yaml",
    "reid": REPO_ROOT / "configs" / "reid.yaml",
    "detector": REPO_ROOT / "configs" / "detector.yaml",
}


def load_config(path: Path | str | None, kind: str | None = None) -> dict[str, Any]:
    """加载 yaml 配置；path 为 None、配置名（"pipeline" 等）或 kind 均指向默认配置文件。

    文件不存在 → 返回空 dict（脚本继续用 argparse 默认值）。
    """
    if isinstance(path, str) and path in DEFAULT_CONFIGS:
        path = DEFAULT_CONFIGS[path]
    if path is None and kind is not None:
        path = DEFAULT_CONFIGS[kind]
    if path is None or not Path(path).exists():
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def merge_cfg(cfg: dict[str, Any], **overrides) -> dict[str, Any]:
    """yaml 配置与 CLI 覆盖合并（覆盖值非 None 时生效）。"""
    out = dict(cfg)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out
