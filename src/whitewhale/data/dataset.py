"""
审核相关数据集工具（load_review_dataset 等）。

load_review_dataset 原属 scripts/fiftyone_review.py（该工具整体被
review_app 取代，仅保留此通用函数供其他入口复用）：
- clusters.csv 候选簇表 → 附加绝对 source_path（可追溯）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from whitewhale.data.image_store import ImageStore


def load_review_dataset(clusters_csv: Path, images_root: Path) -> pd.DataFrame:
    """加载聚类结果，拼出绝对图片路径（可追溯）。"""
    df = pd.read_csv(clusters_csv)
    store = ImageStore(images_root)
    df["source_path"] = df["relative_path"].map(
        lambda p: str(store.resolve(p)))
    return df
