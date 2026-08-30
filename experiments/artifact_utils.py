"""
实验特征产物读取工具。

负责用 embedding 邻接 meta 的 image_id 恢复真实行序，再与当前清单对齐，
避免清单追加/重排后把特征配给错误照片。
"""
from pathlib import Path

import numpy as np
import pandas as pd


def embedding_meta_path(embeddings_path: Path) -> Path:
    """返回统一提取接口为 embedding 写出的邻接 meta 路径。"""
    return embeddings_path.with_name(f"{embeddings_path.stem}_meta.csv")


def load_aligned_embeddings(
    embeddings_path: Path,
    source_csv: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    """按 embedding meta 的 image_id 行序对齐源清单与特征。"""
    embeddings_path = Path(embeddings_path)
    source_csv = Path(source_csv)
    meta_path = embedding_meta_path(embeddings_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"特征缺少行序文件，请重新提取: {meta_path}")

    emb = np.load(embeddings_path)
    meta = pd.read_csv(meta_path, dtype={"image_id": str})
    source = pd.read_csv(source_csv, dtype={"image_id": str})
    for name, frame, path in (("特征 meta", meta, meta_path),
                              ("源清单", source, source_csv)):
        if "image_id" not in frame.columns:
            raise ValueError(f"{name} 缺少 image_id 列: {path}")
        if frame["image_id"].isna().any() or frame["image_id"].duplicated().any():
            raise ValueError(f"{name} 的 image_id 为空或重复: {path}")
    if len(meta) != len(emb):
        raise ValueError(
            f"特征与行序数量不一致: {embeddings_path}={len(emb)}, "
            f"{meta_path}={len(meta)}")

    order = meta[["image_id"]].copy()
    order["_embedding_row"] = np.arange(len(order))
    aligned = order.merge(
        source, on="image_id", how="left", validate="one_to_one", indicator=True)
    missing = aligned.loc[aligned["_merge"] != "both", "image_id"].tolist()
    if missing:
        raise ValueError(
            f"特征 meta 中有 {len(missing)} 个 image_id 不在源清单 {source_csv}: "
            f"{missing[:5]}")
    aligned = aligned.sort_values("_embedding_row", kind="stable").drop(
        columns=["_embedding_row", "_merge"])
    return aligned.reset_index(drop=True), emb
