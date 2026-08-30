"""
检索/聚类评估指标与数据划分（统一评估模块）。

本项目的 individual_id 是批次内已确认身份；跨批次身份仍未对齐。正式评估
可传完整 series 列，保证同一连拍串不会跨 query/gallery。
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd


def _stable_group_rng(seed: int, group: pd.DataFrame) -> np.random.Generator:
    """按组内稳定图像成员派生随机种子，避免仅因身份显示名变更而重切分。"""
    if "image_id" in group.columns:
        members = sorted(group["image_id"].astype(str).tolist())
    else:
        members = sorted(str(index) for index in group.index)
    digest = hashlib.sha256(
        (str(seed) + "\x1f" + "\x1f".join(members)).encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def recall_at_k(scores: np.ndarray, indices: np.ndarray,
                query_ids: np.ndarray, gallery_ids: np.ndarray,
                k_list: tuple[int, ...] = (1, 5, 10),
                gt_sets: list[set[int]] | None = None) -> dict[int, float]:
    """Recall@K。

    Args:
        scores: (Nq, k) 分数
        indices: (Nq, k) gallery 索引
        query_ids / gallery_ids: 身份标签（字符串数组）
        k_list: 要报告的 K 值
        gt_sets: 可选，官方匹配对模式：gt_sets[i] 是第 i 个 query 的
            正样本 gallery 索引集合（Beluga 竞赛协议，同身份图不算正样本，
            避免自匹配虚高）。None 时用身份相等判断。

    Returns:
        {k: recall}
    """
    results = {}
    n_query = indices.shape[0]
    for k in k_list:
        kk = min(k, indices.shape[1])
        hits = 0
        for i in range(n_query):
            cand_idx = indices[i, :kk]
            if gt_sets is not None:
                if any(j in gt_sets[i] for j in cand_idx):
                    hits += 1
            else:
                cand_ids = gallery_ids[cand_idx]
                if (cand_ids == query_ids[i]).any():
                    hits += 1
        results[k] = hits / n_query
    return results


def mean_average_precision(scores: np.ndarray, indices: np.ndarray,
                           query_ids: np.ndarray, gallery_ids: np.ndarray,
                           gt_sets: list[set[int]] | None = None) -> float:
    """mAP：每个 query 的 AP 平均。gt_sets 语义同 recall_at_k。"""
    aps = []
    for i in range(indices.shape[0]):
        cand_idx = indices[i]
        if gt_sets is not None:
            mask = np.array([j in gt_sets[i] for j in cand_idx], dtype=bool)
        else:
            mask = gallery_ids[cand_idx] == query_ids[i]
        if not mask.any():
            aps.append(0.0)
            continue
        n_pos = mask.sum()
        ap = 0.0
        n_hit = 0
        for j, is_pos in enumerate(mask):
            if is_pos:
                n_hit += 1
                ap += n_hit / (j + 1)
        aps.append(ap / n_pos)
    return float(np.mean(aps)) if aps else 0.0


def split_query_gallery(df: pd.DataFrame, identity_col: str = "identity",
                        seed: int = 42,
                        series_col: str | None = None
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """query/gallery 划分防泄漏：可把完整连拍串作为最小划分单元。

    ``series_col`` 为空时保持通用数据集的单图抽样行为。提供时，以全局
    series 为不可分单元选 query；不足两个 series 的身份只进 gallery，
    并确保每个 query 身份都有跨串同体库图。
    """
    q_idx, g_idx = [], []
    if series_col is not None and series_col not in df.columns:
        raise ValueError(f"缺少 series 列: {series_col}")

    # 单图模式保持原有的通用划分行为。
    if series_col is None:
        for _, grp in df.groupby(identity_col, sort=False):
            rng = _stable_group_rng(seed, grp)
            idx = list(grp.index)
            if len(idx) < 2:
                g_idx.extend(idx)
                continue
            rng.shuffle(idx)
            q_idx.append(idx[0])
            g_idx.extend(idx[1:])
        return df.loc[q_idx].copy(), df.loc[g_idx].copy()

    # series 可能同时包含多个身份，因此不能再按身份各自独立
    # 切分。贪心选择全局 query series，每次都保证该串中的所有
    # 身份仍至少有一个其他 series 留在 gallery。
    units = df[series_col].fillna("").astype(str).copy()
    empty = units.str.strip() == ""
    units.loc[empty] = [f"__single_{index}" for index in units.index[empty]]

    identity_groups = list(df.groupby(identity_col, sort=False))
    identity_units = {
        identity: set(units.loc[grp.index].tolist())
        for identity, grp in identity_groups
    }
    unit_identities: dict[str, set[object]] = {}
    for identity, unit_set in identity_units.items():
        for unit in unit_set:
            unit_identities.setdefault(unit, set()).add(identity)

    query_units: set[str] = set()
    for identity, grp in identity_groups:
        rng = _stable_group_rng(seed, grp)
        unique_units = list(dict.fromkeys(units.loc[grp.index].tolist()))
        if len(unique_units) < 2:
            continue
        if identity_units[identity] & query_units:
            continue
        rng.shuffle(unique_units)
        for query_unit in unique_units:
            trial = query_units | {query_unit}
            affected = unit_identities[query_unit]
            if all(identity_units[value] - trial for value in affected):
                query_units.add(query_unit)
                break

    query_mask = units.isin(query_units)
    query = df.loc[query_mask].copy()
    gallery = df.loc[~query_mask].copy()
    query_series = set(units.loc[query.index])
    gallery_series = set(units.loc[gallery.index])
    if not query_series.isdisjoint(gallery_series):
        raise AssertionError("series 划分泄漏：同一完整串同时出现在 query/gallery")
    if not set(query[identity_col]).issubset(set(gallery[identity_col])):
        raise AssertionError("query 中存在没有跨串 gallery 正样本的身份")
    return query, gallery
