"""
检索/聚类评估指标与数据划分（统一评估模块）。

注意：本土数据无 Ground Truth（目录不直接等于身份），本模块的指标
只用于有可靠 individual_id 的公开数据或 leave-one-out 弱标签口径。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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
                        seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """query/gallery 划分防泄漏：同 identity 的 query 与 gallery 不共用同图。

    每个多图身份随机抽 1 张作 query，其余进 gallery；单图身份只进 gallery。
    """
    rng = np.random.default_rng(seed)
    q_idx, g_idx = [], []
    for iid, grp in df.groupby(identity_col):
        idx = list(grp.index)
        if len(idx) < 2:
            g_idx.extend(idx)  # 单图身份只能当 gallery
            continue
        rng.shuffle(idx)
        q_idx.append(idx[0])
        g_idx.extend(idx[1:])
    return df.loc[q_idx].copy(), df.loc[g_idx].copy()
