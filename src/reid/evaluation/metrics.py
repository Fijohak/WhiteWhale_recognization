"""
统一评估接口（仅用于有可靠 individual_id 的公开数据）。

方向调整后：本土数据无 Ground Truth，禁止调用本模块（会报错提示）。

指标：
- Recall@K：query 的真实 identity 是否出现在 Top-K 候选（同 identity 除 query 自身）
- mAP：mean Average Precision
"""
from __future__ import annotations

import numpy as np


def recall_at_k(scores: np.ndarray, indices: np.ndarray,
                query_ids: np.ndarray, gallery_ids: np.ndarray,
                k_list: tuple[int, ...] = (1, 5, 10)) -> dict[int, float]:
    """Recall@K。

    Args:
        scores: (Nq, k) 分数
        indices: (Nq, k) gallery 索引
        query_ids / gallery_ids: 身份标签（字符串数组）
        k_list: 要报告的 K 值

    Returns:
        {k: recall}
    """
    results = {}
    n_query = indices.shape[0]
    for k in k_list:
        kk = min(k, indices.shape[1])
        hits = 0
        for i in range(n_query):
            cand_ids = gallery_ids[indices[i, :kk]]
            # 同一身份命中（query 自身不在 gallery 中，无自匹配问题）
            if (cand_ids == query_ids[i]).any():
                hits += 1
        results[k] = hits / n_query
    return results


def mean_average_precision(scores: np.ndarray, indices: np.ndarray,
                           query_ids: np.ndarray, gallery_ids: np.ndarray) -> float:
    """mAP：每个 query 的 AP 平均。"""
    aps = []
    for i in range(indices.shape[0]):
        cand_ids = gallery_ids[indices[i]]
        mask = cand_ids == query_ids[i]
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
