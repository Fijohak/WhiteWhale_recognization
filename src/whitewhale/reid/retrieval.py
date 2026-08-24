"""
统一检索接口：cosine Top-K 与"图 → 个体"打分。

embedding 约定为 L2 归一化（提取时已完成），cosine 相似度可直接点乘。
"""
from __future__ import annotations

import numpy as np


def cosine_topk(query_emb: np.ndarray, gallery_emb: np.ndarray,
                k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """cosine Top-K 检索。

    Args:
        query_emb: (Nq, D) L2 归一化 embedding
        gallery_emb: (Ng, D) L2 归一化 embedding
        k: 返回 K 个候选

    Returns:
        (scores, indices): 每个 query 的 (k,) 分数与 gallery 索引，分数降序
    """
    # L2 归一化兜底（防未归一化输入）
    def _norm(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    q = _norm(query_emb)
    g = _norm(gallery_emb)
    sim = q @ g.T  # (Nq, Ng)

    k = min(k, sim.shape[1])
    scores = np.zeros((sim.shape[0], k), dtype=np.float32)
    indices = np.zeros((sim.shape[0], k), dtype=np.int64)
    for i in range(sim.shape[0]):
        idx = np.argsort(-sim[i])[:k]
        indices[i] = idx
        scores[i] = sim[i, idx]
    return scores, indices


def score_img_to_individual(img_emb: np.ndarray, gal_emb: np.ndarray,
                            gal_ind: np.ndarray) -> dict[str, float]:
    """单张图 → 每个历史库个体：max over 该个体照片 cos。返回 {ind: score}。

    归档/匹配场景的图-个体分数定义（E5 簇级检索沿用同一口径）。
    """
    sims = img_emb @ gal_emb.T
    out = {}
    for g in np.unique(gal_ind):
        mask = gal_ind == g
        out[str(g)] = float(sims[mask].max())
    return out
