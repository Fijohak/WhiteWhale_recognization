"""
统一检索接口。

当前实现 NumPy 全量 cosine Top-K（数据量小足够）；接口设计保证后续可替换为 Faiss
（同签名，只需改后端）。embedding 约定为 L2 归一化（提取时已完成）。

语义（方向调整后）：
- 只支持 query / gallery 分离（防泄漏：同 encounter 不跨 split）；
- 输出 Top-K 候选 + 分数，全部为"候选"，不自动判定身份。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def faiss_topk(query_emb: np.ndarray, gallery_emb: np.ndarray,
               k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Faiss 后端（可选）。需要 faiss 已安装；接口与 cosine_topk 一致。"""
    import faiss  # noqa: PLC0415

    def _norm(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    q = np.ascontiguousarray(_norm(query_emb), dtype=np.float32)
    g = np.ascontiguousarray(_norm(gallery_emb), dtype=np.float32)
    index = faiss.IndexFlatIP(g.shape[1])  # inner product = cosine (L2 归一化后)
    index.add(g)
    scores, indices = index.search(q, min(k, g.shape[0]))
    return scores.astype(np.float32), indices.astype(np.int64)
