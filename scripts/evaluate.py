"""
检索评估（正式入口 6/7）：对当前特征库做个体级检索评估。

统一度量入口（E1-E5 的专项实验脚本在 experiments/，本入口用于正式特征的
日常评估，不复制实验配置）：

    python scripts/evaluate.py                       # 留一检索评估（R@1 / mAP，个体级）
    python scripts/evaluate.py --mode pairs          # 同个体/跨个体余弦分布 + 阈值建议

输入特征带 confirmed_identity（人工初审标签，Candidate 级）时按个体划分
query/gallery：多图个体抽 1 张作 query，其余入库；单图个体整图入库；
正确个体必在库中（R@1 有定义）。图对个体分数 = max cos（多帧聚合语义）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.reid.evaluation import split_query_gallery  # noqa: E402
from whitewhale.reid.retrieval import cosine_topk  # noqa: E402


def _load(embeddings: Path, meta: Path):
    emb = np.load(embeddings)
    m = pd.read_csv(meta)
    assert len(emb) == len(m), "特征与 meta 行数不一致"
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb, m


def _individual_ap(ind_scores: np.ndarray, true_col: np.ndarray) -> float:
    """个体级 mAP：对每个 query 的个体分数排序，取真值个体位置计算 AP。"""
    aps = []
    for row, t in zip(ind_scores, true_col):
        order = np.argsort(-row)
        rank_pos = int(np.where(order == t)[0][0]) + 1
        aps.append(1.0 / rank_pos)
    return float(np.mean(aps))


def eval_retrieval(emb, meta, seed: int):
    """个体级 leave-one-out 检索：R@1 + mAP（正确个体必在库中）。"""
    keep = meta["confirmed_identity"].notna() & (
        meta["confirmed_identity"].astype(str).str.strip() != "")
    emb, meta = emb[keep.to_numpy()], meta[keep].copy()
    meta["ind"] = meta["confirmed_identity"].astype(str)
    if meta["ind"].nunique() < 2:
        raise SystemExit(f"已确认个体数不足（{meta['ind'].nunique()}），无法评估。"
                         f"请先完成人工初审（confirmed_identity 非空）。")
    q, g = split_query_gallery(meta, identity_col="ind", seed=seed)
    q_idx = [list(meta["image_id"]).index(iid) for iid in q["image_id"]]
    g_idx = [list(meta["image_id"]).index(iid) for iid in g["image_id"]]
    # 图对库照片级 Top-K → 图对个体 = max cos（多帧聚合）
    scores, idx = cosine_topk(emb[q_idx], emb[g_idx], k=len(g_idx))
    ind_names = sorted(g["ind"].unique())
    name2col = {name: i for i, name in enumerate(ind_names)}
    g_inds = g["ind"].to_numpy()
    ind_scores = np.full((len(q_idx), len(ind_names)), -1.0)
    for gi, (score_row, id_row) in enumerate(zip(scores, idx)):
        for s, gi_id in zip(score_row, id_row):
            col = name2col[g_inds[gi_id]]
            if s > ind_scores[gi, col]:
                ind_scores[gi, col] = s
    true_col = np.array([name2col[v] for v in q["ind"].to_numpy()])
    r1 = float((ind_scores.argmax(axis=1) == true_col).mean())
    ap = _individual_ap(ind_scores, true_col)
    print(f"[eval] 个体级检索（query {len(q)} 张 / gallery {len(g)} 张 / "
          f"{len(ind_names)} 个体）")
    print(f"[eval] R@1 = {r1:.3f}   mAP = {ap:.3f}")
    return {"n_query": int(len(q)), "n_gallery": int(len(g)),
            "n_individuals": len(ind_names), "r1": round(r1, 4),
            "map": round(ap, 4), "seed": seed}


def eval_pairs(emb, meta):
    """同个体/跨个体余弦分布 + 阈值建议（FA≤5% 拒识标定，对照 E4 口径）。"""
    keep = meta["confirmed_identity"].notna() & (
        meta["confirmed_identity"].astype(str).str.strip() != "")
    emb, meta = emb[keep.to_numpy()], meta[keep].copy()
    ind = meta["confirmed_identity"].astype(str).to_numpy()
    n = len(emb)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)
    same, cross = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (same if ind[i] == ind[j] else cross).append(float(sim[i, j]))
    same, cross = np.array(same), np.array(cross)
    # FA≤5%：跨个体对 top5% 的分数作阈值（更严格 → 更低的 FA）
    fa5 = float(np.quantile(cross, 0.95)) if len(cross) else np.nan
    fa1 = float(np.quantile(cross, 0.99)) if len(cross) else np.nan
    print(f"[eval] 同个体对 {len(same)} / 跨个体对 {len(cross)}")
    print(f"[eval] 同个体相似度: 中位 {np.median(same):.3f}（p25 {np.quantile(same, .25):.3f}）")
    print(f"[eval] 跨个体相似度: 中位 {np.median(cross):.3f}（p95 {np.quantile(cross, .95):.3f}）")
    print(f"[eval] FA≤5% 阈值建议 ≈ {fa5:.3f}（FA≤1% ≈ {fa1:.3f}，参考值非铁律）")
    return {"n_same": int(len(same)), "n_cross": int(len(cross)),
            "same_median": round(float(np.median(same)), 3),
            "cross_median": round(float(np.median(cross)), 3),
            "fa5_threshold": round(fa5, 3) if not np.isnan(fa5) else None,
            "fa1_threshold": round(fa1, 3) if not np.isnan(fa1) else None}


def main():
    base = REPO_ROOT / "outputs"
    parser = argparse.ArgumentParser(description="特征库检索评估（正式度量入口）")
    parser.add_argument("--embeddings", type=Path,
                        default=base / "embeddings" / "embeddings_metric_r3_yolocrop.npy",
                        help="特征（meta 需含 confirmed_identity）")
    parser.add_argument("--meta", type=Path,
                        default=base / "embeddings" / "embeddings_metric_r3_yolocrop_meta.csv")
    parser.add_argument("--mode", choices=["retrieval", "pairs"], default="retrieval",
                        help="retrieval = 个体级 R@1/mAP；pairs = 分数分布 + 阈值建议")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=base / "reports" / "evaluate")
    args = parser.parse_args()

    emb, meta = _load(args.embeddings, args.meta)
    if args.mode == "retrieval":
        res = eval_retrieval(emb, meta, args.seed)
    else:
        res = eval_pairs(emb, meta)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.mode}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] → {args.out / f'{args.mode}.json'}")


if __name__ == "__main__":
    main()
