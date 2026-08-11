"""
Top-K 检索。
对 Pilot Set embedding 做同调查内（in-session）的余弦相似度 Top-K 检索，
按 labeled 个体计算 rank-1 / rank-5 命中率，作为基线评估指标。

检索仅在同一 session 内进行（跨调查同名编号未合并，不参与比较）。
支持 numpy（N×D 全量计算，406 张规模足够）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def topk_retrieval(embeddings_path: Path, meta_path: Path, pilot_path: Path,
                   out_dir: Path, k: int = 5, metric: str = "cosine"):
    emb = np.load(embeddings_path)
    meta = pd.read_csv(meta_path)
    pilot = pd.read_csv(pilot_path)

    # 对齐：meta 与 pilot 顺序一致（均由 build_pilot_set 生成）
    assert len(emb) == len(meta), f"embedding {len(emb)} 与 meta {len(meta)} 数量不一致"
    df = meta.merge(pilot, on="image_id", how="left", suffixes=("", "_pilot"))
    assert len(df) == len(meta), "merge 后行数变化，请检查 image_id 是否唯一"
    df["session_id"] = df["session_id"].astype(str)

    if metric == "cosine":
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        sim = emb @ emb.T
    else:
        raise ValueError(f"不支持的 metric: {metric}")

    results = []
    for idx, row in df.iterrows():
        # 同 session 内检索，排除自身
        same_session = df["session_id"] == row["session_id"]
        mask = same_session & (np.arange(len(df)) != idx)
        scores = sim[idx, mask.values]
        cand = df[mask.values].copy()
        cand["score"] = scores
        top = cand.sort_values("score", ascending=False).head(k)
        results.append({
            "image_id": row["image_id"],
            "session_id": row["session_id"],
            "split": row["split"],
            "individual_id": row["individual_id"],
            "sequence_guess": row.get("sequence_guess", None),
            "filename": row.get("filename", None),
            "retrieved": top["image_id"].tolist(),
            "retrieved_individuals": top["individual_id"].tolist(),
            "retrieved_scores": [float(s) for s in top["score"].tolist()],
        })
    res = pd.DataFrame(results)

    # 评估：仅对 labeled 个体（有真实标签）计算 rank 命中
    labeled_idx = res.index[res["split"] == "labeled"]
    labeled = res.loc[labeled_idx]
    def rank_hit(row, kk):
        return bool((np.array(row["retrieved_individuals"][:kk]) == row["individual_id"]).any())
    for kk in (1, 5):
        hits = np.zeros(len(res), dtype=bool)
        hits[labeled_idx] = labeled.apply(lambda r: rank_hit(r, kk), axis=1).values
        res[f"rank{kk}_hit"] = hits

    stats = {
        "total": len(res),
        "labeled": int((res["split"] == "labeled").sum()),
        "k": k,
        "rank1": float(res.loc[res["split"] == "labeled", "rank1_hit"].mean()),
        "rank5": float(res.loc[res["split"] == "labeled", "rank5_hit"].mean()),
        "loose_retrieved_top1_individual": res[res["split"] == "loose_known"].apply(
            lambda r: r["retrieved_individuals"][0] if len(r["retrieved_individuals"]) else None,
            axis=1).value_counts().head(10).to_dict(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    # 列表列 JSON 序列化后再存 CSV，保证读回可解析
    res_out = res.copy()
    for c in ("retrieved", "retrieved_individuals", "retrieved_scores"):
        res_out[c] = res_out[c].apply(json.dumps, ensure_ascii=False)
    res_out.to_csv(out_dir / "topk_results.csv", index=False)
    with open(out_dir / "topk_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Top-K 检索完成: {len(res)} 张（labeled {len(labeled)} 参与评估）")
    print(f"  rank-1: {stats['rank1']:.3f}  rank-5: {stats['rank5']:.3f}")
    print(f"  → {out_dir / 'topk_results.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="同调查内 Top-K 检索与基线评估")
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser.add_argument("--embeddings", type=Path, default=base / "embeddings" / "embeddings.npy")
    parser.add_argument("--meta", type=Path, default=base / "embeddings" / "embeddings_meta.csv")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--out", type=Path, default=base / "retrieval")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    topk_retrieval(args.embeddings, args.meta, args.pilot, args.out, args.k)
