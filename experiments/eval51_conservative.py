"""
E5.1 保守口径评估：剔除同串连拍后的同体检索命中率。

动机（用户指出的过拟合风险）：query 与库照片若来自同一连拍串
（连拍号差 ≤ MAX_GAP=2），是"平凡命中"——相邻帧同一次目击、
相似度天然 ≈1.0，不能证明个体识别能力。

保守规则：
- 每张 query 图：从库中剔除与其同串（连拍号差 ≤2）的照片后再打分；
- 剔除后若该个体在库中无照片 → 该 query 无法评估（不计入分母，
  单独统计数量）；
- 簇级分数 = 簇内各图分数对每个候选个体的 mean（每图库子集不同）。

输出：整体 / 历史库 / 新批次 / 按 session，与常规口径对照。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.eval51_common import (  # noqa: E402
    exclude_same_series, load_data, score_img_to_individual, split_probe_gallery)


def main():
    base = REPO_ROOT
    ap = argparse.ArgumentParser(description="E5.1 保守口径（剔除同串连拍）")
    ap.add_argument("--out", type=Path,
                    default=base / "outputs" / "reports" / "cluster_retrieval_v2")
    ap.add_argument("--feats-stem", type=str, default="embeddings_eval51_all",
                    help="特征文件名主干（r4 重训后传 embeddings_eval51_all_r4）")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    meta, emb = load_data(base, stem=args.feats_stem)
    q_rows, gal_idx, gal_ind = split_probe_gallery(meta, emb)

    records, skipped = [], 0
    for target, q_idx in q_rows.items():
        # 每张 query 图：剔除同串后的库子集 + 个体分数
        per_img = []
        for qi in q_idx:
            g_idx, g_ind = exclude_same_series(meta, qi, gal_idx, gal_ind)
            scores = score_img_to_individual(emb[qi], emb, g_idx, g_ind)
            if target not in scores:
                skipped += 1  # 剔除同串后正确个体不在库中 → 无法评估
                continue
            per_img.append(scores)
        if not per_img:
            continue
        # 簇级：候选个体 = 各图并集，分数 = 各图 mean（缺失图忽略）
        cands = sorted({g for s in per_img for g in s})
        cluster = {}
        for g in cands:
            vals = [s[g] for s in per_img if g in s]
            cluster[g] = float(np.mean(vals))
        top = sorted(cluster, key=cluster.get, reverse=True)
        hit_rank = next((r for r, g in enumerate(top[: args.k], 1)
                         if g == target), 0)
        hit1 = top[0] == target
        records.append({
            "individual": target,
            "session": str(meta.loc[q_idx[0], "session_id"]),
            "n_query": int(len(q_idx)),
            "single_r1": float(np.mean([max(s, key=s.get) == target
                                        for s in per_img])),
            "cluster_r1": int(hit1),
            "cluster_r5": int(0 < hit_rank <= 5),
            "ap": float(1.0 / hit_rank) if hit_rank else 0.0,
        })
    df = __import__("pandas").DataFrame(records)

    def summarize(sub, name):
        if len(sub) == 0:
            return {"n_probe_clusters": 0}
        return {
            "n_probe_clusters": int(len(sub)),
            "n_query_images": int(sub["n_query"].sum()),
            "single_R@1": float(sub["single_r1"].mean()),
            "cluster_R@1": float(sub["cluster_r1"].mean()),
            "cluster_R@5": float(sub["cluster_r5"].mean()),
            "cluster_mAP": float(sub["ap"].mean()),
        }

    results = {"overall": summarize(df, "overall")}
    hist = df[df["session"].isin(["20140806 01", "20140806 03"])]
    newb = df[~df["session"].isin(["20140806 01", "20140806 03"])]
    results["history_20140806"] = summarize(hist, "hist")
    results["new_batches"] = summarize(newb, "new")
    results["_meta"] = {
        "skipped_queries_no_same_series_in_gallery": int(skipped),
        "note": "保守口径：query 打分前剔除库中同串（连拍号差≤2）照片，"
                "仅统计跨串命中；剔除后正确个体不在库中的 query 跳过。",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "" if args.feats_stem == "embeddings_eval51_all" \
        else f"_{args.feats_stem.replace('embeddings_eval51_all', '')}"
    out_json = args.out / f"metrics_conservative{tag}.json"
    out_csv = args.out / f"per_individual_conservative{tag}.csv"
    out_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"探针簇 {len(df)} 个 | 跳过(库中无跨串同体) {skipped} 张 query")
    print(f"[整体] 单图 R@1={results['overall'].get('single_R@1', float('nan')):.3f} | "
          f"簇级 R@1={results['overall'].get('cluster_R@1', float('nan')):.3f} | "
          f"簇级 R@5={results['overall'].get('cluster_R@5', float('nan')):.3f} | "
          f"mAP={results['overall'].get('cluster_mAP', float('nan')):.3f}")
    print(f"[历史库] 单图 R@1={results['history_20140806'].get('single_R@1', float('nan')):.3f} | "
          f"簇级 R@1={results['history_20140806'].get('cluster_R@1', float('nan')):.3f}")
    print(f"[新批次] 单图 R@1={results['new_batches'].get('single_R@1', float('nan')):.3f} | "
          f"簇级 R@1={results['new_batches'].get('cluster_R@1', float('nan')):.3f}")
    print(f"[done] -> {out_json}")


if __name__ == "__main__":
    main()
