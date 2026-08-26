"""
E5.1 串抽样版评估（用户方案，2026-08-25）：同串平凡命中的替代解法。

背景：保守口径（打分前剔除同串照片）证实新批次 55% 命中是同串平凡命中，
用户决定不把连拍个体整个丢进干扰池——连拍每张照片都有差异信息，
而是对连拍串做随机抽样减半：

- 每串随机挑出 ceil(串长/2) 张（11 张挑 ≤6 张），挑剩的进干扰池
  （在库、不构成个体候选）；
- 挑出的照片以整串为单位随机分边：整串进 query 或整串进库，
  多串个体兜底保证两侧都有（可评估）；
- 结果：query 与库中候选永远不同串 → 平凡命中被根除，
  连拍照片的一半信息仍保留在库特征中；
- 单串个体（全串连拍）照片全部入库存作检索目标、不出 query，
  其无法拆出跨串 query，如实报告数量。

输出：metrics_series_split.json（整体/历史库/新批次/按 session），
per_individual_series_split.csv（逐个体），与常规/保守口径对照。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.eval51_common import (  # noqa: E402
    load_data, score_img_to_individual, split_probe_gallery_series)


def main():
    base = REPO_ROOT
    ap = argparse.ArgumentParser(description="E5.1 串抽样版评估（防平凡命中）")
    ap.add_argument("--out", type=Path,
                    default=base / "outputs" / "reports" / "cluster_retrieval_v2")
    ap.add_argument("--feats-stem", type=str, default="embeddings_eval51_all",
                    help="特征文件名主干（r4 重训后传 embeddings_eval51_all_r4）")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    meta, emb = load_data(base, stem=args.feats_stem)
    q_rows, gal_idx, gal_ind, info = split_probe_gallery_series(
        meta, emb, seed=args.seed)

    # ---- 逐个体评估（同常规口径，但库中无同串候选） ----
    records = []
    for target, q_idx in q_rows.items():
        per_img = [score_img_to_individual(emb[i], emb, gal_idx, gal_ind)
                   for i in q_idx]
        all_g = sorted(per_img[0].keys())
        cluster = {g: float(np.mean([s[g] for s in per_img])) for g in all_g}
        top = sorted(cluster, key=cluster.get, reverse=True)
        hit_rank = next((r for r, g in enumerate(top[: args.k], 1)
                         if g == target), 0)
        records.append({
            "individual": target,
            "session": str(meta.loc[q_idx[0], "session_id"]),
            "n_query": int(len(q_idx)),
            "single_r1": float(np.mean([max(s, key=s.get) == target
                                        for s in per_img])),
            "cluster_r1": int(top[0] == target),
            "cluster_r5": int(0 < hit_rank <= 5),
            "ap": float(1.0 / hit_rank) if hit_rank else 0.0,
        })
    df = pd.DataFrame(records)

    def summarize(sub, name):
        if len(sub) == 0:
            return {"n_probe_clusters": 0}
        return {
            "n_probe_clusters": int(len(sub)),
            "n_query_images": int(sub["n_query"].sum()),
            "n_gallery_images": int(len(gal_idx)),
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
        "seed": args.seed, **info,
        "note": "串抽样版：每串随机挑 ceil(串长/2) 张，挑剩的进干扰池"
                "（在库不作候选）；挑出的按整串分边进 query/库，"
                "query 与库候选不同串 → 无平凡命中。"
                "单串个体照片全部入库存作检索目标、不出 query。",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "" if args.feats_stem == "embeddings_eval51_all" else f"_{args.feats_stem.replace('embeddings_eval51_all', '')}"
    out_json = args.out / f"metrics_series_split{tag}.json"
    out_csv = args.out / f"per_individual_series_split{tag}.csv"
    out_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"探针簇 {len(df)} 个（{info['n_query_images']} 张 query 图）| "
          f"库 {len(gal_idx)} 张（干扰池 {info['n_interference_pool_images']}）| "
          f"单串个体仅入库 {info['n_gallery_only_individuals']} 个")
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
