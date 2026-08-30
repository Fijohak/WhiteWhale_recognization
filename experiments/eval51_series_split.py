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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.eval51_common import (  # noqa: E402
    evaluate_probe_clusters,
    load_data,
    split_probe_gallery_series,
    summarize_probe_clusters,
)


def format_metric(value) -> str:
    """把空分组指标显示为 N/A，避免无该类 session 时格式化崩溃。"""
    return "N/A" if value is None else f"{float(value):.3f}"


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
    if args.k < 1:
        ap.error("--k 必须为正整数")

    meta, emb = load_data(base, stem=args.feats_stem)
    q_rows, gal_idx, gal_ind, info = split_probe_gallery_series(
        meta, emb, seed=args.seed)

    # 两个 E5 入口共用同一评估函数：同 session 隔离 + 完整同串剔除。
    df = evaluate_probe_clusters(
        meta, emb, q_rows, gal_idx, gal_ind, reciprocal_rank_k=args.k)

    results = {"overall": summarize_probe_clusters(df, args.k)}
    hist = df[df["session"].isin(["20140806 01", "20140806 03"])]
    newb = df[~df["session"].isin(["20140806 01", "20140806 03"])]
    results["history_20140806"] = summarize_probe_clusters(hist, args.k)
    results["new_batches"] = summarize_probe_clusters(newb, args.k)
    results["by_session"] = {
        str(session): summarize_probe_clusters(group, args.k)
        for session, group in df.groupby("session", sort=True)
    }
    summary = results["overall"]

    results["_meta"] = {
        "protocol_version": "e5_session_local_cross_series_v1",
        "features_stem": args.feats_stem,
        "seed": int(args.seed), **info,
        "reciprocal_rank_cutoff": int(args.k),
        "identity_scope": "session_local_confirmed_identity",
        "n_query_images_evaluable": int(summary["n_query_images"]),
        "n_query_images_skipped": int(summary["n_query_images_skipped"]),
        "note": "串抽样版：每串随机挑 ceil(串长/2) 张，挑剩的进干扰池"
                "（在库不作候选）；挑出的按整串分边进 query/库，"
                "每张 query 仅对同 session 的批次内身份打分，跨 session 未对齐"
                "身份不作候选或负类，并再次执行完整同串剔除。"
                "正确身份无有效 gallery 时计为 skipped，不进入指标分母。"
                f"单真值簇排名报告 MRR@{args.k}，不是 mAP。",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "" if args.feats_stem == "embeddings_eval51_all" else f"_{args.feats_stem.replace('embeddings_eval51_all', '')}"
    out_json = args.out / f"metrics_series_split{tag}.json"
    out_csv = args.out / f"per_individual_series_split{tag}.csv"
    out_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"探针簇 {summary['n_probe_clusters']}/"
          f"{summary['n_probe_clusters_total']} 个可评（query "
          f"{summary['n_query_images']}/{summary['n_query_images_total']} 张可评）| "
          f"全批次原始库 {info['n_gallery_images_total_all_sessions']} 张"
          f"（干扰池 {info['n_interference_pool_images']}）| "
          f"单串个体仅入库 {info['n_gallery_only_individuals']} 个")
    print(f"[整体] 单图 R@1={format_metric(results['overall']['single_R@1'])} | "
          f"簇级 R@1={format_metric(results['overall']['cluster_R@1'])} | "
          f"簇级 R@5={format_metric(results['overall']['cluster_R@5'])} | "
          f"MRR@{args.k}={format_metric(results['overall'][f'cluster_MRR@{args.k}'])}")
    print(f"[历史库] 单图 R@1={format_metric(results['history_20140806']['single_R@1'])} | "
          f"簇级 R@1={format_metric(results['history_20140806']['cluster_R@1'])}")
    print(f"[新批次] 单图 R@1={format_metric(results['new_batches']['single_R@1'])} | "
          f"簇级 R@1={format_metric(results['new_batches']['cluster_R@1'])}")
    print(f"[done] -> {out_json}")


if __name__ == "__main__":
    main()
