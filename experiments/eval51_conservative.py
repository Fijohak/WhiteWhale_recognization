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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.eval51_common import (  # noqa: E402
    evaluate_probe_clusters,
    load_data,
    split_probe_gallery,
    summarize_probe_clusters,
)


def format_metric(value) -> str:
    """把空分组指标显示为 N/A，避免无该类 session 时格式化崩溃。"""
    return "N/A" if value is None else f"{float(value):.3f}"


def main():
    base = REPO_ROOT
    ap = argparse.ArgumentParser(description="E5.1 保守口径（剔除同串连拍）")
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
    q_rows, gal_idx, gal_ind = split_probe_gallery(meta, emb, seed=args.seed)
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
        "seed": int(args.seed),
        "reciprocal_rank_cutoff": int(args.k),
        "identity_scope": "session_local_confirmed_identity",
        "n_query_images_evaluable": int(summary["n_query_images"]),
        "n_query_images_skipped": int(summary["n_query_images_skipped"]),
        "note": "保守口径：每张 query 只对同 session 的批次内身份打分，"
                "跨 session 未对齐身份不作为候选或负类；随后完整剔除同串照片。"
                "正确身份无跨串 gallery 时如实计为 skipped，不进入指标分母。"
                f"单真值簇排名报告 MRR@{args.k}，不是 mAP。",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "" if args.feats_stem == "embeddings_eval51_all" \
        else f"_{args.feats_stem.replace('embeddings_eval51_all', '')}"
    out_json = args.out / f"metrics_conservative{tag}.json"
    out_csv = args.out / f"per_individual_conservative{tag}.csv"
    out_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"探针簇 {summary['n_probe_clusters']}/"
          f"{summary['n_probe_clusters_total']} 个可评 | "
          f"query {summary['n_query_images']}/"
          f"{summary['n_query_images_total']} 张可评 | "
          f"跳过 {summary['n_query_images_skipped']} 张")
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
