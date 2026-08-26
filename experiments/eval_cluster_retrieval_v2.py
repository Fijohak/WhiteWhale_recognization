"""
E5.1 簇级检索评估（全量版，2026-08-25）：多帧投票 vs 单图，9 批次全库。

E5（2026-08-17）只覆盖 01/03 两群（当时数据未到位），本实验把同一套
探针/库拆分设计（HappyWhale 同款）扩到全量 1040 张库：

- query：全部多图个体（≥2 张 labeled）照片按 seed 对半拆成 query 簇；
- gallery：其余全部特征（另一半照片 + 22 个单图个体 + 散图/忽略图
  无标签干扰项）——干扰项不构成个体候选，仅作为库规模与特征噪声；
- 单图对照：同一批 query 图逐张独立 Top-1（个体级命中率）；
- 簇级（多帧投票）：簇-个体分数 = 图-个体 max 后簇内 mean。

判定标准（强标签口径）：
- 同体标签 = 当年分组文件夹的个体编号（individual_id），组内照片必属同一
  个体——本实验以此作为命中判定的绝对标准（无此标准则实验无法判定成败）；
- 标签只用于评分（gallery 聚合与对答案），检索打分过程不使用标签；
  query 照片已从库中排除（对半拆），防"自己找自己"的平凡命中；
- 诚实边界：该强标签口径适用于"同批内检索能力"判定；跨批次生产性结论
  仍受"分组未人工核验"约束（E9 教训），需 3.6 核验后另行确认。
- r3 训练见过 01/03 部分个体（训练泄漏），历史库侧数字偏乐观；
- 理想簇上界：query 簇 = 个体照片（真实 HDBSCAN 簇含噪声，会低于此）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    base = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="E5.1 簇级检索评估（全量库）")
    ap.add_argument("--feats", type=Path,
                    default=base / "outputs" / "embeddings" / "embeddings_eval51_all.npy")
    ap.add_argument("--meta", type=Path,
                    default=base / "outputs" / "embeddings" / "embeddings_eval51_all_meta.csv")
    ap.add_argument("--out", type=Path,
                    default=base / "outputs" / "reports" / "cluster_retrieval_v2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7, help="探针/库拆分随机种子（同 E5）")
    args = ap.parse_args()

    meta = pd.read_csv(args.meta, dtype={"session_id": str})
    emb = np.load(args.feats)
    assert len(meta) == emb.shape[0], "特征行数与 meta 不一致"
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # 防御：确保 L2
    meta["ind"] = meta["individual_id"].fillna("").astype(str)
    rng = np.random.default_rng(args.seed)

    def score_img_to_individual(img_emb, gal_idx, gal_ind):
        """单张图 → 每个 gallery 个体：max over 该个体库照片 cos（无标签不入）。"""
        sims = img_emb @ emb[gal_idx].T
        out = {}
        for g in np.unique(gal_ind):
            if g == "":
                continue  # 无标签干扰项不构成个体候选
            mask = gal_ind == g
            out[str(g)] = float(sims[mask].max())
        return out

    # ---- 拆探针/库：多图个体对半拆，单图个体与无标签图整图入库 ----
    gal_rows, q_rows = [], {}
    for ind, sub in meta.groupby("ind"):
        idx = np.array(sub.index.to_numpy(), copy=True)  # index 只读，需复制
        if ind == "" or len(sub) < 2:
            gal_rows.append(idx)
            continue
        rng.shuffle(idx)
        n = len(idx)
        q_rows[ind] = idx[: n // 2]
        gal_rows.append(idx[n // 2:])
    gal_idx = np.concatenate(gal_rows)
    gal_ind = np.asarray([meta.loc[i, "ind"] for i in gal_idx])

    # ---- 逐个体评估：单图 vs 簇级 ----
    records = []
    for target, q_idx in q_rows.items():
        per_img = [score_img_to_individual(emb[i], gal_idx, gal_ind) for i in q_idx]
        all_g = sorted(per_img[0].keys())
        cluster = {g: float(np.mean([s[g] for s in per_img])) for g in all_g}
        top = sorted(cluster, key=cluster.get, reverse=True)
        hit_rank = next((r for r, g in enumerate(top[: args.k], 1) if g == target), 0)
        records.append({
            "individual": target,
            "session": str(meta.loc[q_idx[0], "session_id"]),
            "n_query": int(len(q_idx)),
            "vote1_ratio": float(sum(1 for s in per_img
                                     if max(s, key=s.get) == target) / len(q_idx)),
            "single_r1": float(sum(1 for s in per_img
                                   if max(s, key=s.get) == target) / len(q_idx)),
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
            "vote1_median": float(sub["vote1_ratio"].median()),
        }

    results = {"overall": summarize(df, "overall")}
    for sess, sub in df.groupby("session"):
        results[f"session_{sess}"] = summarize(sub, sess)

    # 01/03（历史库）与其余（新批次）合并口径，便于对照 E5/E10
    hist = df[df["session"].isin(["20140806 01", "20140806 03"])]
    newb = df[~df["session"].isin(["20140806 01", "20140806 03"])]
    results["history_20140806"] = summarize(hist, "hist")
    results["new_batches"] = summarize(newb, "new")

    results["_meta"] = {
        "feats": str(args.feats), "seed": args.seed,
        "n_total_images": int(len(meta)),
        "n_labeled_individuals": int(meta["ind"].ne("").nunique()),
        "n_multi_image_individuals": int(len(q_rows)),
        "note": "判定标准 = individual_id（当年分组文件夹的个体编号，本实验绝对真值）；"
                "标签仅评分用（gallery 聚合/对答案），检索打分不使用标签；"
                "query 照片已排除出库（防自身平凡命中）。"
                "r3 见过 01/03 部分个体（泄漏），历史库侧数字偏乐观；"
                "query=个体照片（理想簇上界）。",
    }
    # 输出文件名带特征后缀（r4 重训不覆盖 r3）：与保守/串抽样脚本的
    # _r4 后缀命名保持一致
    stem = Path(args.feats).stem
    tag = "" if stem == "embeddings_eval51_all" \
        else f"_{stem.replace('embeddings_eval51_all', '')}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"metrics{tag}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(args.out / f"per_individual{tag}.csv", index=False,
              encoding="utf-8-sig")

    print(f"总探针簇 {len(df)} 个（{int(df['n_query'].sum())} 张 query 图）| "
          f"库 {len(gal_idx)} 张")
    print(f"[整体] 单图 R@1={results['overall']['single_R@1']:.3f} | "
          f"簇级 R@1={results['overall']['cluster_R@1']:.3f} | "
          f"簇级 R@5={results['overall']['cluster_R@5']:.3f} | "
          f"簇级 mAP={results['overall']['cluster_mAP']:.3f} | "
          f"vote1 中位={results['overall']['vote1_median']:.2f}")
    print(f"[历史库 01/03] 单图 R@1={results['history_20140806'].get('single_R@1', float('nan')):.3f} | "
          f"簇级 R@1={results['history_20140806'].get('cluster_R@1', float('nan')):.3f}")
    print(f"[新批次] 单图 R@1={results['new_batches'].get('single_R@1', float('nan')):.3f} | "
          f"簇级 R@1={results['new_batches'].get('cluster_R@1', float('nan')):.3f}")
    print(f"[done] -> {args.out / f'metrics{tag}.json'}")


if __name__ == "__main__":
    main()
