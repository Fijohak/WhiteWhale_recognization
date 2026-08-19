"""
簇级检索评估（2026-08-17，实验 E5）：多帧投票 vs 单图检索。

场景：真实归档流程中，新批次先批内聚类成 Candidate Cluster，再与历史库匹配。
本实验用现有数据模拟并对比两种匹配方式（探针/库拆分设计，同 HappyWhale）：

- 单图匹配：每张图独立检索 Top-K（当前基线，B 协议 R@1≈0.495）
- 簇级匹配（多帧投票）：簇内每张图对每个候选个体打分（图-个体 max），
  簇-个体分数 = 簇内 mean 聚合；簇 Top-1 = 排序第一的个体

评估口径：
1. 库内簇级匹配（个体级 R@1/R@5/mAP）：同群内，每个多帧个体 X 的照片按
   seed 拆成 query 簇（一半）与库内照片（一半）；簇检索全库，正确个体
   必须在库中（R@1 有定义）。单图对照 = 同一批 query 图逐张独立 Top-1。
2. 跨群簇级拒识（open-set）：03 群个体簇 → 01 群全库（全部库外）；
   known 侧 = 01 群内部簇匹配分数（正确个体在库中）。
   FA<=5% 时 known recall，对比 E4 单图（52-61%）。

注意：本实验用"个体照片"模拟理想簇（簇纯净），是簇级检索的**上界**；
真实 HDBSCAN 簇含噪声，会低于此。

特征：r3 跨群 HN 微调（embeddings_metric_r3.npy，行序 = pilot_set.csv）。

用法：
    python scripts/eval_cluster_retrieval.py
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
    parser = argparse.ArgumentParser(description="簇级检索评估（多帧投票 vs 单图）")
    parser.add_argument("--pilot", type=Path, default=base / "outputs" / "pilot" / "pilot_set.csv")
    parser.add_argument("--feats", type=Path,
                        default=base / "outputs" / "embeddings" / "embeddings_metric_r3.npy")
    parser.add_argument("--out", type=Path, default=base / "outputs" / "reports" / "cluster_retrieval")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--fa-cap", type=float, default=0.05, help="跨群拒识的 open-set FA 上限")
    parser.add_argument("--seed", type=int, default=7, help="探针/库拆分随机种子")
    args = parser.parse_args()

    p = pd.read_csv(args.pilot)
    emb = np.load(args.feats)
    assert len(p) == emb.shape[0]
    p["sess"] = p["session_id"].astype(int)
    p["ind"] = [str(x) for x in p["individual_id"]]  # 强制纯 Python str
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # 防御：确保 L2
    rng = np.random.default_rng(args.seed)

    def score_img_to_individual(img_emb, gal_idx, gal_ind):
        """单张图 → 每个 gallery 个体：max over 该个体库照片 cos。返回 {ind: score}。"""
        sims = img_emb @ emb[gal_idx].T
        out = {}
        for g in np.unique(gal_ind):
            mask = gal_ind == g
            out[str(g)] = float(sims[mask].max())
        return out

    def split_probe_gallery(sub, ind):
        """个体照片拆成探针簇（一半）与库照片（一半），返回 (q_idx, g_idx)。"""
        idx = np.array(sub.index.to_numpy(), copy=True)  # index 只读，需复制
        rng.shuffle(idx)
        n = len(idx)
        return idx[: n // 2], idx[n // 2:]

    results = {}

    # ---------- 口径 1：库内簇级匹配（两群分别，个体级 R@1/R@5/mAP） ----------
    for sess in (1, 3):
        sub = p[p["sess"] == sess]
        # 拆探针/库：只拆多图个体（>=2 张）；单图个体整图入库
        gal_rows, q_rows = [], {}
        for ind in sorted(sub["ind"].unique()):
            sub_i = sub[sub["ind"] == ind]
            if len(sub_i) >= 2:
                q_idx, g_idx = split_probe_gallery(sub_i, ind)
                q_rows[ind] = q_idx
                gal_rows.append(g_idx)
            else:
                gal_rows.append(sub_i.index.to_numpy())
        gal_idx = np.concatenate(gal_rows)
        gal_ind = np.asarray([sub.loc[i, "ind"] for i in gal_idx])

        r1_single, r1_cluster = [], []
        ap_cluster = []
        votes = []
        for target, q_idx in q_rows.items():
            per_img_scores = [score_img_to_individual(emb[i], gal_idx, gal_ind) for i in q_idx]
            all_g = sorted(per_img_scores[0].keys())
            cluster_score = {g: float(np.mean([s[g] for s in per_img_scores])) for g in all_g}
            top = sorted(cluster_score, key=cluster_score.get, reverse=True)
            # 一致性：簇内多少张图把 target 排第一
            votes.append(sum(1 for s in per_img_scores
                             if max(s, key=s.get) == target) / len(q_idx))
            # 单图对照：同一批探针图逐张独立 Top-1（个体级命中率）
            r1_single.append(sum(1 for s in per_img_scores
                                 if max(s, key=s.get) == target) / len(q_idx))
            # 簇级
            r1_cluster.append(int(top[0] == target))
            ap = 0.0
            for rank, g in enumerate(top[:args.k], start=1):
                if g == target:
                    ap = 1.0 / rank
                    break
            ap_cluster.append(ap)
        results[f"within_sess_{sess}"] = {
            "n_probe_clusters": len(q_rows),
            "n_gallery_images": len(gal_idx),
            "cluster_R@1": float(np.mean(r1_cluster)),
            "single_R@1": float(np.mean(r1_single)),
            "cluster_R@5": float(np.mean([a > 0 and a >= 0.2 for a in ap_cluster])),
            "cluster_mAP": float(np.mean(ap_cluster)),
            "vote1_median": float(np.median(votes)) if votes else None,
        }
        print(f"[sess {sess}] 探针簇 {len(q_rows)} 个 | 簇级 R@1 = {np.mean(r1_cluster):.3f} "
              f"vs 单图 R@1 = {np.mean(r1_single):.3f} | 簇级 mAP = {np.mean(ap_cluster):.3f} "
              f"| Top1 一致票中位 {np.median(votes):.2f}")

    # ---------- 口径 2：跨群簇级拒识（03 簇 → 01 全库；known 侧 = 01 内部） ----------
    gal1 = p[p["sess"] == 1]
    g1_idx = gal1.index.to_numpy()
    g1_ind = np.asarray([gal1.loc[i, "ind"] for i in g1_idx])

    # known 图级对照：01 群每张图对同体其它图 max（E3/E4 同口径，n=25）
    known_img = []
    for ind, sub in gal1.groupby("ind"):
        if len(sub) < 2:
            continue
        si = sub.index.to_numpy()
        sims = emb[si] @ emb[g1_idx].T
        for k, i in enumerate(si):
            same = np.where(g1_ind == ind)[0]
            same = same[g1_idx[same] != i]
            if len(same):
                known_img.append(float(sims[k, same].max()))
    # known 簇级：01 群内部簇匹配分数（探针簇 → 全库，取正确个体分数）
    known_scores, known_multi = [], []
    for target in sorted(gal1["ind"].unique()):
        sub_i = gal1[gal1["ind"] == target]
        if len(sub_i) < 2:
            continue
        q_idx, _ = split_probe_gallery(sub_i, target)
        s = [score_img_to_individual(emb[i], g1_idx, g1_ind) for i in q_idx]
        score = float(np.mean([d[target] for d in s]))
        known_scores.append(score)
        known_multi.append(score)
    # unknown 图级对照：03 群每张图 → 01 全库 max（E3/E4 同口径，n=166）
    unk_img = []
    for _, row in p[p["sess"] == 3].iterrows():
        sims = emb[row.name] @ emb[g1_idx].T
        unk_img.append(float(sims.max()))
    # unknown 簇级：03 群每个个体整簇 → 01 全库（簇内 max）
    unk_scores = []
    for target in sorted(p[p["sess"] == 3]["ind"].unique()):
        q = p[(p["sess"] == 3) & (p["ind"] == target)]
        s = [score_img_to_individual(emb[i], g1_idx, g1_ind) for i in q.index]
        unk_scores.append(float(np.max([max(d.values()) for d in s])))

    def scan_fa5(known, unknown):
        """FA<=5% 时最高 known recall（与 E3/E4 同阈值扫描口径）。"""
        best_recall, best_t = 0.0, None
        for t in np.arange(0.30, 0.96, 0.01):
            fa = float((np.asarray(unknown) >= t).mean())
            if fa <= args.fa_cap:
                rec = float((np.asarray(known) >= t).mean())
                if rec > best_recall:
                    best_recall, best_t = rec, float(t)
        return best_recall, best_t

    rec_img, t_img = scan_fa5(known_img, unk_img)
    rec_clu, t_clu = scan_fa5(known_scores, unk_scores)
    results["cross_group_openset"] = {
        "image_level": {  # E3/E4 同口径对照
            "known_n": len(known_img), "unknown_n": len(unk_img),
            "known_p50": float(np.median(known_img)),
            "unknown_p50": float(np.median(unk_img)),
            "unknown_p95": float(np.percentile(unk_img, 95)),
            f"FA<={args.fa_cap}_known_recall": rec_img, "threshold": t_img,
        },
        "cluster_level": {
            "known_n": len(known_scores), "unknown_n": len(unk_scores),
            "known_p50": float(np.median(known_scores)),
            "unknown_p50": float(np.median(unk_scores)),
            "unknown_p95": float(np.percentile(unk_scores, 95)),
            f"FA<={args.fa_cap}_known_recall": rec_clu, "threshold": t_clu,
        },
    }
    print(f"[openset 图级] known p50={np.median(known_img):.3f} (n={len(known_img)}) | "
          f"unknown p50={np.median(unk_img):.3f} p95={np.percentile(unk_img, 95):.3f} (n={len(unk_img)}) | "
          f"FA<={args.fa_cap} known recall={rec_img:.3f} (t={t_img})")
    print(f"[openset 簇级] known p50={np.median(known_scores):.3f} (n={len(known_scores)}) | "
          f"unknown p50={np.median(unk_scores):.3f} p95={np.percentile(unk_scores, 95):.3f} (n={len(unk_scores)}) | "
          f"FA<={args.fa_cap} known recall={rec_clu:.3f} (t={t_clu})")

    results["_meta"] = {
        "feats": str(args.feats),
        "seed": args.seed,
        "note": "探针/库拆分：多图个体照片按 seed 对半拆成 query 簇与库照片，单图个体整图入库；"
                "簇-个体分数=图-个体max后簇内mean；一致性=簇内Top1票数占比。"
                "理想簇上界（个体照片=纯净簇）；known 侧已从库中排除探针照片。"
                "注意：01/03 多图个体均被 r3 训练见过（训练泄漏），数字为乐观估计。",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"[done] -> {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
