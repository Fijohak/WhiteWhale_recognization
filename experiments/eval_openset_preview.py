"""
跨群未知个体预演（TASKS 6.7 开放集评价第一步，2026-08-17）。

场景：历史库 = 01 群（14 个体），"新批次" = 03 群（29 个体，对 01 库全部为真库外）；
反向对称再测一次（gallery=03, query=01）。这是"新批次数据到达"的预演，
不是"跨时间新个体出现"（后者需要多天数据，当前没有）。

口径：
- known positive ：库内同个体照片间最大相似度（多图个体，leave-one-out）
- known negative ：库内异体照片间最大相似度（closed-set 难度参照，也应被拒）
- unknown        ：库外 query 对库内 gallery 的最大相似度（拒识对象，必须被拒）

阈值语义（防错并优先）：选 t 使 open-set false alarm 足够低，同时 known recall 尽量高；
错并把未知个体并入已知个体直接影响种群统计，宁可多标候选不可错并。

特征：MegaDescriptor-T-224 预训练整图（outputs/embeddings_pool_archival/pilot_full/，
与 E2 同模型同批次，无标签泄漏）。全部为弱标签（Source Group ≠ Confirmed Individual）。

用法：
    python scripts/eval_openset_preview.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30..0.95
BAD_BAND = 0.60  # 误配明细阈值：max >= 0.60 的库外 query 记入"高危误配"清单


def run():
    base = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="跨群未知个体预演")
    parser.add_argument("--pilot", type=Path, default=base / "outputs" / "pilot" / "pilot_set.csv")
    parser.add_argument("--feats", type=Path,
                        default=base / "outputs" / "embeddings_pool_archival" / "pilot_full" / "embeddings.npy")
    parser.add_argument("--out", type=Path, default=base / "outputs" / "reports" / "openset_preview")
    args = parser.parse_args()

    p = pd.read_csv(args.pilot)
    emb = np.load(args.feats)
    assert len(p) == emb.shape[0], "特征行数必须与 pilot 行数一致"
    p["ind"] = p["individual_id"].astype(str)
    p["sess"] = p["session_id"].astype(int)

    sess_gallery = {1: 1, 3: 3}  # 历史库 session
    args.out.mkdir(parents=True, exist_ok=True)
    all_results = {}
    curve_rows = []
    rows_unknown = []

    for gsess, qsess in [(1, 3), (3, 1)]:
        gal = p[p["sess"] == gsess]
        que = p[p["sess"] == qsess]
        gal_idx = gal.index.to_numpy()
        gal_ind = gal["ind"].to_numpy()

        # --- known positive（库内同体，多图个体）---
        pos_scores = []
        for ind, sub in gal.groupby("ind"):
            if len(sub) < 2:
                continue
            si = sub.index.to_numpy()
            sims = emb[si] @ emb[gal_idx].T
            for k, i in enumerate(si):
                same = np.where(gal_ind == ind)[0]
                same = same[gal_idx[same] != i]
                if len(same) == 0:
                    continue
                pos_scores.append(float(sims[k, same].max()))
        pos_scores = np.array(pos_scores)

        # --- known negative（库内异体）---
        neg_scores = []
        for _, row in gal.iterrows():
            other = np.where(gal_ind != row["ind"])[0]
            sims = emb[row.name] @ emb[gal_idx[other]].T
            neg_scores.append(float(sims.max()))
        neg_scores = np.array(neg_scores)

        # --- unknown（库外 query 对库内 gallery）---
        unk_scores, unk_top1 = [], []
        for _, row in que.iterrows():
            sims = emb[row.name] @ emb[gal_idx].T
            m = int(np.argmax(sims))
            unk_scores.append(float(sims[m]))
            unk_top1.append((row["image_id"], row["ind"], gal_ind[m], float(sims[m])))
            rows_unknown.append({"gallery_sess": gsess, "query_sess": qsess,
                                 "image_id": row["image_id"], "unknown_individual": row["ind"],
                                 "top1_known": gal_ind[m], "top1_score": float(sims[m])})
        unk_scores = np.array(unk_scores)

        def stats(x):
            return {"n": int(len(x)), "p50": float(np.median(x)), "p90": float(np.percentile(x, 90)),
                    "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
                    "max": float(x.max()), "_raw": [float(v) for v in x]}

        # --- 阈值扫描 ---
        for t in THRESHOLDS:
            curve_rows.append({
                "gallery_sess": gsess, "query_sess": qsess, "threshold": t,
                "known_recall": float((pos_scores >= t).mean()) if len(pos_scores) else None,
                "closed_set_fa": float((neg_scores >= t).mean()) if len(neg_scores) else None,
                "open_set_fa": float((unk_scores >= t).mean()),
                "n_pos": len(pos_scores), "n_neg": len(neg_scores), "n_unk": len(unk_scores),
            })

        # --- 误配明细（高危清单：库外 query 最高相似度 >= BAD_BAND 的"库内个体"）---
        n_bad = int((unk_scores >= BAD_BAND).sum())
        bad_rows = [{"image_id": i, "unknown_individual": ui, "top1_known": gi, "top1_score": s}
                    for i, ui, gi, s in unk_top1 if s >= BAD_BAND]

        all_results[f"dir_{gsess}_q{qsess}"] = {
            "gallery": f"session {gsess} ({len(gal)} imgs / {gal['ind'].nunique()} individuals)",
            "query": f"session {qsess} ({len(que)} imgs / {que['ind'].nunique()} individuals, ALL unknown)",
            "known_positive": stats(pos_scores),
            "known_negative": stats(neg_scores),
            "unknown": stats(unk_scores),
            "high_risk_mismatch_ge_0.60": {"n": n_bad, "rows": bad_rows[:50]},
        }
        print(f"[dir {gsess}->q{qsess}] known+ p50={np.median(pos_scores):.3f} | "
              f"known- p50={np.median(neg_scores):.3f} | unknown p50={np.median(unk_scores):.3f} "
              f"p95={np.percentile(unk_scores, 95):.3f}")

    curve = pd.DataFrame(curve_rows)
    curve.to_csv(args.out / "threshold_curve.csv", index=False)
    pd.DataFrame(rows_unknown).to_csv(args.out / "unknown_detail.csv", index=False)

    # 推荐阈值：open_set_fa <= 5% 的最高 known_recall（两方向取保守侧）
    recs = {}
    for gsess, qsess in [(1, 3), (3, 1)]:
        sub = curve[(curve["gallery_sess"] == gsess) & (curve["query_sess"] == qsess)]
        cand = sub[sub["open_set_fa"] <= 0.05]
        if len(cand):
            j = int(cand["known_recall"].values.argmax())
            recs[f"dir_{gsess}_q{qsess}"] = (float(cand["known_recall"].iloc[j]),
                                             float(cand["threshold"].iloc[j]))
        else:
            recs[f"dir_{gsess}_q{qsess}"] = (None, None)
    all_results["_recommended_threshold"] = {
        f"g{gs}_q{qs}": {"known_recall": recs[f"dir_{gs}_q{qs}"][0],
                         "threshold": recs[f"dir_{gs}_q{qs}"][1]}
        for gs, qs in [(1, 3), (3, 1)]}
    all_results["_meta"] = {
        "model": "MegaDescriptor-T-224 pretrained (full image)",
        "feats": str(args.feats),
        "note": "跨群未知个体预演：库外 query 全部为真未知（不同群）；结果不能直接外推为跨时间新个体发现",
        "bad_band": BAD_BAND,
    }
    (args.out / "metrics.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    _plot(all_results, args.out)
    print(f"[done] -> {args.out}")
    print(f"[rec] 推荐阈值（open-set FA<=5% 时最高 known recall）: {all_results['_recommended_threshold']}")


def _plot(results, out: Path):
    """三分布直方图（known+ / known- / unknown），两方向上下两张；英文标签避免字体问题。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib 未安装，跳过分布图")
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    for ax, key in zip(axes, ["dir_1_q3", "dir_3_q1"]):
        r = results.get(key)
        if not r:
            continue
        gs, qs = key.split("_")[1], key.split("_")[2][1:]
        for label, field, color in [("known same-indiv", "known_positive", "#2f7f2f"),
                                    ("known diff-indiv", "known_negative", "#d0812f"),
                                    ("unknown (out-of-library)", "unknown", "#c03030")]:
            ax.hist(r[field]["_raw"], bins=40, range=(0.0, 1.0), alpha=0.5,
                    color=color, label=label)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"gallery=session {gs} / query=session {qs}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "similarity_distributions.png", dpi=110)
    plt.close(fig)



if __name__ == "__main__":
    run()
