"""
Anchor-based Top-K 检索（方向调整后主流程）。

语义：高分目录数字子文件夹中的照片 = 代表照片（Anchor），`70-79` 散图 = Unresolved
Image Pool。检索以 **query=Anchor**，**gallery=同调查 Pool（+ 同调查其他 Anchor）**，
输出 Top-K 候选供人工审核 same / different / uncertain。

说明：
- 目录不直接等于个体 ID，因此**不计算**"rank-1 命中率"式准确率（无可靠 ground truth）；
- 仅保留一个弱信号诊断：Anchor 文件夹内多帧（同一次挑选的同个体连拍）能否互相回召，
  明确标注为「同组回召（弱信号，不代表身份）」；
- 默认仅同调查内检索（跨调查同名编号未核验）；--cross-session 可开启跨调查检索，
  但输出一律作为候选，必须人工审核。

输出：
- topk_results.csv       每条 query 的 Top-K 候选（image_id、来源、分数）
- topk_for_review.csv    人工审核表（query Anchor → 候选，含来源批次与同组标记）
- topk_stats.json        弱信号诊断统计
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def topk_retrieval(embeddings_path: Path, meta_path: Path, pilot_path: Path,
                   out_dir: Path, k: int = 10, metric: str = "cosine",
                   cross_session: bool = False):
    emb = np.load(embeddings_path)
    meta = pd.read_csv(meta_path)
    pilot = pd.read_csv(pilot_path)

    # 对齐：meta 与 pilot 顺序一致（均由 build_pilot_set 生成）
    assert len(emb) == len(meta), f"embedding {len(emb)} 与 meta {len(meta)} 数量不一致"
    df = meta.merge(pilot, on="image_id", how="left", suffixes=("", "_pilot"))
    assert len(df) == len(meta), "merge 后行数变化，请检查 image_id 是否唯一"
    df["session_id"] = df["session_id"].astype(str)

    # 分组：Anchor = 高分目录照片；Pool = 散图（loose_known）
    df["is_anchor"] = df["split"] == "labeled"
    n_anchor = int(df["is_anchor"].sum())
    n_pool = int((~df["is_anchor"]).sum())

    if metric == "cosine":
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        sim = emb @ emb.T
    else:
        raise ValueError(f"不支持的 metric: {metric}")

    results = []
    for idx, row in df.iterrows():
        if not row["is_anchor"]:
            continue  # 只以 Anchor 为 query
        # gallery：同调查 Pool（默认）；--cross-session 时含全部 Pool 与其他调查 Anchor
        if cross_session:
            mask = ~df["is_anchor"] | (df.index != idx)
        else:
            mask = (~df["is_anchor"]) & (df["session_id"] == row["session_id"])
        scores = sim[idx, mask.values]
        cand = df[mask.values].copy()
        cand["score"] = scores
        # 同组标记：候选是否与 query 来自同一数字子文件夹（弱证据，非身份）
        same_group = cand["individual_id"] == row["individual_id"]
        cand["same_group"] = same_group
        top = cand.sort_values("score", ascending=False).head(k)
        results.append({
            "query_anchor": row["image_id"],
            "query_session": row["session_id"],
            "query_group": row["individual_id"],
            "query_filename": row.get("filename", None),
            "rank": range(1, len(top) + 1),
            "candidate": top["image_id"].tolist(),
            "candidate_session": top["session_id"].tolist(),
            "candidate_group": top["individual_id"].tolist(),
            "candidate_is_pool": (~top["is_anchor"]).tolist(),
            "candidate_same_group": top["same_group"].tolist(),
            "candidate_score": [float(s) for s in top["score"].tolist()],
            "candidate_filename": top.get("filename", None).tolist()
            if "filename" in top else None,
        })
    res = pd.DataFrame(results)

    # ---- 弱信号诊断：Anchor 文件夹多帧是否互相回召（不代表身份，仅同一次挑选） ----
    # 对含 ≥2 张的 Anchor 组，组内其他照片是否出现在该 Anchor 的 Top-K 中
    diag = {}
    for gid, sub in df[df["is_anchor"]].groupby("individual_id"):
        if len(sub) < 2:
            continue
        for _, q in sub.iterrows():
            siblings = [s for s in sub["image_id"] if s != q["image_id"]]
            if not siblings:
                continue
            qrow = res[res["query_anchor"] == q["image_id"]]
            if qrow.empty:
                continue
            hits = [c for c, g in zip(qrow.iloc[0]["candidate"],
                                      qrow.iloc[0]["candidate_same_group"]) if g]
            diag[gid] = {"n_frames": int(len(sub)),
                         "sibling_in_topk": int(any(h in qrow.iloc[0]["candidate"] for h in siblings))}
    n_groups = len(diag)
    n_hit = sum(1 for v in diag.values() if v["sibling_in_topk"])
    stats = {
        "n_anchor_queries": int(len(res)),
        "n_anchor": n_anchor,
        "n_pool": n_pool,
        "k": k,
        "cross_session": cross_session,
        "weak_diagnostic": {
            "note": "同组回召 = Anchor 文件夹内多帧互相出现在 Top-K。仅反映同一次挑选/连拍的相似性，不代表跨时间身份能力。",
            "anchor_groups_with_2plus": n_groups,
            "groups_sibling_in_topk": n_hit,
            "ratio": (n_hit / n_groups) if n_groups else None,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    # 结果表（JSON 列）
    res_out = res.copy()
    for c in ("candidate", "candidate_session", "candidate_group", "candidate_is_pool",
              "candidate_same_group", "candidate_score"):
        res_out[c] = res_out[c].apply(json.dumps, ensure_ascii=False)
    res_out.to_csv(out_dir / "topk_results.csv", index=False, encoding="utf-8-sig")

    # 人工审核表：展开成每 query-候选一行
    review_rows = []
    for _, r in res.iterrows():
        for i, cid in enumerate(r["candidate"]):
            review_rows.append({
                "query_anchor": r["query_anchor"],
                "query_session": r["query_session"],
                "query_group": r["query_group"],
                "rank": i + 1,
                "candidate": cid,
                "candidate_session": r["candidate_session"][i],
                "candidate_group": r["candidate_group"][i],
                "candidate_is_pool": r["candidate_is_pool"][i],
                "candidate_same_group": r["candidate_same_group"][i],
                "score": r["candidate_score"][i],
                "review": "",  # 人工填写 same / different / uncertain
                "reviewer": "",
                "note": "",
            })
    review = pd.DataFrame(review_rows)
    review.to_csv(out_dir / "topk_for_review.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "topk_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Anchor Top-K 检索完成: {len(res)} 个 Anchor 查询（Pool {n_pool} 张）")
    print(f"  弱信号诊断: 含多帧 Anchor 组 {n_groups}，组内回召 {n_hit}（{n_hit / n_groups:.1%}" if n_groups else "  弱信号诊断: 无多帧 Anchor 组")
    print(f"  → {out_dir / 'topk_for_review.csv'}（供人工审核）")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anchor-based Top-K 检索（候选输出，供人工审核）")
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser.add_argument("--embeddings", type=Path, default=base / "embeddings" / "embeddings.npy")
    parser.add_argument("--meta", type=Path, default=base / "embeddings" / "embeddings_meta.csv")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--out", type=Path, default=base / "retrieval")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--cross-session", action="store_true",
                        help="跨调查检索（输出仅作候选，须人工核验）")
    args = parser.parse_args()
    topk_retrieval(args.embeddings, args.meta, args.pilot, args.out, args.k,
                   cross_session=args.cross_session)
