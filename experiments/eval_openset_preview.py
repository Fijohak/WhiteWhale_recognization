"""
跨批次未对齐相似度预演（2026-08-17 历史实验修订版）。

场景：分别用 01/03 批次作 gallery，反向批次作 query。individual_id 是
批次内已确认个体标签，但跨批次身份尚未对齐；因此 cross-batch query 可能
包含 gallery 中同一只真实个体，不能标为 confirmed unknown 或可靠负例。

口径：
- within-batch confirmed positive：库内同个体的跨连拍串最大相似度
- within-batch different identity：库内已确认异体的最大相似度
- cross_batch_unverified：另一批次 query 对 gallery 的最大相似度，仅作代理分布

脚本只输出描述性分布和各阈值下的代理接受率，不输出 open-set FA 或推荐阈值。
正式拒识阈值必须使用跨批次身份完成对齐后的独立已知/未知集合重新标定。

特征：MegaDescriptor-T-224 预训练整图（outputs/embeddings_pool_archival/pilot_full/，
与 E2 同模型同批次，无标签泄漏）。individual_id 批次内已确认、跨批次未对齐。

用法：
    python experiments/eval_openset_preview.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.artifact_utils import load_aligned_embeddings  # noqa: E402
from whitewhale.data.sequence_groups import annotate_series  # noqa: E402

THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30..0.95
PROXY_SIMILARITY_BAND = 0.60  # 仅用于列出高相似 cross-batch 代理样本
SESSION_BY_CODE = {1: "20140806 01", 3: "20140806 03"}


def distribution_stats(values: np.ndarray) -> dict:
    """输出可为空的描述性分布，避免 NaN 污染 JSON。"""
    array = np.asarray(values, dtype=float)
    result = {"n": int(len(array)), "_raw": [float(value) for value in array]}
    for percentile, name in ((50, "p50"), (90, "p90"),
                             (95, "p95"), (99, "p99")):
        result[name] = (float(np.percentile(array, percentile))
                        if len(array) else None)
    result["max"] = float(array.max()) if len(array) else None
    return result


def display_metric(value: float | None) -> str:
    """终端显示可空指标。"""
    return f"{value:.3f}" if value is not None else "N/A"


def exclude_query_and_same_series(
    meta: pd.DataFrame,
    query_idx: int,
    gallery_idx: np.ndarray,
) -> np.ndarray:
    """剔除 query 自身及其完整连拍串，无法解析串时只排除自身。"""
    query_series = str(meta.loc[query_idx, "series_id"])
    keep = []
    for gallery_index in np.asarray(gallery_idx, dtype=int):
        gallery_series = str(meta.loc[int(gallery_index), "series_id"])
        keep.append(int(gallery_index) != int(query_idx)
                    and not (query_series.strip()
                             and gallery_series == query_series))
    return np.asarray(gallery_idx, dtype=int)[np.asarray(keep, dtype=bool)]


def archive_legacy_unknown_detail(out_dir: Path) -> Path | None:
    """可恢复地归档旧版 unknown 明细，避免与未对齐代理结果并存。"""
    legacy = out_dir / "unknown_detail.csv"
    if not legacy.exists():
        return None
    archive_dir = out_dir / "legacy_pre_alignment"
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / legacy.name
    counter = 1
    while target.exists():
        target = archive_dir / f"unknown_detail_{counter}.csv"
        counter += 1
    legacy.rename(target)
    return target


def run():
    base = REPO_ROOT
    parser = argparse.ArgumentParser(description="跨批次未对齐相似度预演")
    parser.add_argument("--pilot", type=Path, default=base / "outputs" / "pilot" / "pilot_set.csv")
    parser.add_argument("--feats", type=Path,
                        default=base / "outputs" / "embeddings_pool_archival" / "pilot_full" / "embeddings.npy")
    parser.add_argument("--out", type=Path, default=base / "outputs" / "reports" / "openset_preview")
    args = parser.parse_args()

    p, emb = load_aligned_embeddings(args.feats, args.pilot)
    p["ind"] = p["individual_id"].astype(str)
    p["session_id"] = p["session_id"].astype(str)
    annotate_series(p)

    args.out.mkdir(parents=True, exist_ok=True)
    archived_legacy = archive_legacy_unknown_detail(args.out)
    if archived_legacy is not None:
        print(f"[legacy] 旧 unknown 明细已归档：{archived_legacy}")
    all_results = {}
    curve_rows = []
    rows_cross_batch = []

    for gsess, qsess in [(1, 3), (3, 1)]:
        gallery_session = SESSION_BY_CODE[gsess]
        query_session = SESSION_BY_CODE[qsess]
        gal = p[p["session_id"] == gallery_session]
        que = p[p["session_id"] == query_session]
        if gal.empty or que.empty:
            raise ValueError(
                f"清单缺少实验批次：gallery={gallery_session}, query={query_session}")
        gal_idx = gal.index.to_numpy()
        gal_ind = gal["ind"].to_numpy()

        # --- 批次内已确认正例：排除自身和完整同串，只保留跨串同体 ---
        pos_scores = []
        for ind, sub in gal.groupby("ind"):
            if len(sub) < 2:
                continue
            si = sub.index.to_numpy()
            for i in si:
                effective_idx = exclude_query_and_same_series(p, i, gal_idx)
                effective_ind = p.loc[effective_idx, "ind"].astype(str).to_numpy()
                same = effective_idx[effective_ind == ind]
                if not len(same):
                    continue
                pos_scores.append(float((emb[i] @ emb[same].T).max()))
        pos_scores = np.array(pos_scores)

        # --- 批次内已确认异体（描述 closed-set 难度，不称 FA）---
        neg_scores = []
        for _, row in gal.iterrows():
            effective_idx = exclude_query_and_same_series(
                p, int(row.name), gal_idx)
            effective_ind = p.loc[effective_idx, "ind"].astype(str).to_numpy()
            other = effective_idx[effective_ind != row["ind"]]
            if not len(other):
                continue
            sims = emb[row.name] @ emb[other].T
            neg_scores.append(float(sims.max()))
        neg_scores = np.array(neg_scores)

        # --- 跨批次未对齐代理：可能含同一真实个体，不能称 unknown ---
        cross_scores, cross_top1 = [], []
        for _, row in que.iterrows():
            sims = emb[row.name] @ emb[gal_idx].T
            m = int(np.argmax(sims))
            cross_scores.append(float(sims[m]))
            cross_top1.append((row["image_id"], row["ind"], gal_ind[m], float(sims[m])))
            rows_cross_batch.append({
                "gallery_sess": gsess,
                "query_sess": qsess,
                "gallery_session_id": gallery_session,
                "query_session_id": query_session,
                "image_id": row["image_id"],
                "query_batch_identity": row["ind"],
                "top1_gallery_identity": gal_ind[m],
                "top1_score": float(sims[m]),
                "identity_alignment": "cross_batch_unverified",
            })
        cross_scores = np.array(cross_scores)

        # --- 描述性阈值扫描：cross-batch 列只表示代理接受率，不是 FA ---
        for t in THRESHOLDS:
            curve_rows.append({
                "gallery_sess": gsess, "query_sess": qsess, "threshold": t,
                "gallery_session_id": gallery_session,
                "query_session_id": query_session,
                "within_batch_positive_recall": (
                    float((pos_scores >= t).mean()) if len(pos_scores) else None),
                "within_batch_different_identity_accept_rate": (
                    float((neg_scores >= t).mean()) if len(neg_scores) else None),
                "cross_batch_unverified_accept_rate": float((cross_scores >= t).mean()),
                "n_within_batch_positive": len(pos_scores),
                "n_within_batch_different_identity": len(neg_scores),
                "n_cross_batch_unverified": len(cross_scores),
            })

        # 高相似代理清单仅供人工跨批次身份对齐，不判定为误配。
        high_similarity = [
            {"image_id": image_id,
             "query_batch_identity": query_identity,
             "top1_gallery_identity": gallery_identity,
             "top1_score": score}
            for image_id, query_identity, gallery_identity, score in cross_top1
            if score >= PROXY_SIMILARITY_BAND
        ]

        all_results[f"dir_{gsess}_q{qsess}"] = {
            "gallery": (f"session {gallery_session} ({len(gal)} imgs / "
                        f"{gal['ind'].nunique()} individuals)"),
            "query": (f"session {query_session} ({len(que)} imgs / "
                      f"{que['ind'].nunique()} batch-local confirmed identities; "
                      "cross-batch alignment unverified)"),
            "within_batch_confirmed_positive": distribution_stats(pos_scores),
            "within_batch_different_identity": distribution_stats(neg_scores),
            "cross_batch_unverified": distribution_stats(cross_scores),
            "high_similarity_proxy_ge_0.60": {
                "n": len(high_similarity), "rows": high_similarity[:50]},
        }
        direction = all_results[f"dir_{gsess}_q{qsess}"]
        print(f"[dir {gsess}->q{qsess}] within-batch positive p50="
              f"{display_metric(direction['within_batch_confirmed_positive']['p50'])} | "
              f"different-identity p50="
              f"{display_metric(direction['within_batch_different_identity']['p50'])} | "
              f"cross-batch-unverified p50="
              f"{display_metric(direction['cross_batch_unverified']['p50'])} "
              f"p95={display_metric(direction['cross_batch_unverified']['p95'])}")

    curve = pd.DataFrame(curve_rows)
    curve.to_csv(args.out / "threshold_curve.csv", index=False)
    pd.DataFrame(rows_cross_batch).to_csv(
        args.out / "cross_batch_unverified_detail.csv", index=False)

    all_results["_meta"] = {
        "model": "MegaDescriptor-T-224 pretrained (full image)",
        "feats": str(args.feats),
        "calibration_status": "cross_batch_unverified_proxy_only",
        "note": (
            "individual_id 仅批次内确认、跨批次未对齐。cross_batch 分布可能包含"
            "同一真实个体，不是 confirmed unknown/可靠负例；本实验不报告正式 FA，"
            "也不推荐阈值。批次内正例已剔除自身及完整同串照片。"),
        "proxy_similarity_band": PROXY_SIMILARITY_BAND,
        "legacy_artifact_archived_to": (
            str(archived_legacy) if archived_legacy is not None else None),
    }
    (args.out / "metrics.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    _plot(all_results, args.out)
    print(f"[done] -> {args.out}")
    print("[proxy] 未输出推荐阈值；需先完成跨批次身份对齐并构建独立开放集。")


def _plot(results, out: Path):
    """绘制批次内正/异体与跨批次未对齐代理分布。"""
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
        for label, field, color in [
                ("within-batch confirmed same identity",
                 "within_batch_confirmed_positive", "#2f7f2f"),
                ("within-batch different identity",
                 "within_batch_different_identity", "#d0812f"),
                ("cross-batch unverified proxy",
                 "cross_batch_unverified", "#c03030")]:
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
