"""
簇级检索评估（2026-08-17，实验 E5）：多帧投票 vs 单图检索。

场景：真实归档流程中，新批次先批内聚类成 Candidate Cluster，再与历史库匹配。
本实验用现有数据模拟并对比两种匹配方式（探针/库拆分设计，同 HappyWhale）：

- 单图匹配：每张图独立检索 Top-K（当前基线，B 协议 R@1≈0.495）
- 簇级匹配（多帧投票）：簇内每张图对每个候选个体打分（图-个体 max），
  簇-个体分数 = 簇内 mean 聚合；簇 Top-1 = 排序第一的个体

评估口径：
1. 库内簇级匹配（个体级 R@1/R@5/mAP）：同批次内，每个至少包含两个完整
   连拍串的个体按串拆成 query 簇与 gallery；同串永不跨两侧。簇检索全库，
   正确个体必须在库中。单图对照 = 同一批 query 图逐张独立 Top-1。
2. 跨批次相似度代理：03 批次个体簇 → 01 批次图库。individual_id 只在
   批次内确认，跨批次没有完成身份对齐，因此该方向不能当作“全部库外”，
   也不能据此计算正式 open-set FA 或推荐阈值。

注意：本实验用"个体照片"模拟理想簇（簇纯净），是簇级检索的**上界**；
真实 HDBSCAN 簇含噪声，会低于此。

特征：历史 r3 度量学习特征（embeddings_metric_r3.npy，行序由邻接 meta 对齐）。

用法：
    python experiments/eval_cluster_retrieval.py
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

SESSION_BY_CODE = {1: "20140806 01", 3: "20140806 03"}


def exclude_query_and_same_series(
    meta: pd.DataFrame,
    query_idx: np.ndarray,
    gallery_idx: np.ndarray,
) -> np.ndarray:
    """从 gallery 剔除 query 本身及 query 所在的完整连拍串。"""
    query_set = {int(index) for index in np.asarray(query_idx, dtype=int)}
    query_series = {
        str(value) for value in meta.loc[list(query_set), "series_id"]
        if str(value).strip()
    }
    keep = []
    for index in np.asarray(gallery_idx, dtype=int):
        series_id = str(meta.loc[int(index), "series_id"])
        keep.append(int(index) not in query_set
                    and not (series_id.strip() and series_id in query_series))
    return np.asarray(gallery_idx, dtype=int)[np.asarray(keep, dtype=bool)]


def split_identity_by_series(
    meta: pd.DataFrame,
    indices: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """按完整串拆一个个体；少于两个串时只进 gallery。"""
    groups: dict[str, list[int]] = {}
    for index in np.asarray(indices, dtype=int):
        series_id = str(meta.loc[int(index), "series_id"])
        unit = series_id if series_id.strip() else f"__single_{int(index)}"
        groups.setdefault(unit, []).append(int(index))
    units = list(groups.values())
    if len(units) < 2:
        return np.asarray([], dtype=int), np.asarray(indices, dtype=int)
    order = np.arange(len(units))
    rng.shuffle(order)
    n_query_units = max(1, len(units) // 2)
    query = np.concatenate([
        np.asarray(units[position], dtype=int)
        for position in order[:n_query_units]
    ])
    gallery = np.concatenate([
        np.asarray(units[position], dtype=int)
        for position in order[n_query_units:]
    ])
    return query, gallery


def distribution_summary(values: list[float]) -> dict:
    """输出描述性分布；跨批次值不解释为已确认负例。"""
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "p50": float(np.median(array)) if len(array) else None,
        "p95": float(np.percentile(array, 95)) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def mean_or_none(values: list[float]) -> float | None:
    """空评估集合返回 None，避免把 NaN 写入 JSON。"""
    return float(np.mean(values)) if values else None


def display_metric(value: float | None) -> str:
    """终端显示可空指标。"""
    return f"{value:.3f}" if value is not None else "N/A"


def main():
    base = REPO_ROOT
    parser = argparse.ArgumentParser(description="簇级检索评估（多帧投票 vs 单图）")
    parser.add_argument("--pilot", type=Path, default=base / "outputs" / "pilot" / "pilot_set.csv")
    parser.add_argument("--feats", type=Path,
                        default=base / "outputs" / "embeddings" / "embeddings_metric_r3.npy")
    parser.add_argument("--out", type=Path, default=base / "outputs" / "reports" / "cluster_retrieval")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7, help="探针/库拆分随机种子")
    args = parser.parse_args()

    p, emb = load_aligned_embeddings(args.feats, args.pilot)
    p["session_id"] = p["session_id"].astype(str)
    p["ind"] = [str(x) for x in p["individual_id"]]  # 强制纯 Python str
    annotate_series(p)
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

    results = {}

    # ---------- 口径 1：库内簇级匹配（两群分别，个体级 R@1/R@5/mAP） ----------
    for sess in (1, 3):
        sub = p[p["session_id"] == SESSION_BY_CODE[sess]]
        if sub.empty:
            raise ValueError(f"清单缺少实验批次 {SESSION_BY_CODE[sess]}")
        # 以完整连拍串为划分单元；单串个体整组只进 gallery。
        gal_rows, q_rows = [], {}
        for ind in sorted(sub["ind"].unique()):
            sub_i = sub[sub["ind"] == ind]
            q_idx, g_idx = split_identity_by_series(
                p, sub_i.index.to_numpy(), rng)
            if len(q_idx):
                q_rows[ind] = q_idx
            gal_rows.append(g_idx)
        gal_idx = np.concatenate(gal_rows)
        gal_ind = np.asarray([sub.loc[i, "ind"] for i in gal_idx])

        r1_single, r1_cluster = [], []
        ap_cluster = []
        votes = []
        effective_gallery_sizes = []
        skipped_no_cross_series_positive = 0
        for target, q_idx in q_rows.items():
            effective_idx = exclude_query_and_same_series(p, q_idx, gal_idx)
            effective_ind = p.loc[effective_idx, "ind"].astype(str).to_numpy()
            if target not in set(effective_ind):
                skipped_no_cross_series_positive += 1
                continue
            effective_gallery_sizes.append(len(effective_idx))
            per_img_scores = [score_img_to_individual(
                emb[i], effective_idx, effective_ind) for i in q_idx]
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
            "n_evaluable_cross_series_clusters": len(r1_cluster),
            "n_skipped_no_cross_series_positive": skipped_no_cross_series_positive,
            "n_gallery_images": len(gal_idx),
            "n_effective_gallery_images_mean": (
                float(np.mean(effective_gallery_sizes))
                if effective_gallery_sizes else None),
            "cluster_R@1": mean_or_none(r1_cluster),
            "single_R@1": mean_or_none(r1_single),
            "cluster_R@5": mean_or_none(
                [a > 0 and a >= 0.2 for a in ap_cluster]),
            "cluster_mAP": mean_or_none(ap_cluster),
            "vote1_median": float(np.median(votes)) if votes else None,
        }
        session_result = results[f"within_sess_{sess}"]
        print(f"[sess {sess}] 可评估跨串簇 {len(r1_cluster)}/{len(q_rows)} 个 | "
              f"簇级 R@1 = {display_metric(session_result['cluster_R@1'])} "
              f"vs 单图 R@1 = {display_metric(session_result['single_R@1'])} | "
              f"簇级 mAP = {display_metric(session_result['cluster_mAP'])} | "
              f"Top1 一致票中位 {display_metric(session_result['vote1_median'])}")

    # ---------- 口径 2：跨批次未对齐相似度代理（不能解释为真实库外负例） ----------
    gal1 = p[p["session_id"] == SESSION_BY_CODE[1]]
    g1_idx = gal1.index.to_numpy()
    g1_ind = np.asarray([gal1.loc[i, "ind"] for i in g1_idx])

    # 批次内已确认正例：排除 query 本身及完整同串，只保留跨串同体匹配。
    known_img = []
    for ind, sub in gal1.groupby("ind"):
        if len(sub) < 2:
            continue
        si = sub.index.to_numpy()
        for i in si:
            effective_idx = exclude_query_and_same_series(
                p, np.asarray([i]), g1_idx)
            effective_ind = p.loc[effective_idx, "ind"].astype(str).to_numpy()
            same = effective_idx[effective_ind == ind]
            if len(same):
                known_img.append(float((emb[i] @ emb[same].T).max()))
    # 批次内簇级正例：query 与 gallery 严格不相交，并剔除 query 的完整连拍串。
    known_scores = []
    for target in sorted(gal1["ind"].unique()):
        sub_i = gal1[gal1["ind"] == target]
        if len(sub_i) < 2:
            continue
        q_idx, _ = split_identity_by_series(
            p, sub_i.index.to_numpy(), rng)
        if not len(q_idx):
            continue
        effective_idx = exclude_query_and_same_series(p, q_idx, g1_idx)
        effective_ind = p.loc[effective_idx, "ind"].astype(str).to_numpy()
        if target not in set(effective_ind):
            continue
        s = [score_img_to_individual(
            emb[i], effective_idx, effective_ind) for i in q_idx]
        score = float(np.mean([d[target] for d in s]))
        known_scores.append(score)
    # 跨批次相似度：身份尚未对齐，可能混有同一真实个体，只作代理分布。
    cross_batch_img = []
    for _, row in p[p["session_id"] == SESSION_BY_CODE[3]].iterrows():
        sims = emb[row.name] @ emb[g1_idx].T
        cross_batch_img.append(float(sims.max()))
    cross_batch_cluster = []
    for target in sorted(p[p["session_id"] == SESSION_BY_CODE[3]]["ind"].unique()):
        q = p[(p["session_id"] == SESSION_BY_CODE[3]) & (p["ind"] == target)]
        s = [score_img_to_individual(emb[i], g1_idx, g1_ind) for i in q.index]
        cross_batch_cluster.append(float(np.max([max(d.values()) for d in s])))

    results["cross_batch_unverified_proxy"] = {
        "calibration_status": "not_open_set_calibration",
        "warning": (
            "individual_id 仅批次内确认，跨批次身份未对齐；cross_batch 分布可能"
            "包含同一真实个体，不是已确认 unknown/负例，禁止据此报告 FA 或推荐阈值。"),
        "image_level": {
            "within_batch_confirmed_positive": distribution_summary(known_img),
            "cross_batch_unverified": distribution_summary(cross_batch_img),
        },
        "cluster_level": {
            "within_batch_confirmed_positive": distribution_summary(known_scores),
            "cross_batch_unverified": distribution_summary(cross_batch_cluster),
        },
    }
    print("[cross-batch proxy] 跨批次 individual_id 未对齐；仅输出相似度分布，"
          "不计算 open-set FA 或推荐阈值。")

    results["_meta"] = {
        "feats": str(args.feats),
        "seed": args.seed,
        "note": "探针/库拆分：按完整连拍串随机分到 query 或 gallery，少于两串的个体只进库；"
                "簇-个体分数=图-个体max后簇内mean；一致性=簇内Top1票数占比。"
                "理想簇上界（个体照片=纯净簇）；批次内 individual_id 为已确认个体，"
                "评估已从库中排除探针照片及完整同串照片。跨批次身份未对齐，"
                "cross_batch_unverified 只作描述性代理，不是已确认负例。"
                "注意：01/03 多图个体均被 r3 训练见过（训练泄漏），数字为乐观估计。",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"[done] -> {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
