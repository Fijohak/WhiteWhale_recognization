"""
检索评估（正式入口 6/7）：对当前特征库做个体级检索评估。

统一度量入口（E1-E5 的专项实验脚本在 experiments/，本入口用于正式特征的
日常评估，不复制实验配置）：

    python scripts/evaluate.py                       # 跨串检索评估（R@1 / mAP，个体级）
    python scripts/evaluate.py --mode pairs          # 跨串同体/跨体图对分布（仅诊断）

输入特征带 confirmed_identity（已确认个体标签）时，以完整连拍串为最小单元：
至少两个串的个体抽一个整串作 query，其余串入库；不足两个串的个体只入库。
图对个体分数 = max cos（多帧聚合语义）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.data.eval_set import attach_stable_series_from_manifest  # noqa: E402
from whitewhale.data.sequence_groups import series_units  # noqa: E402
from whitewhale.reid.embedding import (  # noqa: E402
    load_verified_embedding_artifact,
    require_generated_artifact_provenance,
)
from whitewhale.reid.evaluation import split_query_gallery  # noqa: E402
from whitewhale.reid.retrieval import cosine_topk  # noqa: E402


def _load(embeddings: Path, meta: Path, manifest: Path | None = None,
          allow_legacy_diagnostic: bool = False):
    """加载特征，并优先从完整 manifest 回填生成期稳定 series。"""
    emb, m, config = load_verified_embedding_artifact(
        embeddings, meta, require_hashes=True)
    if not allow_legacy_diagnostic:
        require_generated_artifact_provenance(config)
    for column in ("session_id", "relative_path"):
        if column not in m.columns:
            raise ValueError(f"meta 缺少 {column}，无法执行同串约束")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if not np.isfinite(emb).all() or np.any(norms == 0):
        raise ValueError("特征含 NaN/Inf/零向量，拒绝生成评估结果")
    emb = emb / norms
    if manifest is not None:
        if not manifest.exists():
            raise FileNotFoundError(f"完整 manifest 不存在：{manifest}")
        full_manifest = pd.read_csv(
            manifest, dtype=str, keep_default_na=False)
        m = attach_stable_series_from_manifest(m, full_manifest)
    elif "series_id" not in m.columns:
        raise ValueError(
            "meta 缺少生成期稳定 series_id，必须传入完整 manifest；"
            "禁止在 embedding 子集上重新分串")
    m["series_unit"] = series_units(m)
    return emb, m


def _individual_ap(ind_scores: np.ndarray, true_col: np.ndarray) -> float:
    """个体级 mAP：对每个 query 的个体分数排序，取真值个体位置计算 AP。"""
    aps = []
    for row, t in zip(ind_scores, true_col):
        order = np.argsort(-row)
        rank_pos = int(np.where(order == t)[0][0]) + 1
        aps.append(1.0 / rank_pos)
    return float(np.mean(aps))


def eval_retrieval(emb, meta, seed: int):
    """批次内、跨串个体检索：不把跨批次未对齐身份当作负例。"""
    keep = meta["confirmed_identity"].notna() & (
        meta["confirmed_identity"].astype(str).str.strip() != "")
    emb, meta = emb[keep.to_numpy()], meta[keep].copy()
    meta["ind"] = meta["confirmed_identity"].astype(str)
    meta["eval_ind"] = pd.Series(
        list(zip(meta["session_id"].astype(str), meta["ind"])),
        index=meta.index,
        dtype=object,
    )
    if meta["eval_ind"].nunique() < 2:
        raise SystemExit(f"已确认个体数不足（{meta['ind'].nunique()}），无法评估。"
                         f"请先完成人工初审（confirmed_identity 非空）。")
    q, g = split_query_gallery(
        meta, identity_col="eval_ind", seed=seed, series_col="series_unit")
    if q.empty:
        raise SystemExit("没有具备至少两个完整 series 的已确认个体，无法做跨串评估。")
    overlap = set(q["series_unit"]) & set(g["series_unit"])
    position = {image_id: index for index, image_id in enumerate(meta["image_id"])}
    hits: list[bool] = []
    aps: list[float] = []
    candidate_counts: list[int] = []
    skipped_no_positive = 0
    skipped_no_negative = 0
    excluded_same_series = 0
    for _, query in q.iterrows():
        g_session = g[
            g["session_id"].astype(str) == str(query["session_id"])
        ]
        # 防御性约束：即使未来的上游划分绕过全局串断言，
        # 单个 query 也绝不参与同 session、同 series 的 gallery 比较。
        same_series = (
            g_session["series_unit"].astype(str) == str(query["series_unit"])
        )
        excluded_same_series += int(same_series.sum())
        candidates = g_session.loc[~same_series]
        has_positive = bool(candidates["eval_ind"].map(
            lambda value: value == query["eval_ind"]).any())
        if not has_positive:
            skipped_no_positive += 1
            continue
        has_negative = bool(candidates["eval_ind"].map(
            lambda value: value != query["eval_ind"]).any())
        if not has_negative:
            skipped_no_negative += 1
            continue

        q_idx = position[query["image_id"]]
        g_idx = [position[iid] for iid in candidates["image_id"]]
        scores, idx = cosine_topk(emb[[q_idx]], emb[g_idx], k=len(g_idx))
        ind_names = sorted(candidates["eval_ind"].unique())
        name2col = {name: i for i, name in enumerate(ind_names)}
        gallery_inds = candidates["eval_ind"].to_numpy()
        ind_scores = np.full(len(ind_names), -1.0)
        for score, gallery_index in zip(scores[0], idx[0]):
            column = name2col[gallery_inds[gallery_index]]
            ind_scores[column] = max(ind_scores[column], score)
        true_col = name2col[query["eval_ind"]]
        hits.append(bool(ind_scores.argmax() == true_col))
        rank = int(np.where(np.argsort(-ind_scores) == true_col)[0][0]) + 1
        aps.append(1.0 / rank)
        candidate_counts.append(len(candidates))

    if not hits:
        raise SystemExit(
            "没有同时具备跨串正样本和跨体负样本的 query，无法评估。")
    r1 = float(np.mean(hits))
    ap = float(np.mean(aps))
    n_individuals = int(meta["eval_ind"].nunique())
    skipped = skipped_no_positive + skipped_no_negative
    print(f"[eval] 批次内跨串个体级检索（有效 query {len(hits)}/{len(q)} 张 / "
          f"gallery {len(g)} 张 / "
          f"{n_individuals} 个批次内身份）")
    if skipped:
        print(f"[eval] 跳过 query {skipped} 张：无跨串正样本 "
              f"{skipped_no_positive}，无跨体负样本 {skipped_no_negative}")
    if overlap:
        print(f"[eval] 防御性过滤：上游划分含 {len(overlap)} 个重叠 series，"
              f"逐 query 共剔除同串 gallery {excluded_same_series} 张次")
    print(f"[eval] R@1 = {r1:.3f}   mAP = {ap:.3f}")
    return {"n_query": int(len(hits)), "n_query_split": int(len(q)),
            "n_query_skipped": int(skipped),
            "n_query_skipped_no_positive": int(skipped_no_positive),
            "n_query_skipped_no_negative": int(skipped_no_negative),
            "n_split_overlap_series": int(len(overlap)),
            "n_excluded_same_series_gallery": int(excluded_same_series),
            "n_gallery": int(len(g)),
            "n_individuals": n_individuals,
            "gallery_per_query_min": int(min(candidate_counts)),
            "gallery_per_query_max": int(max(candidate_counts)),
            "r1": round(r1, 4), "map": round(ap, 4), "seed": seed,
            "protocol": "within_session_cross_series"}


def eval_pairs(emb, meta):
    """跨串同体/跨体图对余弦分布；图对分位数不是部署 FA 标定。"""
    keep = meta["confirmed_identity"].notna() & (
        meta["confirmed_identity"].astype(str).str.strip() != "")
    emb, meta = emb[keep.to_numpy()], meta[keep].copy()
    ind = meta["confirmed_identity"].astype(str).to_numpy()
    series = meta["series_id"].fillna("").astype(str).to_numpy()
    sessions = meta["session_id"].fillna("").astype(str).to_numpy()
    n = len(emb)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)
    same, cross = [], []
    excluded_same_series = 0
    excluded_cross_session = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sessions[i] != sessions[j]:
                excluded_cross_session += 1
                continue
            if series[i] and series[i] == series[j]:
                excluded_same_series += 1
                continue
            (same if ind[i] == ind[j] else cross).append(float(sim[i, j]))
    same, cross = np.array(same), np.array(cross)
    same_median = float(np.median(same)) if len(same) else np.nan
    same_p25 = float(np.quantile(same, 0.25)) if len(same) else np.nan
    cross_median = float(np.median(cross)) if len(cross) else np.nan
    cross_p95 = float(np.quantile(cross, 0.95)) if len(cross) else np.nan
    cross_p99 = float(np.quantile(cross, 0.99)) if len(cross) else np.nan
    print(f"[eval] 跨串同体图对 {len(same)} / 跨体图对 {len(cross)}"
          f"（剔除同串 {excluded_same_series}、跨批次未对齐 {excluded_cross_session}）")
    print(f"[eval] 同体相似度: 中位 {same_median:.3f}（p25 {same_p25:.3f}）")
    print(f"[eval] 跨体相似度: 中位 {cross_median:.3f}（p95 {cross_p95:.3f}）")
    print("[eval] 注意：跨体图对 p95/p99 仅供分布诊断；正式拒识阈值必须用独立"
          " known/unknown 集按每个 query 对全库最大分数重新标定。")
    return {"n_same": int(len(same)), "n_cross": int(len(cross)),
            "n_excluded_same_series": excluded_same_series,
            "n_excluded_cross_session_unverified": excluded_cross_session,
            "same_median": round(same_median, 3) if np.isfinite(same_median) else None,
            "cross_median": round(cross_median, 3) if np.isfinite(cross_median) else None,
            "cross_pair_p95": round(cross_p95, 3) if np.isfinite(cross_p95) else None,
            "cross_pair_p99": round(cross_p99, 3) if np.isfinite(cross_p99) else None,
            "calibration_status": "diagnostic_only_not_open_set_calibration"}


def main():
    base = REPO_ROOT / "outputs"
    parser = argparse.ArgumentParser(description="特征库检索评估（正式度量入口）")
    parser.add_argument("--embeddings", type=Path,
                        default=base / "embeddings" / "embeddings_metric_r4_yolocrop_v2.npy",
                        help="特征（meta 需含 confirmed_identity）")
    parser.add_argument("--meta", type=Path,
                        default=base / "embeddings" / "embeddings_metric_r4_yolocrop_v2_meta.csv")
    parser.add_argument("--manifest", type=Path,
                        default=base / "index" / "dataset_manifest.csv",
                        help="完整数据清单（先全局分串，再按 image_id 回填）")
    parser.add_argument("--mode", choices=["retrieval", "pairs"], default="retrieval",
                        help="retrieval = 跨串个体级 R@1/mAP；pairs = 图对分布诊断（非阈值标定）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-legacy-diagnostic", action="store_true",
        help=("显式允许仅回填 provenance 的历史产物；"
              "仅供诊断，不得当作正式评估结论"))
    parser.add_argument("--out", type=Path, default=base / "reports" / "evaluate")
    args = parser.parse_args()

    emb, meta = _load(
        args.embeddings, args.meta, args.manifest,
        allow_legacy_diagnostic=args.allow_legacy_diagnostic)
    if args.mode == "retrieval":
        res = eval_retrieval(emb, meta, args.seed)
    else:
        res = eval_pairs(emb, meta)
    res["artifact_provenance_mode"] = (
        "legacy_diagnostic" if args.allow_legacy_diagnostic
        else "generated_row_binding_required")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.mode}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] → {args.out / f'{args.mode}.json'}")


if __name__ == "__main__":
    main()
