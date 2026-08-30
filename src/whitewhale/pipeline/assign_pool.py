"""
同群散图划分（Pool Assignment）。

场景（用户确认，2026-08-14）：优先把同群内的未归属散图（70-79 直接文件，
01: 59 张 / 03: 148 张）划分到群内已确认个体，而不是跨群全库检索。
跨群匹配在这个场景不参与，因此 query 与 gallery 都限制在同一群（session）。

流程（E1/E2/E4 结论）：散图与同群已确认个体使用 pipeline.yaml 指向的
同版 r4+YOLO 生成期产物，执行余弦检索并输出 Top-K 个体候选与分数。

语义：输出是 Candidate（候选划归），不代表自动确认；低分散图可能是
新个体候选，需人工审核。所有行保留 image_id / 原路径 / 群 / 候选分数。

阈值：默认 0.50 沿用历史 E4 实验参考值；正式使用前需在当前模型和独立集上重标定。

CLI 入口见 scripts/assign_pool.py。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from whitewhale.config import load_config
from whitewhale.data.history_verify import load_identity_migration
from whitewhale.data.manifest import compute_sha256
from whitewhale.data.sequence_groups import annotate_series, series_units
from whitewhale.reid.embedding import (
    load_verified_embedding_artifact,
    require_compatible_embedding_configs,
    require_generated_artifact_provenance,
)


def _sess_key(s) -> str:
    """把 session 当不透明字符串，仅清理首尾空白，不做数值猜测。"""
    if pd.isna(s) or str(s).strip() == "":
        return ""
    return str(s).strip()


def ensure_series_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """补齐完整 ``series_id`` 与稳定 ``series_unit``，并规范 session。"""
    out = frame.copy()
    if "image_id" not in out.columns:
        raise ValueError("候选清单缺少 image_id")
    if "session_id" not in out.columns:
        if "group" not in out.columns:
            raise ValueError("候选清单缺少 session_id/group")
        out["session_id"] = out["group"]
    out["session_id"] = out["session_id"].map(_sess_key)
    if (out["session_id"] == "").any():
        raise ValueError("候选清单存在空 session_id")

    derived = (out["relative_path"].fillna("").map(
        lambda value: Path(str(value)).name)
        if "relative_path" in out.columns
        else pd.Series("", index=out.index, dtype=object))
    if "filename" not in out.columns:
        out["filename"] = derived
    else:
        filename = out["filename"].fillna("").astype(str)
        out["filename"] = filename.where(filename.str.strip() != "", derived)

    annotate_series(out)
    out["series_unit"] = series_units(out)
    out["group"] = out["session_id"].map(_sess_key)
    return out


def load_series_manifest(manifest_path: Path) -> pd.DataFrame:
    """从全量 manifest 构建唯一串索引，并记录源文件哈希。"""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"全量 manifest 不存在: {manifest_path}")
    manifest = pd.read_csv(
        manifest_path, dtype=str, keep_default_na=False)
    required = {"image_id", "session_id"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"全量 manifest 缺少列: {missing}")
    image_ids = manifest["image_id"].astype(str).str.strip()
    if (image_ids == "").any() or image_ids.duplicated().any():
        raise ValueError("全量 manifest 的 image_id 为空或重复")
    manifest["image_id"] = image_ids
    manifest = ensure_series_metadata(manifest)
    index = manifest[[
        "image_id", "session_id", "filename", "series_id", "series_unit",
    ]].copy()
    index.attrs["provenance"] = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": compute_sha256(manifest_path),
        "manifest_n": int(len(manifest)),
        "series_source": "full_dataset_manifest",
    }
    return index


def attach_series_manifest(
    frame: pd.DataFrame,
    series_index: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """按 image_id 把全量串索引附到特征 meta；缺失或 session 冲突即拒绝。"""
    out = frame.copy()
    if "image_id" not in out.columns:
        raise ValueError(f"{source_name} meta 缺少 image_id")
    image_ids = out["image_id"].astype(str).str.strip()
    if (image_ids == "").any() or image_ids.duplicated().any():
        raise ValueError(f"{source_name} meta 的 image_id 为空或重复")
    if "image_id" not in series_index.columns:
        raise ValueError("全量 manifest 串索引缺少 image_id")
    index_ids = series_index["image_id"].astype(str).str.strip()
    if (index_ids == "").any() or index_ids.duplicated().any():
        raise ValueError("全量 manifest 串索引的 image_id 为空或重复")
    lookup = series_index.assign(image_id=index_ids).set_index("image_id")
    missing_ids = sorted(set(image_ids) - set(lookup.index))
    if missing_ids:
        raise ValueError(
            f"全量 manifest 未覆盖 {source_name} 的 {len(missing_ids)} 张图: "
            f"{missing_ids[:5]}")
    aligned = lookup.loc[image_ids].reset_index()

    if "session_id" in out.columns:
        actual = out["session_id"].map(_sess_key).reset_index(drop=True)
        expected = aligned["session_id"].map(_sess_key)
        mismatch = actual != expected
        if mismatch.any():
            bad = image_ids.reset_index(drop=True)[mismatch].tolist()
            raise ValueError(
                f"{source_name} meta 与全量 manifest 的 session_id 冲突: {bad[:5]}")
    out["session_id"] = aligned["session_id"].to_numpy()
    out["filename"] = aligned["filename"].to_numpy()
    out["series_id"] = aligned["series_id"].to_numpy()
    out["series_unit"] = aligned["series_unit"].to_numpy()
    out["group"] = out["session_id"].map(_sess_key)
    return out.reset_index(drop=True)


def _require_series_metadata(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """下游只接受已由全量索引补齐的串字段，不再从特征子集重算。"""
    required = {"image_id", "group", "series_id", "series_unit"}
    missing = sorted(required - set(frame.columns))
    if missing:
        if frame.empty:
            out = frame.copy()
            for column in missing:
                out[column] = pd.Series(dtype=str)
            return out
        raise ValueError(
            f"{source_name} 缺少串字段 {missing}；请先按全量 manifest 附加")
    return frame.copy()


def _candidate_gallery(q: pd.Series, gallery: pd.DataFrame) -> pd.DataFrame:
    """只保留同 session、非自身且不同完整连拍串的候选图片。"""
    scoped = gallery[gallery["group"] == q["group"]]
    if "image_id" in q.index:
        scoped = scoped[scoped["image_id"].astype(str) != str(q["image_id"])]
    query_series = str(q.get("series_id", "")).strip()
    if query_series:
        scoped = scoped[
            scoped["series_id"].fillna("").astype(str) != query_series]
    return scoped


def _rank_identities(q_emb: np.ndarray, gallery: pd.DataFrame,
                     embedding_col: str, topk: int) -> list[tuple[str, int, float]]:
    """照片相似度按已确认个体取最大值，返回个体级 Top-K。"""
    emb_gal = np.stack(gallery[embedding_col].to_numpy())
    sim = emb_gal @ q_emb
    order = np.argsort(-sim, kind="stable")
    ranked: list[tuple[str, int, float]] = []
    seen = set()
    for position in order:
        identity = str(gallery.iloc[position]["confirmed_identity"]).strip()
        if identity in seen:
            continue
        seen.add(identity)
        ranked.append((identity, int(position), float(sim[position])))
        if len(ranked) >= topk:
            break
    return ranked


def load_pool(emb_path: Path, meta_path: Path,
              series_index: pd.DataFrame | None = None) -> pd.DataFrame:
    """散图池：预提取的 r4+YOLO 裁剪特征 + meta（image_id / relative_path / session_id）。"""
    if series_index is None:
        raise ValueError("load_pool 必须提供由完整 manifest 构建的 series_index")
    emb, pool, config = load_verified_embedding_artifact(
        emb_path, meta_path, require_hashes=True, allow_nonfinite=True)
    pool["_emb"] = list(emb)
    pool = attach_series_manifest(pool, series_index, "pool")
    pool.attrs["artifact_config"] = config
    return pool


def load_gallery(emb_path: Path, meta_path: Path,
                 series_index: pd.DataFrame | None = None):
    """已确认个体的 r4+YOLO 裁剪特征（meta 含 confirmed_identity / 群）。"""
    if series_index is None:
        raise ValueError("load_gallery 必须提供由完整 manifest 构建的 series_index")
    emb, meta, config = load_verified_embedding_artifact(
        emb_path, meta_path, require_hashes=True)
    g = meta[meta["confirmed_identity"].notna()
             & (meta["confirmed_identity"].astype(str).str.strip() != "")].copy()
    g["emb"] = list(emb[g.index.to_numpy()])
    g = attach_series_manifest(g, series_index, "gallery")
    g.attrs["artifact_config"] = config
    return g


def assign_pool(pool: pd.DataFrame, gallery: pd.DataFrame, topk: int,
                threshold: float) -> pd.DataFrame:
    """每张散图 → 同 session、跨完整连拍串的 Top-K 个体候选。"""
    if topk < 1:
        raise ValueError("topk 必须至少为 1")
    pool = _require_series_metadata(pool, "pool")
    gallery = _require_series_metadata(gallery, "gallery")
    rows = []
    for _, q in pool.iterrows():
        base = {
            "image_id": q["image_id"],
            "group": q["group"],
            "relative_path": q.get("relative_path", ""),
            "series_id": q["series_id"],
            "series_unit": q["series_unit"],
        }
        if not np.isfinite(q["_emb"]).all():
            rows.append({"image_id": q["image_id"], "group": q["group"],
                         "relative_path": q.get("relative_path", ""),
                         "series_id": q["series_id"],
                         "series_unit": q["series_unit"],
                         "top1": "", "top1_score": np.nan,
                         "top1_image_id": "", "candidates": "",
                         "candidate_gallery_n": 0,
                         "excluded_same_series_n": 0,
                         "status": "missing_image"})
            continue
        same_session = gallery[gallery["group"] == q["group"]]
        if same_session.empty:
            rows.append({**base, "status": "no_gallery", "top1": "",
                         "top1_score": np.nan, "top1_image_id": "",
                         "candidates": "", "candidate_gallery_n": 0,
                         "excluded_same_series_n": 0})
            continue
        gal = _candidate_gallery(q, same_session)
        nonself = same_session[
            same_session["image_id"].astype(str) != str(q["image_id"])]
        excluded_same_series_n = len(nonself) - len(gal)
        if gal.empty:
            rows.append({**base, "status": "no_cross_series_candidate",
                         "top1": "", "top1_score": np.nan,
                         "top1_image_id": "", "candidates": "",
                         "candidate_gallery_n": 0,
                         "excluded_same_series_n": excluded_same_series_n})
            continue
        identity_best = _rank_identities(q["_emb"], gal, "emb", topk)
        cands = [f"{ident}@{score:.3f}"
                 for ident, _, score in identity_best]
        top_ident, top, top_score = identity_best[0]
        rows.append({
            **base,
            "top1": top_ident,
            "top1_score": top_score,
            "top1_image_id": gal.iloc[top]["image_id"],
            "candidates": "; ".join(cands),
            "candidate_gallery_n": len(gal),
            "excluded_same_series_n": excluded_same_series_n,
            "status": "candidate",
        })
    out = pd.DataFrame(rows, columns=[
        "image_id", "group", "relative_path", "series_id", "series_unit",
        "top1", "top1_score", "top1_image_id", "candidates",
        "candidate_gallery_n", "excluded_same_series_n", "status",
    ])
    # 低分提示（可能新个体）：最高分低于阈值
    if not out.empty:
        out.loc[
            out["top1_score"] < threshold,
            "status",
        ] = "low_confidence_new_candidate"
    return out


def resolve_reviewed_identity(
    raw_identity: object,
    group: str,
    gallery: pd.DataFrame,
    migration: dict[str, str] | None = None,
) -> str:
    """把审核 ID 当不透明字符串；仅允许显式一对一迁移表做兼容。"""
    if pd.isna(raw_identity):
        raise ValueError("reviewed_identity 不能为空")
    raw = str(raw_identity).strip()
    if not raw:
        raise ValueError("reviewed_identity 不能为空")
    scoped = gallery[gallery["group"] == group]["confirmed_identity"].astype(str)
    if raw in set(scoped):
        return raw
    mapping = migration or {}
    qualified = f"{group}_{raw}"
    return mapping.get(raw, mapping.get(qualified, raw))


def load_confirmed_reviews(path: Path) -> pd.DataFrame:
    """以字符串读取审核表，保留前导零与 ``NA`` 等合法不透明 ID。"""
    reviews = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"image_id", "review_status", "reviewed_identity"}
    missing = sorted(required - set(reviews.columns))
    if missing:
        raise ValueError(f"reviews 缺少列: {missing}")
    reviews = reviews.copy()
    for column in required:
        reviews[column] = reviews[column].astype(str).str.strip()
    confirmed = reviews[
        (reviews["review_status"].str.lower() == "confirmed")
        & (reviews["reviewed_identity"] != "")
    ].copy()
    if (confirmed["image_id"] == "").any():
        raise ValueError("confirmed reviews 存在空 image_id")
    duplicated = confirmed["image_id"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = confirmed.loc[duplicated, "image_id"].unique().tolist()
        raise ValueError(
            "confirmed reviews 的 image_id 必须唯一，重复示例: "
            f"{duplicate_ids[:5]}")
    return confirmed


def eval_within_group(gallery: pd.DataFrame) -> dict:
    """评估同 session 跨串 R@1；仅纳入同时有正例和异体负例的 query。"""
    gallery = _require_series_metadata(gallery, "gallery")
    per_session: dict[str, dict] = {}
    total = {
        "n_total": 0,
        "n_evaluated": 0,
        "n_hits": 0,
        "n_skipped_total": 0,
        "n_skipped_no_cross_series_positive": 0,
        "n_skipped_no_within_session_negative": 0,
        "n_skipped_invalid_feature": 0,
    }

    for group, session_gallery in gallery.groupby("group", sort=True):
        counters = {key: 0 for key in total}
        for _, query in session_gallery.iterrows():
            counters["n_total"] += 1
            total["n_total"] += 1
            if not np.isfinite(query["emb"]).all():
                counters["n_skipped_invalid_feature"] += 1
                counters["n_skipped_total"] += 1
                total["n_skipped_invalid_feature"] += 1
                total["n_skipped_total"] += 1
                continue

            candidates = _candidate_gallery(query, session_gallery)
            query_identity = str(query["confirmed_identity"]).strip()
            candidate_identity = candidates["confirmed_identity"].map(
                lambda value: str(value).strip())
            cross_series_positive = candidates[
                (candidate_identity == query_identity)
                & (candidates["series_unit"] != query["series_unit"])
            ]
            within_session_negative = candidates[
                candidate_identity != query_identity]
            has_positive = not cross_series_positive.empty
            has_negative = not within_session_negative.empty
            if not has_positive:
                counters["n_skipped_no_cross_series_positive"] += 1
                total["n_skipped_no_cross_series_positive"] += 1
            if not has_negative:
                counters["n_skipped_no_within_session_negative"] += 1
                total["n_skipped_no_within_session_negative"] += 1
            if not (has_positive and has_negative):
                counters["n_skipped_total"] += 1
                total["n_skipped_total"] += 1
                continue

            ranked = _rank_identities(query["emb"], candidates, "emb", 1)
            hit = ranked[0][0] == query_identity
            counters["n_evaluated"] += 1
            counters["n_hits"] += int(hit)
            total["n_evaluated"] += 1
            total["n_hits"] += int(hit)

        counters["r1"] = (
            counters["n_hits"] / counters["n_evaluated"]
            if counters["n_evaluated"] else None)
        per_session[str(group)] = counters

    within_group = {
        group: metrics["r1"] for group, metrics in per_session.items()}
    return {
        "protocol": "within_session_cross_series",
        "within_group": within_group,
        "per_session": per_session,
        "overall_r1": (
            total["n_hits"] / total["n_evaluated"]
            if total["n_evaluated"] else None),
        **total,
        "cross_session_all_gallery": {
            "status": "not_reported",
            "reason": "跨 session individual_id 未对齐，不能把跨批次样本当确认负例。",
        },
    }


def build_parser(base: Path, cfg: dict | None = None) -> argparse.ArgumentParser:
    """构建散图划分 CLI 参数，确保同一版特征与 meta 成对使用。"""
    cfg = load_config("pipeline") if cfg is None else cfg
    pool_cfg = cfg.get("pool", {})
    query_cfg = cfg.get("query", {})
    parser = argparse.ArgumentParser(description="同群散图划分（群内 Top-K 候选）")
    parser.add_argument("--pool-embeddings", type=Path,
                        default=base / pool_cfg.get(
                            "embeddings",
                            "artifacts/r4_yolocrop_v3/pool/embeddings.npy"),
                        help="散图 r4+YOLO 裁剪特征（预提取）")
    parser.add_argument("--pool-meta", type=Path,
                        default=base / pool_cfg.get(
                            "meta",
                            "artifacts/r4_yolocrop_v3/pool/embeddings_meta.csv"))
    parser.add_argument("--gallery-embeddings", type=Path,
                        default=base / query_cfg.get(
                            "embeddings",
                            "artifacts/r4_yolocrop_v3/gallery/embeddings.npy"),
                        help="已确认个体的 r4+YOLO 裁剪特征")
    parser.add_argument("--gallery-meta", type=Path,
                        default=base / query_cfg.get(
                            "meta",
                            "artifacts/r4_yolocrop_v3/gallery/embeddings_meta.csv"))
    parser.add_argument(
        "--manifest", type=Path,
        default=base / "index" / "dataset_manifest.csv",
        help="全量数据 manifest；连拍串只能从该全集构建，不能用特征子集重算")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="低于此分标记为 low_confidence（历史 E4 参考值；需重标定）")
    parser.add_argument("--out", type=Path, default=base / "pool_assignment")
    parser.add_argument("--eval", action="store_true",
                        help="只跑群内 leave-one-out 评估（不划分散图）")
    parser.add_argument("--reviews", type=Path,
                        default=base / "pool_assignment" / "pool_reviews.csv",
                        help="人工审核记录：confirmed 行并入 gallery 重跑")
    parser.add_argument(
        "--identity-migration", type=Path,
        default=base / "pilot" / "individual_id_migration_v1_to_v2.csv",
        help="显式一对一旧 ID→canonical ID 映射；禁止按数值猜测 ID")
    return parser


def main():
    repo_root = Path(__file__).resolve().parents[3]
    cfg = load_config("pipeline")
    output_root = Path(cfg.get("output_root", "outputs"))
    base = output_root if output_root.is_absolute() else repo_root / output_root
    parser = build_parser(base, cfg)
    args = parser.parse_args()

    series_index = load_series_manifest(args.manifest)
    series_provenance = series_index.attrs["provenance"]
    gallery = load_gallery(
        args.gallery_embeddings, args.gallery_meta, series_index)
    require_generated_artifact_provenance(
        gallery.attrs["artifact_config"])
    if args.eval:
        metrics = eval_within_group(gallery)
        metrics["series_manifest"] = series_provenance
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "series_manifest_provenance.json").write_text(
            json.dumps(series_provenance, indent=2, ensure_ascii=False),
            encoding="utf-8")
        with open(args.out / "within_group_eval.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[eval] 同 session 跨串 R@1: {metrics['within_group']}")
        print(f"[eval] 可评估 {metrics['n_evaluated']}/{metrics['n_total']}；"
              f"跳过 {metrics['n_skipped_total']}（无跨串正例 "
              f"{metrics['n_skipped_no_cross_series_positive']}，"
              f"无批次内异体负例 "
              f"{metrics['n_skipped_no_within_session_negative']}）")
        print("[eval] 跨 session 全库 R@1 不报告：individual_id 尚未跨批次对齐。"
              f" → {args.out / 'within_group_eval.json'}")
        return

    pool = load_pool(args.pool_embeddings, args.pool_meta, series_index)
    require_generated_artifact_provenance(pool.attrs["artifact_config"])
    require_compatible_embedding_configs(
        pool.attrs["artifact_config"], gallery.attrs["artifact_config"],
        left_name="pool", right_name="gallery")

    # 第一遍：不含人工确认回写（基线，线上旧版）
    before = assign_pool(pool, gallery, args.topk, args.threshold)

    # 人工确认回写：reviews 中 confirmed 的散图并入 gallery（特征复用本池），
    # 已确认的图不再作为 query
    n_merged = 0
    if args.reviews and args.reviews.exists():
        conf = load_confirmed_reviews(args.reviews)
        migration = (
            load_identity_migration(args.identity_migration)
            if args.identity_migration and args.identity_migration.exists()
            else {})
        new_rows = []
        for _, r in conf.iterrows():
            qrow = pool[pool["image_id"] == r["image_id"]]
            if qrow.empty:
                print(f"[assign] 警告: review {r['image_id']} 不在散图池，跳过")
                continue
            group = qrow.iloc[0]["group"]
            identity = resolve_reviewed_identity(
                r["reviewed_identity"], group, gallery, migration)
            source = qrow.iloc[0]
            new_rows.append({"image_id": r["image_id"],
                             "confirmed_identity": identity,
                             "session_id": source["session_id"],
                             "relative_path": source.get("relative_path", ""),
                             "filename": source.get("filename", ""),
                             "series_id": source.get("series_id", ""),
                             "series_unit": source.get("series_unit", ""),
                             "emb": source["_emb"], "group": group})
        if new_rows:
            gallery = pd.concat([gallery, pd.DataFrame(new_rows)], ignore_index=True)
            pool = pool[~pool["image_id"].isin(conf["image_id"])].reset_index(drop=True)
            n_merged = len(new_rows)
            print(f"[assign] 并入人工确认散图 {n_merged} 张 → gallery；"
                  f"剩余 query {len(pool)} 张")

    out = assign_pool(pool, gallery, args.topk, args.threshold)

    diff = None
    changed = None
    if n_merged:
        b = before.set_index("image_id")
        a = out.set_index("image_id")
        common = b.index.intersection(a.index)
        diff = pd.DataFrame({
            "image_id": common,
            "before_top1": b.loc[common, "top1"],
            "after_top1": a.loc[common, "top1"],
            "before_score": b.loc[common, "top1_score"],
            "after_score": a.loc[common, "top1_score"],
            "relative_path": a.loc[common, "relative_path"],
        })
        diff["changed"] = diff["before_top1"] != diff["after_top1"]
        changed = diff[diff["changed"]]

    # 所有输入、溯源、review 与计算校验通过后才创建目录并写出结果。
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "series_manifest_provenance.json").write_text(
        json.dumps(series_provenance, indent=2, ensure_ascii=False),
        encoding="utf-8")
    before.to_csv(args.out / "pool_candidates_before.csv",
                  index=False, encoding="utf-8-sig")
    out.to_csv(args.out / "pool_candidates.csv", index=False,
               encoding="utf-8-sig")
    if diff is not None:
        diff.to_csv(args.out / "pool_assignment_diff.csv", index=False,
                    encoding="utf-8-sig")

    # 汇总
    n_low = (out["status"] == "low_confidence_new_candidate").sum()
    print(f"[assign] 散图 {len(pool)} 张（按 session: {pool['group'].value_counts().to_dict()}）"
          f"→ 已确认个体 gallery "
          f"{len(gallery)} 张 / {gallery['confirmed_identity'].nunique()} 个体")
    print(f"[assign] 低分（<{args.threshold}，疑似新个体）: {n_low} 张")
    top1_scores = out["top1_score"].dropna()
    print(f"[assign] Top1 分数: p50={top1_scores.median():.3f} "
          f"p75={top1_scores.quantile(.75):.3f} p90={top1_scores.quantile(.9):.3f}")
    print(f"[assign] → {args.out / 'pool_candidates.csv'}")

    # 回写对比：并入前后 Top1 变化（可追溯）
    if n_merged:
        assert diff is not None and changed is not None
        print(f"[assign] 回写对比: 共 {len(common)} 张可比较，改判 {len(changed)} 张")
        if len(changed):
            pairs = changed.groupby(["before_top1", "after_top1"]).size()
            for (fb, fa), n in pairs.items():
                print(f"[assign]   {fb} → {fa} × {n}")
            print(f"[assign] → {args.out / 'pool_assignment_diff.csv'}")


if __name__ == "__main__":
    main()
