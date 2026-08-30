"""
E5.1 公共逻辑：探针/库拆分 + 连拍串判定 + 评分（供展示与保守评估复用）。

保证展示页、常规评估、同串剔除评估三处使用完全相同的拆分与评分逻辑。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from whitewhale.data.eval_set import attach_stable_series_from_manifest  # noqa: E402
from whitewhale.data.sequence_groups import (  # noqa: E402
    MAX_GAP,
    annotate_series as annotate_complete_series,
)
from whitewhale.reid.embedding import (  # noqa: E402
    load_verified_embedding_artifact,
    read_metadata_csv,
    require_generated_artifact_provenance,
)

SEED = 7
NO_FRAME = 999999  # 无连拍号(如 MO RES 文件)的哨兵值
NO_SERIES = ""     # 无法解析连拍序列时不参与同串判定


def stable_group_rng(sub: pd.DataFrame, seed: int) -> np.random.Generator:
    """按稳定 image_id 成员生成组内 RNG，身份 ID 改名不会改变评估拆分。"""
    members = (sorted(sub["image_id"].astype(str).tolist())
               if "image_id" in sub.columns
               else sorted(str(index) for index in sub.index))
    digest = hashlib.sha256(
        (str(seed) + "\x1f" + "\x1f".join(members)).encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def require_session_local_identities(meta: pd.DataFrame) -> None:
    """确认每个批次内身份 ID 只属于一个 session，避免未对齐身份被误合并。"""
    required = {"ind", "session_id"}
    missing = sorted(required - set(meta.columns))
    if missing:
        raise ValueError(f"E5 元数据缺少必需列: {missing}")
    invalid_session = (
        meta["session_id"].isna()
        | meta["session_id"].astype(str).str.strip().eq("")
    )
    if invalid_session.any():
        raise ValueError(
            f"E5 元数据存在 {int(invalid_session.sum())} 行空 session_id，"
            "无法执行批次隔离"
        )
    labeled = meta[meta["ind"].fillna("").astype(str) != ""].copy()
    if labeled.empty:
        return
    labeled["_session"] = labeled["session_id"].astype(str)
    conflicts = labeled.groupby("ind", sort=False)["_session"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        examples = ", ".join(map(str, conflicts.index[:5]))
        raise ValueError(
            "E5 仅接受 session 内身份：同一 individual_id 出现在多个 session，"
            f"请先按 session 建立身份命名空间；示例: {examples}"
        )


def load_data(base: Path, stem: str = "embeddings_eval51_all") -> tuple[pd.DataFrame, np.ndarray]:
    """读全量特征与 meta（含 individual_id 标签），L2 归一化。

    stem 指定特征文件名主干（r3 默认；r4 重训后传
    "embeddings_eval51_all_r4"）。
    """
    emb_path = base / "outputs" / "embeddings" / f"{stem}.npy"
    meta_path = base / "outputs" / "embeddings" / f"{stem}_meta.csv"
    emb, meta, config = load_verified_embedding_artifact(
        emb_path, meta_path, require_hashes=True)
    require_generated_artifact_provenance(config)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    meta["ind"] = meta["individual_id"].fillna("").astype(str)
    meta["filename"] = meta["relative_path"].map(lambda p: Path(str(p)).name)
    full_manifest = read_metadata_csv(
        base / "outputs" / "index" / "dataset_manifest.csv")
    stable = attach_stable_series_from_manifest(meta, full_manifest)
    for column in ("series_id", "sequence_key", "frame"):
        meta[column] = stable[column]
    return meta, emb


def annotate_series(meta: pd.DataFrame) -> None:
    """按 session、文件名序列键和连续帧段给照片标注 series_id。

    同一文件名序列键内，相邻帧号差不超过 MAX_GAP 的照片通过传递闭包归为
    同串；不同 session 或不同序列键即使帧号相近也不会归为同串。无法解析
    连拍信息的照片保留空 series_id，不参与同串剔除。
    """
    annotate_complete_series(meta)


def split_probe_gallery(meta: pd.DataFrame, emb: np.ndarray, seed: int = SEED):
    """探针/库拆分：多图个体对半拆 query 簇，单图个体与无标签图整图入库。

    返回 (q_rows, gal_idx, gal_ind)：
    - q_rows: {individual_id: np.array(index)}，query 照片；
    - gal_idx: np.array(index)，库照片（含无标签干扰项）；
    - gal_ind: 与 gal_idx 对应的个体标签数组。
    """
    require_session_local_identities(meta)
    gal_rows, q_rows = [], {}
    for ind, sub in meta.groupby("ind"):
        idx = np.array(sub.index.to_numpy(), copy=True)  # index 只读，需复制
        if ind == "" or len(sub) < 2:
            gal_rows.append(idx)
            continue
        stable_group_rng(sub, seed).shuffle(idx)
        n = len(idx)
        q_rows[ind] = idx[: n // 2]
        gal_rows.append(idx[n // 2:])
    gal_idx = np.concatenate(gal_rows)
    gal_ind = np.asarray([meta.loc[i, "ind"] for i in gal_idx])
    return q_rows, gal_idx, gal_ind


def same_series(a: str, b: str) -> bool:
    """两个非空 series_id 相同才视为同串（平凡命中范围）。"""
    return bool(a) and bool(b) and a == b


def score_img_to_individual(img_emb: np.ndarray, emb: np.ndarray,
                            gal_idx: np.ndarray, gal_ind: np.ndarray):
    """单张图 → 每个 gallery 个体：max over 该个体库照片 cos（无标签不入）。"""
    sims = img_emb @ emb[gal_idx].T
    out = {}
    for g in np.unique(gal_ind):
        if g == "":
            continue  # 无标签干扰项不构成个体候选
        mask = gal_ind == g
        out[str(g)] = float(sims[mask].max())
    return out


def restrict_gallery_to_session(meta: pd.DataFrame, q_idx: int,
                                gal_idx: np.ndarray, gal_ind: np.ndarray):
    """只保留 query 所属 session 的 gallery，跨 session 样本不参与相似度计算。"""
    if len(gal_idx) != len(gal_ind):
        raise ValueError("gallery 索引与身份标签长度不一致")
    query_session = str(meta.loc[q_idx, "session_id"])
    keep = np.asarray([
        str(meta.loc[int(index), "session_id"]) == query_session
        for index in gal_idx
    ], dtype=bool)
    return gal_idx[keep], gal_ind[keep]


def series_of(meta: pd.DataFrame, idx: int) -> str:
    """取某行的完整连拍串标识；无法解析时为空。"""
    value = meta.loc[idx, "series_id"]
    return "" if pd.isna(value) else str(value)


def exclude_same_series(meta: pd.DataFrame, q_idx: int,
                        gal_idx: np.ndarray, gal_ind: np.ndarray):
    """保留同 session gallery，并剔除与 query 完整 series_id 相同的照片。

    series_id 同时包含 session、文件名序列键和连续帧段；无法解析连拍信息的
    照片不参与同串判定。跨 session 样本不会成为候选或负类。
    """
    gal_idx, gal_ind = restrict_gallery_to_session(
        meta, q_idx, gal_idx, gal_ind)
    q_series = series_of(meta, q_idx)
    keep = np.asarray([not same_series(q_series, series_of(meta, i))
                       for i in gal_idx], dtype=bool)
    return gal_idx[keep], gal_ind[keep]


def evaluate_probe_clusters(meta: pd.DataFrame, emb: np.ndarray,
                            q_rows: dict[str, np.ndarray],
                            gal_idx: np.ndarray, gal_ind: np.ndarray,
                            reciprocal_rank_k: int = 10):
    """按同 session、跨串 gallery 评估探针簇，并如实记录不可评 query。

    每个探针图先做 session 隔离和完整同串剔除。正确身份在有效 gallery 中
    不存在时，该图只计入 skipped，不进入任何命中率或排名指标。单真值身份
    的排名指标为 MRR@K，不冒充多相关项定义下的 mAP。
    """
    if reciprocal_rank_k < 1:
        raise ValueError("reciprocal_rank_k 必须为正整数")

    records = []
    for target, raw_q_idx in q_rows.items():
        q_idx = np.asarray(raw_q_idx, dtype=int)
        sessions = {str(meta.loc[int(index), "session_id"]) for index in q_idx}
        if len(sessions) != 1:
            raise ValueError(
                f"探针簇 {target!r} 跨越多个 session，批次内身份协议无法评估"
            )
        session = next(iter(sessions))
        per_img = []
        gallery_sizes = []
        candidate_sizes = []
        skipped = 0
        for query_index in q_idx:
            effective_idx, effective_ind = exclude_same_series(
                meta, int(query_index), gal_idx, gal_ind)
            scores = score_img_to_individual(
                emb[int(query_index)], emb, effective_idx, effective_ind)
            if str(target) not in scores:
                skipped += 1
                continue
            per_img.append(scores)
            gallery_sizes.append(int(len(effective_idx)))
            candidate_sizes.append(int(len(scores)))

        n_evaluable = len(per_img)
        record = {
            "individual": str(target),
            "session": session,
            "n_query_total": int(len(q_idx)),
            "n_query": int(n_evaluable),
            "n_query_skipped": int(skipped),
            "n_gallery_images_effective_min": (
                int(min(gallery_sizes)) if gallery_sizes else None),
            "n_gallery_images_effective_max": (
                int(max(gallery_sizes)) if gallery_sizes else None),
            "n_gallery_images_effective_sum": int(sum(gallery_sizes)),
            "n_candidate_identities_effective_min": (
                int(min(candidate_sizes)) if candidate_sizes else None),
            "n_candidate_identities_effective_max": (
                int(max(candidate_sizes)) if candidate_sizes else None),
            "n_candidate_identities_effective_sum": int(sum(candidate_sizes)),
            "single_r1": None,
            "cluster_r1": None,
            "cluster_r5": None,
            "cluster_rank": None,
            "cluster_rr_at_k": None,
            "skipped_reason": (
                "" if n_evaluable else
                "no_same_session_cross_series_positive_in_gallery"),
        }
        if per_img:
            candidates = sorted({candidate for scores in per_img
                                 for candidate in scores})
            cluster_scores = {
                candidate: float(np.mean([
                    scores[candidate] for scores in per_img
                    if candidate in scores
                ]))
                for candidate in candidates
            }
            ranked = sorted(cluster_scores, key=cluster_scores.get, reverse=True)
            rank = ranked.index(str(target)) + 1
            record.update({
                "single_r1": float(np.mean([
                    max(scores, key=scores.get) == str(target)
                    for scores in per_img
                ])),
                "cluster_r1": int(rank == 1),
                "cluster_r5": int(rank <= 5),
                "cluster_rank": int(rank),
                "cluster_rr_at_k": (
                    float(1.0 / rank) if rank <= reciprocal_rank_k else 0.0),
            })
        records.append(record)

    columns = [
        "individual", "session", "n_query_total", "n_query",
        "n_query_skipped", "n_gallery_images_effective_min",
        "n_gallery_images_effective_max", "n_gallery_images_effective_sum",
        "n_candidate_identities_effective_min",
        "n_candidate_identities_effective_max",
        "n_candidate_identities_effective_sum", "single_r1", "cluster_r1",
        "cluster_r5", "cluster_rank", "cluster_rr_at_k", "skipped_reason",
    ]
    return pd.DataFrame(records, columns=columns)


def summarize_probe_clusters(records: pd.DataFrame,
                             reciprocal_rank_k: int = 10) -> dict:
    """汇总簇结果；分母只含真实可评 query，同时完整报告 skipped。"""
    evaluable = records[records["n_query"] > 0] if len(records) else records
    n_query = int(evaluable["n_query"].sum()) if len(evaluable) else 0
    result = {
        "n_probe_clusters_total": int(len(records)),
        "n_probe_clusters": int(len(evaluable)),
        "n_probe_clusters_skipped": int(len(records) - len(evaluable)),
        "n_query_images_total": (
            int(records["n_query_total"].sum()) if len(records) else 0),
        "n_query_images": n_query,
        "n_query_images_skipped": (
            int(records["n_query_skipped"].sum()) if len(records) else 0),
        "n_gallery_images_effective_mean": None,
        "n_gallery_images_effective_min": None,
        "n_gallery_images_effective_max": None,
        "n_candidate_identities_effective_mean": None,
        "n_candidate_identities_effective_min": None,
        "n_candidate_identities_effective_max": None,
        "single_R@1": None,
        "cluster_R@1": None,
        "cluster_R@5": None,
        f"cluster_MRR@{reciprocal_rank_k}": None,
    }
    if not len(evaluable):
        return result

    result.update({
        "n_gallery_images_effective_mean": float(
            evaluable["n_gallery_images_effective_sum"].sum() / n_query),
        "n_gallery_images_effective_min": int(
            evaluable["n_gallery_images_effective_min"].min()),
        "n_gallery_images_effective_max": int(
            evaluable["n_gallery_images_effective_max"].max()),
        "n_candidate_identities_effective_mean": float(
            evaluable["n_candidate_identities_effective_sum"].sum() / n_query),
        "n_candidate_identities_effective_min": int(
            evaluable["n_candidate_identities_effective_min"].min()),
        "n_candidate_identities_effective_max": int(
            evaluable["n_candidate_identities_effective_max"].max()),
        "single_R@1": float(
            (evaluable["single_r1"] * evaluable["n_query"]).sum() / n_query),
        "cluster_R@1": float(evaluable["cluster_r1"].mean()),
        "cluster_R@5": float(evaluable["cluster_r5"].mean()),
        f"cluster_MRR@{reciprocal_rank_k}": float(
            evaluable["cluster_rr_at_k"].mean()),
    })
    return result


def group_series_ids(series_ids: np.ndarray) -> list[np.ndarray]:
    """按完整 series_id 分组；无连拍信息的照片各自作为独立串。"""
    groups: dict[str, list[int]] = {}
    for pos, series_id in enumerate(series_ids):
        key = str(series_id) if series_id else f"__single_{pos}"
        groups.setdefault(key, []).append(pos)
    return [np.asarray(v, dtype=int) for v in groups.values()]


def group_series(frames: np.ndarray) -> list[np.ndarray]:
    """把一组连拍号聚成连拍串（传递闭包）：排序后相邻差 ≤ MAX_GAP 同串。

    - 连拍号 1, 3, 5 → 同一串（间隔 2 帧 = 删 1 帧，容忍）；
    - 无连拍号（NO_FRAME 哨兵）各自独立成串，不与其照片同串。
    返回串内位置数组的列表（按原始 frames 的位置索引）。
    """
    order = np.argsort(frames, kind="stable")
    series, cur = [], [int(order[0])]
    for i in order[1:]:
        last = cur[-1]
        same = (frames[i] != NO_FRAME and frames[last] != NO_FRAME
                and frames[i] - frames[last] <= MAX_GAP)
        if same:
            cur.append(int(i))
        else:
            series.append(np.asarray(cur, dtype=int))
            cur = [int(i)]
    series.append(np.asarray(cur, dtype=int))
    return series


def split_probe_gallery_series(meta: pd.DataFrame, emb: np.ndarray,
                               seed: int = SEED):
    """串抽样版探针/库拆分（防同串平凡命中的用户方案）。

    规则（用户 2026-08-25 指示）：
    - 多图个体照片按连拍串分组，每串随机挑出 ceil(串长/2) 张
      （11 张挑 ≤6 张，挑剩的进干扰池）；
    - 挑出的照片以"整串"为单位随机分边：整串进 query 或整串进库，
      多串个体兜底保证两侧都有（可评估）；
    - 挑剩的照片进干扰池：在库但不构成个体候选（gal_ind=""），
      因此 query 的候选里不可能出现同串照片——平凡命中被根除，
      而连拍照片的一半差异信息仍留在库特征中；
    - 单串个体（全串连拍）照片全部入库存作检索目标、不出 query
      （无法拆出跨串 query），数量在 info 中报告。

    返回 (q_rows, gal_idx, gal_ind, info)：
    - q_rows: {individual_id: np.array(index)}；
    - gal_idx/gal_ind: 库照片与候选标签（干扰池标签为 ""）；
    - info: 各池数量统计。
    """
    require_session_local_identities(meta)
    gal_all, drop_all, q_rows = [], [], {}
    n_gallery_only = 0
    for ind, sub in meta.groupby("ind"):
        idx = np.array(sub.index.to_numpy(), copy=True)
        if ind == "" or len(sub) < 2:
            gal_all.append(idx)
            continue
        rng = stable_group_rng(sub, seed)
        series_ids = meta.loc[idx, "series_id"].to_numpy(dtype=str)
        series = group_series_ids(series_ids)
        kept = []   # 每串挑出的半串（有效池）
        dropped = []  # 每串挑剩的（干扰池）
        for s in series:
            # s 是帧在 idx 内的位置，必须映射回 meta index 再入库/query
            n = len(s)
            pick = np.sort(rng.choice(n, (n + 1) // 2, replace=False))
            kept.append(idx[s[pick]])
            dropped.append(idx[s[np.delete(np.arange(n), pick)]])
        if len(series) == 1:
            # 单串个体：全部入库存作检索目标，不出 query（无法跨串评估）
            gal_all.append(np.concatenate(kept + dropped))
            n_gallery_only += 1
            continue
        # 整串分边：保证多串个体两侧都有（最大串兜底）
        qside = [bool(rng.random() < 0.5) for _ in kept]
        big = int(np.argmax([len(k) for k in kept]))
        if not any(qside):
            qside[big] = True
        if all(qside):
            qside[big] = False
        for k, is_q in zip(kept, qside):
            (q_rows.setdefault(ind, []).append(k) if is_q
             else gal_all.append(k))
        for d in dropped:
            drop_all.append(d)
    # 合并库：干扰池照片候选标签置 ""
    if not q_rows:
        raise ValueError("无 query 个体，检查标签数据")
    q_rows = {ind: np.concatenate(v) for ind, v in q_rows.items()}
    gal_idx = np.concatenate(gal_all + drop_all)
    drop_set = set(np.concatenate(drop_all).tolist())
    gal_ind = np.asarray(["" if int(i) in drop_set else meta.loc[i, "ind"]
                          for i in gal_idx])
    info = {
        "n_gallery_only_individuals": int(n_gallery_only),
        "n_interference_pool_images": int(len(np.concatenate(drop_all))),
        "n_query_images_planned": int(sum(len(v) for v in q_rows.values())),
        "n_gallery_images_total_all_sessions": int(len(gal_idx)),
    }
    return q_rows, gal_idx, gal_ind, info
