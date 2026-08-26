"""
E5.1 公共逻辑：探针/库拆分 + 连拍串判定 + 评分（供展示与保守评估复用）。

保证展示页、常规评估、同串剔除评估三处使用完全相同的拆分与评分逻辑。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from whitewhale.data.sequence_groups import MAX_GAP, parse_ray_frame

SEED = 7
NO_FRAME = 999999  # 无连拍号(如 MO RES 文件)的哨兵值


def load_data(base: Path, stem: str = "embeddings_eval51_all") -> tuple[pd.DataFrame, np.ndarray]:
    """读全量特征与 meta（含 individual_id 标签），L2 归一化。

    stem 指定特征文件名主干（r3 默认；r4 重训后传
    "embeddings_eval51_all_r4"）。
    """
    meta = pd.read_csv(base / "outputs" / "embeddings" / f"{stem}_meta.csv",
                       dtype={"session_id": str})
    emb = np.load(base / "outputs" / "embeddings" / f"{stem}.npy")
    assert len(meta) == emb.shape[0], "特征行数与 meta 不一致"
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    meta["ind"] = meta["individual_id"].fillna("").astype(str)
    meta["filename"] = meta["relative_path"].map(lambda p: Path(str(p)).name)
    meta["frame"] = meta["filename"].map(
        lambda f: (parse_ray_frame(f) or (None, NO_FRAME))[1])
    return meta, emb


def split_probe_gallery(meta: pd.DataFrame, emb: np.ndarray, seed: int = SEED):
    """探针/库拆分：多图个体对半拆 query 簇，单图个体与无标签图整图入库。

    返回 (q_rows, gal_idx, gal_ind)：
    - q_rows: {individual_id: np.array(index)}，query 照片；
    - gal_idx: np.array(index)，库照片（含无标签干扰项）；
    - gal_ind: 与 gal_idx 对应的个体标签数组。
    """
    rng = np.random.default_rng(seed)
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
    return q_rows, gal_idx, gal_ind


def same_series(a: int, b: int) -> bool:
    """连拍号差 ≤ MAX_GAP 视为同串（平凡命中范围）；无连拍号不算同串。"""
    return a != NO_FRAME and b != NO_FRAME and abs(a - b) <= MAX_GAP


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


def frame_of(meta: pd.DataFrame, idx: int) -> int:
    """取某行的连拍号。"""
    return int(meta.loc[idx, "frame"])


def exclude_same_series(meta: pd.DataFrame, q_idx: int,
                        gal_idx: np.ndarray, gal_ind: np.ndarray):
    """保守口径：从库中剔除与 query 同串（连拍号差 ≤ MAX_GAP）的照片。

    返回剔除后的 (gal_idx, gal_ind)；无连拍号照片不剔除（无法判定）。
    """
    q_frame = frame_of(meta, q_idx)
    keep = np.asarray([not same_series(q_frame, frame_of(meta, i))
                       for i in gal_idx])
    return gal_idx[keep], gal_ind[keep]


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
    rng = np.random.default_rng(seed)
    gal_all, drop_all, q_rows = [], [], {}
    n_gallery_only = 0
    for ind, sub in meta.groupby("ind"):
        idx = np.array(sub.index.to_numpy(), copy=True)
        if ind == "" or len(sub) < 2:
            gal_all.append(idx)
            continue
        frames = meta.loc[idx, "frame"].to_numpy(dtype=int)
        series = group_series(frames)
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
        "n_query_images": int(sum(len(v) for v in q_rows.values())),
        "n_gallery_images": int(len(gal_idx)),
    }
    return q_rows, gal_idx, gal_ind, info
