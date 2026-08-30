"""
本地真实图库检索评估（批次内已确认个体）。

用 199 张已提取特征（outputs/embeddings/embeddings.npy）评估检索质量，
pilot_set.csv 的 individual_id 表示批次内已确认个体；编号没有完成跨批次
身份对齐，因此指标只评价清单定义下的批次内检索，不代表跨批次身份识别率。

三个口径：
- A 组代表检索（主口径）：每组挑代表图（分辨率最高）进 gallery，
  组内其余图做 query → 拍一张新照片能否找回所属组；
- B leave-one-out：每张图轮流 query、排除自身 → 客户端当前行为；
- C 跨序列严格：B 基础上，正确命中只算"不同连拍序列"的同组图
  （避免连续帧近乎相同导致虚高），无跨序列样本的 query 跳过。

指标：Recall@1/5/10、完整 gallery mAP，附随机基线期望 R@1 与 Top-1 误配混淆分布。

用法：
    python experiments/local_reid_benchmark.py
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

from whitewhale.reid.retrieval import cosine_topk  # noqa: E402
from whitewhale.data.sequence_groups import annotate_series  # noqa: E402


def load_data(embeddings: Path, meta: Path, pilot: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """加载特征与追溯信息（按 image_id 对齐，行序 = 特征行序）。"""
    emb = np.load(embeddings)
    m = pd.read_csv(meta, dtype={"image_id": str, "session_id": str})
    p = pd.read_csv(pilot, dtype={"image_id": str, "session_id": str})
    assert len(emb) == len(m), f"特征 {len(emb)} 与 meta {len(m)} 行数不一致"
    info = m[["image_id"]].merge(
        p, on="image_id", how="left", validate="one_to_one")
    assert len(info) == len(m), "merge 后行数变化（image_id 是否唯一）"
    if "filename" not in info.columns:
        if "relative_path" not in info.columns:
            raise ValueError("缺少 filename/relative_path，无法判定完整连拍串")
        info["filename"] = info["relative_path"].map(
            lambda value: Path(str(value)).name)
    annotate_series(info)
    return emb, info


def pick_representative(info: pd.DataFrame) -> pd.DataFrame:
    """每组挑代表图：分辨率（宽*高）最大，并列取第一张（确定性）。"""
    area = info["width"] * info["height"]
    info = info.assign(_area=area.fillna(0))
    return info.sort_values("_area", ascending=False).drop_duplicates(
        subset="individual_id", keep="first").drop(columns="_area")


def protocol_anchor(emb: np.ndarray, info: pd.DataFrame):
    """口径 A：组代表检索。返回 (q_idx, g_idx, gt_sets)。

    所有 query 共享同一 gallery（每组一张代表图），g_idx 为
    per-query 的索引列表（协议函数统一输出形状）。
    """
    rep = pick_representative(info)                       # 43 张代表
    g_all = np.flatnonzero(info.index.isin(rep.index))
    q_idx = [i for i in range(len(info)) if i not in set(g_all.tolist())]
    g_idx = [g_all] * len(q_idx)
    id_list = info["individual_id"].tolist()
    g_ids = [id_list[i] for i in g_all]
    g_pos = {gid: [k for k, g in enumerate(g_ids) if g == gid] for gid in set(g_ids)}
    gt_sets = [set(g_pos[id_list[i]]) for i in q_idx]
    return q_idx, g_idx, gt_sets


def protocol_leave_one_out(emb: np.ndarray, info: pd.DataFrame,
                           cross_sequence: bool = False):
    """口径 B/C：每张图轮流 query，排除自身。

    cross_sequence=True 时（口径 C），gallery 整体剔除 query 所在完整连拍串，
    正确命中只来自不同串的同个体图；无跨串正样本的 query 从指标中剔除。
    """
    id_list = info["individual_id"].tolist()
    series_list = info["series_id"].fillna("").astype(str).tolist()
    n = len(info)
    q_idx, g_idx, gt_sets = [], [], []
    for i in range(n):
        query_series = series_list[i]
        g = [j for j in range(n) if j != i and not (
            cross_sequence and query_series and series_list[j] == query_series)]
        if cross_sequence:
            gt_global = [j for j in g if id_list[j] == id_list[i]]
        else:
            gt_global = [j for j in g if id_list[j] == id_list[i]]
        if not gt_global:                # 无正样本（单图组 / 无跨序列样本）
            continue
        q_idx.append(i)
        g_idx.append(g)
        # cosine_topk 返回 gallery 局部位置，gt_sets 必须使用同一坐标系。
        gt_global_set = set(gt_global)
        gt_sets.append({position for position, global_index in enumerate(g)
                        if global_index in gt_global_set})
    return q_idx, g_idx, gt_sets


def evaluate(emb: np.ndarray, q_idx, g_idx, gt_sets, k: int):
    """检索并计算 Recall@K/full mAP；支持每个 query 的 gallery 长度不同。"""
    scores, idx = [], []
    recall_hits = {cutoff: 0 for cutoff in (1, 5, 10)}
    aps = []
    for query_number, (qi, gi) in enumerate(zip(q_idx, g_idx)):
        qi = int(qi)
        gi = np.asarray(gi, dtype=int)      # 统一 fancy index（防标量解包）
        if not len(gi):
            raise ValueError(f"query {qi} 的 gallery 为空")
        display_k = min(k, len(gi))
        full_scores, full_idx = cosine_topk(
            emb[qi:qi + 1], emb[gi], k=len(gi))
        display_scores = full_scores[0, :display_k]
        display_idx = full_idx[0, :display_k]
        scores.append(display_scores)
        idx.append(display_idx)

        positives = gt_sets[query_number]
        if not positives:
            raise ValueError(f"query {qi} 没有正样本位置")
        for cutoff in recall_hits:
            if any(int(position) in positives
                   for position in display_idx[:min(cutoff, display_k)]):
                recall_hits[cutoff] += 1

        # mAP 报告完整 gallery 排名，分母为全部正样本数，避免按“已检出的正例”归一化虚高。
        n_hit = 0
        precision_sum = 0.0
        for rank, position in enumerate(full_idx[0], start=1):
            if int(position) in positives:
                n_hit += 1
                precision_sum += n_hit / rank
        aps.append(precision_sum / len(positives))

    n_query = len(q_idx)
    metrics = {
        "recall_at_1": recall_hits[1] / n_query if n_query else 0.0,
        "recall_at_5": recall_hits[5] / n_query if n_query else 0.0,
        "recall_at_10": recall_hits[10] / n_query if n_query else 0.0,
        "mAP": float(np.mean(aps)) if aps else 0.0,
    }
    return metrics, scores, idx


def random_baseline(gt_sets, galleries) -> float:
    """随机基线期望 R@1：逐 query 计算 |gt| / |gallery| 后取平均。"""
    if not gt_sets:
        return 0.0
    return float(np.mean([
        len(gt) / len(gallery)
        for gt, gallery in zip(gt_sets, galleries)
    ]))


def main():
    base = REPO_ROOT / "outputs"
    parser = argparse.ArgumentParser(description="本地真实图库检索评估（批次内已确认个体）")
    parser.add_argument("--embeddings", type=Path, default=base / "embeddings" / "embeddings.npy")
    parser.add_argument("--meta", type=Path, default=base / "embeddings" / "embeddings_meta.csv")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=base / "reports" / "local_reid_benchmark")
    args = parser.parse_args()

    emb, info = load_data(args.embeddings, args.meta, args.pilot)
    args.out.mkdir(parents=True, exist_ok=True)

    protocols = [
        ("A_representative", protocol_anchor(emb, info)),
        ("B_leave_one_out", protocol_leave_one_out(emb, info)),
        ("C_cross_sequence", protocol_leave_one_out(emb, info, cross_sequence=True)),
    ]

    summary = {}
    all_rows = []
    for name, (q_idx, g_idx, gt_sets) in protocols:
        met, scores, idx = evaluate(emb, q_idx, g_idx, gt_sets, args.k)
        n_gal = np.mean([len(g) for g in g_idx])
        met["random_baseline_r1"] = random_baseline(gt_sets, g_idx)
        met["n_query"] = len(q_idx)
        met["n_gallery_avg"] = float(n_gal)
        summary[name] = met
        print(f"[{name}] query={len(q_idx)} gallery(avg)={n_gal:.0f} "
              f"R@1={met['recall_at_1']:.3f} R@5={met['recall_at_5']:.3f} "
              f"R@10={met['recall_at_10']:.3f} mAP={met['mAP']:.3f} "
              f"随机基线R@1={met['random_baseline_r1']:.3f}")

        # Top-1 混淆分布：未命中 query 统计误配组；
        # "同组非代表"（A 口径：同组其他 query 图）单列，不算误配。
        id_list = info["individual_id"].tolist()
        conf = {}
        same_group = 0
        for i, (qi, gi) in enumerate(zip(q_idx, g_idx)):
            top_position = int(idx[i][0])
            top_g = gi[top_position]
            if top_position in gt_sets[i]:
                continue
            if id_list[top_g] == id_list[qi]:
                same_group += 1          # 同组但非正样本（如另一张 query 图）
                continue
            key = (id_list[qi], id_list[top_g])
            conf[key] = conf.get(key, 0) + 1
        top_conf = sorted(conf.items(), key=lambda kv: -kv[1])[:5]
        print(f"    Top-1 同组非正样本 {same_group} 次；误配最多: " +
              ("; ".join(f"{a}→{b} ×{c}" for (a, b), c in top_conf) or "无"))

        # 明细（可追溯到 image_id / 批次内已确认身份）
        rows = []
        for i, (qi, gi) in enumerate(zip(q_idx, g_idx)):
            for j in range(min(args.k, len(idx[i]))):
                gallery_position = int(idx[i][j])
                gi_j = gi[gallery_position]
                rows.append({
                    "protocol": name,
                    "query_image_id": info.iloc[qi]["image_id"],
                    "query_individual_id": id_list[qi],
                    "rank": j + 1,
                    "cand_image_id": info.iloc[gi_j]["image_id"],
                    "cand_individual_id": id_list[gi_j],
                    "score": float(scores[i][j]),
                    "hit": int(gallery_position in gt_sets[i]),
                })
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(args.out / "topk_candidates.csv", index=False)
    with open(args.out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "note": "individual_id 为批次内已确认个体；跨批次身份未对齐，"
                    "结果不代表跨批次身份识别率。C 口径剔除完整同串照片。",
            "n_images": len(info),
            "n_confirmed_identities": info["individual_id"].nunique(),
            "mAP_scope": "full_gallery",
            "k": args.k, "per_protocol": summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
