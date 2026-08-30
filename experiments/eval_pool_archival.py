"""
散图归档场景检索对比（TASKS 6.1c）：中心裁剪 vs YOLO 检测裁剪。

场景：输入一批散图（70-79 公共池），系统把同个体照片聚到一起、每簇取代表图归档。
散图无个体归属标签（A13：批内有主但具体归属未确认），采用两个可验证口径：

口径 A（完整同串相邻帧互检，弱正样本）：同 session、同文件名序列键、同连续
帧段内帧号差 <= K 的散图对视为同一次目击连拍，在散图池内 Top-K 检索，
看对侧另一张是否命中。
测"同一只的照片能否聚在一起"——归档的核心动作。

口径 B（归档把握度）：散图 -> 同批已确认个体（pilot）的 Top-1 相似度分布；
批内有主假设下，分数越高说明归档到某个已确认个体的把握越大。

特征：MegaDescriptor-T-224 预训练（与 assign_pool 同模型，无标签泄漏）。
注意：YOLO 分支中检测未检出（fallback=True）的图实际走了中心裁剪回退，
本脚本将其特征替换为中心分支特征，保证两种裁剪方式的唯一变量是"裁剪方式"。

用法：
    python experiments/eval_pool_archival.py
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

from whitewhale.config import load_config  # noqa: E402
from whitewhale.data.sequence_groups import annotate_series  # noqa: E402
from whitewhale.reid.embedding import extract_embeddings, make_embedder  # noqa: E402

# 帧号差阈值：<= K 视为同一次目击连拍（弱正样本）
SHOT_K = 2
# 口径 B 分数档位
B_BANDS = [(0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]

def build_adjacent_pairs(df: pd.DataFrame, k: int):
    """完整同串内生成相邻帧弱正对（df 以 image_id 为 index）。"""
    pairs = []
    valid = df[df["series_id"].astype(str).str.strip() != ""]
    for _, sub in valid.groupby("series_id"):
        sub = sub.sort_values("frame")
        idx = list(sub.index)
        for i in range(len(idx) - 1):
            d = sub["frame"].iloc[i + 1] - sub["frame"].iloc[i]
            if 0 < d <= k:
                pairs.append((idx[i], idx[i + 1]))
    return pairs


def topk_hit(emb: np.ndarray, query_idx, gallery_idx, k, excluded):
    """query 在 gallery 内 Top-K，是否命中 excluded 中的任一图。特征已 L2 归一化。"""
    sims = emb[query_idx] @ emb[gallery_idx].T
    order = np.argsort(-sims)
    for pos in order[:k]:
        if gallery_idx[pos] in excluded:
            return True
    return False


def summarize_top1(scores: list[float]) -> dict:
    """汇总 Top-1 分数；无可评估样本时返回显式空指标。"""
    bands = {f"in_[{lo},{hi})": None for lo, hi in B_BANDS}
    if not scores:
        return {
            "n": 0,
            "top1_p50": None,
            "top1_mean": None,
            "ge_0.6": None,
            "ge_0.7": None,
            "lt_0.55": None,
            "bands": bands,
        }

    n_scores = len(scores)
    bands = {
        f"in_[{lo},{hi})": sum(lo <= value < hi for value in scores) / n_scores
        for lo, hi in B_BANDS
    }
    return {
        "n": n_scores,
        "top1_p50": float(np.median(scores)),
        "top1_mean": float(np.mean(scores)),
        "ge_0.6": sum(value >= 0.6 for value in scores) / n_scores,
        "ge_0.7": sum(value >= 0.7 for value in scores) / n_scores,
        "lt_0.55": sum(value < 0.55 for value in scores) / n_scores,
        "bands": bands,
    }


def load_embeddings_with_ids(path: Path) -> tuple[np.ndarray, list[str]]:
    """加载特征及同目录 meta，确保 image_id 与特征行严格对齐。"""
    emb = np.load(path)
    meta_path = path.with_name(f"{path.stem}_meta.csv")
    if not meta_path.exists():
        raise FileNotFoundError(f"特征缺少行序文件，请重新提取: {meta_path}")
    meta = pd.read_csv(meta_path)
    if "image_id" not in meta.columns:
        raise ValueError(f"特征行序文件缺少 image_id 列: {meta_path}")
    if len(meta) != len(emb):
        raise ValueError(
            f"特征与行序数量不一致: {path}={len(emb)}, {meta_path}={len(meta)}"
        )
    image_ids = meta["image_id"].astype(str)
    if image_ids.duplicated().any():
        raise ValueError(f"特征行序存在重复 image_id: {meta_path}")
    return emb, image_ids.tolist()


def resolve_session_id(raw_session: str, available_sessions: set[str]) -> str:
    """把旧式 01/03 批次编号唯一映射到完整 session_id。"""
    raw = str(raw_session).strip()
    if raw in available_sessions:
        return raw

    raw_tail = raw.lstrip("0") or "0"
    matches = []
    for session in available_sessions:
        tail = session.rsplit(" ", 1)[-1]
        if (tail.lstrip("0") or "0") == raw_tail:
            matches.append(session)
    if len(matches) > 1:
        raise ValueError(f"旧批次编号 {raw!r} 对应多个 session_id: {sorted(matches)}")
    return matches[0] if matches else raw


def run():
    base = REPO_ROOT
    images_root = Path(load_config("pipeline").get("data_root", "src_dataset"))
    if not images_root.is_absolute():
        images_root = base / images_root
    parser = argparse.ArgumentParser(description="散图归档场景检索对比（中心 vs YOLO 裁剪）")
    parser.add_argument("--loose", type=Path, default=base / "outputs" / "det_labels" / "pool_loose_fixed.csv")
    parser.add_argument("--pilot", type=Path, default=base / "outputs" / "pilot" / "pilot_set.csv")
    parser.add_argument("--center-crops", type=Path, default=base / "outputs" / "crops_center_pool")
    parser.add_argument("--yolo-crops", type=Path, default=base / "outputs" / "crops_yolo_pool")
    parser.add_argument("--images-root", type=Path, default=images_root)
    parser.add_argument("--feat-dir", type=Path, default=base / "outputs" / "embeddings_pool_archival")
    parser.add_argument("--out", type=Path, default=base / "outputs" / "reports" / "pool_archival")
    parser.add_argument("--model", default="hf-hub:BVRA/MegaDescriptor-T-224")
    parser.add_argument("--reuse-feats", action="store_true", help="跳过特征提取，复用 --feat-dir 已有特征")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    loose = pd.read_csv(args.loose)
    loose["image_id"] = loose["image_id"].astype(str)
    loose["session"] = loose["relative_path"].str.replace(
        "\\", "/", regex=False
    ).str.split("/").str[0]
    pilot_source = pd.read_csv(args.pilot)
    pilot_source["image_id"] = pilot_source["image_id"].astype(str)

    # ---------- 特征提取（模型统一：预训练 MegaDescriptor） ----------
    feat = args.feat_dir
    feat.mkdir(parents=True, exist_ok=True)

    def rel_manifest(images: Path, ids, out_dir: Path):
        """为裁剪目录生成保留原 session 的特征提取清单。"""
        out_dir.mkdir(parents=True, exist_ok=True)
        mf = out_dir / f"{images.name}_manifest.csv"
        session_by_id = loose.set_index("image_id")["session"].to_dict()
        pd.DataFrame({
            "image_id": ids,
            "relative_path": [f"{image_id}.jpg" for image_id in ids],
            "session_id": [session_by_id.get(image_id, "") for image_id in ids],
        }).to_csv(mf, index=False, encoding="utf-8-sig")
        return mf

    if not args.reuse_feats:
        # 散图：中心裁剪池
        center_ids = [p.stem for p in sorted(args.center_crops.glob("*.jpg"))]
        model = make_embedder("megadescriptor")
        extract_embeddings(rel_manifest(args.center_crops, center_ids, feat), model,
                           images_root=args.center_crops, out_path=feat / "center" / "embeddings.npy")
        # 散图：YOLO 裁剪池
        yolo_manifest = pd.read_csv(args.yolo_crops / "crops_manifest.csv")
        yolo_ids = sorted(yolo_manifest["image_id"].tolist())
        extract_embeddings(rel_manifest(args.yolo_crops, yolo_ids, feat), model,
                           images_root=args.yolo_crops, out_path=feat / "yolo" / "embeddings.npy")
        # gallery：pilot 整图（已确认个体，作口径 B 的检索目标）
        extract_embeddings(args.pilot, model, images_root=args.images_root,
                           out_path=feat / "pilot_full" / "embeddings.npy")

    center_emb, center_ids = load_embeddings_with_ids(feat / "center" / "embeddings.npy")
    yolo_emb, yolo_ids = load_embeddings_with_ids(feat / "yolo" / "embeddings.npy")
    pilot_emb, pilot_ids = load_embeddings_with_ids(feat / "pilot_full" / "embeddings.npy")

    # pilot_set 可能已追加新批次，必须按特征 meta 对齐，不能沿用 CSV 当前行号。
    pilot_order = pd.DataFrame({"image_id": pilot_ids})
    pilot = pilot_order.merge(
        pilot_source, on="image_id", how="left", validate="one_to_one"
    )
    if pilot["session_id"].isna().any():
        missing = pilot.loc[pilot["session_id"].isna(), "image_id"].tolist()
        raise ValueError(f"pilot 特征中的 image_id 不在当前 pilot_set: {missing[:5]}")
    pilot["session"] = pilot["session_id"].astype(str).str.strip()
    available_sessions = set(pilot["session"])
    loose["session"] = loose["session"].map(
        lambda value: resolve_session_id(value, available_sessions)
    )
    loose["session_id"] = loose["session"]
    loose["filename"] = loose["relative_path"].map(lambda value: Path(str(value)).name)
    annotate_series(loose)

    yolo_manifest = pd.read_csv(args.yolo_crops / "crops_manifest.csv")
    yolo_manifest["image_id"] = yolo_manifest["image_id"].astype(str)

    # fallback 图（检测未检出，实际走了中心裁剪回退）：YOLO 分支替换为中心分支特征
    fallback_ids = set(yolo_manifest.loc[yolo_manifest["fallback"] == True, "image_id"])  # noqa: E712
    fallback_ids &= set(center_ids) & set(yolo_ids)
    if fallback_ids:
        fb_idx = [yolo_ids.index(i) for i in fallback_ids]
        fb_center = [center_ids.index(i) for i in fallback_ids]
        yolo_emb[fb_idx] = center_emb[fb_center]
        print(f"[eval] fallback 图 {len(fallback_ids)} 张：YOLO 分支特征替换为中心分支（变量纯净）")

    # 统一索引：image_id -> 特征行号（与提取顺序一致）
    pool = loose.set_index("image_id")
    center_of = {i: k for k, i in enumerate(center_ids) if i in pool.index}
    yolo_of = {i: k for k, i in enumerate(yolo_ids) if i in pool.index}

    results = {}
    # ---------- 口径 A：相邻帧互检 ----------
    pairs = build_adjacent_pairs(pool, SHOT_K)
    print(f"[A] 完整同串内帧号差 <= {SHOT_K} 的弱正样本对: {len(pairs)}")
    pair_rows = {br: [] for br in ("center", "yolo")}
    for branch, emb, of in [("center", center_emb, center_of),
                            ("yolo", yolo_emb, yolo_of)]:
        # 索引 -> session 数组（用于限制同批 gallery）
        idx2sess = {j: pool.loc[i, "session"] for i, j in of.items()}
        evaluable_pairs = [(a, b) for a, b in pairs if a in of and b in of]
        out = {"n_evaluable_pairs": len(evaluable_pairs)}
        for k in (1, 5, 10):
            hits = 0
            for a, b in evaluable_pairs:
                qa, qb = of[a], of[b]
                # 同批散图池内检索（排除自身；目标 = 对侧图）
                gallery_a = [j for j in of.values()
                             if j != qa and idx2sess[j] == idx2sess[qa]]
                h_a = topk_hit(emb, qa, gallery_a, k, excluded={of[b]})
                gallery_b = [j for j in of.values()
                             if j != qb and idx2sess[j] == idx2sess[qb]]
                h_b = topk_hit(emb, qb, gallery_b, k, excluded={of[a]})
                if k == 1:
                    pair_rows[branch].append((a, b, int(h_a), int(h_b)))
                if h_a or h_b:
                    hits += 1
            out[f"pair_hit@{k}"] = hits / len(evaluable_pairs) if evaluable_pairs else None
        results[f"A_{branch}"] = out
        print(f"  [{branch}] {out}")
    pd.DataFrame(pair_rows["center"], columns=["a", "b", "hit_a@1", "hit_b@1"]).to_csv(
        args.out / "A_pair_detail_center.csv", index=False)
    pd.DataFrame(pair_rows["yolo"], columns=["a", "b", "hit_a@1", "hit_b@1"]).to_csv(
        args.out / "A_pair_detail_yolo.csv", index=False)

    # ---------- 口径 B：散图 -> 同批已确认个体 Top-1 ----------
    top1_rows = []
    for branch, emb, of in [("center", center_emb, center_of),
                            ("yolo", yolo_emb, yolo_of)]:
        scores: dict[str, list[float]] = {}
        missing_features = 0
        for _, row in pool.iterrows():
            s = str(row["session"])
            gal = pilot[pilot["session"] == s]
            if len(gal) == 0:
                continue
            if row.name not in of:
                missing_features += 1
                continue
            gi = gal.index.to_numpy()
            sims = emb[of[row.name]] @ pilot_emb[gi].T
            top1 = float(sims.max())
            scores.setdefault(s, []).append(top1)
            top1_rows.append({"image_id": row.name, "branch": branch,
                              "session": s, "top1": top1})
        all_scores = [v for v in scores.values() for v in v]
        summary = summarize_top1(all_scores)
        summary["n_missing_features"] = missing_features
        results[f"B_{branch}"] = summary
        if summary["n"]:
            print(f"  [{branch}] Top1 p50={summary['top1_p50']:.3f} "
                  f"ge0.6={summary['ge_0.6']:.3f} "
                  f"lt0.55={summary['lt_0.55']:.3f}")
        else:
            print(f"  [{branch}] 无可评估 Top-1 样本")
    pd.DataFrame(
        top1_rows, columns=["image_id", "branch", "session", "top1"]
    ).to_csv(args.out / "B_top1_detail.csv", index=False)

    results["_meta"] = {
        "model": args.model,
        "shot_k": SHOT_K,
        "weak_label_note": "口径 A 弱正样本：同 session+序列键+连续帧段内帧号差<=K，视为同一次目击（弱证据）",
        "fallback_note": f"fallback 图 {len(fallback_ids)} 张，YOLO 分支特征替换为中心分支",
        "n_loose": len(loose), "n_pairs": len(pairs),
    }
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"[done] -> {args.out / 'metrics.json'}")


if __name__ == "__main__":
    run()
