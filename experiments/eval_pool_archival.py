"""
散图归档场景检索对比（TASKS 6.1c）：中心裁剪 vs YOLO 检测裁剪。

场景：输入一批散图（70-79 公共池），系统把同个体照片聚到一起、每簇取代表图归档。
散图无个体归属标签（A13：批内有主但具体归属未确认），采用两个可验证口径：

口径 A（相邻帧互检，弱正样本）：同批次、连拍帧号差 <= K 的散图对视为同一次目击连拍
（弱证据：批内有主 + 连拍连续性），在散图池内 Top-K 检索，看对侧另一张是否命中。
测"同一只的照片能否聚在一起"——归档的核心动作。

口径 B（归档把握度）：散图 -> 同批已确认个体（pilot）的 Top-1 相似度分布；
批内有主假设下，分数越高说明归档到某个已确认个体的把握越大。

特征：MegaDescriptor-T-224 预训练（与 assign_pool 同模型，无标签泄漏）。
注意：YOLO 分支中检测未检出（fallback=True）的图实际走了中心裁剪回退，
本脚本将其特征替换为中心分支特征，保证两种裁剪方式的唯一变量是"裁剪方式"。

用法：
    python scripts/eval_pool_archival.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from whitewhale.reid.embedding import extract_embeddings, make_embedder  # noqa: E402

# 帧号差阈值：<= K 视为同一次目击连拍（弱正样本）
SHOT_K = 2
# 口径 B 分数档位
B_BANDS = [(0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]

SHOT_RE = re.compile(r"(\d+)_(\d{8})_(\w+)_(\d+)_(\w+)_(\d+)\.JPG$")


def parse_shot(rel_path: str):
    """从 RAY 命名提取批次与连拍帧号；非 RAY 命名返回 (None, None)。"""
    m = SHOT_RE.search(str(rel_path).replace("\\", "/"))
    if not m:
        return None, None
    return m.group(4), int(m.group(6))


def build_adjacent_pairs(df: pd.DataFrame, k: int):
    """同批次内连拍帧号差 <= k 的散图对（df 以 image_id 为 index）。"""
    pairs = []
    for _, sub in df.groupby("session"):
        sub = sub.sort_values("shot")
        idx = list(sub.index)
        for i in range(len(idx) - 1):
            d = sub["shot"].iloc[i + 1] - sub["shot"].iloc[i]
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


def run():
    base = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="散图归档场景检索对比（中心 vs YOLO 裁剪）")
    parser.add_argument("--loose", type=Path, default=base / "outputs" / "det_labels" / "pool_loose_fixed.csv")
    parser.add_argument("--pilot", type=Path, default=base / "outputs" / "pilot" / "pilot_set.csv")
    parser.add_argument("--center-crops", type=Path, default=base / "outputs" / "crops_center_pool")
    parser.add_argument("--yolo-crops", type=Path, default=base / "outputs" / "crops_yolo_pool")
    parser.add_argument("--images-root", type=Path, default=Path("src_dataset"))
    parser.add_argument("--feat-dir", type=Path, default=base / "outputs" / "embeddings_pool_archival")
    parser.add_argument("--out", type=Path, default=base / "outputs" / "reports" / "pool_archival")
    parser.add_argument("--model", default="hf-hub:BVRA/MegaDescriptor-T-224")
    parser.add_argument("--reuse-feats", action="store_true", help="跳过特征提取，复用 --feat-dir 已有特征")
    args = parser.parse_args()

    loose = pd.read_csv(args.loose)
    loose["session"] = loose["relative_path"].str.split("/").str[0].astype(int)
    shot_df = pd.DataFrame(loose["relative_path"].map(parse_shot).tolist(),
                           columns=["sess", "shot"])
    # 解析失败（非 RAY 命名）无帧号 → 不进口径 A
    loose["shot"] = pd.to_numeric(shot_df["shot"], errors="coerce")

    pilot = pd.read_csv(args.pilot)
    pilot["session"] = pd.to_numeric(pilot["session_id"], errors="coerce")

    # ---------- 特征提取（模型统一：预训练 MegaDescriptor） ----------
    feat = args.feat_dir
    feat.mkdir(parents=True, exist_ok=True)

    def rel_manifest(images: Path, ids, out_dir: Path):
        """裁剪目录 -> {image_id, relative_path={image_id}.jpg} 清单（供 extract_embeddings）。"""
        out_dir.mkdir(parents=True, exist_ok=True)
        mf = out_dir / f"{images.name}_manifest.csv"
        mf.write_text("image_id,relative_path\n" + "".join(f"{i},{i}.jpg\n" for i in ids),
                      encoding="utf-8")
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

    center_emb = np.load(feat / "center" / "embeddings.npy")
    yolo_emb = np.load(feat / "yolo" / "embeddings.npy")
    pilot_emb = np.load(feat / "pilot_full" / "embeddings.npy")
    # 特征行序 = 提取时的清单顺序（sorted）
    center_ids = [p.stem for p in sorted(args.center_crops.glob("*.jpg"))]
    yolo_manifest = pd.read_csv(args.yolo_crops / "crops_manifest.csv")
    yolo_ids = sorted(yolo_manifest["image_id"].tolist())

    # fallback 图（检测未检出，实际走了中心裁剪回退）：YOLO 分支替换为中心分支特征
    fallback_ids = set(yolo_manifest.loc[yolo_manifest["fallback"] == True, "image_id"])  # noqa: E712
    fallback_ids &= set(center_ids)
    if fallback_ids:
        fb_idx = [yolo_ids.index(i) for i in fallback_ids]
        fb_center = [center_ids.index(i) for i in fallback_ids]
        yolo_emb[fb_idx] = center_emb[fb_center]
        print(f"[eval] fallback 图 {len(fallback_ids)} 张：YOLO 分支特征替换为中心分支（变量纯净）")

    # 统一索引：image_id -> 特征行号（与提取顺序一致）
    pool = loose.set_index("image_id")
    center_of = {i: k for k, i in enumerate(center_ids)}
    yolo_of = {i: k for k, i in enumerate(yolo_ids)}

    results = {}
    # ---------- 口径 A：相邻帧互检 ----------
    pairs = build_adjacent_pairs(pool, SHOT_K)
    print(f"[A] 帧号差 <= {SHOT_K} 的弱正样本对: {len(pairs)}")
    pair_rows = {br: [] for br in ("center", "yolo")}
    for branch, emb, of in [("center", center_emb, center_of),
                            ("yolo", yolo_emb, yolo_of)]:
        # 索引 -> session 数组（用于限制同批 gallery）
        idx2sess = {j: pool.loc[i, "session"] for i, j in of.items()}
        out = {}
        for k in (1, 5, 10):
            hits = 0
            for a, b in pairs:
                qa, qb = of.get(a), of.get(b)
                if qa is None or qb is None:
                    continue
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
            out[f"pair_hit@{k}"] = hits / len(pairs)
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
        scores = {s: [] for s in (1, 3)}
        for _, row in pool.iterrows():
            s = int(row["session"])
            gal = pilot[pilot["session"] == s]
            if len(gal) == 0:
                continue
            gi = gal.index.to_numpy()
            sims = emb[of[row.name]] @ pilot_emb[gi].T
            top1 = float(sims.max())
            scores[s].append(top1)
            top1_rows.append({"image_id": row.name, "branch": branch,
                              "session": s, "top1": top1})
        all_scores = [v for v in scores.values() for v in v]
        band_stats = {}
        for lo, hi in B_BANDS:
            band_stats[f"in_[{lo},{hi})"] = sum(lo <= v < hi for v in all_scores) / len(all_scores)
        results[f"B_{branch}"] = {
            "n": len(all_scores),
            "top1_p50": float(np.median(all_scores)),
            "top1_mean": float(np.mean(all_scores)),
            "ge_0.6": sum(v >= 0.6 for v in all_scores) / len(all_scores),
            "ge_0.7": sum(v >= 0.7 for v in all_scores) / len(all_scores),
            "lt_0.55": sum(v < 0.55 for v in all_scores) / len(all_scores),
            "bands": band_stats,
        }
        print(f"  [{branch}] Top1 p50={results[f'B_{branch}']['top1_p50']:.3f} "
              f"ge0.6={results[f'B_{branch}']['ge_0.6']:.3f} "
              f"lt0.55={results[f'B_{branch}']['lt_0.55']:.3f}")
    pd.DataFrame(top1_rows).to_csv(args.out / "B_top1_detail.csv", index=False)

    results["_meta"] = {
        "model": args.model,
        "shot_k": SHOT_K,
        "weak_label_note": "口径 A 弱正样本：同批连拍帧号差<=K，视为同一次目击（批内有主假设下的弱证据）",
        "fallback_note": f"fallback 图 {len(fallback_ids)} 张，YOLO 分支特征替换为中心分支",
        "n_loose": len(loose), "n_pairs": len(pairs),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"[done] -> {args.out / 'metrics.json'}")


if __name__ == "__main__":
    run()
