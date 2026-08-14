"""
中华白海豚个体识别：同群散图划分（Pool Assignment）。

场景（用户确认，2026-08-14）：优先把同群内的未归属散图（70-79 直接文件，
01: 59 张 / 03: 148 张）划分到群内已确认个体，而不是跨群全库检索。
跨群匹配在这个场景不参与，因此 query 与 gallery 都限制在同一群（session）。

流程：
散图（中心裁剪 0.55）→ 预训练 MegaDescriptor 特征 → 与同群已确认个体
照片（裁剪特征）余弦检索 → Top-K 个体候选 + 分数。

语义：输出是 Candidate（候选划归），不代表自动确认；低分散图可能是
新个体候选，需人工审核。所有行保留 image_id / 原路径 / 群 / 候选分数。

用法：
    python scripts/assign_pool.py
    python scripts/assign_pool.py --topk 5 --threshold 0.55
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_pool(manifest_csv: Path) -> pd.DataFrame:
    """70-79 直接文件（无数字子文件夹）的散图，保留群信息。"""
    df = pd.read_csv(manifest_csv)
    loose = df[df.relative_path.str.contains(r"^70-79/[^/]+$", regex=True)].copy()
    loose["group"] = loose["session_id"].map({1: "01", 3: "03"})
    return loose


def load_gallery(pilot_csv: Path, emb_path: Path, meta_path: Path):
    """已确认个体的裁剪特征（按 image_id 对齐，保留 confirmed_identity / 群）。"""
    p = pd.read_csv(pilot_csv)
    c = p[p["confirmed_identity"].notna()
           & (p["confirmed_identity"].astype(str).str.strip() != "")].copy()
    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    assert len(emb) == len(meta), "gallery 特征与 meta 行数不一致"
    g = meta.merge(c[["image_id", "confirmed_identity", "session_id"]],
                   on="image_id", how="inner")
    idx = [list(meta["image_id"]).index(iid) for iid in g["image_id"]]
    g["emb"] = list(emb[idx])
    g["group"] = g["session_id"].map({1: "01", 3: "03"})
    return g


def assign_pool(pool: pd.DataFrame, gallery: pd.DataFrame, topk: int,
                threshold: float) -> pd.DataFrame:
    """每张散图 → 同群 Top-K 个体候选（照片级检索 → 个体级聚合）。"""
    rows = []
    for _, q in pool.iterrows():
        if np.isnan(q["_emb"]).any():
            rows.append({"image_id": q["image_id"], "group": q["group"],
                         "relative_path": q["relative_path"],
                         "top1": "", "top1_score": np.nan, "candidates": "",
                         "status": "missing_image"})
            continue
        gal = gallery[gallery["group"] == q["group"]]
        if gal.empty:
            rows.append({"image_id": q["image_id"], "group": q["group"],
                         "status": "no_gallery", "top1": "", "top1_score": np.nan,
                         "candidates": ""})
            continue
        emb_gal = np.stack(gal["emb"].to_numpy())
        sim = emb_gal @ q["_emb"]
        order = np.argsort(-sim)[:topk]
        cands = []
        for j in order:
            ident = gal.iloc[j]["confirmed_identity"]
            tag = f"{int(ident)}" if float(ident).is_integer() else f"{ident}"
            cands.append(f"{tag}@{sim[j]:.3f}")
        top = order[0]
        rows.append({
            "image_id": q["image_id"],
            "group": q["group"],
            "relative_path": q["relative_path"],
            "top1": gal.iloc[top]["confirmed_identity"],
            "top1_score": float(sim[top]),
            "candidates": "; ".join(cands),
            "status": "candidate",
        })
    out = pd.DataFrame(rows)
    # 低分提示（可能新个体）：最高分低于阈值
    out.loc[out["top1_score"] < threshold, "status"] = "low_confidence_new_candidate"
    return out


def eval_within_group(gallery: pd.DataFrame) -> dict:
    """群内 leave-one-out R@1：query 与 gallery 都限同群（与全库对比）。

    用同一批裁剪特征（预训练 MegaDescriptor），量化"限制同群"对检索的影响。
    """
    groups = {}
    for gname, gal in gallery.groupby("group"):
        emb = np.stack(gal["emb"].to_numpy())
        sim = emb @ emb.T
        np.fill_diagonal(sim, -1.0)
        top = sim.argmax(axis=1)
        ids = gal["confirmed_identity"].to_numpy()
        hit = ids[top] == ids
        groups[gname] = float(hit.mean())
    emb_all = np.stack(gallery["emb"].to_numpy())
    sim_all = emb_all @ emb_all.T
    np.fill_diagonal(sim_all, -1.0)
    ids_all = gallery["confirmed_identity"].to_numpy()
    r1_all = float((ids_all[sim_all.argmax(axis=1)] == ids_all).mean())
    return {"within_group": groups, "all_gallery_r1": r1_all}


def main():
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser = argparse.ArgumentParser(description="同群散图划分（群内 Top-K 候选）")
    parser.add_argument("--manifest", type=Path, default=base / "index" / "dataset_manifest.csv")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--gallery-embeddings", type=Path,
                        default=base / "embeddings_crop" / "embeddings.npy",
                        help="群内已确认个体的裁剪特征（预训练 MegaDescriptor）")
    parser.add_argument("--gallery-meta", type=Path,
                        default=base / "embeddings_crop" / "embeddings_meta.csv")
    parser.add_argument("--images-root", type=Path, default=Path("I:/"),
                        help="原始图片根目录（散图裁剪用，只读）")
    parser.add_argument("--crop-ratio", type=float, default=0.55,
                        help="散图中心裁剪窗口（与 crops 实验一致）")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="低于此分标记为 low_confidence（提示可能新个体，阈值待标定）")
    parser.add_argument("--out", type=Path, default=base / "pool_assignment")
    parser.add_argument("--eval", action="store_true",
                        help="只跑群内 leave-one-out 评估（不划分散图）")
    parser.add_argument("--reviews", type=Path,
                        default=base / "pool_assignment" / "pool_reviews.csv",
                        help="人工审核记录：confirmed 行并入 gallery 重跑")
    args = parser.parse_args()

    gallery = load_gallery(args.pilot, args.gallery_embeddings, args.gallery_meta)
    if args.eval:
        metrics = eval_within_group(gallery)
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "within_group_eval.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[eval] 群内 leave-one-out R@1: {metrics['within_group']}")
        print(f"[eval] 全库（不限制同群）R@1: {metrics['all_gallery_r1']:.3f} "
              f"→ 见 {args.out / 'within_group_eval.json'}")
        return

    # 散图裁剪 + 特征（与 crops 实验一致：中心 0.55 方窗）
    from PIL import Image, ImageOps

    from src.model.reid.embedding.base import MegaDescriptorAdapter

    pool = load_pool(args.manifest)
    model = MegaDescriptorAdapter()

    def crop_one(p: Path):
        img = ImageOps.exif_transpose(Image.open(p).convert("RGB"))
        w, h = img.size
        side = int(min(w, h) * args.crop_ratio)
        return img.crop(((w - side) // 2, (h - side) // 2,
                         (w - side) // 2 + side, (h - side) // 2 + side))

    # 散图路径无群前缀（如 70-79/xxx.JPG），需拼群目录；pilot 图已含前缀
    gdir = {1: "01", 3: "03"}
    paths = []
    for _, r in pool.iterrows():
        p = args.images_root / gdir[r["session_id"]] / r["relative_path"]
        if not p.exists():
            p = args.images_root / r["relative_path"]      # 兼容已含前缀的情况
        paths.append(p)
    missing = [i for i, p in enumerate(paths) if not p.exists()]
    if missing:
        print(f"[assign] 警告: {len(missing)} 张散图原图缺失（01 群），将标记为 missing 跳过")
    embs = np.full((len(pool), model.feat_dim), np.nan, dtype=np.float32)
    keep = [i for i, p in enumerate(paths) if p.exists()]
    imgs = [crop_one(paths[i]) for i in keep]
    for s in range(0, len(imgs), 32):          # 分批编码，避免整批驻留
        embs[keep[s:s + 32]] = model.encode(imgs[s:s + 32])
    pool["_emb"] = list(embs)

    # 第一遍：不含人工确认回写（基线，线上旧版）
    before = assign_pool(pool, gallery, args.topk, args.threshold)
    before.to_csv(args.out / "pool_candidates_before.csv",
                  index=False, encoding="utf-8-sig")

    # 人工确认回写：reviews 中 confirmed 的散图并入 gallery（特征复用本池），
    # 已确认的图不再作为 query
    n_merged = 0
    if args.reviews and args.reviews.exists():
        rev = pd.read_csv(args.reviews)
        conf = rev[(rev["review_status"] == "confirmed")
                   & rev["reviewed_identity"].notna()]
        new_rows = []
        for _, r in conf.iterrows():
            qrow = pool[pool["image_id"] == r["image_id"]]
            if qrow.empty:
                print(f"[assign] 警告: review {r['image_id']} 不在散图池，跳过")
                continue
            new_rows.append({"image_id": r["image_id"],
                             "confirmed_identity": float(r["reviewed_identity"]),
                             "session_id": qrow.iloc[0]["session_id"],
                             "emb": qrow.iloc[0]["_emb"],
                             "group": qrow.iloc[0]["group"]})
        if new_rows:
            gallery = pd.concat([gallery, pd.DataFrame(new_rows)], ignore_index=True)
            pool = pool[~pool["image_id"].isin(conf["image_id"])].reset_index(drop=True)
            n_merged = len(new_rows)
            print(f"[assign] 并入人工确认散图 {n_merged} 张 → gallery；"
                  f"剩余 query {len(pool)} 张")

    out = assign_pool(pool, gallery, args.topk, args.threshold)

    args.out.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["_emb"], errors="ignore").to_csv(
        args.out / "pool_candidates.csv", index=False, encoding="utf-8-sig")

    # 汇总
    n_low = (out["status"] == "low_confidence_new_candidate").sum()
    print(f"[assign] 散图 {len(pool)} 张（01: {(pool.group=='01').sum()} / "
          f"03: {(pool.group=='03').sum()}）→ 已确认个体 gallery "
          f"{len(gallery)} 张 / {gallery['confirmed_identity'].nunique()} 个体")
    print(f"[assign] 低分（<{args.threshold}，疑似新个体）: {n_low} 张")
    top1_scores = out["top1_score"].dropna()
    print(f"[assign] Top1 分数: p50={top1_scores.median():.3f} "
          f"p75={top1_scores.quantile(.75):.3f} p90={top1_scores.quantile(.9):.3f}")
    print(f"[assign] → {args.out / 'pool_candidates.csv'}")

    # 回写对比：并入前后 Top1 变化（可追溯）
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
        diff.to_csv(args.out / "pool_assignment_diff.csv", index=False,
                    encoding="utf-8-sig")
        changed = diff[diff["changed"]]
        print(f"[assign] 回写对比: 共 {len(common)} 张可比较，改判 {len(changed)} 张")
        if len(changed):
            pairs = changed.groupby(["before_top1", "after_top1"]).size()
            for (fb, fa), n in pairs.items():
                print(f"[assign]   {fb:.0f} → {fa:.0f} × {n}")
            print(f"[assign] → {args.out / 'pool_assignment_diff.csv'}")


if __name__ == "__main__":
    main()
