"""
批内簇级归档管线（2026-08-17）：新批次 → 检测 → 特征 → 候选聚类 → 簇级匹配 → 审核清单。

场景：项目最终目的 = 加速数据处理。新批次（或散图池）到达时自动走：
  YOLO 背鳍检测裁剪 → r3 特征 → HDBSCAN 批内候选聚类 → 每簇与历史库
  （已确认个体）多帧投票匹配 → 输出归档候选 + 疑似新个体候选 + 噪声。

输出语义（CLAUDE.md）：
- HDBSCAN 簇 = Candidate Cluster，不是个体；-1 噪声合法，不强制分配；
- 匹配结果 = Candidate（候选划归），人工确认后才能叫个体/入库；
- 所有行保留 image_id / relative_path / session_id，可追溯到原图；
- 每簇选代表图（与簇均值特征最接近的一帧，归档用）。

两档阈值（均来自实验标定，非铁律）：
- 簇级（多帧投票）：0.58 = E5 簇级拒识 FA≤5% 标定（known 侧 n=6 偏薄，参考值）；
- 单图（噪声/孤图退化）：0.50 = E4 图级 FA≤5% 区间下限。

用法：
    # 散图池验证（跳过检测/特征，用预提取产物）
    python scripts/pipeline_archival.py --pool
    # 新批次完整流程（清单 CSV：image_id,relative_path[,session_id]）
    python scripts/pipeline_archival.py --input-manifest 批次清单.csv --batch-name 批次名
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def score_img_to_individual(img_emb, gal_idx, gal_emb, gal_ind):
    """单张图 → 每个历史库个体：max over 该个体照片 cos。返回 {ind: score}。"""
    sims = img_emb @ gal_emb[gal_idx].T
    out = {}
    for g in np.unique(gal_ind):
        mask = gal_ind == g
        out[str(g)] = float(sims[mask].max())
    return out


def load_gallery(emb_path: Path, meta_path: Path):
    """历史库：已确认个体的 r3+YOLO 裁剪特征。返回 (emb, ind, info)。"""
    meta = pd.read_csv(meta_path)
    emb = np.load(emb_path)
    assert len(emb) == len(meta)
    keep = meta["confirmed_identity"].notna() & (
        meta["confirmed_identity"].astype(str).str.strip() != "")
    emb = emb[keep.to_numpy()]
    ind = np.asarray([str(x) for x in meta.loc[keep, "confirmed_identity"]])
    if len(ind) == 0:
        raise SystemExit("[pipeline] 历史库为空（无已确认个体），无法进行匹配。请先构建历史库特征。")
    return emb, ind, meta.loc[keep]


def run(args):
    base = Path(__file__).resolve().parents[1] / "outputs"
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 阶段 1-2：检测裁剪 + r3 特征（新批次）或复用散图池产物 ----------
    if args.pool:
        emb_path = base / "embeddings" / "embeddings_pool_r3_yolocrop.npy"
        meta_path = base / "embeddings" / "embeddings_pool_r3_yolocrop_meta.csv"
        print("[pipeline] 散图池模式：复用预提取 r3+YOLO 特征")
    else:
        # 新批次：检测裁剪 → 特征
        from scripts.extract_r3_yolocrop import extract
        man = pd.read_csv(args.input_manifest)
        crops_dir = out_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        det_rows = _detect_all(man, args.images_root, crops_dir, args)
        pd.DataFrame(det_rows).to_csv(out_dir / "crops_manifest.csv",
                                      index=False, encoding="utf-8-sig")
        emb_path = out_dir / "embeddings.npy"
        extract(crops_dir, out_dir / "crops_manifest.csv", args.input_manifest,
                emb_path, args.ckpt)
        meta_path = emb_path.with_name(emb_path.stem + "_meta.csv")

    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    assert len(emb) == len(meta)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    meta["ind"] = [str(x) for x in meta["image_id"]]
    print(f"[pipeline] 批次 {len(meta)} 张（session: {meta['session_id'].value_counts().to_dict()}）")

    # ---------- 阶段 3：HDBSCAN 候选聚类（-1 = 合法噪声） ----------
    try:
        import hdbscan
    except ImportError as e:
        raise SystemExit(f"缺少 hdbscan 依赖: {e}") from e
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size)
    labels = clusterer.fit_predict(emb)
    meta["cluster"] = labels
    meta["cluster_probability"] = clusterer.probabilities_
    n_clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)  # 真实簇数
    n_clustered_images = int((labels >= 0).sum())
    n_noise = int((labels == -1).sum())
    print(f"[pipeline] HDBSCAN: {len(meta)} 张 → {n_clusters} "
          f"候选簇 + 噪声 {n_noise} 张（{n_noise / len(meta):.1%}）")

    # ---------- 阶段 3.5：大簇内再聚类（子簇化） ----------
    # 大簇常为混簇（多只并在一起）。对 >= subcluster_min_size 的簇内部再聚一次，
    # 拆出"纯子簇 + 残噪"，人工审核以子簇为单元（更小更纯，一键判定可行）。
    # 语义：subcluster >= 0 = 纯子簇；-1 = 残噪（逐图退化，同噪声级）。
    meta["subcluster"] = -1
    for c in set(labels):
        if c == -1:
            continue
        grp_idx = np.where(labels == c)[0]
        if len(grp_idx) >= getattr(args, "subcluster_min_size", 4):
            try:
                sub = hdbscan.HDBSCAN(min_cluster_size=2,
                                      min_samples=1).fit_predict(emb[grp_idx])
                meta.loc[meta["cluster"] == c, "subcluster"] = sub
            except Exception as e:
                print(f"[pipeline] 簇 {c} 子簇化失败（保持整簇）: {e}")
                meta.loc[meta["cluster"] == c, "subcluster"] = 0
        else:
            meta.loc[meta["cluster"] == c, "subcluster"] = 0
    n_sub = int((meta["subcluster"] >= 0).sum())
    n_residual = int((meta["subcluster"] == -1).sum())
    print(f"[pipeline] 子簇化: 纯子簇单元 {n_sub} 张，残噪 {n_residual} 张"
          f"（{n_residual / len(meta):.1%}）")

    # ---------- 阶段 4：子簇级匹配历史库（多帧投票） ----------
    # 审核单元 = (cluster, subcluster)：cluster=-1 噪声 / subcluster=-1 残噪 → 逐图退化；
    # 纯子簇（subcluster>=0）→ 子簇内多帧投票（更纯，投票更可靠）。
    gal_emb, gal_ind, _ = load_gallery(args.gallery_embeddings, args.gallery_meta)
    gal_idx = np.arange(len(gal_emb))

    # 清空代表图目录（重跑批次可能残留旧命名的 cluster_xxx.jpg）
    rep_dst = out_dir / "representatives"
    if rep_dst.exists():
        shutil.rmtree(rep_dst)
    rep_dst.mkdir(exist_ok=True)

    rows = []          # 逐图
    cluster_rows = []  # 逐子簇汇总（仅纯子簇）
    for c in sorted(set(labels)):
        for sc in sorted(meta.loc[meta["cluster"] == c, "subcluster"].unique()):
            sub = meta[(meta["cluster"] == c) & (meta["subcluster"] == sc)]
            if c == -1 or sc == -1:
                # 噪声/残噪：点互不相似，不能合并成"簇"；逐图独立匹配（单图退化）
                for _, r in sub.iterrows():
                    s = score_img_to_individual(emb[r.name], gal_idx, gal_emb, gal_ind)
                    t1 = max(s, key=s.get)
                    s1 = s[t1]
                    status = ("noise" if s1 < args.threshold_image
                              else "noise_match_candidate")
                    rows.append({
                        "image_id": r["image_id"], "relative_path": r["relative_path"],
                        "session_id": r["session_id"], "cluster": int(c),
                        "subcluster": int(sc),
                        "cluster_probability": r["cluster_probability"],
                        "top1": t1, "top1_score": round(s1, 4),
                        "vote1_ratio": 1.0, "status": status,
                    })
                continue
            # 纯子簇：图-个体分数 = max → 簇内 mean（多帧投票）
            per_img = [score_img_to_individual(emb[i], gal_idx, gal_emb, gal_ind)
                       for i in sub.index]
            all_g = sorted(per_img[0].keys())
            agg = {g: float(np.mean([s[g] for s in per_img])) for g in all_g}
            top = sorted(agg, key=agg.get, reverse=True)[: args.topk]
            t1 = top[0]
            s1 = agg[t1]
            vote1 = float(np.mean([max(s, key=s.get) == t1 for s in per_img]))
            status = ("match" if s1 >= args.threshold_cluster
                      else "suspected_new")
            # 代表图：与子簇均值特征最接近的一帧（归档用）
            mean_feat = np.mean(emb[sub.index], axis=0)
            rep_i = int((emb[sub.index] @ mean_feat).argmax())
            rep = sub.iloc[rep_i]
            # 复制代表图到 representatives/（新批次模式下裁剪图就在 out_dir 内）
            if args.pool:
                src = base / "crops_yolo_pool" / f"{rep['image_id']}.jpg"
            else:
                src = crops_dir / f"{rep['image_id']}.jpg"
            if src.exists():
                shutil.copy2(src, rep_dst / f"cluster_{c:03d}_sub{sc}.jpg")

            for _, r in sub.iterrows():
                rows.append({
                    "image_id": r["image_id"], "relative_path": r["relative_path"],
                    "session_id": r["session_id"], "cluster": int(c),
                    "subcluster": int(sc),
                    "cluster_probability": r["cluster_probability"],
                    "top1": t1, "top1_score": round(s1, 4),
                    "vote1_ratio": round(vote1, 2), "status": status,
                })
            cluster_rows.append({
                "cluster": int(c), "subcluster": int(sc), "n_members": len(sub),
                "members": "; ".join(sub["image_id"]),
                "rep_image_id": rep["image_id"], "rep_relative_path": rep["relative_path"],
                "top1": t1, "top1_score": round(s1, 4),
                "top2": top[1] if len(top) > 1 else "", "top3": top[2] if len(top) > 2 else "",
                "vote1_ratio": round(vote1, 2), "status": status,
            })

    # ---------- 阶段 5：汇总与落盘 ----------
    if not rows:
        raise SystemExit("[pipeline] 无任何可处理图片，请检查输入清单")
    out_img = pd.DataFrame(rows)
    out_img.to_csv(out_dir / "clusters.csv", index=False, encoding="utf-8-sig")
    if cluster_rows:
        pd.DataFrame(cluster_rows).to_csv(out_dir / "cluster_matches.csv",
                                          index=False, encoding="utf-8-sig")
    cm = pd.DataFrame(cluster_rows)
    n_pure = len(cluster_rows)
    summary = {
        "n_images": int(len(meta)), "n_clusters": n_clusters,
        "n_clustered_images": n_clustered_images, "n_noise": n_noise,
        "n_pure_subclusters": n_pure,
        "noise_ratio": round(n_noise / len(meta), 3),
        # status_counts 口径：有纯子簇时统计纯子簇状态（match/suspected_new，
        # 噪声/残噪逐图状态见 clusters.csv）；全噪声批次（无纯子簇）才统计逐图状态
        "status_counts": (cm["status"].value_counts().to_dict() if cluster_rows
                          else out_img["status"].value_counts().to_dict()),
        "subcluster_size": ({"min": int(cm["n_members"].min()),
                             "median": int(cm["n_members"].median()),
                             "max": int(cm["n_members"].max())} if cluster_rows else {}),
        "threshold": {"cluster": args.threshold_cluster, "image": args.threshold_image},
        "note": "簇 = Candidate Cluster（-1 噪声合法）；子簇 = 大簇内再聚类（subcluster>=0 纯子簇，"
                "-1 残噪逐图）；match/suspected_new 均为候选，须人工核验后才能叫个体。"
                "阈值为实验标定参考值。",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(f"[pipeline] 状态分布: {summary['status_counts']}")
    print(f"[pipeline] → {out_dir}（clusters.csv / cluster_matches.csv / representatives/ / summary.json）")

    # ---------- 阶段 6（可选）：候选簇拼图（人工逐簇审核用） ----------
    if args.sheets:
        from scripts.contact_sheets import build_cluster_contact_sheets
        build_cluster_contact_sheets(out_dir / "clusters.csv",
                                     out_dir / "contact_sheets",
                                     args.images_root, max_sheets=args.max_sheets)


def _detect_all(man: pd.DataFrame, images_root: Path, crops_dir: Path, args) -> list:
    """逐图 YOLO 检测 + 非均匀扩展裁剪；未检出回退中心 0.45 窗。"""
    from PIL import Image
    from ultralytics import YOLO

    from scripts.detect_and_crop import expand_box
    model = YOLO(str(args.det_weights))
    rows, failures = [], []
    for _, r in man.iterrows():
        src = images_root / r["relative_path"]
        if not src.exists():
            failures.append((r["image_id"], "原图不存在"))
            continue
        img = Image.open(src).convert("RGB")
        w, h = img.size
        res = model.predict(str(src), conf=args.det_conf, imgsz=args.det_imgsz,
                            device=args.det_device, verbose=False)
        box, conf = None, 0.0
        if len(res) and len(res[0].boxes):
            b = res[0].boxes[0]
            x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
            conf = float(b.conf[0])
            box = expand_box(x0, y0, x1, y1, w, h,
                             args.det_pad_x, args.det_pad_up, args.det_pad_down)
            fallback = False
        else:
            side = int(min(w, h) * 0.45)
            left, top = (w - side) // 2, (h - side) // 2
            box = [left, top, side, side]
            fallback = True
            failures.append((r["image_id"], "未检出→中心裁剪回退"))
        crop = img.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
        crop.save(crops_dir / f"{r['image_id']}.jpg")
        rows.append({"image_id": r["image_id"], "relative_path": r["relative_path"],
                     "session_id": r.get("session_id", ""),
                     "x": box[0], "y": box[1], "w": box[2], "h": box[3],
                     "det_conf": round(conf, 4), "fallback": fallback})
    print(f"[pipeline] 检测裁剪: {len(rows)} 张，回退 {sum(1 for x in rows if x['fallback'])} 张")
    for fid, reason in failures[:10]:
        print(f"[pipeline]   FAIL {fid}: {reason}")
    return rows


def main():
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser = argparse.ArgumentParser(description="批内簇级归档管线")
    parser.add_argument("--pool", action="store_true",
                        help="散图池验证模式：复用 outputs/embeddings/embeddings_pool_r3_yolocrop.*")
    parser.add_argument("--input-manifest", type=Path,
                        help="新批次清单 CSV（image_id,relative_path[,session_id]）")
    parser.add_argument("--batch-name", type=str, default="batch",
                        help="批次名（输出目录名）")
    parser.add_argument("--images-root", type=Path, default=Path("I:/"))
    parser.add_argument("--ckpt", type=Path,
                        default=base / "metric_learning" / "r3" / "best.pt")
    parser.add_argument("--gallery-embeddings", type=Path,
                        default=base / "embeddings" / "embeddings_metric_r3_yolocrop.npy")
    parser.add_argument("--gallery-meta", type=Path,
                        default=base / "embeddings" / "embeddings_metric_r3_yolocrop_meta.csv")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--subcluster-min-size", type=int, default=4,
                        help="大簇内再聚类的簇大小下限（>= 此值的簇才拆子簇）")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--threshold-cluster", type=float, default=0.58,
                        help="簇级 match 阈值（E5 FA≤5% 标定参考值）")
    parser.add_argument("--threshold-image", type=float, default=0.50,
                        help="单图 match 阈值（噪声/孤簇退化，E4 区间下限）")
    parser.add_argument("--out", type=Path, default=base / "cluster_archival")
    parser.add_argument("--det-weights", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models" / "detectors" / "yolov8n_dorsalfin.pt")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-imgsz", type=int, default=1024)
    parser.add_argument("--det-device", default="cuda")
    parser.add_argument("--det-pad-x", type=float, default=0.30)
    parser.add_argument("--det-pad-up", type=float, default=0.15)
    parser.add_argument("--det-pad-down", type=float, default=0.60)
    parser.add_argument("--sheets", action="store_true", help="生成候选簇拼图（人工审核用）")
    parser.add_argument("--max-sheets", type=int, default=200)
    args = parser.parse_args()
    args.out = args.out / args.batch_name

    if not args.pool and not args.input_manifest:
        raise SystemExit("需要 --pool 或 --input-manifest 指定输入批次")
    run(args)


if __name__ == "__main__":
    main()
