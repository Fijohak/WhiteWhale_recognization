"""
跨时间批次管线驱动（E7 首跑验证，正式流程）。

流程：
1. 历史库 gallery：20140806 01/03 的 labeled（43 组）→ YOLO 裁剪 → r3 特征；
   个体标识 = individual_id（Anchor 组，Candidate 级历史库，人工确认前
   假设组=个体，与 E3-E5 评估口径一致）；
2. 新批次逐个（按 session）跑批内归档管线（见 whitewhale.pipeline.archival）：
   检测裁剪 → r3 特征 → HDBSCAN 批内候选聚类 → 子簇化 →
   簇级多帧投票匹配历史库 → 代表图 + 候选簇拼图（人工审核材料）。

输出：
- outputs/crops_yolo_gallery/                 历史库裁剪图
- outputs/embeddings/embeddings_metric_r3_yolocrop_v2.npy(+meta)  历史库特征
- outputs/cluster_archival/cross_time/<session>/  每批结果

CLI 入口见 scripts/run_cross_time_batch.py。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from whitewhale.data.image_store import get_image_store
from whitewhale.detection.detector import detect_and_crop
from whitewhale.pipeline.archival import run as run_archival
from whitewhale.reid.embedding import extract_embeddings, make_embedder

BASE = Path(__file__).resolve().parents[3]
# 数据根只从 config 读取一次（消除硬编码）
DATA_ROOT = get_image_store().root
MANIFEST = BASE / "outputs" / "index" / "dataset_manifest.csv"
PILOT = BASE / "outputs" / "pilot" / "pilot_set.csv"
CKPT = BASE / "outputs" / "metric_learning" / "r3" / "best.pt"
DET_WEIGHTS = BASE / "models" / "detectors" / "yolov8n_dorsalfin.pt"
GALLERY_SESSIONS = ["20140806 01", "20140806 03"]
OUT_ROOT = BASE / "outputs" / "cluster_archival" / "cross_time"
GAL_NPY = BASE / "outputs" / "embeddings" / "embeddings_metric_r3_yolocrop_v2.npy"
GAL_META = GAL_NPY.with_name(GAL_NPY.stem + "_meta.csv")


def build_gallery() -> None:
    """历史库：20140806 的 labeled → 检测裁剪 → r3 特征 → confirmed_identity=individual_id。"""
    m = pd.read_csv(MANIFEST, dtype={"session_id": str})
    gal = m[m["session_id"].isin(GALLERY_SESSIONS) & (m["label_status"] == "labeled")]
    gal_csv = OUT_ROOT / "gallery_manifest.csv"
    gal_csv.parent.mkdir(parents=True, exist_ok=True)
    # relative_path 需含 session 前缀（manifest 内层路径无前缀，与 pilot_set 约定一致）
    gal = gal.assign(relative_path=gal["session_id"] + "/" + gal["relative_path"])
    gal[["image_id", "relative_path", "session_id"]].to_csv(gal_csv, index=False)
    print(f"[gallery] 历史库 {len(gal)} 张（{gal['session_id'].value_counts().to_dict()}）")

    crops_dir = BASE / "outputs" / "crops_yolo_gallery"
    man = detect_and_crop(gal, DATA_ROOT, crops_dir, DET_WEIGHTS,
                          preview=False)
    model = make_embedder("metric-learning", metric_ckpt=CKPT)
    extract_embeddings(
        man, model, crops_dir=crops_dir, out_path=GAL_NPY,
        merge_from=None, missing="nan",
        model_cfg={"model": model.name, "crop": "yolo",
                   "ckpt": str(CKPT), "preprocess": "Resize256+CenterCrop224"})

    meta = pd.read_csv(GAL_META)
    # Candidate 级历史库：Anchor 组标识即历史库个体（人工确认后升级为 confirmed）
    ind = pd.read_csv(PILOT)[["image_id", "individual_id"]]
    meta = meta.merge(ind, on="image_id", how="left")
    meta["confirmed_identity"] = meta["individual_id"].fillna("")
    meta.drop(columns=["individual_id"], inplace=True)
    meta.to_csv(GAL_META, index=False, encoding="utf-8-sig")
    n = meta["confirmed_identity"].astype(str).str.strip().ne("").sum()
    print(f"[gallery] 历史库特征 {len(meta)} 张 / {n} 个个体 → {GAL_NPY}")


def build_query_manifest(session: str, m: pd.DataFrame) -> Path:
    """新批次清单：该 session 的 labeled + loose_known。"""
    q = m[m["session_id"] == session]
    q = q[q["label_status"].isin(["labeled", "loose_known"])]
    p = OUT_ROOT / "manifests" / f"{session}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    # relative_path 需含 session 前缀（archival 直接拼 images_root 读图）
    q = q.assign(relative_path=q["session_id"] + "/" + q["relative_path"])
    q[["image_id", "relative_path", "session_id"]].to_csv(p, index=False)
    print(f"[query] {session}: {len(q)} 张"
          f"（labeled {int((q['label_status'] == 'labeled').sum())} / "
          f"散图 {int((q['label_status'] == 'loose_known').sum())}）")
    return p


def run_batch(session: str, m: pd.DataFrame) -> None:
    man = build_query_manifest(session, m)
    args = argparse.Namespace(
        pool=False, input_manifest=man, images_root=DATA_ROOT, ckpt=CKPT,
        gallery_embeddings=GAL_NPY, gallery_meta=GAL_META,
        min_cluster_size=3, subcluster_min_size=4, topk=3,
        threshold_cluster=0.58, threshold_image=0.50,
        out=OUT_ROOT, det_weights=DET_WEIGHTS,
        det_conf=0.25, det_imgsz=1024, det_device="cuda",
        det_pad_x=0.30, det_pad_up=0.15, det_pad_down=0.60,
        sheets=True, max_sheets=200,
    )
    args.out = OUT_ROOT / session
    run_archival(args)
    print(f"[pipeline] {session} 完成 → {args.out}")


def main():
    ap = argparse.ArgumentParser(description="跨时间批次管线驱动（历史库=20140806）")
    ap.add_argument("--only-gallery", action="store_true", help="只构建历史库特征")
    ap.add_argument("--skip-gallery", action="store_true", help="跳过历史库构建（复用已有特征）")
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="只跑指定 session（默认全部新批次）")
    args = ap.parse_args()

    if not args.skip_gallery:
        build_gallery()
    if args.only_gallery:
        return
    m = pd.read_csv(MANIFEST, dtype={"session_id": str})
    sessions = args.sessions or sorted(
        s for s in m["session_id"].unique() if s not in GALLERY_SESSIONS)
    for s in sessions:
        run_batch(s, m)


if __name__ == "__main__":
    main()
