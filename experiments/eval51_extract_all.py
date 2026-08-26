"""
E5.1 全量特征提取：9 批次 1040 张全部 YOLO 裁剪 + 度量学习特征。

正式链路（cross_time.build_gallery）只处理 20140806 两群，本脚本把它
泛化到全量 manifest，供 E5.1 全库同体检索评估使用。

流程：manifest 全量 → YOLO 检测裁剪（detect_and_crop，未检出回退中心窗）
→ embedding（extract_embeddings）→ 输出 npy + meta（merge pilot 个体标签）。

--ckpt 指定权重（r3 默认；r4 重训后传 outputs/metric_learning/r4/best.pt），
--out 指定输出文件（默认 embeddings_eval51_all.npy，r4 用别名避免覆盖）。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from whitewhale.detection.detector import detect_and_crop  # noqa: E402
from whitewhale.reid.embedding import extract_embeddings, make_embedder  # noqa: E402

BASE = REPO_ROOT
MANIFEST = BASE / "outputs" / "index" / "dataset_manifest.csv"
PILOT = BASE / "outputs" / "pilot" / "pilot_set.csv"
DET = BASE / "models" / "detectors" / "yolov8n_dorsalfin.pt"
CROPS = BASE / "outputs" / "crops_yolo_eval51"
DEFAULT_OUT = BASE / "outputs" / "embeddings" / "embeddings_eval51_all.npy"


def main():
    ap = argparse.ArgumentParser(description="E5.1 全量特征提取")
    ap.add_argument("--ckpt", type=Path,
                    default=BASE / "outputs" / "metric_learning" / "r3" / "best.pt")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    CKPT, OUT = args.ckpt, args.out

    m = pd.read_csv(MANIFEST, dtype={"session_id": str})
    # relative_path 需含 session 前缀（与 pilot_set / cross_time 约定一致）
    m = m.assign(relative_path=m["session_id"] + "/" + m["relative_path"])
    print(f"[extract] 全量清单 {len(m)} 张（9 批次）| 权重 {CKPT}")

    man = detect_and_crop(m, Path("I:/"), CROPS, DET, preview=False)
    model = make_embedder("metric-learning", metric_ckpt=CKPT)
    extract_embeddings(
        man, model, crops_dir=CROPS, out_path=OUT, missing="nan",
        model_cfg={"model": model.name, "crop": "yolo",
                   "ckpt": str(CKPT), "preprocess": "Resize256+CenterCrop224"})

    # 个体标签：labeled 才有（loose_known / ignored 无归属 → NaN，仅作干扰项）
    p = pd.read_csv(PILOT)[["image_id", "individual_id"]]
    meta = pd.read_csv(OUT.with_name(OUT.stem + "_meta.csv"))
    meta = meta.merge(p, on="image_id", how="left")
    meta.to_csv(OUT.with_name(OUT.stem + "_meta.csv"), index=False,
                encoding="utf-8-sig")
    print(f"[done] 特征 {len(meta)} 张 → {OUT}")
    print(f"[done] 带个体标签 {int(meta['individual_id'].notna().sum())} 张")


if __name__ == "__main__":
    main()
