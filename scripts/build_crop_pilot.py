"""
从裁剪清单构造"裁剪版 pilot"（供 extract_embeddings / local_reid_benchmark 使用）。

背景：local_reid_benchmark 需要 individual_id / sequence_guess / session_id / width /
height 等追溯字段（来自 pilot_set.csv），而裁剪 manifest 只有 image_id 与裁剪信息。
本脚本把两者按 image_id merge，并把 relative_path 重指向裁剪图（{image_id}.jpg），
使评估链路对裁剪目录透明：--images-root 指到裁剪输出目录即可。

用法：
    python scripts/build_crop_pilot.py --crops-manifest outputs/crops_yolo/crops_manifest.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 评估链路需要的字段
KEEP = ["image_id", "relative_path", "individual_id", "sequence_guess", "session_id",
        "label", "source_group", "width", "height"]


def main():
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser = argparse.ArgumentParser(description="构造裁剪版 pilot")
    parser.add_argument("--crops-manifest", type=Path,
                        default=base / "crops_yolo" / "crops_manifest.csv")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--out", type=Path, default=base / "crops_yolo" / "crops_pilot.csv")
    args = parser.parse_args()

    crops = pd.read_csv(args.crops_manifest)
    pilot = pd.read_csv(args.pilot)
    df = crops.merge(pilot[["image_id", "individual_id", "sequence_guess", "session_id",
                            "label", "source_group", "width", "height"]],
                     on="image_id", how="left")
    # relative_path 重指向裁剪图（相对 --images-root=裁剪输出目录）
    df["relative_path"] = df["image_id"] + ".jpg"
    df = df[KEEP]
    df.to_csv(args.out, index=False)
    print(f"[crop-pilot] {len(df)} 行 → {args.out}")
    print(f"[crop-pilot] 缺 individual_id: {df['individual_id'].isna().sum()} 行")


if __name__ == "__main__":
    main()
