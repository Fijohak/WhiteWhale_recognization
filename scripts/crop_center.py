"""
中华白海豚个体识别：背鳍中心裁剪。

依据（用户确认，2026-08-13）：80 分以上照片是背鳍特写，背鳍位于画面正中央，
约占画面 20%。原始特征从整图提取，缩到 224px 时背鳍仅剩 ~100px，细节大量丢失；
中心裁剪后背鳍占满输入尺寸，分辨率细节提升约 2-4 倍，同时去除海面背景干扰。

输出：
- outputs/crops/{image_id}.jpg      裁剪图（按 image_id 命名，可追溯）
- outputs/crops/crops_manifest.csv  裁剪清单（image_id / crop 相对路径 /
  原图相对路径 / 追溯字段，可直接作后续训练的 pilot 使用）

原始数据只读；裁剪只做中心方窗（不缩放、不旋转、不增强）。
用法：
    python scripts/crop_center.py                # 默认 45% 边长（面积约 20%）
    python scripts/crop_center.py --ratio 0.5    # 自定义窗口
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def crop_one(img: Image.Image, ratio: float) -> Image.Image:
    """中心方窗裁剪：边长 = min(w,h) × ratio（保留背鳍+少量身体上下文）。"""
    w, h = img.size
    side = int(min(w, h) * ratio)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def main():
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser = argparse.ArgumentParser(description="背鳍中心裁剪（中心方窗）")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--images-root", type=Path, default=Path("I:/"),
                        help="原始图片根目录（只读）")
    parser.add_argument("--out", type=Path, default=base / "crops",
                        help="裁剪图输出目录")
    parser.add_argument("--ratio", type=float, default=0.45,
                        help="窗口边长比例（min 边长 × ratio；面积约 ratio²）")
    args = parser.parse_args()

    p = pd.read_csv(args.pilot)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in p.iterrows():
        src = args.images_root / r["relative_path"]
        if not src.exists():
            print(f"[crop] 跳过（原图不存在）: {src}")
            continue
        img = Image.open(src)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        crop = crop_one(img, args.ratio)
        crop_path = args.out / f"{r['image_id']}.jpg"
        crop.save(crop_path, "JPEG", quality=95)
        rows.append({
            "image_id": r["image_id"],
            "relative_path": f"{args.out.name}/{r['image_id']}.jpg",  # 相对 outputs/
            "source_relative_path": r["relative_path"],
            "source_group": r.get("source_group", ""),
            "session_id": r.get("session_id", ""),
            "quality_band": r.get("quality_band", ""),
            "confirmed_identity": r.get("confirmed_identity", ""),
            "crop_window": f"center {args.ratio:.2f}",
        })
    out_csv = args.out / "crops_manifest.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[crop] {len(rows)}/{len(p)} 张中心裁剪（{args.ratio:.0%} 边长）→ {args.out}")
    print(f"[crop] 清单 → {out_csv}")


if __name__ == "__main__":
    main()
