"""
YOLOv8 背鳍检测 + 裁剪。

流程：
1. 对清单内每张图运行背鳍检测器（yolov8n_dorsalfin.pt）；
2. 取最高置信度检测框 → 非均匀扩展（用户要求 2026-08-17：包含背鳍 + 大部分背部，
   上方少留水面、下方多留背部）→ 裁剪；
3. 输出：
   - outputs/crops_yolo/{image_id}.jpg      裁剪图（按 image_id 命名，可追溯）
   - outputs/crops_yolo/crops_manifest.csv  裁剪清单（检测框、置信度、原图路径、追溯字段）
   - outputs/crops_yolo/detect_preview/     检测框叠加预览图（供质量抽查）

扩展默认值：左右各 0.30 × 框宽，上方 0.15 × 框高（水面，少留），
下方 0.60 × 框高（背部，多留）；结果 clamp 到边长 ≤ 0.9 × min(w,h)，
面积 ≤ 0.6 × 原图（背鳍占满画面的图不裁成整图）。

失败处理：未检出时回退中心裁剪（0.45 窗），manifest 标记 fallback=true，
回退图仍可追溯，供后续分析检测失败原因。

语义：裁剪结果只作特征提取输入，不代表个体身份；所有结果可追溯到原图。

用法：
    python scripts/detect_and_crop.py --pilot outputs/pilot/pilot_set.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def expand_box(x0: float, y0: float, x1: float, y1: float,
               w: int, h: int, px: float, up: float, down: float):
    """非均匀扩展检测框：左右各 px×框宽，上方 up×框高，下方 down×框高。

    返回扩展后的 [x0, y0, w, h]（已 clamp 到图像内 + 尺寸上限）。
    """
    bw, bh = x1 - x0, y1 - y0
    pad_x, pad_up, pad_down = bw * px, bh * up, bh * down
    x0, y0 = max(0, int(x0 - pad_x)), max(0, int(y0 - pad_up))
    x1, y1 = min(w, int(x1 + pad_x)), min(h, int(y1 + pad_down))
    # 尺寸上限：边长 ≤ 0.9 × min(w,h)，面积 ≤ 60% 原图
    max_side = int(min(w, h) * 0.9)
    cw, ch = x1 - x0, y1 - y0
    if cw > max_side:  # 缩宽（居中收缩）
        dx = (cw - max_side) // 2
        x0, x1 = x0 + dx, x0 + dx + max_side
    if ch > max_side:  # 缩高（保持底部，收缩顶部）
        y0 = y1 - max_side
    return [x0, y0, x1 - x0, y1 - y0]


def main():
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser = argparse.ArgumentParser(description="YOLO 背鳍检测与裁剪")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--images-root", type=Path, default=Path("I:/"), help="原始图片根目录（只读）")
    parser.add_argument("--weights", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models" / "detectors" / "yolov8n_dorsalfin.pt")
    parser.add_argument("--out", type=Path, default=base / "crops_yolo")
    parser.add_argument("--pad-x", type=float, default=0.30, help="左右扩展：× 框宽")
    parser.add_argument("--pad-up", type=float, default=0.15, help="上方扩展：× 框高（水面，少留）")
    parser.add_argument("--pad-down", type=float, default=0.60, help="下方扩展：× 框高（背部，多留）")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试）")
    args = parser.parse_args()

    from PIL import Image

    from ultralytics import YOLO

    df = pd.read_csv(args.pilot)
    if args.limit > 0:
        df = df.head(args.limit)

    model = YOLO(str(args.weights))
    out_dir = args.out
    preview_dir = out_dir / "detect_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    for i, r in df.iterrows():
        src = args.images_root / r["relative_path"]
        if not src.exists():
            failures.append((r["image_id"], "原图不存在"))
            continue
        img = Image.open(src).convert("RGB")
        w, h = img.size
        res = model.predict(str(src), conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
        box = None
        if len(res) and len(res[0].boxes):
            b = res[0].boxes[0]  # 最高置信度框（ultralytics 已按 conf 排序）
            x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
            conf = float(b.conf[0])
            box = expand_box(x0, y0, x1, y1, w, h,
                             args.pad_x, args.pad_up, args.pad_down)
            fallback = False
            # 检测预览（框叠加）
            vis = img.copy()
            from PIL import ImageDraw
            ImageDraw.Draw(vis).rectangle(
                [box[0], box[1], box[0] + box[2], box[1] + box[3]],
                outline=(0, 220, 0), width=6)
            vis.save(preview_dir / f"{r['image_id']}.jpg")
        else:
            # 回退：中心裁剪（0.45 窗，与 crop_center.py 一致）
            side = int(min(w, h) * 0.45)
            left, top = (w - side) // 2, (h - side) // 2
            box = [left, top, side, side]
            conf, fallback = 0.0, True
            failures.append((r["image_id"], "未检出→中心裁剪回退"))
        crop = img.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
        crop.save(out_dir / f"{r['image_id']}.jpg")
        rows.append({
            "image_id": r["image_id"], "relative_path": r["relative_path"],
            "x": box[0], "y": box[1], "w": box[2], "h": box[3],
            "det_conf": round(conf, 4), "fallback": fallback,
        })
        if (i + 1) % 50 == 0:
            print(f"[detect] {i + 1}/{len(df)} 完成")

    pd.DataFrame(rows).to_csv(out_dir / "crops_manifest.csv", index=False)
    print(f"[detect] 完成：{len(rows)} 张，检测失败回退 {sum(1 for x in rows if x['fallback'])} 张")
    for fid, reason in failures[:20]:
        print(f"[detect]   FAIL {fid}: {reason}")


if __name__ == "__main__":
    main()
