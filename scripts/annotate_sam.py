"""
背鳍框 SAM 辅助标注脚本。

流程：
1. 对 pilot 图片跑 SAM（vit_b）生成候选掩码；
2. 启发式选框：背鳍特写位于画面中央（2026-08-13 用户确认），
   取"中央得分"最高的掩码作为背鳍框（中央得分 = 掩码中心到图像中心的距离 + 面积占比）；
3. 输出：
   - outputs/det_labels/sam_candidates.csv  候选框（image_id + 像素 bbox + 中心得分）
   - outputs/det_labels/preview/{image_id}.jpg  框叠加预览图，供人工抽查修正

语义：
- 本脚本输出的是"辅助预标注"，未经人工抽查的框**不得用于训练**；
- 抽查确认的框由人工另存到 det_labels/confirmed/ 后才进入 YOLO 训练；
- 原始数据只读；每个框可追溯 image_id / relative_path / 中心得分。

用法：
    python scripts/annotate_sam.py --limit 20        # 试点：前 20 张
    python scripts/annotate_sam.py                   # 全量 pilot（199 张）
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# SAM 依赖延迟导入（未安装时仍可查看帮助）
def _load_sam(checkpoint: Path, device: str):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to(device)
    # 降低点密度与 NMS 阈值，避免碎掩码过多
    return SamAutomaticMaskGenerator(
        sam, points_per_side=32, pred_iou_thresh=0.88,
        stability_score_thresh=0.92, min_mask_region_area=800,
    )


def norm_box(box, img_w: int, img_h: int):
    """规范化 bbox 为 [x0, y0, x1, y1] 并 clamp 到图像内。

    防御：SAM 不同版本可能返回 xyxy 或 xywh；边缘掩码可能越界。
    """
    if len(box) != 4:
        raise ValueError(f"非法 bbox: {box}")
    x0, y0, x1, y1 = (float(v) for v in box)
    # 若为 xywh（x1<=x0 或 y1<=y0），转 xyxy
    if x1 <= x0 or y1 <= y0:
        x1, y1 = x0 + x1, y0 + y1
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return [max(0.0, x0), max(0.0, y0), min(img_w, x1), min(img_h, y1)]


def pick_dorsal_fin(masks, img_w: int, img_h: int):
    """启发式挑选背鳍框：中央得分最高 + 面积占比合理的掩码。

    依据（用户确认 2026-08-13）：80 分以上照片是背鳍特写，
    背鳍位于画面正中央、约占画面 20%（面积比 0.05~0.5 放宽范围）。
    中央得分 = 掩码中心距画面中心的加权倒数 × 面积比因子。
    """
    cx, cy = img_w / 2, img_h / 2
    best, best_score = None, -1.0
    for m in masks:
        area_ratio = m["area"] / (img_w * img_h)
        if not (0.005 <= area_ratio <= 0.50):
            continue
        x0, y0, x1, y1 = norm_box(m["bbox"], img_w, img_h)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dist = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5 / max(img_w, img_h)
        # 中心权重 2.0、面积权重 1.0：位置优先，面积次之
        score = 1.0 / (dist + 1e-3) + 0.5 * min(area_ratio / 0.2, 2.0)
        if score > best_score:
            best, best_score = m, score
    return best, best_score


def pad_box(box, pad_ratio: float, img_w: int, img_h: int):
    """按边长比例向外扩展 bbox（保留背鳍周围身体上下文，对齐 0.45 中心窗语义）。"""
    x, y, w, h = box
    pad_x, pad_y = w * pad_ratio, h * pad_ratio
    x0 = max(0, int(x - pad_x))
    y0 = max(0, int(y - pad_y))
    x1 = min(img_w, int(x + w + pad_x))
    y1 = min(img_h, int(y + h + pad_y))
    return [x0, y0, x1 - x0, y1 - y0]


def main():
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser = argparse.ArgumentParser(description="SAM 辅助背鳍框预标注")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--images-root", type=Path, default=Path("I:/"), help="原始图片根目录（只读）")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models" / "detectors" / "sam_vit_b_01ec64.pth")
    parser.add_argument("--out", type=Path, default=base / "det_labels")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张（试点）")
    parser.add_argument("--pad-ratio", type=float, default=0.20, help="bbox 外扩比例")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    df = pd.read_csv(args.pilot)
    if args.limit > 0:
        df = df.head(args.limit)

    import torch
    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[annotate] SAM 加载中（{args.checkpoint.name}，device={dev}）...")
    generator = _load_sam(args.checkpoint, dev)

    out_csv = args.out / "sam_candidates.csv"
    preview_dir = args.out / "preview"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    for i, r in df.iterrows():
        src = args.images_root / r["relative_path"]
        if not src.exists():
            failures.append((r["image_id"], "原图不存在"))
            continue
        img = Image.open(src).convert("RGB")
        w, h = img.size
        # 缩放到 SAM 标准输入（≤1024），避免 8GB 显卡 OOM；bbox 再映射回原图坐标
        scale = max(w, h) / 1024.0
        if scale > 1:
            new_w, new_h = int(w / scale), int(h / scale)
            img_small = img.resize((new_w, new_h), Image.LANCZOS)
        else:
            img_small, new_w, new_h = img, w, h
        sx, sy = w / new_w, h / new_h
        masks = generator.generate(np.asarray(img_small))
        best, score = pick_dorsal_fin(masks, new_w, new_h)
        if best is None:
            failures.append((r["image_id"], "无候选掩码"))
            continue
        x0, y0, x1, y1 = norm_box(best["bbox"], new_w, new_h)
        x0, y0 = x0 * sx, y0 * sy
        x1, y1 = x1 * sx, y1 * sy
        box = pad_box([x0, y0, x1 - x0, y1 - y0], args.pad_ratio, w, h)
        rows.append({
            "image_id": r["image_id"], "relative_path": r["relative_path"],
            "x": box[0], "y": box[1], "w": box[2], "h": box[3],
            "center_score": round(score, 3),
            "sam_iou": round(float(best["predicted_iou"]), 3),
            "n_masks": len(masks),
        })
        # 预览图：背鳍框（绿）+ 所有掩码轮廓（浅蓝，映射回原图坐标）
        vis = img.copy()
        d = ImageDraw.Draw(vis, "RGBA")
        for m in masks:
            bx0, by0, bx1, by1 = norm_box(m["bbox"], new_w, new_h)
            d.rectangle([bx0 * sx, by0 * sy, bx1 * sx, by1 * sy],
                        outline=(100, 180, 255, 90), width=1)
        # pad_box 返回 xywh，画框前转 xyxy
        d.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    outline=(0, 220, 0, 255), width=4)
        vis.save(preview_dir / f"{r['image_id']}.jpg")
        if (i + 1) % 20 == 0:
            print(f"[annotate] {i + 1}/{len(df)} 完成，失败 {len(failures)}")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[annotate] 完成：候选 {len(rows)} 张，失败 {len(failures)} 张")
    print(f"[annotate] 预览图：{preview_dir}")
    for fid, reason in failures:
        print(f"[annotate]   FAIL {fid}: {reason}")


if __name__ == "__main__":
    main()
