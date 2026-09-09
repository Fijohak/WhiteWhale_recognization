#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成「背鳍检测」申报 demo 图（YOLOv8 检测框 + 置信度 + 模型标注）。

产物：
    outputs/demo/detection_demo_grid.jpg      网格拼图（默认 3×2，申报主图）
    outputs/demo/singles/det_01_xxx.jpg       单张大图（带同样标注，供单独插图）

用法（conda activate torch 后，在仓库根目录执行）：
    python demo/make_detection_demo.py                      # 自动挑 6 张检测置信最高的图
    python demo/make_detection_demo.py --n 4 --cols 2       # 2×2
    python demo/make_detection_demo.py --images "20140806 03/80 and above/11/0796_20140417_HBi_03_RAY_1809.JPG"
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from whitewhale.config import load_config  # noqa: E402

# --------------------------------------------------------------------------
# 字体（中文 + 英文粗体；找不到回退 PIL 默认字体）
# --------------------------------------------------------------------------
FONT_CANDIDATES = {
    "zh": [
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ],
    "en_bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
    "en": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
}

# 配色（与审核网页主色一致）
C_NAVY = (30, 58, 95)
C_GREEN = (24, 200, 110)
C_CYAN = (90, 200, 245)
C_WHITE = (255, 255, 255)
C_DARK = (35, 35, 35)
C_GREY = (120, 120, 120)
C_BG = (246, 247, 249)

IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES[kind]:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _segments(text: str, f_zh, f_en):
    """按 CJK / 拉丁字符切分文本段（Droid Sans Fallback 不含拉丁字形，须混排）。"""
    def is_cjk(ch: str) -> bool:
        return ("\u2e80" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f"
                or "\uff00" <= ch <= "\uffef")

    segs, buf, cur = [], "", None
    for ch in text:
        c = is_cjk(ch)
        if buf and c != cur:
            segs.append((buf, f_zh if cur else f_en))
            buf = ""
        cur = c
        buf += ch
    if buf:
        segs.append((buf, f_zh if cur else f_en))
    return segs


def mixed_metrics(draw, text: str, f_zh, f_en) -> tuple[int, int, int]:
    """混排文本的 (宽度, ascent, descent)。"""
    w = asc = desc = 0
    for seg, f in _segments(text, f_zh, f_en):
        w += draw.textlength(seg, font=f)
        a, d = f.getmetrics()
        asc, desc = max(asc, a), max(desc, d)
    return w, asc, desc


def draw_mixed(draw, xy_baseline, text: str, f_zh, f_en, fill):
    """按基线混排绘制中英文文本。"""
    x, y = xy_baseline
    for seg, f in _segments(text, f_zh, f_en):
        draw.text((x, y), seg, font=f, fill=fill, anchor="ls")
        x += draw.textlength(seg, font=f)


def text_size(draw, text, fnt) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_tag(draw, xy, text, fnt, bg, fg, pad=6, radius=4):
    """带背景的圆角标签（左上角对齐到 xy）。"""
    x, y = xy
    w, h = text_size(draw, text, fnt)
    draw.rounded_rectangle([x, y - pad, x + w + 2 * pad, y + h + pad],
                           radius=radius, fill=bg)
    draw.text((x + pad, y), text, font=fnt, fill=fg)
    return w + 2 * pad, h + 2 * pad


def draw_dashed_rect(draw, box, fill, width=3, dash=12, gap=8):
    x0, y0, x1, y1 = box
    for seg in ((x0, y0, x1, y0), (x1, y0, x1, y1),
                (x1, y1, x0, y1), (x0, y1, x0, y0)):
        sx, sy, ex, ey = seg
        length = max(abs(ex - sx), abs(ey - sy))
        dx, dy = (ex - sx) / max(length, 1), (ey - sy) / max(length, 1)
        pos, step = 0.0, dash + gap
        while pos < length:
            e = min(pos + dash, length)
            draw.line([(sx + dx * pos, sy + dy * pos), (sx + dx * e, sy + dy * e)],
                      fill=fill, width=width)
            pos += step


# --------------------------------------------------------------------------
# 候选图片
# --------------------------------------------------------------------------
def collect_candidates(data_root: Path, per_batch: int, seed: int) -> list[Path]:
    """从每个批次的高分目录随机采样候选图（避免扫描全量 1000+ 张）。"""
    rng = random.Random(seed)
    out: list[Path] = []
    for batch in sorted(p for p in data_root.iterdir() if p.is_dir()):
        pool: list[Path] = []
        for band in ("80 and above", "70-79"):
            d = batch / band
            if not d.exists():
                continue
            pool += [p for p in d.rglob("*") if p.suffix in IMG_EXTS]
        if not pool:
            continue
        rng.shuffle(pool)
        out += pool[:per_batch]
    return out


def seq_key(p: Path) -> str:
    """连拍去重键：去掉文件名末尾的连拍号（RAY_0010 / 0323）。"""
    stem = p.stem
    if "_" in stem:
        return str(p.parent) + "/" + stem.rsplit("_", 1)[0]
    return str(p.parent) + "/" + stem


# --------------------------------------------------------------------------
# 单张渲染
# --------------------------------------------------------------------------
def render_cell(path: Path, det: dict | None, cell_w: int, cell_h: int,
                footer_h: int, model_tag: str, show_crop_box: bool,
                caption: str) -> Image.Image:
    canvas = Image.new("RGB", (cell_w, cell_h), C_WHITE)
    draw = ImageDraw.Draw(canvas)
    body_h = cell_h - footer_h

    im = Image.open(path).convert("RGB")
    # cover 填充：以检测框中心（无框则图中心）为裁剪焦点，避免 letterbox 白边
    if det and det["conf"] > 0:
        bx = det["box"]
        fcx, fcy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
    else:
        fcx, fcy = im.width / 2, im.height / 2
    scale = max(cell_w / im.width, body_h / im.height)
    sw, sh = int(im.width * scale + 0.5), int(im.height * scale + 0.5)
    ox = int(min(max(fcx * scale - cell_w / 2, 0), max(sw - cell_w, 0)))
    oy = int(min(max(fcy * scale - body_h / 2, 0), max(sh - body_h, 0)))
    im = im.resize((sw, sh), Image.LANCZOS).crop((ox, oy, ox + cell_w, oy + body_h))
    canvas.paste(im, (0, 0))
    ox, oy = -ox, -oy  # 原图坐标 → 画布坐标偏移

    f_small, f_tag = font("en", 15), font("en_bold", 17)
    f_zh, f_en = font("zh", 15), font("en", 15)

    if det and det["conf"] > 0:
        # ① 先画扩展裁剪框（底层，青色虚线）
        if show_crop_box:
            cx0, cy0 = ox + det["crop"][0] * scale, oy + det["crop"][1] * scale
            cx1, cy1 = cx0 + det["crop"][2] * scale, cy0 + det["crop"][3] * scale
            draw_dashed_rect(draw, [cx0, cy0, cx1, cy1], C_CYAN, width=3)
            if cy1 + 26 < body_h:
                draw_tag(draw, (cx0, cy1 + 4), "crop → ReID", f_small,
                         C_CYAN, (8, 40, 55), pad=4)

        # ② 再画检测框与置信度标签（顶层，绿色实线，压在裁剪框之上）
        x0, y0 = ox + det["box"][0] * scale, oy + det["box"][1] * scale
        x1, y1 = ox + det["box"][2] * scale, oy + det["box"][3] * scale
        draw.rectangle([x0, y0, x1, y1], outline=C_GREEN, width=5)
        label = f"dorsal fin  {det['conf']:.2f}"
        tw, th = text_size(draw, label, f_tag)
        ty = y0 - th - 14 if y0 - th - 14 >= 0 else min(y0 + 6, body_h - th - 10)
        draw_tag(draw, (x0, ty), label, f_tag, C_GREEN, (12, 40, 25))
    else:
        draw_tag(draw, (ox + 12, oy + 12), "no detection  (fallback)",
                 f_tag, (200, 60, 60), C_WHITE)

    # 右下角模型角标
    mt = f"model: {model_tag}"
    mw, mh = text_size(draw, mt, f_small)
    draw.rectangle([cell_w - mw - 20, body_h - mh - 16, cell_w - 6, body_h - 6],
                   fill=(0, 0, 0, 0))
    draw.rounded_rectangle([cell_w - mw - 20, body_h - mh - 16, cell_w - 8, body_h - 8],
                           radius=4, fill=(0, 0, 0))
    draw.text((cell_w - mw - 14, body_h - mh - 12), mt, font=f_small, fill=C_WHITE)

    # 底部信息条
    draw.rectangle([0, body_h, cell_w, cell_h], fill=C_DARK)
    draw_mixed(draw, (12, body_h + footer_h - 13), caption, f_zh, f_en, C_WHITE)
    return canvas


def render_single(path: Path, det: dict | None, max_side: int,
                  model_tag: str) -> Image.Image:
    """单张大图（等比缩放，标注同网格单元）。"""
    im = Image.open(path).convert("RGB")
    scale = min(1.0, max_side / max(im.width, im.height))
    nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    canvas = im.resize((nw, nh), Image.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    f_tag, f_small = font("en_bold", max(16, int(26 * scale) + 6)), font("en", 16)
    if det and det["conf"] > 0:
        x0, y0 = det["box"][0] * scale, det["box"][1] * scale
        x1, y1 = det["box"][2] * scale, det["box"][3] * scale
        draw.rectangle([x0, y0, x1, y1], outline=C_GREEN, width=max(4, int(6 * scale)))
        label = f"dorsal fin  {det['conf']:.2f}"
        tw, th = text_size(draw, label, f_tag)
        ty = y0 - th - 16 if y0 - th - 16 > 0 else y0 + 8
        draw_tag(draw, (x0, ty), label, f_tag, C_GREEN, (12, 40, 25))
    mt = f"model: {model_tag}"
    mw, mh = text_size(draw, mt, f_small)
    draw.rounded_rectangle([nw - mw - 22, nh - mh - 18, nw - 8, nh - 8],
                           radius=4, fill=(0, 0, 0))
    draw.text((nw - mw - 15, nh - mh - 14), mt, font=f_small, fill=C_WHITE)
    return canvas


# --------------------------------------------------------------------------
def main() -> None:
    cfg = load_config("pipeline")
    ap = argparse.ArgumentParser(description="生成背鳍检测 demo 图")
    ap.add_argument("--weights", default=cfg.get("detector_checkpoint",
                                                 "models/detectors/yolov8n_dorsalfin.pt"))
    ap.add_argument("--data-root", default=cfg.get("data_root", "src_dataset/"))
    ap.add_argument("--images", nargs="*", default=None,
                    help="手动指定图片（相对 data_root 或绝对/仓库相对路径），指定后不做自动挑选")
    ap.add_argument("--n", type=int, default=6, help="网格图片数（自动挑选时生效）")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--cell-w", type=int, default=760)
    ap.add_argument("--cell-h", type=int, default=560)
    ap.add_argument("--footer-h", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--per-batch", type=int, default=14, help="每批次采样候选数")
    ap.add_argument("--max-per-batch", type=int, default=2, help="同一批次最多入选张数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-crop-box", action="store_true", help="不画扩展裁剪框")
    ap.add_argument("--out", default="outputs/demo")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = REPO_ROOT / weights
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_tag = weights.stem  # yolov8n_dorsalfin

    # ---------- 1. 选图 ----------
    if args.images:
        paths = []
        for s in args.images:
            p = Path(s)
            paths.append(p if p.is_absolute() else (data_root / s if (data_root / s).exists() else REPO_ROOT / s))
    else:
        cand = collect_candidates(data_root, args.per_batch, args.seed)
        print(f"[demo] 候选 {len(cand)} 张，跑检测挑选 top-{args.n} ...")
        from ultralytics import YOLO

        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = YOLO(str(weights))
        confs: dict[Path, float] = {}
        ratios: dict[Path, float] = {}
        for p in cand:
            r = model.predict(str(p), conf=args.conf, imgsz=args.imgsz,
                              device=device, verbose=False)
            if len(r) and len(r[0].boxes):
                b = r[0].boxes[0]
                confs[p] = float(b.conf[0])
                x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
                with Image.open(p) as im:
                    ratios[p] = (x1 - x0) * (y1 - y0) / (im.width * im.height)
            else:
                confs[p], ratios[p] = 0.0, 0.0
        ranked = sorted(cand, key=lambda p: -confs[p])
        # 连拍去重 + 每批次限额（保证跨批次多样性）+ 框占比适中（特写/漏检图不适合展示）
        picked: list[Path] = []
        seen_seq, per_count = set(), {}
        for p in ranked:
            if confs[p] <= 0 or not (0.02 <= ratios[p] <= 0.45):
                continue
            k = seq_key(p)
            if k in seen_seq:
                continue
            b = p.relative_to(data_root).parts[0]
            if per_count.get(b, 0) >= args.max_per_batch:
                continue
            seen_seq.add(k)
            per_count[b] = per_count.get(b, 0) + 1
            picked.append(p)
            if len(picked) >= args.n:
                break
        paths = picked
        print(f"[demo] 入选 {len(paths)} 张："
              + ", ".join(f"{p.parents[1].name}/{p.name}({confs[p]:.2f})" for p in paths))
    if not paths:
        raise SystemExit("[demo] 没有可用的图片")

    # ---------- 2. 检测（取最高置信框 + 扩展裁剪框） ----------
    from ultralytics import YOLO

    import torch
    from whitewhale.detection.detector import expand_box

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(weights))
    crop_cfg = cfg.get("crop", {})
    dets: list[dict | None] = []
    for p in paths:
        r = model.predict(str(p), conf=args.conf, imgsz=args.imgsz,
                          device=device, verbose=False)
        if not (len(r) and len(r[0].boxes)):
            dets.append(None)
            continue
        b = r[0].boxes[0]
        im = Image.open(p)
        w, h = im.size
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
        crop = expand_box(x0, y0, x1, y1, w, h,
                          crop_cfg.get("pad_x", 0.30), crop_cfg.get("pad_up", 0.15),
                          crop_cfg.get("pad_down", 0.60))
        dets.append({"box": [x0, y0, x1, y1], "crop": crop, "conf": float(b.conf[0])})

    # ---------- 3. 渲染网格 ----------
    cols = min(args.cols, len(paths))
    rows = (len(paths) + cols - 1) // cols
    gap, margin = 16, 18
    header_h, legend_h = 104, 74
    grid_w = margin * 2 + cols * args.cell_w + (cols - 1) * gap
    grid_h = (header_h + margin + rows * args.cell_h + (rows - 1) * gap
              + margin + legend_h)
    canvas = Image.new("RGB", (grid_w, grid_h), C_BG)
    draw = ImageDraw.Draw(canvas)

    # 标题栏
    draw.rectangle([0, 0, grid_w, header_h], fill=C_NAVY)
    f_h1, f_h1e = font("zh", 30), font("en_bold", 30)
    f_h2, f_h2e = font("zh", 16), font("en", 15)
    f_leg, f_lege = font("zh", 16), font("en", 15)
    title = "中华白海豚背鳍检测与个体识别  ·  检测结果示例"
    _, asc1, _ = mixed_metrics(draw, title, f_h1, f_h1e)
    draw_mixed(draw, (margin + 6, 18 + asc1), title, f_h1, f_h1e, C_WHITE)
    sub = (f"检测模型：YOLOv8n-dorsalfin（自训练 Ultralytics YOLOv8-nano）    "
           f"conf≥{args.conf}  imgsz={args.imgsz}  device={device}    "
           f"特征模型：MegaDescriptor-T-224 + ArcFace 微调（r4）")
    _, asc2, _ = mixed_metrics(draw, sub, f_h2, f_h2e)
    draw_mixed(draw, (margin + 6, 62 + asc2), sub, f_h2, f_h2e, (190, 212, 235))

    for i, (p, det) in enumerate(zip(paths, dets)):
        try:
            batch_name = p.relative_to(data_root).parts[0]
        except ValueError:
            batch_name = p.parents[2].name if len(p.parts) >= 3 else p.parent.name
        cell = render_cell(p, det, args.cell_w, args.cell_h, args.footer_h,
                           model_tag, not args.no_crop_box,
                           f"{batch_name}  ·  {p.name}")
        r, c = divmod(i, cols)
        x = margin + c * (args.cell_w + gap)
        y = header_h + margin + r * (args.cell_h + gap)
        canvas.paste(cell, (x, y))
        draw.rectangle([x - 2, y - 2, x + args.cell_w + 1, y + args.cell_h + 1],
                       outline=(210, 216, 226), width=2)

    # 图例
    ly = grid_h - legend_h
    draw.rectangle([0, ly, grid_w, grid_h], fill=(233, 237, 243))
    draw.rectangle([0, ly, grid_w, ly + 2], fill=C_NAVY)
    lx, ty = margin + 6, ly + 14
    _, al, _ = mixed_metrics(draw, "背鳍", f_leg, f_lege)
    base = ty + al
    draw.rectangle([lx, base - 15, lx + 34, base - 1], fill=C_GREEN)
    draw_mixed(draw, (lx + 44, base), "背鳍检测框（标签 = 类别 + 置信度 confidence）",
               f_leg, f_lege, C_DARK)
    lx += 430
    draw.rectangle([lx, base - 15, lx + 34, base - 1], outline=C_CYAN, width=3)
    draw_mixed(draw, (lx + 44, base), "非均匀扩展裁剪区 → 特征提取输入（ReID）",
               f_leg, f_lege, C_DARK)
    lx += 430
    draw_mixed(draw, (lx, base), "识别结果 = Candidate，个体身份须人工核验",
               f_leg, f_lege, C_GREY)

    out_grid = out_dir / "detection_demo_grid.jpg"
    canvas.save(out_grid, quality=94)
    print(f"[demo] 网格图 → {out_grid}  ({grid_w}×{grid_h})")

    # 单图
    single_dir = out_dir / "singles"
    single_dir.mkdir(parents=True, exist_ok=True)
    for i, (p, det) in enumerate(zip(paths, dets), 1):
        img = render_single(p, det, 1280, model_tag)
        sp = single_dir / f"det_{i:02d}_{p.stem}.jpg"
        img.save(sp, quality=94)
    print(f"[demo] 单图 {len(paths)} 张 → {single_dir}")


if __name__ == "__main__":
    main()
