"""
拼图（contact sheet）生成。
按聚类簇或个体分组输出图片拼图（每簇一张），便于人工核验候选簇。
真实运行需要 I 盘图片；--mock 模式生成占位色块验证布局逻辑。
"""
import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

GRID_W, GRID_H = 4, 3  # 每张拼图网格


def load_image(images_root: Path, rel_path: str):
    """真实模式下读取图片；mock 模式返回占位图。"""
    global MOCK
    if MOCK:
        return None
    return Image.open(images_root / rel_path).convert("RGB")


def render_placeholder(size, label):
    img = Image.new("RGB", size, (240, 245, 250))
    ImageDraw.Draw(img).text((10, size[1] // 2 - 10), label, fill=(120, 130, 140))
    return img


def render_sheet(imgs, titles, out_path: Path, cell_w=256, cell_h=256):
    n = len(imgs)
    cols = GRID_W
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    for i, (im, t) in enumerate(zip(imgs, titles)):
        r, c = divmod(i, cols)
        if im is None:
            im = render_placeholder((cell_w, cell_h), t)
        im.thumbnail((cell_w - 8, cell_h - 40))
        sheet.paste(im, (c * cell_w + 4, r * cell_h + 4))
        ImageDraw.Draw(sheet).text((c * cell_w + 4, (r + 1) * cell_h - 32),
                                   t, fill=(60, 70, 80))
    sheet.save(out_path)


def build_contact_sheets(clusters_path: Path, out_dir: Path,
                         images_root: Path, mock: bool = False,
                         min_size: int = 1, max_sheets: int = 200):
    global MOCK
    MOCK = mock
    df = pd.read_csv(clusters_path)
    groups = df[df["cluster"] >= 0].groupby("cluster")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_sheets = 0
    for cluster_id, sub in groups:
        if len(sub) < min_size:
            continue
        sub = sub.sort_values("cluster_probability", ascending=False)
        imgs = [load_image(images_root, p) for p in sub["relative_path"]]
        titles = [f"{r['image_id']} {r['individual_id']}" for _, r in sub.iterrows()]
        render_sheet(imgs, titles, out_dir / f"cluster_{cluster_id:03d}.jpg")
        n_sheets += 1
        if n_sheets >= max_sheets:
            break
    print(f"拼图: {n_sheets} 张 → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按聚类簇生成人工核验拼图")
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser.add_argument("--clusters", type=Path, default=base / "clusters" / "clusters.csv")
    parser.add_argument("--out", type=Path, default=base / "contact_sheets")
    parser.add_argument("--images-root", type=Path, default=Path("I:/"))
    parser.add_argument("--mock", action="store_true", help="占位图验证布局")
    parser.add_argument("--min-size", type=int, default=2, help="少于该规模的簇不输出拼图")
    parser.add_argument("--max-sheets", type=int, default=200)
    args = parser.parse_args()
    build_contact_sheets(args.clusters, args.out, args.images_root,
                         args.mock, args.min_size, args.max_sheets)
