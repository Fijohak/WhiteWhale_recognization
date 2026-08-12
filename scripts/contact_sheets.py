"""
拼图（contact sheet）生成（Anchor 版 + 候选簇版）。

- 默认按 Anchor 组（高分目录数字子文件夹 = 代表照片组，非个体 ID）输出拼图；
- --cluster 模式按 HDBSCAN 候选簇（Candidate Cluster）分组输出，供人工审核
  逐簇核对是否为同一只个体（-1 噪声单独一张，仅提醒不强制分配）。

真实运行需要 I 盘图片；--mock 模式生成占位色块验证布局逻辑。
"""
import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

GRID_W, GRID_H = 4, 3  # 每张拼图网格
MOCK = False


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


def build_contact_sheets(pilot_path: Path, out_dir: Path,
                         images_root: Path, mock: bool = False,
                         max_sheets: int = 200):
    global MOCK
    MOCK = mock
    df = pd.read_csv(pilot_path)
    df["session_id"] = df["session_id"].astype(str)
    groups = df.groupby("individual_id")  # Anchor 组标识（非个体 ID）

    out_dir.mkdir(parents=True, exist_ok=True)
    n_sheets = 0
    for gid, sub in groups:
        sub = sub.sort_values("sequence_guess", na_position="last")
        imgs = [load_image(images_root, p) for p in sub["relative_path"]]
        titles = [f"{r['session_id']}/{r['group_id']} {r['filename']}" for _, r in sub.iterrows()]
        render_sheet(imgs, titles, out_dir / f"anchor_{gid.replace('/', '_')}.jpg")
        n_sheets += 1
        if n_sheets >= max_sheets:
            break
    print(f"拼图（Anchor 组）: {n_sheets} 张 → {out_dir}")
    print("注：Anchor 组是代表照片候选分组，非已确认个体身份")


def build_cluster_contact_sheets(clusters_csv: Path, out_dir: Path,
                                 images_root: Path, mock: bool = False,
                                 max_sheets: int = 200):
    """按 HDBSCAN 候选簇分组生成拼图（人工逐簇审核用）。

    - cluster >= 0：Candidate Cluster（候选个体分组，需人工确认）；
    - cluster = -1：噪声，单张拼图提示，不参与合并。
    """
    global MOCK
    MOCK = mock
    df = pd.read_csv(clusters_csv)
    df["session_id"] = df["session_id"].astype(str)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_sheets = 0
    for cluster_id, sub in df.groupby("cluster"):
        sub = sub.sort_values("sequence_guess", na_position="last")
        label = "noise" if cluster_id == -1 else f"cluster_{cluster_id:03d}"
        imgs = [load_image(images_root, p) for p in sub["relative_path"]]
        titles = [
            f"{r['session_id']}/{r.get('group_id', '')} {r['filename']}"
            for _, r in sub.iterrows()
        ]
        render_sheet(imgs, titles, out_dir / f"{label}.jpg")
        n_sheets += 1
        if n_sheets >= max_sheets:
            break
    print(f"拼图（候选簇）: {n_sheets} 张 → {out_dir}")
    print("注：cluster 是 Candidate Cluster（-1=噪声），人工确认后才能叫个体")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成人工核验拼图（Anchor 组 / 候选簇）")
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--clusters", type=Path, default=base / "clusters" / "clusters.csv")
    parser.add_argument("--out", type=Path, default=base / "contact_sheets")
    parser.add_argument("--images-root", type=Path, default=Path("I:/"))
    parser.add_argument("--cluster", action="store_true",
                        help="按 HDBSCAN 候选簇分组（默认按 Anchor 组）")
    parser.add_argument("--mock", action="store_true", help="占位图验证布局")
    parser.add_argument("--max-sheets", type=int, default=200)
    args = parser.parse_args()
    if args.cluster:
        build_cluster_contact_sheets(args.clusters, args.out, args.images_root,
                                     args.mock, args.max_sheets)
    else:
        build_contact_sheets(args.pilot, args.out, args.images_root,
                             args.mock, args.max_sheets)
