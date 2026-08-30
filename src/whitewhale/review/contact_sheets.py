"""
拼图（contact sheet）生成（已确认个体组版 + 候选簇版）。

- build_contact_sheets：按批次内已确认个体（高分目录数字子文件夹）输出拼图；
- build_cluster_contact_sheets：按 HDBSCAN 候选簇（Candidate Cluster）分组
  输出，供人工审核逐簇核对（-1 噪声单独一张，仅提醒不强制分配）。

真实运行需要图片根（src_dataset）；mock 模式生成占位色块验证布局逻辑。
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from whitewhale.data.image_store import ImageStore

GRID_W, GRID_H = 4, 3  # 每张拼图网格
MOCK = False


def load_image(images_root: Path, rel_path: str):
    """真实模式下读取图片；mock 模式返回占位图。"""
    if MOCK:
        return None
    return ImageStore(images_root).open(rel_path)


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
    groups = df.groupby("individual_id")  # 含 session 命名空间的批次内已确认个体 ID

    out_dir.mkdir(parents=True, exist_ok=True)
    n_sheets = 0
    for gid, sub in groups:
        sub = sub.sort_values("sequence_guess", na_position="last")
        imgs = [load_image(images_root, p) for p in sub["relative_path"]]
        titles = [
            f"{r['session_id']}/{r.get('group_id', '')} {Path(r['relative_path']).name}"
            for _, r in sub.iterrows()
        ]
        render_sheet(imgs, titles, out_dir / f"anchor_{gid.replace('/', '_')}.jpg")
        n_sheets += 1
        if n_sheets >= max_sheets:
            break
    print(f"拼图（已确认个体组）: {n_sheets} 张 → {out_dir}")
    print("注：individual_id 仅在批次内确认；跨批次同名编号不自动视为同一只")


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
        if "sequence_guess" in df.columns:  # 散图池等清单无此列，跳过排序
            sub = sub.sort_values("sequence_guess", na_position="last")
        label = "noise" if cluster_id == -1 else f"cluster_{cluster_id:03d}"
        imgs = [load_image(images_root, p) for p in sub["relative_path"]]
        titles = [
            f"{r['session_id']}/{r.get('group_id', '')} {Path(r['relative_path']).name}"
            for _, r in sub.iterrows()
        ]
        render_sheet(imgs, titles, out_dir / f"{label}.jpg")
        n_sheets += 1
        if n_sheets >= max_sheets:
            break
    print(f"拼图（候选簇）: {n_sheets} 张 → {out_dir}")
    print("注：cluster 是 Candidate Cluster（-1=噪声），人工确认后才能叫个体")
