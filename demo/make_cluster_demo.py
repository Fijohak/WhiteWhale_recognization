#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成「相似海豚聚类」申报 demo 网页（中心代表图 + 环绕成员图 + 连线相似度）。

数据源 = 批内归档管线的真实产物（clusters.csv + embeddings.npy + crops/），
代表图规则与管线一致：簇内与均值特征最接近的一帧。

产物：
    outputs/demo/cluster_demo.html     单文件网页（图片 base64 内嵌，可直接双击打开截图）

用法（conda activate torch 后，在仓库根目录执行）：
    python demo/make_cluster_demo.py
    python demo/make_cluster_demo.py --batch "20140419 02" --max-clusters 2 --max-members 12
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.config import load_config  # noqa: E402

PALETTE = {
    "high": "#1a7f37",    # ≥0.70
    "mid": "#2e86c1",     # ≥0.55
    "low": "#d98200",     # < 0.55
}


def b64(img: Image.Image, size: int) -> str:
    """PIL 图 → 高质量缩放 → base64 data URI。"""
    im = img.convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_picture(crops_dir: Path, image_id: str, data_root: Path,
                 rel_path: str) -> Image.Image | None:
    """优先背鳍裁剪图，缺失回退原图。"""
    for p in (crops_dir / f"{image_id}.jpg", data_root / str(rel_path)):
        if p.exists():
            return Image.open(p)
    return None


def ring_positions(n: int, cx: float, cy: float, rx: float, ry: float,
                   start: float = -90.0) -> list[tuple[float, float]]:
    """椭圆环均匀布点：按弧长均匀采样（等角度分布在椭圆两端会挤压重叠）。"""
    if n <= 0:
        return []
    th = np.linspace(0, 2 * np.pi, 4001)
    pts = np.stack([rx * np.cos(th), ry * np.sin(th)], axis=1)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = (np.arange(n) + 0.5) * s[-1] / n
    idx = np.searchsorted(s, targets)
    frac = (targets - s[idx - 1]) / np.maximum(s[idx] - s[idx - 1], 1e-9)
    angles = th[idx - 1] + frac * (th[idx] - th[idx - 1]) + np.deg2rad(start + 90.0)
    return [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in angles]


def build_cluster_block(cid: int, scid: int, members: list[dict], rep: dict,
                        status: str, top1: str, top1_score: float,
                        W: int, H: int) -> str:
    """生成一个簇的可视化块（SVG 连线 + 绝对定位图片）。"""
    cx, cy = W / 2, H / 2 + 8
    rep_size, mem_size = 300, 100
    others = [m for m in members if m["image_id"] != rep["image_id"]]

    # 布局：≤18 张单环（清晰，连线互不遮挡）；更多则双环
    if len(others) <= 18:
        rings = [(others, 438, 258, 0.0)]
    else:
        k = len(others)
        inner_n = (k + 1) // 2
        rings = [(others[:inner_n], 276, 172, 0.0),
                 (others[inner_n:], 438, 262, 360.0 / (2 * max(k - inner_n, 1)))]

    svg, cards = [], []
    for group, rx, ry, off in rings:
        pts = ring_positions(len(group), cx, cy, rx, ry, start=-90 + off)
        for m, (x, y) in zip(group, pts):
            s = m["sim"]
            color = PALETTE["high"] if s >= 0.70 else (
                PALETTE["mid"] if s >= 0.55 else PALETTE["low"])
            width = 1.6 + max(0.0, (s - 0.5)) * 5.0
            svg.append(
                f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" '
                f'stroke="{color}" stroke-width="{width:.1f}" stroke-linecap="round" '
                f'opacity="0.85"/>')
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
            mx, my = x - mem_size / 2, y - mem_size / 2
            cards.append(f"""
      <div class="node" style="left:{mx:.1f}px;top:{my:.1f}px;width:{mem_size}px">
        <img src="{m['src']}" alt="member">
        <div class="cap" style="color:{color}">sim {s:.2f}</div>
      </div>""")

    mean_sim = float(np.mean([m["sim"] for m in others])) if others else 1.0
    badge_cls = "ok" if status == "match" else "new"
    badge_txt = "匹配历史个体" if status == "match" else "疑似新个体"
    rep_card = f"""
      <div class="rep" style="left:{cx - rep_size / 2:.1f}px;top:{cy - rep_size / 2:.1f}px;
                              width:{rep_size}px;height:{rep_size}px">
        <img src="{rep['src']}" alt="representative">
        <div class="rep-tag">代表图 · Representative</div>
      </div>"""
    info = f"""
  <div class="cl-head">
    <span class="cl-title">Cluster {cid:03d}.{scid}　候选个体簇</span>
    <span class="badge {badge_cls}">{badge_txt}</span>
    <span class="cl-meta">成员 {len(members)} 张 · 平均相似度 {mean_sim:.3f}</span>
    <span class="cl-meta">Top-1 历史个体：{top1 or '—'}（score {top1_score:.3f}）</span>
  </div>"""
    svg_el = (f'<svg class="wires" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
              + "".join(svg) + "</svg>")
    return (f'{info}<div class="stage" style="width:{W}px;height:{H}px">'
            f'{svg_el}{"".join(cards)}{rep_card}</div>')


HTML_TMPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>中华白海豚个体识别 · 相似个体聚类结果</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
         background: #eef1f6; color: #1f2937; padding: 26px 0 40px; }}
  .wrap {{ width: {W}px; margin: 0 auto; }}
  header {{ background: linear-gradient(135deg, #1e3a5f 0%, #2c5f8d 100%);
            color: #fff; border-radius: 14px; padding: 22px 28px;
            box-shadow: 0 4px 18px rgba(30,58,95,.25); }}
  header h1 {{ font-size: 25px; font-weight: 700; letter-spacing: .5px; }}
  header .sub {{ margin-top: 8px; font-size: 14px; opacity: .92; line-height: 1.7; }}
  .pipe {{ margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }}
  .pipe span {{ background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
                padding: 5px 12px; border-radius: 16px; font-size: 12.5px; }}
  .pipe i {{ color: #ffd166; font-style: normal; margin: 0 2px; }}
  .card {{ background: #fff; border-radius: 14px; margin-top: 22px; overflow: hidden;
           box-shadow: 0 2px 12px rgba(0,0,0,.08); border: 1px solid #e3e8f0; }}
  .cl-head {{ padding: 14px 22px; background: #f5f8fc; border-bottom: 1px solid #e3e8f0;
              display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
  .cl-title {{ font-size: 17px; font-weight: 700; color: #1e3a5f; }}
  .badge {{ font-size: 12px; font-weight: 600; color: #fff; padding: 3px 12px; border-radius: 12px; }}
  .badge.ok {{ background: #1a7f37; }}
  .badge.new {{ background: #d9a400; }}
  .cl-meta {{ font-size: 12.5px; color: #5b6b7f; }}
  .stage {{ position: relative; margin: 0 auto; }}
  .wires {{ position: absolute; inset: 0; }}
  .node {{ position: absolute; text-align: center; }}
  .node img {{ width: 100%; border-radius: 10px; border: 3px solid #fff;
               box-shadow: 0 2px 10px rgba(0,0,0,.22); display: block;
               transition: transform .15s; }}
  .node img:hover {{ transform: scale(1.6); z-index: 20; position: relative; }}
  .cap {{ margin-top: 5px; font-size: 12px; font-weight: 700; }}
  .rep {{ position: absolute; }}
  .rep img {{ width: 100%; height: 100%; object-fit: cover; border-radius: 16px;
              border: 5px solid #1e3a5f; box-shadow: 0 8px 26px rgba(30,58,95,.35); }}
  .rep-tag {{ position: absolute; left: 50%; transform: translateX(-50%); bottom: -30px;
              background: #1e3a5f; color: #fff; font-size: 12px; font-weight: 600;
              padding: 4px 14px; border-radius: 12px; white-space: nowrap; }}
  .legend {{ display: flex; gap: 26px; align-items: center; justify-content: center;
             padding: 12px 0 20px; color: #5b6b7f; font-size: 13px; }}
  .legend i {{ display: inline-block; width: 30px; height: 4px; border-radius: 2px;
               margin-right: 7px; vertical-align: middle; }}
  footer {{ margin-top: 8px; padding: 14px 22px; background: #fff8e6;
            border: 1px solid #f0dca8; border-radius: 12px; color: #8a6d1f; font-size: 13px;
            line-height: 1.7; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>🐬 中华白海豚个体识别 · 相似个体聚类结果</h1>
    <div class="sub">
      批次 <b>{batch}</b>　|　批内候选聚类（HDBSCAN，min_cluster_size={min_size}）　|　
      特征：MegaDescriptor-T-224 + ArcFace 度量学习微调（r4）　|　检测：YOLOv8n-dorsalfin
    </div>
    <div class="pipe">
      <span>① 背鳍检测 <i>YOLOv8n-dorsalfin</i></span> →
      <span>② 非均匀扩展裁剪</span> →
      <span>③ 特征提取 <i>MegaDescriptor-r4</i></span> →
      <span>④ 批内聚类 <i>HDBSCAN</i></span> →
      <span>⑤ 代表图选取 + 人工审核</span>
    </div>
  </header>
  {blocks}
  <div class="legend">
    <span><i style="background:{c_high}"></i>相似度 ≥ 0.70（高置信同一体）</span>
    <span><i style="background:{c_mid}"></i>0.55 – 0.70</span>
    <span><i style="background:{c_low}"></i>&lt; 0.55（须人工复核）</span>
    <span>中心 = 簇代表图（与簇内均值特征最接近的一帧）</span>
  </div>
  <footer>
    说明：聚类结果一律为 <b>Candidate（候选）</b>，连线数值为成员图与代表图的余弦相似度；
    须经人工审核确认后方构成个体身份。代表图用于归档，连线仅反映特征空间相似度，不等于亲缘关系。
  </footer>
</div>
</body>
</html>
"""


def main() -> None:
    cfg = load_config("pipeline")
    ap = argparse.ArgumentParser(description="生成相似个体聚类 demo 网页")
    ap.add_argument("--batch", default="20140419 02", help="批次名（cross_time 下的子目录）")
    ap.add_argument("--archival-root", default="outputs/cluster_archival/cross_time")
    ap.add_argument("--data-root", default=cfg.get("data_root", "src_dataset/"))
    ap.add_argument("--crops-dir", default=None, help="裁剪图目录（默认 {archival}/{batch}/crops）")
    ap.add_argument("--max-clusters", type=int, default=2, help="展示簇数（按成员数降序）")
    ap.add_argument("--max-members", type=int, default=18, help="每簇最多展示成员数")
    ap.add_argument("--min-members", type=int, default=3)
    ap.add_argument("--width", type=int, default=1040)
    ap.add_argument("--height", type=int, default=700)
    ap.add_argument("--out", default="outputs/demo/cluster_demo.html")
    args = ap.parse_args()

    root = Path(args.archival_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    bdir = root / args.batch
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    crops_dir = Path(args.crops_dir) if args.crops_dir else bdir / "crops"

    cl = pd.read_csv(bdir / "clusters.csv")
    meta = pd.read_csv(bdir / "embeddings_meta.csv")
    emb = np.load(bdir / "embeddings.npy")
    assert len(meta) == len(emb), "embeddings 与 meta 行数不一致"
    pos = {iid: i for i, iid in enumerate(meta["image_id"])}

    # 纯子簇（cluster>=0 且 subcluster>=0）按成员数降序
    groups = []
    for (c, sc), g in cl[cl["cluster"] >= 0].groupby(["cluster", "subcluster"]):
        if sc < 0 or len(g) < args.min_members:
            continue
        groups.append((int(c), int(sc), g))
    groups.sort(key=lambda t: -len(t[2]))
    groups = groups[: args.max_clusters]
    if not groups:
        raise SystemExit("[demo] 没有满足条件的候选簇（试试降低 --min-members 或换批次）")

    blocks = []
    for c, sc, g in groups:
        idx = [pos[i] for i in g["image_id"] if i in pos]
        sub = emb[idx]
        mean_feat = sub.mean(axis=0)
        mean_feat /= (np.linalg.norm(mean_feat) + 1e-12)
        rep_local = int(np.argmax(sub @ mean_feat))
        rep_row = g.iloc[rep_local]
        rep_vec = sub[rep_local]
        sims = sub @ rep_vec

        rows = []
        for r, s in zip(g.itertuples(), sims):
            pic = load_picture(crops_dir, r.image_id, data_root, r.relative_path)
            if pic is None:
                continue
            rows.append({"image_id": r.image_id, "sim": float(s), "img": pic})
        if len(rows) < args.min_members:
            continue
        rows.sort(key=lambda m: -m["sim"])          # 与代表图相似度降序
        rows = rows[: args.max_members]
        rep_m = next((m for m in rows if m["image_id"] == rep_row["image_id"]), rows[0])
        for m in rows:                              # 代表图用更高分辨率
            m["src"] = b64(m.pop("img"), 620 if m is rep_m else 220)
        rows.remove(rep_m)
        members = [rep_m] + rows

        blocks.append(build_cluster_block(
            c, sc, members, rep_m, str(rep_row["status"]),
            str(rep_row["top1"]), float(rep_row["top1_score"]),
            args.width, args.height))
        print(f"[demo] Cluster {c:03d}.{sc}: {len(members)} 张，"
              f"代表图 {rep_row['image_id']}，status={rep_row['status']}")

    html = HTML_TMPL.format(
        W=args.width + 40, batch=args.batch, blocks="".join(
            f'<div class="card">{b}</div>' for b in blocks),
        min_size=cfg.get("clustering", {}).get("min_cluster_size", 3),
        c_high=PALETTE["high"], c_mid=PALETTE["mid"], c_low=PALETTE["low"])
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[demo] → {out}  ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
