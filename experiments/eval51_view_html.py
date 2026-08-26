"""
E5.1 新批次划分与命中对照展示页（给人工核对用）。

对每个新批次 query 个体：列出每张 query 图与它命中的 Top-1 库图，
标注分数与是否"同串连拍"（平凡命中）。用浏览器打开生成的 HTML 查看原图。
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.eval51_common import (  # noqa: E402
    load_data, same_series, score_img_to_individual, split_probe_gallery)

IMAGES_ROOT = Path("I:/")


def img_uri(row) -> str:
    """原图 file:// URI（meta.relative_path 不含 session 前缀时补上）。"""
    p = IMAGES_ROOT / str(row["relative_path"])
    if not p.exists():
        p = IMAGES_ROOT / f"{row['session_id']}/{row['relative_path']}"
    return p.as_uri() if p.exists() else ""


def frame_label(row) -> str:
    f = int(row["frame"])
    return f"{f:04d}" if f != 999999 else "无连拍号"


def main():
    base = REPO_ROOT
    meta, emb = load_data(base)
    q_rows, gal_idx, gal_ind = split_probe_gallery(meta, emb)

    # 只展示新批次（历史库另有用途，本次核对聚焦新批次划分）
    newb_sessions = [s for s in meta["session_id"].unique()
                     if s not in ("20140806 01", "20140806 03")]
    blocks = []
    stats = {"rows": 0, "hit": 0, "hit_same_series": 0}
    for target in sorted(q_rows):
        if meta.loc[q_rows[target][0], "session_id"] not in newb_sessions:
            continue
        q_idx = q_rows[target]
        rows_html = []
        for qi in q_idx:
            per_img = score_img_to_individual(emb[qi], emb, gal_idx, gal_ind)
            top_ind = max(per_img, key=per_img.get)
            sims = emb[qi] @ emb[gal_idx].T
            top_pos = int(np.argmax(sims))
            top_row = meta.loc[gal_idx[top_pos]]
            same = same_series(int(meta.loc[qi, "frame"]), int(top_row["frame"]))
            hit = top_ind == target
            stats["rows"] += 1
            if hit:
                stats["hit"] += 1
                if same:
                    stats["hit_same_series"] += 1
            color = "#c62828" if same else ("#2e7d32" if hit else "#555")
            tag = "同串平凡命中" if same else ("命中" if hit else "未命中")
            rows_html.append(f"""
            <div class="row">
              <div class="imgbox"><img src="{img_uri(meta.loc[qi])}">
                <div class="cap">QUERY {meta.loc[qi, 'filename']}<br>连拍号 {frame_label(meta.loc[qi])}</div></div>
              <div class="arrow">→</div>
              <div class="imgbox"><img src="{img_uri(top_row)}">
                <div class="cap">TOP1 {top_row['filename']}<br>连拍号 {frame_label(top_row)}</div></div>
              <div class="info" style="color:{color}">
                <b>{tag}</b><br>Top1 个体: {top_ind or '无标签'}<br>分数: {per_img[top_ind]:.3f}<br>
                同体分数: {per_img.get(target, float('nan')):.3f}
              </div>
            </div>""")
        blocks.append(f"""
        <section><h3>{target}（{len(q_idx)} 张 query）</h3>{''.join(rows_html)}</section>""")

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>E5.1 新批次划分与命中对照</title>
<style>
  body {{ font-family: "Microsoft YaHei", sans-serif; margin: 24px; background: #f5f5f5; }}
  h3 {{ background: #1E3A5F; color: #fff; padding: 8px 12px; border-radius: 4px; }}
  .row {{ display: flex; align-items: center; gap: 12px; background: #fff;
         margin: 8px 0; padding: 10px; border-radius: 6px; }}
  .imgbox img {{ height: 140px; max-width: 260px; object-fit: contain; border: 1px solid #ccc; background: #000; }}
  .cap {{ font-size: 11px; color: #666; max-width: 260px; }}
  .arrow {{ font-size: 22px; color: #999; }}
  .info {{ font-size: 13px; min-width: 220px; }}
  .summary {{ background: #fff8e1; border-left: 4px solid #E8B339; padding: 10px 14px; margin-bottom: 16px; }}
</style></head><body>
<h2>E5.1 新批次 query 划分与 Top-1 命中对照（共 {len(blocks)} 个个体）</h2>
<div class="summary">统计: {stats['rows']} 张 query 图, 命中 {stats['hit']} 张（{stats['hit']/max(stats['rows'],1):.1%}）,
其中<b style="color:#c62828">同串连拍平凡命中 {stats['hit_same_series']} 张</b>。
红 = 同串连拍（相邻帧，相似度天然高，不计入能力证据）；绿 = 非同串命中；灰 = 未命中。</div>
{''.join(blocks)}
</body></html>"""
    out = REPO_ROOT / "outputs" / "reports" / "cluster_retrieval_v2" / "eval51_new_batches_view.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[view] 生成 {out}")
    print(f"[view] 统计: query {stats['rows']} 张 | 命中 {stats['hit']} | "
          f"其中同串平凡命中 {stats['hit_same_series']}（{stats['hit_same_series']/max(stats['hit'],1):.1%} 的命中）")


if __name__ == "__main__":
    main()
