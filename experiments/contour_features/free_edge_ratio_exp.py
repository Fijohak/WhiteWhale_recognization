"""
4.3 分割改进快试：自由边比例对轮廓特征区分度的影响（E10 后续）。

E10 结论：Otsu 分割不稳定 → 同体轮廓一致性 ≈ 跨体水平，区分度不足。
本实验检查一个可快速验证的变量：自由边比例（free_edge cut_ratio）——
取更多的轮廓（45% → 55% → 65%）是否提供更多个体区分信息。

流程（历史库 199 张）：
- 每张图 Otsu 分割一次（分割瓶颈相同，不重复测），
  对 cut_ratio ∈ {0.45, 0.55, 0.65} 各提取 27 维轮廓特征；
- 同体/跨体相似度分布 + 检索 R@1/R@5（leave-one-out，与 E10 同口径）。

输出：outputs/reports/contour_features/free_edge_ratio.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contour_features import contour_feature, feature_sim, free_edge, \
    segment_fin  # noqa: E402
from experiment_contour import retrieval_eval  # noqa: E402

BASE = Path(__file__).resolve().parents[2]
CROPS = BASE / "outputs" / "crops_yolo"
REPORT = BASE / "outputs" / "reports" / "contour_features"
RATIOS = (0.45, 0.55, 0.65)


def main():
    meta = pd.read_csv(REPORT / "features_meta.csv")
    ids = [str(iid) for iid in meta["image_id"]]
    print(f"[data] {len(ids)} 张 / {meta['confirmed_identity'].nunique()} 个体")

    # Otsu 分割只做一次，free_edge 取不同比例
    masks = {}
    for iid in ids:
        img = np.asarray(Image.open(CROPS / f"{iid}.jpg").convert("RGB"))
        masks[iid] = segment_fin(img)
    print("[seg] Otsu 分割完成（每张一次）")

    labels = np.asarray([str(x) for x in meta["confirmed_identity"]])
    same_idx, diff_idx = [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            (same_idx if labels[i] == labels[j] else diff_idx).append((i, j))
    same_idx = np.asarray(same_idx)
    diff_idx = np.asarray(diff_idx)

    results: dict = {"_meta": {
        "data": "历史库 199 张 r3 yolo 裁剪（与 E10 同口径）",
        "question": "自由边比例 45%→65% 能否改善轮廓特征区分度",
        "sim": "cosine", "query": "多图个体 leave-one-out"}}

    for ratio in RATIOS:
        F = np.stack([contour_feature(free_edge(masks[iid], cut_ratio=ratio))
                      for iid in ids])
        ok = ~np.isnan(F).any(axis=1)
        if not ok.all():
            print(f"[ratio={ratio}] 特征提取失败 {int((~ok).sum())} 张")
        F = F[ok]
        lab = labels[ok]
        n = len(F)
        sidx = [(i, j) for i in range(n) for j in range(i + 1, n)
                if lab[i] == lab[j]]
        didx = [(i, j) for i in range(n) for j in range(i + 1, n)
                if lab[i] != lab[j]]
        same = [feature_sim(F[i], F[j]) for i, j in sidx]
        diff = [feature_sim(F[i], F[j]) for i, j in didx]
        same, diff = np.asarray(same), np.asarray(diff)
        best, bt = 0.0, 0.0
        for t in np.linspace(0.30, 0.95, 131):
            acc = ((same >= t).mean() + (diff < t).mean()) / 2
            if acc > best:
                best, bt = float(acc), float(t)
        r = retrieval_eval(F, meta[ok], [ids[k] for k in np.where(ok)[0]])
        row = {"同体_p50": float(np.median(same)),
               "跨体_p50": float(np.median(diff)),
               "区分_acc": best, "区分_threshold": bt,
               "R@1": r["R@1"], "R@5": r["R@5"], "n_query": r["n_query"]}
        results[f"cut_ratio_{ratio}"] = row
        print(f"[ratio={ratio}] 同体 p50={row['同体_p50']:.3f} | "
              f"跨体 p50={row['跨体_p50']:.3f} | acc={best:.3f} | "
              f"R@1={r['R@1']:.3f}")

    (REPORT / "free_edge_ratio.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] -> {REPORT / 'free_edge_ratio.json'}")


if __name__ == "__main__":
    main()
