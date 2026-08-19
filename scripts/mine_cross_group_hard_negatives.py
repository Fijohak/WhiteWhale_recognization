"""
挖掘跨群 hard negative 照片对（供 r3 跨群 hard negative 微调用）。

语义（README §5.2 确认）：01 与 03 是两个独立海豚群，跨群对默认不同个体——
因此跨群高相似对是可靠的 hard negative（即使群内编号相同，如 01_5 vs 03_5）。

输入：pilot_set.csv（取 confirmed_identity 非空 = 伪标签训练照片）+ 特征文件（行序 = pilot 行序）
输出：outputs/metric_learning/r3/hard_negatives.csv（a, b, sess_a, sess_b, sim）

用法：
    python scripts/mine_cross_group_hard_negatives.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser = argparse.ArgumentParser(description="挖掘跨群 hard negative 照片对")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--feats", type=Path, default=base / "embeddings" / "embeddings_metric.npy")
    parser.add_argument("--out", type=Path, default=base / "metric_learning" / "r3" / "hard_negatives.csv")
    parser.add_argument("--min-sim", type=float, default=0.6, help="相似度阈值，高于此的跨群对视为 hard negative")
    args = parser.parse_args()

    p = pd.read_csv(args.pilot)
    d = p[p["confirmed_identity"].notna()
          & (p["confirmed_identity"].astype(str).str.strip() != "")].copy()
    d["sess"] = d["session_id"].astype(int)
    emb = np.load(args.feats)
    assert len(p) == emb.shape[0], "特征行数必须与 pilot 行数一致"

    # 特征按 pilot 行序；d 是 p 的子集，用 index 定位行号
    rows = []
    sims = emb @ emb.T
    idx = d.index.to_numpy()
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if d.loc[i, "sess"] == d.loc[j, "sess"]:
                continue
            s = float(sims[i, j])
            if s >= args.min_sim:
                rows.append({"a": d.loc[i, "image_id"], "sess_a": int(d.loc[i, "sess"]),
                             "b": d.loc[j, "image_id"], "sess_b": int(d.loc[j, "sess"]),
                             "sim": s})
    out = pd.DataFrame(rows).sort_values("sim", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[mine] 跨群对 sim>={args.min_sim}: {len(out)} 对")
    if len(out):
        print(f"[mine] 涉及照片 {out[['a','b']].to_numpy().ravel()!r}"[:200])
        print(f"[mine] sim p50={out['sim'].median():.3f} max={out['sim'].max():.3f}")
        print(f"[mine] 涉及个体（按群内编号）: {sorted(out[['a','b']].to_numpy().ravel().tolist())[:10]} ...")
    print(f"[mine] -> {args.out}")


if __name__ == "__main__":
    main()
