"""
HDBSCAN 候选聚类（辅助工具，方向调整后降为辅助）。

数据语义（2026-08-11 确认）：目录不直接等于个体 ID。本脚本仅对 embedding 做
HDBSCAN 聚类（允许噪声点），输出 **Candidate Cluster**（候选分组，仅供人工审核
参考），**不得直接当作真实个体**。聚类结果与 Anchor 组标识对照，用于诊断特征
是否按拍摄批次/背景分组（基线诊断输入之一）。

注意：输入 pilot_set.csv 现仅含高分 Anchor 照片（199 张），`individual_id` 为
Anchor 组标识（非已确认个体身份）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 分析性对照（图片与几何信息，始终可用）
import PIL.Image


def hdbscan_cluster(embeddings_path: Path, meta_path: Path, pilot_path: Path,
                    out_dir: Path, min_cluster_size: int = 3):
    try:
        import hdbscan
    except ImportError as e:
        raise SystemExit(f"缺少 hdbscan 依赖: {e}") from e

    emb = np.load(embeddings_path)
    meta = pd.read_csv(meta_path)
    pilot = pd.read_csv(pilot_path)
    assert len(emb) == len(meta), "embedding 与 meta 数量不一致"
    df = meta.merge(pilot, on="image_id", how="left", suffixes=("", "_pilot"))
    df["session_id"] = df["session_id"].astype(str)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(emb)
    df["cluster"] = labels
    df["cluster_probability"] = clusterer.probabilities_

    # 簇统计：簇内涉及的 Anchor 组标识与 session（仅对照用，不代表个体身份）
    cluster_stats = {}
    for c in sorted(set(labels)):
        sub = df[df["cluster"] == c]
        cluster_stats[int(c)] = {
            "size": len(sub),
            "anchor_groups": sorted(sub["individual_id"].unique().tolist()),
            "sessions": sorted(sub["session_id"].unique().tolist()),
            "note": "Candidate Cluster：仅供人工审核参考，不是真实个体。",
        }
    # 噪声比例（-1 为噪声）
    noise = df[df["cluster"] == -1]
    stats = {
        "n_images": len(df),
        "n_clusters": len(set(labels)) - (1 if -1 in set(labels) else 0),
        "n_noise": len(noise),
        "noise_ratio": float(len(noise) / len(df)),
        "min_cluster_size": min_cluster_size,
        "cluster_stats": cluster_stats,
        "note": "HDBSCAN 为辅助工具，结果只能叫 Candidate Cluster。",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "clusters.csv", index=False)
    with open(out_dir / "cluster_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"HDBSCAN（辅助）: {len(df)} 张 → {stats['n_clusters']} 候选簇 + 噪声 {stats['n_noise']} 张"
          f"（{stats['noise_ratio']:.1%}）")
    print(f"  → {out_dir / 'clusters.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDBSCAN 候选聚类（辅助，输出仅供人工审核）")
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser.add_argument("--embeddings", type=Path, default=base / "embeddings" / "embeddings.npy")
    parser.add_argument("--meta", type=Path, default=base / "embeddings" / "embeddings_meta.csv")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--out", type=Path, default=base / "clusters")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    args = parser.parse_args()
    hdbscan_cluster(args.embeddings, args.meta, args.pilot, args.out, args.min_cluster_size)
