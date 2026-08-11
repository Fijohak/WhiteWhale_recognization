"""
HDBSCAN 候选聚类。
对 embedding 做 HDBSCAN 聚类（允许噪声点），聚类结果与 labeled 个体对照，
用于判断：聚类是否按个体分开，还是按拍摄批次/背景分开（基线诊断输入之一）。
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

    # 集群统计
    cluster_stats = {}
    for c in sorted(set(labels)):
        sub = df[df["cluster"] == c]
        cluster_stats[int(c)] = {
            "size": len(sub),
            "labeled_individuals": sorted(sub[sub["individual_id"] != "loose_unknown"]["individual_id"].unique().tolist()),
            "sessions": sorted(sub["session_id"].unique().tolist()),
            "noise_ratio": None,
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
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "clusters.csv", index=False)
    with open(out_dir / "cluster_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"HDBSCAN: {len(df)} 张 → {stats['n_clusters']} 簇 + 噪声 {stats['n_noise']} 张"
          f"（{stats['noise_ratio']:.1%}）")
    print(f"  → {out_dir / 'clusters.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDBSCAN 候选聚类")
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser.add_argument("--embeddings", type=Path, default=base / "embeddings" / "embeddings.npy")
    parser.add_argument("--meta", type=Path, default=base / "embeddings" / "embeddings_meta.csv")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv")
    parser.add_argument("--out", type=Path, default=base / "clusters")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    args = parser.parse_args()
    hdbscan_cluster(args.embeddings, args.meta, args.pilot, args.out, args.min_cluster_size)
