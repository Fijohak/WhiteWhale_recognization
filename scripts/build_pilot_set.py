"""
Pilot Set 清单生成器。
从 dataset_manifest.csv 中筛选 labeled（43 个体 199 张）与 loose_known（207 张散图），
输出可直接用于 embedding / 检索 / 聚类的清单文件。
"""
import argparse
from pathlib import Path

import pandas as pd


def build_pilot_set(manifest_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(manifest_path)
    df["session_id"] = df["session_id"].astype(str)

    labeled = df[df["label_status"] == "labeled"].copy()
    loose = df[df["label_status"] == "loose_known"].copy()

    # 个体 ID 按 {session}_{group_id} 组合（跨调查同名编号未合并）
    labeled["individual_id"] = labeled["session_id"] + "_" + labeled["group_id"].astype(str)
    loose["individual_id"] = "loose_unknown"

    pilot = pd.concat([labeled, loose], ignore_index=True)
    pilot["split"] = "pilot"
    pilot.loc[pilot["label_status"] == "labeled", "split"] = "labeled"
    pilot.loc[pilot["label_status"] == "loose_known", "split"] = "loose_known"

    out_dir.mkdir(parents=True, exist_ok=True)
    pilot_path = out_dir / "pilot_set.csv"
    pilot.to_csv(pilot_path, index=False)

    # 汇总统计
    stats = {
        "total": len(pilot),
        "labeled_images": len(labeled),
        "loose_known_images": len(loose),
        "individual_count": labeled["individual_id"].nunique(),
        "single_image_individuals": int(
            (labeled.groupby("individual_id").size() == 1).sum()
        ),
        "multi_image_individuals": int(
            (labeled.groupby("individual_id").size() > 1).sum()
        ),
        "max_images_per_individual": int(labeled.groupby("individual_id").size().max()),
        "session_distribution": pilot["session_id"].value_counts().to_dict(),
        "quality_band_distribution": pilot["quality_band"].value_counts(dropna=False).to_dict(),
    }
    stats_path = out_dir / "pilot_set_stats.json"
    import json

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Pilot Set: {len(pilot)} 张")
    print(f"  labeled: {len(labeled)} 张 / {stats['individual_count']} 个体"
          f"（单图 {stats['single_image_individuals']}，多图 {stats['multi_image_individuals']}，"
          f"最多 {stats['max_images_per_individual']} 张）")
    print(f"  loose_known: {len(loose)} 张")
    print(f"  session: {stats['session_distribution']}")
    print(f"  输出: {pilot_path}")
    print(f"  统计: {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 Pilot Set 清单")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "index" / "dataset_manifest.csv")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "pilot")
    args = parser.parse_args()
    build_pilot_set(args.manifest, args.out)
