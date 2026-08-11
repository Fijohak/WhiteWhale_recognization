"""
Pilot Set 清单生成器（Anchor-based，方向调整后）。

数据语义（2026-08-11 确认）：高分目录数字子文件夹 = 历史挑选的代表照片（Anchor），
**不代表个体 ID**。`70-79` 散图暂不处理（用户明确：现阶段不处理散图）。

输出：
- pilot_set.csv      仅含高分 Anchor 照片（labeled，199 张），relative_path 含会话根前缀（01/、03/）
- pilot_set_stats.json  Anchor 组统计

字段语义：individual_id = {session}_{group_id}，仅作 Anchor 组标识（候选起点），
**不是已确认个体身份**。后续人工审核后另立 confirmed_identity。
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def build_pilot_set(manifest_path: Path, out_dir: Path) -> None:
    # 以字符串读入 session_id，避免 "01" 被 pandas 解析为整数 1（路径会变成 1/...）
    df = pd.read_csv(manifest_path, dtype={"session_id": str})
    df["session_id"] = df["session_id"].str.zfill(2)

    # 仅取已分组照片（高分目录子文件夹内的照片 = Anchor）
    anchors = df[df["label_status"] == "labeled"].copy()
    # 散图（loose_known）暂不纳入 Pilot（用户决定现阶段不处理散图）

    # 含根前缀的完整相对路径（相对数据根 I:\，含 01/ 03/），供下游直接拼根读取
    anchors["relative_path"] = anchors["session_id"] + "/" + anchors["relative_path"]

    # Anchor 组标识 = {session}_{group_id}（跨调查同名编号未合并，非全局 ID）
    anchors["individual_id"] = anchors["session_id"] + "_" + anchors["group_id"].astype(str)
    anchors["split"] = "labeled"

    out_dir.mkdir(parents=True, exist_ok=True)
    pilot_path = out_dir / "pilot_set.csv"
    anchors.to_csv(pilot_path, index=False, encoding="utf-8-sig")

    # 汇总统计
    stats = {
        "total": len(anchors),
        "n_anchor_groups": anchors["individual_id"].nunique(),
        "single_image_groups": int((anchors.groupby("individual_id").size() == 1).sum()),
        "multi_image_groups": int((anchors.groupby("individual_id").size() > 1).sum()),
        "max_images_per_group": int(anchors.groupby("individual_id").size().max()),
        "session_distribution": anchors["session_id"].value_counts().to_dict(),
        "quality_band_distribution": anchors["quality_band"].value_counts(dropna=False).to_dict(),
        "note": "仅含高分目录子文件夹照片（Anchor）。散图（loose_known）暂不纳入。individual_id 是 Anchor 组标识，非已确认个体 ID。",
    }
    stats_path = out_dir / "pilot_set_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Pilot Set（Anchor）: {len(anchors)} 张 / {stats['n_anchor_groups']} 组")
    print(f"  单图组 {stats['single_image_groups']}，多图组 {stats['multi_image_groups']}，最多 {stats['max_images_per_group']} 张")
    print(f"  session: {stats['session_distribution']}")
    print(f"  输出: {pilot_path}")
    print(f"  统计: {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 Pilot Set（Anchor-only）清单")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "index" / "dataset_manifest.csv")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "pilot")
    args = parser.parse_args()
    build_pilot_set(args.manifest, args.out)
