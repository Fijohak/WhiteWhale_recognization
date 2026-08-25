"""
散图连拍整串划归（正式入口，待办 1.9 准备阶段）：
散图池 → 按拍摄序列分串清单 + A9 抽样核验清单。

用法：
    python scripts/group_sequences.py \
        --manifest outputs/index/dataset_manifest.csv \
        --sample 10

输出（outputs/index/）：
- sequence_groups.csv          全部散图连拍串（n_frames≥2）
- sequence_sample_checklist.csv A9 抽样核验清单（逐帧文件名+原图路径）

⚠️ 仅生成清单，不执行划归：A9（连拍号=连续拍摄序列）抽样核验通过后，
清单才能用于整串划归（配合 assign_pool 的匹配结果）。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.data.sequence_groups import build_checklist_and_groups  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="散图连拍串分组清单（1.9 准备）")
    parser.add_argument("--manifest", type=Path,
                        default=Path("outputs/index/dataset_manifest.csv"),
                        help="数据清单 dataset_manifest.csv")
    parser.add_argument("--out", type=Path, default=Path("outputs/index"),
                        help="输出目录（sequence_groups.csv + 抽样清单）")
    parser.add_argument("--sample", type=int, default=10,
                        help="A9 抽样核验串数（0 = 不出抽样清单）")
    args = parser.parse_args()

    result = build_checklist_and_groups(args.manifest, args.out, args.sample)
    print(f"散图总数：{result['loose_images']} 张")
    print(f"连拍串（n≥2 帧）：{result['sequence_count']} 串")
    print(f"串清单：{result['sequences_csv']}")
    if result["sample_csv"]:
        print(f"A9 抽样核验清单：{result['sample_csv']}")
    print("注意: ", result["note"])


if __name__ == "__main__":
    main()
