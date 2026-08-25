"""
历史库核验回填（正式入口）：核验汇总表 → 可信基准 + pilot_set 回填。

用法：
    python scripts/finalize_history_verify.py \
        --summary outputs/review/history_verify_summary.csv \
        --pilot outputs/pilot/pilot_set.csv \
        --out outputs/review

判定规则与输出说明见 docs/history_verify_crossyear.md §一 1.5。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.data.history_verify import mark_verified  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="历史库核验结果回填（3.6 步骤 4）")
    parser.add_argument("--summary", type=Path, required=True,
                        help="核验汇总表 history_verify_summary.csv")
    parser.add_argument("--pilot", type=Path,
                        default=Path("outputs/pilot/pilot_set.csv"),
                        help="pilot_set.csv（将更新 review_status=verified，改前备份）")
    parser.add_argument("--out", type=Path, default=Path("outputs/review"),
                        help="可信基准表输出目录")
    parser.add_argument("--no-backup", action="store_true",
                        help="不备份 pilot_set.csv（默认备份）")
    args = parser.parse_args()

    result = mark_verified(args.summary, args.pilot, args.out,
                           backup=not args.no_backup)
    print(f"回填完成：{result['verified_groups']} 组 / "
          f"{result['verified_images']} 张照片标记为 verified")
    print(f"可信基准：{result['benchmark_csv']}")
    if result["pilot_backup"]:
        print(f"pilot_set 备份：{result['pilot_backup']}")


if __name__ == "__main__":
    main()
