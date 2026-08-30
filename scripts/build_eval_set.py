"""
人工评估集划分草案（正式入口，待办 3.2）：
确认个体表 → 按完整连拍串自动划分 query/gallery 草案。

用法：
    python scripts/build_eval_set.py \
        --confirmed outputs/review/confirmed_individuals.csv \
        --manifest outputs/index/dataset_manifest.csv \
        --out outputs/index

输出（outputs/index/）：
- eval_set_draft.csv         逐图 split 草案（人工确认后作评估集）
- eval_set_draft_stats.json  覆盖统计

⚠️ 草案须人工确认；A9（连拍号=序列）核验通过前不作正式评估集。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.data.eval_set import build_draft  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="评估集划分草案（3.2）")
    parser.add_argument("--confirmed", type=Path,
                        default=Path("outputs/review/confirmed_individuals.csv"),
                        help="人工确认个体表（status=confirmed 行）")
    parser.add_argument("--manifest", type=Path,
                        default=Path("outputs/index/dataset_manifest.csv"),
                        help="数据清单（由 filename + session_id 计算完整 series）")
    parser.add_argument("--out", type=Path, default=Path("outputs/index"),
                        help="输出目录")
    args = parser.parse_args()

    result = build_draft(args.confirmed, args.manifest, args.out)
    print(f"评估集草案：{result['n_images']} 张 / {result['n_individuals']} 个体")
    print(f"  query {result['n_query']} 张（{result['n_individuals_with_query']}"
          f" 个体）/ gallery {result['n_gallery']} 张")
    print(f"  跨日期个体 {result['multi_session_individuals']} 个"
          f"（当前确认数据均为批内，跨日期待 3.6/3.8）")
    print(f"  草案：{result['draft_csv']}")
    print(f"  统计：{result['stats_json']}")
    print("注意: ", result["note"])


if __name__ == "__main__":
    main()
