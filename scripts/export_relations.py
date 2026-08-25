"""
确认关系表导出（正式入口，待办 3.3）：
确认个体表 → confirmed_same 对 + possibly_same / confirmed_different 空表。

用法：
    python scripts/export_relations.py \
        --confirmed outputs/review/confirmed_individuals.csv \
        --out outputs/review

输出（outputs/review/）：
- relations_confirmed_same.csv       确认同体对（有数据）
- relations_confirmed_different.csv  确认异体对（无数据源，仅表头）
- relations_possibly_same.csv        疑似同体对（待 3.8，仅表头）
- relations_note.json                三张表来源与说明
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.data.relations import build_relations  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="确认关系表导出（3.3）")
    parser.add_argument("--confirmed", type=Path,
                        default=Path("outputs/review/confirmed_individuals.csv"),
                        help="人工确认个体表")
    parser.add_argument("--out", type=Path, default=Path("outputs/review"),
                        help="输出目录")
    args = parser.parse_args()

    paths = build_relations(args.confirmed, args.out)
    print("关系表导出完成：")
    for key in ("confirmed_same", "confirmed_different", "possibly_same"):
        print(f"  {key}: {paths[key]}")
    print(f"  说明: {paths['note_json']}")


if __name__ == "__main__":
    main()
