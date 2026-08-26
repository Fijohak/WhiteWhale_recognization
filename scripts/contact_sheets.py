"""
拼图（contact sheet）生成（正式辅助工具）：Anchor 组版 + 候选簇版。

用法：
    python scripts/contact_sheets.py                       # 默认 Anchor 组
    python scripts/contact_sheets.py --cluster            # 按 HDBSCAN 候选簇分组

真实运行需要图片根（src_dataset）；--mock 生成占位色块验证布局逻辑。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.config import load_config  # noqa: E402
from whitewhale.review.contact_sheets import (  # noqa: E402
    build_cluster_contact_sheets, build_contact_sheets)


def main():
    base = REPO_ROOT / "outputs"
    cfg = load_config("pipeline")
    parser = argparse.ArgumentParser(description="生成候选照片拼图（人工审核辅助）")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv",
                        help="Anchor 组清单（默认模式）")
    parser.add_argument("--clusters", type=Path, default=base / "clusters" / "clusters.csv",
                        help="候选簇照片表（--cluster 模式）")
    parser.add_argument("--images-root", type=Path,
                        default=Path(cfg.get("data_root", "src_dataset")),
                        help="图片根目录（含 01/ 03/ 子目录）")
    parser.add_argument("--cluster", action="store_true",
                        help="按 HDBSCAN 候选簇分组（默认按 Anchor 组）")
    parser.add_argument("--out", type=Path, default=base / "contact_sheets")
    parser.add_argument("--mock", action="store_true", help="mock 模式（占位色块，不读图）")
    parser.add_argument("--max-sheets", type=int, default=200)
    args = parser.parse_args()

    if args.cluster:
        build_cluster_contact_sheets(args.clusters, args.out, args.images_root,
                                     mock=args.mock, max_sheets=args.max_sheets)
    else:
        build_contact_sheets(args.pilot, args.out, args.images_root,
                             mock=args.mock, max_sheets=args.max_sheets)


if __name__ == "__main__":
    main()
