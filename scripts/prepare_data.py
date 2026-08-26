"""
数据准备（正式入口 1/7）：扫描数据集 → 生成 Manifest → 生成 Pilot Set 清单。

用法：
    python scripts/prepare_data.py scan          # 扫描 src_dataset 下 9 个批次 → outputs/index/
    python scripts/prepare_data.py build-pilot   # Manifest → outputs/pilot/pilot_set.csv

数据语义：
- 高分目录数字子文件夹 = Anchor 代表照片组（individual_id 仅作组标识，
  非已确认个体身份）；70-79 散图 = loose_known（暂不处理）。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.config import load_config  # noqa: E402
from whitewhale.data.manifest import build_pilot_set, scan_dataset  # noqa: E402


def main():
    cfg = load_config("pipeline")
    default_roots = cfg.get("data_roots") or [Path(cfg.get("data_root", "src_dataset")) / d
                                              for d in ("01", "03")]
    parser = argparse.ArgumentParser(description="数据集扫描与 Pilot Set 生成")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="扫描数据根生成 Manifest")
    p_scan.add_argument("--data-roots", nargs="+", type=Path,
                        default=default_roots, help="数据根目录（默认读 configs/pipeline.yaml）")
    p_scan.add_argument("--output-dir", type=Path,
                        default=Path("outputs/index"), help="索引输出目录")
    p_scan.add_argument("--sha256", action="store_true", help="计算 SHA-256 检测完全重复文件")

    p_pilot = sub.add_parser("build-pilot", help="从 Manifest 生成 Pilot Set（Anchor）")
    p_pilot.add_argument("--manifest", type=Path, default=Path("outputs/index/dataset_manifest.csv"))
    p_pilot.add_argument("--out", type=Path, default=Path("outputs/pilot"))
    args = parser.parse_args()

    if args.cmd == "scan":
        scan_dataset(data_roots=args.data_roots, output_dir=args.output_dir,
                     include_sha256=args.sha256)
    elif args.cmd == "build-pilot":
        build_pilot_set(args.manifest, args.out)


if __name__ == "__main__":
    main()
