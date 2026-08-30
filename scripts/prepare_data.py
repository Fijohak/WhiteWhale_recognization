"""
数据准备（正式入口 1/7）：扫描数据集 → 生成 Manifest → 生成 Pilot Set 清单。

用法：
    python scripts/prepare_data.py scan          # 扫描 src_dataset 下 9 个批次 → outputs/index/
    python scripts/prepare_data.py build-pilot   # Manifest → outputs/pilot/pilot_set.csv

数据语义：
- 高分目录数字子文件夹 = 批次内已确认个体（individual_id 含 session
  命名空间，不表示跨批次全局身份已对齐）；70-79 散图 = loose_known（暂不处理）。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.config import load_config  # noqa: E402
from whitewhale.data.manifest import build_pilot_set, scan_dataset  # noqa: E402


def _repo_path(path: Path) -> Path:
    """CLI/配置相对路径统一按仓库根解析。"""
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    cfg = load_config("pipeline")
    output_root = _repo_path(Path(cfg.get("output_root", "outputs")))
    default_roots = cfg.get("data_roots") or [Path(cfg.get("data_root", "src_dataset")) / d
                                              for d in ("01", "03")]
    default_roots = [_repo_path(Path(path)) for path in default_roots]
    parser = argparse.ArgumentParser(description="数据集扫描与 Pilot Set 生成")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="扫描数据根生成 Manifest")
    p_scan.add_argument("--data-roots", nargs="+", type=Path,
                        default=default_roots, help="数据根目录（默认读 configs/pipeline.yaml）")
    p_scan.add_argument("--output-dir", type=Path,
                        default=output_root / "index", help="索引输出目录")
    p_scan.add_argument("--sha256", action="store_true", help="计算 SHA-256 检测完全重复文件")

    p_pilot = sub.add_parser("build-pilot", help="从 Manifest 生成批次内已确认个体库")
    p_pilot.add_argument("--manifest", type=Path,
                         default=output_root / "index" / "dataset_manifest.csv")
    p_pilot.add_argument("--out", type=Path, default=output_root / "pilot")
    args = parser.parse_args()

    if args.cmd == "scan":
        args.data_roots = [_repo_path(path) for path in args.data_roots]
        args.output_dir = _repo_path(args.output_dir)
        scan_dataset(data_roots=args.data_roots, output_dir=args.output_dir,
                     include_sha256=args.sha256)
    elif args.cmd == "build-pilot":
        args.manifest = _repo_path(args.manifest)
        args.out = _repo_path(args.out)
        build_pilot_set(args.manifest, args.out)


if __name__ == "__main__":
    main()
