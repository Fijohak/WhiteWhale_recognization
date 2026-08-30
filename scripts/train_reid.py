"""
度量学习训练（正式入口 5/7）：确认个体标签 ArcFace + 可选批内 hard negative。

r1/r2（纯 CE）与批内 hard-negative（CE + λ×triplet）共用同一入口，差异由
--hard-negative 控制。训练必须显式指定新的 --out，避免覆盖生产权重：

    python scripts/train_reid.py --out outputs/metric_learning/r6_candidate
    python scripts/train_reid.py --out outputs/metric_learning/ce_candidate --no-hard-negative

两阶段训练：冻结 backbone 训 head → 解冻低学习率微调（HN 链路只使用
同一 session 内已确认的不同个体作负样本）。配置默认值读 configs/reid.yaml。

语义：confirmed_identity 只接收已确认个体标签；批次内 individual_id 已确认，
但现有多图样本时间间隔短，指标不能外推为跨年能力。不使用水平翻转
（左右侧背鳍特征可能不同）；划分按个体（同一体不跨 train/val）。
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.config import load_config  # noqa: E402
from whitewhale.reid.training import extract_features, load_confirmed, run_training  # noqa: E402


def guard_training_output(out_dir: Path, overwrite: bool) -> None:
    """默认拒绝覆盖已有训练产物，必须显式 --overwrite 才允许复跑同目录。"""
    protected = [out_dir / name for name in (
        "best.pt", "best_stage1.pt", "history.csv", "metrics.json")]
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        raise SystemExit(
            f"FATAL: 输出目录已有训练产物，拒绝覆盖: {existing[0]}。"
            "请换一个 --out；确认需要复跑同版本时显式加 --overwrite。")


def guard_embedding_output(out_path: Path, overwrite: bool) -> None:
    """训练后提取也必须成组三文件防覆盖，避免新旧特征与 meta/config 混用。"""
    protected = [
        out_path,
        out_path.with_name(f"{out_path.stem}_meta.csv"),
        out_path.with_name(f"{out_path.stem}_config.json"),
    ]
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        raise SystemExit(
            f"FATAL: 特征输出已有产物，拒绝覆盖: {existing[0]}。"
            "请换 --embeddings-out，或确认后显式加 --overwrite。")


def _repo_path(path: Path) -> Path:
    """CLI 相对路径统一按仓库根解析，不依赖启动时工作目录。"""
    return path if path.is_absolute() else REPO_ROOT / path


def _promote_training_outputs(staging: Path, destination: Path) -> None:
    """训练全部成功后再把暂存产物逐个原子替换到版本目录。"""
    required = ["best.pt", "history.csv", "metrics.json"]
    missing = [name for name in required if not (staging / name).exists()]
    if missing:
        raise RuntimeError(f"训练暂存目录缺少核心产物: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    for source in staging.iterdir():
        if source.is_file():
            os.replace(source, destination / source.name)


def main():
    pipeline_cfg = load_config("pipeline")
    output_root = Path(pipeline_cfg.get("output_root", "outputs"))
    base = _repo_path(output_root)
    cfg = load_config("reid")
    parser = argparse.ArgumentParser(description="度量学习训练（确认个体标签 ArcFace + 可选批内 HN）")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv",
                        help="已确认个体清单（含非空 confirmed_identity）")
    parser.add_argument("--images-root", type=Path,
                        default=_repo_path(Path(pipeline_cfg.get("data_root", "src_dataset"))),
                        help="图片根目录（只读）")
    parser.add_argument("--out", type=Path, required=True,
                        help="新训练版本输出目录（必须显式指定，避免覆盖生产权重）")
    parser.add_argument("--overwrite", action="store_true",
                        help="允许覆盖 --out 中已有训练产物（高风险，默认拒绝）")
    parser.add_argument("--val-n", type=int, default=cfg.get("val_n", 6),
                        help="验证个体数（按个体留出）")
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    configured_test_sessions = cfg.get("test_sessions", [])
    if isinstance(configured_test_sessions, str):
        configured_test_sessions = [configured_test_sessions]
    parser.add_argument(
        "--test-session", action="append",
        default=list(configured_test_sessions),
        help="完整隔离为独立测试集的 session；可重复指定，默认不启用",
    )
    parser.add_argument("--epochs-stage1", type=int, default=cfg.get("epochs_stage1", 20),
                        help="阶段一：冻结 backbone 训 head")
    parser.add_argument("--epochs-stage2", type=int, default=cfg.get("epochs_stage2", 25),
                        help="阶段二：解冻微调")
    parser.add_argument("--lr-head", type=float, default=cfg.get("lr_head", 0.001))
    parser.add_argument("--lr-backbone", type=float, default=cfg.get("lr_backbone", 0.000005))
    parser.add_argument("--batch", type=int, default=cfg.get("batch", 16))
    # ---- 批内 hard negative ----
    parser.add_argument("--hard-negative", dest="hard_negative", action="store_true",
                        default=cfg.get("hard_negative", True),
                        help="同 session 已确认负样本的 batch-hard triplet（默认开）")
    parser.add_argument("--no-hard-negative", dest="hard_negative", action="store_false",
                        help="纯 ArcFace CE（r1/r2 链路）")
    parser.add_argument("--init-ckpt", type=Path,
                        default=None,
                        help="可选初始化权重；默认从原始 MegaDescriptor 开始，避免继承旧版跨批次 CE")
    parser.add_argument("--batches-per-epoch", type=int,
                        default=cfg.get("batches_per_epoch", 40))
    parser.add_argument("--lambda-hn", type=float, default=cfg.get("lambda_hn", 0.2),
                        help="triplet 损失权重")
    parser.add_argument("--extract", action="store_true",
                        help="训练完成后用 best.pt 重新提取 pilot 特征（--embeddings-out）")
    parser.add_argument("--embeddings-out", type=Path,
                        default=None,
                        help="--extract 的输出特征路径（默认按 --out 目录名生成）")
    args = parser.parse_args()

    for field in ("pilot", "images_root", "out", "init_ckpt"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, _repo_path(Path(value)))
    guard_training_output(args.out, args.overwrite)
    if args.embeddings_out is None:
        args.embeddings_out = base / "embeddings" / f"embeddings_metric_{args.out.name}.npy"
    else:
        args.embeddings_out = _repo_path(args.embeddings_out)
    if args.extract:
        guard_embedding_output(args.embeddings_out, args.overwrite)

    df = load_confirmed(args.pilot, args.images_root)
    if df["confirmed_identity"].nunique() < args.val_n + 2:
        raise SystemExit(
            f"个体数 {df['confirmed_identity'].nunique()} 过少，无法留出 {args.val_n} 个验证个体")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=f".{args.out.name}.training-", dir=args.out.parent) as tmp_dir:
        staging = Path(tmp_dir)
        run_training(args, df, staging)
        _promote_training_outputs(staging, args.out)
    if args.extract:
        extract_features(args, args.out, args.pilot, args.images_root)


if __name__ == "__main__":
    main()
