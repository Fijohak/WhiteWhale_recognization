"""
度量学习训练（正式入口 5/7）：伪标签 ArcFace + 可选跨群 hard negative。

r1/r2（纯 CE）与 r3（CE + λ×跨群 triplet）共用同一入口，差异由
--hard-negative 控制（默认开 = r3 链路，当前正式特征源）：

    python scripts/train_reid.py --hard-negative          # r3（跨群 HN 微调）
    python scripts/train_reid.py --no-hard-negative       # r1/r2（纯 ArcFace）

两阶段训练：冻结 backbone 训 head → 解冻低学习率微调（HN 链路加
λ×跨群 triplet）。配置默认值读 configs/reid.yaml，可被 CLI 覆盖。

语义：伪标签来自人工初审（Candidate 级，非专家复核）；不使用水平翻转
（左右侧背鳍特征可能不同）；划分按个体（同一体不跨 train/val）。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.config import load_config  # noqa: E402
from whitewhale.reid.training import extract_features, load_confirmed, run_training  # noqa: E402


def main():
    base = REPO_ROOT / "outputs"
    cfg = load_config("reid")
    parser = argparse.ArgumentParser(description="度量学习训练（伪标签 ArcFace + 可选跨群 HN）")
    parser.add_argument("--pilot", type=Path, default=base / "pilot" / "pilot_set.csv",
                        help="人工初审清单（含 confirmed_identity，Candidate 级标签）")
    parser.add_argument("--images-root", type=Path,
                        default=Path(load_config("pipeline").get("data_root", "src_dataset")),
                        help="图片根目录（只读）")
    parser.add_argument("--out", type=Path, default=base / "metric_learning" / "r3",
                        help="输出目录（best.pt / history.csv / metrics.json）")
    parser.add_argument("--val-n", type=int, default=cfg.get("val_n", 6),
                        help="验证个体数（按个体留出）")
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    parser.add_argument("--epochs-stage1", type=int, default=cfg.get("epochs_stage1", 20),
                        help="阶段一：冻结 backbone 训 head")
    parser.add_argument("--epochs-stage2", type=int, default=cfg.get("epochs_stage2", 25),
                        help="阶段二：解冻微调")
    parser.add_argument("--lr-head", type=float, default=cfg.get("lr_head", 0.001))
    parser.add_argument("--lr-backbone", type=float, default=cfg.get("lr_backbone", 0.000005))
    parser.add_argument("--batch", type=int, default=cfg.get("batch", 16))
    # ---- 跨群 hard negative（r3 链路）----
    parser.add_argument("--hard-negative", dest="hard_negative", action="store_true",
                        default=cfg.get("hard_negative", True),
                        help="跨群 triplet 辅助损失（默认开 = r3）")
    parser.add_argument("--no-hard-negative", dest="hard_negative", action="store_false",
                        help="纯 ArcFace CE（r1/r2 链路）")
    parser.add_argument("--init-ckpt", type=Path,
                        default=base / "metric_learning" / "r2" / "best.pt",
                        help="HN 链路初始化权重（建议 r2 训练产物）")
    parser.add_argument("--batches-per-epoch", type=int,
                        default=cfg.get("batches_per_epoch", 40))
    parser.add_argument("--lambda-hn", type=float, default=cfg.get("lambda_hn", 0.2),
                        help="triplet 损失权重")
    parser.add_argument("--extract", action="store_true",
                        help="训练完成后用 best.pt 重新提取 pilot 特征（--embeddings-out）")
    parser.add_argument("--embeddings-out", type=Path,
                        default=base / "embeddings" / "embeddings_metric_r3.npy",
                        help="--extract 的输出特征路径")
    args = parser.parse_args()

    df = load_confirmed(args.pilot, args.images_root)
    if df["confirmed_identity"].nunique() < args.val_n + 2:
        raise SystemExit(
            f"个体数 {df['confirmed_identity'].nunique()} 过少，无法留出 {args.val_n} 个验证个体")
    args.out.mkdir(parents=True, exist_ok=True)
    run_training(args, df, args.out)
    if args.extract:
        extract_features(args, args.out, args.pilot, args.images_root)


if __name__ == "__main__":
    main()
