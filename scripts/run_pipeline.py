"""
批内归档管线（正式入口 2/7）：新批次 → 检测裁剪 → r3 特征 → 候选聚类 →
子簇化 → 簇级多帧投票匹配历史库 → 审核清单。

用法：
    python scripts/run_pipeline.py --pool                      # 散图池验证（复用预提取产物）
    python scripts/run_pipeline.py --input-manifest 清单.csv \
        --batch-name 批次名                                    # 新批次完整流程

输出（out/batch_name/ 下）：clusters.csv（逐图）/ cluster_matches.csv（簇级）/
representatives/（代表图）/ summary.json / contact_sheets/（--sheets）。

数据语义：簇 = Candidate Cluster（-1 噪声合法）；match/suspected_new 均为
候选，人工核验后才能叫个体。阈值来自实验标定（E5 簇级 0.58 / E4 单图 0.50）。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.config import load_config  # noqa: E402
from whitewhale.pipeline.archival import run  # noqa: E402


def main():
    base = REPO_ROOT / "outputs"
    cfg = load_config("pipeline")
    retr = cfg.get("retrieval", {})
    clut = cfg.get("clustering", {})
    crop = cfg.get("crop", {})
    parser = argparse.ArgumentParser(description="批内簇级归档管线")
    parser.add_argument("--pool", action="store_true",
                        help="散图池模式：复用 outputs/embeddings/ 预提取 r4+YOLO 特征")
    parser.add_argument("--input-manifest", type=Path, default=None,
                        help="新批次图片清单（image_id, relative_path[, session_id]）")
    parser.add_argument("--batch-name", default="batch",
                        help="批次标识（输出目录名）")
    parser.add_argument("--images-root", type=Path,
                        default=Path(cfg.get("data_root", "src_dataset")),
                        help="图片根目录（只读）")
    parser.add_argument("--ckpt", type=Path,
                        default=REPO_ROOT / cfg.get("reid_checkpoint",
                                                    "outputs/metric_learning/r4/best.pt"),
                        help="r4 微调权重（新批次模式）")
    parser.add_argument("--gallery-embeddings", type=Path,
                        default=base / "embeddings" / "embeddings_metric_r4_yolocrop_v2.npy",
                        help="历史库特征（已确认个体）")
    parser.add_argument("--gallery-meta", type=Path,
                        default=base / "embeddings" / "embeddings_metric_r4_yolocrop_v2_meta.csv")
    parser.add_argument("--min-cluster-size", type=int, default=clut.get("min_cluster_size", 3))
    parser.add_argument("--subcluster-min-size", type=int,
                        default=clut.get("subcluster_min_size", 4),
                        help="大于等于此成员的簇才做内部子簇化")
    parser.add_argument("--topk", type=int, default=retr.get("topk", 3))
    parser.add_argument("--threshold-cluster", type=float,
                        default=retr.get("threshold_cluster", 0.58),
                        help="簇级多帧投票阈值（E5 标定 FA≤5%）")
    parser.add_argument("--threshold-image", type=float,
                        default=retr.get("threshold_image", 0.50),
                        help="单图阈值（E4 标定区间下限，噪声/残噪退化用）")
    parser.add_argument("--out", type=Path, default=base / "cluster_archival")
    parser.add_argument("--sheets", action="store_true", help="生成候选簇拼图（人工审核）")
    parser.add_argument("--max-sheets", type=int, default=200)
    parser.add_argument("--det-weights", type=Path,
                        default=REPO_ROOT / cfg.get("detector_checkpoint",
                                                    "models/detectors/yolov8n_dorsalfin.pt"))
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-imgsz", type=int, default=1024)
    parser.add_argument("--det-device", default="cuda")
    parser.add_argument("--det-pad-x", type=float, default=crop.get("pad_x", 0.30))
    parser.add_argument("--det-pad-up", type=float, default=crop.get("pad_up", 0.15))
    parser.add_argument("--det-pad-down", type=float, default=crop.get("pad_down", 0.60))
    args = parser.parse_args()

    if args.pool == bool(args.input_manifest):
        parser.error("必须且只能指定其一：--pool（散图池）或 --input-manifest（新批次）")
    # 输出目录 = out/batch_name（archival.run 约定 out 为完整输出目录）
    args.out = args.out / args.batch_name
    run(args)


if __name__ == "__main__":
    main()
