"""
本地图库个体查询网页（正式入口 4/7）：上传单张照片 → 检测裁剪 → r3 特征 →
全库 Top-K 检索 → 三态判定（known / unknown）。

用法：
    python scripts/launch_query.py --port 8000
    python scripts/launch_query.py --no-detect        # 关掉检测裁剪（整图直查）
    浏览器打开 http://127.0.0.1:8000

gallery 与查询统一为 "YOLO 检测裁剪 + r3 跨群 HN 微调特征"；查询模型自动
匹配 gallery 特征（读 embedding 旁 config），显式 --model 时必须同源。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.config import load_config  # noqa: E402
from whitewhale.query import build_app  # noqa: E402


def main():
    base = REPO_ROOT / "outputs"
    cfg = load_config("pipeline")
    query_cfg = cfg.get("query", {})
    crop_cfg = cfg.get("crop", {})
    parser = argparse.ArgumentParser(description="本地图库个体查询 Web 客户端")
    parser.add_argument("--embeddings", type=Path,
                        default=base / query_cfg.get(
                            "embeddings", "embeddings/embeddings_metric_r3_yolocrop.npy"),
                        help="gallery 特征（r3 微调 + YOLO 裁剪，见同名 _config.json）")
    parser.add_argument("--meta", type=Path,
                        default=base / query_cfg.get(
                            "meta", "embeddings/embeddings_metric_r3_yolocrop_meta.csv"))
    parser.add_argument("--pilot", type=Path,
                        default=base / query_cfg.get("pilot", "pilot/pilot_set.csv"))
    parser.add_argument("--images-root", type=Path,
                        default=Path(cfg.get("data_root", "src_dataset")),
                        help="图片根目录（含 01/ 03/ 子目录）")
    parser.add_argument("--model", type=str, default=None,
                        help="查询模型覆盖（默认自动匹配 gallery 特征模型）："
                             "megadescriptor / dinov2 / metric-learning")
    parser.add_argument("--dinov2-weight", type=str, default=None,
                        help="DINOv2 官方权重 .pth（gallery 为 dinov2 特征时必填）")
    parser.add_argument("--metric-ckpt", type=Path,
                        default=REPO_ROOT / cfg.get(
                            "reid_checkpoint", "outputs/metric_learning/r3/best.pt"),
                        help="伪标签微调权重（gallery 为 metric-learning 特征时使用）")
    parser.add_argument("--k", type=int, default=query_cfg.get("k", 10))
    parser.add_argument("--threshold", type=float,
                        default=query_cfg.get("threshold", 0.55),
                        help="三态判定阈值（E4 标定 FA≤5% 区间 0.5-0.6 的中值，"
                             "r3 特征下使用）")
    parser.add_argument("--detect", action="store_true", default=True,
                        help="查询图先走 YOLO 背鳍检测裁剪（默认开；未检出回退整图）")
    parser.add_argument("--no-detect", dest="detect", action="store_false")
    parser.add_argument("--det-weights", type=Path,
                        default=REPO_ROOT / cfg.get(
                            "detector_checkpoint", "models/detectors/yolov8n_dorsalfin.pt"))
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-imgsz", type=int, default=1024)
    parser.add_argument("--det-device", default="cuda")
    parser.add_argument("--det-pad-x", type=float, default=crop_cfg.get("pad_x", 0.30))
    parser.add_argument("--det-pad-up", type=float, default=crop_cfg.get("pad_up", 0.15))
    parser.add_argument("--det-pad-down", type=float, default=crop_cfg.get("pad_down", 0.60))
    parser.add_argument("--port", type=int, default=query_cfg.get("port", 8000))
    parser.add_argument("--host", default=query_cfg.get("host", "127.0.0.1"))
    args = parser.parse_args()

    import uvicorn
    app = build_app(args)
    print(f"[query_app] gallery {app.state.n_gallery} 张 / 模型 {app.state.model_name} / "
          f"阈值 {args.threshold} / 检测裁剪 {'开' if args.detect else '关'}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
