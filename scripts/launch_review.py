"""
人工审核网页（正式入口 3/7）：按候选簇/子簇单元逐簇审核 → 导出确认结果。

用法：
    python scripts/launch_review.py --port 8001        # 启动审核网页
    python scripts/launch_review.py --export           # 导出 Confirmed Individual

操作即时写入 outputs/review/review_annotations.csv（可追溯、可恢复）；
确认后运行 --export 导出 outputs/review/confirmed_individuals.csv。

数据语义：簇号 = Candidate Cluster，审核确认后才叫个体；
uncertain / reject 为合法状态，不强制分配。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.config import load_config  # noqa: E402
from whitewhale.review.app import build_app, export_confirmed  # noqa: E402


def main():
    base = REPO_ROOT / "outputs"
    cfg = load_config("pipeline")
    review_cfg = cfg.get("review", {})
    parser = argparse.ArgumentParser(description="中华白海豚个体识别：人工审核网页")
    parser.add_argument("--clusters", type=Path,
                        default=base / "clusters" / "clusters.csv",
                        help="候选簇照片表（行 = 待审核照片）")
    parser.add_argument("--annotations", type=Path,
                        default=base / "review" / "review_annotations.csv",
                        help="审核标注保存位置（每次操作即时写入）")
    parser.add_argument("--images-root", type=Path,
                        default=Path(cfg.get("data_root", "I:/")),
                        help="图片根目录（含 01/ 03/ 子目录）")
    parser.add_argument("--embeddings", type=Path,
                        default=base / "embeddings" / "embeddings.npy",
                        help="特征库（相似度提醒用，可选）")
    parser.add_argument("--embeddings-meta", type=Path,
                        default=base / "embeddings" / "embeddings_meta.csv",
                        help="特征库 meta（含 image_id）")
    parser.add_argument("--cluster-filter", default="all",
                        help="启动时默认筛选：all / noise / 簇号（如 1）")
    parser.add_argument("--export", action="store_true", help="导出审核结果后退出")
    parser.add_argument("--out", type=Path,
                        default=base / "review" / "confirmed_individuals.csv",
                        help="导出路径（--export 时生效）")
    parser.add_argument("--port", type=int, default=review_cfg.get("port", 8001))
    parser.add_argument("--host", default=review_cfg.get("host", "127.0.0.1"))
    parser.add_argument("--reviewer", default="",
                        help="审核者标识（多人审核时区分，写入标注文件 reviewer 列）")
    parser.add_argument("--history-lookup", type=Path, default=None,
                        help="历史库对照照片目录（按个体分文件夹，如 review_package/history_lookup；"
                             "跨时间审核时指定，页面可点开历史个体照片比对）")
    parser.add_argument("--history-quality", type=Path, default=None,
                        help="历史对照图质量表 CSV（filename,quality=clear/low；低质对照图前端可隐藏）")
    parser.add_argument("--batch-embeddings", type=Path, default=None,
                        help="批次特征 npy（pipeline 产物，如 cross_time/<批次>/embeddings.npy；"
                             "用于簇内相似度辅助：混簇中离群者沉底标红）")
    args = parser.parse_args()

    if args.export:
        export_confirmed(args)
        return

    import uvicorn
    app = build_app(args)
    print(f"[review] 审核网页就绪: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
