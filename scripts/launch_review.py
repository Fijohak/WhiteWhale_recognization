"""
人工审核网页（正式入口 3/7）：按候选簇/子簇单元多人独立审核 → 保守裁决。

用法：
    python scripts/launch_review.py --clusters 候选簇.csv --reviewer alice \
        --reviewer-roster alice bob carol --port 8001
    python scripts/launch_review.py --clusters 候选簇.csv --export \
        --reviewer-roster alice bob carol --min-reviewers 3

操作即时写入 outputs/review/review_annotations.csv（可追溯、可恢复）；
多人票按 image_id+reviewer 并存；确认后运行 --export 导出投票汇总与
outputs/review/confirmed_individuals.csv。命名审核票存在时，个人参数不能绕过共识导出。

数据语义：簇号 = Candidate Cluster，审核确认后才叫个体；
uncertain / reject 为合法状态，不强制分配。
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from whitewhale.config import load_config  # noqa: E402
from whitewhale.review.app import (  # noqa: E402
    build_app,
    export_confirmed,
    normalize_reviewer_roster,
    resolve_reviewer_id,
)


def _repo_path(path: Path) -> Path:
    """CLI 相对路径统一按仓库根解析，不依赖启动时工作目录。"""
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    cfg = load_config("pipeline")
    base = _repo_path(Path(cfg.get("output_root", "outputs")))
    data_root = _repo_path(Path(cfg.get("data_root", "src_dataset")))
    review_cfg = cfg.get("review", {})
    parser = argparse.ArgumentParser(description="中华白海豚个体识别：人工审核网页")
    parser.add_argument("--clusters", type=Path,
                        required=True,
                        help="候选簇照片表（行 = 待审核照片）")
    parser.add_argument("--annotations", type=Path,
                        default=base / "review" / "review_annotations.csv",
                        help="审核标注保存位置（每次操作即时写入）")
    parser.add_argument("--images-root", type=Path,
                        default=data_root,
                        help="图片根目录（含 01/ 03/ 子目录）")
    parser.add_argument("--embeddings", type=Path,
                        default=None,
                        help="生成期特征库（相似度提醒用；必须与 --embeddings-meta 同时给出）")
    parser.add_argument("--embeddings-meta", type=Path,
                        default=None,
                        help="特征库 meta（含 image_id；必须有生成期行绑定溯源）")
    parser.add_argument("--cluster-filter", default="all",
                        help="启动时默认筛选：all / noise / 簇号（如 1）")
    parser.add_argument("--export", action="store_true", help="导出审核结果后退出")
    parser.add_argument("--out", type=Path,
                        default=base / "review" / "confirmed_individuals.csv",
                        help="导出路径（--export 时生效）")
    parser.add_argument("--summary-out", type=Path,
                        default=base / "review" / "review_vote_summary.csv",
                        help="多人原始票据与保守裁决汇总（--export 时生效）")
    parser.add_argument("--min-reviewers", type=int, default=3,
                        help="形成一致裁决所需的最少独立审核人数（默认 3）")
    parser.add_argument("--port", type=int, default=review_cfg.get("port", 8001))
    parser.add_argument("--host", default=review_cfg.get("host", "127.0.0.1"))
    parser.add_argument("--reviewer", default="",
                        help="审核者标识（多人审核时区分，写入标注文件 reviewer 列）")
    parser.add_argument(
        "--reviewer-roster", nargs="+",
        default=review_cfg.get("reviewer_roster"), metavar="ID",
        help=("受控审核人 canonical ID 名单（至少 3 人，可在 pipeline.review."
              "reviewer_roster 配置）；仅做名单约束，不等同于账号认证"),
    )
    parser.add_argument("--history-lookup", type=Path, default=None,
                        help="历史库对照照片目录（按个体分文件夹，如 review_package/history_lookup；"
                             "跨时间审核时指定，页面可点开历史个体照片比对）")
    parser.add_argument("--history-quality", type=Path, default=None,
                        help="历史对照图质量表 CSV（filename,quality=clear/low；低质对照图前端可隐藏）")
    parser.add_argument("--batch-embeddings", type=Path, default=None,
                        help="批次特征 npy（pipeline 产物，如 cross_time/<批次>/embeddings.npy；"
                             "用于簇内相似度辅助：混簇中离群者沉底标红）")
    args = parser.parse_args()

    for field in (
            "clusters", "annotations", "images_root", "embeddings",
            "embeddings_meta", "out", "summary_out", "history_lookup",
            "history_quality", "batch_embeddings"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, _repo_path(Path(value)))

    try:
        args.reviewer_roster = normalize_reviewer_roster(
            args.reviewer_roster,
            min_reviewers=max(3, int(args.min_reviewers)),
        )
        if args.export:
            export_confirmed(args)
            return
        args.reviewer = resolve_reviewer_id(args.reviewer, args.reviewer_roster)
    except ValueError as exc:
        parser.error(str(exc))

    import uvicorn
    if str(args.host).strip().casefold() not in {"127.0.0.1", "localhost", "::1"}:
        print("[review] 警告：reviewer roster 只是受控名单，不是登录认证；"
              "当前服务监听非本机地址，请仅在受信任网络中使用。")
    app = build_app(args)
    print(f"[review] 审核网页就绪: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
