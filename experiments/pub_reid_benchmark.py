"""
主路线 A：公开数据 zero-shot Re-ID 实验（Recall@K / mAP）。

语义（方向调整后）：
- 只用有可靠 individual_id 的公开数据（Happywhale / BelugaID）；
- query / gallery 分离，同一 identity 的 query 与 gallery 必须来自不同 encounter
  （Happywhale 无 encounter 字段 → 同一 identity 不跨 split 混用，且对同一图像去重）；
- 不训练、不微调，只用预训练 backbone（MegaDescriptor / DINOv2）提取特征；
- 输出指标与候选结果，全部可追溯（保留 image_id / 原路径 / identity / split）。

用法示例：
    python experiments/pub_reid_benchmark.py --dataset happywhale --model megadescriptor
    python experiments/pub_reid_benchmark.py --dataset happywhale --model dinov2 --mock
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pub_reid.dataset.happywhale import HappywhaleAdapter  # noqa: E402
from whitewhale.reid.embedding import (  # noqa: E402
    DINOv2Adapter,
    MegaDescriptorAdapter,
)
from whitewhale.reid.evaluation import mean_average_precision, recall_at_k  # noqa: E402
from whitewhale.reid.retrieval import cosine_topk  # noqa: E402


def load_model(name: str, mock: bool, dinov2_weight: str | None = None):
    """按名字实例化模型；mock 返回 None 表示随机特征模式。"""
    if mock or name == "mock":
        return None
    if name == "megadescriptor":
        return MegaDescriptorAdapter()
    if name == "dinov2":
        return DINOv2Adapter(weight_path=dinov2_weight)
    raise ValueError(f"未知模型: {name}")


def mock_encode(n, dim, rng):
    """mock：随机 L2 归一化特征，用于离线验证 pipeline（不跑真模型）。"""
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def split_query_gallery(df: pd.DataFrame, identity_col: str,
                        seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 identity 划分 query / gallery。

    无 encounter 字段的数据集（Happywhale）：每个 identity 最多一张作 query，
    其余进 gallery，且同一 identity 的 query/gallery 不共享同一张图（防泄漏）。
    每个 identity 至少 2 张才参与（query + gallery 都非空）。
    """
    rng = np.random.default_rng(seed)
    query_rows, gallery_rows = [], []
    for _, grp in df.groupby(identity_col, sort=False):
        if len(grp) < 2:
            continue
        # 按位置选 query（保证每个 identity 至多一张），其余进 gallery
        q_pos = int(rng.integers(0, len(grp)))
        q_label = grp.index[q_pos]
        query_rows.append(grp.loc[[q_label]])  # 1 行 DataFrame
        gallery_rows.append(grp.drop(index=q_label))
    q_df = pd.concat(query_rows, axis=0) if query_rows else df.iloc[0:0]
    g_df = pd.concat(gallery_rows, axis=0) if gallery_rows else df.iloc[0:0]
    return q_df, g_df


def load_beluga_scenarios(data_root: Path, max_scenarios: int | None = None
                          ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """加载 Beluga 10 个官方 scenario 的 (query_df, gallery_df) 列表。

    Beluga 协议：query 图在 database 内（自匹配），评估必须排除 query 自身。
    每个 scenario 是独立任务（官方划分，不跨 scenario 混用）。
    """
    import zipfile

    zip_path = data_root / "beluga-id-test.zip"
    meta = pd.read_csv(zipfile.ZipFile(zip_path).open("private_test_metadata.csv"))
    meta["image_id"] = meta["image_id"].astype(str)
    identity_map = {f"beluga__{r.original_whale_id}" for _, r in meta.iterrows()}
    del identity_map  # 身份已在 adapter 中 namespace

    query_list, gallery_list = [], []
    scenario_files = sorted((data_root / "beluga-scenarios").glob("query_*.csv")) \
        if (data_root / "beluga-scenarios").exists() else []
    # 从 zip 内读 scenario 定义
    with zipfile.ZipFile(zip_path) as z:
        q_files = sorted(n for n in z.namelist() if n.startswith("code-execution/queries/scenario"))
        d_files = sorted(n for n in z.namelist() if n.startswith("code-execution/databases/scenario"))
        for qf, df_f in zip(q_files, d_files):
            if max_scenarios and len(query_list) >= max_scenarios:
                break
            q_csv = pd.read_csv(z.open(qf))
            d_csv = pd.read_csv(z.open(df_f))
            q_ids = q_csv["query_image_id"].astype(str)
            d_ids = d_csv["database_image_id"].astype(str)
            q_rows = meta[meta["image_id"].isin(q_ids)].copy()
            d_rows = meta[meta["image_id"].isin(d_ids)].copy()
            # 把 zip 内路径映射为已解图路径
            img_dir = data_root / "test_images"
            q_rows["image_path"] = q_rows["image_id"].apply(lambda x: str(img_dir / f"{x}.jpg"))
            d_rows["image_path"] = d_rows["image_id"].apply(lambda x: str(img_dir / f"{x}.jpg"))
            q_rows["identity"] = "beluga__" + q_rows["original_whale_id"].astype(str)
            d_rows["identity"] = "beluga__" + d_rows["original_whale_id"].astype(str)
            # 只保留有身份映射的（防御：scenario 里可能有 meta 外图片）
            query_list.append(q_rows)
            gallery_list.append(d_rows)
    return query_list, gallery_list


def main():
    parser = argparse.ArgumentParser(description="主路线 A：公开数据 zero-shot Re-ID")
    parser.add_argument("--dataset", choices=["happywhale", "beluga"], default="happywhale")
    parser.add_argument("--model", choices=["megadescriptor", "dinov2"], default="megadescriptor")
    parser.add_argument("--data-root", type=Path,
                        default=Path("D:/dolphin_data/happywhale"),
                        help="数据目录（beluga 数据集请传 D:/dolphin_data/beluga）")
    parser.add_argument("--cache-images", action="store_true",
                        help="把 parquet 内嵌图片落盘到 data-root/images（首次运行用）")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "outputs" / "pub_reid")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-query", type=int, default=0,
                        help="限制 query 数（0=全部；调试用）")
    parser.add_argument("--mock", action="store_true", help="离线验证模式（随机特征）")
    parser.add_argument("--dinov2-weight", type=str, default=None,
                        help="DINOv2 官方权重 .pth 本地路径（网络不可用时离线加载）")
    args = parser.parse_args()

    out_dir = args.out / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 数据
    if args.dataset == "happywhale":
        if args.cache_images:
            data = HappywhaleAdapter(image_cache_dir=args.data_root / "images").load(args.data_root)
        else:
            data = HappywhaleAdapter().load(args.data_root)
        df = data.df
        if "image_path" not in df.columns or df["image_path"].isna().all():
            raise SystemExit(
                "metadata 无落盘图片（image_path 全空）。请先运行 --cache-images 落盘图片。")

        # 2. query / gallery 划分
        q_df, g_df = split_query_gallery(df, identity_col="identity")
        if args.max_query:
            q_df = q_df.head(args.max_query)
        print(f"[data] {data.n_images} 图 / {data.n_identities} 个体 → "
              f"query {len(q_df)} / gallery {len(g_df)}")

        # 3. 特征
        model = load_model(args.model, args.mock, dinov2_weight=args.dinov2_weight)
        rng = np.random.default_rng(42)
        if model is None:
            print("[model] MOCK 随机特征（离线验证模式）")
            q_emb = mock_encode(len(q_df), 768, rng)
            g_emb = mock_encode(len(g_df), 768, rng)
        else:
            print(f"[model] {model.name} ({model.feat_dim}D)")
            q_emb = model.encode_paths(q_df["image_path"].tolist(), batch_size=args.batch_size)
            g_emb = model.encode_paths(g_df["image_path"].tolist(), batch_size=args.batch_size)

        # 4. 检索 + 评估（无自匹配问题：query 不在 gallery）
        scores, idx = cosine_topk(q_emb, g_emb, k=args.k)
        q_ids = q_df["identity"].values
        g_ids = g_df["identity"].values
        rec = recall_at_k(scores, idx, q_ids, g_ids, k_list=(1, 5, 10))
        ap = mean_average_precision(scores, idx, q_ids, g_ids)
        print(f"[metric] Recall@1={rec[1]:.3f}  Recall@5={rec[5]:.3f}  "
              f"Recall@10={rec[10]:.3f}  mAP={ap:.3f}")

        # 5. 输出：全部可追溯
        rows = []
        for i in range(len(q_df)):
            for j in range(args.k):
                rows.append({
                    "query_image_id": q_df.iloc[i]["image_name"],
                    "query_identity": q_ids[i],
                    "query_split": q_df.iloc[i]["split"],
                    "rank": j + 1,
                    "cand_image_id": g_df.iloc[idx[i, j]]["image_name"],
                    "cand_identity": g_ids[idx[i, j]],
                    "score": float(scores[i, j]),
                    "hit": int(g_ids[idx[i, j]] == q_ids[i]),
                })
        cand = pd.DataFrame(rows)
        cand.to_csv(out_dir / "topk_candidates.csv", index=False)
        agg = cand[cand["rank"] == 1].groupby("query_identity").agg(
            n_query=("query_image_id", "count"), n_hit=("hit", "sum"),
            mean_top1_score=("score", "mean")).reset_index()
        agg.to_csv(out_dir / "top1_summary.csv", index=False)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "dataset": args.dataset, "model": args.model, "mock": bool(args.mock),
                "n_images": data.n_images, "n_identities": data.n_identities,
                "n_query": len(q_df), "n_gallery": len(g_df),
                "recall_at_1": rec[1], "recall_at_5": rec[5],
                "recall_at_10": rec[10], "mAP": ap,
            }, f, indent=2, ensure_ascii=False)

    elif args.dataset == "beluga":
        from pub_reid.dataset.beluga import BelugaTestAdapter

        # 先确保 test 图已解压
        data = BelugaTestAdapter().load(args.data_root)
        q_list, g_list = load_beluga_scenarios(args.data_root)

        # 官方 GT 匹配对（private_test_labels.csv）：
        # query_id 格式 scenarioXX-testNNNN，database_image_id 是正样本图。
        # 协议：同身份的其他图不算正样本，只认官方匹配对（防自匹配虚高）。
        import zipfile

        with zipfile.ZipFile(args.data_root / "beluga-id-test.zip") as z:
            labels = pd.read_csv(z.open("private_test_labels.csv"))
        labels["query_id"] = labels["query_id"].astype(str)
        labels["database_image_id"] = labels["database_image_id"].astype(str)
        db_index = {id_: i for i, id_ in enumerate(data.df["image_id"].tolist())}

        model = load_model(args.model, args.mock, dinov2_weight=args.dinov2_weight)
        rng = np.random.default_rng(42)
        results = []
        for i, (q_rows, g_rows) in enumerate(zip(q_list, g_list), start=1):
            scen = f"scenario{i:02d}"
            # 每行的 query_image_id 即 query（官方定义：每行一条 query）
            q_ids = q_rows["image_id"].astype(str).tolist()
            # 排除 query 自身图（Beluga 协议：query 在 database 内）
            g_safe = g_rows[~g_rows["image_id"].isin(q_ids)].reset_index(drop=True)
            # 官方正样本：labels 里 (scenarioXX-testNNNN -> db 图) 的 db 索引
            lab = labels[labels["query_id"] == f"{scen}-{q_ids[0]}"][["database_image_id"]]
            gt = {}
            for qid in q_ids:
                db_ids = labels.loc[
                    labels["query_id"] == f"{scen}-{qid}", "database_image_id"]
                gt[qid] = {db_index[d] for d in db_ids if d in db_index}
            if model is None:
                q_emb = mock_encode(len(q_rows), 768, rng)
                g_emb = mock_encode(len(g_safe), 768, rng)
            else:
                q_emb = model.encode_paths(q_rows["image_path"].tolist(), batch_size=args.batch_size)
                g_emb = model.encode_paths(g_safe["image_path"].tolist(), batch_size=args.batch_size)
            scores, idx = cosine_topk(q_emb, g_emb, k=args.k)
            gt_sets = [gt[qid] for qid in q_ids]
            rec = recall_at_k(scores, idx, None, None, k_list=(1, 5, 10), gt_sets=gt_sets)
            ap = mean_average_precision(scores, idx, None, None, gt_sets=gt_sets)
            results.append({"scenario": scen, "n_query": len(q_rows),
                            "n_gallery": len(g_safe), "recall@1": rec[1],
                            "recall@5": rec[5], "recall@10": rec[10], "mAP": ap})
            print(f"[{scen}] query={len(q_rows)} gallery={len(g_safe)} "
                  f"Recall@1={rec[1]:.3f} Recall@5={rec[5]:.3f} "
                  f"Recall@10={rec[10]:.3f} mAP={ap:.3f}")
        summary = pd.DataFrame(results)
        summary.to_csv(out_dir / "scenario_summary.csv", index=False)
        if len(summary):
            avg = summary[["recall@1", "recall@5", "recall@10", "mAP"]].mean()
            print(f"[avg] Recall@1={avg['recall@1']:.3f} Recall@5={avg['recall@5']:.3f} "
                  f"Recall@10={avg['recall@10']:.3f} mAP={avg['mAP']:.3f}")
            with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump({
                    "dataset": args.dataset, "model": args.model, "mock": bool(args.mock),
                    "n_scenarios": len(summary),
                    "recall_at_1": avg["recall@1"], "recall_at_5": avg["recall@5"],
                    "recall_at_10": avg["recall@10"], "mAP": avg["mAP"],
                    "per_scenario": results,
                }, f, indent=2, ensure_ascii=False)

    print(f"[out] {out_dir}")


if __name__ == "__main__":
    main()
