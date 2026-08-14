"""
FiftyOne 人工审核流程（本地数据候选聚类审核）。

数据语义（方向调整后）：
- clusters.csv 里的 cluster 只是 **Candidate Cluster**（候选分组），不是真实个体；
- HDBSCAN 的 -1 是合法噪声，不强制分配；
- 人工在 FiftyOne App 中逐簇确认后，写入 confirmed_identity 才是 Confirmed Individual；
- 所有行保留 image_id / source_path / source_group / session 追溯。

用法：
    python scripts/fiftyone_review.py                      # 加载/重建审核数据集
    python scripts/fiftyone_review.py --export confirmed   # 导出人工审核结果

需要 fiftyone 已安装（pip install fiftyone）。
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_review_dataset(clusters_csv: Path, images_root: Path) -> pd.DataFrame:
    """加载聚类结果，拼出绝对图片路径（可追溯）。"""
    df = pd.read_csv(clusters_csv)
    df["source_path"] = df["relative_path"].map(
        lambda p: str(images_root / p))
    return df


def build_dataset(df: pd.DataFrame, dataset_name: str = "local_reid_review"):
    """构建/更新 FiftyOne 数据集。

    标签字段：
    - cluster: Candidate Cluster（-1 = 噪声，不强制分配）
    - anchor_group / source_group: 原目录组（弱信息，仅参考）
    - session_id / quality_band: 拍摄批次与评分区间
    - confirmed_identity: 人工审核字段（初始空）
    """
    import fiftyone as fo

    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)
    ds = fo.Dataset(dataset_name)
    samples = []
    for _, row in df.iterrows():
        sample = fo.Sample(filepath=row["source_path"])
        sample["image_id"] = row["image_id"]
        sample["cluster"] = int(row["cluster"])
        sample["cluster_probability"] = float(row.get("cluster_probability", 0.0))
        sample["anchor_group"] = str(row.get("individual_id", ""))
        sample["source_group"] = str(row.get("source_group", ""))
        sample["session_id"] = str(row.get("session_id", ""))
        sample["quality_band"] = str(row.get("quality_band", ""))
        sample["confirmed_identity"] = ""
        samples.append(sample)
    ds.add_samples(samples)
    ds.persistent = True
    counts = ds.count_values("cluster")
    n_clusters = sum(1 for k in counts if int(k) >= 0)
    print(f"[fiftyone] 数据集 {dataset_name}: {len(samples)} 张 / "
          f"{n_clusters} 候选簇 + 噪声（-1）{counts.get(-1, 0)} 张")
    print(f"          簇分布: {sorted(counts.items())}")
    print("          审核方式：App 中按 cluster 字段分组/排序，逐簇核对；")
    print("          或运行 scripts/contact_sheets.py --cluster 生成网格图离线审核。")
    return ds


def export_confirmed(dataset_name: str, out_csv: Path) -> None:
    """导出人工审核结果。

    审核以 tag 为准（App 中缩略图上直接打，最直观）：
    - CI-xxx  （如 CI-001）：该照片属于个体 CI-xxx，同一只海豚用同一个 tag；
    - uncertain：无法判断，留给后续；
    - reject：确认不是任何已审核个体（噪声 / 新个体候选）。
    兼容旧的 confirmed_identity 字段（字段与 tag 都有时以 tag 为准）。
    """
    import fiftyone as fo

    ds = fo.load_dataset(dataset_name)
    rows = []
    for sample in ds:
        ci = [t for t in sample.tags if t.startswith("CI-")]
        ident = ci[0] if ci else (sample.confirmed_identity or "")
        if not ident:
            continue
        status = "uncertain" if "uncertain" in sample.tags else "reject" if "reject" in sample.tags else "confirmed"
        rows.append({
            "image_id": sample.image_id,
            "confirmed_identity": ident,
            "status": status,
            "cluster": sample.cluster,
            "source_path": sample.filepath,
            "tags": ",".join(sorted(sample.tags)),
        })
    if not rows:
        print("[fiftyone] 尚无人工确认的个体（无 CI-* tag / confirmed_identity 全空）。")
        return
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[fiftyone] 导出 {len(out)} 条人工确认 → {out_csv}")
    print(f"          个体数: {out['confirmed_identity'].nunique()}（{sorted(out['confirmed_identity'].unique())}）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FiftyOne 人工审核流程")
    base = Path(__file__).resolve().parents[1] / "outputs"
    parser.add_argument("--clusters", type=Path,
                        default=base / "clusters" / "clusters.csv")
    parser.add_argument("--images-root", type=Path, default=Path("I:/"),
                        help="图片根目录（含 01/ 03/ 子目录）")
    parser.add_argument("--dataset", default="local_reid_review")
    parser.add_argument("--export", choices=["confirmed"], default=None,
                        help="导出人工审核结果")
    parser.add_argument("--out", type=Path, default=base / "review" / "confirmed_individuals.csv")
    args = parser.parse_args()

    try:
        import fiftyone  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"缺少 fiftyone: pip install fiftyone ({e})") from e

    if args.export == "confirmed":
        export_confirmed(args.dataset, args.out)
    else:
        # 防丢失保护：数据集已存在且含人工审核结果时禁止重建覆盖
        try:
            import fiftyone as fo  # noqa: F401
            if fo.dataset_exists(args.dataset):
                ds = fo.load_dataset(args.dataset)
                reviewed = sum(1 for s in ds if s.confirmed_identity)
                if reviewed > 0:
                    raise SystemExit(
                        f"数据集 {args.dataset} 已有 {reviewed} 条人工审核结果，"
                        f"禁止重建覆盖。请先导出："
                        f"python scripts/fiftyone_review.py --export confirmed，"
                        f"或确认丢弃后手动删除数据集再重建。")
        except SystemExit:
            raise
        except Exception:
            pass  # fiftyone 不可用时按原逻辑走（构建时自然会报缺依赖）
        df = load_review_dataset(args.clusters, args.images_root)
        build_dataset(df, args.dataset)
