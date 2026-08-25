"""
人工评估集自动划分（待办 3.2 草案版）。

目标：从已确认个体（confirmed_individuals.csv 的 confirmed 行）中
自动划分 query / gallery 草案，供人工确认后作为评估集。
规则（README §1 核心语义 + §2.6）：
- 最小划分单元 = 拍摄序列（sequence_guess），同序列照片不跨 split（防泄漏）；
- 同一 confirmed_identity 有 ≥2 个序列才可作 query（保证 gallery 有同体）；
  取图像数最多的序列作 query，其余进 gallery；
- 只有 1 个序列的个体整进 gallery（无法作 query）；
- 输出带 quality_band / session_id，供人工核对"多个体 × 多角度"覆盖。

⚠️ 本输出是**草案**：按 Sequence 划分成立的前提 A9（文件名连拍号 = 连续
拍摄序列）须经抽样核验；人工确认草案后才算正式评估集。

CLI 入口见 scripts/build_eval_set.py。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_confirmed_with_manifest(confirmed_csv: Path,
                                 manifest_csv: Path) -> pd.DataFrame:
    """读确认个体表（status=confirmed），按 image_id 对齐 manifest 字段。"""
    for csv in (confirmed_csv, manifest_csv):
        if not csv.exists():
            raise FileNotFoundError(f"文件不存在：{csv}")
    confirmed = pd.read_csv(confirmed_csv, dtype={"session_id": str,
                                                  "confirmed_identity": str})
    confirmed = confirmed[confirmed["status"] == "confirmed"].copy()
    confirmed = confirmed[confirmed["confirmed_identity"].notna()].copy()
    if confirmed.empty:
        raise ValueError("确认个体表中没有 status=confirmed 的行，"
                         "请先完成人工审核（3.1）")
    manifest = pd.read_csv(manifest_csv, dtype={"session_id": str})
    keep = ["image_id", "session_id", "sequence_guess", "sequence_source",
            "quality_band", "relative_path"]
    # confirmed 表里的 session_id 是审核内部编号（如 "1"），以 manifest 为准
    confirmed = confirmed.drop(columns=["session_id"], errors="ignore")
    df = confirmed.merge(manifest[keep], on="image_id", how="left")
    missing = df["sequence_guess"].isna().sum()
    if missing:
        raise ValueError(f"{missing} 张确认照片在 manifest 中找不到"
                         f"（image_id 不匹配），请检查输入")
    return df


def build_eval_split(df: pd.DataFrame) -> pd.DataFrame:
    """按个体 × 序列划分 query/gallery，返回逐图 split 草案。

    无序列照片（MO 拍摄等无连拍号）无法按序列安全划分，一律进 gallery；
    有序列照片按序列划分：序列数 ≥2 才出 query（图像数最多的序列）。
    """
    has_seq = df["sequence_guess"].notna() & (df["sequence_guess"] != "")
    rows = []
    for iid, grp in df.groupby("confirmed_identity", sort=False):
        grp = grp.copy()
        grp["split"] = "gallery"  # 默认：无序列或不足 2 序列的照片
        with_seq = grp[has_seq[grp.index]]
        seq_sizes = with_seq.groupby("sequence_guess").size()
        if len(seq_sizes) >= 2:
            query_seq = seq_sizes.idxmax()  # 图像数最多的序列作 query
            grp.loc[with_seq.index, "split"] = with_seq["sequence_guess"].map(
                lambda s: "query" if s == query_seq else "gallery")
        rows.append(grp)
    out = pd.concat(rows, ignore_index=True)
    out = out[["image_id", "confirmed_identity", "session_id",
               "sequence_guess", "sequence_source", "quality_band",
               "relative_path", "split"]].rename(
        columns={"confirmed_identity": "individual_id"})
    return out.sort_values(["individual_id", "split"]).reset_index(drop=True)


def build_draft(confirmed_csv: Path, manifest_csv: Path,
                out_dir: Path) -> dict:
    """一站式：确认表 + manifest → 评估集草案 CSV + 统计。"""
    df = load_confirmed_with_manifest(confirmed_csv, manifest_csv)
    draft = build_eval_split(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    draft_path = out_dir / "eval_set_draft.csv"
    draft.to_csv(draft_path, index=False, encoding="utf-8-sig")

    n_individuals = draft["individual_id"].nunique()
    sessions_per_individual = draft.groupby("individual_id")["session_id"].nunique()
    stats = {
        "n_images": int(len(draft)),
        "n_individuals": int(n_individuals),
        "n_query": int((draft["split"] == "query").sum()),
        "n_gallery": int((draft["split"] == "gallery").sum()),
        "n_individuals_with_query": int(
            draft[draft["split"] == "query"]["individual_id"].nunique()),
        "multi_session_individuals": int((sessions_per_individual > 1).sum()),
        "sessions": sorted(draft["session_id"].dropna().unique().tolist()),
        "note": "草案：按 Sequence 划分（A9 核验通过前不得作正式评估集）；"
                "跨日期（多 session）同体需 3.6/3.8 完成后再补。",
    }
    stats_path = out_dir / "eval_set_draft_stats.json"
    stats_path.write_text(__import__("json").dumps(
        stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"draft_csv": str(draft_path), "stats_json": str(stats_path),
            **stats}
