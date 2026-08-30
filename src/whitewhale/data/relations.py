"""
确认关系表导出（待办 3.3）。

关系语义（README §1 + §2.6）：
- confirmed_same    人工确认同一只（组内任意两张，整组确认过）；
- confirmed_different 人工确认不同个体——当前审核流程只有"整组确认/不确定/
  排除"，无显式跨体确认源，表结构就绪、数据为空；
- possibly_same     疑似同一只待核验——可靠来源待 3.8 跨年匹配人工审核
  （默认落在 possibly_same 关系），当前数据为空。

三张表统一列：image_id_a / image_id_b / relation / individual_id /
source_group_a / source_group_b / session_id_a / session_id_b / source。
source = 数据来源描述（确认个体表 / 跨年匹配审核，可追溯）。

CLI 入口见 scripts/export_relations.py。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RELATION_COLUMNS = ["image_id_a", "image_id_b", "relation",
                    "individual_id", "source_group_a", "source_group_b",
                    "session_id_a", "session_id_b", "source"]
RELATION_FILES = {
    "confirmed_same": "relations_confirmed_same.csv",
    "confirmed_different": "relations_confirmed_different.csv",
    "possibly_same": "relations_possibly_same.csv",
}


def load_confirmed(confirmed_csv: Path) -> pd.DataFrame:
    """读人工确认个体表（仅 status=confirmed 且有确认身份的行）。"""
    if not confirmed_csv.exists():
        raise FileNotFoundError(f"确认个体表不存在：{confirmed_csv}")
    columns = pd.read_csv(confirmed_csv, nrows=0).columns
    identifier_columns = {
        "image_id", "session_id", "confirmed_identity", "source_group", "status",
    }
    df = pd.read_csv(
        confirmed_csv,
        dtype={column: str for column in columns if column in identifier_columns},
        keep_default_na=False)
    df = df[df["status"] == "confirmed"].copy()
    df = df[df["confirmed_identity"].notna()
            & (df["confirmed_identity"] != "")].copy()
    required = {"image_id", "session_id", "source_group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"确认个体表缺少追溯列: {sorted(missing)}")
    image_ids = df["image_id"].astype(str)
    if ((image_ids.str.strip() == "").any()
            or image_ids.duplicated().any()):
        raise ValueError("确认个体表 image_id 为空或重复")
    sessions = df["session_id"].fillna("").astype(str).str.strip()
    if (sessions == "").any():
        raise ValueError("确认个体表存在空 session_id，无法建立批次内关系")
    df["confirmed_identity"] = df["confirmed_identity"].astype(str).str.strip()
    df["session_id"] = sessions
    return df


def build_relations(confirmed_csv: Path,
                    out_dir: Path) -> dict[str, str]:
    """导出三张关系表（confirmed_different / possibly_same 无数据源，仅表头）。"""
    confirmed = load_confirmed(confirmed_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- confirmed_same：同 confirmed_identity 内任意两两（整组已确认） ----
    parts = []
    # identity 是不透明字符串；分组键直接用 (session, identity)，不靠前缀猜测。
    for (_, identity), grp in confirmed.groupby(
            ["session_id", "confirmed_identity"], sort=False):
        ordered = grp.sort_values("image_id", kind="stable").reset_index(drop=True)
        pair_rows = []
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                left, right = ordered.iloc[i], ordered.iloc[j]
                pair_rows.append({
                    "image_id_a": str(left["image_id"]),
                    "image_id_b": str(right["image_id"]),
                    "relation": "confirmed_same",
                    "individual_id": identity,
                    "source_group_a": str(left["source_group"]),
                    "source_group_b": str(right["source_group"]),
                    "session_id_a": str(left["session_id"]),
                    "session_id_b": str(right["session_id"]),
                    "source": "confirmed_individuals.csv",
                })
        if pair_rows:
            parts.append(pd.DataFrame(pair_rows, columns=RELATION_COLUMNS))
    same = pd.concat(parts, ignore_index=True) if parts else \
        pd.DataFrame(columns=RELATION_COLUMNS)
    same_path = out_dir / RELATION_FILES["confirmed_same"]
    same.to_csv(same_path, index=False, encoding="utf-8-sig")

    # ---- 另外两张：结构就绪，无可靠数据源（见模块 docstring） ----
    for key in ("confirmed_different", "possibly_same"):
        path = out_dir / RELATION_FILES[key]
        pd.DataFrame(columns=RELATION_COLUMNS).to_csv(
            path, index=False, encoding="utf-8-sig")

    # 说明文件：为什么这两张为空 + 数据源何时会有
    note = {
        "confirmed_same": {"n_pairs": int(len(same)),
                           "source": "confirmed_individuals.csv 同 identity 内两两",
                           "note": "整组人工确认过 → 组内任意两张 = 确认同体"},
        "confirmed_different": {
            "n_pairs": 0,
            "source": "无（当前审核只有整组确认/不确定/排除，无显式跨体确认）",
            "note": "不同 identity 只是'未确认相同'，不等于'确认不同'；"
                    "待审核流程支持显式跨体判定后填充"},
        "possibly_same": {
            "n_pairs": 0,
            "source": "无（待 3.8 跨年匹配人工审核）",
            "note": "3.8 审核结果默认落在 possibly_same（README §2.6），"
                    "经多人确认后才升级为 confirmed_same"},
    }
    note_path = out_dir / "relations_note.json"
    note_path.write_text(json.dumps(note, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return {key: str(out_dir / RELATION_FILES[key]) for key in RELATION_FILES} \
        | {"note_json": str(note_path)}
