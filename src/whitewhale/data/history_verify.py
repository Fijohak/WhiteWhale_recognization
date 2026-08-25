"""
历史库核验回填（待办 3.6 步骤 4 自动化，操作手册 docs/history_verify_crossyear.md §一 1.5）。

核验汇总表（history_verify_summary.csv）→ 生成可信基准
history_verified_individuals.csv，并把 pilot_set.csv 中对应照片的
review_status 更新为 verified（改前自动备份）。

判定规则（手册 §一 1.5）：
- 结论 = "通过" 的组才登记；"需拆分" / "需复核" 不进可信基准；
- 结论与组内标注必须自洽：结论 "通过" ⇒ n_uncertain=0 且 n_reject=0，
  否则拒绝回填（防止错并——错并是种群统计低估的最危险错误）；
- 通过组名必须在 pilot_set.csv 中存在，防止组名笔误。

CLI 入口见 scripts/finalize_history_verify.py。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

# 手册 §1.5 定义的汇总表列
SUMMARY_COLUMNS = ["group", "n_images", "n_confirmed", "n_uncertain",
                   "n_reject", "结论"]
VALID_VERDICTS = {"通过", "需拆分", "需复核"}
VERDICT_PASS = "通过"


def load_summary(summary_csv: Path) -> pd.DataFrame:
    """读取核验汇总表，校验列与结论取值。"""
    if not summary_csv.exists():
        raise FileNotFoundError(f"核验汇总表不存在：{summary_csv}")
    df = pd.read_csv(summary_csv, dtype={"group": str})
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in SUMMARY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"汇总表缺少列：{missing}（应有 {SUMMARY_COLUMNS}）")
    bad = sorted(set(df["结论"].astype(str).str.strip())
                 - VALID_VERDICTS - {""})
    if bad:
        raise ValueError(f"结论取值非法（应为 通过/需拆分/需复核）：{bad}")
    df["group"] = df["group"].str.strip()
    return df


def check_consistency(df: pd.DataFrame) -> list[str]:
    """数据自洽检查：结论 '通过' 的组必须无不确定、无排除；数值列非负。

    返回问题清单（空 = 无问题）。
    """
    problems: list[str] = []
    for _, row in df.iterrows():
        verdict = str(row["结论"]).strip()
        group = row["group"]
        counts = {}
        for col in ("n_images", "n_confirmed", "n_uncertain", "n_reject"):
            value = row[col]
            if pd.isna(value) or int(value) < 0:
                problems.append(f"{group}: {col} 缺失或为负")
            else:
                counts[col] = int(value)
        # 总数自洽：n_confirmed + n_uncertain + n_reject == n_images
        if len(counts) == 4 and counts["n_confirmed"] + counts["n_uncertain"] \
                + counts["n_reject"] != counts["n_images"]:
            problems.append(
                f"{group}: 数量自洽校验失败（确认 {counts['n_confirmed']} + "
                f"不确定 {counts['n_uncertain']} + 排除 {counts['n_reject']} "
                f"≠ 总数 {counts['n_images']}），请核对汇总表")
        if verdict == VERDICT_PASS:
            if int(row["n_uncertain"]) != 0 or int(row["n_reject"]) != 0:
                problems.append(
                    f"{group}: 结论=通过 但 n_uncertain={row['n_uncertain']} "
                    f"n_reject={row['n_reject']}（自相矛盾，拒绝回填）")
    return problems


def mark_verified(summary_csv: Path, pilot_csv: Path, out_dir: Path,
                  verified_date: str | None = None,
                  backup: bool = True) -> dict:
    """按汇总表回填：生成可信基准表 + 更新 pilot_set 的 review_status。

    - 结论=通过的组 → history_verified_individuals.csv（可信基准）；
    - pilot_set.csv 中这些组对应的照片 review_status 置为 verified，
      修改前自动备份 pilot_set.csv；
    - 返回统计信息（组数 / 照片数 / 输出路径）。
    """
    summary = load_summary(summary_csv)
    problems = check_consistency(summary)
    if problems:
        raise ValueError("汇总表自洽检查未通过，已拒绝回填：\n  - "
                         + "\n  - ".join(problems))

    if not pilot_csv.exists():
        raise FileNotFoundError(f"pilot_set.csv 不存在：{pilot_csv}")

    verified_groups = summary[summary["结论"].str.strip() == VERDICT_PASS].copy()
    if verified_groups.empty:
        raise ValueError("没有结论=通过的组，无可回填内容（请先完成核验）")

    pilot = pd.read_csv(pilot_csv, dtype={"session_id": str})
    known_ids = set(pilot["individual_id"].astype(str))
    unknown = sorted(set(verified_groups["group"]) - known_ids)
    if unknown:
        raise ValueError(f"通过组在 pilot_set.csv 中不存在（组名笔误？）：{unknown}")

    # ---- 生成可信基准表 ----
    verified_date = verified_date or date.today().isoformat()
    benchmark = verified_groups[["group", "n_images"]].rename(
        columns={"group": "individual_id", "n_images": "n_images"})
    benchmark["verified_date"] = verified_date
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = out_dir / "history_verified_individuals.csv"
    benchmark.to_csv(benchmark_path, index=False, encoding="utf-8-sig")

    # ---- 更新 pilot_set（改前备份） ----
    backup_path = None
    if backup:
        backup_path = pilot_csv.with_name(
            f"pilot_set.csv.bak_{pd.Timestamp.now():%Y%m%d_%H%M%S}")
        backup_path.write_bytes(pilot_csv.read_bytes())
    pass_ids = set(verified_groups["group"])
    mask = pilot["individual_id"].astype(str).isin(pass_ids)
    n_updated = int(mask.sum())
    # 交叉验证：实际更新照片数与汇总表登记的 n_images 必须一致
    claimed = int(verified_groups["n_images"].sum())
    if n_updated != claimed:
        raise ValueError(
            f"回填数量不一致：pilot_set 中通过组实际 {n_updated} 张，"
            f"汇总表登记 {claimed} 张（组名或 n_images 填错？），已拒绝回填")
    pilot.loc[mask, "review_status"] = "verified"
    pilot.to_csv(pilot_csv, index=False, encoding="utf-8-sig")

    return {
        "verified_groups": int(len(benchmark)),
        "verified_images": n_updated,
        "benchmark_csv": str(benchmark_path),
        "pilot_backup": str(backup_path) if backup_path else None,
    }
