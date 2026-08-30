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
- 旧核验包中的数值化组名可通过 pilot 同目录的
  individual_id_migration_v1_to_v2.csv 自动映射为保留源标签的 canonical ID。

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
MIGRATION_COLUMNS = ["individual_id_legacy", "individual_id_canonical"]


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
    duplicate_groups = sorted(df.loc[df["group"].duplicated(keep=False), "group"].unique())
    if duplicate_groups:
        problems.append(f"group 重复，无法确定唯一组级结论：{duplicate_groups}")
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


def load_identity_migration(migration_csv: Path) -> dict[str, str]:
    """读取一对一旧 ID → canonical ID 映射，拒绝含糊或空白映射。"""
    migration = pd.read_csv(migration_csv, dtype=str, keep_default_na=False)
    missing = [column for column in MIGRATION_COLUMNS if column not in migration.columns]
    if missing:
        raise ValueError(f"身份迁移表缺少列：{missing}（应有 {MIGRATION_COLUMNS}）")
    migration = migration[MIGRATION_COLUMNS].apply(lambda column: column.str.strip())
    if (migration == "").any().any():
        raise ValueError("身份迁移表含空白 legacy/canonical ID，拒绝自动映射")
    if migration["individual_id_legacy"].duplicated().any():
        raise ValueError("身份迁移表存在重复 legacy ID，拒绝含糊映射")
    if migration["individual_id_canonical"].duplicated().any():
        raise ValueError("身份迁移表不是一对一映射，拒绝自动合并身份")
    return dict(zip(migration["individual_id_legacy"],
                    migration["individual_id_canonical"]))


def mark_verified(summary_csv: Path, pilot_csv: Path, out_dir: Path,
                  verified_date: str | None = None,
                  backup: bool = True,
                  migration_csv: Path | None = None) -> dict:
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

    pilot = pd.read_csv(pilot_csv, dtype=str, keep_default_na=False)
    if "individual_id" not in pilot.columns:
        raise ValueError("pilot_set.csv 缺少 individual_id")
    known_ids = set(pilot["individual_id"].astype(str))

    # 历史核验包在 v1 中曾把 05 读成 5.0。只接受显式的一对一迁移表，
    # 不靠 float/补零猜测组名；canonical 组名本身保持不变。
    migration_path = migration_csv or pilot_csv.with_name(
        "individual_id_migration_v1_to_v2.csv")
    migrated_groups = 0
    if migration_path.exists():
        mapping = load_identity_migration(migration_path)
        original = verified_groups["group"].copy()
        verified_groups["group"] = verified_groups["group"].map(
            lambda group: mapping.get(group, group))
        migrated_groups = int((original != verified_groups["group"]).sum())
    elif migration_csv is not None:
        raise FileNotFoundError(f"身份迁移表不存在：{migration_path}")

    duplicate_after_mapping = sorted(verified_groups.loc[
        verified_groups["group"].duplicated(keep=False), "group"].unique())
    if duplicate_after_mapping:
        raise ValueError(
            f"身份迁移后出现重复通过组，拒绝重复计数：{duplicate_after_mapping}")
    unknown = sorted(set(verified_groups["group"]) - known_ids)
    if unknown:
        raise ValueError(f"通过组在 pilot_set.csv 中不存在（组名笔误？）：{unknown}")

    # 所有一致性检查必须先于任何输出写入，避免拒绝回填时留下貌似有效的基准表。
    pass_ids = set(verified_groups["group"])
    mask = pilot["individual_id"].astype(str).isin(pass_ids)
    n_updated = int(mask.sum())
    claimed = int(verified_groups["n_images"].sum())
    if n_updated != claimed:
        raise ValueError(
            f"回填数量不一致：pilot_set 中通过组实际 {n_updated} 张，"
            f"汇总表登记 {claimed} 张（组名或 n_images 填错？），已拒绝回填")

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
    pilot.loc[mask, "review_status"] = "verified"
    pilot.to_csv(pilot_csv, index=False, encoding="utf-8-sig")

    return {
        "verified_groups": int(len(benchmark)),
        "verified_images": n_updated,
        "migrated_groups": migrated_groups,
        "identity_migration_csv": str(migration_path) if migration_path.exists() else None,
        "benchmark_csv": str(benchmark_path),
        "pilot_backup": str(backup_path) if backup_path else None,
    }
