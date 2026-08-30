"""
人工评估集自动划分（待办 3.2 草案版）。

目标：从已确认个体（confirmed_individuals.csv 的 confirmed 行）中
自动划分 query / gallery 草案，供人工确认后作为评估集。
规则（README §1 核心语义 + §2.6）：
- 最小划分单元 = 完整拍摄串（session + 文件名序列键 + 连续帧段），同串不跨 split；
- confirmed_identity 只在所属 session 内有效，未对齐的跨 session
  同名标签不视为同一个体；
- 同一 confirmed_identity 有 ≥2 个序列才可作 query（保证 gallery 有同体）；
  优先取图像数最多的序列，若多身份共享一串则按全局串单元调整；
- 只有 1 个序列的个体整进 gallery（无法作 query）；
- 输出带 quality_band / session_id，供人工核对"多个体 × 多角度"覆盖。

⚠️ 本输出是**草案**：按 Sequence 划分成立的前提 A9（文件名连拍号 = 连续
拍摄序列）须经抽样核验；人工确认草案后才算正式评估集。

CLI 入口见 scripts/build_eval_set.py。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from whitewhale.data.sequence_groups import annotate_series


SERIES_META_COLUMNS = ("series_id", "sequence_key", "frame")


def attach_stable_series_from_manifest(
        frame: pd.DataFrame, full_manifest: pd.DataFrame) -> pd.DataFrame:
    """在完整 manifest 上一次分串，再按 image_id 回填到子集。"""
    for name, value in (("frame", frame), ("full_manifest", full_manifest)):
        if "image_id" not in value.columns:
            raise ValueError(f"{name} 缺少 image_id")
        if value["image_id"].duplicated().any():
            raise ValueError(f"{name}.image_id 不唯一，无法安全回填 series")

    manifest = full_manifest.copy()
    annotate_series(manifest)
    manifest_ids = set(manifest["image_id"])
    missing = [value for value in frame["image_id"] if value not in manifest_ids]
    if missing:
        raise ValueError(
            f"{len(missing)} 张子集图像不在完整 manifest，无法确定稳定 series")

    mapping = manifest.set_index("image_id")
    out = frame.copy()
    if "session_id" in out.columns:
        expected_session = out["image_id"].map(mapping["session_id"]).astype(str)
        actual_session = out["session_id"].astype(str)
        if not actual_session.equals(expected_session):
            raise ValueError("frame.session_id 与完整 manifest 不一致")
    for column in SERIES_META_COLUMNS:
        out[column] = out["image_id"].map(mapping[column])
    return out


def _session_scoped_identity(session_id: object, identity: object) -> str:
    """生成便于阅读的展示 ID；内部分组不使用该字符串。"""
    if pd.isna(session_id) or not str(session_id).strip():
        raise ValueError("评估集划分缺少 session_id，无法确定个体命名空间")
    if pd.isna(identity) or not str(identity).strip():
        raise ValueError("评估集划分缺少 confirmed_identity")
    session = str(session_id).strip()
    raw_identity = str(identity).strip()
    prefix = f"{session}_"
    return raw_identity if raw_identity.startswith(prefix) else f"{prefix}{raw_identity}"


def _enforce_global_series_partition(df: pd.DataFrame) -> pd.DataFrame:
    """把同一完整串作为全局单元，并尽量为可评估身份保留 query。"""
    out = df.copy()
    parsed = out["series_id"].fillna("").astype(str).str.strip() != ""
    units = out["series_id"].fillna("").astype(str).copy()
    units.loc[~parsed] = "__single_" + out.loc[~parsed, "image_id"].astype(str)

    # 先撤回逐身份划分造成的冲突串，再以全局单元重选。
    query_units = set(units[out["split"] == "query"])
    gallery_units = set(units[out["split"] == "gallery"])
    conflicts = query_units & gallery_units
    if conflicts:
        out.loc[units.isin(conflicts), "split"] = "gallery"

    identity_groups = list(out.groupby("_identity_key", sort=False))
    identity_units = {
        identity: set(units.loc[grp.index].tolist())
        for identity, grp in identity_groups
    }
    unit_identities: dict[str, set[object]] = {}
    for identity, unit_set in identity_units.items():
        for unit in unit_set:
            unit_identities.setdefault(unit, set()).add(identity)

    query_units = set(units[out["split"] == "query"])
    for identity, grp in identity_groups:
        if identity_units[identity] & query_units:
            continue
        parsed_grp = grp[parsed.loc[grp.index]]
        sizes = units.loc[parsed_grp.index].groupby(
            units.loc[parsed_grp.index], sort=False).size()
        if len(sizes) < 2:
            continue
        candidates = sorted(
            sizes.index.tolist(), key=lambda value: (-int(sizes[value]), str(value)))
        for candidate in candidates:
            trial = query_units | {candidate}
            affected = unit_identities[candidate]
            if all(identity_units[value] - trial for value in affected):
                out.loc[units == candidate, "split"] = "query"
                query_units.add(candidate)
                break

    final_query = set(units[out["split"] == "query"])
    final_gallery = set(units[out["split"] == "gallery"])
    if not final_query.isdisjoint(final_gallery):
        raise AssertionError("series 划分泄漏：同一完整串同时出现在 query/gallery")
    query_ids = set(out.loc[out["split"] == "query", "_identity_key"])
    gallery_ids = set(out.loc[out["split"] == "gallery", "_identity_key"])
    if not query_ids.issubset(gallery_ids):
        raise AssertionError("query 中存在没有跨串 gallery 正样本的身份")
    return out


def load_confirmed_with_manifest(confirmed_csv: Path,
                                 manifest_csv: Path) -> pd.DataFrame:
    """读确认个体表（status=confirmed），按 image_id 对齐 manifest 字段。"""
    for csv in (confirmed_csv, manifest_csv):
        if not csv.exists():
            raise FileNotFoundError(f"文件不存在：{csv}")
    confirmed = pd.read_csv(
        confirmed_csv, dtype=str, keep_default_na=False)
    confirmed = confirmed[confirmed["status"] == "confirmed"].copy()
    confirmed = confirmed[
        confirmed["confirmed_identity"].astype(str).str.strip() != ""
    ].copy()
    if confirmed.empty:
        raise ValueError("确认个体表中没有 status=confirmed 的行，"
                         "请先完成人工审核（3.1）")
    manifest = pd.read_csv(manifest_csv, dtype=str, keep_default_na=False)
    keep = ["image_id", "session_id", "filename", "sequence_guess",
            "sequence_source", "quality_band", "relative_path"]
    # confirmed 表里的 session_id 是审核内部编号（如 "1"），以 manifest 为准
    confirmed = confirmed.drop(columns=["session_id"], errors="ignore")
    df = confirmed.merge(
        manifest[keep], on="image_id", how="left", validate="one_to_one")
    missing = df["session_id"].isna().sum()
    if missing:
        raise ValueError(f"{missing} 张确认照片在 manifest 中找不到"
                         f"（image_id 不匹配），请检查输入")
    df = attach_stable_series_from_manifest(df, manifest)
    df["series_source"] = df["series_id"].map(
        lambda value: "filename_ray_frame" if str(value).strip() else "unparsed")
    return df


def build_eval_split(df: pd.DataFrame,
                     full_manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    """按个体 × 序列划分 query/gallery，返回逐图 split 草案。

    无序列照片（MO 拍摄等无连拍号）无法按序列安全划分，一律进 gallery；
    有序列照片按序列划分：序列数 ≥2 才出 query（图像数最多的序列）。
    """
    if full_manifest is not None:
        df = attach_stable_series_from_manifest(df, full_manifest)
    elif "series_id" not in df.columns:
        raise ValueError(
            "缺少生成期稳定 series_id：请传入 full_manifest，"
            "禁止在评估子集上重新分串")
    if "series_source" not in df.columns:
        df["series_source"] = df["series_id"].map(
            lambda value: "filename_ray_frame" if str(value).strip() else "unparsed")
    df = df.copy()
    identity_keys = []
    display_ids = []
    for session, identity in zip(df["session_id"], df["confirmed_identity"]):
        display_ids.append(_session_scoped_identity(session, identity))
        identity_keys.append((str(session).strip(), str(identity).strip()))
    df["_identity_key"] = pd.Series(identity_keys, index=df.index, dtype=object)

    # 常规数据保持现有展示 ID；只对 startswith 旧规则造成的
    # 真实碰撞项使用长度前缀消歧，不让展示列丢失身份区分。
    display_to_keys: dict[str, set[tuple[str, str]]] = {}
    for display, key in zip(display_ids, identity_keys):
        display_to_keys.setdefault(display, set()).add(key)
    collided = {
        display for display, keys in display_to_keys.items() if len(keys) > 1
    }
    df["_individual_id"] = [
        (f"{len(session)}:{session}|{len(identity)}:{identity}"
         if display in collided else display)
        for display, (session, identity) in zip(display_ids, identity_keys)
    ]
    has_seq = df["series_id"].notna() & (df["series_id"].astype(str).str.strip() != "")
    rows = []
    for _, grp in df.groupby("_identity_key", sort=False):
        grp = grp.copy()
        grp["split"] = "gallery"  # 默认：无序列或不足 2 序列的照片
        with_seq = grp[has_seq[grp.index]]
        seq_sizes = with_seq.groupby("series_id").size()
        if len(seq_sizes) >= 2:
            query_seq = seq_sizes.idxmax()  # 图像数最多的序列作 query
            grp.loc[with_seq.index, "split"] = with_seq["series_id"].map(
                lambda s: "query" if s == query_seq else "gallery")
        rows.append(grp)
    out = pd.concat(rows, ignore_index=True)
    out = _enforce_global_series_partition(out)
    out = out[["image_id", "_individual_id", "session_id",
               "series_id", "series_source", "sequence_key", "frame",
               "sequence_guess", "sequence_source", "quality_band",
               "relative_path", "split"]].rename(
        columns={"_individual_id": "individual_id"})
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
        "note": "草案：按完整 series 划分（A9 核验通过前不得作正式评估集）；"
                "跨日期（多 session）同体需 3.6/3.8 完成后再补。",
    }
    stats_path = out_dir / "eval_set_draft_stats.json"
    stats_path.write_text(__import__("json").dumps(
        stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"draft_csv": str(draft_path), "stats_json": str(stats_path),
            **stats}
