"""
人工审核网页（面向领域专家，无需学习 FiftyOne）。

界面按候选簇（子簇单元）分组显示照片，每位审核人的三种操作即时写入
outputs/review/review_annotations.csv（可追溯、可恢复）：
- 个体名 + [确认]：归属某个体（如 CI-001，同一只海豚必须同名）；
- [不确定] / [排除]：特殊状态（uncertain / reject，不产出确认记录）。

数据语义（项目红线）：
- 簇号 = Candidate Cluster，审核确认后才叫个体；
- uncertain / reject 必须保留（不能强制分配）；
- 命名审核票相互隔离；正式导出仅接受人数达标且完全一致的结论，
  人数不足或任意冲突保留为 uncertain。

CLI 入口见 scripts/launch_review.py。
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unicodedata
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, cast

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from filelock import FileLock, Timeout as FileLockTimeout

from whitewhale.data.image_store import ImageStore
from whitewhale.reid.embedding import (
    load_verified_embedding_artifact,
    require_generated_artifact_provenance,
)

# 审核标注约定：confirmed = CI-xxx 个体名；uncertain / reject 为特殊状态
UNCERTAIN = "uncertain"
REJECT = "reject"
ANNOTATION_COLUMNS = ["image_id", "label", "reviewer", "reviewed_at"]
ANNOTATION_LOCK_TIMEOUT_SECONDS = 10.0
REVIEWER_ROSTER_SCHEMA_VERSION = 1
_RESERVED_REVIEWER_IDS = {"(legacy)", "consensus"}


def normalize_reviewer_id(reviewer: object) -> str:
    """生成审核人 canonical ID，用于唯一计票与持久化。"""
    if reviewer is None:
        return ""
    return unicodedata.normalize("NFKC", str(reviewer)).strip().casefold()


def _normalize_annotation_label(label: object) -> str:
    """保留个体名原文，但把大小写/全角形式的保留状态统一为 canonical 值。"""
    if label is None:
        return ""
    value = str(label).strip()
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return normalized if normalized in {UNCERTAIN, REJECT} else value


def normalize_reviewer_roster(reviewers: Iterable[object] | None,
                              *, min_reviewers: int = 1) -> tuple[str, ...]:
    """校验受控审核人名单，并返回按配置顺序排列的 canonical ID。"""
    if min_reviewers < 1:
        raise ValueError("min_reviewers 必须为正整数")
    if reviewers is None:
        raise ValueError("必须提供 reviewer roster，不能仅凭自由文本证明审核人身份")
    if isinstance(reviewers, (str, bytes)):
        reviewers = [reviewers]

    canonical = [normalize_reviewer_id(value) for value in reviewers]
    if any(not value for value in canonical):
        raise ValueError("reviewer roster 不能包含空审核人 ID")
    reserved = sorted(set(canonical) & _RESERVED_REVIEWER_IDS)
    if reserved:
        raise ValueError(f"reviewer roster 使用了保留 ID: {reserved}")
    if len(set(canonical)) != len(canonical):
        raise ValueError("reviewer roster 经 Unicode/大小写规范化后存在重复 ID")
    if len(canonical) < min_reviewers:
        raise ValueError(
            f"reviewer roster 仅有 {len(canonical)} 人，少于要求的 {min_reviewers} 人")
    return tuple(canonical)


def resolve_reviewer_id(reviewer: object,
                        reviewer_roster: Iterable[object] | None) -> str:
    """把 CLI 输入解析为 roster 内的 canonical ID，未登记者立即拒绝。"""
    canonical = normalize_reviewer_id(reviewer)
    if not canonical:
        raise ValueError("启动审核网页必须提供非空 --reviewer")
    roster = normalize_reviewer_roster(reviewer_roster)
    if canonical not in roster:
        raise ValueError(f"审核人 {reviewer!r} 不在 reviewer roster 中")
    return canonical


def _canonical_roster_binding(
    reviewer_roster: Iterable[object],
) -> tuple[str, ...]:
    """生成与 CLI 排列顺序无关的固定审核名单。"""
    return tuple(sorted(normalize_reviewer_roster(reviewer_roster)))


def _reviewer_roster_fingerprint(roster: Iterable[str]) -> str:
    """计算固定名单的可复核 SHA-256 指纹。"""
    encoded = json.dumps(
        list(roster), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reviewer_roster_sidecar(csv_path: Path) -> Path:
    """返回与原始票文件相邻的固定 reviewer roster 元数据路径。"""
    csv_path = Path(csv_path)
    return csv_path.with_name(csv_path.name + ".reviewer_roster.json")


def _ensure_reviewer_roster_locked(
    csv_path: Path,
    reviewer_roster: Iterable[object],
) -> tuple[str, ...]:
    """在 annotations 锁内初始化或校验跨运行固定的审核名单。"""
    roster = _canonical_roster_binding(reviewer_roster)
    fingerprint = _reviewer_roster_fingerprint(roster)
    sidecar = _reviewer_roster_sidecar(csv_path)
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"reviewer roster sidecar 无法读取: {sidecar}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"reviewer roster sidecar 格式无效: {sidecar}")
        saved_values = payload.get("canonical_reviewer_roster")
        if not isinstance(saved_values, list):
            raise ValueError(f"reviewer roster sidecar 缺少 canonical 名单: {sidecar}")
        try:
            saved = _canonical_roster_binding(saved_values)
        except ValueError as exc:
            raise ValueError(f"reviewer roster sidecar 名单无效: {sidecar}") from exc
        saved_fingerprint = payload.get("fingerprint_sha256")
        if (payload.get("schema_version") != REVIEWER_ROSTER_SCHEMA_VERSION
                or saved_values != list(saved)
                or saved_fingerprint != _reviewer_roster_fingerprint(saved)):
            raise ValueError(f"reviewer roster sidecar 内容或指纹无效: {sidecar}")
        if saved_fingerprint != fingerprint:
            raise ValueError(
                "reviewer roster 与 annotations 已固定名单不匹配；"
                f"固定名单为 {list(saved)}")
        return saved

    # 命名票的有效性依赖历史 roster；sidecar 丢失后不能从本次 CLI 参数猜回。
    # 仅空表和纯匿名旧表可以在第一次进入多人审核时建立固定名单。
    existing = load_annotation_records(csv_path)
    named_reviewers = sorted(set(
        existing.loc[existing["reviewer"] != "", "reviewer"].tolist()))
    if named_reviewers:
        raise ValueError(
            "reviewer roster sidecar 缺失，但 annotations 已包含命名审核票；"
            "拒绝根据本次 CLI 名单重建，请从可信备份恢复 sidecar 或执行显式迁移。"
            f"已发现 reviewer: {named_reviewers}")

    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REVIEWER_ROSTER_SCHEMA_VERSION,
        "canonical_reviewer_roster": list(roster),
        "fingerprint_sha256": fingerprint,
    }
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=".tmp", dir=sidecar.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, sidecar)
    finally:
        tmp.unlink(missing_ok=True)
    return roster


def _resolve_history_path(root: Path, *parts: str) -> Path | None:
    """解析历史对照路径并确保真实目标仍位于配置根目录内。"""
    try:
        resolved_root = root.resolve()
        candidate = resolved_root.joinpath(*map(str, parts)).resolve()
        candidate.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    return candidate


def _fmt(v):
    """数值格式化：空 / NaN → 空字符串，否则保留 4 位小数（供前端直接展示）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        s = str(v)
        return s if s not in ("", "nan") else ""
    return round(f, 4) if not np.isnan(f) else ""


def _int_col(v):
    """数字列解析：兼容 int/float/字符串（含 "1.0" 浮点写法）；解析失败 → -1（噪声/残噪语义）。"""
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return -1


def _validate_photos_frame(df: pd.DataFrame) -> pd.DataFrame:
    """校验审核清单主键与图片路径，避免模糊或错位审核。"""
    required = {"image_id", "relative_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"候选簇照片表缺少必需列: {sorted(missing)}")
    if df.empty:
        raise ValueError("候选簇照片表为空")

    out = df.copy()
    for column in sorted(required):
        values = out[column].fillna("").astype(str).str.strip()
        blank_rows = (values == "").to_numpy().nonzero()[0]
        if len(blank_rows):
            rows = [int(index) + 2 for index in blank_rows[:5]]
            raise ValueError(f"{column} 含空值（CSV 行 {rows}）")
        out[column] = values

    folded_ids = out["image_id"].map(str.casefold)
    duplicates = out.loc[folded_ids.duplicated(keep=False), "image_id"].tolist()
    if duplicates:
        raise ValueError(f"image_id 重复或仅大小写不同: {duplicates[:5]}")
    return out


def load_photos(clusters_csv: Path) -> pd.DataFrame:
    """以字符串安全模式加载并校验候选簇照片表。"""
    columns = pd.read_csv(clusters_csv, nrows=0, keep_default_na=False).columns
    required = {"image_id", "relative_path"}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"候选簇照片表缺少必需列: {sorted(missing)}")
    identifier_columns = {
        "image_id", "individual_id", "source_group", "session_id",
        "relative_path", "confirmed_identity",
    }
    df = pd.read_csv(
        clusters_csv, keep_default_na=False,
        dtype={column: str for column in columns if column in identifier_columns})
    df = _validate_photos_frame(df)
    for col in ("cluster", "individual_id", "source_group", "session_id",
                "quality_band", "relative_path"):
        if col not in df.columns:
            df[col] = ""
    # 子簇列（子簇化管线产物）：旧格式无此列 → 整簇视为一个子簇（0），噪声为 -1
    if "subcluster" not in df.columns:
        def _old_sc(x):
            try:
                return -1 if int(float(x)) == -1 else 0
            except (TypeError, ValueError):
                return -1
        df["subcluster"] = df["cluster"].apply(_old_sc)
    return df


def load_annotation_records(csv_path: Path) -> pd.DataFrame:
    """读取一人一票长表，并把旧版两列 CSV 迁移到内存中的统一格式。

    旧文件没有 ``reviewer`` / ``reviewed_at`` 列时，记为空审核人与空时间；
    数字形式的个体名按字符串读取，避免 ``0001`` 被破坏为 ``1``。
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    missing = {"image_id", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"标注文件缺少必需列: {sorted(missing)}")
    if "reviewer" not in df.columns:
        df["reviewer"] = ""
    if "reviewed_at" not in df.columns:
        df["reviewed_at"] = ""
    df = df[ANNOTATION_COLUMNS].copy()
    for col in ANNOTATION_COLUMNS:
        df[col] = df[col].fillna("").astype(str)
    for column in ("image_id", "label"):
        df[column] = df[column].str.strip()
        blank_rows = (df[column] == "").to_numpy().nonzero()[0]
        if len(blank_rows):
            rows = [int(index) + 2 for index in blank_rows[:5]]
            raise ValueError(f"标注文件 {column} 含空值（CSV 行 {rows}）")
    df["reviewer"] = df["reviewer"].map(normalize_reviewer_id)
    df["label"] = df["label"].map(_normalize_annotation_label)
    # 容忍手工合并或旧程序留下的重复行，同一人对同一图以最后一票为准。
    return df.drop_duplicates(["image_id", "reviewer"], keep="last").reset_index(drop=True)


def load_annotations(csv_path: Path, reviewer: str = "", *,
                     image_ids: Iterable[object] | None = None) -> dict[str, str]:
    """读取指定审核人的 ``{image_id: label}``，兼容旧版单人 CSV。"""
    records = load_annotation_records(csv_path)
    reviewer = normalize_reviewer_id(reviewer)
    own = records[records["reviewer"] == reviewer]
    if image_ids is not None:
        allowed = {str(image_id).strip() for image_id in image_ids}
        own = own[own["image_id"].isin(allowed)]
    return dict(zip(own["image_id"], own["label"]))


class _PersistentFileLock(FileLock):
    """保留稳定锁文件，只释放 ``filelock`` 获取的系统 advisory lock。"""

    def _fallback_to_soft_lock(self) -> None:
        """文件系统不支持 advisory lock 时拒绝降级为可被误删的软锁。"""
        raise OSError(
            f"当前文件系统不支持系统级 advisory lock: {self.lock_file}")

    def _release(self) -> None:
        # filelock 3.25 会在释放时删除锁文件，可能产生 unlink/recreate 竞态；
        # 获取、重试及平台适配仍复用 FileLock，仅覆盖其删除动作。
        fd = cast(int, self._context.lock_file_fd)
        self._context.lock_file_fd = None
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextmanager
def _annotation_file_lock(csv_path: Path):
    """用系统 advisory lock 串行化写入；锁随进程退出自动释放。"""
    lock_path = csv_path.with_name(csv_path.name + ".lock")
    lock = _PersistentFileLock(
        lock_path, timeout=ANNOTATION_LOCK_TIMEOUT_SECONDS)
    try:
        with lock.acquire():
            yield
    except FileLockTimeout as exc:
        raise TimeoutError(f"等待标注文件写入锁超时: {lock_path}") from exc


def load_embeddings(embeddings_path, meta_path):
    """严格加载生成期特征产物；未配置时才关闭相似度提示。"""
    if embeddings_path is None and meta_path is None:
        return None, {}
    if embeddings_path is None or meta_path is None:
        raise ValueError("相似度提示必须同时配置 embeddings 与 meta")

    emb, meta, config = load_verified_embedding_artifact(
        Path(embeddings_path), Path(meta_path))
    require_generated_artifact_provenance(config)
    if config.get("row_binding") != "embedding_row_i_to_meta_image_id_i":
        raise ValueError("相似度产物 row_binding 语义不受支持")
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    image_ids = meta["image_id"].astype(str).tolist()
    return emb, {image_id: index for index, image_id in enumerate(image_ids)}


def _validate_embedding_alignment(photos: pd.DataFrame, index: dict[str, int],
                                  *, artifact_name: str,
                                  exact: bool = False) -> None:
    """按 image_id 校验审核清单与提示特征，禁止位置猜测或静默缺行。"""
    photo_ids = set(photos["image_id"].astype(str))
    artifact_ids = set(index)
    missing = sorted(photo_ids - artifact_ids)
    extra = sorted(artifact_ids - photo_ids) if exact else []
    if missing or extra:
        details = []
        if missing:
            details.append(f"缺少审核图片 {missing[:5]}")
        if extra:
            details.append(f"含清单外图片 {extra[:5]}")
        raise ValueError(f"{artifact_name} 的 image_id 与审核清单未严格对齐："
                         + "；".join(details))


def save_annotations(csv_path: Path, photos: pd.DataFrame, ann: dict,
                     reviewer: str = "", *,
                     reviewer_roster: Iterable[object] | None = None,
                     replace_image_ids: Iterable[object] | None = None) -> None:
    """合并写回当前审核人的票，不覆盖其他审核人的原始判断。

    仍使用临时文件 + replace 原子替换，避免审核进程中断留下半个 CSV。
    命名审核人必须属于显式 roster；空审核人仅保留给旧数据迁移测试。
    """
    photos = _validate_photos_frame(photos)
    known_ids = set(photos["image_id"])
    target_ids = (known_ids if replace_image_ids is None else
                  {str(image_id).strip() for image_id in replace_image_ids})
    unknown_targets = sorted(target_ids - known_ids)
    if unknown_targets:
        raise ValueError(f"待更新票包含未知 image_id: {unknown_targets[:5]}")
    reviewer = normalize_reviewer_id(reviewer)
    fixed_roster = None
    if reviewer:
        fixed_roster = normalize_reviewer_roster(
            reviewer_roster, min_reviewers=3)
        reviewer = resolve_reviewer_id(reviewer, fixed_roster)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with _annotation_file_lock(csv_path):
        if reviewer:
            _ensure_reviewer_roster_locked(csv_path, fixed_roster)
        # 必须在获取锁后重新读取，才能合并其他进程刚写入的票。
        existing = load_annotation_records(csv_path)
        if not reviewer:
            has_named_votes = bool((existing["reviewer"] != "").any())
            if (_reviewer_roster_sidecar(csv_path).exists()
                    or has_named_votes):
                raise ValueError(
                    "匿名 reviewer 写入仅用于无 roster sidecar 且全为匿名票的"
                    " legacy 迁移；当前 annotations 已进入或曾进入命名审核流程")
        own_old = existing[existing["reviewer"] == reviewer]
        old_by_id = own_old.set_index("image_id").to_dict("index")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = []
        for image_id, label in ann.items():
            image_id = str(image_id).strip()
            label = _normalize_annotation_label(label)
            if not image_id or not label:
                continue
            if image_id not in known_ids:
                raise ValueError(f"标注包含未知 image_id: {image_id}")
            if image_id not in target_ids:
                continue
            previous = old_by_id.get(image_id, {})
            if previous and previous.get("label") == label:
                # 兼容旧两列文件：未改旧票继续保留空时间，不伪造历史审核时刻。
                reviewed_at = previous.get("reviewed_at", "")
            else:
                reviewed_at = now
            rows.append({
                "image_id": image_id,
                "label": label,
                "reviewer": reviewer,
                "reviewed_at": reviewed_at,
            })
        # 只替换当前清单内该审核人的票；共享 annotations 中其他批次原始票保留。
        other = existing[
            (existing["reviewer"] != reviewer)
            | (~existing["image_id"].isin(target_ids))
        ]
        current = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
        df = pd.concat([other, current], ignore_index=True)
        if not df.empty:
            df = df.sort_values(["reviewer", "image_id"], kind="stable")
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{csv_path.name}.", suffix=".tmp", dir=csv_path.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            df.to_csv(tmp, index=False, encoding="utf-8-sig",
                      columns=ANNOTATION_COLUMNS)
            os.replace(tmp, csv_path)
        finally:
            tmp.unlink(missing_ok=True)


def summarize_annotations(
    records: pd.DataFrame,
    min_reviewers: int = 3,
    *,
    reviewer_roster: Iterable[object] | None = None,
) -> pd.DataFrame:
    """按图片汇总多人独立票，产出可复核的保守裁决。

    达到 ``min_reviewers`` 且所有票完全一致时才产生 ``agreed`` 结论；
    人数不足标记 ``pending``，任何分歧标记 ``conflict``，两者的裁决均保留为 uncertain。
    ``vote_counts`` 和 ``reviewer_votes`` 保留每个结论的原始票据；
    旧版匿名票只供追溯，不与命名审核人合并为“两人一致”。
    """
    columns = [
        "image_id", "adjudicated_label", "adjudication_status",
        "n_reviewers", "min_reviewers_required", "max_vote_count", "agreement_ratio",
        "vote_counts", "eligible_vote_counts", "reviewer_votes",
        "eligible_reviewers", "ineligible_reviewers", "reviewer_roster",
    ]
    if min_reviewers < 3:
        raise ValueError("min_reviewers 必须至少为 3")
    roster = normalize_reviewer_roster(
        reviewer_roster, min_reviewers=min_reviewers)
    roster_set = set(roster)
    if records.empty:
        return pd.DataFrame(columns=columns)
    data = records.copy()
    for col in ("image_id", "label", "reviewer"):
        if col not in data.columns:
            raise ValueError(f"标注汇总缺少必需列: {col}")
        data[col] = data[col].fillna("").astype(str)
    data["reviewer"] = data["reviewer"].map(normalize_reviewer_id)
    data["image_id"] = data["image_id"].str.strip()
    data["label"] = data["label"].map(_normalize_annotation_label)
    data = data[(data["image_id"] != "") & (data["label"] != "")]
    data = data.drop_duplicates(["image_id", "reviewer"], keep="last")

    rows = []
    for image_id, group in data.groupby("image_id", sort=True):
        votes = {
            (reviewer if reviewer else "(legacy)"): label
            for reviewer, label in zip(group["reviewer"], group["label"])
        }
        # 旧版匿名票无法证明与命名审核人相互独立，不参与多人裁决。
        named = group[group["reviewer"] != ""]
        eligible = named[named["reviewer"].isin(roster_set)]
        ineligible = sorted(set(named["reviewer"]) - roster_set)
        all_counts = group["label"].value_counts().sort_index().to_dict()
        counts = eligible["label"].value_counts().sort_index().to_dict()
        n_reviewers = len(eligible)
        max_count = max(counts.values()) if counts else 0
        if not counts:
            status = "pending"
            adjudicated = UNCERTAIN
        elif len(counts) > 1:
            status = "conflict"
            adjudicated = UNCERTAIN
        elif n_reviewers < min_reviewers:
            status = "pending"
            adjudicated = UNCERTAIN
        else:
            status = "agreed"
            adjudicated = next(iter(counts))
        rows.append({
            "image_id": image_id,
            "adjudicated_label": adjudicated,
            "adjudication_status": status,
            "n_reviewers": n_reviewers,
            "min_reviewers_required": min_reviewers,
            "max_vote_count": max_count,
            "agreement_ratio": round(max_count / n_reviewers, 4) if n_reviewers else 0.0,
            "vote_counts": json.dumps(all_counts, ensure_ascii=False, sort_keys=True),
            "eligible_vote_counts": json.dumps(
                counts, ensure_ascii=False, sort_keys=True),
            "reviewer_votes": json.dumps(votes, ensure_ascii=False, sort_keys=True),
            "eligible_reviewers": json.dumps(
                sorted(eligible["reviewer"].tolist()), ensure_ascii=False),
            "ineligible_reviewers": json.dumps(ineligible, ensure_ascii=False),
            "reviewer_roster": json.dumps(list(roster), ensure_ascii=False),
        })
    return pd.DataFrame(rows, columns=columns)


def build_app(args, photos: pd.DataFrame | None = None) -> FastAPI:
    """构建审核应用（photos 可注入，便于测试）。"""
    photos = _validate_photos_frame(
        photos) if photos is not None else load_photos(args.clusters)
    min_reviewers = int(getattr(args, "min_reviewers", 3))
    roster = normalize_reviewer_roster(
        getattr(args, "reviewer_roster", None),
        min_reviewers=max(3, min_reviewers),
    )
    reviewer = resolve_reviewer_id(getattr(args, "reviewer", ""), roster)
    annotations_path = Path(args.annotations)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with _annotation_file_lock(annotations_path):
        roster = _ensure_reviewer_roster_locked(annotations_path, roster)
        ann: dict = load_annotations(
            annotations_path, reviewer=reviewer, image_ids=photos["image_id"])
    store = ImageStore(args.images_root)
    for rel_path in photos["relative_path"]:
        store.resolve(rel_path)
    # 历史库对照照片目录（跨时间审核用，按个体分文件夹）；未指定 → 不提供对照
    history_root = (Path(args.history_lookup).resolve()
                    if getattr(args, "history_lookup", None) else None)
    # 历史对照图质量表（filename → clear/low；未提供 → 全部默认清晰，低质图前端可隐藏）
    hist_quality: dict[str, str] = {}
    if getattr(args, "history_quality", None) and Path(args.history_quality).exists():
        try:
            hq = pd.read_csv(args.history_quality, dtype={"filename": str})
            hist_quality = dict(zip(hq["filename"], hq["quality"]))
        except Exception:
            hist_quality = {}
    # 批次特征库（簇内相似度辅助：把混簇中可疑的"离群者"沉底标红；可选）
    batch_emb, batch_idx = None, {}
    if getattr(args, "batch_embeddings", None) is not None:
        meta_p = Path(args.batch_embeddings).with_name(
            Path(args.batch_embeddings).stem + "_meta.csv")
        batch_emb, batch_idx = load_embeddings(args.batch_embeddings, meta_p)
        _validate_embedding_alignment(
            photos, batch_idx, artifact_name="批次相似度产物", exact=True)
    # 每张照片与其子簇内均值的相似度（低 = 可能是混入的别的个体）
    # 按 (cluster, subcluster) 分组：同一簇的不同子簇可能是不同个体，不能共用一个混均值
    in_sim: dict[str, float] = {}
    if batch_emb is not None:
        for (cl, sc), grp in photos.groupby(["cluster", "subcluster"]):
            if _int_col(cl) == -1 or _int_col(sc) == -1 or len(grp) < 2:
                continue
            rows = [(str(r["image_id"]), batch_idx[str(r["image_id"])])
                    for _, r in grp.iterrows()
                    if str(r["image_id"]) in batch_idx]
            if len(rows) < 2:
                continue
            ids, idxs = zip(*rows)
            sub = batch_emb[list(idxs)]
            sims = sub @ sub.mean(axis=0)
            for iid, s in zip(ids, sims):
                in_sim[iid] = round(float(s), 3)
    # 特征库（相似度辅助，可选：缺失时审核功能照常，只是没有相似提醒）
    emb, emb_idx = load_embeddings(
        getattr(args, "embeddings", None), getattr(args, "embeddings_meta", None))
    if emb is not None:
        _validate_embedding_alignment(
            photos, emb_idx, artifact_name="相似度提示产物")

    app = FastAPI(title="中华白海豚个体审核")

    def gkey_of(cl: int, sc: int) -> str:
        """审核单元分组键：噪声 → "noise"；子簇 → "簇号.子簇号"（残噪 -1 也在内）。"""
        return "noise" if cl == -1 else f"{cl}.{sc}"

    def photo_list(cluster_filter: str | None = None) -> list[dict]:
        # 已命名个体（相似度辅助的比对目标；ann 动态变化，每次现算）
        named = [(iid, lab) for iid, lab in ann.items()
                 if lab and lab not in (UNCERTAIN, REJECT) and iid in emb_idx]
        sim_all = None
        if emb is not None and named:
            sim_all = emb @ emb[[emb_idx[iid] for iid, _ in named]].T  # (N, M)
        items = []
        for _, r in photos.iterrows():
            iid = str(r["image_id"])
            cl = _int_col(r["cluster"])
            sc = _int_col(r["subcluster"])
            gk = gkey_of(cl, sc)
            label = ann.get(iid, "")
            if cluster_filter and cluster_filter != "all":
                if cluster_filter == "noise":
                    # 噪声池 + 残噪子簇（逐图处理的单元都归入"噪声"筛选）
                    if cl != -1 and sc != -1:
                        continue
                elif cluster_filter == "unreviewed":
                    if label:
                        continue
                elif cluster_filter == "reviewed":
                    if not label:
                        continue
                elif cluster_filter.isdigit():
                    # 旧式簇号筛选：匹配该簇全部单元（子簇 + 残噪）
                    if not gk.startswith(cluster_filter + "."):
                        continue
                elif gk != cluster_filter:
                    continue
            # 与已命名个体的最相似 Top-2（排除自身，低于 0.40 不提示）
            similar = []
            if sim_all is not None and iid in emb_idx:
                row = sim_all[emb_idx[iid]]
                for j in np.argsort(-row)[:2]:
                    if named[j][0] == iid:
                        continue
                    if float(row[j]) < 0.40:
                        continue
                    similar.append({"name": named[j][1],
                                    "score": round(float(row[j]), 2)})
            items.append({
                "image_id": iid,
                "cluster": cl,
                "subcluster": sc,
                "gkey": gk,
                "source_group": str(r.get("individual_id", "")),
                "session_id": str(r.get("session_id", "")),
                "quality_band": str(r.get("quality_band", "")),
                "label": label,
                "similar": similar,
                # 跨时间管线产物（pipeline_archival）带簇级匹配建议；旧格式无这些列 → 空
                "top1": "" if str(r.get("top1", "")) in ("", "nan") else str(r.get("top1", "")),
                "top1_score": _fmt(r.get("top1_score", "")),
                "vote1_ratio": _fmt(r.get("vote1_ratio", "")),
                "status": str(r.get("status", "")),
                "in_sim": in_sim.get(iid, ""),
            })
        return items

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = (Path(__file__).parent / "review_app.html").read_text(encoding="utf-8")
        return html

    @app.get("/api/state")
    def state(filter: str = "all"):
        """照片列表（?filter=all|noise|unreviewed|reviewed|单元键如 1.0 / 旧式簇号 1）。"""
        from collections import Counter
        cnt = Counter(gkey_of(_int_col(r["cluster"]), _int_col(r["subcluster"]))
                      for _, r in photos.iterrows())
        labels = {k: v for k, v in ann.items() if v}
        # 名字 → 使用位置（与筛选无关；供前端跨单元同名警告，filter 视图下依然有效）
        gk_by_id = {str(r["image_id"]): gkey_of(_int_col(r["cluster"]), _int_col(r["subcluster"]))
                    for _, r in photos.iterrows()}
        name_locations: dict[str, dict] = {}
        for iid, lab in labels.items():
            if lab in (UNCERTAIN, REJECT) or iid not in gk_by_id:
                continue
            if lab in name_locations:
                name_locations[lab]["count"] += 1
            else:
                name_locations[lab] = {"key": gk_by_id[iid], "count": 1}
        return {
            "n_total": len(photos),
            "n_reviewed": len(labels),
            "reviewer": reviewer,
            "clusters": {str(k): v for k, v in sorted(cnt.items())},
            "photos": photo_list(filter),
            "name_locations": name_locations,
            "has_history": history_root is not None,
        }

    @app.post("/api/annotate")
    async def annotate(request: Request):
        """提交标注：{image_id, action: confirm|uncertain|reject|clear, identity}。"""
        body = await request.json()
        image_id = str(body.get("image_id", ""))
        if image_id not in set(photos["image_id"]):
            return JSONResponse({"error": f"未知 image_id: {image_id}"}, status_code=400)
        action = body.get("action", "")
        updated_ann = dict(ann)
        if action == "confirm":
            identity = body.get("identity")
            ident = "" if identity is None else str(identity).strip()
            if not ident:
                return JSONResponse({"error": "确认需填写个体名"}, status_code=400)
            if unicodedata.normalize("NFKC", ident).casefold() in {UNCERTAIN, REJECT}:
                return JSONResponse(
                    {"error": "个体名不能使用 uncertain/reject 保留状态"},
                    status_code=400)
            updated_ann[image_id] = ident
        elif action == "uncertain":
            updated_ann[image_id] = UNCERTAIN
        elif action == "reject":
            updated_ann[image_id] = REJECT
        elif action == "clear":
            updated_ann.pop(image_id, None)
        else:
            return JSONResponse({"error": f"未知操作: {action}"}, status_code=400)
        save_annotations(
            args.annotations, photos, updated_ann, reviewer=reviewer,
            reviewer_roster=roster, replace_image_ids=[image_id])
        ann.clear()
        ann.update(updated_ann)
        return {"image_id": image_id, "label": ann.get(image_id, ""),
                "n_reviewed": len([v for v in ann.values() if v])}

    @app.post("/api/annotate_batch")
    async def annotate_batch(request: Request):
        """整簇批量标注：{image_ids, action, identity} —— 簇级审核一次操作全簇成员。"""
        body = await request.json()
        ids = [str(x) for x in body.get("image_ids", [])]
        known = set(photos["image_id"])
        unknown = [i for i in ids if i not in known]
        if unknown:
            return JSONResponse({"error": f"未知 image_id: {unknown[:3]}"}, status_code=400)
        action = body.get("action", "")
        updated_ann = dict(ann)
        if action == "confirm":
            identity = body.get("identity")
            ident = "" if identity is None else str(identity).strip()
            if not ident:
                return JSONResponse({"error": "确认需填写个体名"}, status_code=400)
            if unicodedata.normalize("NFKC", ident).casefold() in {UNCERTAIN, REJECT}:
                return JSONResponse(
                    {"error": "个体名不能使用 uncertain/reject 保留状态"},
                    status_code=400)
            for iid in ids:
                updated_ann[iid] = ident
        elif action == "uncertain":
            for iid in ids:
                updated_ann[iid] = UNCERTAIN
        elif action == "reject":
            for iid in ids:
                updated_ann[iid] = REJECT
        elif action == "clear":
            for iid in ids:
                updated_ann.pop(iid, None)
        else:
            return JSONResponse({"error": f"未知操作: {action}"}, status_code=400)
        save_annotations(
            args.annotations, photos, updated_ann, reviewer=reviewer,
            reviewer_roster=roster, replace_image_ids=ids)
        ann.clear()
        ann.update(updated_ann)
        return {"n_reviewed": len([v for v in ann.values() if v]),
                "labels": {i: ann.get(i, "") for i in ids}}

    @app.get("/api/history/{individual_id}")
    def history(individual_id: str):
        """历史个体对照照片清单（history_lookup/<个体>/ 目录内图片；未配置目录 → 空）。"""
        if not history_root:
            return {"photos": []}
        d = _resolve_history_path(history_root, individual_id)
        if d is None:
            return Response(status_code=400)
        if not d.is_dir():
            return {"photos": []}
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        photos_list = []
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                from urllib.parse import quote
                photos_list.append({
                    "name": f.name,
                    "url": f"/api/history_photo/{quote(individual_id)}/{quote(f.name)}",
                    "quality": hist_quality.get(f.name, ""),
                })
        return {"photos": photos_list}

    @app.get("/api/history_photo/{individual_id}/{name}")
    def history_photo(individual_id: str, name: str):
        """历史个体对照照片原图。"""
        if not history_root:
            return Response(status_code=404)
        p = _resolve_history_path(history_root, individual_id, name)
        if p is None:
            return Response(status_code=400)
        if not p.is_file():
            return Response(status_code=404)
        ext = p.suffix.lower().lstrip(".")
        media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                 "bmp": "image/bmp"}.get(ext, "image/jpeg")
        return Response(p.read_bytes(), media_type=media)

    @app.get("/api/image/{image_id}")
    def image_file(image_id: str, full: int = 0):
        """缩略图（宽 ≤ 480px）或原图（?full=1，审核放大看背鳍细节，不压缩）。"""
        hit = photos[photos["image_id"] == image_id]
        if hit.empty:
            return Response(status_code=404)
        rel = str(hit.iloc[0]["relative_path"])
        if not store.exists(rel):
            return Response(content=f"图片不存在: {store.resolve(rel)}", status_code=404)
        data = store.read_bytes(rel) if full else store.thumbnail(rel, 480, 82)
        # 原图按真实扩展名给媒体类型（多数为 JPG）
        ext = Path(rel).suffix.lower().lstrip(".")
        media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                 "bmp": "image/bmp"}.get(ext, "image/jpeg")
        return Response(data, media_type=media)

    return app


def _write_csv_transaction(outputs: list[tuple[Path, pd.DataFrame]]) -> None:
    """先完成全部 staging，再替换目标；任一替换失败则恢复旧文件。"""
    if not outputs:
        return
    targets = [Path(target) for target, _ in outputs]
    resolved = [os.path.normcase(str(target.resolve())) for target in targets]
    if len(set(resolved)) != len(resolved):
        raise ValueError("同一导出事务中不能重复指定输出路径")

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for target, frame in outputs:
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
            os.close(fd)
            tmp_path = Path(tmp_name)
            staged[target] = tmp_path
            frame.to_csv(tmp_path, index=False, encoding="utf-8-sig")

        for target in targets:
            if target.exists():
                backup = target.with_name(
                    f".{target.name}.{uuid.uuid4().hex}.bak")
                os.replace(target, backup)
                backups[target] = backup

        for target in targets:
            os.replace(staged[target], target)
            installed.append(target)
    except Exception as exc:
        rollback_errors = []
        for target in reversed(installed):
            try:
                target.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        for target, backup in reversed(list(backups.items())):
            try:
                os.replace(backup, target)
            except OSError as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        if rollback_errors:
            raise RuntimeError(
                "审核导出失败且旧文件恢复不完整；保留 .bak 文件供人工恢复: "
                + "；".join(rollback_errors)) from exc
        raise
    finally:
        for tmp_path in staged.values():
            tmp_path.unlink(missing_ok=True)

    for backup in backups.values():
        try:
            backup.unlink(missing_ok=True)
        except OSError:
            # 新文件已经完整提交；保留备份比误报失败后让操作者重试更安全。
            pass


@contextmanager
def _output_file_locks(paths: Iterable[Path]):
    """按稳定顺序锁住所有累计输出，避免共享任一目标时并发丢更新。"""
    unique = {
        os.path.normcase(str(Path(path).resolve())): Path(path)
        for path in paths
    }
    ordered = [unique[key] for key in sorted(unique)]
    for path in ordered:
        path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in ordered:
            stack.enter_context(_annotation_file_lock(path))
        yield


def _load_existing_table(path: Path, *, table_name: str) -> pd.DataFrame:
    """以字符串安全模式读取累计表，并拒绝无法安全合并的主键。"""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "image_id" not in existing.columns:
        raise ValueError(f"既有{table_name}缺少 image_id，拒绝覆盖: {path}")
    existing = existing.copy()
    existing["image_id"] = existing["image_id"].astype(str).str.strip()
    if (existing["image_id"] == "").any():
        raise ValueError(f"既有{table_name}含空 image_id，拒绝覆盖: {path}")
    folded = existing["image_id"].map(str.casefold)
    if folded.duplicated().any():
        raise ValueError(f"既有{table_name}含重复 image_id，拒绝覆盖: {path}")
    return existing


def _merge_cumulative_table(existing: pd.DataFrame, current: pd.DataFrame,
                            *, replace_ids: set[str],
                            preferred_columns: list[str]) -> pd.DataFrame:
    """按 image_id 替换当前批次行，同时保留其他批次的累计记录。"""
    if existing.empty:
        merged = current.copy()
    else:
        kept = existing[~existing["image_id"].isin(replace_ids)].copy()
        merged = pd.concat([kept, current], ignore_index=True, sort=False)
    columns = preferred_columns + [
        column for column in merged.columns if column not in preferred_columns]
    merged = merged.reindex(columns=columns).fillna("")
    if not merged.empty:
        folded = merged["image_id"].astype(str).map(str.casefold)
        if folded.duplicated().any():
            raise ValueError("累计表合并后 image_id 发生大小写碰撞，拒绝写入")
        merged = merged.sort_values("image_id", kind="stable").reset_index(drop=True)
    return merged


def export_confirmed(args) -> None:
    """导出审核结果与多人票据汇总。

    只导出达到 ``min_reviewers``（至少 3）且完全一致的命名审核人裁决；
    旧匿名票仅供追溯，不能绕过共识形成 confirmed。
    """
    photos = load_photos(args.clusters)
    store = ImageStore(args.images_root)
    for rel_path in photos["relative_path"]:
        store.resolve(rel_path)
    min_reviewers = int(getattr(args, "min_reviewers", 3))
    roster = normalize_reviewer_roster(
        getattr(args, "reviewer_roster", None),
        min_reviewers=max(3, min_reviewers),
    )
    known_ids = set(photos["image_id"])
    out_columns = [
        "image_id", "confirmed_identity", "status", "cluster", "subcluster",
        "source_group", "session_id", "source_path", "reviewer",
        "consensus_reviewers",
    ]
    annotations_path = Path(args.annotations)
    out_path = Path(args.out)
    summary_out = getattr(args, "summary_out", None)
    summary_path = Path(summary_out) if summary_out is not None else None
    output_paths = [out_path] + ([summary_path] if summary_path is not None else [])
    normalized_outputs = {
        os.path.normcase(str(path.resolve())) for path in output_paths}
    if len(normalized_outputs) != len(output_paths):
        raise ValueError("确认库与投票汇总不能写入同一路径")
    if os.path.normcase(str(annotations_path.resolve())) in normalized_outputs:
        raise ValueError("原始票文件不能与确认库或投票汇总共用路径")

    # 锁住累计输出后再锁原始票；从票据快照到两文件提交均处在同一临界区。
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with _output_file_locks(output_paths):
        with _annotation_file_lock(annotations_path):
            fixed_roster = _ensure_reviewer_roster_locked(
                annotations_path, roster)
            records = load_annotation_records(annotations_path)
            # annotations 是累计原始票表；本次裁决只取当前清单。
            records = records[
                records["image_id"].isin(known_ids)].reset_index(drop=True)
            summary = summarize_annotations(
                records, min_reviewers=min_reviewers,
                reviewer_roster=fixed_roster)

            agreed = summary[summary["adjudication_status"] == "agreed"]
            ann = dict(zip(
                agreed["image_id"], agreed["adjudicated_label"]))
            consensus_reviewers = dict(zip(
                agreed["image_id"], agreed["eligible_reviewers"]))
            rows = []
            for _, row in photos.iterrows():
                image_id = str(row["image_id"])
                label = ann.get(image_id, "")
                if not label or label in (UNCERTAIN, REJECT):
                    continue
                rows.append({
                    "image_id": image_id,
                    "confirmed_identity": label,
                    "status": "confirmed",
                    "cluster": row.get("cluster", ""),
                    "subcluster": row.get("subcluster", ""),
                    "source_group": row.get("individual_id", ""),
                    "session_id": row.get("session_id", ""),
                    "source_path": row.get("relative_path", ""),
                    "reviewer": "consensus",
                    "consensus_reviewers": consensus_reviewers[image_id],
                })
            current_out = pd.DataFrame(rows, columns=out_columns)
            existing_out = _load_existing_table(
                out_path, table_name="确认库")
            out = _merge_cumulative_table(
                existing_out, current_out,
                replace_ids=known_ids,
                preferred_columns=out_columns,
            )
            outputs = [(out_path, out)]
            if summary_path is not None:
                existing_summary = _load_existing_table(
                    summary_path, table_name="投票汇总")
                cumulative_summary = _merge_cumulative_table(
                    existing_summary, summary,
                    replace_ids=known_ids,
                    preferred_columns=summary.columns.tolist(),
                )
                outputs.append((summary_path, cumulative_summary))
            _write_csv_transaction(outputs)

    if summary_out is not None:
        print(f"[review] 投票汇总累计 {len(cumulative_summary)} 条 → {summary_out}")
    n_id = out["confirmed_identity"].nunique() if len(out) else 0
    print(f"[review] 本批新增/更新 {len(current_out)} 条；确认库共 {len(out)} 条 → {args.out}")
    print(f"[review] 个体数: {n_id}，个体: {sorted(out['confirmed_identity'].unique()) if n_id else '无'}")
    # 按“已有任意票”统计覆盖率；匿名旧票也只计审计覆盖，不计有效审核人数。
    total = len(photos)
    known_summary = summary[summary["image_id"].isin(set(photos["image_id"].astype(str)))]
    done = len(known_summary)
    progress_labels = known_summary["adjudicated_label"].tolist()
    print(f"[review] 进度: 已审 {done}/{total}"
          f"（未审 {total - done}，uncertain {progress_labels.count(UNCERTAIN)}，"
          f"reject {progress_labels.count(REJECT)}）")
