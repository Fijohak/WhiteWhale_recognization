# -*- coding: utf-8 -*-
"""
scan_dataset.py — 中华白海豚数据集扫描与 Manifest 生成

功能：
- 支持多个数据根目录（如 I:\\01、I:\\03），跳过 ZIP / 视频 / 非图片文件；
- 解析调查批次(session)、评分区间(quality_band)、候选分组(group_id)；
- 依据 70-79 / 80 and above 子文件夹生成个体标签（labeled）；
- 70-79 散图标记为 loose_known（属于本调查已分组个体之一，归属未知）；
- 低于 70 分及 MO/RAY/miscellaneous 等目录标记为 ignored；
- 计算 SHA-256（可选）、读取尺寸、记录损坏文件；
- 输出 dataset_manifest.csv / dataset_stats.json / dataset_tree.txt / unreadable_files.csv。

数据语义依据（2026-08-07 确认）：
- I:\\01 与 I:\\03 是同一天(20140417)的两群海豚，是两个独立调查批次；
- 80 and above / 70-79 下的子文件夹 = 已分好的白海豚个体，可作为标签；
- 每个个体目前基本只有一张照片（极端少样本）；
- 70-79 散图属于本调查已分组个体之一，但具体归属未确认（candidate-label）。

用法：
    python scripts/scan_dataset.py --data-roots "I:\\01" "I:\\03" --output-dir outputs/index
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

# 常用图片扩展名（只扫描这些格式）
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
}

# 用于训练的质量区间（子文件夹 = 个体）；用归一化后的规范名
TRAIN_QUALITY_BANDS = {"70_79", "80_and_above"}

# 评分区间别名归一化
QUALITY_ALIASES = {
    "80 and above": "80_and_above",
    "80+": "80_and_above",
    "70-79": "70_79",
    "60-69": "60_69",
    "50-59": "50_59",
    "below 50": "below_50",
}

# 拍摄者代码目录（非个体标签来源）
PHOTOGRAPHER_DIRS = {"MO", "RAY", "DEREK"}

# 文件名中形如 20140417 的日期（用于 date_guess）
DATE_PATTERN = re.compile(r"(?<!\d)(20\d{6}|19\d{6})(?!\d)")

# 文件名连拍号推测：NNNN_YYYYMMDD_XXX_XX_摄影师_连拍号.JPG 取最后数字段
SEQUENCE_PATTERN = re.compile(r"^(.+)_(\d+)$")


def normalize_text(value: str) -> str:
    """归一化目录名用于别名匹配（小写、压缩空白）。"""
    return re.sub(r"\s+", " ", value.strip().lower())


def stable_image_id(relative_path: Path) -> str:
    """由相对路径生成稳定的项目内图片编号（与路径绑定，可重复）。"""
    digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:16]
    return f"IMG_{digest}"


def extract_date(*values: str) -> str:
    """从路径或文件名中提取 YYYYMMDD 日期，返回 YYYY-MM-DD；找不到返回空串。"""
    for value in values:
        match = DATE_PATTERN.search(value)
        if match:
            raw = match.group(1)
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return ""


def detect_quality(parts: tuple[str, ...]) -> str:
    """在路径片段中查找评分区间（归一化别名），返回规范名或 unknown。"""
    for part in parts:
        normalized = normalize_text(part)
        if normalized in QUALITY_ALIASES:
            return QUALITY_ALIASES[normalized]
    return "unknown"


def guess_sequence_id(filename: str, session_guess: str, source_group: str) -> tuple[str, str, str]:
    """
    尝试从文件名推测连续拍摄序列。
    规则（启发式）：如果文件名去掉扩展名后以 '_数字' 结尾，把最后一段视为帧号，
    前面部分作为序列 key；否则退回父目录级分组。
    """
    stem = Path(filename).stem
    match = SEQUENCE_PATTERN.match(stem)
    if match:
        sequence = f"{session_guess}::{match.group(1)}"
        return sequence, "filename_trailing_number", "medium"
    fallback = f"{session_guess}::{source_group}"
    return fallback, "parent_folder_fallback", "low"


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """分块计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[int | None, int | None, str, str]:
    """读取图片尺寸与格式；损坏/无法识别时返回 read_status 异常原因。"""
    try:
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or path.suffix.lstrip(".").upper()
            image.verify()
        return width, height, image_format, "readable"
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as error:
        return None, None, "", f"unreadable:{type(error).__name__}"


def build_tree_preview(root: Path, max_depth: int = 4, max_entries_per_dir: int = 30) -> str:
    """生成缩略目录树文本（只读预览，用于人工核对目录语义）。"""
    lines = [f"{root.name}/"]

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if depth >= max_depth:
            return
        try:
            entries = sorted(
                directory.iterdir(),
                key=lambda item: (not item.is_dir(), item.name.lower()),
            )
        except OSError:
            lines.append(f"{prefix}└── [无法读取]")
            return
        shown = entries[:max_entries_per_dir]
        for index, entry in enumerate(shown):
            is_last = index == len(shown) - 1 and len(entries) <= len(shown)
            connector = "└── " if is_last else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")
            if entry.is_dir():
                child_prefix = prefix + ("    " if is_last else "│   ")
                walk(entry, child_prefix, depth + 1)
        hidden_count = len(entries) - len(shown)
        if hidden_count > 0:
            lines.append(f"{prefix}└── ... 另有 {hidden_count} 项")

    walk(root, "", 0)
    return "\n".join(lines)


def classify_path(dir_parts: tuple[str, ...], root_name: str) -> dict[str, Any]:
    """
    核心：根据路径层级判定标签状态与标签值。
    - session_guess 取数据根目录名（如 01、03），不使用相对路径第一层
    - 出现 TRAIN_QUALITY_BANDS 区间：区间之后若还有目录层 → 子文件夹 = 个体 labeled；
      若区间下直接是文件（无目录层）→ loose_known（散图，归属待确认）
    - 其他区间 / MO / RAY / DEREK / miscellaneous / nn relationship → ignored
    注意：dir_parts 只包含目录部分，不含文件名。
    """
    result: dict[str, Any] = {
        "quality_band": "unknown",
        "group_id": "",
        "label": "",
        "label_status": "unknown",
        "relation_note": "",
        "session_guess": root_name,
    }

    # 在目录路径中寻找训练区间，确定标签层级
    for idx, part in enumerate(dir_parts):
        normalized = normalize_text(part)
        if normalized in QUALITY_ALIASES:
            band = QUALITY_ALIASES[normalized]
            result["quality_band"] = band
            if band in TRAIN_QUALITY_BANDS:
                # 该区间之后若还有目录层 → 子文件夹 = 个体
                rest = dir_parts[idx + 1:]
                if rest:
                    group_dir = rest[0]
                    # 规范化 group_id：去掉 "（sj of 01)" 等括号备注，只留编号部分
                    group_id = re.sub(r"\s*[（(].*?[)）]\s*$", "", group_dir).strip()
                    result["group_id"] = group_id
                    result["label"] = f"{root_name}_{group_id}"
                    result["label_status"] = "labeled"
                    # 记录关系备注（如 “sj of 01”）
                    if "sj of" in normalize_text(group_dir):
                        result["relation_note"] = "sj_of"
                else:
                    # 区间下直接是文件 → 散图（属于本调查已分组个体之一，归属未知）
                    result["label_status"] = "loose_known"
            else:
                result["label_status"] = "ignored"
            return result

    # 未命中评分区间：MO/RAY/DEREK/miscellaneous/nn relationship 等目录
    result["label_status"] = "ignored"
    return result


def collect_band_group_ids(root: Path) -> dict[str, dict[str, list[str]]]:
    """
    预扫描每个数据根下训练区间（70-79 / 80 and above）的直接子文件夹名。
    返回 {root_name: {band: [group_id, ...]}}，
    例如 {"01": {"70_79": ["05", "11"], "80_and_above": ["01", ...]}}。
    散图的候选集合只取同一调查、同一评分区间内的分组。
    """
    result: dict[str, dict[str, list[str]]] = {}
    if not root.is_dir():
        return result
    for band_dir in root.iterdir():
        if not band_dir.is_dir():
            continue
        # 先转规范名（"70-79" → "70_79"），再判断是否训练区间
        band_name = QUALITY_ALIASES.get(normalize_text(band_dir.name))
        if band_name in TRAIN_QUALITY_BANDS:
            groups = sorted(
                entry.name
                for entry in band_dir.iterdir()
                if entry.is_dir()
            )
            result.setdefault(root.name, {}).setdefault(band_name, []).extend(groups)
    return result


def scan_dataset(data_roots: list[Path], output_dir: Path, include_sha256: bool) -> None:
    """扫描所有数据根目录，汇总生成 Manifest 与统计信息。"""
    for root in data_roots:
        if not root.exists():
            raise FileNotFoundError(f"数据目录不存在：{root}")
        if not root.is_dir():
            raise NotADirectoryError(f"路径不是目录：{root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    unreadable_rows: list[dict[str, str]] = []

    # 预扫描各根的候选个体分组列表（散图候选集合用）
    band_group_ids: dict[str, list[str]] = {}
    for root in data_roots:
        band_group_ids.update(collect_band_group_ids(root))

    for root in data_roots:
        all_files = sorted(path for path in root.rglob("*") if path.is_file())
        for path in all_files:
            extension = path.suffix.lower()
            if extension not in IMAGE_EXTENSIONS:
                continue

            relative_path = path.relative_to(root)
            parts = relative_path.parts

            session_guess = root.name
            source_group = path.parent.name or "unknown"
            date_guess = extract_date(path.name, relative_path.as_posix(), session_guess)

            sequence_guess, sequence_source, sequence_confidence = guess_sequence_id(
                filename=path.name,
                session_guess=session_guess,
                source_group=source_group,
            )

            width, height, image_format, read_status = inspect_image(path)

            sha256 = ""
            if include_sha256:
                try:
                    sha256 = compute_sha256(path)
                except OSError:
                    read_status = read_status if read_status != "readable" else "hash_failed"

            # classify_path 只接收目录部分（不含文件名），避免散图文件名被误判为 group_id
            dir_parts = relative_path.parent.parts
            # session 取数据根目录名（如 01、03），保证跨调查同名分组不合并
            meta = classify_path(dir_parts, root.name)

            # 散图候选集合：同一调查、同一评分区间内的全部个体标签
            candidate_groups = ""
            if meta["label_status"] == "loose_known":
                band_groups = band_group_ids.get(root.name, {}).get(meta["quality_band"], [])
                candidate_groups = ";".join(
                    f"{root.name}_{g}" for g in band_groups
                )

            row = {
                "image_id": stable_image_id(relative_path),
                "relative_path": relative_path.as_posix(),
                "filename": path.name,
                "extension": extension,
                "file_size_bytes": path.stat().st_size,
                "session_guess": session_guess,
                "session_id": session_guess if session_guess in {"01", "03"} else "",
                "date_guess": date_guess,
                "quality_band": meta["quality_band"],
                "source_group": source_group,
                "group_id": meta["group_id"],
                "label": meta["label"],
                "label_status": meta["label_status"],
                "relation_note": meta["relation_note"],
                "candidate_groups": candidate_groups,
                "sequence_guess": sequence_guess,
                "sequence_source": sequence_source,
                "sequence_confidence": sequence_confidence,
                "width": width if width is not None else "",
                "height": height if height is not None else "",
                "image_format": image_format,
                "read_status": read_status,
                "sha256": sha256,
                "side": "unknown",
                "crop_status": "pending",
                "review_status": "unreviewed",
                "candidate_cluster": "",
                "confirmed_identity": "",
            }
            rows.append(row)

            if read_status != "readable":
                unreadable_rows.append({
                    "relative_path": relative_path.as_posix(),
                    "read_status": read_status,
                })

    # ---- 写 Manifest ----
    fieldnames = [
        "image_id", "relative_path", "filename", "extension", "file_size_bytes",
        "session_guess", "session_id", "date_guess", "quality_band", "source_group",
        "group_id", "label", "label_status", "relation_note", "candidate_groups",
        "sequence_guess", "sequence_source", "sequence_confidence",
        "width", "height", "image_format", "read_status", "sha256",
        "side", "crop_status", "review_status", "candidate_cluster", "confirmed_identity",
    ]
    manifest_path = output_dir / "dataset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---- 写异常文件表 ----
    unreadable_path = output_dir / "unreadable_files.csv"
    with unreadable_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["relative_path", "read_status"])
        writer.writeheader()
        writer.writerows(unreadable_rows)

    # ---- 统计 ----
    session_counter = Counter(row["session_guess"] for row in rows)
    quality_counter = Counter(row["quality_band"] for row in rows)
    label_counter = Counter(row["label_status"] for row in rows)
    extension_counter = Counter(row["extension"] for row in rows)
    read_counter = Counter(row["read_status"] for row in rows)

    # 每个个体的图片数（labeled）
    group_to_images: dict[str, int] = Counter(
        row["label"] for row in rows if row["label_status"] == "labeled"
    )
    # 每个个体出现的调查批次（跨调查同名编号核查用）
    label_to_sessions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["label_status"] == "labeled" and row["label"]:
            label_to_sessions[row["label"]].add(row["session_guess"])

    duplicate_groups = 0
    duplicate_images = 0
    if include_sha256:
        hash_counter = Counter(row["sha256"] for row in rows if row["sha256"])
        duplicate_hashes = {d: c for d, c in hash_counter.items() if c > 1}
        duplicate_groups = len(duplicate_hashes)
        duplicate_images = sum(duplicate_hashes.values())

    stats = {
        "data_roots": [str(r.resolve()) for r in data_roots],
        "total_images": len(rows),
        "readable_images": read_counter.get("readable", 0),
        "unreadable_or_problematic_images": len(unreadable_rows),
        "session_count": len(session_counter),
        "label_status_distribution": dict(label_counter.most_common()),
        "labeled_individual_count": len(group_to_images),
        "labeled_individuals": {
            "single_image": sum(1 for n in group_to_images.values() if n == 1),
            "multi_image": sum(1 for n in group_to_images.values() if n > 1),
            "max_images_per_individual": max(group_to_images.values(), default=0),
        },
        "cross_session_label_overlap": {
            label: sorted(sessions)
            for label, sessions in label_to_sessions.items()
            if len(sessions) > 1
        },
        "extensions": dict(extension_counter.most_common()),
        "quality_distribution": dict(quality_counter.most_common()),
        "session_distribution": dict(session_counter.most_common()),
        "read_status_distribution": dict(read_counter.most_common()),
        "sha256_enabled": include_sha256,
        "exact_duplicate_groups": duplicate_groups,
        "images_in_exact_duplicate_groups": duplicate_images,
        "warnings": [
            "session_guess、quality_band、group_id 均由路径解析；跨调查同名编号未核验。",
            "loose_known 散图属于本调查已分组个体之一，但具体归属未确认（candidate-label）。",
            "sequence_guess 必须经过抽样检查后才能用于数据划分。",
        ],
    }

    stats_path = output_dir / "dataset_stats.json"
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    # ---- 写目录树预览（合并所有根）----
    tree_lines = []
    for root in data_roots:
        tree_lines.append(build_tree_preview(root))
    tree_path = output_dir / "dataset_tree.txt"
    tree_path.write_text("\n\n".join(tree_lines), encoding="utf-8")

    print(f"扫描完成：{len(rows)} 张图片")
    print(f"  labeled（已分组个体）: {label_counter['labeled']}")
    print(f"  loose_known（70-79 散图，归属待确认）: {label_counter['loose_known']}")
    print(f"  ignored（低于70分/拍摄者目录等）: {label_counter['ignored']}")
    print(f"Manifest：{manifest_path.resolve()}")
    print(f"统计信息：{stats_path.resolve()}")
    print(f"目录预览：{tree_path.resolve()}")
    print(f"异常文件：{unreadable_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描中华白海豚数据集并建立 Manifest。")
    parser.add_argument(
        "--data-roots", nargs="+", type=Path, required=True,
        help="数据根目录列表，如 I:\\01 I:\\03（多个根各自独立解析）。",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/index"),
        help="索引结果输出目录。",
    )
    parser.add_argument(
        "--sha256", action="store_true",
        help="计算 SHA-256 以检测完全重复文件；数据很大时会增加耗时。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scan_dataset(
        data_roots=args.data_roots,
        output_dir=args.output_dir,
        include_sha256=args.sha256,
    )
