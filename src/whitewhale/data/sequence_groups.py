"""
散图连拍整串划归（待办 1.9 准备）：按拍摄序列给散图池分组。

目标：散图池（loose_known）按连拍号分组为"连拍串"，
后续任一帧匹配上某个体即可整串划归候选（一次确认一串，提升收集效率）。

文件名解析（README §2.3 已确认双模式）：
- RAY/DH/RZ 等拍摄：`编号_日期_地点_批次_[相机_]人员_连拍号.JPG`
  → 序列 key = 去掉首段编号与末段连拍号（如 `20140417_SZi_01_RAY`），
    帧号 = 末段连拍号（如 0024）；
- MO 拍摄：`RES20001.JPG`（无连拍信息，README A10）→ 不分组。

⚠️ 前提（README A9）：连拍号 = 连续拍摄序列仅为推测，须抽样核验后才
能用于数据划分。因此本脚本**只做分组与抽样材料输出，不执行任何划归**：
- 输出全部散图连拍串清单（sequence_groups.csv），同串切分假设：
  连拍号间隔 ≤ MAX_GAP 视为同串（默认 2，容忍中间删帧）；
- --sample N 输出抽样核验清单（含原图相对路径，人工对照文件名确认），
  核验通过（A9 成立）后清单才可用于划归。

注意：manifest 的 sequence_guess 将编号段并入 key（每张一 key），
对 RAY 格式不可用，故本模块自行解析文件名。
CLI 入口见 scripts/group_sequences.py。
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# RAY 格式：编号_序列key_连拍号（key 含日期/地点/批次/相机/人员）
RAY_NAME_PATTERN = re.compile(r"^(\d+)_(.+)_(\d+)$")
# 连拍号间隔超过此值视为新串（容忍中间删 1 帧）
MAX_GAP = 2
# 连拍串清单统一列（空结果也输出表头，保证下游可读）
SEQ_COLUMNS = ["sequence_id", "session_id", "n_frames", "image_ids",
               "filenames", "frame_numbers", "candidate_groups",
               "sequence_source"]


def load_loose_manifest(manifest_csv: Path) -> pd.DataFrame:
    """读取 manifest 并过滤散图（loose_known），session_id 保字符串。"""
    if not manifest_csv.exists():
        raise FileNotFoundError(f"manifest 不存在：{manifest_csv}")
    df = pd.read_csv(manifest_csv, dtype={"session_id": str})
    loose = df[df["label_status"] == "loose_known"].copy()
    return loose


def parse_ray_frame(filename: str) -> tuple[str, int] | None:
    """解析 RAY 格式文件名 → (序列 key, 连拍号)；无法解析返回 None。

    示例：`0155_20140418_SCi_01_RAY_0024.JPG` → ("20140418_SCi_01_RAY", 24)。
    MO 文件（RES20001.JPG 等）不匹配 → None。
    """
    stem = Path(filename).stem
    match = RAY_NAME_PATTERN.match(stem)
    if not match:
        return None
    return match.group(2), int(match.group(3))


def split_by_gap(frames: list[int], max_gap: int = MAX_GAP) -> list[list[int]]:
    """按连拍号间隔切串：间隔 > max_gap 处断开（返回分组下标序列）。

    空列表返回空；单元素返回 [[x]]。
    """
    if not frames:
        return []
    groups = []
    current = [frames[0]]
    for prev, cur in zip(frames, frames[1:]):
        if cur - prev > max_gap:
            groups.append(current)
            current = []
        current.append(cur)
    groups.append(current)
    return groups


def build_sequence_groups(loose: pd.DataFrame,
                          min_frames: int = 2) -> pd.DataFrame:
    """散图 → 连拍串清单（按文件名连拍号分组，单帧不成串）。

    返回每串一行：序列 key、帧数、帧 image_id / filename / 连拍号列表、
    候选个体并集、解析来源（filename_ray_frame / unparsed 不成串）。
    """
    if loose.empty:
        return pd.DataFrame(columns=SEQ_COLUMNS)
    rows = []
    by_key: dict[tuple[str, str], list] = {}
    for row in loose.itertuples():
        parsed = parse_ray_frame(row.filename)
        if parsed is None:
            continue  # MO 等无连拍信息，不分组
        key, frame = parsed
        by_key.setdefault((row.session_id, key), []).append((frame, row))
    for (session, key), items in sorted(by_key.items()):
        items.sort(key=lambda x: x[0])  # 按连拍号升序
        frames = [f for f, _ in items]
        for chunk in split_by_gap(frames):
            if len(chunk) < min_frames:
                continue
            chunk_items = [it[1] for it in items if it[0] in chunk]
            candidates = [g for r in chunk_items
                          for g in str(getattr(r, "candidate_groups", "")).split(";")
                          if g]
            rows.append({
                "sequence_id": f"{key}#{chunk[0]:04d}",
                "session_id": session,
                "n_frames": len(chunk),
                "image_ids": ";".join(r.image_id for r in chunk_items),
                "filenames": ";".join(r.filename for r in chunk_items),
                "frame_numbers": ";".join(f"{f:04d}" for f in chunk),
                "candidate_groups": "|".join(dict.fromkeys(candidates)),
                "sequence_source": "filename_ray_frame",
            })
    out = pd.DataFrame(rows, columns=SEQ_COLUMNS)
    if not out.empty:
        out = out.sort_values(["session_id", "sequence_id"]).reset_index(drop=True)
    return out


def sample_checklist(sequence_groups: pd.DataFrame, loose: pd.DataFrame,
                     n: int = 10) -> pd.DataFrame:
    """抽样核验清单（A9）：随机抽 n 串，列出每串所有帧的文件名与路径。

    人工打开原图对照文件名，确认连拍号确实对应连续拍摄的同一只海豚。
    返回逐帧明细（非逐串），便于直接看图核验。
    """
    if sequence_groups.empty:
        return pd.DataFrame(columns=[
            "sequence_id", "session_id", "filename", "relative_path",
            "frame_number", "sequence_source"])
    picked = sequence_groups.sample(min(n, len(sequence_groups)),
                                    random_state=0)  # 固定种子可复现
    picked_long = picked.assign(
        iid=picked["image_ids"].str.split(";")).explode("iid")
    frames = loose.merge(picked_long[["iid", "sequence_id"]],
                         left_on="image_id", right_on="iid", how="inner")
    frames["frame_number"] = frames["filename"].map(
        lambda f: (parse_ray_frame(f) or (None, None))[1])
    return frames[["sequence_id", "session_id", "filename",
                   "relative_path", "frame_number", "sequence_source"]]


def build_checklist_and_groups(manifest_csv: Path, out_dir: Path,
                               sample_n: int = 10) -> dict:
    """一站式：读 manifest → 分串清单 + 抽样核验清单，写 outputs/index/。"""
    loose = load_loose_manifest(manifest_csv)
    groups = build_sequence_groups(loose)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups_path = out_dir / "sequence_groups.csv"
    groups.to_csv(groups_path, index=False, encoding="utf-8-sig")

    sample_path = None
    if sample_n > 0:
        sample = sample_checklist(groups, loose, sample_n)
        sample_path = out_dir / "sequence_sample_checklist.csv"
        sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

    return {
        "loose_images": int(len(loose)),
        "sequence_count": int(len(groups)),
        "sequences_csv": str(groups_path),
        "sample_csv": str(sample_path) if sample_path else None,
        "note": "清单仅供 A9 抽样核验；核验通过前不得用于划归。",
    }
