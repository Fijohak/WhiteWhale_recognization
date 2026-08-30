"""
背鳍检测与裁剪（YOLOv8）。

统一两处历史实现（scripts/detect_and_crop.py 的独立裁剪工具与
scripts/pipeline_archival.py 的 _detect_all）的检测+扩展裁剪逻辑：

- expand_box：非均匀扩展检测框（左右 0.30×框宽、上方 0.15×框高、下方
  0.60×框高），clamp 到图像内 + 尺寸上限；
- detect_and_crop：逐图检测 → 扩展 → 裁剪 → crops_manifest；
  未检出回退中心 0.45 窗（fallback=true，可追溯）。

语义：裁剪结果只作特征提取输入，不代表个体身份；所有结果可追溯到原图。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from whitewhale.data.image_store import ImageStore, validate_safe_image_ids
from whitewhale.data.manifest import compute_sha256

YOLO_FALLBACK_POLICY = "center_square_min_side_0.45"


def resolve_yolo_device(device: str | int | None) -> str | int | None:
    """把项目的 ``auto`` 设备值转换为 Ultralytics 的自动选择语义。"""
    if device is None:
        return None
    if isinstance(device, str) and device.strip().lower() in {"", "auto"}:
        return None
    return device


def expand_box(x0: float, y0: float, x1: float, y1: float,
               w: int, h: int, px: float, up: float, down: float):
    """非均匀扩展检测框：左右各 px×框宽，上方 up×框高，下方 down×框高。

    返回扩展后的 [x0, y0, w, h]（已 clamp 到图像内 + 尺寸上限）。
    """
    bw, bh = x1 - x0, y1 - y0
    pad_x, pad_up, pad_down = bw * px, bh * up, bh * down
    x0, y0 = max(0, int(x0 - pad_x)), max(0, int(y0 - pad_up))
    x1, y1 = min(w, int(x1 + pad_x)), min(h, int(y1 + pad_down))
    # 尺寸上限：边长 ≤ 0.9 × min(w,h)，面积 ≤ 60% 原图
    max_side = int(min(w, h) * 0.9)
    cw, ch = x1 - x0, y1 - y0
    if cw > max_side:  # 缩宽（居中收缩）
        dx = (cw - max_side) // 2
        x0, x1 = x0 + dx, x0 + dx + max_side
    if ch > max_side:  # 缩高（保持底部，收缩顶部）
        y0 = y1 - max_side
    return [x0, y0, x1 - x0, y1 - y0]


def center_fallback_box(w: int, h: int, ratio: float = 0.45) -> list[int]:
    """返回检测失败时的中心正方形回退框，与离线裁剪保持同一语义。"""
    side = max(1, int(min(w, h) * ratio))
    left, top = (w - side) // 2, (h - side) // 2
    return [left, top, side, side]


def yolo_crop_provenance(weights: Path, conf: float, imgsz: int,
                         pad_x: float, pad_up: float, pad_down: float) -> dict:
    """生成可核对的 YOLO 裁剪配置，供 embedding config 与查询端比较。"""
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"检测器权重不存在：{weights}")
    return {
        "crop": "yolo",
        "crop_schema_version": 1,
        "detector_checkpoint_file": str(weights.resolve()),
        "detector_checkpoint_sha256": compute_sha256(weights),
        "detector_conf": float(conf),
        "detector_imgsz": int(imgsz),
        "detector_pad_x": float(pad_x),
        "detector_pad_up": float(pad_up),
        "detector_pad_down": float(pad_down),
        "detector_fallback_policy": YOLO_FALLBACK_POLICY,
    }


def detect_and_crop(df: pd.DataFrame, images_root: Path, out_dir: Path,
                    weights: Path, conf: float = 0.25, imgsz: int = 1024,
                    device: str | int | None = "auto", pad_x: float = 0.30,
                    pad_up: float = 0.15, pad_down: float = 0.60,
                    preview: bool = True) -> pd.DataFrame:
    """逐图 YOLO 检测 + 非均匀扩展裁剪；未检出回退中心 0.45 窗。

    Args:
        df: 输入清单（image_id, relative_path[, session_id]）
        images_root: 原图根目录（只读）
        out_dir: 输出裁剪图目录（{image_id}.jpg + crops_manifest.csv + preview/）
        weights: 检测器权重（yolov8n_dorsalfin.pt）
        其余为检测/扩展参数。

    Returns:
        裁剪清单 DataFrame（image_id / relative_path / session_id /
        x / y / w / h / det_conf / fallback），行序与输入一致。
    """
    if "image_id" not in df.columns or "relative_path" not in df.columns:
        raise ValueError("检测清单必须包含 image_id 和 relative_path")
    validate_safe_image_ids(df["image_id"])
    store = ImageStore(images_root)
    sources = [store.resolve(path) for path in df["relative_path"]]
    missing = [
        (str(image_id), str(path))
        for image_id, path in zip(df["image_id"], sources)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"检测清单有 {len(missing)} 张原图不存在，已在检测前终止；"
            f"示例: image_id={missing[0][0]!r}, path={missing[0][1]}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    yolo_device = resolve_yolo_device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "detect_preview"
    if preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    for i, r in df.iterrows():
        src = store.resolve(r["relative_path"])
        img = store.open(r["relative_path"])
        w, h = img.size
        res = model.predict(str(src), conf=conf, imgsz=imgsz, device=yolo_device,
                            verbose=False)
        if len(res) and len(res[0].boxes):
            b = res[0].boxes[0]  # 最高置信度框（ultralytics 已按 conf 排序）
            x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
            conf_det = float(b.conf[0])
            box = expand_box(x0, y0, x1, y1, w, h, pad_x, pad_up, pad_down)
            fallback = False
            if preview:  # 检测预览（框叠加）
                from PIL import ImageDraw

                vis = img.copy()
                ImageDraw.Draw(vis).rectangle(
                    [box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    outline=(0, 220, 0), width=6)
                vis.save(preview_dir / f"{r['image_id']}.jpg")
        else:
            # 回退：中心裁剪（0.45 窗，与历史 crop_center.py 语义一致）
            box = center_fallback_box(w, h)
            conf_det, fallback = 0.0, True
            failures.append((r["image_id"], "未检出→中心裁剪回退"))
        crop = img.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
        crop.save(out_dir / f"{r['image_id']}.jpg")
        rows.append({
            "image_id": r["image_id"], "relative_path": r["relative_path"],
            "session_id": r.get("session_id", ""),
            "x": box[0], "y": box[1], "w": box[2], "h": box[3],
            "det_conf": round(conf_det, 4), "fallback": fallback,
        })
        if (i + 1) % 50 == 0:
            print(f"[detect] {i + 1}/{len(df)} 完成")

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "crops_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"[detect] 完成：{len(rows)} 张，检测失败回退 "
          f"{int(out['fallback'].sum()) if len(out) else 0} 张")
    for fid, reason in failures[:20]:
        print(f"[detect]   FAIL {fid}: {reason}")
    return out
