"""
统一图片目录访问入口。

所有模块不再各自拼接 images_root/relative_path，而是通过 get_image_store()
获取 ImageStore 实例统一访问原图目录：

- resolve：相对路径 → 绝对路径（含越界防护）；
- open / read_bytes：读图；
- thumbnail：压缩缩略图（进程内缓存，审核/查询网页共用）。

数据根目录只从 configs/pipeline.yaml 读取一次（load_config("pipeline") 的
data_root），任何入口只需修改该配置即可切换数据目录。
"""
from __future__ import annotations

import io
import re
from collections import OrderedDict
from pathlib import Path, PureWindowsPath
from typing import Iterable

from PIL import Image, ImageOps

# 进程内有界 LRU 缩略图缓存；key 必须包含数据根，避免多盘同相对路径串图。
_THUMB_CACHE_MAX_ITEMS = 256
_THUMB_CACHE: OrderedDict[tuple[str, str, int, int], bytes] = OrderedDict()

_SAFE_IMAGE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z", re.ASCII)
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def validate_safe_image_ids(values: Iterable[object]) -> None:
    """校验 image_id 可安全作为跨平台文件 basename，且不会覆盖碰撞。"""
    seen: dict[str, str] = {}
    for position, raw in enumerate(values):
        missing = raw is None
        try:
            missing = missing or bool(raw != raw)  # NaN
        except (TypeError, ValueError):
            missing = missing or str(raw) in {"<NA>", "NaT"}
        if missing:
            raise ValueError(f"image_id 不能为空（第 {position + 1} 行）")

        image_id = str(raw)
        if not image_id:
            raise ValueError(f"image_id 不能为空（第 {position + 1} 行）")
        if not _SAFE_IMAGE_ID.fullmatch(image_id):
            raise ValueError(
                f"不安全的 image_id {image_id!r}（第 {position + 1} 行）；"
                "只允许 ASCII 字母、数字、下划线和连字符，且不能作为路径")
        if image_id.upper() in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"image_id {image_id!r} 是 Windows 保留名")
        key = image_id.casefold()
        if key in seen:
            raise ValueError(
                f"image_id 重复或在 Windows 上发生大小写碰撞: "
                f"{seen[key]!r} / {image_id!r}")
        seen[key] = image_id


class ImageStore:
    """原图目录访问入口：相对路径 → 绝对路径 / PIL / 缩略图字节。"""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def resolve(self, rel_path: str) -> Path:
        """相对路径 → 绝对路径（含越界防护）。

        越界（如 ../ 逃逸出数据根）直接报错，防止清单污染读任意文件。
        """
        root_abs = self.root.resolve()
        relative = Path(rel_path)
        windows_path = PureWindowsPath(rel_path)
        if relative.anchor or windows_path.drive or windows_path.root:
            raise ValueError(f"路径越界: {rel_path} 不是数据根内的相对路径")
        p = (root_abs / relative).resolve()
        try:
            p.relative_to(root_abs)
        except ValueError:
            raise ValueError(f"路径越界: {rel_path} 超出数据根 {self.root}")
        return p

    def exists(self, rel_path: str) -> bool:
        return self.resolve(rel_path).exists()

    def open(self, rel_path: str) -> Image.Image:
        """打开图片并转 RGB（统一预处理）。"""
        return Image.open(self.resolve(rel_path)).convert("RGB")

    def read_bytes(self, rel_path: str) -> bytes:
        return self.resolve(rel_path).read_bytes()

    def thumbnail(self, rel_path: str, max_w: int = 480,
                  quality: int = 82) -> bytes:
        """压缩缩略图（带缓存），审核/查询网页共用。"""
        key = (str(self.root.resolve()), str(rel_path), int(max_w), int(quality))
        hit = _THUMB_CACHE.get(key)
        if hit is not None:
            _THUMB_CACHE.move_to_end(key)
            return hit
        img = self.open(rel_path)
        img = ImageOps.exif_transpose(img)
        if img.width > max_w:
            img = img.resize(
                (max_w, int(img.height * max_w / img.width)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        data = buf.getvalue()
        _THUMB_CACHE[key] = data
        _THUMB_CACHE.move_to_end(key)
        while len(_THUMB_CACHE) > _THUMB_CACHE_MAX_ITEMS:
            _THUMB_CACHE.popitem(last=False)
        return data


def get_image_store(root: Path | str | None = None) -> ImageStore:
    """全局入口：root 缺省时自动读 configs/pipeline.yaml 的 data_root。"""
    if root is not None:
        return ImageStore(root)
    from whitewhale.config import load_config
    cfg = load_config("pipeline")
    return ImageStore(cfg.get("data_root", "src_dataset"))
