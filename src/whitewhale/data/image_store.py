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
from pathlib import Path

from PIL import Image, ImageOps

# 进程内缩略图缓存（图片只读，无需过期）
_THUMB_CACHE: dict[str, bytes] = {}


class ImageStore:
    """原图目录访问入口：相对路径 → 绝对路径 / PIL / 缩略图字节。"""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def resolve(self, rel_path: str) -> Path:
        """相对路径 → 绝对路径（含越界防护）。

        越界（如 ../ 逃逸出数据根）直接报错，防止清单污染读任意文件。
        """
        root_abs = self.root.resolve()
        p = (self.root / rel_path).resolve()
        if not str(p).startswith(str(root_abs)):
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
        key = f"{rel_path}|{max_w}|{quality}"
        hit = _THUMB_CACHE.get(key)
        if hit is not None:
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
        return data


def get_image_store(root: Path | str | None = None) -> ImageStore:
    """全局入口：root 缺省时自动读 configs/pipeline.yaml 的 data_root。"""
    if root is not None:
        return ImageStore(root)
    from whitewhale.config import load_config
    cfg = load_config("pipeline")
    return ImageStore(cfg.get("data_root", "src_dataset"))
