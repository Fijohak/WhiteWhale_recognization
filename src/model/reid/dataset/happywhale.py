"""
Happywhale 数据集 adapter（主路线 A：公开数据定量评估）。

数据来源：HF 镜像 GATE-engine/happy-whale-dolphin-classification（parquet 分片）。
parquet 每行 image 列为 dict{bytes, path}，bytes 内嵌 JPEG 二进制，无需本地解压。

映射到统一 schema：
- image_path   → 保存到本地 images 缓存目录后的路径（可选，见 load 的 extract_images）
- identity     → happywhale__{individual_name}（namespace 防跨源冲突）
- species      → species_name（注意官方拼写：bottlenose_dolpin 是原始数据错误，保留原样不纠错）
- encounter_id → 无（Happywhale 官方不提供 encounter 分组，报告限制，不做 encounter-safe split）
- split        → 按 parquet 分片归属（train/val/test 由文件名决定）

注意：
- 官方 individual_name 存在个别异常（如 'new_individual'），属于数据本身问题，保留不处理。
- 不复制图片进项目仓库，只落盘到 data_root 下的缓存目录。
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from PIL import Image

from src.model.reid.dataset.base import DatasetAdapter, ReIDData


class HappywhaleAdapter(DatasetAdapter):
    name = "happywhale"
    has_identity = True

    def __init__(self, image_cache_dir: str | Path | None = None):
        """image_cache_dir：解出的图片落盘目录；None 时只建 metadata（不落图）。"""
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir else None

    def load(self, data_root: str | None = None) -> ReIDData:
        data_root = Path(data_root) if data_root else Path("D:/dolphin_data/happywhale")
        # 过滤 .part 等未下载完成的分片（避免读到损坏文件）
        parquet_files = sorted(
            f for f in data_root.glob("*.parquet")
            if not f.name.endswith((".part", ".tmp")))
        if not parquet_files:
            raise FileNotFoundError(f"{data_root} 下没有 parquet 文件")

        rows = []
        for f in parquet_files:
            split = "train" if "train" in f.name else ("val" if "val" in f.name else "test")
            df = pd.read_parquet(f, columns=["image", "species_name", "individual_name"])
            for _, r in df.iterrows():
                img = r["image"]
                if not isinstance(img, dict) or "bytes" not in img:
                    continue  # 跳过损坏行（不静默产出空 embedding，此处只是 metadata）
                rows.append({
                    "image_bytes": img["bytes"],
                    "image_name": img.get("path", f.name),
                    "identity": f"happywhale__{r['individual_name']}",
                    "species": r["species_name"],
                    "source_dataset": "happywhale",
                    "encounter_id": None,
                    "date": None,
                    "viewpoint": None,
                    "split": split,
                })

        meta = pd.DataFrame(rows)
        # 缓存目录存在时按名字补 image_path（幂等，不重复解图）
        cache = self.image_cache_dir or (data_root / "images")
        if cache.is_dir():
            meta["image_path"] = [
                str(cache / f"{i:06d}.jpg") for i in range(len(meta))]
        elif self.image_cache_dir:
            self._extract_images(meta)
        return self._normalize(meta)

    def _extract_images(self, meta: pd.DataFrame) -> None:
        """把 bytes 落盘为图片文件，image_path 指向缓存目录。"""
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, r in meta.iterrows():
            name = f"{i:06d}.jpg"
            p = self.image_cache_dir / name
            if not p.exists():  # 幂等：已存在跳过
                p.write_bytes(r["image_bytes"])
            paths.append(str(p))
        meta["image_path"] = paths
