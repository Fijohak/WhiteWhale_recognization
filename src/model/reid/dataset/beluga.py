"""
BelugaID 数据集 adapter（主路线 A：公开数据定量评估）。

数据来源：LILA 官方 beluga-id-test.zip（竞赛 test 集，含 3402 图 / 978 身份 / 10 scenario）。
zip 内：
- images/test*.jpg             → 图片
- private_test_labels.csv      → GT 匹配对（query_id, database_image_id）
- private_test_metadata.csv    → 每张图的 original_whale_id / viewpoint / date（⚠️ 非官方字段，仅参考）
- code-execution/queries/*.csv → 每个 scenario 的 query 列表
- code-execution/databases/*.csv → 每个 scenario 的 database 列表

语义（方向调整后）：
- 这是官方 Re-ID benchmark：query 图在 database 内（自匹配），评估时必须排除 query 自身；
- original_whale_id 是竞赛未公开身份，**只用于内部研究评估**，不可再分发；
- identity → beluga__{original_whale_id}（namespace 防跨源冲突）；
- 每个 scenario 是一个独立评估任务（query 少、database 大），不跨 scenario 混用。
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from src.model.reid.dataset.base import DatasetAdapter, ReIDData


class BelugaTestAdapter(DatasetAdapter):
    """BelugaID 竞赛 test 集（zip 内全部数据，首次加载自动解图到 data_root/test_images/）。"""

    name = "beluga"
    has_identity = True

    def __init__(self, extract_images: bool = True):
        self.extract_images = extract_images

    def load(self, data_root: str | None = None) -> ReIDData:
        data_root = Path(data_root) if data_root else Path("D:/dolphin_data/beluga")
        zip_path = data_root / "beluga-id-test.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"缺少 {zip_path}（LILA 直链下载）")

        z = zipfile.ZipFile(zip_path)
        meta = pd.read_csv(z.open("private_test_metadata.csv"))

        # 首次加载把图解压到 test_images/（幂等）
        img_dir = data_root / "test_images"
        if self.extract_images:
            img_dir.mkdir(parents=True, exist_ok=True)
            missing = [n for n in z.namelist()
                       if n.startswith("code-execution/images/") and not (img_dir / n.split("/")[-1]).exists()]
            for i, n in enumerate(missing):
                (img_dir / n.split("/")[-1]).write_bytes(z.read(n))
            print(f"[beluga] 解图 {len(missing)} 张 → {img_dir}")

        rows = []
        for _, r in meta.iterrows():
            rows.append({
                "image_path": str(img_dir / f"{r['image_id']}.jpg"),
                "image_id": r["image_id"],  # test0000 ...（自匹配排除用）
                "identity": f"beluga__{r['original_whale_id']}",
                "species": "beluga_whale",
                "source_dataset": "beluga",
                "encounter_id": None,  # test 集无 encounter 字段
                "date": r.get("date"),
                "viewpoint": r.get("viewpoint"),
                "split": "test",
                "original_image_id": r.get("original_image_id"),
            })

        df = pd.DataFrame(rows)
        # 官方 meta 存在重复行（如 test0007 出现两次）：同图去重，避免
        # 检索/评估把同图当独立样本（image_path 由 image_id 决定，重复行同路径）
        df = df.drop_duplicates(subset="image_id")
        return self._normalize(df)
