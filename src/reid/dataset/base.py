"""
统一数据集接口（方向调整后）。

所有公开/本土数据集通过 adapter 转换为统一 metadata DataFrame：

    image_path | identity | species | source_dataset | encounter_id | date | viewpoint | split

约定：
- identity 必须 namespace：happywhale__id_001 / ndd20__id_014（禁止跨源 ID 冲突）；
- 缺失字段用 None，不编造；
- 数据集划分默认 encounter-safe（未提供 encounter 时报告限制）；
- 本土数据（LocalChineseWhiteDolphin）不产生 identity，只保留
  image_id / relative_path / session_id / group_id / quality_band / is_anchor。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class ReIDData:
    """统一数据集结构。"""

    df: pd.DataFrame  # 统一 metadata 表（列见模块 docstring）
    name: str         # 数据集名（happywhale / ndd20 / beluga / local）
    has_identity: bool  # 是否含可靠 individual_id（决定能否做定量评估）

    @property
    def n_images(self) -> int:
        return len(self.df)

    @property
    def n_identities(self) -> int | None:
        return self.df["identity"].nunique() if self.has_identity else None

    @property
    def n_species(self) -> int | None:
        return self.df["species"].nunique() if "species" in self.df else None


class DatasetAdapter(ABC):
    """数据集适配器基类：任意源 → 统一 ReIDData。"""

    name: str = "base"
    has_identity: bool = False

    @abstractmethod
    def load(self, data_root: str | None = None) -> ReIDData:
        """加载数据集并返回统一结构。data_root 为本地数据目录。"""

    def _normalize(self, df: pd.DataFrame) -> ReIDData:
        """确保输出列齐全，缺失列补 None。"""
        required = ["image_path", "identity", "species", "source_dataset",
                    "encounter_id", "date", "viewpoint", "split"]
        for c in required:
            if c not in df.columns:
                df[c] = None
        return ReIDData(df=df, name=self.name, has_identity=self.has_identity)
