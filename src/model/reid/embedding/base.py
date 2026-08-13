"""
统一 embedding 接口（方向调整后）。

所有 backbone 通过 adapter 实现 EmbeddingModel.encode(images) -> embeddings，
保证 dataset / retrieval / evaluation 不依赖具体 backbone。

当前实现：
- DINOv2Adapter（facebookresearch/dinov2，hf 权重）
- MegaDescriptorAdapter（BVRA/MegaDescriptor-T-224，timm hf-hub）

约定：
- 输出 L2 归一化 embedding（cosine 相似度直接可用）；
- 所有模型权重通过 hf-mirror 环境变量（HF_ENDPOINT）下载；
- 图片读取失败必须报错（禁止静默 0 向量）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class EmbeddingModel(ABC):
    """embedding 提取器接口。"""

    name: str = "base"
    feat_dim: int = 0

    @abstractmethod
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """输入 PIL 图片列表，返回 L2 归一化 embedding (N, D) float32。"""

    def encode_paths(self, paths: list[Path | str], batch_size: int = 32) -> np.ndarray:
        """从路径批量提取（分批读图避免整批驻留内存，失败即抛错）。

        batch_size: 每批读入的图片数；GPU/大图集可调小控制内存峰值。
        """
        outs = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            outs.append(self.encode(imgs))
        return np.concatenate(outs, axis=0)


class _HFImageModel(EmbeddingModel):
    """hf 权重模型的通用实现（预处理 + forward + L2）。"""

    def _load(self):
        raise NotImplementedError

    def __init__(self, device: str = "auto"):
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        self._load()

    def _preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        import torchvision.transforms as T

        tf = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return torch.stack([tf(im) for im in images]).to(self.device)

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.feat_dim), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            x = self._preprocess(images)
            out = self.model(x)
        emb = out.float().cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms


class DINOv2Adapter(_HFImageModel):
    """DINOv2 通用自监督视觉表征（主路线 A 对照组）。

    权重来源二选一：
    - 默认 timm 在线下载（vit_base_patch14_dinov2.lvd142m = 官方权重转换，
      走 HF 通道）；网络不可用时传 weight_path 加载官方 .pth（离线）。
    官方权重键与 timm 模型 174/174 匹配，仅多一个预训练用的 mask_token。

    注意：模型默认固定 518x518（官方推荐分辨率，pos_embed 训练尺寸匹配），
    不使用 224 输入（224 需插值 pos_embed 且损失空间信息）。
    """

    name = "dinov2"
    input_size = 518
    feat_dim = 768

    def __init__(self, device: str = "auto", weight_path: str | None = None):
        self.weight_path = weight_path
        super().__init__(device)

    def _load(self):
        import timm

        self.model = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=self.weight_path is None, num_classes=0)
        if self.weight_path:
            self._load_official_weight()
        self.model.eval()
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
        self.feat_dim = self.model.num_features

    def _load_official_weight(self):
        """加载 facebookresearch/dinov2 官方 .pth（离线）。

        官方权重含预训练任务键 mask_token，timm 模型无此键 → 剔除。
        """
        import torch

        sd = torch.load(self.weight_path, map_location="cpu")
        sd = sd.get("model", sd)
        sd = {k: v for k, v in sd.items() if k in self.model.state_dict()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            raise ValueError(f"DINOv2 权重加载不完整，缺 {len(missing)} 个键: "
                             f"{sorted(missing)[:5]}")


class MegaDescriptorAdapter(_HFImageModel):
    """MegaDescriptor-T-224（WildlifeTools，timm hf-hub 加载）。"""

    name = "megadescriptor"
    input_size = 224
    feat_dim = 768

    def _load(self):
        import timm

        self.model = timm.create_model(
            "hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True, num_classes=0)
        self.model.eval()
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
        self.feat_dim = self.model.num_features
