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

    def encode_paths(self, paths: list[Path | str]) -> np.ndarray:
        """从路径批量提取（逐个读图，失败即抛错）。"""
        imgs = []
        for p in paths:
            imgs.append(Image.open(p).convert("RGB"))
        return self.encode(imgs)


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
    """DINOv2 通用自监督视觉表征（主路线 A 对照组）。"""

    name = "dinov2"
    input_size = 518
    feat_dim = 768

    def _load(self):
        # DINOv2 权重在 hf，通过 HF_ENDPOINT 镜像下载
        torch.hub.set_dir(str(Path.home() / ".cache" / "torch" / "hub"))
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", source="local"
        ).to(self.device) if False else None  # placeholder，见下方真实加载
        # 真实加载：dinov2 无 timm 包装，走 torch.hub（需 github 可达）。
        # 若 github 不可达，改用 hf 上的 dinov2 权重（huggingface.co/facebook/dinov2-base）。
        from huggingface_hub import PyTorchModelHubMixin  # noqa: F401

        raise NotImplementedError(
            "DINOv2 加载需适配：优先用 hf 镜像的 facebook/dinov2-base 权重。"
            "待数据就绪后实现，见 src/reid/embedding/dinov2.py。")


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
