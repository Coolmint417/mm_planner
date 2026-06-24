from __future__ import annotations

import torch
from torch import nn

from config import ModelConfig


class _MockDINOv2Backbone(nn.Module):
    """Small offline backbone used only when explicitly enabled in config."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward_features(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        fmap = self.net(rgb)
        tokens = fmap.flatten(2).transpose(1, 2)
        cls = tokens.mean(dim=1)
        return {"x_norm_patchtokens": tokens, "x_norm_clstoken": cls}


class DINOv2VisionEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_patch_tokens = cfg.use_patch_tokens
        self.freeze_vision_encoder = cfg.freeze_vision_encoder

        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                cfg.vision_model_name,
            )
        except Exception:
            if not cfg.allow_mock_vision_encoder:
                raise
            self.backbone = _MockDINOv2Backbone(cfg.dinov2_feature_dim)

        if self.freeze_vision_encoder:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.attn_pool = nn.Linear(cfg.dinov2_feature_dim, 1)
        self.proj = nn.Sequential(
            nn.Linear(cfg.dinov2_feature_dim, cfg.vision_output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.vision_output_dim),
        )

    def train(self, mode: bool = True) -> "DINOv2VisionEncoder":
        super().train(mode)
        if self.freeze_vision_encoder:
            self.backbone.eval()
        return self

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb: [B, 3, H, W]
        if self.freeze_vision_encoder:
            with torch.no_grad():
                features = self.backbone.forward_features(rgb)
        else:
            features = self.backbone.forward_features(rgb)

        # patch_tokens: [B, N, C], cls_token: [B, C]
        if self.use_patch_tokens:
            patch_tokens = features["x_norm_patchtokens"]
            scores = self.attn_pool(patch_tokens)
            weights = torch.softmax(scores, dim=1)
            pooled = (patch_tokens * weights).sum(dim=1)
        else:
            pooled = features["x_norm_clstoken"]

        # z_img: [B, vision_output_dim]
        return self.proj(pooled)
