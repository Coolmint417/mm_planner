from __future__ import annotations

import torch
from torch import nn

from config import ModelConfig


class MultimodalFusionTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.image_proj = nn.Linear(cfg.vision_output_dim, cfg.fusion_dim)
        self.traj_proj = nn.Linear(cfg.traj_token_dim, cfg.fusion_dim)
        self.waypoint_proj = nn.Linear(cfg.waypoint_token_dim, cfg.fusion_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.fusion_dim))
        self.modality_embedding = nn.Parameter(torch.zeros(1, 4, cfg.fusion_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.fusion_dim,
            nhead=cfg.fusion_heads,
            dim_feedforward=cfg.fusion_ff_dim,
            dropout=cfg.fusion_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.fusion_layers)
        self.norm = nn.LayerNorm(cfg.fusion_dim)

    def forward(
        self,
        z_img: torch.Tensor,
        z_traj: torch.Tensor,
        z_waypoint: torch.Tensor,
    ) -> torch.Tensor:
        # z_img: [B, vision_output_dim], z_traj: [B, traj_token_dim]
        # z_waypoint: [B, waypoint_token_dim]
        image_token = self.image_proj(z_img).unsqueeze(1)
        traj_token = self.traj_proj(z_traj).unsqueeze(1)
        waypoint_token = self.waypoint_proj(z_waypoint).unsqueeze(1)
        cls_token = self.cls_token.expand(z_img.size(0), -1, -1)

        # tokens: [B, 4, fusion_dim]
        tokens = torch.cat([cls_token, image_token, traj_token, waypoint_token], dim=1)
        tokens = tokens + self.modality_embedding
        encoded = self.encoder(tokens)

        # h: [B, fusion_dim]
        return self.norm(encoded[:, 0])
