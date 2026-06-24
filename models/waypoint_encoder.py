from __future__ import annotations

import torch
from torch import nn

from mm_planner.config import ModelConfig


class TaskWaypointEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.waypoint_dim, cfg.waypoint_token_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.waypoint_token_dim),
        )
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, cfg.n_waypoints, cfg.waypoint_token_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.waypoint_token_dim,
            nhead=cfg.waypoint_encoder_heads,
            dim_feedforward=cfg.waypoint_encoder_ff_dim,
            dropout=cfg.waypoint_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.waypoint_encoder_layers)
        self.attn_pool = nn.Linear(cfg.waypoint_token_dim, 1)

    def forward(
        self,
        task_waypoints: torch.Tensor,
        task_waypoint_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # task_waypoints: [B, M, waypoint_dim], task_waypoint_mask: [B, M] or None
        x = self.input_proj(task_waypoints)
        x = x + self.pos_embedding[:, : x.size(1)]

        # encoded: [B, M, waypoint_token_dim]
        encoded = self.encoder(x, src_key_padding_mask=task_waypoint_mask)
        scores = self.attn_pool(encoded)
        if task_waypoint_mask is not None:
            scores = scores.masked_fill(task_waypoint_mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)

        # z_waypoint: [B, waypoint_token_dim]
        return (encoded * weights).sum(dim=1)
