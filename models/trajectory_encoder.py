from __future__ import annotations

import torch
from torch import nn

from mm_planner.config import ModelConfig


class HistoricalTrajectoryEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mode_embedding = nn.Embedding(
            cfg.num_motion_modes,
            cfg.mode_embedding_dim,
        )
        input_dim = cfg.mode_embedding_dim + cfg.traj_continuous_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, cfg.traj_token_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.traj_token_dim),
        )
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, cfg.n_traj_encoder, cfg.traj_token_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.traj_token_dim,
            nhead=cfg.traj_encoder_heads,
            dim_feedforward=cfg.traj_encoder_ff_dim,
            dropout=cfg.traj_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.traj_encoder_layers)
        self.attn_pool = nn.Linear(cfg.traj_token_dim, 1)

    def forward(
        self,
        traj_mode_ids: torch.Tensor,
        traj_continuous: torch.Tensor,
    ) -> torch.Tensor:
        # traj_mode_ids: [B, T], traj_continuous: [B, T, traj_continuous_dim]
        # position: 3
        # velocity: 3
        # acceleration: 3
        # attitude quaternion: 4
        # angular velocity: 3
        mode_tokens = self.mode_embedding(traj_mode_ids)
        x = torch.cat([mode_tokens, traj_continuous], dim=-1)
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, : x.size(1)]

        # encoded: [B, T, traj_token_dim]
        encoded = self.encoder(x)
        weights = torch.softmax(self.attn_pool(encoded), dim=1)

        # z_traj: [B, traj_token_dim]
        return (encoded * weights).sum(dim=1)
