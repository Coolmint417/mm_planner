from __future__ import annotations

import torch
from torch import nn

from mm_planner.config import ModelConfig


class TaskWaypointEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.waypoint_dim + 1, cfg.waypoint_token_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.waypoint_token_dim),
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
        task_waypoints: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        # task_waypoints: list of [M_i, waypoint_dim] tensors.
        if isinstance(task_waypoints, torch.Tensor):
            samples = [task_waypoints[i] for i in range(task_waypoints.size(0))]
        else:
            samples = task_waypoints

        return torch.stack([self._encode_one(sample) for sample in samples], dim=0)

    def _encode_one(self, waypoints: torch.Tensor) -> torch.Tensor:
        if waypoints.ndim != 2:
            raise ValueError(f"Expected task waypoints [M, D], got {tuple(waypoints.shape)}")

        if waypoints.size(0) == 0:
            return waypoints.new_zeros((self.attn_pool.in_features,))

        x = self._append_order_feature(waypoints)
        x = self.input_proj(x).unsqueeze(0)

        # encoded: [1, M, waypoint_token_dim]
        encoded = self.encoder(x)
        scores = self.attn_pool(encoded)
        weights = torch.softmax(scores, dim=1)

        # z_waypoint: [waypoint_token_dim]
        return (encoded * weights).sum(dim=1).squeeze(0)

    def _append_order_feature(self, waypoints: torch.Tensor) -> torch.Tensor:
        count = waypoints.size(0)
        if count == 1:
            order = waypoints.new_zeros((1, 1))
        else:
            order = torch.linspace(
                0.0,
                1.0,
                steps=count,
                device=waypoints.device,
                dtype=waypoints.dtype,
            ).unsqueeze(-1)
        return torch.cat([waypoints, order], dim=-1)
