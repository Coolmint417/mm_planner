from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mm_planner.config import ModelConfig


class ModePredictionHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.num_modes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, fusion_dim], mode_logits: [B, num_modes]
        return self.net(h)


class AutoregressiveWaypointHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.num_steps = cfg.pred_num_flight_waypoints
        self.waypoint_dim = cfg.pred_waypoint_dim
        self.predict_delta = cfg.autoregressive_predict_delta
        self.max_waypoint_delta = cfg.max_waypoint_delta
        self.min_sigma = cfg.min_action_sigma

        self.context_proj = nn.Linear(cfg.fusion_dim, cfg.autoregressive_hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.pred_waypoint_dim, cfg.autoregressive_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRUCell(
            cfg.autoregressive_hidden_dim,
            cfg.autoregressive_hidden_dim,
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(cfg.autoregressive_hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, 2 * cfg.pred_waypoint_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        teacher_waypoints: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        # h: [B, fusion_dim], teacher_waypoints: [B, N, pred_waypoint_dim] or None
        batch_size = h.size(0)
        hidden = torch.tanh(self.context_proj(h))
        prev_waypoint = h.new_zeros(batch_size, self.waypoint_dim)
        mus: list[torch.Tensor] = []
        sigmas: list[torch.Tensor] = []

        for step in range(self.num_steps):
            gru_input = self.input_proj(prev_waypoint)
            hidden = self.gru(gru_input, hidden)
            raw = self.out_mlp(hidden)
            raw_mu, raw_sigma = raw.chunk(2, dim=-1)
            sigma = F.softplus(raw_sigma) + self.min_sigma

            if self.predict_delta:
                delta_mu = torch.tanh(raw_mu) * self.max_waypoint_delta
                waypoint_mu = prev_waypoint + delta_mu
            else:
                waypoint_mu = raw_mu

            mus.append(waypoint_mu)
            sigmas.append(sigma)

            use_teacher = (
                self.training
                and teacher_waypoints is not None
                and teacher_forcing_ratio > 0.0
            )
            if use_teacher:
                # mask: [B, 1], previous waypoint for the next decoding step.
                mask = (
                    torch.rand(batch_size, 1, device=h.device)
                    < teacher_forcing_ratio
                )
                prev_waypoint = torch.where(
                    mask,
                    teacher_waypoints[:, step],
                    waypoint_mu,
                )
            else:
                prev_waypoint = waypoint_mu

        # mu/sigma: [B, N, pred_waypoint_dim]
        return {
            "mu": torch.stack(mus, dim=1),
            "sigma": torch.stack(sigmas, dim=1),
        }


class AutoregressiveSequenceHead(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        output_dim: int,
        num_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps or cfg.pred_num_flight_waypoints
        self.output_dim = output_dim
        self.min_sigma = cfg.min_action_sigma

        self.context_proj = nn.Linear(cfg.fusion_dim, cfg.autoregressive_hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(output_dim, cfg.autoregressive_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRUCell(
            cfg.autoregressive_hidden_dim,
            cfg.autoregressive_hidden_dim,
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(cfg.autoregressive_hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, 2 * output_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        teacher_sequence: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        # h: [B, fusion_dim], teacher_sequence: [B, N, output_dim] or None
        batch_size = h.size(0)
        hidden = torch.tanh(self.context_proj(h))
        prev_output = h.new_zeros(batch_size, self.output_dim)
        mus: list[torch.Tensor] = []
        sigmas: list[torch.Tensor] = []

        for step in range(self.num_steps):
            gru_input = self.input_proj(prev_output)
            hidden = self.gru(gru_input, hidden)
            raw = self.out_mlp(hidden)
            mu, raw_sigma = raw.chunk(2, dim=-1)
            sigma = F.softplus(raw_sigma) + self.min_sigma
            mus.append(mu)
            sigmas.append(sigma)

            use_teacher = (
                self.training
                and teacher_sequence is not None
                and teacher_forcing_ratio > 0.0
            )
            if use_teacher:
                # mask: [B, 1], previous output for the next decoding step.
                mask = (
                    torch.rand(batch_size, 1, device=h.device)
                    < teacher_forcing_ratio
                )
                prev_output = torch.where(
                    mask,
                    teacher_sequence[:, step],
                    mu,
                )
            else:
                prev_output = mu

        # mu/sigma: [B, N, output_dim]
        return {
            "mu": torch.stack(mus, dim=1),
            "sigma": torch.stack(sigmas, dim=1),
        }


# Backward-compatible alias for code that imported the earlier non-AR name.
SequencePredictionHead = AutoregressiveSequenceHead


class FlightVelocityDeltaHead(AutoregressiveSequenceHead):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, output_dim=cfg.pred_velocity_delta_dim)

    def forward(
        self,
        h: torch.Tensor,
        teacher_velocity_deltas: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        # h: [B, fusion_dim], output mu/sigma: [B, N, 4]
        # Each step is [dv_x, dv_y, dv_z, ddyaw].
        return super().forward(
            h,
            teacher_sequence=teacher_velocity_deltas,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )


class FlightAccelerationDeltaHead(AutoregressiveSequenceHead):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg, output_dim=cfg.pred_acceleration_delta_dim)

    def forward(
        self,
        h: torch.Tensor,
        teacher_acceleration_deltas: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        # h: [B, fusion_dim], output mu/sigma: [B, N, 3]
        # Each step is [da_x, da_y, da_z].
        return super().forward(
            h,
            teacher_sequence=teacher_acceleration_deltas,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )


class CrawlActionHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.crawl_action_dim),
            nn.Tanh(),
        )
        self.register_buffer(
            "scale",
            torch.tensor(
                [cfg.max_crawl_vx, cfg.max_crawl_vy, cfg.max_crawl_yaw_rate],
                dtype=torch.float32,
            ),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, fusion_dim], crawl_action: [B, 3]
        return self.net(h) * self.scale


class TransitionHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.max_approach_speed = cfg.max_approach_speed
        self.shared = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.contact_pos = nn.Linear(cfg.hidden_dim, 3)
        self.surface_normal = nn.Linear(cfg.hidden_dim, 3)
        self.yaw = nn.Linear(cfg.hidden_dim, 1)
        self.approach_speed = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        # h: [B, fusion_dim]
        shared = self.shared(h)
        normal = self.surface_normal(shared)

        # Each output keeps batch dimension B.
        return {
            "contact_pos": self.contact_pos(shared),
            "surface_normal": F.normalize(normal, dim=-1),
            "yaw": self.yaw(shared),
            "approach_speed": torch.sigmoid(self.approach_speed(shared))
            * self.max_approach_speed,
        }
