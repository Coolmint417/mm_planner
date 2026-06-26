from __future__ import annotations

import torch
from torch import nn

from config import ModelConfig
from models.fusion_transformer import MultimodalFusionTransformer
from models.heads import (
    AutoregressiveWaypointHead,
    CrawlActionHead,
    FlightAccelerationDeltaHead,
    FlightVelocityDeltaHead,
    ModePredictionHead,
)
from models.trajectory_encoder import HistoricalTrajectoryEncoder
from models.vision_encoder import DINOv2VisionEncoder
from models.waypoint_encoder import TaskWaypointEncoder


class MultimodalPlanner(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = DINOv2VisionEncoder(cfg)
        self.trajectory_encoder = HistoricalTrajectoryEncoder(cfg)
        self.waypoint_encoder = TaskWaypointEncoder(cfg)
        self.fusion = MultimodalFusionTransformer(cfg)

        self.mode_head = ModePredictionHead(cfg)
        self.flight_head = AutoregressiveWaypointHead(cfg)
        self.flight_velocity_delta_head = FlightVelocityDeltaHead(cfg)
        self.flight_acceleration_delta_head = FlightAccelerationDeltaHead(cfg)
        self.crawl_head = CrawlActionHead(cfg)

    def forward(
        self,
        rgb: torch.Tensor,
        traj_mode_ids: torch.Tensor,
        traj_continuous: torch.Tensor,
        task_waypoints: list[torch.Tensor] | torch.Tensor,
        teacher_flight_waypoints: torch.Tensor | None = None,
        teacher_flight_velocity_deltas: torch.Tensor | None = None,
        teacher_flight_acceleration_deltas: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor | dict[str, torch.Tensor]]]:
        # rgb: [B, 3, H, W]
        # traj_mode_ids: [B, T], traj_continuous: [B, T, traj_continuous_dim]
        # task_waypoints: list of [M_i, waypoint_dim] tensors.
        z_img = self.vision_encoder(rgb)
        z_traj = self.trajectory_encoder(traj_mode_ids, traj_continuous)
        z_waypoint = self.waypoint_encoder(task_waypoints)

        # h: [B, fusion_dim]
        h = self.fusion(z_img, z_traj, z_waypoint)

        # Heads produce planner outputs with documented public shapes.
        return {
            "mode_logits": self.mode_head(h),
            "flight_waypoints": self.flight_head(
                h,
                teacher_waypoints=teacher_flight_waypoints,
                teacher_forcing_ratio=teacher_forcing_ratio,
            ),
            "flight_velocity_deltas": self.flight_velocity_delta_head(
                h,
                teacher_velocity_deltas=teacher_flight_velocity_deltas,
                teacher_forcing_ratio=teacher_forcing_ratio,
            ),
            "flight_acceleration_deltas": self.flight_acceleration_delta_head(
                h,
                teacher_acceleration_deltas=teacher_flight_acceleration_deltas,
                teacher_forcing_ratio=teacher_forcing_ratio,
            ),
            "crawl_action": self.crawl_head(h),
            #  "transition": self.transition_head(h),
            "fused_feature": h,
        }
