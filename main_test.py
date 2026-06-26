from __future__ import annotations

import argparse
import copy

import torch

from mm_planner.config import cfg
from mm_planner.models import MultimodalPlanner
from mm_planner.utils import (
    count_total_parameters,
    count_trainable_parameters,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mock-vision",
        action="store_true",
        help="Use a small local vision backbone instead of downloading DINOv2.",
    )
    args = parser.parse_args()

    run_cfg = copy.deepcopy(cfg)
    run_cfg.model.allow_mock_vision_encoder = args.mock_vision

    torch.manual_seed(run_cfg.model.seed)
    device = torch.device(
        run_cfg.model.device if torch.cuda.is_available() else "cpu"
    )

    model = MultimodalPlanner(run_cfg.model).to(device)
    model.train()

    batch_size = 2
    height = width = run_cfg.model.image_size

    rgb = torch.randn(batch_size, 3, height, width, device=device)
    traj_mode_ids = torch.randint(
        low=0,
        high=run_cfg.model.num_motion_modes,
        size=(batch_size, run_cfg.model.n_traj_encoder),
        device=device,
    )
    traj_continuous = torch.randn(
        batch_size,
        run_cfg.model.n_traj_encoder,
        run_cfg.model.traj_continuous_dim,
        device=device,
    )
    task_waypoints = [
        torch.randn(3, run_cfg.model.waypoint_dim, device=device),
        torch.randn(8, run_cfg.model.waypoint_dim, device=device),
    ]
    teacher_flight_waypoints = torch.randn(
        batch_size,
        run_cfg.model.pred_num_flight_waypoints,
        run_cfg.model.pred_waypoint_dim,
        device=device,
    )
    teacher_flight_velocity_deltas = torch.randn(
        batch_size,
        run_cfg.model.pred_num_flight_waypoints,
        run_cfg.model.pred_velocity_delta_dim,
        device=device,
    )
    teacher_flight_acceleration_deltas = torch.randn(
        batch_size,
        run_cfg.model.pred_num_flight_waypoints,
        run_cfg.model.pred_acceleration_delta_dim,
        device=device,
    )

    outputs = model(
        rgb=rgb,
        traj_mode_ids=traj_mode_ids,
        traj_continuous=traj_continuous,
        task_waypoints=task_waypoints,
        teacher_flight_waypoints=teacher_flight_waypoints,
        teacher_flight_velocity_deltas=teacher_flight_velocity_deltas,
        teacher_flight_acceleration_deltas=teacher_flight_acceleration_deltas,
        teacher_forcing_ratio=run_cfg.training.teacher_forcing_ratio,
    )

    flight_waypoints = outputs["flight_waypoints"]
    flight_velocity_deltas = outputs["flight_velocity_deltas"]
    flight_acceleration_deltas = outputs["flight_acceleration_deltas"]
    assert isinstance(flight_waypoints, dict)
    assert isinstance(flight_velocity_deltas, dict)
    assert isinstance(flight_acceleration_deltas, dict)
    print(f"mode_logits shape: {tuple(outputs['mode_logits'].shape)}")
    print(f"flight_waypoints.mu shape: {tuple(flight_waypoints['mu'].shape)}")
    print(f"flight_waypoints.sigma shape: {tuple(flight_waypoints['sigma'].shape)}")
    print(
        "flight_velocity_deltas.mu shape: "
        f"{tuple(flight_velocity_deltas['mu'].shape)}"
    )
    print(
        "flight_velocity_deltas.sigma shape: "
        f"{tuple(flight_velocity_deltas['sigma'].shape)}"
    )
    print(
        "flight_acceleration_deltas.mu shape: "
        f"{tuple(flight_acceleration_deltas['mu'].shape)}"
    )
    print(
        "flight_acceleration_deltas.sigma shape: "
        f"{tuple(flight_acceleration_deltas['sigma'].shape)}"
    )
    print(f"crawl_action shape: {tuple(outputs['crawl_action'].shape)}")
    print(f"fused_feature shape: {tuple(outputs['fused_feature'].shape)}")
    print(f"total parameters: {count_total_parameters(model):,}")
    print(f"trainable parameters: {count_trainable_parameters(model):,}")


if __name__ == "__main__":
    main()
