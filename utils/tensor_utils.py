from __future__ import annotations

import torch
from torch import nn


def waypoint_smoothness_loss(waypoints: torch.Tensor) -> torch.Tensor:
    # waypoints: [B, N, D]
    if waypoints.size(1) < 3:
        return waypoints.new_zeros(())
    velocity = waypoints[:, 1:] - waypoints[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    return torch.mean(acceleration**2)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
