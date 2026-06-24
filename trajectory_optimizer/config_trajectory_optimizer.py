from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrajectoryOptimizerConfig:
    # Polynomial order for x/y/z. Degree 8 gives 9 coefficients per segment.
    k_pos: int = 8

    # Polynomial order for yaw. Degree 5 gives 6 coefficients per segment.
    k_yaw: int = 5

    # Default waypoint spacing used when segment_durations is not provided.
    waypoint_dt: float = 0.2

    # Objective weights for integral of squared 4th derivative.
    snap_weight_pos: float = 1.0
    snap_weight_yaw: float = 0.1

    # IPOPT settings.
    ipopt_print_level: int = 0
    print_time: bool = False
    max_iter: int = 500

    # Evaluation behavior outside [0, total_time].
    clamp_query_time: bool = True
