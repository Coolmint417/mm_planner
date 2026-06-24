# Trajectory Optimizer

This module post-processes model waypoint outputs into a smooth piecewise
polynomial trajectory.

## Configuration

Edit `config_trajectory_optimizer.py`:

- `k_pos = 8`: polynomial degree for x/y/z
- `k_yaw = 5`: polynomial degree for yaw
- `waypoint_dt = 0.2`: default time spacing between model waypoints

The optimizer minimizes the integral of squared 4th derivative and enforces C4
continuity at segment connections.

## Example

```python
import numpy as np

from trajectory_optimizer import (
    PolynomialTrajectoryOptimizer,
    TrajectoryOptimizerConfig,
)

positions = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.2, 0.1],
    [1.0, 0.0, 0.2],
])
yaws = np.array([0.0, 0.1, 0.2])
velocities = [
    np.array([0.0, 0.0, 0.0]),
    None,                          # no velocity constraint at this knot
    np.array([0.0, 0.0, 0.0]),
]                                  # optional
accelerations = np.zeros((3, 3))   # optional
yaw_rates = np.zeros(3)            # optional

optimizer = PolynomialTrajectoryOptimizer(TrajectoryOptimizerConfig())
trajectory = optimizer.optimize(
    positions=positions,
    yaws=yaws,
    velocities=velocities,
    accelerations=accelerations,
    yaw_rates=yaw_rates,
)

state = trajectory.evaluate(0.2)
print(state.position)
print(state.velocity)
print(state.acceleration)
print(state.yaw)
print(state.yaw_rate)
```

`trajectory.pos_coeffs` has shape `[num_segments, k_pos + 1, 3]`.
`trajectory.yaw_coeffs` has shape `[num_segments, k_yaw + 1]`.

For `velocities`, `accelerations`, and `yaw_rates`, each knot can be left
unconstrained by using `None` or `NaN`. At an unconstrained internal knot, the
optimizer enforces derivative continuity between neighboring polynomial
segments instead of pinning that derivative to a target value.
