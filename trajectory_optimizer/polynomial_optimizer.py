from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import casadi as ca
import numpy as np

from .config_trajectory_optimizer import TrajectoryOptimizerConfig


@dataclass
class TrajectoryState:
    t: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    snap: np.ndarray
    yaw: float
    yaw_rate: float
    yaw_acceleration: float
    yaw_jerk: float
    yaw_snap: float


class PiecewisePolynomialTrajectory:
    """Piecewise polynomial trajectory with increasing-power coefficients."""

    def __init__(
        self,
        pos_coeffs: np.ndarray,
        yaw_coeffs: np.ndarray,
        segment_durations: np.ndarray,
        config: TrajectoryOptimizerConfig | None = None,
    ) -> None:
        self.pos_coeffs = np.asarray(pos_coeffs, dtype=np.float64)
        self.yaw_coeffs = np.asarray(yaw_coeffs, dtype=np.float64)
        self.segment_durations = np.asarray(segment_durations, dtype=np.float64)
        self.config = config or TrajectoryOptimizerConfig()
        self.knot_times = np.concatenate(
            [np.zeros(1), np.cumsum(self.segment_durations)]
        )

        if self.pos_coeffs.ndim != 3 or self.pos_coeffs.shape[2] != 3:
            raise ValueError("pos_coeffs must have shape [S, k_pos + 1, 3]")
        if self.yaw_coeffs.ndim != 2:
            raise ValueError("yaw_coeffs must have shape [S, k_yaw + 1]")
        if self.pos_coeffs.shape[0] != self.segment_durations.shape[0]:
            raise ValueError("position coefficient segment count mismatch")
        if self.yaw_coeffs.shape[0] != self.segment_durations.shape[0]:
            raise ValueError("yaw coefficient segment count mismatch")

    @property
    def total_time(self) -> float:
        return float(self.knot_times[-1])

    def evaluate(self, t: float) -> TrajectoryState:
        segment, tau, query_t = self._locate_segment(t)
        return TrajectoryState(
            t=query_t,
            position=self._eval_pos(segment, tau, derivative=0),
            velocity=self._eval_pos(segment, tau, derivative=1),
            acceleration=self._eval_pos(segment, tau, derivative=2),
            jerk=self._eval_pos(segment, tau, derivative=3),
            snap=self._eval_pos(segment, tau, derivative=4),
            yaw=float(self._eval_yaw(segment, tau, derivative=0)),
            yaw_rate=float(self._eval_yaw(segment, tau, derivative=1)),
            yaw_acceleration=float(self._eval_yaw(segment, tau, derivative=2)),
            yaw_jerk=float(self._eval_yaw(segment, tau, derivative=3)),
            yaw_snap=float(self._eval_yaw(segment, tau, derivative=4)),
        )

    __call__ = evaluate

    def _locate_segment(self, t: float) -> tuple[int, float, float]:
        query_t = float(t)
        if self.config.clamp_query_time:
            query_t = float(np.clip(query_t, 0.0, self.total_time))
        elif query_t < 0.0 or query_t > self.total_time:
            raise ValueError(f"t={t} is outside [0, {self.total_time}]")

        if query_t >= self.total_time:
            segment = self.segment_durations.shape[0] - 1
        else:
            segment = int(np.searchsorted(self.knot_times, query_t, side="right") - 1)
            segment = max(0, min(segment, self.segment_durations.shape[0] - 1))
        tau = query_t - float(self.knot_times[segment])
        return segment, tau, query_t

    def _eval_pos(self, segment: int, tau: float, derivative: int) -> np.ndarray:
        coeffs = self.pos_coeffs[segment]
        return np.array(
            [
                _eval_poly_np(coeffs[:, axis], tau, derivative)
                for axis in range(3)
            ],
            dtype=np.float64,
        )

    def _eval_yaw(self, segment: int, tau: float, derivative: int) -> float:
        return _eval_poly_np(self.yaw_coeffs[segment], tau, derivative)


class PolynomialTrajectoryOptimizer:
    def __init__(self, config: TrajectoryOptimizerConfig | None = None) -> None:
        self.config = config or TrajectoryOptimizerConfig()
        if self.config.k_pos < 8:
            raise ValueError("k_pos must be at least 8 for C4 snap optimization")
        if self.config.k_yaw < 5:
            raise ValueError("k_yaw must be at least 5 for C4 yaw continuity")

    def optimize(
        self,
        positions: np.ndarray,
        yaws: np.ndarray,
        segment_durations: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
        accelerations: np.ndarray | None = None,
        yaw_rates: np.ndarray | None = None,
    ) -> PiecewisePolynomialTrajectory:
        positions = _as_2d(positions, width=3, name="positions")
        yaws = np.asarray(yaws, dtype=np.float64).reshape(-1)
        if positions.shape[0] != yaws.shape[0]:
            raise ValueError("positions and yaws must have the same knot count")
        if positions.shape[0] < 2:
            raise ValueError("at least two knots are required")

        segment_durations = self._segment_durations(
            knot_count=positions.shape[0],
            segment_durations=segment_durations,
        )
        velocities, velocity_mask = _optional_masked_2d(
            velocities,
            positions.shape[0],
            3,
            "velocities",
        )
        accelerations, acceleration_mask = _optional_masked_2d(
            accelerations,
            positions.shape[0],
            3,
            "accelerations",
        )
        yaw_rates, yaw_rate_mask = _optional_masked_1d(
            yaw_rates,
            positions.shape[0],
            "yaw_rates",
        )

        opti = ca.Opti()
        pos_vars = [
            opti.variable(segment_durations.shape[0], self.config.k_pos + 1)
            for _ in range(3)
        ]
        yaw_var = opti.variable(segment_durations.shape[0], self.config.k_yaw + 1)

        objective = 0
        for axis in range(3):
            objective += self.config.snap_weight_pos * _snap_integral_ca(
                pos_vars[axis],
                segment_durations,
                self.config.k_pos,
            )
        objective += self.config.snap_weight_yaw * _snap_integral_ca(
            yaw_var,
            segment_durations,
            self.config.k_yaw,
        )
        opti.minimize(objective)

        self._add_position_constraints(
            opti,
            pos_vars,
            positions,
            segment_durations,
            velocities,
            velocity_mask,
            accelerations,
            acceleration_mask,
        )
        self._add_yaw_constraints(
            opti,
            yaw_var,
            yaws,
            segment_durations,
            yaw_rates,
            yaw_rate_mask,
        )
        self._set_initial_guess(
            opti,
            pos_vars,
            yaw_var,
            positions,
            yaws,
            segment_durations,
        )

        opti.solver(
            "ipopt",
            {"expand": True, "print_time": self.config.print_time},
            {
                "print_level": self.config.ipopt_print_level,
                "max_iter": self.config.max_iter,
            },
        )
        solution = opti.solve()

        pos_coeffs = np.zeros(
            (segment_durations.shape[0], self.config.k_pos + 1, 3),
            dtype=np.float64,
        )
        for axis in range(3):
            pos_coeffs[:, :, axis] = solution.value(pos_vars[axis])
        yaw_coeffs = solution.value(yaw_var)

        return PiecewisePolynomialTrajectory(
            pos_coeffs=pos_coeffs,
            yaw_coeffs=yaw_coeffs,
            segment_durations=segment_durations,
            config=self.config,
        )

    def _segment_durations(
        self,
        knot_count: int,
        segment_durations: np.ndarray | None,
    ) -> np.ndarray:
        if segment_durations is None:
            durations = np.full(knot_count - 1, self.config.waypoint_dt, dtype=np.float64)
        else:
            durations = np.asarray(segment_durations, dtype=np.float64).reshape(-1)
        if durations.shape[0] != knot_count - 1:
            raise ValueError("segment_durations must have length knot_count - 1")
        if np.any(durations <= 0.0):
            raise ValueError("all segment durations must be positive")
        return durations

    def _add_position_constraints(
        self,
        opti: ca.Opti,
        pos_vars: list[ca.MX],
        positions: np.ndarray,
        durations: np.ndarray,
        velocities: np.ndarray | None,
        velocity_mask: np.ndarray,
        accelerations: np.ndarray | None,
        acceleration_mask: np.ndarray,
    ) -> None:
        num_segments = durations.shape[0]
        for axis, coeff in enumerate(pos_vars):
            for segment in range(num_segments):
                duration = float(durations[segment])
                opti.subject_to(_poly_ca(coeff, segment, 0.0, 0) == positions[segment, axis])
                opti.subject_to(
                    _poly_ca(coeff, segment, duration, 0)
                    == positions[segment + 1, axis]
                )
                if velocity_mask[segment]:
                    opti.subject_to(
                        _poly_ca(coeff, segment, 0.0, 1) == velocities[segment, axis]
                    )
                if velocity_mask[segment + 1]:
                    opti.subject_to(
                        _poly_ca(coeff, segment, duration, 1)
                        == velocities[segment + 1, axis]
                    )
                if acceleration_mask[segment]:
                    opti.subject_to(
                        _poly_ca(coeff, segment, 0.0, 2)
                        == accelerations[segment, axis]
                    )
                if acceleration_mask[segment + 1]:
                    opti.subject_to(
                        _poly_ca(coeff, segment, duration, 2)
                        == accelerations[segment + 1, axis]
                    )

            for knot in range(1, num_segments):
                left_segment = knot - 1
                right_segment = knot
                left_duration = float(durations[left_segment])
                continuity_orders = [3, 4]
                if not velocity_mask[knot]:
                    continuity_orders.append(1)
                if not acceleration_mask[knot]:
                    continuity_orders.append(2)
                for derivative in sorted(continuity_orders):
                    opti.subject_to(
                        _poly_ca(coeff, left_segment, left_duration, derivative)
                        == _poly_ca(coeff, right_segment, 0.0, derivative)
                    )

    def _add_yaw_constraints(
        self,
        opti: ca.Opti,
        yaw_var: ca.MX,
        yaws: np.ndarray,
        durations: np.ndarray,
        yaw_rates: np.ndarray | None,
        yaw_rate_mask: np.ndarray,
    ) -> None:
        num_segments = durations.shape[0]
        for segment in range(num_segments):
            duration = float(durations[segment])
            opti.subject_to(_poly_ca(yaw_var, segment, 0.0, 0) == yaws[segment])
            opti.subject_to(
                _poly_ca(yaw_var, segment, duration, 0) == yaws[segment + 1]
            )
            if yaw_rate_mask[segment]:
                opti.subject_to(
                    _poly_ca(yaw_var, segment, 0.0, 1) == yaw_rates[segment]
                )
            if yaw_rate_mask[segment + 1]:
                opti.subject_to(
                    _poly_ca(yaw_var, segment, duration, 1) == yaw_rates[segment + 1]
                )

        for knot in range(1, num_segments):
            left_segment = knot - 1
            right_segment = knot
            left_duration = float(durations[left_segment])
            continuity_orders = [2, 3, 4]
            if not yaw_rate_mask[knot]:
                continuity_orders.append(1)
            for derivative in sorted(continuity_orders):
                opti.subject_to(
                    _poly_ca(yaw_var, left_segment, left_duration, derivative)
                    == _poly_ca(yaw_var, right_segment, 0.0, derivative)
                )

    def _set_initial_guess(
        self,
        opti: ca.Opti,
        pos_vars: list[ca.MX],
        yaw_var: ca.MX,
        positions: np.ndarray,
        yaws: np.ndarray,
        durations: np.ndarray,
    ) -> None:
        for segment, duration in enumerate(durations):
            for axis, coeff in enumerate(pos_vars):
                guess = np.zeros(self.config.k_pos + 1)
                guess[0] = positions[segment, axis]
                guess[1] = (
                    positions[segment + 1, axis] - positions[segment, axis]
                ) / duration
                opti.set_initial(coeff[segment, :], guess)

            yaw_guess = np.zeros(self.config.k_yaw + 1)
            yaw_guess[0] = yaws[segment]
            yaw_guess[1] = (yaws[segment + 1] - yaws[segment]) / duration
            opti.set_initial(yaw_var[segment, :], yaw_guess)


def _as_2d(value: np.ndarray, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape [N, {width}]")
    return array


def _optional_masked_2d(
    value: object | None,
    length: int,
    width: int,
    name: str,
) -> tuple[np.ndarray | None, np.ndarray]:
    mask = np.zeros(length, dtype=bool)
    if value is None:
        return None, mask

    object_array = np.asarray(value, dtype=object)
    if object_array.ndim == 2 and object_array.shape == (length, width):
        array = np.asarray(value, dtype=np.float64)
        row_is_nan = np.any(np.isnan(array), axis=1)
        mask = ~row_is_nan
        array[row_is_nan] = 0.0
        return array, mask

    if object_array.ndim != 1 or object_array.shape[0] != length:
        raise ValueError(f"{name} must have length {length}")

    array = np.zeros((length, width), dtype=np.float64)
    for index, item in enumerate(object_array):
        if _is_missing(item):
            continue
        row = np.asarray(item, dtype=np.float64).reshape(-1)
        if row.shape[0] != width:
            raise ValueError(f"{name}[{index}] must have length {width} or be None")
        if np.any(np.isnan(row)):
            continue
        array[index] = row
        mask[index] = True
    return array, mask


def _optional_masked_1d(
    value: object | None,
    length: int,
    name: str,
) -> tuple[np.ndarray | None, np.ndarray]:
    mask = np.zeros(length, dtype=bool)
    if value is None:
        return None, mask

    object_array = np.asarray(value, dtype=object).reshape(-1)
    if object_array.shape[0] != length:
        raise ValueError(f"{name} must have length {length}")

    array = np.zeros(length, dtype=np.float64)
    for index, item in enumerate(object_array):
        if _is_missing(item):
            continue
        scalar = float(item)
        if np.isnan(scalar):
            continue
        array[index] = scalar
        mask[index] = True
    return array, mask


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _poly_ca(coeff: ca.MX, segment: int, tau: float, derivative: int) -> ca.MX:
    degree = coeff.shape[1] - 1
    expr = 0
    for power in range(derivative, degree + 1):
        scale = factorial(power) / factorial(power - derivative)
        expr += coeff[segment, power] * scale * (tau ** (power - derivative))
    return expr


def _snap_integral_ca(coeff: ca.MX, durations: np.ndarray, degree: int) -> ca.MX:
    objective = 0
    for segment, duration in enumerate(durations):
        for i in range(4, degree + 1):
            scale_i = factorial(i) / factorial(i - 4)
            for j in range(4, degree + 1):
                scale_j = factorial(j) / factorial(j - 4)
                power = i + j - 7
                objective += (
                    coeff[segment, i]
                    * coeff[segment, j]
                    * scale_i
                    * scale_j
                    * (float(duration) ** power)
                    / power
                )
    return objective


def _eval_poly_np(coeffs: np.ndarray, tau: float, derivative: int) -> float:
    degree = coeffs.shape[0] - 1
    value = 0.0
    for power in range(derivative, degree + 1):
        scale = factorial(power) / factorial(power - derivative)
        value += coeffs[power] * scale * (tau ** (power - derivative))
    return float(value)
