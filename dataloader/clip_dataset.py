from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info

from config import DataConfig, ModelConfig


@dataclass(frozen=True)
class ClipInfo:
    path: Path
    num_frames: int
    start_time: float
    end_time: float


class ClipWindowDataset(Dataset):
    """Samples random temporal windows from recorded mm_planner npz clips."""

    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        seed: int = 42,
    ) -> None:
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.seed = seed
        self.data_dir = Path(data_cfg.data_dir).expanduser()
        self.clip_infos = self._discover_clips()
        self._clip_cache: dict[Path, dict[str, np.ndarray]] = {}

        if not self.clip_infos:
            raise FileNotFoundError(
                f"No usable clips found under {self.data_dir} "
                f"with pattern {data_cfg.clip_pattern}"
            )

    def __len__(self) -> int:
        return self.data_cfg.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = self._rng_for_index(index)
        clip_info = self.clip_infos[int(rng.integers(0, len(self.clip_infos)))]
        clip = self._load_clip(clip_info.path)
        anchor_time = self._sample_anchor_time(clip_info, rng)
        anchor_index = self._index_at_or_after(clip["time"], anchor_time)

        history_indices = self._sample_time_indices(
            clip["time"],
            anchor_time,
            count=self.model_cfg.n_traj_encoder,
            dt=-self.data_cfg.history_dt,
            reverse_to_chronological=True,
        )
        future_indices = self._sample_time_indices(
            clip["time"],
            anchor_time,
            count=self.model_cfg.pred_num_flight_waypoints,
            dt=self.model_cfg.pred_flight_waypoint_dt,
            start_step=1,
        )

        rgb = self._build_rgb(clip, anchor_index)
        traj_mode_ids = self._build_history_modes(clip, history_indices)
        traj_continuous = self._build_history_continuous(clip, history_indices)
        task_waypoints = self._build_task_waypoints(clip, anchor_index)

        future_waypoints = self._build_future_waypoints(clip, future_indices, anchor_index)
        future_velocities = self._linear_velocity(clip, future_indices)
        future_accelerations = self._linear_acceleration(clip, future_indices)
        future_yaw = self._yaw(clip["qpos"][future_indices])
        future_yaw_rate = self._yaw_rate(clip, future_indices)

        anchor_velocity = self._linear_velocity(clip, np.asarray([anchor_index]))[0]
        anchor_acceleration = self._linear_acceleration(clip, np.asarray([anchor_index]))[0]
        anchor_yaw_rate = self._yaw_rate(clip, np.asarray([anchor_index]))[0]

        velocity_deltas = np.concatenate(
            [
                future_velocities - anchor_velocity,
                (future_yaw_rate - anchor_yaw_rate)[:, None],
            ],
            axis=-1,
        )
        acceleration_deltas = future_accelerations - anchor_acceleration

        target_mode_id = self._mode_at(clip, anchor_index)

        return {
            "rgb": torch.from_numpy(rgb).float(),
            "traj_mode_ids": torch.from_numpy(traj_mode_ids).long(),
            "traj_continuous": torch.from_numpy(traj_continuous).float(),
            "task_waypoints": torch.from_numpy(task_waypoints).float(),
            "target_mode_id": torch.tensor(target_mode_id, dtype=torch.long),
            "teacher_flight_waypoints": torch.from_numpy(future_waypoints).float(),
            "teacher_flight_velocity_deltas": torch.from_numpy(velocity_deltas).float(),
            "teacher_flight_acceleration_deltas": torch.from_numpy(acceleration_deltas).float(),
            "future_positions": torch.from_numpy(clip["qpos"][future_indices, :3]).float(),
            "future_velocities": torch.from_numpy(future_velocities).float(),
            "future_accelerations": torch.from_numpy(future_accelerations).float(),
            "future_yaw": torch.from_numpy(future_yaw).float(),
            "future_yaw_rate": torch.from_numpy(future_yaw_rate).float(),
            "anchor_index": torch.tensor(anchor_index, dtype=torch.long),
            "anchor_time": torch.tensor(float(clip["time"][anchor_index]), dtype=torch.float32),
        }

    def _discover_clips(self) -> list[ClipInfo]:
        paths = sorted(self.data_dir.rglob(self.data_cfg.clip_pattern))
        clips: list[ClipInfo] = []
        required_future = (
            self.model_cfg.pred_num_flight_waypoints
            * self.model_cfg.pred_flight_waypoint_dt
        )
        required_history = (self.model_cfg.n_traj_encoder - 1) * self.data_cfg.history_dt

        for path in paths:
            try:
                with np.load(path, allow_pickle=False) as npz:
                    if not self._has_required_fields(npz):
                        continue
                    time = np.asarray(npz["time"], dtype=np.float64)
                    if time.ndim != 1 or time.size < 2:
                        continue
                    if time[-1] - time[0] < required_history + required_future:
                        continue
                    clips.append(
                        ClipInfo(
                            path=path,
                            num_frames=int(time.size),
                            start_time=float(time[0]),
                            end_time=float(time[-1]),
                        )
                    )
            except Exception:
                continue
        return clips

    def _has_required_fields(self, npz: Any) -> bool:
        required = {"time", "qpos", "qvel", self.data_cfg.rgb_key}
        return required.issubset(set(npz.files))

    def _load_clip(self, path: Path) -> dict[str, np.ndarray]:
        if self.data_cfg.cache_clips and path in self._clip_cache:
            return self._clip_cache[path]

        with np.load(path, allow_pickle=False) as npz:
            clip = {key: npz[key].copy() for key in npz.files}

        if self.data_cfg.cache_clips:
            self._clip_cache[path] = clip
        return clip

    def _rng_for_index(self, index: int) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        seed = self.seed + index * 1009 + worker_id * 9176
        return np.random.default_rng(seed)

    def _sample_anchor_time(
        self,
        clip_info: ClipInfo,
        rng: np.random.Generator,
    ) -> float:
        history = (self.model_cfg.n_traj_encoder - 1) * self.data_cfg.history_dt
        future = (
            self.model_cfg.pred_num_flight_waypoints
            * self.model_cfg.pred_flight_waypoint_dt
        )
        low = clip_info.start_time + history
        high = clip_info.end_time - future
        return float(rng.uniform(low, high))

    def _sample_time_indices(
        self,
        times: np.ndarray,
        anchor_time: float,
        count: int,
        dt: float,
        start_step: int = 0,
        reverse_to_chronological: bool = False,
    ) -> np.ndarray:
        sample_times = np.asarray(
            [anchor_time + dt * step for step in range(start_step, start_step + count)],
            dtype=np.float64,
        )
        if reverse_to_chronological:
            sample_times = sample_times[::-1]
        indices = np.searchsorted(times, sample_times, side="left")
        return np.clip(indices, 0, len(times) - 1).astype(np.int64)

    def _index_at_or_after(self, times: np.ndarray, timestamp: float) -> int:
        return int(np.clip(np.searchsorted(times, timestamp, side="left"), 0, len(times) - 1))

    def _build_rgb(self, clip: dict[str, np.ndarray], anchor_index: int) -> np.ndarray:
        image = clip[self.data_cfg.rgb_key][anchor_index]
        image = image.astype(np.float32) / 255.0
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected RGB image [H, W, 3], got {image.shape}")
        image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image).unsqueeze(0)
        if tensor.shape[-2:] != (self.model_cfg.image_size, self.model_cfg.image_size):
            tensor = F.interpolate(
                tensor,
                size=(self.model_cfg.image_size, self.model_cfg.image_size),
                mode="bilinear",
                align_corners=False,
            )
        mean = torch.tensor(self.data_cfg.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(self.data_cfg.image_std).view(1, 3, 1, 1)
        return ((tensor - mean) / std).squeeze(0).numpy()

    def _build_history_modes(
        self,
        clip: dict[str, np.ndarray],
        indices: np.ndarray,
    ) -> np.ndarray:
        mode_key = self._available_mode_key(clip)
        if not mode_key:
            return np.zeros((len(indices),), dtype=np.int64)
        modes = np.asarray(clip[mode_key][indices], dtype=np.int64)
        return np.clip(modes, 0, self.model_cfg.num_motion_modes - 1)

    def _build_history_continuous(
        self,
        clip: dict[str, np.ndarray],
        indices: np.ndarray,
    ) -> np.ndarray:
        qpos = clip["qpos"][indices]
        qvel = clip["qvel"][indices]
        pos = qpos[:, :3]
        vel = qvel[:, :3]
        acc = self._linear_acceleration(clip, indices)
        quat = qpos[:, 3:7]
        omega = qvel[:, 3:6] if qvel.shape[1] >= 6 else np.zeros_like(vel)
        continuous = np.concatenate([pos, vel, acc, quat, omega], axis=-1)
        return continuous.astype(np.float32)

    def _build_task_waypoints(
        self,
        clip: dict[str, np.ndarray],
        anchor_index: int,
    ) -> np.ndarray:
        if self.data_cfg.task_waypoints_key in clip:
            raw = np.asarray(clip[self.data_cfg.task_waypoints_key], dtype=np.float32)
            raw = raw.reshape(-1, raw.shape[-1])
        else:
            raw = self._default_task_waypoint(clip, anchor_index)

        if raw.shape[0] == 0:
            raw = self._default_task_waypoint(clip, anchor_index)

        count = raw.shape[0]
        dims = min(self.model_cfg.waypoint_dim, raw.shape[1])
        waypoints = np.zeros((count, self.model_cfg.waypoint_dim), dtype=np.float32)
        waypoints[:count, :dims] = raw[:count, :dims]
        return waypoints

    def _default_task_waypoint(
        self,
        clip: dict[str, np.ndarray],
        anchor_index: int,
    ) -> np.ndarray:
        position = clip["qpos"][-1, :3]
        yaw = self._yaw(clip["qpos"][[-1]])[0]
        current = clip["qpos"][anchor_index, :3]
        direction = position - current
        waypoint = np.zeros((1, self.model_cfg.waypoint_dim), dtype=np.float32)
        waypoint[0, :3] = position
        if self.model_cfg.waypoint_dim > 3:
            waypoint[0, 3] = yaw
        if self.model_cfg.waypoint_dim > 4:
            waypoint[0, 4:7] = direction[: min(3, self.model_cfg.waypoint_dim - 4)]
        return waypoint

    def _build_future_waypoints(
        self,
        clip: dict[str, np.ndarray],
        indices: np.ndarray,
        anchor_index: int,
    ) -> np.ndarray:
        qpos = clip["qpos"]
        future_pos = qpos[indices, :3]
        future_yaw = self._yaw(qpos[indices])[:, None]
        waypoint = np.concatenate([future_pos, future_yaw], axis=-1)

        if self.data_cfg.relative_targets:
            anchor_pos = qpos[anchor_index, :3]
            anchor_yaw = self._yaw(qpos[[anchor_index]])[0]
            waypoint[:, :3] -= anchor_pos
            waypoint[:, 3] = self._wrap_angle(waypoint[:, 3] - anchor_yaw)

        return waypoint.astype(np.float32)

    def _available_mode_key(self, clip: dict[str, np.ndarray]) -> str:
        if self.data_cfg.mode_key in clip:
            return self.data_cfg.mode_key
        if self.data_cfg.fallback_mode_key in clip:
            return self.data_cfg.fallback_mode_key
        return ""

    def _mode_at(self, clip: dict[str, np.ndarray], index: int) -> int:
        mode_key = self._available_mode_key(clip)
        if not mode_key:
            return 0
        mode = int(np.asarray(clip[mode_key][index]).reshape(-1)[0])
        return int(np.clip(mode, 0, self.model_cfg.num_modes - 1))

    def _linear_velocity(
        self,
        clip: dict[str, np.ndarray],
        indices: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(clip["qvel"][indices, :3], dtype=np.float32)

    def _linear_acceleration(
        self,
        clip: dict[str, np.ndarray],
        indices: np.ndarray,
    ) -> np.ndarray:
        times = np.asarray(clip["time"], dtype=np.float64)
        velocities = np.asarray(clip["qvel"][:, :3], dtype=np.float32)
        acc = np.zeros_like(velocities)
        if len(times) > 1:
            for dim in range(3):
                acc[:, dim] = np.gradient(velocities[:, dim], times)
        return acc[indices].astype(np.float32)

    def _yaw_rate(
        self,
        clip: dict[str, np.ndarray],
        indices: np.ndarray,
    ) -> np.ndarray:
        qvel = clip["qvel"]
        if qvel.shape[1] >= 6:
            return np.asarray(qvel[indices, 5], dtype=np.float32)
        return np.zeros((len(indices),), dtype=np.float32)

    def _yaw(self, qpos: np.ndarray) -> np.ndarray:
        quat = qpos[:, 3:7].astype(np.float64)
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return np.arctan2(siny_cosp, cosy_cosp).astype(np.float32)

    def _wrap_angle(self, angle: np.ndarray) -> np.ndarray:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi


def collate_clip_samples(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    keys = samples[0].keys()
    batch: dict[str, torch.Tensor | list[torch.Tensor]] = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        if key == "task_waypoints":
            batch[key] = values
        else:
            batch[key] = torch.stack(values, dim=0)
    return batch


def build_dataloader(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    dataset = ClipWindowDataset(data_cfg=data_cfg, model_cfg=model_cfg, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_clip_samples,
        drop_last=True,
    )
