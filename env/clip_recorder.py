from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


@dataclass
class ClipRecorderConfig:
    output_dir: str = "data/demos"
    episode_name: str = "demo"
    camera_name: str = "fpv_camera"
    width: int = 224
    height: int = 224
    fps: float = 20.0
    clip_seconds: float = 0.0
    save_rgb: bool = True
    compressed: bool = True

    @property
    def frames_per_clip(self) -> int:
        return max(1, int(round(self.fps * self.clip_seconds)))


class ClipRecorder:
    """Records joystick-triggered MuJoCo demonstration clips."""

    def __init__(self, model: mujoco.MjModel, config: ClipRecorderConfig) -> None:
        self.model = model
        self.config = config
        self.output_dir = Path(config.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_dir = self._make_episode_dir()
        self.renderer = (
            mujoco.Renderer(model, height=config.height, width=config.width)
            if config.save_rgb
            else None
        )
        self.frame_dt = 1.0 / config.fps
        self.next_sample_time = 0.0
        self.clip_index = 0
        self.active = False
        self.segment_start_time = 0.0
        self.buffer: dict[str, list[np.ndarray]] = {}
        self.clip_metadata: dict[str, np.ndarray] = {}
        self.current_clip_metadata: dict[str, np.ndarray] = {}
        self.metadata = {
            "format": "mm_planner_demo_clip_v1",
            "config": asdict(config),
            "qpos_dim": int(model.nq),
            "qvel_dim": int(model.nv),
            "ctrl_dim": int(model.nu),
            "sensor_dim": int(model.nsensordata),
        }
        self._write_metadata()
        print(f"Recording clips to: {self.episode_dir}")

    @property
    def has_max_clip_length(self) -> bool:
        return self.config.clip_seconds > 0.0

    def _make_episode_dir(self) -> Path:
        base = self.output_dir / self.config.episode_name
        if not base.exists():
            base.mkdir(parents=True)
            return base
        index = 1
        while True:
            candidate = self.output_dir / f"{self.config.episode_name}_{index:03d}"
            if not candidate.exists():
                candidate.mkdir(parents=True)
                return candidate
            index += 1

    def _write_metadata(self) -> None:
        metadata_path = self.episode_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def sample(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        extra: dict[str, Any],
    ) -> None:
        if not self.active:
            return
        if data.time + 1e-9 < self.next_sample_time:
            return
        self.next_sample_time = data.time + self.frame_dt

        frame = self._build_frame(model, data, extra)
        for key, value in frame.items():
            self.buffer.setdefault(key, []).append(value)

        if (
            self.has_max_clip_length
            and self.num_buffered_frames >= self.config.frames_per_clip
        ):
            self.flush()
            self.begin(data.time)

    def begin(self, current_time: float) -> None:
        if self.active:
            return
        self.buffer.clear()
        self.current_clip_metadata = {
            key: np.asarray(value).copy()
            for key, value in self.clip_metadata.items()
        }
        self.active = True
        self.segment_start_time = float(current_time)
        self.next_sample_time = float(current_time)
        print(f"Start recording clip {self.clip_index:06d} at t={current_time:.3f}s")

    def set_clip_metadata(self, metadata: dict[str, Any]) -> None:
        self.clip_metadata = {
            key: np.asarray(value).copy()
            for key, value in metadata.items()
        }

    def end(self) -> None:
        if not self.active:
            return
        self.active = False
        self.flush()

    @property
    def num_buffered_frames(self) -> int:
        if not self.buffer:
            return 0
        first_key = next(iter(self.buffer))
        return len(self.buffer[first_key])

    def _build_frame(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        extra: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        frame: dict[str, np.ndarray] = {
            "time": np.asarray(data.time, dtype=np.float64),
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "ctrl": data.ctrl.copy(),
            "sensordata": data.sensordata.copy(),
        }
        if self.renderer is not None:
            self.renderer.update_scene(data, camera=self.config.camera_name)
            frame["rgb"] = self.renderer.render().copy()

        for key, value in extra.items():
            frame[key] = np.asarray(value)
        return frame

    def flush(self) -> None:
        if self.num_buffered_frames == 0:
            return

        arrays = {
            key: np.stack(values, axis=0)
            for key, values in self.buffer.items()
        }
        arrays.update(self.current_clip_metadata)
        arrays["metadata_json"] = np.asarray(json.dumps(self.metadata))
        output_path = self.episode_dir / f"clip_{self.clip_index:06d}.npz"
        if self.config.compressed:
            np.savez_compressed(output_path, **arrays)
        else:
            np.savez(output_path, **arrays)

        print(f"Saved {output_path} ({self.num_buffered_frames} frames)")
        self.clip_index += 1
        self.buffer.clear()

    def close(self) -> None:
        self.end()
        if self.renderer is not None:
            self.renderer.close()
