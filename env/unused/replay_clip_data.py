from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer as viewer
import numpy as np


DEFAULT_XML_PATH = Path(__file__).resolve().parent / "scene.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay recorded mm_planner demonstration clips in MuJoCo.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to one clip_*.npz file or an episode directory containing clips.",
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--hide-left-ui", action="store_true", default=True)
    parser.add_argument("--hide-right-ui", action="store_true", default=True)
    parser.add_argument("--fpv-camera", default="fpv_camera")
    parser.add_argument("--fpv-width", type=int, default=640)
    parser.add_argument("--fpv-height", type=int, default=480)
    parser.add_argument("--no-fpv-window", action="store_true")
    parser.add_argument(
        "--ignore-ctrl",
        action="store_true",
        help="Replay only qpos/qvel and leave MuJoCo controls untouched.",
    )
    return parser.parse_args()


class FpvWindow:
    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str,
        width: int,
        height: int,
    ) -> None:
        self.camera_name = camera_name
        self.window_name = f"FPV: {camera_name}"
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width, height)

    def render(self, data: mujoco.MjData) -> bool:
        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = self.renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, bgr)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        self.renderer.close()
        cv2.destroyWindow(self.window_name)


def discover_clips(path: Path) -> list[Path]:
    path = path.expanduser()
    if path.is_file():
        return [path]
    if path.is_dir():
        clips = sorted(path.glob("clip_*.npz"))
        if clips:
            return clips
    raise FileNotFoundError(f"No clip_*.npz files found at {path}")


def validate_clip(clip: np.lib.npyio.NpzFile, model: mujoco.MjModel) -> int:
    if "qpos" not in clip:
        raise KeyError("Clip does not contain required field: qpos")
    qpos = clip["qpos"]
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos shape {qpos.shape} does not match model.nq={model.nq}")

    if "qvel" in clip:
        qvel = clip["qvel"]
        if qvel.ndim != 2 or qvel.shape[1] != model.nv:
            raise ValueError(f"qvel shape {qvel.shape} does not match model.nv={model.nv}")
        if qvel.shape[0] != qpos.shape[0]:
            raise ValueError("qvel and qpos have different frame counts")

    if "ctrl" in clip:
        ctrl = clip["ctrl"]
        if ctrl.ndim != 2 or ctrl.shape[1] != model.nu:
            raise ValueError(f"ctrl shape {ctrl.shape} does not match model.nu={model.nu}")
        if ctrl.shape[0] != qpos.shape[0]:
            raise ValueError("ctrl and qpos have different frame counts")

    return int(qpos.shape[0])


def frame_delay_seconds(
    clip: np.lib.npyio.NpzFile,
    frame_index: int,
    speed: float,
    fallback_dt: float,
) -> float:
    if speed <= 0.0:
        return 0.0
    if "time" not in clip or frame_index <= 0:
        return fallback_dt / speed

    times = clip["time"]
    dt = float(times[frame_index] - times[frame_index - 1])
    if dt <= 0.0:
        dt = fallback_dt
    return dt / speed


def apply_frame(
    data: mujoco.MjData,
    clip: np.lib.npyio.NpzFile,
    frame_index: int,
    ignore_ctrl: bool,
) -> None:
    data.qpos[:] = clip["qpos"][frame_index]
    if "qvel" in clip:
        data.qvel[:] = clip["qvel"][frame_index]
    if not ignore_ctrl and "ctrl" in clip:
        data.ctrl[:] = clip["ctrl"][frame_index]
    if "time" in clip:
        data.time = float(clip["time"][frame_index])


def replay_clip(
    handle: viewer.Handle,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip_path: Path,
    args: argparse.Namespace,
    fpv_window: FpvWindow | None,
) -> bool:
    with np.load(clip_path, allow_pickle=False) as clip:
        num_frames = validate_clip(clip, model)
        start_frame = min(max(args.start_frame, 0), num_frames - 1)
        fallback_dt = model.opt.timestep
        if "time" in clip and num_frames > 1:
            diffs = np.diff(clip["time"])
            positive_diffs = diffs[diffs > 0.0]
            if positive_diffs.size > 0:
                fallback_dt = float(np.median(positive_diffs))

        print(f"Replaying {clip_path} ({num_frames} frames)")
        for frame_index in range(start_frame, num_frames):
            if not handle.is_running():
                return False

            tick = time.monotonic()
            apply_frame(data, clip, frame_index, args.ignore_ctrl)
            mujoco.mj_forward(model, data)
            handle.sync()
            if fpv_window is not None and not fpv_window.render(data):
                return False

            delay = frame_delay_seconds(clip, frame_index, args.speed, fallback_dt)
            elapsed = time.monotonic() - tick
            if delay > elapsed:
                time.sleep(delay - elapsed)

    return True


def main() -> None:
    args = parse_args()
    clips = discover_clips(args.path)
    model = mujoco.MjModel.from_xml_path(str(args.xml.expanduser()))
    data = mujoco.MjData(model)
    fpv_window = None if args.no_fpv_window else FpvWindow(
        model=model,
        camera_name=args.fpv_camera,
        width=args.fpv_width,
        height=args.fpv_height,
    )

    try:
        with viewer.launch_passive(
            model,
            data,
            show_left_ui=not args.hide_left_ui,
            show_right_ui=not args.hide_right_ui,
        ) as handle:
            while handle.is_running():
                for clip_path in clips:
                    if not replay_clip(handle, model, data, clip_path, args, fpv_window):
                        return
                if not args.loop:
                    break
                args.start_frame = 0
    finally:
        if fpv_window is not None:
            fpv_window.close()


if __name__ == "__main__":
    main()
