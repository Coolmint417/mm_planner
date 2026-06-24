from __future__ import annotations

import argparse

import mujoco.viewer as viewer

from clip_recorder import ClipRecorderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joystick teaching with multimodal clip recording.",
    )
    parser.add_argument("--output-dir", default="data/demos")
    parser.add_argument("--episode-name", default="perching_demo")
    parser.add_argument("--camera-name", default="fpv_camera")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=0.0,
        help="Optional max segment length. 0 disables automatic splitting.",
    )
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--uncompressed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import perching_uav_teach as teach

    config = ClipRecorderConfig(
        output_dir=args.output_dir,
        episode_name=args.episode_name,
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
        fps=args.fps,
        clip_seconds=args.clip_seconds,
        save_rgb=not args.no_rgb,
        compressed=not args.uncompressed,
    )
    teach.set_recording_config(config)
    try:
        viewer.launch(loader=teach.load_callback)
    finally:
        teach.close_recorder()


if __name__ == "__main__":
    main()
