# Route-Conditioned Multimodal Flying-Crawling Drone Planner

This package implements a modular PyTorch planner for a flying-crawling drone.
It fuses onboard RGB observations, historical self-motion, and ordered task
waypoints, then predicts motion mode, future flight waypoints, crawling action,
and transition actions.

## Structure

```text
mm_planner/
├── config.py
├── main_test.py
├── models/
│   ├── vision_encoder.py
│   ├── trajectory_encoder.py
│   ├── waypoint_encoder.py
│   ├── fusion_transformer.py
│   ├── heads.py
│   └── planner.py
└── utils/
    └── tensor_utils.py
```

## Quick Test

The default vision encoder loads DINOv2 with `torch.hub.load`.

```bash
python -m mm_planner.main_test
```

For an offline smoke test, use the local mock vision backbone:

```bash
python -m mm_planner.main_test --mock-vision
```

## Joystick Demonstration Clip Recording

The MuJoCo teaching environment can record clips while you fly with the
joystick. Push joystick axis 4 above zero to start a clip, then push it below
zero to end and save that clip. You can repeat this many times in one run.

Run it from this package directory:

```bash
conda run -n mm_planner python env/collect_clip_data.py \
  --output-dir data/demos \
  --episode-name perching_demo \
  --fps 20 \
  --width 224 \
  --height 224
```

You can add `--clip-seconds 10` if you also want a maximum automatic segment
length. The default `0` means only axis 4 decides when clips end.

For state-only debugging without camera rendering:

```bash
conda run -n mm_planner python env/collect_clip_data.py --no-rgb
```

The PySide collection app provides live world-follow and FPV views plus editable
task waypoints. Waypoints are shown only in the world-follow view and saved into
each clip when joystick axis 4 starts recording:

```bash
conda run -n mm_planner python env/AppClipCollector.py \
  --output-dir data/demos \
  --episode-name perching_demo
```

Each saved clip includes `task_waypoints_world` (`[N, 3]`) and
`task_waypoints` (`[N, 7]`, planner feature format).

Each episode is written to `data/demos/<episode-name>/` and split into
`clip_000000.npz`, `clip_000001.npz`, and so on. Each clip contains:

- `rgb`: `[T, H, W, 3]` uint8 camera frames from `fpv_camera`, unless `--no-rgb`
- `time`: `[T]`
- `qpos`, `qvel`, `ctrl`, `sensordata`
- `mode_id`: current teaching control mode
- `joystick_axes`: roll, pitch, throttle, yaw, record axis, control-mode axis, takeoff/land axis
- `roll_command`, `pitch_command`, `yaw_rate_command`, `throttle_command`
- `thrust_command`, `target_omega`, `target_velocity`
- `yaw_command`, `position_target`

To visualize recorded data in MuJoCo:

```bash
conda run -n mm_planner python env/replay_clip_data.py data/demos/perching_demo
```

This opens the MuJoCo viewer plus a separate first-person OpenCV window rendered
from `fpv_camera`. Press `q` or `Esc` in the first-person window to stop replay.

You can also replay a single clip and loop it:

```bash
conda run -n mm_planner python env/replay_clip_data.py \
  data/demos/perching_demo/clip_000000.npz \
  --loop \
  --speed 0.5
```

Useful replay options:

- `--fpv-camera fpv_camera`
- `--fpv-width 640 --fpv-height 480`
- `--no-fpv-window`

For a richer dataset browser, use the PySide viewer:

```bash
conda run -n mm_planner python env/clip_viewer_app.py \
  --data-dir data/demos/perching_demo
```

The app provides a clip list, synchronized world/FPV MuJoCo renders, playback
controls, a frame slider, and tabs for position, joystick, control, and command
curves. In the world render, enable `Future points` and set `N` to draw red
markers at the future UAV positions for `0.2s, 0.4s, ..., 0.2*N s`. It needs a
graphical desktop/OpenGL context.

## Inputs

- `rgb`: `[B, 3, H, W]`
- `traj_mode_ids`: `[B, n_traj_encoder]`
- `traj_continuous`: `[B, n_traj_encoder, traj_continuous_dim]`
- `task_waypoints`: `[B, n_waypoints, waypoint_dim]`
- `task_waypoint_mask`: optional `[B, n_waypoints]`, where `True` means padding

## Outputs

- `mode_logits`: `[B, num_modes]`
- `flight_waypoints.mu/sigma`: `[B, pred_num_flight_waypoints, pred_waypoint_dim]`
- `flight_velocity_deltas.mu/sigma`: `[B, pred_num_flight_waypoints, 4]`
- `flight_acceleration_deltas.mu/sigma`: `[B, pred_num_flight_waypoints, 3]`
- `crawl_action`: `[B, 3]`
- `transition.contact_pos`: `[B, 3]`
- `transition.surface_normal`: `[B, 3]`
- `transition.yaw`: `[B, 1]`
- `transition.approach_speed`: `[B, 1]`
- `fused_feature`: `[B, fusion_dim]`

The flight waypoint, velocity-delta, and acceleration-delta heads all generate
Gaussian action distributions autoregressively. Each returns `mu` and positive
`sigma`, and each supports optional teacher forcing during training.
