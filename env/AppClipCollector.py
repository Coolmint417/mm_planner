from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from clip_recorder import ClipRecorderConfig


teach = None


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "demos"


class ImageView(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 320)
        self.setStyleSheet("background: #111; color: #ddd;")

    def set_rgb(self, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class WaypointTable(QTableWidget):
    def __init__(self) -> None:
        super().__init__(0, 3)
        self.setHorizontalHeaderLabels(["x", "y", "z"])
        self.horizontalHeader().setStretchLastSection(True)

    def add_waypoint(self, point: np.ndarray | None = None) -> None:
        if point is None:
            point = np.zeros(3)
        row = self.rowCount()
        self.insertRow(row)
        for col, value in enumerate(point):
            item = QTableWidgetItem(f"{float(value):.3f}")
            item.setTextAlignment(Qt.AlignCenter)
            self.setItem(row, col, item)

    def remove_selected_waypoint(self) -> None:
        rows = sorted({index.row() for index in self.selectedIndexes()}, reverse=True)
        for row in rows:
            self.removeRow(row)

    def waypoints(self) -> np.ndarray:
        points = []
        for row in range(self.rowCount()):
            values = []
            for col in range(3):
                item = self.item(row, col)
                text = item.text() if item is not None else "0"
                values.append(float(text))
            points.append(values)
        if not points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(points, dtype=np.float32)


class ClipCollectorWindow(QMainWindow):
    def __init__(
        self,
        output_dir: Path,
        episode_name: str,
        fps: float,
        save_rgb: bool,
        render_fps: int,
    ) -> None:
        super().__init__()
        self.setWindowTitle("mm_planner Clip Collector")
        self.resize(1500, 900)

        self.output_dir = output_dir
        self.episode_name = episode_name
        self.record_fps = fps
        self.save_rgb = save_rgb
        self.render_fps = render_fps
        self.model = None
        self.data = None
        self.world_renderer = None
        self.fpv_renderer = None
        self.world_camera = mujoco.MjvCamera()
        self.configure_world_tracking_camera()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step_and_render)

        self._build_ui()
        self.start_simulation()

    def _build_ui(self) -> None:
        self.world_view = ImageView("World Follow Camera")
        self.fpv_view = ImageView("FPV Camera")
        image_splitter = QSplitter(Qt.Horizontal)
        image_splitter.addWidget(self.world_view)
        image_splitter.addWidget(self.fpv_view)

        self.output_edit = QLineEdit(str(self.output_dir))
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.choose_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)
        output_row.addWidget(browse_button)

        self.episode_edit = QLineEdit(self.episode_name)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 120.0)
        self.fps_spin.setValue(self.record_fps)
        self.fps_spin.setSingleStep(1.0)

        form = QFormLayout()
        form.addRow("Output", output_row)
        form.addRow("Episode", self.episode_edit)
        form.addRow("Record FPS", self.fps_spin)

        self.waypoint_table = WaypointTable()
        self.waypoint_table.add_waypoint(np.array([0.5, 0.0, 0.3]))
        self.waypoint_table.add_waypoint(np.array([1.0, 0.0, 0.5]))

        add_button = QPushButton("Add")
        add_button.clicked.connect(lambda: self.waypoint_table.add_waypoint())
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.waypoint_table.remove_selected_waypoint)
        reset_button = QPushButton("Restart Sim")
        reset_button.clicked.connect(self.restart_simulation)

        button_row = QHBoxLayout()
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        button_row.addWidget(reset_button)

        self.status_label = QLabel(
            "Axis 4 > 0 starts recording, axis 4 < 0 saves.\n"
            "In position mode, axis 7 negative->positive takes off, "
            "positive->negative lands."
        )
        self.status_label.setWordWrap(True)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addLayout(form)
        left_layout.addWidget(QLabel("Task Waypoints (world frame)"))
        left_layout.addWidget(self.waypoint_table)
        left_layout.addLayout(button_row)
        left_layout.addWidget(self.status_label)

        root = QSplitter(Qt.Horizontal)
        root.addWidget(left_panel)
        root.addWidget(image_splitter)
        root.setSizes([360, 1140])
        self.setCentralWidget(root)

    def choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Choose output directory",
            self.output_edit.text(),
        )
        if directory:
            self.output_edit.setText(directory)

    def start_simulation(self) -> None:
        if teach is None:
            raise RuntimeError("perching_uav_teach module is not initialized")
        config = ClipRecorderConfig(
            output_dir=self.output_edit.text(),
            episode_name=self.episode_edit.text(),
            fps=float(self.fps_spin.value()),
            save_rgb=self.save_rgb,
        )
        teach.set_recording_config(config)
        teach.set_recording_metadata_provider(self.current_clip_metadata)
        self.model, self.data = teach.load_callback()
        self.configure_world_tracking_camera()
        self.world_renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.fpv_renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.timer.start(max(1, int(1000 / self.render_fps)))

    def configure_world_tracking_camera(self) -> None:
        self.world_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.world_camera.trackbodyid = -1
        if self.model is not None:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "Body",
            )
            if body_id >= 0:
                self.world_camera.trackbodyid = body_id
            else:
                self.world_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.world_camera.distance = 1.0
        self.world_camera.azimuth = 0.0
        self.world_camera.elevation = -15

    def restart_simulation(self) -> None:
        self.timer.stop()
        self.close_renderers()
        teach.close_recorder()
        self.start_simulation()

    def close_renderers(self) -> None:
        if self.world_renderer is not None:
            self.world_renderer.close()
            self.world_renderer = None
        if self.fpv_renderer is not None:
            self.fpv_renderer.close()
            self.fpv_renderer = None

    def current_clip_metadata(self) -> dict[str, np.ndarray]:
        world_points = self.waypoint_table.waypoints()
        planner_points = self.make_planner_waypoints(world_points)
        return {
            "task_waypoints_world": world_points,
            "task_waypoints": planner_points,
            "task_waypoint_count": np.asarray(world_points.shape[0], dtype=np.int64),
        }

    def make_planner_waypoints(self, world_points: np.ndarray) -> np.ndarray:
        if self.data is None or world_points.size == 0:
            return np.zeros((0, 7), dtype=np.float32)
        pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7].copy()
        rotation = self.quat_wxyz_to_mat(quat)
        relative = (rotation.T @ (world_points - pos).T).T
        distance = np.linalg.norm(relative, axis=1, keepdims=True)
        relative_yaw = np.zeros((world_points.shape[0], 1), dtype=np.float32)
        is_current = np.zeros((world_points.shape[0], 1), dtype=np.float32)
        if world_points.shape[0] > 0:
            is_current[0, 0] = 1.0
        goal_type = np.zeros((world_points.shape[0], 1), dtype=np.float32)
        return np.concatenate(
            [relative, relative_yaw, distance, is_current, goal_type],
            axis=1,
        ).astype(np.float32)

    @staticmethod
    def quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def step_and_render(self) -> None:
        if self.model is None or self.data is None:
            return
        target_dt = 1.0 / self.render_fps
        steps = max(1, int(round(target_dt / self.model.opt.timestep)))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self.render_views()
        self.update_status()

    def render_views(self) -> None:
        points = self.waypoint_table.waypoints()
        if self.world_camera.type == mujoco.mjtCamera.mjCAMERA_FREE:
            self.world_camera.lookat[:] = self.data.qpos[:3]
        self.world_renderer.update_scene(self.data, camera="fpv_camera_far")  # self.world_camera
        self.add_waypoint_markers(self.world_renderer, points)
        self.world_view.set_rgb(self.world_renderer.render())

        self.fpv_renderer.update_scene(self.data, camera="fpv_camera")
        self.fpv_view.set_rgb(self.fpv_renderer.render())

    def add_waypoint_markers(self, renderer: mujoco.Renderer, points: np.ndarray) -> None:
        scene = renderer.scene
        for index, point in enumerate(points):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            rgba = np.array([0.1, 0.9, 0.25, 1.0], dtype=np.float32)
            if index == 0:
                rgba = np.array([1.0, 0.2, 0.1, 1.0], dtype=np.float32)
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.05, 0.05, 0.05], dtype=np.float64),
                point.astype(np.float64),
                np.eye(3, dtype=np.float64).reshape(-1),
                rgba,
            )
            scene.ngeom += 1

    def update_status(self) -> None:
        recorder = teach.clip_recorder
        if recorder is None:
            self.status_label.setText("Recorder is not initialized.")
            return
        state = "RECORDING" if recorder.active else "idle"
        frames = recorder.num_buffered_frames
        points = self.waypoint_table.waypoints().shape[0]
        self.status_label.setText(
            f"State: {state}\n"
            f"Buffered frames: {frames}\n"
            f"Task waypoints: {points}\n"
            "Axis 4 > 0 starts recording, axis 4 < 0 saves.\n"
            "Position mode axis 7: negative->positive takeoff, "
            "positive->negative landing."
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self.timer.stop()
        self.close_renderers()
        teach.close_recorder()
        mujoco.set_mjcb_control(None)
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PySide joystick clip collector.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-name", default="perching_demo")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--no-rgb", action="store_true")
    return parser.parse_args()


def main() -> None:
    global teach
    args = parse_args()
    import perching_uav_teach

    teach = perching_uav_teach
    app = QApplication(sys.argv)
    window = ClipCollectorWindow(
        output_dir=args.output_dir,
        episode_name=args.episode_name,
        fps=args.fps,
        save_rgb=not args.no_rgb,
        render_fps=args.render_fps,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
