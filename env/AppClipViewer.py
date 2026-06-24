from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


DEFAULT_XML_PATH = Path(__file__).resolve().parent / "scene.xml"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "demos"


class ImageView(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background: #111; color: #ddd;")

    def set_rgb(self, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
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


class ClipViewer(QMainWindow):
    def __init__(
        self,
        data_dir: Path,
        xml_path: Path,
        world_camera: str,
        fpv_camera: str,
        follow_world_camera: bool,
    ) -> None:
        super().__init__()
        self.setWindowTitle("mm_planner Clip Viewer")
        self.resize(1500, 950)

        self.xml_path = xml_path
        self.world_camera = world_camera
        self.fpv_camera = fpv_camera
        self.follow_world_camera = follow_world_camera
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.world_renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.fpv_renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.world_follow_camera = mujoco.MjvCamera()
        self.world_follow_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.world_follow_camera.distance = 1.4
        self.world_follow_camera.azimuth = -35.0
        self.world_follow_camera.elevation = -25.0

        self.data_dir = data_dir
        self.clip_paths: list[Path] = []
        self.clip: dict[str, np.ndarray] = {}
        self.current_frame = 0
        self.is_playing = False
        self.plot_lines: list[pg.InfiniteLine] = []

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self._build_ui()
        self.load_directory(data_dir)

    def _build_ui(self) -> None:
        toolbar = QToolBar("Controls")
        self.addToolBar(toolbar)

        open_action = QAction("Open Dir", self)
        open_action.triggered.connect(self.choose_directory)
        toolbar.addAction(open_action)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        toolbar.addWidget(self.play_button)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.set_frame_from_slider)
        toolbar.addWidget(self.frame_slider)

        self.frame_label = QLabel("Frame: 0 / 0")
        toolbar.addWidget(self.frame_label)

        toolbar.addWidget(QLabel(" FPS "))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 240)
        self.fps_spin.setValue(10)
        self.fps_spin.valueChanged.connect(self.update_timer_interval)
        toolbar.addWidget(self.fps_spin)

        self.show_future_points_checkbox = QCheckBox("Future points")
        self.show_future_points_checkbox.setChecked(True)
        self.show_future_points_checkbox.stateChanged.connect(lambda _: self.render_frame())
        toolbar.addWidget(self.show_future_points_checkbox)

        toolbar.addWidget(QLabel(" N "))
        self.future_points_spin = QSpinBox()
        self.future_points_spin.setRange(0, 200)
        self.future_points_spin.setValue(5)
        self.future_points_spin.valueChanged.connect(lambda _: self.render_frame())
        toolbar.addWidget(self.future_points_spin)

        self.clip_list = QListWidget()
        self.clip_list.currentRowChanged.connect(self.load_clip_by_row)

        self.metadata_label = QLabel("No clip loaded")
        self.metadata_label.setAlignment(Qt.AlignTop)
        self.metadata_label.setWordWrap(True)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Clips"))
        left_layout.addWidget(self.clip_list, 3)
        left_layout.addWidget(QLabel("Metadata"))
        left_layout.addWidget(self.metadata_label, 1)

        self.world_view = ImageView("World Camera")
        self.fpv_view = ImageView("FPV Camera")
        image_splitter = QSplitter(Qt.Horizontal)
        image_splitter.addWidget(self.world_view)
        image_splitter.addWidget(self.fpv_view)

        self.plot_tabs = QTabWidget()
        self.position_plot = self._make_plot("Position qpos[:3]")
        self.joystick_plot = self._make_plot("Joystick Axes")
        self.ctrl_plot = self._make_plot("Controls")
        self.command_plot = self._make_plot("Commands")
        self.plot_tabs.addTab(self.position_plot, "Position")
        self.plot_tabs.addTab(self.joystick_plot, "Joystick")
        self.plot_tabs.addTab(self.ctrl_plot, "Control")
        self.plot_tabs.addTab(self.command_plot, "Command")

        self.flight_state_label = QLabel("Unknown")
        self.flight_state_label.setAlignment(Qt.AlignCenter)
        self.flight_state_label.setMinimumWidth(150)
        self.flight_state_label.setStyleSheet(
            "QLabel {"
            "background: #1f2329;"
            "color: #d5d9e0;"
            "border: 1px solid #3a3f47;"
            "font-size: 18px;"
            "font-weight: 600;"
            "padding: 14px;"
            "}"
        )

        plot_panel = QWidget()
        plot_layout = QHBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.plot_tabs, 1)
        plot_layout.addWidget(self.flight_state_label)

        center_panel = QSplitter(Qt.Vertical)
        center_panel.addWidget(image_splitter)
        center_panel.addWidget(plot_panel)
        center_panel.setSizes([560, 360])

        root_splitter = QSplitter(Qt.Horizontal)
        root_splitter.addWidget(left_panel)
        root_splitter.addWidget(center_panel)
        root_splitter.setSizes([280, 1220])
        self.setCentralWidget(root_splitter)

    def _make_plot(self, title: str) -> pg.PlotWidget:
        plot = pg.PlotWidget(title=title)
        plot.addLegend()
        plot.showGrid(x=True, y=True, alpha=0.25)
        return plot

    def choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Open demonstration directory",
            str(self.data_dir),
        )
        if directory:
            self.load_directory(Path(directory))

    def load_directory(self, data_dir: Path) -> None:
        self.data_dir = data_dir.expanduser()
        if not self.data_dir.exists():
            return

        self.clip_paths = sorted(self.data_dir.glob("clip_*.npz"))
        if not self.clip_paths:
            for episode_dir in sorted(self.data_dir.iterdir()):
                if episode_dir.is_dir():
                    self.clip_paths.extend(sorted(episode_dir.glob("clip_*.npz")))

        self.clip_list.clear()
        for path in self.clip_paths:
            self.clip_list.addItem(str(path.relative_to(self.data_dir)))

        if self.clip_paths:
            self.clip_list.setCurrentRow(0)

    def load_clip_by_row(self, row: int) -> None:
        if row < 0 or row >= len(self.clip_paths):
            return
        self.load_clip(self.clip_paths[row])

    def load_clip(self, path: Path) -> None:
        try:
            with np.load(path, allow_pickle=False) as npz:
                self.clip = {key: npz[key].copy() for key in npz.files}
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load clip", str(exc))
            return

        if "qpos" not in self.clip:
            QMessageBox.critical(self, "Invalid clip", "Missing required qpos field")
            return

        self.is_playing = False
        self.play_button.setText("Play")
        self.current_frame = 0
        num_frames = self.num_frames
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, num_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.update_timer_interval()
        self.update_metadata(path)
        self.refresh_plots()
        self.render_frame()

    @property
    def num_frames(self) -> int:
        if "qpos" not in self.clip:
            return 0
        return int(self.clip["qpos"].shape[0])

    def update_metadata(self, path: Path) -> None:
        lines = [f"File: {path.name}", f"Frames: {self.num_frames}"]
        if "time" in self.clip and self.num_frames > 0:
            duration = float(self.clip["time"][-1] - self.clip["time"][0])
            lines.append(f"Duration: {duration:.3f} s")
        if "qpos" in self.clip and self.num_frames > 1:
            displacement = np.linalg.norm(self.clip["qpos"][-1, :3] - self.clip["qpos"][0, :3])
            max_step = np.max(np.linalg.norm(np.diff(self.clip["qpos"][:, :3], axis=0), axis=1))
            lines.append(f"Position displacement: {displacement:.4f} m")
            lines.append(f"Max position step: {max_step:.4f} m")
        for key in sorted(self.clip):
            value = self.clip[key]
            if key == "metadata_json":
                continue
            lines.append(f"{key}: {value.shape} {value.dtype}")
        self.metadata_label.setText("\n".join(lines))

    def refresh_plots(self) -> None:
        for plot in (
            self.position_plot,
            self.joystick_plot,
            self.ctrl_plot,
            self.command_plot,
        ):
            plot.clear()
        self.plot_lines.clear()

        x = self.x_axis()
        self._plot_array(self.position_plot, x, self.clip.get("qpos"), ["x", "y", "z"], 3)
        self._plot_array(
            self.joystick_plot,
            x,
            self.clip.get("joystick_axes"),
            ["roll", "pitch", "throttle", "yaw", "record", "mode", "takeoff_land"],
            7,
        )
        self._plot_array(self.ctrl_plot, x, self.clip.get("ctrl"), None, 9)

        command_series = {
            "thrust": self.clip.get("thrust_command"),
            "yaw_rate": self.clip.get("yaw_rate_command"),
        }
        for name, values in command_series.items():
            if values is not None:
                self.command_plot.plot(x, values, name=name)
        self._plot_array(
            self.command_plot,
            x,
            self.clip.get("target_omega"),
            ["omega_x", "omega_y", "omega_z"],
            3,
        )

        for plot in (
            self.position_plot,
            self.joystick_plot,
            self.ctrl_plot,
            self.command_plot,
        ):
            line = pg.InfiniteLine(pos=self.current_x(), angle=90, pen="y")
            plot.addItem(line)
            self.plot_lines.append(line)

    def _plot_array(
        self,
        plot: pg.PlotWidget,
        x: np.ndarray,
        values: np.ndarray | None,
        labels: list[str] | None,
        max_dims: int,
    ) -> None:
        if values is None:
            return
        if values.ndim == 1:
            plot.plot(x, values, name=labels[0] if labels else "value")
            return
        dims = min(values.shape[1], max_dims)
        for dim in range(dims):
            name = labels[dim] if labels and dim < len(labels) else f"dim{dim}"
            plot.plot(x, values[:, dim], name=name)

    def x_axis(self) -> np.ndarray:
        if "time" in self.clip:
            return self.clip["time"]
        return np.arange(self.num_frames)

    def current_x(self) -> float:
        x = self.x_axis()
        if x.size == 0:
            return 0.0
        return float(x[self.current_frame])

    def set_frame_from_slider(self, value: int) -> None:
        self.current_frame = int(value)
        self.render_frame()

    def toggle_playback(self) -> None:
        if self.num_frames == 0:
            return
        self.is_playing = not self.is_playing
        self.play_button.setText("Pause" if self.is_playing else "Play")
        if self.is_playing:
            self.timer.start()
        else:
            self.timer.stop()

    def update_timer_interval(self) -> None:
        self.timer.setInterval(max(1, int(1000 / self.fps_spin.value())))

    def next_frame(self) -> None:
        if self.num_frames == 0:
            return
        if self.current_frame >= self.num_frames - 1:
            self.toggle_playback()
            return
        self.current_frame += 1
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        self.render_frame()

    def render_frame(self) -> None:
        if self.num_frames == 0:
            return

        frame = self.current_frame
        self.data.qpos[:] = self.clip["qpos"][frame]
        if "qvel" in self.clip:
            self.data.qvel[:] = self.clip["qvel"][frame]
        if "ctrl" in self.clip:
            self.data.ctrl[:] = self.clip["ctrl"][frame]
        if "time" in self.clip:
            self.data.time = float(self.clip["time"][frame])
        mujoco.mj_forward(self.model, self.data)

        if self.follow_world_camera:
            self.world_follow_camera.lookat[:] = self.data.qpos[:3]
            self.world_renderer.update_scene(self.data, camera=self.world_follow_camera)
        else:
            self.world_renderer.update_scene(self.data, camera=self.world_camera)
        self.add_future_position_markers()
        self.world_view.set_rgb(self.world_renderer.render())
        self.fpv_renderer.update_scene(self.data, camera=self.fpv_camera)
        self.fpv_view.set_rgb(self.fpv_renderer.render())

        self.frame_label.setText(f"Frame: {frame + 1} / {self.num_frames}")
        self.update_flight_state_label()
        for line in self.plot_lines:
            line.setPos(self.current_x())

    def update_flight_state_label(self) -> None:
        mode = self.current_position_auto_mode()
        text, color = {
            0: ("正常飞行", "#2f8f5b"),
            1: ("起飞状态", "#d8941f"),
            2: ("降落状态", "#b84d4d"),
        }.get(mode, ("未知状态", "#545b66"))
        self.flight_state_label.setText(text)
        self.flight_state_label.setStyleSheet(
            "QLabel {"
            f"background: {color};"
            "color: white;"
            "border: 1px solid rgba(255, 255, 255, 0.22);"
            "font-size: 18px;"
            "font-weight: 600;"
            "padding: 14px;"
            "}"
        )

    def current_position_auto_mode(self) -> int | None:
        values = self.clip.get("position_auto_mode")
        if values is None or self.num_frames == 0:
            return None

        value = np.asarray(values[self.current_frame]).reshape(-1)
        if value.size == 0:
            return None
        return int(value[0])

    def add_future_position_markers(self) -> None:
        if not self.show_future_points_checkbox.isChecked():
            return
        if "qpos" not in self.clip:
            return

        num_points = self.future_points_spin.value()
        if num_points <= 0:
            return

        indices = self.future_position_indices(num_points=num_points, dt_seconds=0.5)
        scene = self.world_renderer.scene
        for point_order, frame_index in enumerate(indices, start=1):
            if frame_index <= self.current_frame or frame_index >= self.num_frames:
                continue
            if scene.ngeom >= len(scene.geoms):
                break

            pos = self.clip["qpos"][frame_index, :3].astype(np.float64)
            radius = 0.015 + 0.002 * min(point_order, 8)
            alpha = max(0.35, 1.0 - 0.04 * (point_order - 1))
            rgba = np.array([1.0, 0.0, 0.0, alpha], dtype=np.float32)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, radius, radius], dtype=np.float64),
                pos,
                np.eye(3).reshape(-1),
                rgba,
            )
            scene.ngeom += 1

    def future_position_indices(self, num_points: int, dt_seconds: float) -> list[int]:
        if self.num_frames == 0:
            return []

        if "time" in self.clip:
            times = self.clip["time"]
            current_time = float(times[self.current_frame])
            indices = []
            for step in range(1, num_points + 1):
                target_time = current_time + dt_seconds * step
                frame_index = int(np.searchsorted(times, target_time, side="left"))
                if frame_index < self.num_frames:
                    indices.append(frame_index)
            return indices

        frame_step = max(1, int(round(dt_seconds * self.fps_spin.value())))
        return [
            self.current_frame + frame_step * step
            for step in range(1, num_points + 1)
            if self.current_frame + frame_step * step < self.num_frames
        ]

    def closeEvent(self, event) -> None:  # noqa: N802
        self.timer.stop()
        self.world_renderer.close()
        self.fpv_renderer.close()
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PySide clip viewer for mm_planner.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--world-camera", default="world_camera")
    parser.add_argument("--fpv-camera", default="fpv_camera")
    parser.add_argument(
        "--fixed-world-camera",
        action="store_true",
        help="Use the XML world camera instead of the default follow camera.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = ClipViewer(
        data_dir=args.data_dir,
        xml_path=args.xml,
        world_camera=args.world_camera,
        fpv_camera=args.fpv_camera,
        follow_world_camera=not args.fixed_world_camera,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
