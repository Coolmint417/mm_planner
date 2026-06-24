import numpy as np
import os
import math
import errno
import fcntl
import struct
import mujoco 
import mujoco.viewer as viewer 

from clip_recorder import ClipRecorder
from controller.rate_controller import RatePIDController
from controller.att_controller import AttitudePIDController
from controller.gyro_filter import GyroLowPassFilter
from controller.altitude_controller import AltitudeHoldPIDController
from controller.se3_controller import SE3Controller, VehicleState

from controller.motor_mixer import Mixer

import geometry 

VEHICLE_MASS = 2.1
GRAVITY = 9.8
HOVER_THRUST = VEHICLE_MASS * GRAVITY

MAX_ROLL = math.radians(25.0)
MAX_PITCH = math.radians(25.0)
MAX_YAW_RATE = math.radians(90.0)
MAX_XY_SPEED = 1.0
MAX_Z_SPEED = 0.7
MAX_POSITION_TARGET_Z = 10.0
MIN_POSITION_TARGET_Z = 0.0

# 根据你的遥控器通道顺序改这里即可。Linux joystick 轴值范围会归一化到 [-1, 1]。
AXIS_ROLL = 0
AXIS_PITCH = 1
AXIS_THROTTLE = 2
AXIS_YAW = 3
AXIS_RECORD = 4
AXIS_CTRL_MODE = 5  
AXIS_TAKEOFF_LAND = 7

CTRL_MODE_ANGLE = 0         # PX4 Manual mode / Betaflight angle mode
CTRL_MODE_ALTITUDE = 1      # PX4 Altitude mode
CTRL_MODE_POSITION = 2      # PX4 Position mode

POSITION_MODE_TAKEOFF_Z = 1.0
POSITION_MODE_LAND_Z = 0.05
POSITION_AUTO_NONE = 0
POSITION_AUTO_TAKEOFF = 1
POSITION_AUTO_LAND = 2
TAKEOFF_RELATIVE_HEIGHT = 0.8
TAKEOFF_REACHED_RADIUS = 0.1
TAKEOFF_SETTLE_SPEED = 1.0
TAKEOFF_KV = 2.0
LAND_DESCENT_SPEED = 0.25
LAND_STILL_SPEED = 0.08
LAND_STILL_TIME = 1.0


class JoystickReader:
    JS_EVENT_FORMAT = "=IhBB"
    JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)
    JS_EVENT_AXIS = 0x02
    JS_EVENT_INIT = 0x80

    def __init__(self, path="/dev/input/js0", deadband=0.05):
        self.path = path
        self.deadband = deadband
        self.fd = None
        self.axes = {}
        self.open_failed = False
        self.open()

    def open(self):
        try:
            self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
            flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            print(f"Joystick connected: {self.path}")
        except OSError as exc:
            self.fd = None
            self.open_failed = True
            print(f"Joystick unavailable ({self.path}): {exc}. Use neutral command.")

    def update(self):
        if self.fd is None:
            return

        while True:
            try:
                event = os.read(self.fd, self.JS_EVENT_SIZE)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    print(f"Joystick read error: {exc}")
                    os.close(self.fd)
                    self.fd = None
                break

            if len(event) != self.JS_EVENT_SIZE:
                break

            _, value, event_type, number = struct.unpack(self.JS_EVENT_FORMAT, event)
            event_type &= ~self.JS_EVENT_INIT
            if event_type == self.JS_EVENT_AXIS:
                axis_value = value / 32767.0
                if abs(axis_value) < self.deadband:
                    axis_value = 0.0
                self.axes[number] = float(np.clip(axis_value, -1.0, 1.0))

    def axis(self, number, default=0.0):
        return self.axes.get(number, default)


def euler_to_rotation(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def joystick_to_world_velocity(roll_axis, pitch_axis, throttle, yaw):
    def throttle_to_vertical_speed(throttle):
        throttle = float(np.clip(throttle, 0.0, 1.0))
        if throttle > 0.6:
            return (throttle - 0.6) / 0.4 * MAX_Z_SPEED
        if throttle < 0.4:
            return -(0.4 - throttle) / 0.4 * MAX_Z_SPEED
        return 0.0
    forward_speed = pitch_axis * MAX_XY_SPEED
    left_speed = -roll_axis * MAX_XY_SPEED
    vertical_speed = throttle_to_vertical_speed(throttle)

    forward_world = np.array([math.cos(yaw), math.sin(yaw), 0.0])  # X axis in the NWU coordination
    left_world = np.array([-math.sin(yaw), math.cos(yaw), 0.0])  # Y axis in the NWU coordination
    return forward_speed * forward_world + left_speed * left_world + np.array([0.0, 0.0, vertical_speed])

rate_controller = RatePIDController(kp=np.array([5e-1, 5e-1, 5e-1]), 
                                    ki=np.array([1e-1, 1e-1, 1e-1]), 
                                    kd=np.array([1e-2, 1e-2, 1e-2]), 
                                    integral_limit=np.array([0.3, 0.3, 0.3]), 
                                    torque_limit=np.array([1.0, 1.0, 0.4]))
attitude_controller = AttitudePIDController()
gyro_filter = GyroLowPassFilter(cutoff_freq=30.0)
mixer = Mixer()
altitude_controller = AltitudeHoldPIDController(
    mass=VEHICLE_MASS,
    gravity=GRAVITY,
    thrust_limit=(0.0, 4.0 * mixer.const_thrust * mixer.max_speed ** 2),
)
position_controller = SE3Controller(with_position_error=True)
position_controller.set_parameters(kx=8.0, kv=6.0, kR=8.0)

joystick = JoystickReader()
yaw_command = 0.0
position_target = np.zeros(3)
recording_config = None
clip_recorder = None
record_axis_was_positive = False
recording_metadata_provider = None
takeoff_land_axis_last_sign = 0
position_auto_mode = POSITION_AUTO_NONE
position_auto_target = np.zeros(3)
land_still_time = 0.0


xml_path = os.path.join(os.path.dirname(__file__), './scene.xml')


def set_recording_config(config):
    global recording_config
    recording_config = config


def set_recording_metadata_provider(provider):
    global recording_metadata_provider
    recording_metadata_provider = provider


def close_recorder():
    global clip_recorder
    if clip_recorder is not None:
        clip_recorder.close()
        clip_recorder = None


def load_callback(m=None, d=None):
    global last_time, yaw_command, control_mode, position_target, clip_recorder
    global record_axis_was_positive, takeoff_land_axis_last_sign
    global position_auto_mode, position_auto_target, land_still_time
    mujoco.set_mjcb_control(None)
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    last_time = 0.0
    yaw_command = 0.0
    control_mode = -1
    record_axis_was_positive = False
    takeoff_land_axis_last_sign = 0
    position_auto_mode = POSITION_AUTO_NONE
    position_auto_target = d.qpos[:3].copy()
    land_still_time = 0.0
    position_target = d.qpos[:3].copy()
    attitude_controller.reset()
    rate_controller.reset()
    gyro_filter.reset()
    altitude_controller.reset(d.qpos[2])
    close_recorder()
    if recording_config is not None:
        clip_recorder = ClipRecorder(m, recording_config)

    if m is not None:
        mujoco.set_mjcb_control(lambda m, d: control_callback(m, d))  # 设置控制回调函数
    return m, d


def _update_recording_trigger(d):
    global record_axis_was_positive
    if clip_recorder is None:
        return

    record_axis = joystick.axis(AXIS_RECORD)
    record_axis_is_positive = record_axis > 0.0
    record_axis_is_negative = record_axis < 0.0

    if record_axis_is_positive and not record_axis_was_positive:
        if recording_metadata_provider is not None:
            clip_recorder.set_clip_metadata(recording_metadata_provider())
        clip_recorder.begin(d.time)
    elif record_axis_is_negative and record_axis_was_positive:
        clip_recorder.end()

    if record_axis_is_positive:
        record_axis_was_positive = True
    elif record_axis_is_negative:
        record_axis_was_positive = False


def _axis_sign(value):
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _sync_takeoff_land_axis_state():
    global takeoff_land_axis_last_sign
    sign = _axis_sign(joystick.axis(AXIS_TAKEOFF_LAND))
    if sign != 0:
        takeoff_land_axis_last_sign = sign


def _update_position_mode_takeoff_land_trigger(d):
    global takeoff_land_axis_last_sign

    sign = _axis_sign(joystick.axis(AXIS_TAKEOFF_LAND))
    if sign == 0:
        return

    if takeoff_land_axis_last_sign < 0 and sign > 0:
        _start_position_auto_takeoff(d)
    elif takeoff_land_axis_last_sign > 0 and sign < 0:
        _start_position_auto_land(d)

    takeoff_land_axis_last_sign = sign


def _start_position_auto_takeoff(d):
    global position_auto_mode, position_auto_target, position_target, land_still_time
    position_auto_mode = POSITION_AUTO_TAKEOFF
    position_auto_target = d.qpos[:3].copy()
    position_auto_target[2] = np.clip(
        max(POSITION_MODE_TAKEOFF_Z, d.qpos[2] + TAKEOFF_RELATIVE_HEIGHT),
        MIN_POSITION_TARGET_Z,
        MAX_POSITION_TARGET_Z,
    )
    position_target = position_auto_target.copy()
    land_still_time = 0.0
    print(f"Position auto takeoff: target = {position_auto_target}")


def _start_position_auto_land(d):
    global position_auto_mode, position_auto_target, position_target, land_still_time
    position_auto_mode = POSITION_AUTO_LAND
    position_auto_target = d.qpos[:3].copy()
    position_target = d.qpos[:3].copy()
    land_still_time = 0.0
    print("Position auto landing: joystick input blocked, descending slowly")


def _update_position_auto_control(d, dt):
    global position_auto_mode, position_target, land_still_time

    if position_auto_mode == POSITION_AUTO_TAKEOFF:
        position_target = position_auto_target.copy()
        target_velocity = TAKEOFF_KV * (position_target - d.qpos[:3].copy())
        z_error = abs(d.qpos[2] - position_auto_target[2])
        speed = np.linalg.norm(d.qvel[:3])
        if z_error < TAKEOFF_REACHED_RADIUS and speed < TAKEOFF_SETTLE_SPEED:
            position_auto_mode = POSITION_AUTO_NONE
            _sync_takeoff_land_axis_state()
            print("Position auto takeoff complete: joystick input restored")
        return target_velocity

    if position_auto_mode == POSITION_AUTO_LAND:
        target_velocity = np.array([0.0, 0.0, -LAND_DESCENT_SPEED])

        position_target[2] = np.clip(
            position_target[2] - LAND_DESCENT_SPEED * dt,
            MIN_POSITION_TARGET_Z,
            MAX_POSITION_TARGET_Z,
        )
        if np.linalg.norm(d.qvel[:3]) < LAND_STILL_SPEED:
            land_still_time += dt
        else:
            land_still_time = 0.0

        if land_still_time >= LAND_STILL_TIME:
            position_auto_mode = POSITION_AUTO_NONE
            position_target = d.qpos[:3].copy()
            land_still_time = 0.0
            _sync_takeoff_land_axis_state()
            print("Position auto landing complete: joystick input restored")
        return target_velocity

    return None


def _record_frame(
    m,
    d,
    roll_command,
    pitch_command,
    yaw_rate_command,
    throttle_command,
    thrust_command,
    target_omega,
    target_velocity,
):
    if clip_recorder is None:
        return

    joystick_axes = np.array(
        [
            joystick.axis(AXIS_ROLL),
            joystick.axis(AXIS_PITCH),
            joystick.axis(AXIS_THROTTLE),
            joystick.axis(AXIS_YAW),
            joystick.axis(AXIS_RECORD),
            joystick.axis(AXIS_CTRL_MODE),
            joystick.axis(AXIS_TAKEOFF_LAND),
        ],
        dtype=np.float32,
    )
    extra = {
        "mode_id": np.asarray(control_mode, dtype=np.int64),
        "joystick_axes": joystick_axes,
        "roll_command": np.asarray(roll_command, dtype=np.float32),
        "pitch_command": np.asarray(pitch_command, dtype=np.float32),
        "yaw_rate_command": np.asarray(yaw_rate_command, dtype=np.float32),
        "throttle_command": np.asarray(throttle_command, dtype=np.float32),
        "thrust_command": np.asarray(thrust_command, dtype=np.float32),
        "target_omega": np.asarray(target_omega, dtype=np.float32),
        "target_velocity": np.asarray(target_velocity, dtype=np.float32),
        "yaw_command": np.asarray(yaw_command, dtype=np.float32),
        "position_target": np.asarray(position_target, dtype=np.float32),
        "position_auto_mode": np.asarray(position_auto_mode, dtype=np.int64),
    }
    clip_recorder.sample(m, d, extra)


def control_callback(m, d):
    global last_time, yaw_command, control_mode, position_target
    global position_auto_mode
    
    _sensor_data = d.sensordata
    gyro_x = _sensor_data[0]
    gyro_y = _sensor_data[1]
    gyro_z = _sensor_data[2]
    quat_w = _sensor_data[6]
    quat_x = _sensor_data[7]
    quat_y = _sensor_data[8]
    quat_z = _sensor_data[9]

    dt = d.time - last_time
    if dt <= 0:
        dt = m.opt.timestep

    raw_omega = np.array([gyro_x, gyro_y, gyro_z])
    omega = gyro_filter.update(raw_omega, dt)
    #omega = np.array([gyro_x, gyro_y, gyro_z])  # 角速度
    current_rotation = geometry.GeoQuaternion(quat_x, quat_y, quat_z, quat_w).getRotationMatrix()

    joystick.update()
    _update_recording_trigger(d)
    requested_control_mode = CTRL_MODE_ALTITUDE
    if joystick.axis(AXIS_CTRL_MODE) < -0.5:
        requested_control_mode = CTRL_MODE_ANGLE
    elif joystick.axis(AXIS_CTRL_MODE) > 0.5:
        requested_control_mode = CTRL_MODE_POSITION

    if requested_control_mode != control_mode:
        control_mode = requested_control_mode
        if control_mode == CTRL_MODE_ANGLE:
            print("Switch to angle mode")
        elif control_mode == CTRL_MODE_ALTITUDE:
            print("Switch to altitude mode")
            altitude_controller.reset(d.qpos[2])
        elif control_mode == CTRL_MODE_POSITION:
            print("Switch to position mode")
            position_target = d.qpos[:3].copy()
            position_auto_mode = POSITION_AUTO_NONE
            _sync_takeoff_land_axis_state()

    roll_command = 0.0
    pitch_command = 0.0
    yaw_rate_command = 0.0
    throttle_command = 0.5
    target_velocity = np.zeros(3)

    if control_mode == CTRL_MODE_ALTITUDE or control_mode == CTRL_MODE_ANGLE:
        
        roll_command = joystick.axis(AXIS_ROLL) * MAX_ROLL
        pitch_command = joystick.axis(AXIS_PITCH) * MAX_PITCH
        yaw_rate_command = -joystick.axis(AXIS_YAW) * MAX_YAW_RATE
        throttle_command = 0.5 * (joystick.axis(AXIS_THROTTLE) + 1.0)
        if control_mode == CTRL_MODE_ALTITUDE:
            thrust_command = altitude_controller.update(
                current_z=d.qpos[2],
                throttle=throttle_command,
                dt=dt,
            )
        else:
            thrust_command = throttle_command * HOVER_THRUST / 0.5

        yaw_command += yaw_rate_command * dt
        target_rotation = euler_to_rotation(roll_command, pitch_command, yaw_command)
        target_omega = attitude_controller.update(
            current_rotation=current_rotation,
            target_rotation=target_rotation,
            feedforward_omega=np.array([0.0, 0.0, yaw_rate_command]),
            dt=dt,
        )
    elif control_mode == CTRL_MODE_POSITION:
        if position_auto_mode == POSITION_AUTO_NONE:
            _update_position_mode_takeoff_land_trigger(d)

        auto_target_velocity = _update_position_auto_control(d, dt)
        if auto_target_velocity is None:
            throttle_command = 0.5 * (joystick.axis(AXIS_THROTTLE) + 1.0)
            yaw_rate_command = -joystick.axis(AXIS_YAW) * MAX_YAW_RATE
            yaw_command += yaw_rate_command * dt

            target_velocity = joystick_to_world_velocity(
                roll_axis=joystick.axis(AXIS_ROLL),
                pitch_axis=joystick.axis(AXIS_PITCH),
                throttle=throttle_command,
                yaw=yaw_command,
            )
            position_target += target_velocity * dt
            position_target[2] = np.clip(position_target[2], MIN_POSITION_TARGET_Z, MAX_POSITION_TARGET_Z)
        else:
            throttle_command = 0.5
            yaw_rate_command = 0.0
            target_velocity = auto_target_velocity

        current_state = VehicleState(
            pos=d.qpos[:3].copy(),
            vel=d.qvel[:3].copy(),
            rotation=current_rotation,
            omega=omega,
        )
        target_state = VehicleState(
            pos=position_target.copy(),
            vel=target_velocity,
            rotation=np.eye(3),
            omega=np.zeros(3),
        )
        position_controller.set_current_state(current_state)
        position_controller.set_goal_state(target_state)
        target_rate = position_controller.control_update(
            goal_yaw=yaw_command,
            goal_acc=np.zeros(3),
        )
        thrust_command = float(np.clip(
            target_rate[0],
            0.0,
            4.0 * mixer.const_thrust * mixer.max_speed ** 2,
        ))
        target_omega = np.asarray(target_rate[1:], dtype=float)

    actuation = rate_controller.update(
        target_thrust = thrust_command,
        target_omega = target_omega,
        current_omega = omega,
        dt = dt
    )
    
    ## 混控器工作

    #print(actuation)
    motor_speed_square = mixer.motor_distribute(
        thrust = actuation[0],
        mx = actuation[1],
        my = actuation[2],
        mz = actuation[3]
    )

    for i in range(4):
        d.ctrl[i + 1] = 0
        d.ctrl[i + 5] = motor_speed_square[i]

    _record_frame(
        m=m,
        d=d,
        roll_command=roll_command,
        pitch_command=pitch_command,
        yaw_rate_command=yaw_rate_command,
        throttle_command=throttle_command,
        thrust_command=thrust_command,
        target_omega=target_omega,
        target_velocity=target_velocity,
    )
    last_time = d.time

if __name__ == '__main__':
    viewer.launch(loader=load_callback)
    
    
