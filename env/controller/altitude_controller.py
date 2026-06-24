import numpy as np


class AltitudeHoldPIDController:
    """
    无人机定高控制器。

    油门摇杆量 throttle 在 [0, 1]：
      - throttle > upper_deadband: 上升，目标上升速度线性增加
      - throttle < lower_deadband: 下降，目标下降速度线性增加
      - lower_deadband <= throttle <= upper_deadband: 保持当前目标高度

    输入:
      - current_z: 无人机在世界坐标系下的 z 轴位置 (m)
      - throttle: 油门摇杆归一化量 [0, 1]
      - dt: 控制周期 (s)

    输出:
      - target_thrust: 目标总推力 (N)
    """
    def __init__(
        self,
        mass=2.1,
        gravity=9.8,
        kp=18.0,
        ki=0.0,
        kd=10.0,
        max_climb_rate=1.0,
        max_descent_rate=0.8,
        lower_deadband=0.4,
        upper_deadband=0.6,
        integral_limit=0.6,
        thrust_limit=(0.0, 73.872),
        height_limit=(0.0, 10.0),
        z_velocity_lpf_alpha=0.25,
    ):
        if not 0.0 <= lower_deadband < upper_deadband <= 1.0:
            raise ValueError("deadband must satisfy 0 <= lower < upper <= 1")

        self.mass = float(mass)
        self.gravity = float(gravity)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.max_climb_rate = float(max_climb_rate)
        self.max_descent_rate = float(max_descent_rate)
        self.lower_deadband = float(lower_deadband)
        self.upper_deadband = float(upper_deadband)
        self.integral_limit = float(integral_limit)
        self.thrust_limit = thrust_limit
        self.height_limit = height_limit
        self.z_velocity_lpf_alpha = float(np.clip(z_velocity_lpf_alpha, 0.0, 1.0))

        self.target_z = 0.0
        self.integral_error = 0.0
        self.last_z = 0.0
        self.z_velocity = 0.0
        self.initialized = False

    def reset(self, current_z=None):
        self.integral_error = 0.0
        self.z_velocity = 0.0
        if current_z is None:
            self.target_z = 0.0
            self.last_z = 0.0
            self.initialized = False
            return

        self.target_z = float(current_z)
        self.last_z = float(current_z)
        self.initialized = True

    def _target_velocity_from_throttle(self, throttle):
        throttle = float(np.clip(throttle, 0.0, 1.0))
        if throttle > self.upper_deadband:
            ratio = (throttle - self.upper_deadband) / (1.0 - self.upper_deadband)
            return ratio * self.max_climb_rate
        if throttle < self.lower_deadband:
            ratio = (self.lower_deadband - throttle) / self.lower_deadband
            return -ratio * self.max_descent_rate
        return 0.0

    def update(self, current_z, throttle, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")

        current_z = float(current_z)
        if not self.initialized:
            self.reset(current_z)

        target_vz = self._target_velocity_from_throttle(throttle)
        
        self.target_z += target_vz * dt
        self.target_z = np.clip(self.target_z, self.height_limit[0], self.height_limit[1])

        measured_vz = (current_z - self.last_z) / dt
        self.z_velocity += self.z_velocity_lpf_alpha * (measured_vz - self.z_velocity)

        z_error = self.target_z - current_z
        self.integral_error += z_error * dt
        self.integral_error = float(
            np.clip(self.integral_error, -self.integral_limit, self.integral_limit)
        )

        acc_command = (
            self.kp * z_error
            + self.ki * self.integral_error
            + self.kd * (target_vz - self.z_velocity)
        )
        target_thrust = self.mass * (self.gravity + acc_command)

        if self.thrust_limit is not None:
            target_thrust = float(
                np.clip(target_thrust, self.thrust_limit[0], self.thrust_limit[1])
            )

        self.last_z = current_z
        return target_thrust

