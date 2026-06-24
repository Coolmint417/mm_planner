import numpy as np

class RatePIDController:
    """
    角速度环 PID 控制器
    输入:
      - target_thrust: 目标总推力 (N)
      - target_omega: 目标角速度 [wx, wy, wz] (rad/s)
      - current_omega: 当前角速度 [wx, wy, wz] (rad/s)
      - dt: 控制周期 (s)
    输出:
      - (target_thrust, tau_x, tau_y, tau_z)
    """
    def __init__(
        self,
        kp=np.array([0.0, 0.0, 0.0]),
        ki=np.array([0.0, 0.0, 0.0]),
        kd=np.array([0.0, 0.0, 0.0]),
        integral_limit=np.array([np.inf, np.inf, np.inf]),
        torque_limit=np.array([np.inf, np.inf, np.inf]),
    ):
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)

        self.integral_limit = np.asarray(integral_limit, dtype=float)
        self.torque_limit = np.asarray(torque_limit, dtype=float)

        self.integral_error = np.zeros(3, dtype=float)
        self.last_error = np.zeros(3, dtype=float)
        self.has_last = False

    def set_gains(self, kp, ki, kd):
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)

    def set_integral_limit(self, integral_limit):
        self.integral_limit = np.asarray(integral_limit, dtype=float)

    def set_torque_limit(self, torque_limit):
        self.torque_limit = np.asarray(torque_limit, dtype=float)

    def reset(self):
        self.integral_error[:] = 0.0
        self.last_error[:] = 0.0
        self.has_last = False

    def update(self, target_thrust, target_omega, current_omega, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")

        target_omega = np.asarray(target_omega, dtype=float)
        current_omega = np.asarray(current_omega, dtype=float)
        error = target_omega - current_omega

        self.integral_error += error * dt
        np.clip(
            self.integral_error,
            -self.integral_limit,
            self.integral_limit,
            out=self.integral_error,
        )

        if self.has_last:
            d_error = (error - self.last_error) / dt
        else:
            d_error = np.zeros(3, dtype=float)
            self.has_last = True

        torque = self.kp * error + self.ki * self.integral_error + self.kd * d_error
        np.clip(torque, -self.torque_limit, self.torque_limit, out=torque)

        self.last_error = error
        return target_thrust, torque[0], torque[1], torque[2]