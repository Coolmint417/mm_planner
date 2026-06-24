import numpy as np
import geometry

class AttitudePIDController:
    """
    姿态外环 PID 控制器。
    输入当前姿态 R、目标姿态 R_des，输出内环需要跟踪的目标机体系角速度。
    """
    def __init__(
        self,
        kp=np.array([10.0, 10.0, 6.0]),
        ki=np.array([4.0, 4.0, 2.5]),
        kd=np.array([0.25, 0.25, 0.10]),
        integral_limit=np.array([0.25, 0.25, 0.15]),
        rate_limit=np.array([6.0, 6.0, 3.0]),
    ):
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.integral_limit = np.asarray(integral_limit, dtype=float)
        self.rate_limit = np.asarray(rate_limit, dtype=float)
        self.integral_error = np.zeros(3, dtype=float)
        self.last_error = np.zeros(3, dtype=float)
        self.has_last = False

    def reset(self):
        self.integral_error[:] = 0.0
        self.last_error[:] = 0.0
        self.has_last = False

    def update(self, current_rotation, target_rotation, feedforward_omega, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")

        attitude_error = 0.5 * geometry.veemap(
            target_rotation.T @ current_rotation - current_rotation.T @ target_rotation
        )
        self.integral_error += attitude_error * dt
        np.clip(
            self.integral_error,
            -self.integral_limit,
            self.integral_limit,
            out=self.integral_error,
        )

        if self.has_last:
            d_error = (attitude_error - self.last_error) / dt
        else:
            d_error = np.zeros(3, dtype=float)
            self.has_last = True

        correction = (
            self.kp * attitude_error
            + self.ki * self.integral_error
            + self.kd * d_error
        )
        target_omega = np.asarray(feedforward_omega, dtype=float) - correction
        np.clip(target_omega, -self.rate_limit, self.rate_limit, out=target_omega)

        self.last_error = attitude_error
        return target_omega
