import math

import numpy as np


class GyroLowPassFilter:
    """
    三轴陀螺仪一阶低通滤波器。

    输入:
      - gyro: 原始角速度 [gx, gy, gz] (rad/s)
      - dt: 控制周期 (s)

    输出:
      - filtered_gyro: 滤波后的角速度 [gx, gy, gz] (rad/s)
    """
    def __init__(
        self,
        cutoff_freq=30.0,
        initial_value=np.zeros(3),
    ):
        if cutoff_freq <= 0:
            raise ValueError("cutoff_freq must be positive")

        self.cutoff_freq = float(cutoff_freq)
        self.filtered_gyro = np.asarray(initial_value, dtype=float).copy()
        self.initialized = False

    def set_cutoff_freq(self, cutoff_freq):
        if cutoff_freq <= 0:
            raise ValueError("cutoff_freq must be positive")
        self.cutoff_freq = float(cutoff_freq)

    def reset(self, value=None):
        if value is None:
            self.filtered_gyro[:] = 0.0
            self.initialized = False
            return

        self.filtered_gyro = np.asarray(value, dtype=float).copy()
        self.initialized = True

    def update(self, gyro, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")

        gyro = np.asarray(gyro, dtype=float)
        if gyro.shape != (3,):
            raise ValueError("gyro must be a 3D vector")

        if not self.initialized:
            self.filtered_gyro = gyro.copy()
            self.initialized = True
            return self.filtered_gyro.copy()

        rc = 1.0 / (2.0 * math.pi * self.cutoff_freq)
        alpha = dt / (rc + dt)
        self.filtered_gyro += alpha * (gyro - self.filtered_gyro)
        return self.filtered_gyro.copy()

