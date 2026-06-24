
import numpy as np

class Mixer:
    def __init__(self):

        self.motor_profile = {
            "const_thrust": 2.052e-2,  # 电机推力系数 (N/krpm^2) 注意结果单位为力(N)
            "const_torque": 3.842e-4,  # 电机反扭系数 (Nm/krpm^2) 注意结果单位为扭矩(Nm)
            "max_speed": 30,  # 电机最大转速(krpm)
            "length": 0.22/2.0,   # 电机力臂长度 单位m
            "comment": "thrust model of t-motor F80 with T6143 propeller"
        }


        self.const_thrust  = self.motor_profile["const_thrust"]
        self.const_torque  = self.motor_profile["const_torque"]
        self.length  = self.motor_profile["length"]
        self.max_speed  = self.motor_profile["max_speed"]

        self.calc_mat()

    def calc_mat(self):
        # 动力分配正向矩阵
        self.mat = np.array([
            [self.const_thrust, self.const_thrust, self.const_thrust, self.const_thrust],     # F total
            [self.const_thrust * self.length / np.sqrt(2), self.const_thrust * self.length / np.sqrt(2),
             -self.const_thrust * self.length / np.sqrt(2), -self.const_thrust * self.length / np.sqrt(2)],    # Mx + + - -
            [-self.const_thrust * self.length / np.sqrt(2), self.const_thrust * self.length / np.sqrt(2),
             self.const_thrust * self.length / np.sqrt(2), -self.const_thrust * self.length / np.sqrt(2)],    # My - + + -
            [-self.const_torque, self.const_torque, -self.const_torque, self.const_torque]                    # Mz - + - +
        ])
        
        # 动力分配逆向矩阵
        self.inv_mat = np.linalg.inv(self.mat)

    # 动力分配
    # thrust: 机体总推力 单位N
    # mx, my, mz: 三轴扭矩 单位Nm
    def motor_distribute(self, thrust, mx, my, mz):
        motor_speed_square = np.matmul(self.inv_mat, np.array([thrust, mx, my, mz]))
        np.clip(motor_speed_square, 0, self.max_speed ** 2, out=motor_speed_square)
        return motor_speed_square

    def load_motor_profile(self, filename):
        self.motor_profile = np.load(filename, allow_pickle=True).item()
        return self.motor_profile
    
    def save_motor_profile(self, filename):
        np.save(filename, self.motor_profile)

if __name__ == '__main__':
    # 计算测试
    thrust = 20  # 总推力输出为0.4N
    Mx = 1
    My = 1
    Mz = 0.0

    mixer = Mixer()
    motor_speed = mixer.motor_distribute(thrust, Mx, My, Mz)
    print(f"Motor Speed:{motor_speed}")