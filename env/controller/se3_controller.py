# SE3 控制器

import numpy as np
import geometry

class VehicleState:
    # 四元数顺序 x y z w
    def __init__(self, pos = np.zeros(3), vel = np.zeros(3), rotation = np.eye(3), omega = np.zeros(3)):
        self.position = pos     # x y z
        self.velocity = vel     # x y z
        self.omega = omega      # x y z

        self.rotation = rotation  # SO3 object

    def update(self, pos, vel, rotation, omega):
        self.position = pos
        self.velocity = vel
        self.rotation = rotation
        self.omega = omega

class SE3Controller:
    def __init__(self, with_position_error = True):
        self.goal_state: VehicleState = None
        self.current_state: VehicleState = None
        self.command_rotation = np.eye(3)

        self.kR = 0.0  # SO3控制反馈系数
        self.with_position_error = with_position_error
        self.gravity = 9.8

        self.vehicle_mass = 2.1 # (kg)
        self.vehicle_inertia = np.array([
            [7.7e-3, 0, 0],
            [0, 7.255e-3, 0],
            [0, 0, 1.35e-2]
        ])  # (kg*m^2)

    def set_parameters(self, kx, kv, kR):
        self.kx = kx
        self.kv = kv
        self.kR = kR
    def set_current_state(self, state: VehicleState):
        self.current_state = state

    def set_goal_state(self, state: VehicleState):
        self.goal_state = state
    
    def update_linear_error(self):
        if self.goal_state is None or self.current_state is None:
            print("Error: goal or current state is None")
            return
        
        e_x = self.current_state.position - self.goal_state.position  
        e_v = self.current_state.velocity - self.goal_state.velocity

        return e_x, e_v
    
    # 控制更新函数(外部调用)

    def control_update(self, goal_yaw, goal_acc: np.ndarray):

        e_x, e_v = self.update_linear_error()
        # 位置速度控制(线性控制)
        if self.with_position_error:
            thrust_vector = self.vehicle_mass * goal_acc + self.vehicle_mass * self.gravity * np.array([0, 0, 1]) - self.kx * e_x - self.kv * e_v
        else:
            thrust_vector = self.vehicle_mass * goal_acc + self.vehicle_mass * self.gravity * np.array([0, 0, 1]) - self.kv * e_v
        thrust = np.dot(thrust_vector, self.current_state.rotation[:,2])

        b_3 = thrust_vector / np.linalg.norm(thrust_vector)

        b_c = np.asarray([np.cos(goal_yaw), np.sin(goal_yaw), 0])
        if np.linalg.norm(np.cross(b_3, b_c)) < 1e-6:
            b_c = np.asarray([np.cos(goal_yaw) * np.cos(np.pi / 6), np.sin(goal_yaw) * np.cos(np.pi / 6), np.sin(np.pi / 6)])

        b_2 = np.cross(b_3, b_c)
        b_1 = np.cross(b_2, b_3)
        R_goal = np.asarray([b_1, b_2, b_3]).T
        R_curr = self.current_state.rotation  # 当前旋转矩阵
        
        e_R = 0.5 * geometry.veemap(R_goal.T @ R_curr - R_curr.T @ R_goal)

        omega_des = -self.kR * e_R + self.goal_state.omega
        # R_curr.T @
        return thrust, omega_des[0], omega_des[1], omega_des[2]



