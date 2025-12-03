import mujoco
import numpy as np
import math
import sys
# 导入所有必要的库
from dm_control.mujoco import Physics  # 关键：导入 dm_control 的 Physics 封装
from dm_control.utils.inverse_kinematics import qpos_from_site_pose # 关键：导入 IK 函数
# from .casadi_ik import Kinematics
# --- 笛卡尔轨迹生成器 (使用 dm_control IK 的完整版本) ---
class CartesianTrajectoryGenerator:
    """
    生成笛卡尔空间(x, y, z)的轨迹，并同时计算出对应的关节角度。
    本版本使用 dm_control.inverse_kinematics.qpos_from_site_pose 进行逆运动学求解。
    """
    def __init__(self, model_path: str, ee_site_name: str, num_joints: int, 
                 idx=1, time_horizon=60, time_steps_per_sec = 5):
        """
        初始化轨迹生成器。

        Args:
            model_path (str): MuJoCo XML模型的路径。
            ee_site_name (str): 末端执行器 (end-effector) 在XML中的 <site> 名称。
            num_joints (int): 机械臂的关节数量。
            idx (int): 轨迹平面设置 (0 for x-y plane, 1 for y-z plane)。
            time_horizon (float): 轨迹的总时长（秒）。
            time_steps_per_sec (int): 每秒的轨迹点数量。
        """
        # 轨迹参数
        self.idx = idx
        self.time_horizon = time_horizon
        self.time_steps = time_steps_per_sec * time_horizon
        self.time_vector = np.linspace(0, self.time_horizon, self.time_steps)
        
        # 机器人和IK参数
        self.model_path = model_path
        self.ee_site_name = ee_site_name
        self.num_joints = num_joints
        
        # 这些参数可以根据您的机器人工作空间进行调整
        self.traj_scale = 0.5

        # 初始化IK求解器所需的MuJoCo模型和数据
        self._initialize_ik_solver()

    def _initialize_ik_solver(self):
        """加载模型并准备IK计算环境。"""
        print("正在为IK求解器初始化MuJoCo模型...")
        try:
            # 使用 dm_control 的 Physics 对象封装模型和数据
            # qpos_from_site_pose 函数需要这个类型的输入
            self.physics = Physics.from_xml_path(self.model_path)
            
            # 获取所有可动的、非自由浮动的关节名称
            # 这对于调用qpos_from_site_pose至关重要
            self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

            if not self.joint_names:
                raise RuntimeError("在模型中没有找到任何可驱动的关节。")

            print(f"找到 {len(self.joint_names)} 个可动关节: {self.joint_names}")
            if len(self.joint_names) < self.num_joints:
                print(f"警告：模型中找到的可动关节数量 ({len(self.joint_names)}) 少于指定的 num_joints ({self.num_joints})")
        
        except Exception as e:
            print(f"错误：无法从'{self.model_path}'加载IK模型。 {e}")
            sys.exit(1)
            
        # 检查 site 是否存在
        try:
            _ = self.physics.model.name2id(self.ee_site_name, 'site')
        except KeyError:
            raise ValueError(f"错误: 在模型中找不到名为 '{self.ee_site_name}' 的 site。请检查XML文件。")
        
        # 设置一个合理的初始关节姿态 (通常是全零姿态)
        home_qpos = np.zeros(self.physics.model.nq)
        with self.physics.reset_context():
            self.physics.data.qpos[:] = home_qpos
        
        print("IK求解器初始化完成。")

    def _solve_ik(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        """
        内部IK求解函数，使用 dm_control 的 qpos_from_site_pose。

        Args:
            target_pos (np.ndarray): 目标位置 [x, y, z]。
            target_quat (np.ndarray): 目标姿态四元数 [w, x, y, z]。

        Returns:
            np.ndarray: 求解出的关节角度（弧度），如果失败则返回 None。
        """
        # qpos_from_site_pose 会自动使用 self.physics.data.qpos 作为初始猜测值，
        # 这样可以利用上一步的解，提高求解速度和连续性。

        # 调用 dm_control 的 IK 函数
        ik_result = qpos_from_site_pose(
            physics=self.physics,                   # 传入 dm_control 的 Physics 对象
            site_name=self.ee_site_name,            # 目标 site 名称
            target_pos=target_pos,                  # 目标位置
            target_quat=target_quat,                # 目标姿态
            joint_names=self.joint_names,           # 指定哪些关节可以动
            inplace=True,                           # 设置为True，直接修改self.physics.data，效率更高
            max_steps=100,                          # 最大迭代次数
            tol=1e-6,                               # 求解精度阈值
            rot_weight=0.5,                         # 姿态误差的权重 (0.0到1.0之间)
            regularization_strength=1e-2            # 正则化强度，防止奇异点问题，增加稳定性
        )
        
        # 检查求解结果
        if ik_result.success:
            # 求解成功，返回与机械臂相关的关节角度
            # self.physics.data.qpos 已经被 inplace 修改，直接复制即可
            return self.physics.data.qpos[:self.num_joints].copy()
        else:
            # 如果求解失败，返回None
            return None

    def generate(self, traj_name='Fig8', target_orientation=np.array([1.0, 0.0, 0.0, 0.0])):
        """
        生成指定的笛卡尔轨迹和对应的关节角度轨迹。
        这个方法将一系列的笛卡尔坐标点，通过逆运动学，转换为一系列的关节角度。

        Args:
            traj_name (str): 轨迹名称 ('Fig8', 'Circle')。
            target_orientation (np.ndarray): 整个轨迹中末端执行器要保持的目标姿态 (四元数 [w, x, y, z])。

        Returns:
            tuple: 包含三个元素的元组:
                - np.ndarray: 形状为 (N, 3) 的笛卡尔坐标点云。
                - np.ndarray: 形状为 (N, num_joints) 的关节角度轨迹（弧度）。
                - np.ndarray: 对应的时间向量。
        """
        # 1. 生成笛卡尔坐标点 (x, y, z)
        #    这部分定义了机械臂末端应该遵循的路径。
        t_param = 1.6 + 0.02 * np.linspace(0, self.time_horizon * 5, len(self.time_vector))
        print(f"正在生成 '{traj_name}' 笛卡尔轨迹 (平面设置 idx={self.idx})")

        if traj_name == 'Fig8':
            if self.idx == 1: # Y-Z平面
                a = 0.2 * self.traj_scale
                b = 0.2 * self.traj_scale
                x = 0.4 * np.ones((len(t_param), 1))
                z = np.expand_dims(0.2 + 2 * a * np.sin(t_param) * np.cos(t_param) / (1 + np.sin(t_param)**2), axis=1)
                y = np.expand_dims(b * np.cos(t_param) / (1 + np.sin(t_param)**2), axis=1)
            else: # X-Y平面
                a = 0.2 * self.traj_scale 
                b = 0.2 * self.traj_scale
                z = 0.2 * np.ones((len(t_param), 1))
                x = np.expand_dims(0.3 + 2 * a * np.sin(t_param) * np.cos(t_param) / (1 + np.sin(t_param)**2), axis=1)
                y = np.expand_dims(b * np.cos(t_param) / (1 + np.sin(t_param)**2), axis=1)
            xyz_coords = np.concatenate((x, y, z), axis=1)
        
        elif traj_name == 'Circle':
            if self.idx == 1: # Y-Z平面
                radius = 0.1
                x = 0.4 * np.ones((len(t_param), 1))
                center_y, center_z = 0.0, 0.2
                y = np.expand_dims(center_y + radius * np.cos(t_param), axis=1)
                z = np.expand_dims(center_z + radius * np.sin(t_param), axis=1)
            else: # X-Y平面
                radius = 0.1
                z = 0.2 * np.ones((len(t_param), 1))
                center_x, center_y = 0.3, 0.0
                x = np.expand_dims(center_x + radius * np.cos(t_param), axis=1)
                y = np.expand_dims(center_y + radius * np.sin(t_param), axis=1)
            xyz_coords = np.concatenate((x, y, z), axis=1)
        else:
            raise ValueError(f"未知的轨迹名称: {traj_name}")

        print(f"笛卡尔轨迹生成完毕, 共 {len(xyz_coords)} 个点。")
        
        # 2. 求解逆运动学，将笛卡尔轨迹转换为关节角度轨迹
        print("开始将笛卡尔轨迹转换为关节角度 (使用 dm_control IK)...")
        joint_angles_trajectory = []
        
        # 重置物理状态到home位置，作为第一次IK求解的起点
        with self.physics.reset_context():
            self.physics.data.qpos[:] = np.zeros(self.physics.model.nq)
            
        for i, pos in enumerate(xyz_coords):
            # 保存当前成功解，以备下次失败时使用
            last_successful_qpos = self.physics.data.qpos.copy()
            
            # 对每个笛卡尔坐标点调用IK求解器
            q_sol = self._solve_ik(pos, target_orientation)
            
            if q_sol is not None:
                # 如果求解成功，将关节角度添加到轨迹列表中
                joint_angles_trajectory.append(q_sol)
                # 因为在 _solve_ik 中设置了 inplace=True，
                # self.physics.data.qpos 已经被更新为新的解，
                # 它将自动作为下一次求解的初始猜测值。
            else:
                # 如果求解失败
                print(f"警告: 逆运动学在时间步 {i} (目标位置: {np.round(pos, 3)}) 求解失败。")
                
                # 使用上一个成功的结果来填充，以保持轨迹的连续性
                if joint_angles_trajectory:
                    joint_angles_trajectory.append(joint_angles_trajectory[-1])
                    # 【重要】将物理状态重置回上一个成功点，避免从一个坏的姿态开始下一次求解
                    with self.physics.reset_context():
                        self.physics.data.qpos[:] = last_successful_qpos
                else:
                    # 如果连第一个点都失败了，说明目标点可能完全不可达，或者初始姿态太差
                    raise RuntimeError("轨迹的第一个点IK求解失败,请检查目标位置是否在机器人工作空间内。")
        
        print("关节角度轨迹转换完成。")
        
        # 返回生成的所有数据，这些数据将用于后续的控制和可视化
        return xyz_coords, np.array(joint_angles_trajectory), self.time_vector

# class CartesianTrajectoryGenerator_pinocchio:
#     """
#     生成笛卡尔空间轨迹，并使用 Pinocchio+CasADi 进行逆运动学求解，
#     最终输出关节角度轨迹。
#     """
#     def __init__(self, arm_model_path: str, ee_site_name: str, num_joints: int, 
#                  idx=1, time_horizon=60, time_steps_per_sec=5):
#         """
#         初始化轨迹生成器及内置的IK求解器。

#         Args:
#             arm_model_path (str): 用于IK的机械臂模型路径 (e.g., "so101_new_calib.xml")。
#             ee_site_name (str): 末端执行器在XML中的 <site> 名称。
#             num_joints (int): 机械臂的关节数量。
#             idx (int): 轨迹平面设置 (0 for x-y plane, 1 for y-z plane)。
#             time_horizon (float): 轨迹的总时长（秒）。
#             time_steps_per_sec (int): 每秒的轨迹点数量。
#         """
#         # 轨迹参数
#         self.idx = idx
#         self.time_horizon = time_horizon
#         self.time_steps_per_sec = time_steps_per_sec
#         self.total_steps = int(time_horizon * time_steps_per_sec)
#         self.time_vector = np.linspace(0, self.time_horizon, self.total_steps)
#         self.traj_scale = 0.5
        
#         # IK参数
#         self.arm_model_path = arm_model_path
#         self.ee_site_name = ee_site_name
#         self.num_joints = num_joints
        
#         # 初始化IK求解器
#         self._initialize_ik_solver()

#     def _initialize_ik_solver(self):
#         """加载模型并准备IK计算环境。"""
#         print("正在为IK求解器初始化 Pinocchio+CasADi 模型...")
#         try:
#             self.ik_solver = Kinematics(self.ee_site_name)
#             self.ik_solver.buildFromMJCF(self.arm_model_path)
#             print("IK求解器初始化完成。")
#         except Exception as e:
#             print(f"错误：无法从'{self.arm_model_path}'初始化IK模型。 {e}")
#             sys.exit(1)

#     def _solve_ik(self, target_tf: np.ndarray) -> np.ndarray:
#         """
#         内部IK求解函数，使用 Pinocchio+CasADi。

#         Args:
#             target_tf (np.ndarray): 4x4的目标变换矩阵。

#         Returns:
#             np.ndarray: 求解出的关节角度（弧度），如果失败则返回 None。
#         """
#         q_sol, info = self.ik_solver.ik(target_tf)
#         if info['success']:
#             return q_sol
#         else:
#             return None

#     def generate(self, traj_name='Fig8', target_orientation_matrix=np.eye(3)):
#         """
#         生成指定的笛卡尔轨迹并求解对应的关节角度轨迹。

#         Args:
#             traj_name (str): 轨迹名称 ('Fig8', 'Circle')。
#             target_orientation_matrix (np.ndarray): 3x3的旋转矩阵，定义末端执行器姿态。

#         Returns:
#             tuple: 包含三个元素的元组:
#                 - np.ndarray: 形状为 (N, 3) 的笛卡尔坐标点云。
#                 - np.ndarray: 形状为 (N, num_joints) 的关节角度轨迹（弧度）。
#                 - np.ndarray: 对应的时间向量。
#         """
#         # 1. 生成笛卡尔坐标点 (x, y, z)
#         t_param = 1.6 + 0.02 * np.linspace(0, self.time_horizon * 5, len(self.time_vector))
#         print(f"正在生成 '{traj_name}' 笛卡尔轨迹...")

#         if traj_name == 'Fig8':
#             if self.idx == 1: # Y-Z平面
#                 a = 0.2 * self.traj_scale
#                 b = 0.2 * self.traj_scale
#                 x = 0.4 * np.ones((len(t_param), 1))
#                 z = np.expand_dims(0.2 + 2 * a * np.sin(t_param) * np.cos(t_param) / (1 + np.sin(t_param)**2), axis=1)
#                 y = np.expand_dims(b * np.cos(t_param) / (1 + np.sin(t_param)**2), axis=1)
#             else: # X-Y平面
#                 a = 0.2 * self.traj_scale 
#                 b = 0.2 * self.traj_scale
#                 z = 0.2 * np.ones((len(t_param), 1))
#                 x = np.expand_dims(0.3 + 2 * a * np.sin(t_param) * np.cos(t_param) / (1 + np.sin(t_param)**2), axis=1)
#                 y = np.expand_dims(b * np.cos(t_param) / (1 + np.sin(t_param)**2), axis=1)
#             xyz_coords = np.concatenate((x, y, z), axis=1)
        
#         elif traj_name == 'Circle':
#             if self.idx == 1: # Y-Z平面
#                 radius = 0.1
#                 x = 0.4 * np.ones((len(t_param), 1))
#                 center_y, center_z = 0.0, 0.2
#                 y = np.expand_dims(center_y + radius * np.cos(t_param), axis=1)
#                 z = np.expand_dims(center_z + radius * np.sin(t_param), axis=1)
#             else: # X-Y平面
#                 radius = 0.1
#                 z = 0.2 * np.ones((len(t_param), 1))
#                 center_x, center_y = 0.3, 0.0
#                 x = np.expand_dims(center_x + radius * np.cos(t_param), axis=1)
#                 y = np.expand_dims(center_y + radius * np.sin(t_param), axis=1)
#             xyz_coords = np.concatenate((x, y, z), axis=1)
#         else:
#             raise ValueError(f"未知的轨迹名称: {traj_name}")
        
#         # 2. 求解逆运动学
#         print("开始将笛卡尔轨迹转换为关节角度 (使用 Pinocchio+CasADi IK)...")
#         joint_angles_trajectory = []
#         target_tf = np.eye(4)
#         target_tf[:3, :3] = target_orientation_matrix

#         for i, pos in enumerate(xyz_coords):
#             # 更新目标变换矩阵的位置部分
#             target_tf[:3, 3] = pos
            
#             # 调用内部IK求解器
#             q_sol = self._solve_ik(target_tf)
            
#             if q_sol is not None:
#                 joint_angles_trajectory.append(q_sol)
#             else:
#                 print(f"警告: 逆运动学在时间步 {i} (目标位置: {np.round(pos, 3)}) 求解失败。")
#                 if joint_angles_trajectory:
#                     # 使用上一个成功的结果来填充，保持轨迹连续性
#                     joint_angles_trajectory.append(joint_angles_trajectory[-1])
#                 else:
#                     raise RuntimeError("轨迹的第一个点IK求解失败, 请检查目标位置和姿态。")

#         print("关节角度轨迹转换完成。")
#         return xyz_coords, np.array(joint_angles_trajectory), self.time_vector


