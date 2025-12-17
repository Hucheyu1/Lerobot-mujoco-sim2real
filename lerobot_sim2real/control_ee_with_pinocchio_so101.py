import mujoco
import numpy as np
import mujoco.viewer
import time
import sys
import casadi_ik  # 导入你的 pinocchio+casadi IK 模块
from so101_mujoco import ZMQCommunicator
import os
import math
import mujoco_viewer
import matplotlib.pyplot as plt
# --- 修改后的主仿真类 ---
joint_offsets = [
    0,    # - (Motor 1 position: +1.49)
    0,    # - (Motor 2 position: +7.54)
    0,     # - (Motor 3 position: +0.7)
    0,    # - (Motor 4 position: -41.31)
    0,      # - (Motor 5 position: -0.7)
    # -31.97     # - (Motor 6 position: +4.48)
    -30
]

def sim_to_real(q_sim_deg, offsets):
    """MuJoCo 角度 (度) -> 真实机器人指令角度 (度)"""
    return [sim - off for sim, off in zip(q_sim_deg, offsets)]
# --- 笛卡尔轨迹生成器 (融合 Pinocchio+CasADi IK) ---
class CartesianTrajectoryGenerator:
    """
    生成笛卡尔空间轨迹，并使用 Pinocchio+CasADi 进行逆运动学求解，
    最终输出关节角度轨迹。
    """
    def __init__(self, arm_model_path: str, ee_site_name: str, num_joints: int, 
                 idx=1, time_horizon=60, time_steps_per_sec=5):
        """
        初始化轨迹生成器及内置的IK求解器。

        Args:
            arm_model_path (str): 用于IK的机械臂模型路径 (e.g., "so101_new_calib.xml")。
            ee_site_name (str): 末端执行器在XML中的 <site> 名称。
            num_joints (int): 机械臂的关节数量。
            idx (int): 轨迹平面设置 (0 for x-y plane, 1 for y-z plane)。
            time_horizon (float): 轨迹的总时长（秒）。
            time_steps_per_sec (int): 每秒的轨迹点数量。
        """
        # 轨迹参数
        self.idx = idx
        self.time_horizon = time_horizon
        self.time_steps_per_sec = time_steps_per_sec
        self.total_steps = int(time_horizon * time_steps_per_sec)
        self.time_vector = np.linspace(0, self.time_horizon, self.total_steps)
        self.traj_scale = 0.5
        
        # IK参数
        self.arm_model_path = arm_model_path
        self.ee_site_name = ee_site_name
        self.num_joints = num_joints
        
        # 初始化IK求解器
        self._initialize_ik_solver()

    def _initialize_ik_solver(self):
        """加载模型并准备IK计算环境。"""
        print("正在为IK求解器初始化 Pinocchio+CasADi 模型...")
        try:
            self.ik_solver = casadi_ik.Kinematics(self.ee_site_name)
            self.ik_solver.buildFromMJCF(self.arm_model_path)
            print("IK求解器初始化完成。")
        except Exception as e:
            print(f"错误：无法从'{self.arm_model_path}'初始化IK模型。 {e}")
            sys.exit(1)

    def _solve_ik(self, target_tf: np.ndarray) -> np.ndarray:
        """
        内部IK求解函数，使用 Pinocchio+CasADi。

        Args:
            target_tf (np.ndarray): 4x4的目标变换矩阵。

        Returns:
            np.ndarray: 求解出的关节角度（弧度），如果失败则返回 None。
        """
        q_sol, info = self.ik_solver.ik(target_tf)
        if info['success']:
            return q_sol
        else:
            return None

    def generate(self, traj_name='Fig8', target_orientation_matrix=np.eye(3)):
        """
        生成指定的笛卡尔轨迹并求解对应的关节角度轨迹。

        Args:
            traj_name (str): 轨迹名称 ('Fig8', 'Circle')。
            target_orientation_matrix (np.ndarray): 3x3的旋转矩阵，定义末端执行器姿态。

        Returns:
            tuple: 包含三个元素的元组:
                - np.ndarray: 形状为 (N, 3) 的笛卡尔坐标点云。
                - np.ndarray: 形状为 (N, num_joints) 的关节角度轨迹（弧度）。
                - np.ndarray: 对应的时间向量。
        """
        # 1. 生成笛卡尔坐标点 (x, y, z)
        t_param = 1.6 + 0.02 * np.linspace(0, self.time_horizon * 5, len(self.time_vector))
        print(f"正在生成 '{traj_name}' 笛卡尔轨迹...")

        if traj_name == 'Fig8':
            if self.idx == 1: # Y-Z平面
                a = 0.2 * self.traj_scale
                b = 0.2 * self.traj_scale
                x = 0.4 * np.ones((len(t_param), 1))
                z = np.expand_dims(0.2 + 2 * a * np.sin(t_param) * np.cos(t_param) / (1 + np.sin(t_param)**2), axis=1)
                y = np.expand_dims(b * np.cos(t_param) / (1 + np.sin(t_param)**2), axis=1)
            else: # X-Y平面
                a = 0.25 * self.traj_scale 
                b = 0.25 * self.traj_scale
                z = 0.055 * np.ones((len(t_param), 1))
                x = np.expand_dims(0.35 + 2 * a * np.sin(t_param) * np.cos(t_param) / (1 + np.sin(t_param)**2), axis=1)
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
        elif traj_name == 'FigStar':
            # --- 1. 定义几何参数 ---
            # 外部圆（五角星的五个尖角所在圆）的参数
            a = 0.22 * self.traj_scale  # 使用 a 来控制整体尺寸（外部半径）
            center_x, center_y, center_z = 0.35, 0.0, 0.055 # 统一中心点
            # 定义五角星的半径和中心点
            radius = a         # 外部半径（尖角到中心）
            # 根据 self.idx 确定 2D 形状的中心偏移
            if self.idx == 1:
                 # Y-Z 平面：形状中心是 (center_y, center_z)
                center_prime_0 = center_y
                center_prime_1 = center_z
            else:
                # X-Y 平面：形状中心是 (center_x, center_y)
                center_prime_0 = center_x
                center_prime_1 = center_y      
            eradius = radius * np.sin(np.pi / 10.0) / np.sin(3 * np.pi / 10.0)
            # --- 2. 计算 11 个关键点 (5个尖角 + 5个凹陷 + 1个闭合点) ---
            Star_points_2D = np.zeros((11, 2))
            for i in range(5):
                # 尖角 (Outer Points)
                theta_outer = (2 * np.pi / 5) * i + (np.pi / 2) 
                Star_points_2D[2 * i, 0] = np.cos(theta_outer) * radius + center_prime_0
                Star_points_2D[2 * i, 1] = np.sin(theta_outer) * radius + center_prime_1
                
                # 凹陷点 (Inner Points)
                theta_inner = (2 * np.pi / 5) * i + (np.pi / 2) + (np.pi / 5)
                Star_points_2D[2 * i + 1, 0] = np.cos(theta_inner) * eradius + center_prime_0
                Star_points_2D[2 * i + 1, 1] = np.sin(theta_inner) * eradius + center_prime_1
            # 闭合轨迹：第 11 个点 = 第 1 个点
            Star_points_2D[-1, :] = Star_points_2D[0, :]
            # --- 3. 轨迹插值（使用与您代码类似的线性插值方法） ---
            # 计算总时间步数
            num_steps = len(t_param) # 使用您的 t 或 self.time_vector 
            # 假设轨迹分段均匀 (10个线段)
            num_segments = 10 
            refs = np.zeros((num_steps, 2))
            # 计算每段轨迹包含的步数
            each_num = num_steps // num_segments
            current_step = 0
            for i in range(num_segments):
                start_point = Star_points_2D[i, :]
                end_point = Star_points_2D[i + 1, :]
                # 确保最后一段占满剩余所有步数
                num_interp_points = each_num if i < num_segments - 1 else num_steps - current_step
                for j in range(num_interp_points):
                    t_ = j / (num_interp_points - 1) if num_interp_points > 1 else 0.0
                    
                    refs[current_step + j, :] = t_ * end_point + (1 - t_) * start_point
                current_step += num_interp_points
            # --- 4. 组装 3D 坐标 (Y-Z平面，X固定) ---
            if self.idx==1:
                x = center_x * np.ones((num_steps, 1)) 
                y = refs[:, 0].reshape(-1, 1) # Y 对应 2D 坐标的第一个分量 (x')
                z = refs[:, 1].reshape(-1, 1) # Z 对应 2D 坐标的第二个分量 (y')
            else:
                x = refs[:, 0].reshape(-1, 1) 
                y = refs[:, 1].reshape(-1, 1) 
                z = center_z * np.ones((num_steps, 1))    
            xyz_coords = np.concatenate( (x, y, z), axis = 1)
        else:
            raise ValueError(f"未知的轨迹名称: {traj_name}")
        
        # 2. 求解逆运动学
        print("开始将笛卡尔轨迹转换为关节角度 (使用 Pinocchio+CasADi IK)...")
        joint_angles_trajectory = []
        target_tf = np.eye(4)
        target_tf[:3, :3] = target_orientation_matrix

        for i, pos in enumerate(xyz_coords):
            # 更新目标变换矩阵的位置部分
            target_tf[:3, 3] = pos
            
            # 调用内部IK求解器
            q_sol = self._solve_ik(target_tf)
            
            if q_sol is not None:
                joint_angles_trajectory.append(q_sol)
            else:
                print(f"警告: 逆运动学在时间步 {i} (目标位置: {np.round(pos, 3)}) 求解失败。")
                if joint_angles_trajectory:
                    # 使用上一个成功的结果来填充，保持轨迹连续性
                    joint_angles_trajectory.append(joint_angles_trajectory[-1])
                else:
                    raise RuntimeError("轨迹的第一个点IK求解失败, 请检查目标位置和姿态。")

        print("关节角度轨迹转换完成。")
        return xyz_coords, np.array(joint_angles_trajectory), self.time_vector

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path, communicator, cartesian_points, joint_angle_traj, num_joints, draw_num, use_noise=False):
        """
        初始化参数
        :param path: XML 模型路径
        :param communicator: 通信器实例
        :param cartesian_points: 笛卡尔空间轨迹点 (N, 3)
        :param joint_angle_traj: 关节空间轨迹 (N, num_joints)
        :param num_joints: 机器人的关节数量 (例如 5, 6, 7)
        """
        # 调用父类构造函数 (根据你提供的基类签名)
        super().__init__(path, 1.5, azimuth=135, elevation=-30)
        
        self.path = path
        self.communicator = communicator
        self.use_noise = use_noise
        # 保存轨迹数据
        self.cartesian_points = cartesian_points
        self.joint_angle_traj = joint_angle_traj
        self.num_joints = num_joints
        
        # 轨迹播放进度计数器
        self.traj_index = 0
        self.total_frames = len(joint_angle_traj)
        self.actual_traj = []
        # 预计算采样步长 (防止点太多卡顿)
        # 保证屏幕上最多显示 300-500 个红点
        self.draw_step = max(1, len(cartesian_points) // draw_num)
        print(f"轨迹总长: {self.total_frames}, 绘图采样步长: {self.draw_step}")
        # --- 新增：读取 Home Keyframe ---
        self.home_qpos = None
        try:
            # 1. 获取名为 "home" 的关键帧 ID
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            
            # 2. 如果 ID 有效 (>=0)，则读取对应的 qpos 数据
            if key_id >= 0:
                # model.key_qpos 是一个 (nkey, nq) 的数组
                self.home_qpos = self.model.key_qpos[key_id].copy()
                print(f"已加载 'home' 姿态: {self.home_qpos}")
            else:
                print("警告: XML中未找到名为 'home' 的 <keyframe>")
        except Exception as e:
            print(f"读取 Keyframe 出错: {e}")

        # --- 2. 回归轨迹相关的变量 ---
        self.return_traj = None     # 存储生成的回归路径
        self.return_index = 0       # 回归播放进度
        self.return_duration = 2.0  # 回归过程耗时 2秒

        # --- 关节和末端执行器定义 ---
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        self.joint_ids = [self.model.joint(name).id for name in self.joint_names]
        
        try:
            self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        except ValueError:
            raise ValueError("在模型中未找到名为 'gripper' 的 site。请检查XML文件。")

    def runBefore(self):
        """
        在仿真循环开始前执行一次
        """
        # 将机器人复位到轨迹的起始位置
        if self.total_frames > 0:
            self.data.qpos[:self.num_joints] = self.joint_angle_traj[0][:self.num_joints]
            mujoco.mj_forward(self.model, self.data)
        print("仿真即将开始...")
        # --- 1. 绘制静态轨迹 (红色小球) ---
        # 注意：必须每帧都画，因为 viewer.sync() 会清空 user_scn
        for pt in self.cartesian_points[::self.draw_step]:
            # 安全检查：如果 geometry 满了就不画了
            if self.handle.user_scn.ngeom >= self.handle.user_scn.maxgeom:
                break
            
            mujoco.mjv_initGeom(
                self.handle.user_scn.geoms[self.handle.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.002, 0, 0],       # 2mm 红球
                pos=pt,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 0.3]       # 半透明红
            )
            self.handle.user_scn.ngeom += 1

    def runFunc(self):
        """
        每一帧都会被调用的函数
        """
        # --- 代码：重力补偿 ---
        # `data.qfrc_bias` 存储了由重力、科里奥利力等产生的偏置力矩。
        # 对于静止或慢速运动的机器人，它主要就是重力力矩
        step_start = time.time()

        self.data.qfrc_applied[:] =  self.data.qfrc_bias[:]
        # --- 2. 机器人运动控制 ---
        # 如果轨迹还没播完
        if self.traj_index < self.total_frames:
            if self.use_noise:
                # 定义扰动的最大强度（例如：±0.01 弧度）
                angle_disturbance_magnitude = 0.02
                # 选项 A: 扰动关节位置 (qpos)
                # qpos 尺寸为 (nq,)，即广义坐标位置维度
                # 生成一个随机扰动角度 (均匀分布在 [-magnitude, +magnitude])
                angle_disturb = np.random.uniform(
                    low=-angle_disturbance_magnitude,
                    high=angle_disturbance_magnitude,
                    size=self.num_joints  # 扰动作用于所有关节位置
                )
                # 将扰动加到当前的关节角度上
                self.data.qpos[:self.num_joints] = self.joint_angle_traj[self.traj_index][:self.num_joints] + angle_disturb
            else:
                self.data.qpos[:self.num_joints] = self.joint_angle_traj[self.traj_index][:self.num_joints]
            # 前向动力学计算
            mujoco.mj_forward(self.model, self.data)
            qpos = self.data.qpos[self.joint_ids].copy()
            # qvel = self.data.qvel[self.joint_ids].copy()
            ee_pos = self.data.site_xpos[self.ee_site_id].copy()
            self.actual_traj.append(np.concatenate([ee_pos, qpos]).astype(np.float32))
            # 绘制当前目标点 (绿色大球)
            current_pos = ee_pos
            if self.handle.user_scn.ngeom <= self.handle.user_scn.maxgeom:
                mujoco.mjv_initGeom(
                    self.handle.user_scn.geoms[self.handle.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.002, 0, 0],    # 1cm 绿球
                    pos=current_pos,
                    mat=np.eye(3).flatten(),
                    rgba=[0, 1, 0, 1]     # 不透明绿
                )
                self.handle.user_scn.ngeom += 1
            
            # 进度 +1
            self.traj_index += 1
            
            # 如果觉得播放太快，可以在这里加一点点延时，但通常不建议在 GUI 线程 sleep 太久
            # time.sleep(0.002) 
            
        # --- 阶段 2: 主轨迹刚结束，生成回归路径 (只执行一次) ---
        elif self.return_traj is None and self.home_qpos is not None:
            print("主轨迹播放完毕，生成回归 Home 的路径...")
            np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),"traj_noise.npy"),np.array(self.actual_traj))
            start_qpos = self.joint_angle_traj[-1][:self.num_joints]  # 当前位置
            end_qpos = self.home_qpos[:self.num_joints]  # 目标位置
            
            # 计算需要多少帧 (假设 timestep=0.002, 持续 2秒 => 1000帧)
            steps = int(self.return_duration / 0.1)
            
            # 生成插值轨迹 (Shape: [steps, num_joints])
            # 使用 linspace 生成平滑过渡
            # 如果想要非线性平滑(ease-in-out)，可以使用 cos 函数处理 steps
            self.return_traj = np.linspace(start_qpos, end_qpos, steps)
            
            self.return_index = 0 # 准备开始播放回归

        # --- 阶段 3: 播放回归轨迹 ---
        elif self.return_traj is not None and self.return_index < len(self.return_traj):
            # 设置回归过程中的关节角度
            self.data.qpos[:self.num_joints] = self.return_traj[self.return_index][:self.num_joints]
            mujoco.mj_forward(self.model, self.data)
            
            # 这里不再绘制绿球，因为已经在“回家”路上了
            self.return_index += 1

        # --- 阶段 4: 全部结束，保持 Home 姿态 ---
        else:
            if self.home_qpos is not None:
                self.data.qpos[:self.num_joints] = self.home_qpos[:self.num_joints]
            else:
                # 如果没有 home，就停在轨迹终点
                self.data.qpos[:self.num_joints] = self.joint_angle_traj[-1][:self.num_joints]
                
            mujoco.mj_forward(self.model, self.data)

        # --- 3. 通信器逻辑 (保留你的原始逻辑) ---
        sim_joint_rad = self.data.qpos[:6].copy() # 注意：如果是纯位置回放，ctrl可能为0，除非你在别处设置了
        # 将弧度转换为角度
        sim_joint_deg = [math.degrees(q) for q in sim_joint_rad]        
        q_real_target_deg = sim_to_real(sim_joint_deg, joint_offsets)
        self.communicator.send_data(q_real_target_deg)

        time_until_next_step = 0.02 - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        # time.sleep(0.01)  # 控制发送频

# --- 主程序 (变得非常简洁) ---
if __name__ == "__main__":
    # --- 配置 ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    SCENE_XML_PATH = os.path.join(project_root, "SOARM101", "SO101", "scene_with_table.xml")
    ARM_XML_PATH = os.path.join(project_root, "SOARM101", "SO101", "so101_new_calib.xml")
    EE_SITE_NAME = 'gripperframe'
    NUM_JOINTS = 5
    use_noise =  True
    use_plot = False
    # --- 步骤 1: 初始化MuJoCo环境 ---
    try:
        model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"错误: 无法加载MuJoCo场景 '{SCENE_XML_PATH}'. {e}")
        sys.exit(1)
        
    # --- 步骤 2: 创建全功能轨迹生成器实例 ---
    traj_generator = CartesianTrajectoryGenerator(
        arm_model_path=ARM_XML_PATH,
        ee_site_name=EE_SITE_NAME,
        num_joints=NUM_JOINTS,
        idx = 0,
        time_horizon=60,
        time_steps_per_sec=10
    )
    # --- 步骤 3: 一行代码生成所有轨迹数据 ---
    # 角度（度）
    angle_degrees = 90  # 0 90

    # 转换为弧度
    angle_radians = math.radians(angle_degrees)

    # 计算 cos 和 sin 值
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    aix = "X"
    # 构建绕 X 轴旋转的矩阵
    if aix == "X":
        target_orientation = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    # 构建绕 Y 轴旋转的矩阵
    elif aix == "Y":
        target_orientation = np.array([
            [c,  0, s],
            [0,  1, 0],
            [-s, 0, c]
        ])
    # 构建绕 Z 轴旋转的矩阵
    else:
        target_orientation = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
    # 调用generate方法，它会完成笛卡尔轨迹生成和IK求解两项工作
    cartesian_points, joint_angle_traj, time_vec = traj_generator.generate(
        traj_name='FigStar',  # Circle, Fig8, FigStar
        target_orientation_matrix=target_orientation
    )
    zmq_communicator = ZMQCommunicator("tcp://127.0.0.1:5555")

    try:
        if not use_plot:
            # 实例化播放器
            test = Test(
                path=SCENE_XML_PATH,
                communicator = zmq_communicator, # 传入你的通信器
                cartesian_points=cartesian_points,
                joint_angle_traj=joint_angle_traj,
                num_joints = NUM_JOINTS,
                draw_num = 300,
                use_noise = use_noise
            )
            # 启动
            test.run_loop()

    except KeyboardInterrupt:
        print("仿真程序被用户中断")
    finally:
        # 清理通信资源
        zmq_communicator.cleanup()
        if use_plot:
            traj_noise = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"traj_noise.npy"))
            pos = traj_noise[:,:3]
            fig, ax= plt.subplots(figsize=(10, 8))
            ax.plot(
                    cartesian_points[:, 0], cartesian_points[:, 1],
                    color='black', 
                    alpha = 1.0,
                    label='IBKN-δMPC-UKF'
                )
            ax.plot(
                    pos[:, 0], pos[:, 1],
                    color='red', 
                    alpha = 1.0,
                    label='IBKN-δMPC'
                )
            # 建议：添加图例、标签和网格以便观察
            ax.set_xlabel('X Position',fontsize=18)
            ax.set_ylabel('Y Position',fontsize=18)
            # 3. 设置刻度字体大小
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.legend(fontsize=18)
            ax.grid(True)
            plt.savefig(
                os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__))), "traj.png"), format="png",\
                    dpi=500, bbox_inches='tight'
            )
            plt.show()