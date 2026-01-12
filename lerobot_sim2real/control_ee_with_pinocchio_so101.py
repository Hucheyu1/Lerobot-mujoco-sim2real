import math
import os
import sys
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco_viewer
import numpy as np
from so101_mujoco import ZMQCommunicator
from TrajectoryGenerator import CartesianTrajectoryGenerator_Pinocchio

# --- 修改后的主仿真类 ---
joint_offsets = [
    0,  # - (Motor 1 position: +1.49)
    0,  # - (Motor 2 position: +7.54)
    0,  # - (Motor 3 position: +0.7)
    0,  # - (Motor 4 position: -41.31)
    0,  # - (Motor 5 position: -0.7)
    # -31.97     # - (Motor 6 position: +4.48)
    -30,
]


def sim_to_real(q_sim_deg, offsets):
    """MuJoCo 角度 (度) -> 真实机器人指令角度 (度)"""
    return [sim - off for sim, off in zip(q_sim_deg, offsets)]


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
        self.return_traj = None  # 存储生成的回归路径
        self.return_index = 0  # 回归播放进度
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
            self.data.qpos[: self.num_joints] = self.joint_angle_traj[0][: self.num_joints]
            mujoco.mj_forward(self.model, self.data)
        print("仿真即将开始...")
        # --- 1. 绘制静态轨迹 (红色小球) ---
        # 注意：必须每帧都画，因为 viewer.sync() 会清空 user_scn
        for pt in self.cartesian_points[:: self.draw_step]:
            # 安全检查：如果 geometry 满了就不画了
            if self.handle.user_scn.ngeom >= self.handle.user_scn.maxgeom:
                break

            mujoco.mjv_initGeom(
                self.handle.user_scn.geoms[self.handle.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.002, 0, 0],  # 2mm 红球
                pos=pt,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 0.3],  # 半透明红
            )
            self.handle.user_scn.ngeom += 1

        # self.sync()
        # sim_joint_rad = self.data.qpos[:6].copy()
        # # 将弧度转换为角度
        # sim_joint_deg = [math.degrees(q) for q in sim_joint_rad]
        # q_real_target_deg = sim_to_real(sim_joint_deg, joint_offsets)
        # self.communicator.send_data(q_real_target_deg)
        # input("按 Enter 开始...")

    def runFunc(self):
        """
        每一帧都会被调用的函数
        """
        # --- 代码：重力补偿 ---
        # `data.qfrc_bias` 存储了由重力、科里奥利力等产生的偏置力矩。
        # 对于静止或慢速运动的机器人，它主要就是重力力矩
        step_start = time.time()

        self.data.qfrc_applied[:] = self.data.qfrc_bias[:]
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
                    size=self.num_joints,  # 扰动作用于所有关节位置
                )
                # 将扰动加到当前的关节角度上
                self.data.qpos[: self.num_joints] = (
                    self.joint_angle_traj[self.traj_index][: self.num_joints] + angle_disturb
                )
            else:
                self.data.qpos[: self.num_joints] = self.joint_angle_traj[self.traj_index][: self.num_joints]
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
                    size=[0.002, 0, 0],  # 1cm 绿球
                    pos=current_pos,
                    mat=np.eye(3).flatten(),
                    rgba=[0, 1, 0, 1],  # 不透明绿
                )
                self.handle.user_scn.ngeom += 1

            # 进度 +1
            self.traj_index += 1

            # 如果觉得播放太快，可以在这里加一点点延时，但通常不建议在 GUI 线程 sleep 太久
            # time.sleep(0.002)

        # --- 阶段 2: 主轨迹刚结束，生成回归路径 (只执行一次) ---
        elif self.return_traj is None and self.home_qpos is not None:
            print("主轨迹播放完毕，生成回归 Home 的路径...")
            np.save(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "traj_noise.npy"), np.array(self.actual_traj)
            )
            start_qpos = self.joint_angle_traj[-1][: self.num_joints]  # 当前位置
            end_qpos = self.home_qpos[: self.num_joints]  # 目标位置

            # 计算需要多少帧 (假设 timestep=0.002, 持续 2秒 => 1000帧)
            steps = int(self.return_duration / 0.1)

            # 生成插值轨迹 (Shape: [steps, num_joints])
            # 使用 linspace 生成平滑过渡
            # 如果想要非线性平滑(ease-in-out)，可以使用 cos 函数处理 steps
            self.return_traj = np.linspace(start_qpos, end_qpos, steps)

            self.return_index = 0  # 准备开始播放回归

        # --- 阶段 3: 播放回归轨迹 ---
        elif self.return_traj is not None and self.return_index < len(self.return_traj):
            # 设置回归过程中的关节角度
            self.data.qpos[: self.num_joints] = self.return_traj[self.return_index][: self.num_joints]
            mujoco.mj_forward(self.model, self.data)

            # 这里不再绘制绿球，因为已经在“回家”路上了
            self.return_index += 1

        # --- 阶段 4: 全部结束，保持 Home 姿态 ---
        else:
            if self.home_qpos is not None:
                self.data.qpos[: self.num_joints] = self.home_qpos[: self.num_joints]
            else:
                # 如果没有 home，就停在轨迹终点
                self.data.qpos[: self.num_joints] = self.joint_angle_traj[-1][: self.num_joints]

            mujoco.mj_forward(self.model, self.data)

        # --- 3. 通信器逻辑 (保留你的原始逻辑) ---
        sim_joint_rad = self.data.qpos[:6].copy()  # 注意：如果是纯位置回放，ctrl可能为0，除非你在别处设置了
        # 将弧度转换为角度
        sim_joint_deg = [math.degrees(q) for q in sim_joint_rad]
        q_real_target_deg = sim_to_real(sim_joint_deg, joint_offsets)
        self.communicator.send_data(q_real_target_deg)

        # 3. 进入忙等待循环：只要当前时间还没到 target_time，就一直循环发送
        # while time.time() - step_start < 0.02:
        #     # 再次发送相同的数据（起到 Keep-Alive 或高频刷新的作用）
        #     self.communicator.send_data(q_real_target_deg)

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
    EE_SITE_NAME = "gripperframe"
    NUM_JOINTS = 5
    use_noise = False
    use_plot = False
    idx = 0  # 0 or 1
    traj_name = "Heart"  # Rectangle, Fig8, FigStar, Heart, Helix, Lissajous
    # --- 步骤 1: 初始化MuJoCo环境 ---
    try:
        model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"错误: 无法加载MuJoCo场景 '{SCENE_XML_PATH}'. {e}")
        sys.exit(1)

    # --- 步骤 2: 创建全功能轨迹生成器实例 ---
    traj_generator = CartesianTrajectoryGenerator_Pinocchio(
        arm_model_path=ARM_XML_PATH,
        ee_site_name=EE_SITE_NAME,
        num_joints=NUM_JOINTS,
        idx=idx,
        time_horizon=60,
        time_steps_per_sec=10,
    )
    # --- 步骤 3: 一行代码生成所有轨迹数据 ---
    # 角度（度）
    angle_degrees = 90 if idx == 0 else 0  # 0 90

    # 转换为弧度
    angle_radians = math.radians(angle_degrees)

    # 计算 cos 和 sin 值
    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    aix = "X"
    # 构建绕 X 轴旋转的矩阵
    if aix == "X":
        target_orientation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    # 构建绕 Y 轴旋转的矩阵
    elif aix == "Y":
        target_orientation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    # 构建绕 Z 轴旋转的矩阵
    else:
        target_orientation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # 调用generate方法，它会完成笛卡尔轨迹生成和IK求解两项工作
    cartesian_points, joint_angle_traj, time_vec = traj_generator.generate(
        traj_name=traj_name,
        target_orientation_matrix=target_orientation,
    )
    zmq_communicator = ZMQCommunicator("tcp://127.0.0.1:5555")

    try:
        if not use_plot:
            # 实例化播放器
            test = Test(
                path=SCENE_XML_PATH,
                communicator=zmq_communicator,  # 传入你的通信器
                cartesian_points=cartesian_points,
                joint_angle_traj=joint_angle_traj,
                num_joints=NUM_JOINTS,
                draw_num=300,
                use_noise=use_noise,
            )
            # 启动
            test.run_loop()

    except KeyboardInterrupt:
        print("仿真程序被用户中断")
    finally:
        # 清理通信资源
        zmq_communicator.cleanup()
        if use_plot:
            traj_noise = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "traj_noise.npy"))
            pos = traj_noise[:, :3]
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(cartesian_points[:, 0], cartesian_points[:, 1], color="black", alpha=1.0, label="IBKN-δMPC-UKF")
            ax.plot(pos[:, 0], pos[:, 1], color="red", alpha=1.0, label="IBKN-δMPC")
            # 建议：添加图例、标签和网格以便观察
            ax.set_xlabel("X Position", fontsize=18)
            ax.set_ylabel("Y Position", fontsize=18)
            # 3. 设置刻度字体大小
            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.legend(fontsize=18)
            ax.grid(True)
            plt.savefig(
                os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__))), "traj.png"),
                format="png",
                dpi=500,
                bbox_inches="tight",
            )
            plt.show()
