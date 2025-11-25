import mujoco
import numpy as np
import mujoco_viewer
from TrajectoryGenerator import CartesianTrajectoryGenerator
from so101_mujoco import ZMQCommunicator
import time
import os
import math
# --- 修改后的主仿真类 ---
joint_offsets = [
    0,    # - (Motor 1 position: +1.49)
    0,    # - (Motor 2 position: +7.54)
    0,     # - (Motor 3 position: +0.7)
    0,    # - (Motor 4 position: -41.31)
    0,      # - (Motor 5 position: -0.7)
    -31.97     # - (Motor 6 position: +4.48)
]

def sim_to_real(q_sim_deg, offsets):
    """MuJoCo 角度 (度) -> 真实机器人指令角度 (度)"""
    return [sim - off for sim, off in zip(q_sim_deg, offsets)]

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path, communicator, cartesian_points, joint_angle_traj, num_joints):
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
        
        # 保存轨迹数据
        self.cartesian_points = cartesian_points
        self.joint_angle_traj = joint_angle_traj
        self.num_joints = num_joints
        
        # 轨迹播放进度计数器
        self.traj_index = 0
        self.total_frames = len(joint_angle_traj)
        
        # 预计算采样步长 (防止点太多卡顿)
        # 保证屏幕上最多显示 300-500 个红点
        self.draw_step = max(1, len(cartesian_points) // 500)
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

    def runBefore(self):
        """
        在仿真循环开始前执行一次
        """
        # 将机器人复位到轨迹的起始位置
        if self.total_frames > 0:
            self.data.qpos[:self.num_joints] = self.joint_angle_traj[0]
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

        # --- 2. 机器人运动控制 ---
        # 如果轨迹还没播完
        if self.traj_index < self.total_frames:
            # 设置当前帧的关节角度
            self.data.qpos[:self.num_joints] = self.joint_angle_traj[self.traj_index]
            
            # 前向动力学计算
            mujoco.mj_forward(self.model, self.data)
            
            # 绘制当前目标点 (绿色大球)
            current_pos = self.cartesian_points[self.traj_index]
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
            
            start_qpos = self.joint_angle_traj[-1]       # 当前位置
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
            self.data.qpos[:self.num_joints] = self.return_traj[self.return_index]
            mujoco.mj_forward(self.model, self.data)
            
            # 这里不再绘制绿球，因为已经在“回家”路上了
            self.return_index += 1

        # --- 阶段 4: 全部结束，保持 Home 姿态 ---
        else:
            if self.home_qpos is not None:
                self.data.qpos[:self.num_joints] = self.home_qpos[:self.num_joints]
            else:
                # 如果没有 home，就停在轨迹终点
                self.data.qpos[:self.num_joints] = self.joint_angle_traj[-1]
                
            mujoco.mj_forward(self.model, self.data)

        # --- 3. 通信器逻辑 (保留你的原始逻辑) ---
        sim_joint_rad = self.data.qpos[:6].copy() # 注意：如果是纯位置回放，ctrl可能为0，除非你在别处设置了
        # 将弧度转换为角度
        sim_joint_deg = [math.degrees(q) for q in sim_joint_rad]        
        q_real_target_deg = sim_to_real(sim_joint_deg, joint_offsets)
        self.communicator.send_data(q_real_target_deg)
        time.sleep(0.01)  # 控制发送频

if __name__ == "__main__":
    # --- 0. 基本配置 ---
    # 示例参数，请替换为您自己的模型信息
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    MODEL_XML_PATH = os.path.join(project_root, "model", "SO101", "scene_with_table.xml")
    EE_SITE_NAME = 'gripperframe' # 你的XML里定义的夹爪中心的 <site>
    NUM_JOINTS = 5 # 你的机器人关节数量

    # 创建生成器实例
    traj_generator = CartesianTrajectoryGenerator(
        model_path=MODEL_XML_PATH,
        ee_site_name=EE_SITE_NAME,
        num_joints=NUM_JOINTS
    )

    # 定义末端执行器在整个轨迹中要保持的姿态 (例如，垂直向下)
    target_quat = None # 绕X轴旋转90度
    # target_quat = np.array([0, 0, 1, 0]) 

    # 调用generate方法，反解出关节角度
    cartesian_points, joint_angle_traj, time_vec = traj_generator.generate(
        traj_name='Fig8',  # Circle, Fig8
        target_orientation=target_quat
    )
    
    zmq_communicator = ZMQCommunicator("tcp://127.0.0.1:5555")
    try:
        # 实例化播放器
        test = Test(
            path=MODEL_XML_PATH,
            communicator = zmq_communicator, # 传入你的通信器
            cartesian_points=cartesian_points,
            joint_angle_traj=joint_angle_traj,
            num_joints = NUM_JOINTS
        )
        # 启动
        test.run_loop()

    except KeyboardInterrupt:
        print("仿真程序被用户中断")
    finally:
        # 清理通信资源
        zmq_communicator.cleanup()

