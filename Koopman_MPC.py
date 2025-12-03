import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import mujoco
import numpy as np
from utility.ZMQ import ZMQCommunicator
from control.TrajectoryGenerator import CartesianTrajectoryGenerator
from control.MPC_Controler import MPCController
import time
import math
from args import Args
import torch
from SOARM101.SOARM101_Env import SOARM101Env

from models.init_model import init_model
# --- 修改后的主仿真类 ---
joint_offsets = [
    0,    # - (Motor 1 position: +1.49)
    0,    # - (Motor 2 position: +7.54)
    0,     # - (Motor 3 position: +0.7)
    0,    # - (Motor 4 position: -41.31)
    0,      # - (Motor 5 position: -0.7)
    -41.97     # - (Motor 6 position: +4.48) 
]

def sim_to_real(q_sim_deg, offsets):
    """MuJoCo 角度 (度) -> 真实机器人指令角度 (度)"""
    return [sim - off for sim, off in zip(q_sim_deg, offsets)]

class Test:
    def __init__(self, path, Controller, communicator, cartesian_points, joint_angle_traj, num_joints, draw_num):
        """
        初始化参数
        :param path: XML 模型路径
        :param communicator: 通信器实例
        :param cartesian_points: 笛卡尔空间轨迹点 (N, 3)
        :param joint_angle_traj: 关节空间轨迹 (N, num_joints)
        :param num_joints: 机器人的关节数量 (例如 5, 6, 7)
        """
        
        self.path = path
        self.communicator = communicator
        self.env = SOARM101Env(path, render_mode=True)
        # 控制相关
        self.mpc_controller = Controller
        self.H = Controller.H
        # 保存轨迹数据
        self.actual_traj = []
        self.cartesian_points = cartesian_points # (N, 3)
        self.joint_angle_traj = joint_angle_traj # (N, 5)
        self.state_all_ref = np.hstack([cartesian_points, joint_angle_traj])
        self.num_joints = num_joints
        
        # 轨迹播放进度计数器
        self.traj_index = 0
        self.total_frames = len(joint_angle_traj)
        
        # 预计算采样步长 (防止点太多卡顿)
        # 保证屏幕上最多显示 300-500 个红点
        self.draw_step = max(1, len(cartesian_points) // draw_num)
        print(f"轨迹总长: {self.total_frames}, 绘图采样步长: {self.draw_step}")
        # --- 新增：读取 Home Keyframe ---
        self.home_qpos = None
        try:
            # 1. 获取名为 "home" 的关键帧 ID
            key_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            
            # 2. 如果 ID 有效 (>=0)，则读取对应的 qpos 数据
            if key_id >= 0:
                # model.key_qpos 是一个 (nkey, nq) 的数组
                self.home_qpos = self.env.model.key_qpos[key_id].copy()
                self.home_ctrl = self.env.model.key_ctrl[key_id].copy()
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
            self.env.data.qpos[:self.num_joints] = self.joint_angle_traj[0]
            mujoco.mj_forward(self.env.model, self.env.data)
            self.state_tensor = torch.DoubleTensor(self.state_all_ref[0]).unsqueeze(0)
        print("仿真即将开始...")
        # --- 1. 绘制静态轨迹 (红色小球) ---
        # 注意：必须每帧都画，因为 viewer.sync() 会清空 user_scn
        for pt in self.cartesian_points[::self.draw_step]:
            # 安全检查：如果 geometry 满了就不画了
            if self.env.viewer.user_scn.ngeom >= self.env.viewer.user_scn.maxgeom:
                break
            
            mujoco.mjv_initGeom(
                self.env.viewer.user_scn.geoms[self.env.viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.001, 0, 0],       # 2mm 红球
                pos=pt,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 0.3]       # 半透明红
            )
            self.env.viewer.user_scn.ngeom += 1

    def runFunc(self):
        """
        每一帧都会被调用的函数
        """
        # --- 代码：重力补偿 ---
        # `data.qfrc_bias` 存储了由重力、科里奥利力等产生的偏置力矩。
        # 对于静止或慢速运动的机器人，它主要就是重力力矩
        step_start = time.time()

        self.env.data.qfrc_applied[:] =  self.env.data.qfrc_bias[:]
        # --- 2. 机器人运动控制 ---
        # 如果轨迹还没播完
        if self.traj_index < self.total_frames:

            self.runMPC()
            # 前向动力学计算
            mujoco.mj_forward(self.env.model, self.env.data)
            
            # 绘制当前目标点 (绿色大球)
            # current_pos = self.cartesian_points[self.traj_index]
            current_pos = self.actual_traj[self.traj_index][:3]
            if self.env.viewer.user_scn.ngeom <= self.env.viewer.user_scn.maxgeom:
                mujoco.mjv_initGeom(
                    self.env.viewer.user_scn.geoms[self.env.viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.001, 0, 0],    # 1cm 绿球
                    pos=current_pos,
                    mat=np.eye(3).flatten(),
                    rgba=[0, 1, 0, 1]     # 不透明绿
                )
                self.env.viewer.user_scn.ngeom += 1
            
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
            print("回归 Home...")
            # 设置回归过程中的关节角度
            self.env.data.qpos[:self.num_joints] = self.return_traj[self.return_index]
            mujoco.mj_forward(self.env.model, self.env.data)
            # 这里不再绘制绿球，因为已经在“回家”路上了
            self.return_index += 1
            s_next , _, _, _, _ = self.env.step(self.home_ctrl) # (8,)
        # --- 阶段 4: 全部结束，保持 Home 姿态 ---
        else:
            if self.home_qpos is not None:
                self.env.data.qpos[:self.num_joints] = self.home_qpos[:self.num_joints]
            else:
                # 如果没有 home，就停在轨迹终点
                self.env.data.qpos[:self.num_joints] = self.joint_angle_traj[-1]
                
            mujoco.mj_forward(self.env.model, self.env.data)
            s_next , _, _, _, _ = self.env.step(self.home_ctrl) # (8,)

        # --- 3. 通信器逻辑 (保留你的原始逻辑) ---
        sim_joint_rad = self.env.data.qpos[:6].copy() 
        # 将弧度转换为角度
        sim_joint_deg = [math.degrees(q) for q in sim_joint_rad]        
        q_real_target_deg = sim_to_real(sim_joint_deg, joint_offsets)
        self.communicator.send_data(q_real_target_deg)

        time_until_next_step = 0.02 - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        # time.sleep(0.01)  # 控制发送频
    
    def runMPC(self):
        # --- 2. MPC 控制计算 ---
        ref_traj_segment = self.state_all_ref[self.traj_index + 1 : self.traj_index + self.H + 1]
        if self.mpc_controller.state_full:
            lifted_ref = np.zeros((self.H, self.mpc_controller.Nkoopman))
            for i in range(len(ref_traj_segment)):
                lifted_ref[i] = self.mpc_controller.Psi_o(torch.DoubleTensor(ref_traj_segment[i]).unsqueeze(0)).flatten() 
            ref_flat = lifted_ref.flatten().reshape(-1, 1)
        else:  
            ref_flat = ref_traj_segment.flatten().reshape(-1, 1)

        z0 = self.mpc_controller.Psi_o(self.state_tensor) # (32,1)
        
        if self.mpc_controller.args.MPC_type == 'mpc':
            p = np.vstack([ref_flat, z0.reshape(-1, 1)])
        if self.mpc_controller.args.MPC_type == 'delta_mpc':
            p = np.vstack([ref_flat, z0.reshape(-1, 1), self.mpc_controller.u_prev.reshape(-1, 1)])

        # b. 计算控制输入
        u, a = self.mpc_controller.get_control(p)
        self.mpc_controller.u_prev = u # 保存当前控制以备下次使用

        # --- 3. 应用控制并步进仿真 ---
        s_next , _, _, _, _ = self.env.step(a) # (8,)
        self.state_tensor = torch.DoubleTensor(s_next).unsqueeze(0) # (1,8)
        self.actual_traj.append(self.state_tensor.detach().cpu().numpy().reshape(-1))
 
    def is_running(self):
        return self.env.viewer.is_running()
    
    def run_loop(self):
        self.runBefore()
        while self.is_running():
            self.runFunc()

if __name__ == "__main__":
    # --- 0. 基本配置 ---
    # 示例参数，请替换为您自己的模型信息
    args = Args()
    args.device = "cpu"
    MODEL_XML_PATH = args.xml_path
    EE_SITE_NAME = 'gripperframe' # 你的XML里定义的夹爪中心的 <site>
    NUM_JOINTS = args.u_dim # 你的机器人关节数量

    # 创建生成器实例
    traj_generator = CartesianTrajectoryGenerator(
        model_path=MODEL_XML_PATH,
        ee_site_name=EE_SITE_NAME,
        num_joints=NUM_JOINTS,
        idx=1, 
        time_horizon = 60, 
        time_steps_per_sec = 5
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
    model = init_model(args)
    model.double()
    load_model_path = args.output_dir + "/best_model.pt"
    model.load_state_dict(torch.load(load_model_path, map_location=torch.device('cpu')))
    MPC_Controller = MPCController(model, args)

    try:
        # 实例化播放器
        test = Test(
            path=MODEL_XML_PATH,
            Controller = MPC_Controller,
            communicator = zmq_communicator, # 传入你的通信器
            cartesian_points=cartesian_points,
            joint_angle_traj=joint_angle_traj,
            num_joints = NUM_JOINTS,
            draw_num = 300
        )
        # 启动
        test.run_loop()

    except KeyboardInterrupt:
        print("仿真程序被用户中断")
    finally:
        # 清理通信资源
        zmq_communicator.cleanup()

