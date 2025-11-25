import os
import time
import mujoco_viewer
import numpy as np
import math
import zmq
import json
import sys # 导入 sys 模块以实现初始化失败时退出

# ==================== 配置常量 ====================
# 机器人关节名称（顺序必须与偏移量数组和MuJoCo模型一致）
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper"
]

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


class ZMQCommunicator:
    """ZMQ通信封装类, 负责处理与外部的通信"""
    
    # 1. 修改默认地址为 TCP，并重命名参数为更通用的 'address'
    def __init__(self, address="tcp://127.0.0.1:5555"):
        """初始化ZMQ通信
        
        Args:
            address: ZMQ绑定的地址 (例如: "tcp://127.0.0.1:5555" 或 "ipc:///tmp/...")
        """
        self.address = address
        self.context = None
        self.socket = None
        self._initialize()
    
    def _initialize(self):
        """初始化ZMQ上下文和socket"""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            # 使用 self.address 进行绑定
            self.socket.bind(self.address)
            print(f"ZMQ发布者启动, 绑定到:{self.address}")
        except zmq.ZMQError as e:
            print(f"ZMQ初始化失败: {e}")
            self.cleanup()
            # 2. 初始化失败时直接退出程序，防止后续错误
            print("程序因ZMQ初始化失败而终止。")
            sys.exit(1)
    
    # 3. 修正类型提示，因为我们发送的是列表(list)
    def send_data(self, data: list):
        """发送数据
        
        Args:
            data: 要发送的数据（例如关节角度列表）
        """
        if not self.socket:
            print("ZMQ socket未初始化, 无法发送数据")
            return
            
        try:
            # 转换为JSON字符串并发布
            json_data = json.dumps(data)
            self.socket.send_string(json_data)
        except zmq.ZMQError as e:
            print(f"发送数据失败: {e}")
    
    def cleanup(self):
        """清理ZMQ资源"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        print("ZMQ资源已释放")


class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path, communicator):
        super().__init__(path, 1.5, azimuth=135, elevation=-30)
        self.path = path
        self.communicator = communicator

    def runBefore(self):
        pass
    
    def runFunc(self):
        # 获取仿真中的前6个关节的角度（弧度）
        sim_joint_rad = self.data.qpos[:6].copy()
        # 将弧度转换为角度
        sim_joint_deg = [math.degrees(q) for q in sim_joint_rad]
        # b. 将 MuJoCo 角度转换为真实机器人指令
        q_real_target_deg = sim_to_real(sim_joint_deg, joint_offsets)

        # c. 构建发送给真实机器人的动作字典
        # action_to_send = {
        #     f"{name}.pos": angle for name, angle in zip(JOINT_NAMES, q_real_target_deg)
        # }
        
        # 使用通信器发送数据
        self.communicator.send_data(q_real_target_deg)
        # time.sleep(0.01)  # 控制发送频率


if __name__ == "__main__":
    # 4. 显式创建使用 TCP 地址的通信器实例
    zmq_communicator = ZMQCommunicator("tcp://127.0.0.1:5555")
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    xml_path = os.path.join(project_root, "model", "SO101", "scene_with_table.xml")
    try:
        # 将通信器实例传入Test类
        test = Test(xml_path, zmq_communicator)
        test.run_loop()
    except KeyboardInterrupt:
        print("仿真程序被用户中断")
    finally:
        # 清理通信资源
        zmq_communicator.cleanup()