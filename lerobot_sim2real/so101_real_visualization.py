import zmq
import json
import os
import time
import math
import logging
from typing import Any
import numpy as np
import rerun as rr

# --- lerobot 框架的导入 ---
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# ==================== 配置常量 ====================
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper"
]
REAL_ROBOT_PORT = "COM24"
ZMQ_IP = "127.0.0.1"
ZMQ_PORT = "5555"

# ==================== Rerun 可视化函数 (来自脚本1) ====================
def _init_rerun(session_name: str = "lerobot_realtime_control") -> None:
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    rr.spawn(memory_limit=memory_limit)

def log_rerun_data(observation: dict[str | Any], action: dict[str | Any]):
    # 遍历观测数据
    for obs, val in observation.items():
        entity_path_prefix = f"observation/{obs}"
        if isinstance(val, float):
            rr.log(entity_path_prefix, rr.Scalar(val))
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                for i, v in enumerate(val):
                    rr.log(f"{entity_path_prefix}/{i}", rr.Scalar(float(v)))
            else:
                rr.log(entity_path_prefix, rr.Image(val), static=True)

    # 遍历动作数据
    for act, val in action.items():
        entity_path_prefix = f"action/{act}"
        if isinstance(val, float):
            rr.log(entity_path_prefix, rr.Scalar(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"{entity_path_prefix}/{i}", rr.Scalar(float(v)))

# ==================== 主函数 (融合了两个脚本的优点) ====================
def main():
    # --- 1. 初始化 ---
    logging.basicConfig(level=logging.WARNING)
    _init_rerun()

    print(f"1. 初始化机器人 (端口: {REAL_ROBOT_PORT}) 使用 lerobot 框架...")
    robot_config = SO101FollowerConfig(port=REAL_ROBOT_PORT, id="so101_follower", use_degrees=True)
    robot = SO101Follower(config=robot_config)

    print(f"2. 初始化 ZMQ 订阅者 (连接到: tcp://{ZMQ_IP}:{ZMQ_PORT})...")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{ZMQ_IP}:{ZMQ_PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    # 使用 Poller 来实现非阻塞的、可中断的等待
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    try:
        # --- 2. 连接硬件 ---
        print("3. 连接机器人硬件...")
        robot.connect()
        print("   机器人连接成功！")

        print("4. 进入实时控制循环。现在可以按 Ctrl+C 随时中断。")
        
        # --- 3. 实时控制主循环 (来自脚本2的健壮结构) ---
        while True:
            # 使用带超时的 poller 等待消息，这样循环就不会卡死
            socks = dict(poller.poll(timeout=100))

            # 只有在我们的 socket 收到消息时才执行后续代码
            if socket in socks and socks[socket] == zmq.POLLIN:
                data_string = socket.recv_string().strip()
                if not data_string:
                    continue

                # --- 核心控制逻辑 (来自脚本1) ---
                # 假设从ZMQ收到的是弧度列表
                joint_pos_deg_list = json.loads(data_string)
                
                action_to_send = {
                    f"{name}.pos": angle for name, angle in zip(JOINT_NAMES, joint_pos_deg_list)
                }

                # b. 发送动作到机器人
                robot.send_action(action_to_send)
                
                # c. 获取机器人状态的观察值
                observation = robot.get_observation()

                # d. 将观察值和动作都记录到 Rerun
                log_rerun_data(observation, action_to_send)

    except KeyboardInterrupt:
        print("程序被用户中断。")
    except Exception as e:
        print(f"在主循环中发生错误: {e}")
    finally:
        # --- 4. 安全清理 (来自脚本2的健壮结构) ---
        print("正在安全关闭...")
        
        if not socket.closed:
            socket.close()
        if not context.closed:
            context.term()
        print("   ZMQ 已关闭。")
        
        if robot and robot.is_connected:
            print("   断开机器人连接...")
            robot.disconnect()
            print("   机器人已安全断开。")
        
        print("所有资源已安全释放。")

if __name__ == "__main__":
    main()