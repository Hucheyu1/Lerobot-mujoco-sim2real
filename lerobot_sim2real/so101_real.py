import zmq
import json
import time
from real_robot import create_real_robot

# ==================== 配置常量 ====================
# ... (这部分保持不变)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
ZMQ_IP = "127.0.0.1"
ZMQ_PORT = "5555"
REAL_ROBOT_PORT = "COM24"

def main():
    # ==================== 阶段 1 & 2 (几乎不变) ====================
    print("1. 配置机器人...")
    robot = create_real_robot("so101")

    print(f"2. 初始化ZMQ订阅者, 连接到 tcp://{ZMQ_IP}:{ZMQ_PORT}")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{ZMQ_IP}:{ZMQ_PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    # ==================== 新增: 初始化 Poller ====================
    # Poller 是ZMQ中用于处理非阻塞IO的核心工具
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)  # 告诉poller监控socket上的“可读”事件

    try:
        # ==================== 阶段 3: 连接硬件 (不变) ====================
        print("3. 连接机器人硬件...")
        robot.connect()
        print("   机器人连接成功！")

        # ==================== 阶段 4: 实时控制主循环 (核心修改) ====================
        print("4. 进入实时控制循环。现在可以按 Ctrl+C 随时中断。")
        while True:
            # a. 使用 poller 带超时地等待消息
            #    等待最多100毫秒。如果在100毫秒内有消息，它会立即返回。
            #    如果没有消息，它会等待100毫秒然后返回一个空字典。
            #    这个超时机制是让 Ctrl+C 生效的关键！
            socks = dict(poller.poll(timeout=100))

            # b. 检查我们的socket是否在返回的字典中，且事件是“可读”
            if socket in socks and socks[socket] == zmq.POLLIN:
                # c. 只有在确定有消息时才接收，这个调用不会阻塞
                data_string = socket.recv_string().strip()
                
                # --- 后续逻辑与原来完全相同，但现在被包裹在if块内 ---
                if not data_string:
                    continue

                joint_pos_deg_list = json.loads(data_string)

                if len(joint_pos_deg_list) != len(JOINT_NAMES):
                    print(f"警告：收到的数据长度 ({len(joint_pos_deg_list)}) 与期望的关节数量 ({len(JOINT_NAMES)}) 不匹配。")
                    continue
                    
                action_to_send = {
                    f"{name}.pos": angle for name, angle in zip(JOINT_NAMES, joint_pos_deg_list)
                }
                
                robot.send_action(action_to_send)
                # print(f"已发送动作: {action_to_send}")

            # 如果没有消息，循环会自然地结束本次迭代，然后开始下一次
            # 这就给了Python解释器处理 KeyboardInterrupt 的机会

    except KeyboardInterrupt:
        print("程序被用户中断。")
    except Exception as e:
        print(f"在主循环中发生错误: {e}")
    finally:
        # ==================== 阶段 5: 安全清理资源 (不变) ====================
        print("正在安全关闭...")
        
        # a. 关闭 ZMQ
        if not socket.closed:
            socket.close()
        if not context.closed:
            context.term()
        print("   ZMQ 已关闭。")
        
        # b. 断开机器人连接
        if robot and robot.is_connected:
            print("   断开机器人连接...")
            robot.disconnect()
            print("   机器人已安全断开。")

if __name__ == "__main__":
    main()