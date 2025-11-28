# 这段脚本的核心功能是读取并显示机械臂上所有6个电机当前的角度位置。

import time
# from lerobot_sim2real.real_robot import create_real_robot
# --- lerobot 框架的导入 ---
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


# ==================== 配置 ====================
# 机器人串口
REAL_ROBOT_PORT = "COM24" # !!! 修改为你的串口 !!!

def main():
    # --- 1. 创建机器人实例 ---

    print("配置机器人...")
    robot_config = SO101FollowerConfig(port=REAL_ROBOT_PORT, id="so101_follower", use_degrees=True)
    robot = SO101Follower(config=robot_config)

    try:
        # --- 2. 连接到机器人 ---
        # 这个方法会处理所有电机的初始化和配置
        robot.connect()
        print("机器人连接成功！")

        # 等待一小会儿，确保电机状态稳定可读
        time.sleep(0.1) 

        # --- 3. 一次性读取所有关节的位置 ---
        # 这就是替代你原来 for 循环的核心方法
        print("正在读取所有关节的位置...")
        observation = robot.get_observation()
        
        # --- 4. 打印结果 ---
        # observation 是一个字典，我们可以优雅地打印它
        print("当前机器人状态:")
        for key, value in observation.items():
            if key.endswith(".pos"): # 只打印位置信息
                joint_name = key.removesuffix(".pos")
                # 格式化输出，使其更美观
                print(f"  - {joint_name:<15}: {value:.2f}°")

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # --- 5. 安全断开连接 ---
        if robot and robot.is_connected:
            robot.disconnect()
            print("机器人已断开连接。")

if __name__ == "__main__":
    main()