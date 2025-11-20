import time
import logging

# 配置日志记录，以便看到lerobot库的输出信息
logging.basicConfig(level=logging.INFO)

from real_robot import create_real_robot

def main():
    # ==================== 阶段 1: 配置与实例化 ====================
    print("1. 配置机器人...")
    robot = create_real_robot("so101")

    try:
        # ==================== 阶段 2: 连接硬件 ====================
        print("2. 连接机器人...")
        robot.connect()
        print("   机器人连接成功！")

        # ==================== 阶段 3: 主控制循环 ====================
        print("3. 开始执行控制序列...")

        # 定义一个“初始/归位”姿态
        home_position = {
            "shoulder_pan.pos": -2.37,
            "shoulder_lift.pos": -96.62,
            "elbow_flex.pos": 97.10,
            "wrist_flex.pos": 44.88,
            "wrist_roll.pos": 1.05,
            "gripper.pos": 31.97,  # 夹爪闭合
        }
        
        # 定义一个“工作”姿态
        work_position = {
            "shoulder_pan.pos": 0,
            "shoulder_lift.pos": 0,
            "elbow_flex.pos": 0,
            "wrist_flex.pos": 0,
            "wrist_roll.pos": 0,
            "gripper.pos": 50,   # 夹爪张开
        }

        # --- 序列 1: 移动到初始位置 ---
        print(" 动作1: 移动到初始位置...")
        robot.send_action(home_position)
        time.sleep(3) # 等待3秒让机器人完成移动

        # --- 序列 2: 移动到工作位置 ---
        print("动作2: 移动到工作位置...")
        robot.send_action(work_position)
        time.sleep(3) # 等待3秒

        # --- 序列 3: 返回初始位置 ---
        print("动作3: 返回初始位置...")
        robot.send_action(home_position)
        time.sleep(3)

        print("控制序列执行完毕！")

    except Exception as e:
        print(f"在控制过程中发生错误: {e}")
    finally:
        # ==================== 阶段 4: 断开连接 ====================
        # 无论成功还是失败，都确保断开连接
        if robot.is_connected:
            print("4. 断开机器人连接...")
            robot.disconnect()
            print("   机器人已安全断开。")

if __name__ == "__main__":
    main()