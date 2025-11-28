import logging
import time

# 导入 lerobot 的基础类
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# ==================== 1. 定义增强的机器人控制类 ====================

class ControllableSO101Robot(SO101Follower):
    """
    一个扩展了 SO101Follower 的机器人控制类。
    它提供了一个统一的方法来控制电机的三种扭矩模式：
    'on', 'off', 'damping'。
    """
    
    def set_damping_mode(self):
        """
        [内部方法] 使用同步写入将所有电机设置为阻尼模式 (Torque Enable = 2)。
        """
        # 这个方法是实现新功能的核心，但用户通常不需要直接调用它。
        damping_command = {name: 2 for name in self.joint_names}
        self.bus.sync_write("Torque_Enable", damping_command)

    def set_torque_mode(self, mode: str):
        """
        设置所有关节的扭矩模式。

        Args:
            mode (str): 目标模式。可选项: 'on', 'off', 'damping'。
        """
        print(f"--- 正在设置扭矩模式为: '{mode}' ---")
        
        if mode == "on":
            # 调用父类已有的 enable_torque 方法 (对应模式 1)
            self.enable_torque()
        elif mode == "off":
            # 调用父类已有的 disable_torque 方法 (对应模式 0)
            self.disable_torque()
        elif mode == "damping":
            # 调用我们自己添加的 set_damping_mode 方法 (对应模式 2)
            self.set_damping_mode()
        else:
            # 错误处理，防止输入无效模式
            raise ValueError(f"无效的扭矩模式 '{mode}'。有效选项为: 'on', 'off', 'damping'。")
        
        print("设置完成！")


# ==================== 2. 编写交互式演示脚本 ====================

def main():
    # 配置
    REAL_ROBOT_PORT = "COM24"  # !!! 修改为你的串口 !!!
    logging.basicConfig(level=logging.WARNING)
    
    # 实例化我们新的、增强的机器人对象
    robot_config = SO101FollowerConfig(port=REAL_ROBOT_PORT, id="so101_follower", use_degrees=True)
    robot = ControllableSO101Robot(config=robot_config)

    try:
        # 连接机器人
        robot.connect()
        print("机器人已连接。默认模式为 'on' (扭矩开启)。")
        input("请尝试轻轻推动机械臂，感受其锁定状态。按 Enter 继续...")
        
        # --- 演示 1: 切换到阻尼模式 ---
        robot.set_torque_mode("damping")
        print("现在你可以手动移动机械臂进行示教。")
        input("请移动机械臂，感受平滑的阻力。按 Enter 切换到下一模式...")

        # --- 演示 2: 切换到扭矩关闭模式 ---
        robot.set_torque_mode("off")
        print("现在机械臂完全无力，可以自由移动（小心重力！）。")
        input("请再次移动机械臂，感受无阻力状态。按 Enter 恢复锁定...")

        # --- 演示 3: 恢复扭矩开启模式 ---
        robot.set_torque_mode("on")
        print("机械臂已恢复锁定状态。")
        input("请再次尝试推动机械臂，确认其已锁定。按 Enter 结束程序...")

    except ValueError as e:
        print(f"输入错误: {e}")
    except Exception as e:
        print(f"在演示过程中发生错误: {e}")
    finally:
        # 确保无论发生什么，都安全断开连接
        if robot and robot.is_connected:
            print("程序结束，正在断开连接...")
            # robot.disconnect() 会自动将扭矩设置为 'on'
            robot.disconnect()
            print("机器人已安全断开。")

if __name__ == "__main__":
    main()