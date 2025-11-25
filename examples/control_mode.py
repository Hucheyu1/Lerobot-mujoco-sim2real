import time
import logging
from typing import Any

# 假设你提供的原始类保存在 lerobot/robots/so101_follower.py 中
# 或者你可以把下面的 SO101Follower 类定义直接贴在这个文件上面
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.motors.feetech import OperatingMode

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridSO101Follower(SO101Follower):
    """
    SO-101 的增强版。
    既支持原来的 '位置控制' (send_action 传入 .pos),
    也支持切换到 '速度控制' (send_action 传入 .vel)。
    """

    def __init__(self, config: SO101FollowerConfig):
        super().__init__(config)
        # 记录当前模式，默认为位置模式
        self._current_mode = "position" 

    def set_control_mode(self, mode: str):
        """
        高层接口：切换控制模式。
        :param mode: 'position' (位置模式) 或 'velocity' (速度模式)
        """
        if mode not in ["position", "velocity"]:
            raise ValueError("Mode must be 'position' or 'velocity'")

        if mode == self._current_mode:
            logger.info(f"Already in {mode} mode.")
            return

        logger.info(f"Switching to {mode} mode...")

        # 1. 必须先关闭扭矩，否则无法修改 Operating_Mode
        self.bus.disable_torque()
        time.sleep(0.1) # 安全缓冲

        # 2. 确定 Feetech 的模式值
        # 0: Position Control, 1: Speed Closed-Loop Control
        # 注意：如果 lerobot 库里没有 OperatingMode.VELOCITY，直接用数字 1
        target_val = 0 if mode == "position" else 1 

        # 3. 批量写入模式
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, target_val)

        # 4. 重新开启扭矩
        self.bus.enable_torque()
        
        # 5. 更新内部状态
        self._current_mode = mode
        logger.info(f"Switched to {mode} mode successfully.")

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        重写父类的 send_action。
        根据当前模式，自动决定是发送位置指令还是速度指令。
        """
        if not self.is_connected:
            raise Exception("Robot not connected")

        # --- 情况 A: 当前是位置模式 ---
        if self._current_mode == "position":
            # 检查 action 里是否有 .pos 数据，如果有，直接调用父类原来的逻辑
            # 父类逻辑处理了安全限位等操作，非常完善
            if any(k.endswith(".pos") for k in action):
                return super().send_action(action)
            else:
                logger.warning("In Position Mode but received no .pos actions.")
                return {}

        # --- 情况 B: 当前是速度模式 ---
        elif self._current_mode == "velocity":
            # 提取 .vel 后缀的数据
            goal_vel = {
                key.removesuffix(".vel"): val 
                for key, val in action.items() 
                if key.endswith(".vel")
            }

            if not goal_vel:
                return {}

            # 直接写入 Goal_Speed 寄存器
            # 在模式 1 下，Goal_Speed 就是目标速度
            self.bus.sync_write("Goal_Speed", goal_vel)
            
            # 返回发送的数据用于记录
            return {f"{motor}.vel": val for motor, val in goal_vel.items()}

        return {}

    def stop(self):
        """紧急停止：发送 0 速度"""
        if self._current_mode == "velocity":
            zeros = {f"{name}.vel": 0.0 for name in self.bus.motors}
            self.send_action(zeros)
        else:
            # 位置模式下，读取当前位置并作为目标位置发送（原地锁住）
            current = self.get_observation()
            hold_pos = {k: v for k, v in current.items() if k.endswith(".pos")}
            self.send_action(hold_pos)


# ==========================================
# 演示代码：如何使用这个增强类
# ==========================================

def main():
    # 1. 初始化配置 (替换为你的实际端口)
    config = SO101FollowerConfig(port="COM24", id="so101_follower", use_degrees=True)
    
    # 2. 使用我们要的增强类
    robot = HybridSO101Follower(config)

    try:
        print("1. 连接机器人...")
        robot.connect()
        
        # ================= 位置控制演示 =================
        print(">>> 当前模式: 位置控制 (Position)")
        print("   移动到一个测试位置...")
        
        target_pos = {
            "shoulder_pan.pos": 0,
            "shoulder_lift.pos": -90,
            "elbow_flex.pos": 90,
            "wrist_flex.pos": 0,
            "wrist_roll.pos": 0,
            "gripper.pos": 50
        }
        robot.send_action(target_pos)
        time.sleep(3) # 等待到达

        # ================= 切换模式 =================
        input("按 Enter 切换到 [速度控制] 模式...")
        robot.set_control_mode("velocity")

        # ================= 速度控制演示 =================
        print(">>> 当前模式: 速度控制 (Velocity)")
        print("   注意：这里是让电机转动，而不是去某个坐标")

        # 动作 1: 手腕旋转
        print("   动作: 手腕旋转 (30度/秒)...")
        vel_action = {
            "shoulder_pan.vel": 0,
            "shoulder_lift.vel": 0,
            "elbow_flex.vel": 0,
            "wrist_flex.vel": 0,
            "wrist_roll.vel": 30.0, # 速度值
            "gripper.vel": 0
        }
        robot.send_action(vel_action)
        time.sleep(2.0) # 转2秒

        # 动作 2: 停止
        print("   停止！")
        robot.stop()
        time.sleep(1.0)

        # 动作 3: 反转
        print("   动作: 手腕反转 (-30度/秒)...")
        vel_action["wrist_roll.vel"] = -30.0
        robot.send_action(vel_action)
        time.sleep(2.0)

        # 停止
        robot.stop()

        # ================= 切回位置 =================
        input("按 Enter 切回 [位置控制] 模式并退出...")
        robot.set_control_mode("position")
        print(">>> 已切回位置模式，机器人将保持在当前姿态。")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if robot.is_connected:
            # 断开前先停止，比较安全
            robot.stop()
            robot.disconnect()
            print("机器人已断开。")

if __name__ == "__main__":
    main()