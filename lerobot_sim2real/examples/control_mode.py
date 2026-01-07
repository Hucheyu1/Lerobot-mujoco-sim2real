import logging
import time
from typing import Any

from lerobot.motors.feetech import OperatingMode

# 假设你的环境已经安装了 lerobot 并且路径正确
# 根据你提供的文件结构导入
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

logger = logging.getLogger(__name__)


class HybridSO101Follower(SO101Follower):
    """
    SO-101 增强版驱动类。
    支持在运行时动态切换 '位置控制' 和 '速度控制'。
    """

    def __init__(self, config: SO101FollowerConfig):
        super().__init__(config)
        # 记录当前模式，初始为位置模式 (lerobot 默认)
        self._current_mode = "position"

    def set_control_mode(self, mode: str):
        """
        切换控制模式。
        :param mode: 'position' (位置闭环) 或 'velocity' (速度闭环)
        """
        if mode not in ["position", "velocity"]:
            raise ValueError("Mode must be 'position' or 'velocity'")

        if mode == self._current_mode:
            logger.info(f"Already in {mode} mode.")
            return

        logger.info(f"Switching to {mode} mode...")

        # 1. 必须先关闭扭矩！
        # Feetech 舵机规定：只有在扭矩关闭(Torque Off)状态下才能修改 Operating_Mode
        self.bus.disable_torque()

        # 给一点时间让总线处理
        time.sleep(0.1)

        # 2. 确定 Feetech 的 Operating_Mode 值
        # 查阅 STS3215 手册:
        # 0 = Position Control Mode (位置控制)
        # 1 = Speed Closed-loop Control Mode (速度闭环/轮模式)
        # 注意：这里我们直接使用整数值，以防 OperatingMode枚举类中没有定义 VELOCITY
        target_val = 0 if mode == "position" else 1

        # 3. 批量写入模式寄存器
        # 我们遍历总线上的所有电机进行设置
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, target_val)

        # 4. 重新开启扭矩
        self.bus.enable_torque()

        # 5. 更新内部状态
        self._current_mode = mode
        logger.info(f"Switched to {mode} mode successfully.")

        # 安全措施：如果是切回位置模式，建议同步一下当前位置作为目标，防止跳变
        if mode == "position":
            self.stop()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        重写父类的 send_action 以支持速度指令。
        """
        if not self.is_connected:
            raise Exception("Robot not connected")

        # ================= 位置模式逻辑 =================
        if self._current_mode == "position":
            # 过滤出 .pos 指令
            pos_action = {k: v for k, v in action.items() if k.endswith(".pos")}

            if not pos_action:
                return {}

            # 直接调用父类方法，复用其安全限位逻辑
            return super().send_action(action)

        # ================= 速度模式逻辑 =================
        elif self._current_mode == "velocity":
            # 提取 .vel 后缀的数据
            # key 格式转换: "wrist_roll.vel" -> "wrist_roll"
            goal_vel = {key.removesuffix(".vel"): val for key, val in action.items() if key.endswith(".vel")}

            if not goal_vel:
                return {}

            # 直接写入 Goal_Speed 寄存器
            # STS3215 在模式 1 下，Goal_Speed 寄存器(地址通常是 44) 控制转速
            # 单位通常是：符号代表方向，数值代表速度大小
            # 注意：这里的 normalize=False，意味着我们需要传原生单位或自己处理归一化
            # 如果需要 normalize，需要在 MotorsBus 里配置 normalized_data
            self.bus.sync_write("Goal_Velocity", goal_vel, normalize=False)

            return {f"{motor}.vel": val for motor, val in goal_vel.items()}

        return {}

    def stop(self):
        """
        安全停止函数
        """
        if self._current_mode == "velocity":
            # 速度模式下，发送 0 速度
            zeros = {f"{name}.vel": 0 for name in self.bus.motors}
            self.send_action(zeros)
        else:
            # 位置模式下，读取当前实际位置，并将其设为目标位置（原地保持）
            current_obs = self.get_observation()
            hold_pos = {k: v for k, v in current_obs.items() if k.endswith(".pos")}
            self.send_action(hold_pos)


# ==========================================
# 演示 Demo
# ==========================================
def main():
    # 配置日志输出
    logging.basicConfig(level=logging.INFO)

    # 1. 初始化配置 (请修改为你的实际端口)
    # Windows 示例: "COM3", Linux/Mac 示例: "/dev/ttyUSB0"
    PORT = "COM24"
    config = SO101FollowerConfig(port=PORT, id="so101_follower", use_degrees=True)

    # 2. 实例化增强后的机器人
    robot = HybridSO101Follower(config)

    try:
        print(f"Connecting to robot on {PORT}...")
        robot.connect()

        # -------------------------------------------------
        # 阶段 1: 位置控制测试
        # -------------------------------------------------
        print("\n=== [阶段 1] 位置控制模式 ===")
        print("Moving to Home position...")

        # 定义一个安全的位置 (单位: 度, 因为 use_degrees=True)
        home_pos = {
            "shoulder_pan.pos": -6,
            "shoulder_lift.pos": -98,
            "elbow_flex.pos": 97,
            "wrist_flex.pos": 15,
            "wrist_roll.pos": 0,
            "gripper.pos": 33,
        }
        robot.send_action(home_pos)
        time.sleep(3.0)  # 等待运动完成

        # -------------------------------------------------
        # 阶段 2: 切换到速度模式
        # -------------------------------------------------
        print("\n=== [阶段 2] 切换到速度模式 ===")
        # 此时机械臂会瞬间失去扭矩力维持，然后立刻恢复扭矩
        robot.set_control_mode("velocity")

        # -------------------------------------------------
        # 阶段 3: 速度控制测试 (只转动腕部)
        # -------------------------------------------------
        print("Rotating Wrist Roll joint continuously...")

        # STS3215 速度单位说明：
        # 通常原生单位大概是 steps/sec 或者内部单位。
        # 如果 normalize=False，这里的 200 是原生值。
        # 如果觉得太快或太慢，请调整数值。STS3215 最大速度约数千。
        SPEED_VAL = 200

        # 动作 A: 正向旋转
        vel_cmd = {"wrist_roll.vel": SPEED_VAL}
        # 其他关节默认为 0 (不给指令即不写入，或者需要显式给0以防漂移)
        # 为了安全，最好给其他关节发 0
        for m in robot.bus.motors:
            if m != "wrist_roll":
                vel_cmd[f"{m}.vel"] = 0

        robot.send_action(vel_cmd)
        time.sleep(2.0)  # 旋转 2 秒

        # 动作 B: 停止
        print("Stopping...")
        robot.stop()
        time.sleep(1.0)

        # 动作 C: 反向旋转
        print("Rotating Wrist Roll joint REVERSE...")
        vel_cmd["wrist_roll.vel"] = -SPEED_VAL
        robot.send_action(vel_cmd)
        time.sleep(2.0)

        robot.stop()
        print("Stopped.")

        # -------------------------------------------------
        # 阶段 4: 切回位置模式
        # -------------------------------------------------
        print("\n=== [阶段 3] 切回位置模式 ===")
        robot.set_control_mode("position")

        print("Moving back to Home position...")
        robot.send_action(home_pos)
        time.sleep(2.0)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if robot.is_connected:
            print("Disconnecting...")
            robot.stop()
            robot.disconnect()
            print("Done.")


if __name__ == "__main__":
    main()
