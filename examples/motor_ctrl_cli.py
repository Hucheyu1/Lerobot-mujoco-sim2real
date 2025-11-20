import argparse
import logging
from typing import Dict
import sys
# 导入我们之前设计的增强版机器人控制器
from joint_mode import ControllableSO101Robot
from lerobot.robots.so101_follower import SO101FollowerConfig

# ==================== 配置 ====================
# 硬件和关节映射
REAL_ROBOT_PORT = "COM24"  # !!! 修改为你的串口 !!!
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper"
]

# ==================== 辅助函数 ====================
def get_joint_name_from_id(joint_id: int) -> str:
    """根据1-6的ID获取关节名称"""
    if not 1 <= joint_id <= len(JOINT_NAMES):
        raise ValueError(f"关节ID必须在1到{len(JOINT_NAMES)}之间")
    return JOINT_NAMES[joint_id - 1]

def print_joint_diagnostics(robot: ControllableSO101Robot, joint_name: str):
    """打印指定关节的详细诊断信息"""
    print(f"--- 正在获取关节 '{joint_name}' 的诊断信息 ---")
    registers_to_read = [
        "Present_Position", "Present_Speed", "Present_Load",
        "Goal_Position", "Goal_Speed", "Torque_Limit",
        "P_Gain", "I_Gain", "D_Gain",
        "Hardware_Error_Status",
    ]
    # 使用底层bus对象的 read 方法一次性读取多个寄存器
    try:
        data: Dict[str, int] = robot.bus.read(registers_to_read, {joint_name})
        for reg, value in data[joint_name].items():
            print(f"  - {reg:<25}: {value}")
    except Exception as e:
        print(f"读取诊断信息失败: {e}")

# ==================== 主程序 ====================
def main():
    # --- 1. 设置命令行解析器 (argparse) ---
    parser = argparse.ArgumentParser(
        description="一个基于 lerobot 框架的机械臂控制和调试命令行工具。",
        formatter_class=argparse.RawTextHelpFormatter  # 保持帮助信息格式
    )
    subparsers = parser.add_subparsers(dest="operation", help="可执行的操作")

    # --- 定义所有子命令 ---
    # set-pos
    p_setpos = subparsers.add_parser("set-pos", help="设置单个关节的目标位置 (角度)")
    p_setpos.add_argument("joint_id", type=int, help="关节ID (1-6)")
    p_setpos.add_argument("angle", type=float, help="目标角度 (度)")

    # get-pos
    p_getpos = subparsers.add_parser("get-pos", help="获取单个关节的当前位置 (角度)")
    p_getpos.add_argument("joint_id", type=int, help="关节ID (1-6)")

    # set-torque (操作整个机器人)
    p_settorque = subparsers.add_parser("set-torque", help="设置所有关节的扭矩模式")
    p_settorque.add_argument("mode", choices=["on", "off", "damping"], help="模式: on, off, damping")

    # get-diagnostics (替代 get_config)
    p_getdiag = subparsers.add_parser("get-diagnostics", help="获取单个关节的详细诊断信息")
    p_getdiag.add_argument("joint_id", type=int, help="关节ID (1-6)")
    
    # set-pid (低级调试)
    p_setpid = subparsers.add_parser("set-pid", help="设置单个关节的PID参数 (低级)")
    p_setpid.add_argument("joint_id", type=int, help="关节ID (1-6)")
    p_setpid.add_argument("p", type=int, help="P Gain")
    p_setpid.add_argument("i", type=int, help="I Gain")
    p_setpid.add_argument("d", type=int, help="D Gain")

    # ... 你可以按照这个模式添加更多命令，如 get-speed, set-raw-pos 等 ...
    
    # 解析命令行参数
    args = parser.parse_args()

    if not args.operation:
        parser.print_help()
        sys.exit(1)

    # --- 2. 初始化机器人 ---
    logging.basicConfig(level=logging.WARNING)
    robot_config = SO101FollowerConfig(port=REAL_ROBOT_PORT, use_degrees=True)
    robot = ControllableSO101Robot(config=robot_config)
    
    try:
        print("--- 正在连接机器人 ---")
        robot.connect()

        # --- 3. 执行请求的操作 ---
        if args.operation == "set-pos":
            joint_name = get_joint_name_from_id(args.joint_id)
            action = {f"{joint_name}.pos": args.angle}
            print(f"发送指令: 将关节 '{joint_name}' (ID: {args.joint_id}) 移动到 {args.angle}°")
            robot.send_action(action)

        elif args.operation == "get-pos":
            joint_name = get_joint_name_from_id(args.joint_id)
            obs = robot.get_observation()
            position = obs.get(f"{joint_name}.pos", "N/A")
            print(f"关节 '{joint_name}' (ID: {args.joint_id}) 的当前位置: {position}°")

        elif args.operation == "set-torque":
            robot.set_torque_mode(args.mode)
            print(f"所有关节的扭矩模式已设置为 '{args.mode}'")

        elif args.operation == "get-diagnostics":
            joint_name = get_joint_name_from_id(args.joint_id)
            print_joint_diagnostics(robot, joint_name)
            
        elif args.operation == "set-pid":
            joint_name = get_joint_name_from_id(args.joint_id)
            print(f"设置关节 '{joint_name}' 的PID为 P={args.p}, I={args.i}, D={args.d}")
            # 使用底层 bus 对象进行低级寄存器写入
            robot.bus.write("P_Gain", {joint_name: args.p})
            robot.bus.write("I_Gain", {joint_name: args.i})
            robot.bus.write("D_Gain", {joint_name: args.d})
            print("设置完成。")

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
    finally:
        # --- 4. 确保安全断开 ---
        if robot and robot.is_connected:
            robot.disconnect()
            print("--- 连接已断开 ---")

if __name__ == "__main__":
    # 假设你已经将之前的 ControllableSO101Robot 类保存在了
    # `your_project/controllable_robot.py` 文件中
    
    # sys.path.insert(0, ".") # 将当前目录添加到path，以便导入
    main()