# This code builds upon following:
# https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_arm_ik.py
# https://github.com/ccrpRepo/mocap_retarget/blob/master/src/mocap/src/robot_ik.py

import casadi          
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin               
import os

class Kinematics:
    def __init__(self, ee_frame) -> None:
        # ee_frame 是末端执行器（End-Effector）的名称，比如 "left_finger"
        self.frame_name = ee_frame

    def buildFromMJCF(self, mcjf_file):
        # 从MuJoCo的MJCF文件加载机器人模型
        self.arm = pin.RobotWrapper.BuildFromMJCF(mcjf_file)
        # 加载后立即创建优化求解器
        self.createSolver()

    def buildFromURDF(self, urdf_file):
        # 从URDF文件加载机器人模型
        self.arm = pin.RobotWrapper.BuildFromURDF(urdf_file)
        self.createSolver()

    def createSolver(self):
        self.model = self.arm.model
        self.data = self.arm.data

        # 1. 创建CasADi的符号模型 (Symbolic Model)
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        # 2. 定义符号变量
        self.cq = casadi.SX.sym("q", self.model.nq, 1)  # 符号变量：关节角度 q
        self.cTf = casadi.SX.sym("tf", 4, 4)            # 符号变量：目标位姿矩阵 T
        # 3. 定义误差函数 (Error Functions)
        # 运行一次符号形式的正向运动学，这样 cdata 中就包含了用符号变量cq表示的末端位姿
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        
        # 获取末端执行器的ID
        self.ee_id = self.model.getFrameId(self.frame_name) if self.frame_name=="gripperframe" else self.model.getFrameId(self.frame_name, pin.FrameType.BODY)
        # a. 位置误差函数：末端位置 - 目标位置
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.ee_id].translation - self.cTf[:3,3]
                )
            ],
        )
        # b. 姿态误差函数：使用李代数 log3 计算两个旋转矩阵的差异
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.ee_id].rotation @ self.cTf[:3,:3].T)
                )
            ],
        )

        # 4. 定义优化问题 (Optimization Problem)
        self.opti = casadi.Opti()
        # a. 定义决策变量 (Decision Variable)
        self.var_q = self.opti.variable(self.model.nq)
        # b. 定义参数 (Parameters)
        self.var_q_last = self.opti.parameter(self.model.nq)   # 上一时刻的关节角度 (用于平滑)
        self.param_tf = self.opti.parameter(4, 4)   # 目标位姿 T
        # c. 定义代价函数 (Cost Function)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf)) # 位置误差的平方和
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf))         # 姿态误差的平方和
        self.regularization_cost = casadi.sumsqr(self.var_q)    # 让关节角度尽量小 (几乎没用，权重为0)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)  # 平滑代价：新旧角度差的平方和


        # d. 设置约束 (Constraints)
        # 关节角度必须在模型的物理限制范围内
        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit,
            self.var_q,
            self.model.upperPositionLimit)
        )
        # e. 设置总的最小化目标
        # 这是一个带权重的多目标优化
        self.opti.minimize(10.0 * self.translational_cost + 0.1*self.rotation_cost + 0.0 * self.regularization_cost + 0.1 * self.smooth_cost)

        # 5. 配置求解器 (IPOPT)
        # IPOPT 是一个非常强大的开源非线性优化求解器
        opts = {
            'ipopt':{
                'print_level': 0,
                'max_iter': 500,
                'tol': 1e-4,
                # 'hessian_approximation':"limited-memory"
            },
            'print_time':False,# print or not
            'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.model.nq)
      
    def ik(self, T , current_arm_motor_q = None, current_arm_motor_dq = None):
        # 1. 设置初始猜测值和参数
        if current_arm_motor_q is not None:
            self.init_data = current_arm_motor_q
        # 为优化器提供一个好的初始猜测值（通常是当前的关节角度），这能大大加快收敛速度
        self.opti.set_initial(self.var_q, self.init_data)

        # 设置本次求解的具体参数值
        self.opti.set_value(self.param_tf, T)                 # 本次的目标位姿
        self.opti.set_value(self.var_q_last, self.init_data)  # 上一时刻的角度（用于平滑）

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            # self.smooth_filter.add_data(sol_q)
            # sol_q = self.smooth_filter.filtered_data

            if current_arm_motor_dq is not None:
                v = current_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.model, self.data, sol_q, v, np.zeros(self.model.nv))
            sol_tauff = np.concatenate([sol_tauff, np.zeros(self.model.nq - sol_tauff.shape[0])], axis=0)
            
            info = {"sol_tauff": sol_tauff, "success": True}

            dof = np.zeros(self.model.nq)
            dof[:len(sol_q)] = sol_q
            return dof, info
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            # self.smooth_filter.add_data(sol_q)
            # sol_q = self.smooth_filter.filtered_data

            if current_arm_motor_dq is not None:
                v = current_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.model, self.data, sol_q, v, np.zeros(self.model.nv))
            # import ipdb; ipdb.set_trace()
            sol_tauff = np.concatenate([sol_tauff, np.zeros(self.model.nq - sol_tauff.shape[0])], axis=0)

            print(f"sol_q:{sol_q} \nmotorstate: \n{current_arm_motor_q} \nright_pose: \n{T}")

            info = {"sol_tauff": sol_tauff * 0.0, "success": False}

            dof = np.zeros(self.model.nq)
            # dof[:len(sol_q)] = current_arm_motor_q
            dof[:len(sol_q)] = self.init_data
            
            raise e

if __name__ == "__main__":
    
    arm = Kinematics("gripperframe") 
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    MODEL_XML_PATH = os.path.join(project_root, "model", "SO101", "so101_new_calib.xml")
    arm.buildFromMJCF(MODEL_XML_PATH)
    # arm.buildFromURDF("model/SO101/so101_new_calib.urdf")
    theta = np.pi
    tf = np.array([
            [1, 0, 0, 0.1],
            [0, np.cos(theta), -np.sin(theta), 0.2],
            [0, np.sin(theta), np.cos(theta), 0.3],
        ])
    tf = np.vstack((tf, [0, 0, 0, 1]))
    dof, info = arm.ik(tf)
    print(f"DoF: {dof}, Info: {info}")


# 常见 FrameType 包括：

# 类型	含义
# pin.FrameType.JOINT	关节 frame
# pin.FrameType.BODY / FIXED_JOINT	刚体或固定连接 frame
# pin.FrameType.SITE	<site> 定义的辅助点（最常用作 end-effector）
# pin.FrameType.OP_FRAME	自定义操作 frame（较少直接使用）