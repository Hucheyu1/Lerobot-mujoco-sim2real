import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple

class SOARM101Env(gym.Env):
    """
    一个专门为Koopman算子训练收集数据的SOARM-101环境。

    状态 x (Observation):
    - 5x 关节角度 (rad)
    - 3x 末端执行器位置 (m)
    
    控制输入 u (Action):
    - 5x 关节目标速度 ([-2, 2])
    """
    
    # 尽管不用于RL，但提供元数据是好习惯
    metadata = {"render_modes": ["human"]}

    def __init__(self, xml_path: str, dt: float = 0.02, render_mode = False):
        """
        Args:
            xml_path (str): MuJoCo模型的XML文件路径。
            dt (float): 环境的控制时间步长 (s)。
            render_mode (Optional[str]): 渲染模式，'human'开启可视化。
        """
        super().__init__()

        # --- MuJoCo 初始化 ---
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"MuJoCo XML 文件未找到: {xml_path}")
        
        # 计算需要执行多少个物理步骤以达到期望的dt
        self.frame_skip = max(1, int(np.round(dt / self.model.opt.timestep)))
        self.dt = self.model.opt.timestep * self.frame_skip
        print(f"环境控制步长(dt): {self.dt:.4f}s (执行 {self.frame_skip} 个物理步骤)")
        
        self.data = mujoco.MjData(self.model)
        
        # --- 关节和末端执行器定义 ---
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        self.joint_ids = [self.model.joint(name).id for name in self.joint_names]
        
        try:
            self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        except ValueError:
            raise ValueError("在模型中未找到名为 'gripper' 的 site。请检查XML文件。")

        # --- 状态与控制输入定义 ---(目前无用，保留强化学习拓展)
        # 控制输入 u (Action): 5个关节的目标速度
        self.udim = 5
        self.max_speed = 0.5 # 设定一个物理上的最大关节速度 (rad/s)
        self.action_space = spaces.Box(low = -self.max_speed, high = self.max_speed, shape=(self.udim,), dtype=np.float32)
        # 状态 x (Observation): [qpos, ee_pos]
        self.xdim = 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.xdim,), dtype=np.float32)

        # --- 渲染相关 ---
        self.render_mode = render_mode
        self.viewer = None
        if self.render_mode :
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _get_state(self) -> np.ndarray:
        """获取当前系统的状态 x。"""
        qpos = self.data.qpos[self.joint_ids].copy()
        # qvel = self.data.qvel[self.joint_ids].copy()
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        
        return np.concatenate([ee_pos, qpos]).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境到一个初始状态。
        
        Args:
            seed: 随机种子。
            options (dict, optional): 可选参数，例如可以传入一个 'initial_state'。
        """
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        
        # 允许通过 options 指定一个初始状态
        if options and 'initial_state' in options:
            initial_qpos = options['initial_state'][:5]
            initial_qvel = options['initial_state'][5:10]
        else:
            # 默认重置到一个随机的初始角度和零速度
            initial_qpos = self.np_random.uniform(low=-0.3, high=0.3, size=self.udim)
            initial_qvel = np.zeros(self.udim)

        self.data.qpos[self.joint_ids] = initial_qpos
        self.data.qvel[self.joint_ids] = initial_qvel
        
        # 必须调用 mj_forward 来更新所有派生量 (如 site_xpos)
        mujoco.mj_forward(self.model, self.data)
        
        initial_state = self._get_state()
        
        return initial_state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        施加控制输入 u (action)，并返回系统的下一个状态 x'。

        Args:
            action (np.ndarray): 控制输入 u, 归一化到 [-2, 2]。

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]: 
            (next_state, 0.0, False, False, {})
        """
        # 重力补偿
        # self.data.qfrc_applied[:] =  self.data.qfrc_bias[:]
        # 1. 将归一化的 action 缩放到真实的物理单位 (目标速度)
        target_velocity = action[:self.udim] 
        # 2. 将目标速度施加到控制器
        # 注意：这里我们使用的是速度控制，所以直接设置 qvel 是不正确的。
        # 应该使用 MuJoCo 的执行器。假设XML中定义了 velocity 类型的执行器。
        # 如果是力矩控制，就设置 data.ctrl 为力矩。
        # 为简单起见，我们假设XML中配置了 velocity 执行器，其输入直接是目标速度。
        self.data.ctrl[:self.udim] = target_velocity
        
        # 3. 执行物理模拟
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        # 4. 获取模拟后的新状态
        next_state = self._get_state()
        
        # 5. 渲染 (如果需要)
        if self.render_mode:
            self.render()
            
        # 6. 返回标准 Gym 格式，但奖励和 done 信号是固定的
        return next_state, 0.0, False, False, {}

    def render(self):
        """渲染环境的当前帧。"""
        if self.render_mode and self.viewer is not None:
            self.viewer.sync()

    def close(self):
        """关闭环境并释放资源。"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None



