import numpy as np
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import math
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
sys.path.insert(0, project_root)
from SOARM101.SOARM101_Env import SOARM101Env
from args import Args
import matplotlib.pyplot as plt


class Collater():
    def __init__(self, x_dim: int, u_dim: int, device: str = "cuda"):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.device = device

    def __call__(self, batch_list: list):
        # 解压批次列表：batch_list是包含多个样本的列表
        # 每个样本是一个元组，我们只需要第一个元素（数据张量）
        batch_data = list(zip(*batch_list))[0]
        # 将列表中的张量堆叠成一个批次张量
        # 维度：[batch_size, seq_len, features]
        batch_data = torch.stack(batch_data, dim=0)
        return dict(
            x=batch_data[:, :, self.u_dim:self.u_dim+self.x_dim].to(self.device),
            u=batch_data[:, :, :self.u_dim].to(self.device)
        )

class SineInputGenerator:
    def __init__(self, traj_num, udim=5, 
                 freq_range=(0.01, 0.05), amp_range=(0.05, 0.15), 
                 sine_mask=None, mode="sin"):
        """
        初始化每条轨迹的正弦参数：
        - freq_range: 每个维度频率范围
        - amp_range: 每个维度幅值范围
        - sine_mask: (traj_num, udim) 的 bool 矩阵，指示哪些维度使用正弦，其余为 0
        'sin' 固定频率，'chirp' 扫频
        """
        self.traj_num = traj_num
        self.udim = udim
        self.mode = mode
        self.freq_start = freq_range[0]
        self.freq_end = freq_range[1]

        self.freq_table = np.random.uniform(freq_range[0], freq_range[1], size=(traj_num, udim))
        self.amp_table = np.random.uniform(amp_range[0], amp_range[1], size=(traj_num, udim))
        self.phase_table = np.random.uniform(0, 2*np.pi, size=(traj_num, udim))

        if sine_mask is None:
            self.sine_mask = np.ones((traj_num, udim), dtype=bool)  # 默认所有维度都用正弦
        else:
            self.sine_mask = sine_mask.astype(bool)

    def __call__(self, t, traj_i, T_total=200):
        if self.mode == "sin":
            freq = self.freq_table[traj_i]
        elif self.mode == "chirp":
            # 时间归一化因子
            ratio = t / T_total
            # 线性扫频：从 freq_start 到 freq_end
            freq = self.freq_table[traj_i] + (self.freq_end - self.freq_start) * ratio
            # freq = np.full(self.udim, freq)
        
        amp = self.amp_table[traj_i]
        phase = self.phase_table[traj_i]
        mask = self.sine_mask[traj_i]

        u = np.zeros(self.udim)
        sine_values = amp * np.sin(2 * np.pi * freq * t + phase)
        u[mask] = sine_values[mask]
        return u
# --- 重构后的数据生成器 ---

class SOARM101DataGenerator:
    def __init__(self, args: Args) -> None:
        self.args = args
        self.udim = self.args.u_dim  # 控制维度
        self.xdim = self.args.x_dim  # Koopman模型所需的状态维度

        # 核心改动：实例化基于物理的MuJoCo环境
        # 注意：这里我们不需要可视化，所以 render_mode=None
        print("正在初始化物理仿真环境...")
        self.env = SOARM101Env(xml_path=self.args.xml_path)
        print("物理环境初始化完成。")
        self.collate_fn = Collater(self.args.x_dim, self.args.u_dim, self.args.device)
        
    def generate_physics_based_data(self, traj_num, steps, input_type):
        """
        使用MuJoCo物理引擎生成数据。
        数据格式: [traj_num, steps+1, u_dim + x_dim] = [traj_num, steps+1, 5+8=13]
        """
        if input_type in {'sin', 'chirp'}:
            # 使用您已有的输入信号生成器
            generator = SineInputGenerator(
                traj_num=traj_num,
                udim=self.udim,
                freq_range=(0.0025, 0.05),
                amp_range=(-0.5, 0.5),
                mode=input_type
            )

        # 创建数据存储数组
        data_array = np.empty((traj_num, steps + 1, self.udim + self.xdim))

        for traj_i in tqdm(range(traj_num), desc=f"Generating '{input_type}' data"):
            # 1. 重置环境以获得初始状态
            s0, _ = self.env.reset()
            # 2. 生成初始控制输入 u_0
            if input_type in {'sin', 'chirp'}:
                u0 = generator(0, traj_i)
            else: # 'random'
                u0 = (np.random.rand(self.udim) - 0.5) * 2 * 0.5
            
            # 3. 存储初始数据点 (t=0)
            data_array[traj_i, 0, :] = np.concatenate([u0, s0])
            
            # 4. 在循环中生成轨迹
            for i in range(1, steps + 1):
                # a. 获取上一步的控制输入 u_{i-1} (即 u0)
                u_prev = data_array[traj_i, i-1, :self.udim]
                
                # b. 将 u_{i-1} 应用到环境中，得到真实的下一状态 s_i
                s_next, _, _, _, _ = self.env.step(u_prev)
                
                # c. 生成当前的控制输入 u_i
                if input_type in {'sin', 'chirp'}:
                    u_current = generator(i, traj_i)
                else: # 'random'
                    u_current = (np.random.rand(self.udim) - 0.5) * 2 * 0.5
                # d. 存储数据点 (u_i, s_i)
                data_array[traj_i, i, :] = np.concatenate([u_current, s_next])

        return data_array

    def generate_and_save_data(self):
        """生成并保存所有数据文件（逻辑与您原始代码一致）。"""
        data_dir_save = self.args.data_dir_save
        if not os.path.exists(data_dir_save):
            os.makedirs(data_dir_save)
        os.makedirs(data_dir_save, exist_ok=True)
        train_data_path = self.args.data_dir_load_train  
        val_data_path = self.args.data_dir_load_val  
        # 参数
        test_samples, test_steps = self.args.test_samples, self.args.test_steps
        train_samples, train_steps = self.args.train_samples, self.args.train_steps
        
        if os.path.exists(train_data_path):
            self.train_data = np.load(train_data_path)
        else:
            # 生成训练数据 (random)
            print(f"生成训练数据: {train_samples}条轨迹，每条{train_steps}步")
            self.train_data = self.generate_physics_based_data(train_samples, train_steps, "random")
            np.save(train_data_path, self.train_data)
            print(f"训练数据保存到: {train_data_path}, 形状: {self.train_data.shape}")

        if os.path.exists(val_data_path):
            self.val_data = np.load(val_data_path)
        else:        
            # 生成验证数据 (random)
            print(f"生成验证数据: {test_samples}条轨迹，每条{test_steps}步")
            self.val_data = self.generate_physics_based_data(test_samples, test_steps, "random")
            np.save(val_data_path, self.val_data)
            print(f"验证数据保存到: {val_data_path}, 形状: {self.val_data.shape}")
        
        self.test_data_dict={}
        # 生成测试数据 (三种类型)
        for test_type in ['random', 'sin', 'chirp']:
            test_data_path = os.path.join(project_root, self.args.env, "data", f"test_data_{test_type}_{self.args.test_samples}_{self.args.test_steps}.npy")
            if os.path.exists(test_data_path):
                test_data = np.load(test_data_path)
            else:
                print(f"生成'{test_type}'测试数据: {test_samples}条轨迹，每条{test_steps}步")
                test_data = self.generate_physics_based_data(test_samples, test_steps, test_type)
                np.save(test_data_path, test_data)
                print(f"'{test_type}'测试数据保存到: {test_data_path}, 形状: {test_data.shape}")
            self.test_data_dict[test_type] = test_data
        # 关闭环境
        self.env.close()

      
    def get_train_loader(self):
        train_data_tensor = torch.tensor(self.train_data, dtype=torch.float32)
        train_dataset = TensorDataset(train_data_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, \
                                  collate_fn=self.collate_fn, shuffle=True)
        
        
        val_data_tensor = torch.tensor(self.val_data, dtype=torch.float32)
        val_dataset = TensorDataset(val_data_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.args.eval_batch_size,
                                collate_fn=self.collate_fn, shuffle=False)

        return train_loader, val_loader
    
    def get_test_loader(self, test_type):
        test_data = self.test_data_dict[test_type]
        test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
        test_dataset = TensorDataset(test_data_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.args.eval_batch_size,
                                collate_fn=self.collate_fn, shuffle=False)
    
        return test_loader
        

def visualize_3d_trajectory(data_generator, cartesian_points, traj_type='train'):
    """
    可视化指定类型和索引的轨迹数据中的三维位置。
    
    Args:
        data_generator: 您的 SOARM101DataGenerator 实例。
        traj_type (str): 要可视化的轨迹类型 ('random', 'sin', 'chirp')。
        traj_index (int): 要可视化的轨迹在批次中的索引。
    """
    
    # 1. 获取数据
    if traj_type in ['random', 'sin', 'chirp']:
        data_array = data_generator.test_data_dict.get(traj_type)
        if data_array is None:
            print(f"警告：找不到 {traj_type} 类型的测试数据。请确认数据已生成。")
            return
    elif traj_type == 'train':
        data_array = data_generator.train_data
    else:
        print(f"无效的轨迹类型: {traj_type}")
        return

    reshaped_data = data_array[:,0,:]
    
    # 2. 提取 X, Y, Z 坐标
    # 提取所有轨迹的 X, Y, Z 坐标
    X = reshaped_data[:, 0]
    Y = reshaped_data[:, 1]
    Z = reshaped_data[:, 2]

    # 3. 绘制三维图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    # c=Z：颜色根据 Z 坐标变化，s=2：点的大小为 2
    scatter = ax.scatter(X, Y, Z, 
                         c=Z, 
                         cmap='viridis', 
                         s=0.1, 
                         alpha=1.0)
    # 添加颜色条（Color Bar），显示 Z 坐标的映射
    fig.colorbar(scatter, ax=ax, label='Z Position')
    # 绘制轨迹
    ax.plot(cartesian_points[:, 0], cartesian_points[:, 1], cartesian_points[:, 2])
    # 设置坐标轴标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Data Acquisition Distribution ({traj_type})')
    plt.show()

# --- 主数据收集脚本 ---
if __name__ == "__main__":
    from control.TrajectoryGenerator import CartesianTrajectoryGenerator,CartesianTrajectoryGenerator_pinocchio
    # 1. 配置参数
    args = Args()
    # 2. 创建生成器实例
    data_generate = SOARM101DataGenerator(args)
    # 3. 执行数据生成和保存
    data_generate.generate_and_save_data()
    print("所有数据生成任务已完成！")
    print(data_generate.train_data[0])
    train_loader,val_loader = data_generate.get_train_loader()
    print("训练数据加载成功")
    for test_type in ['random', 'sin', 'chirp']:
        test_loader = data_generate.get_test_loader(test_type)
    print("测试数据加载成功")


    MODEL_XML_PATH = args.xml_path
    EE_SITE_NAME = 'gripperframe' # 你的XML里定义的夹爪中心的 <site>
    NUM_JOINTS = args.u_dim # 你的机器人关节数量
    use_pinocchio = False
    # 创建生成器实例
    if not use_pinocchio:
        traj_generator = CartesianTrajectoryGenerator(
            model_path=MODEL_XML_PATH,
            ee_site_name=EE_SITE_NAME,
            num_joints=NUM_JOINTS,
            idx=1, 
            time_horizon = 60, 
            time_steps_per_sec = 5
        )
        # 定义末端执行器在整个轨迹中要保持的姿态 (例如，垂直向下)
        target_quat = None # 绕X轴旋转90度
        # target_quat = np.array([0, 0, 1, 0]) 
        # 调用generate方法，反解出关节角度
        cartesian_points, joint_angle_traj, time_vec = traj_generator.generate(
            traj_name='Fig8',  # Circle, Fig8
            target_orientation=target_quat
        )
    else:
        ARM_XML_PATH = args.arm_xml_path
        traj_generator = CartesianTrajectoryGenerator_pinocchio(
            arm_model_path=ARM_XML_PATH,
            ee_site_name=EE_SITE_NAME,
            num_joints=NUM_JOINTS,
            idx = 1,
            time_horizon=60,
            time_steps_per_sec=5
        )
        # 角度（度）
        angle_degrees = 0  # 0  90.0
        # 转换为弧度
        angle_radians = math.radians(angle_degrees)
        # 计算 cos 和 sin 值
        c = math.cos(angle_radians)
        s = math.sin(angle_radians)
        # 构建绕 X 轴旋转的矩阵
        target_orientation = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
        # 调用generate方法，它会完成笛卡尔轨迹生成和IK求解两项工作
        cartesian_points, joint_angle_traj, time_vec = traj_generator.generate(
            traj_name='Fig8',  # Circle, Fig8
            target_orientation_matrix=target_orientation
        )

    print("\n正在生成三维轨迹可视化图...")
    visualize_3d_trajectory(data_generate, cartesian_points, traj_type='train')

 
    print(joint_angle_traj)
    N_joints = joint_angle_traj.shape[1]
    plt.figure(figsize=(10, 6))
    # 遍历并绘制每个关节的角度轨迹
    for i in range(N_joints):
        # joint_angle_traj[:, i] 选择了所有时间步的第 i 个关节数据
        plt.plot(time_vec, joint_angle_traj[:, i], label=f'Joint {i + 1} Angle')
    # 设置图表标题和标签
    plt.title('Five Joint Angle Trajectories Over Time')
    plt.xlabel('Time (s)')  # 假设 time_vec 是以秒为单位
    plt.ylabel('Joint Angle (rad)') # 假设关节角度是以弧度为单位

    # 显示图例、网格并调整布局
    plt.legend(loc='best')
    plt.grid(True) 
    plt.show()