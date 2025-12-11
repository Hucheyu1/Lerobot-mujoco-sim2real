import argparse
import os
class Args:
    def __init__(self):
        # 初始化参数解析器
        self.parser = argparse.ArgumentParser(description="Koopman 模型训练/测试参数配置")
        self._add_args()  # 添加所有参数
        self.args = self.parser.parse_args()  # 解析命令行参数
        self.process_args()  # 动态处理路径等衍生参数

    def _add_args(self):
        # 核心参数
        self.parser.add_argument("--model", type=str, default="all", 
                                choices=["DKUC", "DBKN", "IKN", "IBKN", "all",
                                         'KoopmanLSTMlinear_KAN', "Koopformer",'KANKoopman'],
                                help="模型类型,可选值: DKUC, DBKN, IKN, IBKN, all (默认: all)")
        self.parser.add_argument("--mode", type=str, default="train",
                                choices=["train", "test"],
                                help="运行模式, 可选值: train / test (默认: train)")
        self.parser.add_argument("--env", type=str, default="SOARM101",
                                help="环境名称（用于路径生成）(默认: SOARM101)")
        self.parser.add_argument("--suffix", type=str, default="12_11",
                                help="实验后缀（用于路径区分）")
        self.parser.add_argument("--seed", type=int, default=42,
                                help="随机种子 (默认: 42)")

        # 数据收集参数
        self.parser.add_argument("--train_samples", type=int, default=50000,
                                help="训练数据样本数 (默认: 50000)")
        self.parser.add_argument("--train_steps", type=int, default=20,
                                help="每个训练样本的时间步长 (默认: 16)")
        self.parser.add_argument("--test_samples", type=int, default=2000,
                                help="测试数据样本数 (默认: 2000)")
        self.parser.add_argument("--test_steps", type=int, default=200,
                                help="每个测试样本的时间步长 (默认: 200)")
        self.parser.add_argument("--test_type", type=str, default="all",
                                choices=["sin", "random", "chirp", "all"],
                                help="测试数据类型, 可选值: sin / random / chirp / all (默认: all)")
        # 预测参数
        self.parser.add_argument("--pre_length", type=int, default=5,
                                help="预测长度 (默认: 5)")

        # 网络设计参数
        self.parser.add_argument("--u_z", action="store_true", default=False,
                                help="是否使用u-z关联(Koopman线性/双线性网络）(默认: False)")
        self.parser.add_argument("--is_load", action="store_true", default=False,
                                help="是否加载预训练模型 (默认: False)")
        self.parser.add_argument("--x_dim", type=int, default=8,
                                help="状态维度 (默认: 8)")
        self.parser.add_argument("--u_dim", type=int, default=5,
                                help="控制输入维度 (默认: 5)")

        # 训练参数
        self.parser.add_argument("--lr", type=float, default=1e-3,
                                help="学习率 (默认: 1e-3)")
        self.parser.add_argument("--num_epochs", type=int, default=500,
                                help="训练轮数 (默认: 300)")
        self.parser.add_argument("--batch_size", type=int, default=128,
                                help="训练批次大小 (默认: 128)")
        self.parser.add_argument("--eval_batch_size", type=int, default=128,
                                help="评估批次大小 (默认: 128)")
        self.parser.add_argument("--log_interval", type=int, default=100,
                                help="日志打印间隔（单位：迭代次数）(默认: 100)")
        self.parser.add_argument("--eval_interval", type=int, default=10,
                                help="评估间隔(单位: epoch) (默认: 10)")
        self.parser.add_argument("--loss_name", type=str, default="mse",
                                choices=["weighted_mse", "mse", "nmse"],
                                help="损失函数类型,可选值:weighted_mse / mse / nmse (默认: mse)")
        self.parser.add_argument("--gamma", type=float, default=0.8,
                                help="加权损失系数 (默认: 0.8)")
        self.parser.add_argument("--device", type=str, default="cuda",
                                choices=["cpu", "cuda"],
                                help="计算设备, 可选值: cpu / cuda (默认: cuda)")
        
        # 控制实验参数
        self.parser.add_argument("--MPC_type", type=str, default="delta_mpc",
                                choices=["delta_mpc", "mpc"],
                                help="MPC类型,可选值: delta_mpc / mpc (默认: mpc)")
        self.parser.add_argument("--traj_name", type=str, default="Fig8",
                                choices=["FigStar", "Fig8", "love"],
                                help="轨迹形状, 可选值: FigStar / Fig8 / love")
        self.parser.add_argument("--UKF", action="store_true", default=False,
                                help="是否使用无迹卡尔曼滤波")
        self.parser.add_argument("--noise", action="store_true", default=False,
                                help="是否添加噪声")
        self.parser.add_argument("--render", action="store_true", default=True,
                                help="是否渲染")

        self.parser.add_argument("--use_stable", type=bool, default=False,
                                help="是否使用稳定Koopman")
        self.parser.add_argument("--use_decoder", type=bool, default=False,
                                help="是否训练解码器")
        
    def process_args(self):
        project_root = os.path.abspath('.') 
        """动态处理衍生参数（和原 Tap 类的 process_args 功能一致）"""
        self.args.xml_path = os.path.join(project_root, self.args.env, "SO101", "scene_with_table_v.xml")

        # 动态生成输出目录
        self.args.output_dir = os.path.join(project_root, "results", self.args.env, self.args.suffix, self.args.model)

        # 动态生成数据目录
        self.args.data_dir_save = os.path.join(project_root, self.args.env, "data")
        self.args.data_dir_load_train = os.path.join(project_root, self.args.env, "data", f"train_data_{self.args.train_samples}_{self.args.train_steps}.npy")
        self.args.data_dir_load_test = os.path.join(project_root, self.args.env, "data", f"test_data_{self.args.test_type}_{self.args.test_samples}_{self.args.test_steps}.npy")
        self.args.data_dir_load_val = os.path.join(project_root, self.args.env, "data", f"val_data_{self.args.test_samples}_{self.args.test_steps}.npy")

        # 网络层维度配置
        self.args.layers = [self.args.x_dim, 32, 32, 32, 16]  # 调整输出维度以适应SOARM101
        """Koopman线性/双线性网络的全连接层维度（输入维度=x_dim)"""

        # 可逆网络参数
        self.args.x_blocks = [2, 2]
        """可逆网络x分支的块数"""
        self.args.x_channels = [12, 16]
        """可逆网络x分支的通道数"""   
        self.args.x_hiddens = [64, 128]
        """可逆网络x分支的隐藏层维度"""
        self.args.u_blocks = [2, 2]
        """可逆网络u分支的块数"""
        self.args.u_channels = [12, 16]
        """可逆网络u分支的通道数"""
        self.args.u_hiddens = [64, 128]
        """可逆网络u分支的隐藏层维度"""

        # transform网络参数
        self.args.seq_len = 12
        self.args.patch_len = 4 
        self.args.d_model = 16

        # KAN网络
        self.args.kan_layers = [self.args.x_dim, 32, 16]
        self.args.kan_params = None

        # LSTM网络
        self.args.LSTM_Hidden = 32
        self.args.LSTM_encode_layers = [self.args.LSTM_Hidden, 32, 16]

    def __getattr__(self, name):
        """方便直接通过 Args 实例访问参数（如 args.model 而非 args.args.model)"""
        return getattr(self.args, name)

# 使用示例（和原代码完全兼容）
if __name__ == "__main__":
    # 解析命令行参数（用法和原 Tap 完全一致）
    args = Args()
    # 打印解析后的参数（验证效果）
    print("="*50)
    print("运行参数汇总：")
    print(f"模型: {args.model}")
    print(f"模式: {args.mode}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练数据路径: {args.data_dir_load_train}")
    print(f"学习率: {args.lr}")
    print(f"设备: {args.device}")
    print("="*50)