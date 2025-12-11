from torch import nn, Tensor
import torch

class KoopmanNet(nn.Module):
    """
    Base class for all Koopman network
    """
    def __init__(self):
        super(KoopmanNet, self).__init__()

    def x_encoder(self, x: Tensor):
        raise NotImplementedError

    def u_encoder(self, x: Tensor, u: Tensor):
        raise NotImplementedError

    def koopman_operation(self, x_emb: Tensor, u_emb: Tensor):
        raise NotImplementedError

    def x_decoder(self, x_emb: Tensor):
        raise NotImplementedError

    def u_decoder(self, u_emb: Tensor):
        raise NotImplementedError


# --- 辅助函数：保持不变 ---
def _orth(w: torch.Tensor) -> torch.Tensor:
    """
    一个辅助函数, 用于将输入的方阵正交化 (orthogonalize)。
    """
    return torch.linalg.qr(w)[0]

def gaussian_init_(n_units, std=1):    
    """高斯初始化函数"""
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

class StableKoopmanOperator(KoopmanNet):
    """
    一个严格稳定的离散时间库普曼算子模块。
    实现了动力学: z_{t+1} = K @ z_t + B @ u_t
    """
    def __init__(self, x_dim: int, u_dim: int, encoder_layers, use_stable=False,use_decoder=False, rho_max: float = 0.99):
        super().__init__()
        self.u_dim = u_dim
        self.x_dim = x_dim
        # 提取非线性特征的维度
        if use_decoder:
            self.Nkoopman = encoder_layers[-1]
        else:
            self.Nkoopman = encoder_layers[-1] + x_dim
        self.use_stable = use_stable
        self.use_decoder = use_decoder
        self.rho_max = rho_max

        if use_stable:
            # 状态转移矩阵 K 的参数 (Orthogonal-Diagonal-Orthogonal 分解)
            self.U_raw = nn.Parameter(torch.randn(self.Nkoopman, self.Nkoopman))
            self.V_raw = nn.Parameter(torch.randn(self.Nkoopman, self.Nkoopman))
            self.S_raw = nn.Parameter(torch.randn(self.Nkoopman))
        else:
            # koopman矩阵
            self.lA = nn.Linear(self.Nkoopman, self.Nkoopman,bias=False)
            self.lA.weight.data = gaussian_init_(self.Nkoopman, std=1)
            U, _, V = torch.svd(self.lA.weight.data)
            self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        # 控制矩阵
        self.lB = nn.Linear(u_dim, self.Nkoopman, bias=False)
        # 解码矩阵 C：将潜在状态 z 解码回原始状态 x
        if not use_decoder:
            self.lC = nn.Linear(self.Nkoopman, self.x_dim, bias=False)
            # 手动初始化权重（单位矩阵 + 零填充）
            with torch.no_grad():
                self.lC.weight.data[:self.x_dim, :self.x_dim] = torch.eye(self.x_dim)
                self.lC.weight.data[:, self.x_dim:] = 0.0
            self.lC.weight.requires_grad = False

    def get_koopman_matrix_K(self) -> torch.Tensor:
        """
        动态构建严格稳定的库普man矩阵 K。
        通过限制其奇异值来保证谱半径小于 rho_max。
        """
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        # Sigmoid 将奇异值限制在 (0, 1) 区间, 再乘以 rho_max
        Sigma = torch.sigmoid(self.S_raw) * self.rho_max
        K = U @ torch.diag(Sigma) @ V.T
        return K
    
    def koopman_operation(self, x_emb: Tensor, u_emb: Tensor):
        """
        在Koopman空间中进行线性演化。
        """
        if self.use_stable:
            K = self.get_koopman_matrix_K()
            return x_emb @ K.T + self.lB(u_emb)
        else:
            return self.lA(x_emb)+self.lB(u_emb)