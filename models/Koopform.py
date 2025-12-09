import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from .utility import PatchTST_Backbone
from .base_model import StableKoopmanOperator
from .Kan import KAN


class Koopformer(StableKoopmanOperator):
    def __init__(self, x_dim, u_dim, seq_len, d_model, use_stable, use_decoder):
        super().__init__(x_dim, u_dim, [d_model], use_stable, use_decoder)
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_embed = nn.Linear(x_dim, d_model)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model = d_model,  # 输入特征维度
            nhead = 4,  # 多头注意力机制的头数
            dim_feedforward = 96,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2)
        self.u_encode_net = nn.Identity()
        if use_decoder:
            self.lC = nn.Linear(self.Nkoopman, self.x_dim)
        

    def x_encoder(self, x_history: torch.Tensor) -> torch.Tensor:
        # 输入: (B, T, C), 输出: (B, T, d_model)
        if x_history.dim() == 2:
            x_history = x_history.unsqueeze(0)
        x = self.input_embed(x_history)
        # Transformer 编码器
        feat = self.transformer_encoder(x)
        if self.use_decoder:
            return feat[:, -1, :]
        else:
            return torch.cat([x_history[:,-1,:], feat[:, -1, :]], dim=-1)
        
    def x_decoder(self, x_emb: torch.Tensor) -> torch.Tensor:
        return self.lC(x_emb)
    
    def u_encoder(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.u_encode_net(u)
    
    def u_decoder(self, u_emb: torch.Tensor) -> torch.Tensor:
        return u_emb  

class Koopformer_KAN(Koopformer):
    """继承 Koopformer, 只替换 Linear 层为 KAN"""
    def __init__(self, x_dim, u_dim, seq_len, d_model, use_stable=True, use_decoder=True):
        # 先调用父类构造函数
        super().__init__(x_dim, u_dim, seq_len, d_model, use_stable, use_decoder)

        # 替换解码器为 KAN（如果启用）
        if use_decoder:
            self.lC = KAN([self.Nkoopman, self.x_dim])

class Koopformer_PatchTST(StableKoopmanOperator):
    """
    Koopformer_PatchTST: 一个融合了 PatchTST 编码器和严格稳定库普曼算子的动力学模型。

    该模型利用 PatchTST_Backbone 作为其状态编码器(x_encoder)，将一个时间序列窗口
    (历史状态)高效地编码成一个单一的潜在状态向量 z。然后, 在一个由稳定库普曼算子控制的潜在空间中进行动力学传播。

    动力学方程: z_{t+1} = K @ z_t + B @ u_t
    """
    def __init__(self, x_dim: int, u_dim: int, seq_len: int, patch_len: int, d_model: int, rho_max: float = 0.99):
        """
        初始化函数。

        参数:
            x_dim (int): 原始状态空间维度。
            u_dim (int): 控制输入维度。
            seq_len (int): 输入历史序列的长度。
            patch_len (int): PatchTST 的 patch 长度。
            d_model (int): 模型的潜在空间维度 (Nkoopman)。
            rho_max (float): 状态转移矩阵 K 的最大谱半径。
        """
        super().__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.Nkoopman = d_model + x_dim 

        # --- 1. 编码器和解码器 ---
        # 状态编码器：使用 PatchTST_Backbone
        self.x_encode_net = PatchTST_Backbone(
            input_dim = x_dim, 
            seq_len = seq_len, 
            patch_len = patch_len, 
            d_model=d_model
        )
        
        # 控制编码器：在这里我们假设 u_t 是一个瞬时值，所以用恒等映射。
        # 如果 u 也是一个序列，可以设计更复杂的编码器。
        self.u_encode_net = nn.Identity()

        # 解码矩阵 C：将潜在状态 z 解码回原始状态 x
        self.lC = nn.Linear(self.Nkoopman, self.x_dim, bias=False)
        # 手动初始化权重（单位矩阵 + 零填充）
        with torch.no_grad():
            self.lC.weight.data[:self.x_dim, :self.x_dim] = torch.eye(self.x_dim)
            self.lC.weight.data[:, self.x_dim:] = 0.0
        self.lC.weight.requires_grad = False

    def x_encoder(self, x_history: torch.Tensor) -> torch.Tensor:
        """
        将历史状态序列 x_history 编码到潜在空间，得到当前潜在状态 z_t。
        
        参数:
            x_history (Tensor): 形状为 (B, seq_len, x_dim) 的历史数据。
        返回:
            Tensor: 当前潜在状态 z_t, 形状为 (B, Nkoopman)。
        """
        # PatchTST_Backbone 直接输出代表整个序列的潜在向量 z_t
        feat = self.x_encode_net(x_history)
        return torch.cat([x_history[:,-1,:], feat], dim=-1)
          
    def x_decoder(self, x_emb: torch.Tensor) -> torch.Tensor:
        """
        将潜在状态 z 解码回原始状态 x。
        
        参数:
            x_emb (Tensor): 潜在状态 z (例如 z_{t+1})，形状为 (B, Nkoopman)。
        返回:
            Tensor: 预测的下一时刻状态 x_{t+1}，形状为 (B, x_dim)。
        """
        return self.lC(x_emb)
    
    def u_encoder(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        对控制输入 u 进行编码。这里 u 是用于预测的下一时刻的控制输入。
        
        参数:
            x (Tensor): 当前状态（此处未使用）。
            u (Tensor): 用于预测的下一时刻的控制输入，形状 (B, u_dim)。
        """
        return self.u_encode_net(u)
    
    def u_decoder(self, u_emb: torch.Tensor) -> torch.Tensor:
        """对编码后的控制 u_emb 进行解码（此处为恒等映射）。"""
        return u_emb    
