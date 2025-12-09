from .base_model import KoopmanNet, StableKoopmanOperator
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from .Kan import KAN

# ==============================================================================
# 2. 融合后的 KAN-Koopman 实现
# ==============================================================================
class KANKoopmanNet(StableKoopmanOperator):
    def __init__(self, x_dim, u_dim, encoder_layers, kan_params=None,use_stable=False,use_decoder=False):
        """
        初始化 KAN-Koopman 网络, 融合了 Koopmanlinear 的设计思想。
        
        Args:
            state_dim (int): 原始状态x的维度。
            control_dim (int): 控制输入u的维度。
            encoder_layers (list): KAN编码器(特征提取部分)的网络结构, 
                                 例如 [64, 128]。最后一层的大小是特征维度。
            kan_params (dict, optional): 传递给KAN模块的超参数, 如 grid_size, spline_order 等。
        """
        
        super().__init__(x_dim, u_dim, encoder_layers, use_stable, use_decoder)
        # 如果未提供KAN参数，使用默认值
        if kan_params is None:
            kan_params = {}

        # --- 1. 构建 KAN 特征提取器 (对应 Koopmanlinear 的 x_encode_net) ---
        self.x_encode_net = KAN(layers_hidden=encoder_layers, **kan_params)
        # --- 2. 控制编码器: 恒等映射 (来自 Koopmanlinear) ---
        self.u_encode_net = nn.Identity()

        if use_decoder:
            self.lC = KAN([self.Nkoopman, self.x_dim])

    # --- 实现基类的方法 ---
    def x_encoder(self, x: Tensor):
        """
        编码器: 提取非线性特征并与原始状态拼接。
        """
        features = self.x_encode_net(x)
        if self.use_decoder:
            return features
        else:
            return torch.cat([x, features], dim=-1)
    def u_encoder(self, x: Tensor, u: Tensor = None):
        """
        控制编码器: 恒等映射。
        """
        return self.u_encode_net(u)

    def x_decoder(self, x_emb: Tensor):
        """
        解码器: 简单的线性投影。
        """
        return self.lC(x_emb)
        
    def u_decoder(self, u_emb: Tensor):
        # Koopmanlinear 中 u_decoder 是恒等映射
        return u_emb
    
    def get_regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        获取KAN特征提取器的正则化损失
        """
        # 现在只有一个KAN模块，所以直接调用它的正则化损失
        return self.x_encode_net.regularization_loss(regularize_activation, regularize_entropy)