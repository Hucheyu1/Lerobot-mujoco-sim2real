import torch
from torch import nn
import numpy as np

# --------------------------------------------------------------------------- #
# 3) 位置编码与 Patch 嵌入 (Positional encodings & patch embed)                 #
# --------------------------------------------------------------------------- #

class SinCosPosEnc(nn.Module):
    """
    基于正弦和余弦函数的位置编码(Sinusoidal Positional Encoding)。
    这种编码方式为模型提供了关于序列中每个元素位置的信息。
    """
    def __init__(self, d_model: int, max_len: int = 10_000):
        """
        初始化函数。
        
        参数:
            d_model (int): 模型的嵌入维度。
            max_len (int): 预先计算编码的最大序列长度。
        """
        super().__init__()
        # 创建一个从 0到 max_len-1 的位置张量，形状为 [max_len, 1]
        pos = torch.arange(max_len).float().unsqueeze(1)
        
        # 计算除数 `div`，用于缩放不同维度的频率
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10_000.0) / d_model))
        
        # 创建一个全零的位置编码矩阵 `pe`，形状为 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 使用 sin 函数填充偶数索引的维度
        pe[:, 0::2] = torch.sin(pos * div)
        # 使用 cos 函数填充奇数索引的维度
        pe[:, 1::2] = torch.cos(pos * div)
        
        # 将 `pe` 注册为模型的缓冲区（buffer）。
        # 这意味着 `pe` 是模型状态的一部分，但不会被视为可训练的参数。
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
            x (Tensor): 输入张量，形状为 (B, T, d)，其中 B 是批量大小，T 是序列长度，d 是嵌入维度。
        
        返回:
            Tensor: 添加了位置编码的输出张量。
        """
        # 将输入 x 与其对应长度的位置编码相加
        return x + self.pe[: x.size(1)]


class PatchEmbed1D(nn.Module):
    """

    一维 Patch 嵌入模块。
    将一个长时序(time series)分割成多个“补丁”(Patches), 然后将每个 Patch 线性投影到指定的嵌入维度 d_model。
    这类似于 ViT (Vision Transformer) 中的图像 Patch 化操作。
    """
    def __init__(self, in_ch: int, d_model: int,
                 patch_len: int, stride: int):
        """
        初始化函数。
        
        参数:
            in_ch (int): 输入通道数(即时间序列的变量数)——对应机械臂状态。
            d_model (int): 目标嵌入维度。
            patch_len (int): 每个 Patch 的长度——“视野”是 patch_len 个时间步。
            stride (int): 卷积操作的步长，每次移动 stride 步。
        """
        super().__init__()
        # 使用一维卷积层来实现 Patch 化和线性投影
        self.conv = nn.Conv1d(in_ch, d_model,
                              kernel_size=patch_len, stride=stride)

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
            x (Tensor): 输入张量，形状为 (B, T, C)，其中 B 是批量大小, T 是序列长度, C 是状态维度。
            
        返回:
            Tensor: Patch 嵌入后的输出张量，形状为 (B, P, d)，其中 P 是 Patch 的数量, d 是嵌入维度。
        """
        # (B, T, C) -> (B, C, T)，交换维度以适应 Conv1d 的输入要求
        x = x.permute(0, 2, 1)
        
        # 应用卷积操作，输出形状为 (B, d, P)，其中 P = T // stride
        x = self.conv(x)
        
        # (B, d, P) -> (B, P, d)，交换维度以匹配 Transformer 的输入格式
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    """
    序列分解模块。
    将输入的时间序列分解为趋势项 (Trend) 和季节项 (Seasonality/Remainder)。
    常用于 Informer 和 Autoformer 模型中。
    """
    def __init__(self, k: int = 3):
        """
        初始化函数。
        
        参数:
            k (int): 移动平均的核大小(kernel size)。
        """
        super().__init__()
        # 使用一维平均池化层作为移动平均滤波器
        self.avg = nn.AvgPool1d(k, stride=1, padding=k // 2)

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
            x (Tensor): 输入张量，形状为 (B, T, C)。
            
        返回:
            Tuple[Tensor, Tensor]: 趋势项和季节项（残差项）。
        """
        # 计算趋势项。需要先将维度转置为 (B, C, T) 以适应 AvgPool1d，然后再转置回来
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        
        # --- 确保趋势项的长度与输入 x 的长度一致 ---------------------- #
        if trend.size(1) != x.size(1):  # ### 修正 1：检查长度是否不匹配 ###
            # 由于填充（padding）可能导致输出长度与输入不完全一致，
            # 这里进行截断，确保它们有相同的序列长度。
            trend = trend[:, : x.size(1)]  # ### 修正 2：截断趋势项 ###
            
        # 季节项（或残差项）等于原始序列减去趋势项
        return trend, x - trend

class PatchTST_Backbone(nn.Module):
    """
    PatchTST 模型的主干网络部分。
    它负责将输入的时间序列编码为一个固定维度的潜在表示 (latent representation)。
    """
    def __init__(self, input_dim: int, seq_len: int,
                 patch_len: int, d_model: int = 48,
                 num_layers: int = 2, num_heads: int = 4,
                 dim_ff: int = 96):
        """
        初始化函数。
        
        参数:
            input_dim (int): 输入序列的特征维度（变量数）。
            seq_len (int): 输入序列的长度。
            patch_len (int): 每个 Patch 的长度。
            d_model (int): 模型内部的嵌入维度。
            num_layers (int): Transformer编码器的层数。
            num_heads (int): 多头注意力机制的头数。
            dim_ff (int): 前馈神经网络的隐藏层维度。
        """
        super().__init__()
        # 1. Patch化和嵌入层
        self.patch = PatchEmbed1D(input_dim, d_model,
                                  patch_len, patch_len)
        # 2. 计算 Patch 的数量
        n_patches = int(np.ceil(seq_len / patch_len))
        # 3. 位置编码层
        self.pos = SinCosPosEnc(d_model, max_len=n_patches + 1) # +1 是为 [CLS] token 预留位置
        # 4. 标准的 Transformer 编码器层
        enc = nn.TransformerEncoderLayer(d_model, num_heads,
                                         dim_ff, batch_first=True)
        # 5. 堆叠多个编码器层，构成完整的 Transformer 编码器
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        # 6. [CLS] token，用于聚合整个序列的信息，类似于 BERT
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
            x (Tensor): 输入张量，形状为 (B, T, C)。
            
        返回:
            Tensor: 序列的潜在表示，形状为 (B, d_model)。
        """
        # 1. 将输入序列 x 进行 Patch 化和嵌入, (B, T, C) -> (B, P, d_model)
        x = self.patch(x)
        # 2. 准备 [CLS] token，并扩展其批量大小以匹配输入
        cls = self.cls.expand(x.size(0), -1, -1)
        # 3. 将 [CLS] token 拼接到 Patch 序列的开头
        # 说明: 一个可学习的、d_model 维的 [CLS] 向量被复制 B 次，并拼接到每个样本的 Patch 序列的最前面。形状变化: (B, P, d_model) -> (B, P+1, d_model)。
        x = torch.cat([cls, x], dim=1)
        # 4. 添加位置编码并通过 Transformer 编码器,(B, P+1, d_model) -> (B, P+1, d_model) (形状不变，但内容被深度处理)
        x = self.encoder(self.pos(x))
        # 5. 返回 [CLS] token 对应的输出，作为整个序列的聚合表示
        return x[:, 0]