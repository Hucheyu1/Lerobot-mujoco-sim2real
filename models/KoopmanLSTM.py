import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from .base_model import StableKoopmanOperator
from .Kan import KAN

class LSTMBlock(nn.Module):
    """
    A block that encapsulates LSTM functionality.
    """
    def __init__(self, input_dim, hidden_dim, only_last = True, num_layers=1, bidirectional=False, dropout=0.0):
        """
        Initializes the LSTM block.
        :param input_dim: 每个时间步输入向量的维度，即状态维度或嵌入维度（例如 17)
        :param hidden_dim: 每个时间步 LSTM 输出的隐藏状态维度
        :param only_last : 
        :param num_layers:  LSTM 堆叠的层数（默认 1)
        :param bidirectional: 是否是双向 LSTM(默认 False)
        :param dropout: 是否在多层之间加 dropout,只有当 num_layers > 1 时才生效
        """
        super(LSTMBlock, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first = True)
        # [B, T ,D]
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.only_last = only_last
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x) # [B, T ,D]--(B, T, H)
        return lstm_out[:, -1, :]  # 取最后一个时间步的输出

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

class KoopmanLSTMlinear(StableKoopmanOperator):
    def __init__(self, x_dim, u_dim, seq_len, encode_layers, LSTM_Hidden, use_stable, use_decoder):

        super().__init__(x_dim, u_dim, encode_layers, use_stable, use_decoder)
        self.seq_len = seq_len
        self.input_embed = nn.Linear(x_dim, LSTM_Hidden)
        self.lstm_block = LSTMBlock(input_dim = LSTM_Hidden, hidden_dim= LSTM_Hidden , only_last=True)

        Layers = OrderedDict()
        for layer_i in range(len(encode_layers)-1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i],encode_layers[layer_i+1])
            if layer_i != len(encode_layers)-2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()
                
        # 状态输入编码器
        self.x_encode_net = nn.Sequential(Layers)
        # 控制输入编码器：恒等映射（直接返回原始控制输入）
        self.u_encode_net = nn.Identity()
        if use_decoder:
            self.lC = nn.Linear(self.Nkoopman, self.x_dim)

    def x_encoder(self, x_history: torch.Tensor) -> torch.Tensor:
        # 输入: (B, T, C), 输出: (B, T, d_model)
        if x_history.dim() == 2:
            x_history = x_history.unsqueeze(0)
        x = self.input_embed(x_history)
        x = self.lstm_block(x)    
        feat = self.x_encode_net(x)
        if self.use_decoder:
            return feat
        else:
            return torch.cat([x_history[:,-1,:], feat], dim=-1)
    
    def x_decoder(self, x_emb):
        return self.lC(x_emb)
    
    def u_encoder(self, x , u):
        return self.u_encode_net(u)
    
    def u_decoder(self, u_emb):
        return u_emb    
    
class KoopmanLSTMlinear_KAN(KoopmanLSTMlinear):
    """继承 KoopmanLSTMlinear, 只替换 Linear 层为 KAN"""
    def __init__(self, x_dim, u_dim, seq_len, encode_layers, LSTM_Hidden, use_stable, use_decoder):
        # 先调用父类构造函数
        super().__init__(x_dim, u_dim, seq_len, encode_layers, LSTM_Hidden, use_stable, use_decoder)

        # 替换输入嵌入为 KAN
        self.x_encode_net = KAN(LSTM_Hidden , encode_layers[-1])

        # 替换解码器为 KAN（如果启用）
        if use_decoder:
            self.lC = KAN([self.Nkoopman, self.x_dim])
    
class KoopmanLSTMBlinear(KoopmanLSTMlinear):
    def __init__(self, x_dim, u_dim, seq_len, encode_layers, LSTM_Hidden,use_stable, use_decoder, u_z):
        super().__init__(x_dim, u_dim, seq_len, encode_layers, LSTM_Hidden,use_stable, use_decoder)
        # 双线性部分
        self.H = nn.Linear(self.Nkoopman * self.u_dim, self.Nkoopman, bias=False)
        nn.init.normal_(self.H.weight, mean=0.0, std=1e-3)
        self.u_z = u_z
    def koopman_operation(self, x_emb, u_emb):
        # u_emb: u_dim   x_emb:Nkoopman
        if self.use_stable:
            K = self.get_koopman_matrix_K()
            linear_term = x_emb @ K.T + self.lB(u_emb)
        else:
            linear_term = self.lA(x_emb) + self.lB(u_emb)
        
        # 双线性项  u_dim*Nkoopman
        if self.u_z:
            z_kron_u = torch.einsum('bi,bj->bij', u_emb, x_emb).reshape(x_emb.shape[0], -1)
        else:
            z_kron_u = torch.einsum('bi,bj->bij', x_emb, u_emb).reshape(x_emb.shape[0], -1)
        bilinear_term = self.H(z_kron_u)
        return linear_term + bilinear_term
    
    def build_permutation_matrix(self, n, m):
        # 返回 P ∈ R^{nm × nm}，将 vec(u⊗z) → vec(z⊗u)
        if self.u_z:
            P = np.zeros((n * m, n * m))
            for i in range(n):
                for j in range(m):
                    row = i * m + j
                    col = j * n + i
                    P[col, row] = 1
        else:
            P = np.eye(n * m)
        return P   
    
    def get_Hi_list(self):
        Hd = self.H.weight.clone()
        if self.u_z:
            H_blocks = [Hd[:, i*self.Nkoopman:(i+1)*self.Nkoopman] for i in range(self.u_dim)]  # 每块 shape: (N, N)
        else:
            H_blocks = [Hd[:, j*self.u_dim:(j+1)*self.u_dim] for j in range(self.Nkoopman)]  # 每块 shape: (N, m)        
        return H_blocks

    def get_Hi_numpy(self):
        P = self.build_permutation_matrix(self.u_dim, self.Nkoopman)  # 交换矩阵
        Hd = self.H.weight.cpu().detach().numpy() @ P.T # (32, 224) 转成 z⊗u 的 H
        H_hat_list = []
        for j in range(self.Nkoopman):
            start_idx = j * self.u_dim
            end_idx = (j+1) * self.u_dim
            H_hat_j = Hd[:, start_idx:end_idx].copy()  # (32, 7)
            H_hat_list.append(H_hat_j)    
        return H_hat_list
