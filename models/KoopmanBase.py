import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from .base_model import KoopmanNet

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

class Koopmanlinear(KoopmanNet):
    def __init__(self, x_dim, u_dim, encode_layers):

        super(KoopmanNet, self).__init__()
        self.Nkoopman = encode_layers[-1] + x_dim
        self.u_dim = u_dim
        self.x_dim = x_dim

        Layers = OrderedDict()
        for layer_i in range(len(encode_layers)-1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i],encode_layers[layer_i+1])
            if layer_i != len(encode_layers)-2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()

        # 状态输入编码器
        self.x_encode_net = nn.Sequential(Layers)
        # 控制输入编码器：恒等映射（直接返回原始控制输入）
        self.u_encode_net = nn.Identity()
        # koopman矩阵
        self.lA = nn.Linear(self.Nkoopman, self.Nkoopman,bias=False)
        self.lA.weight.data = gaussian_init_(self.Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(self.u_dim, self.Nkoopman, bias=False)
        # nn.init.xavier_uniform_(self.lB.weight, gain=nn.init.calculate_gain('linear'))
        # 解码矩阵
        self.lC = nn.Linear(self.Nkoopman, self.x_dim, bias=False)
        # 手动初始化权重（单位矩阵 + 零填充）
        with torch.no_grad():
            self.lC.weight.data[:self.x_dim, :self.x_dim] = torch.eye(self.x_dim)
            self.lC.weight.data[:, self.x_dim:] = 0.0
        self.lC.weight.requires_grad = False
        
    def x_encoder(self, x):
        feat = self.x_encode_net(x)
        return torch.cat([x, feat], dim=-1)
          # concat 原始 + 特征
    
    def x_decoder(self, x_emb):
        return self.lC(x_emb)
    
    def koopman_operation(self, x_emb, u_emb):
        return self.lA(x_emb)+self.lB(u_emb)
    
    def u_encoder(self, x , u):
        return self.u_encode_net(u)
    
    def u_decoder(self, u_emb):
        return u_emb    
    
class KoopmanBlinear(Koopmanlinear):
    def __init__(self, x_dim, u_dim, encode_layers, u_z):
        super().__init__(x_dim, u_dim, encode_layers)
        # 双线性部分
        self.H = nn.Linear(self.Nkoopman * self.u_dim, self.Nkoopman, bias=False)
        nn.init.zeros_(self.H.weight)
        self.u_z = u_z
    def koopman_operation(self, x_emb, u_emb):
        # u_emb: u_dim   x_emb:Nkoopman
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
    
class DKN(Koopmanlinear):
    def __init__(self, x_dim, u_dim, encode_layers):
        super().__init__(x_dim, u_dim, encode_layers)
        # 双线性部分
        BLayers = OrderedDict()
        bilinear_layers = [x_dim+u_dim, 64, 64, 64, u_dim]
        for i in range(len(bilinear_layers)-1):
            BLayers[f"linear_{i}"] = nn.Linear(bilinear_layers[i], bilinear_layers[i+1])
            if i != len(bilinear_layers)-2:
                BLayers[f"relu_{i}"] = nn.ReLU()
        self.u_encode_net = nn.Sequential(BLayers)
    def u_encoder(self, x , u):
        return self.u_encode_net(torch.cat([x, u], dim=-1))
    
class DKAC(Koopmanlinear):
    def __init__(self, x_dim, u_dim, encode_layers):
        super().__init__(x_dim, u_dim, encode_layers)
        # 双线性部分
        BLayers = OrderedDict()
        bilinear_layers = [x_dim, 64, 64, 64, u_dim]
        for i in range(len(bilinear_layers)-1):
            BLayers[f"linear_{i}"] = nn.Linear(bilinear_layers[i], bilinear_layers[i+1])
            if i != len(bilinear_layers)-2:
                BLayers[f"relu_{i}"] = nn.ReLU()
        self.u_encode_net = nn.Sequential(BLayers)
    def u_encoder(self, x , u):
        gu = self.u_encode_net(x)
        return gu * u
 