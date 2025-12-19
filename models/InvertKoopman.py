import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import StableKoopmanOperator
from torch import nn, Tensor

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n].contiguous()
    x2 = x[:, n:].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class injective_pad(nn.Module):
    """
    Expansion of the input dimension by pad zero

    Input dim: [B, D] = batch size * feature dimension

    Args:
        pad_size (int): The number of pad zero
    """
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size

    def forward(self, x):
        x = F.pad(x, (0, self.pad_size), 'constant', 0)
        return x

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size]


class imlp_block(nn.Module):
    """
    Args:
        in_ch (int): The dimension of input features, total
        hidden_ch (int): The dimension of hidden dimension
        out_ch (int): The dimension of output features, local (total = local * 2)
    """
    def __init__(self, in_ch, out_ch, hidden_ch):
        '''build invertible MLP bottleneck block'''
        super(imlp_block, self).__init__()
        self.pad_size = 2 * out_ch - in_ch
        self.inj_pad = injective_pad(self.pad_size)

        if self.pad_size !=0:
            in_ch = out_ch * 2

        layers = []
        layers.append(nn.Linear(in_ch//2, hidden_ch))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_ch, hidden_ch))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_ch, out_ch))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """bijective or injective block forward"""
        if self.pad_size != 0:
            x = merge(x[0], x[1])
            x = self.inj_pad.forward(x)
            x1, x2 = split(x)
            x = (x1, x2)
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """bijective or injective block inverse"""
        x2, y1 = x[0], x[1]
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.pad_size != 0:
            x = merge(x1, x2)
            x = self.inj_pad.inverse(x)
            x1, x2 = split(x)
            x = (x1, x2)
        else:
            x = (x1, x2)
        return x


class iMLPNet(nn.Module):
    """
    Build the iMLPNet

    Args:
        nBlocks (list): The number of imlp_block in the defined net (depth)
        nChannels (list): The output dimension of the given block
        nHiddens (list): The hidden dimension of the given block
    """
    def __init__(self, nBlocks, nChannels, nHiddens, in_shape=None):
        super(iMLPNet, self).__init__()
        self.in_ch = in_shape
        self.nBlocks = nBlocks

        self.stack = self.imlp_stack(imlp_block, nBlocks, nChannels, nHiddens, in_ch=self.in_ch)

    def imlp_stack(self, _block, nBlocks, nChannels, nHiddens, in_ch):
        """Create stack of imlp blocks"""
        block_list = nn.ModuleList()
        hiddens = []
        channels = []
        for channel, depth, hidden in zip(nChannels, nBlocks, nHiddens):
            hiddens = hiddens + ([hidden]*depth)
            channels = channels + ([channel]*depth)
        for channel, hidden in zip(channels, hiddens):
            block_list.append(
                _block(
                    in_ch,
                    channel,
                    hidden,
                )
            )
            in_ch = channel * 2
        return block_list

    def forward(self, x):
        """imlpnet forward"""
        n = self.in_ch//2
        out = (x[:, :n], x[:, n:])
        for block in self.stack:
            out = block.forward(out)
        out_bij = merge(out[0], out[1])
        return out_bij

    def inverse(self, out_bij):
        """imlpnet inverse"""
        out = split(out_bij)
        for i in range(len(self.stack)):
            out = self.stack[-1-i].inverse(out)
        out = merge(out[0], out[1])
        x = out
        return x


class InvertKoopmanNetLinear(StableKoopmanOperator):
    def __init__(
            self,
            x_dim,            # 状态维度
            x_blocks,         # 状态编码器块数
            x_channels,       # 状态编码器通道数
            x_hiddens,        # 状态编码器隐藏层大小
            u_dim,            # 控制输入维度
            u_blocks,         # 控制输入编码器块数
            u_channels,       # 控制输入编码器通道数
            u_hiddens,        # 控制输入编码器隐藏层大小
            use_stable,
            use_decoder=True
    ):
        super().__init__(x_dim, u_dim, [x_channels[-1] * 2], use_stable, use_decoder)
        self.x_dim = x_dim
        self.x_blocks = x_blocks
        self.x_channels = x_channels
        self.x_hiddens = x_hiddens

        self.u_dim = u_dim
        self.u_blocks = u_blocks
        self.u_channels = u_channels
        self.u_hiddens = u_hiddens

        self.Nkoopman = x_channels[-1] * 2
        self.x_encode_net = iMLPNet(
            nBlocks=self.x_blocks,
            nChannels=self.x_channels,
            nHiddens=self.x_hiddens,
            in_shape=self.x_dim
        )

        # 控制输入编码器：恒等映射（直接返回原始控制输入）
        self.u_encode_net = nn.Identity()

    def x_encoder(self, x: Tensor):
        return self.x_encode_net(x)

    def u_encoder(self, x: Tensor, u: Tensor):
        return self.u_encode_net(u)

    def x_decoder(self, x_emb: Tensor):
        return self.x_encode_net.inverse(x_emb)

    def u_decoder(self, u_emb: Tensor):
        return u_emb
    
class InvertKoopmanNetBLinear(InvertKoopmanNetLinear):
    def __init__(
            self,
            x_dim,            # 状态维度
            x_blocks,         # 状态编码器块数
            x_channels,       # 状态编码器通道数
            x_hiddens,        # 状态编码器隐藏层大小
            u_dim,            # 控制输入维度
            u_blocks,         # 控制输入编码器块数
            u_channels,       # 控制输入编码器通道数
            u_hiddens,        # 控制输入编码器隐藏层大小
            u_z,
            use_stable
    ):
        super().__init__(    
            x_dim,            
            x_blocks,        
            x_channels,       
            x_hiddens,        
            u_dim,            
            u_blocks,         
            u_channels,       
            u_hiddens,
            use_stable
            )       
        
        self.H = nn.Linear(self.Nkoopman * self.u_dim, self.Nkoopman, bias=False)
        nn.init.zeros_(self.H.weight)
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
