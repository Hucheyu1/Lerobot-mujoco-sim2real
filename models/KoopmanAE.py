from collections import OrderedDict

import torch
import torch.nn as nn

from .base_model import KoopmanNet


def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std / n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega


class KoopmanAutoencoder(KoopmanNet):
    def __init__(self, x_dim, u_dim, encode_layers, decode_layers):
        super(KoopmanNet, self).__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.Nkoopman = encode_layers[-1]  # Koopman嵌入维度 = encoder最后一层

        # 编码器：x → z
        eLayers = OrderedDict()
        for layer_i in range(len(encode_layers) - 1):
            eLayers[f"linear_{layer_i}"] = nn.Linear(encode_layers[layer_i], encode_layers[layer_i + 1])
            if layer_i != len(encode_layers) - 2:
                eLayers[f"relu_{layer_i}"] = nn.ReLU()
        self.x_encode_net = nn.Sequential(eLayers)

        # 解码器：z → x̂
        dLayers = OrderedDict()
        for layer_i in range(len(decode_layers) - 1):
            dLayers[f"linear_{layer_i}"] = nn.Linear(decode_layers[layer_i], decode_layers[layer_i + 1])
            if layer_i != len(decode_layers) - 2:
                dLayers[f"relu_{layer_i}"] = nn.ReLU()
        self.lC = nn.Sequential(dLayers)

        # 控制输入保持线性直接映射（也可换成 MLP）
        self.u_encode_net = nn.Identity()
        # Koopman动力学矩阵
        self.lA = nn.Linear(self.Nkoopman, self.Nkoopman, bias=False)
        self.lA.weight.data = gaussian_init_(self.Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(self.u_dim, self.Nkoopman, bias=False)

    def x_encoder(self, x):
        return self.x_encode_net(x)

    def x_decoder(self, x_emb):
        return self.lC(x_emb)

    def u_encoder(self, x, u):
        return self.u_encode_net(u)

    def u_decoder(self, u_emb):
        return u_emb

    def koopman_operation(self, x_emb, u_emb):
        return self.lA(x_emb) + self.lB(u_emb)


class KoopmanBAutoencoder(KoopmanAutoencoder):
    def __init__(self, x_dim, u_dim, encode_layers, decode_layers):
        super().__init__(x_dim, u_dim, encode_layers, decode_layers)
        # 双线性部分
        self.H = nn.Linear(self.Nkoopman * self.u_dim, self.Nkoopman, bias=False)

    def koopman_operation(self, x_emb, u_emb):
        # u_emb: u_dim   x_emb:Nkoopman
        linear_term = self.lA(x_emb) + self.lB(u_emb)
        # 双线性项  u_dim*Nkoopman
        z_kron_u = torch.einsum("bi,bj->bij", u_emb, x_emb).reshape(x_emb.shape[0], -1)
        bilinear_term = self.H(z_kron_u)
        return linear_term + bilinear_term
