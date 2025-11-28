import torch
import torch.nn as nn
from .base_model import KoopmanNet
from torch import Tensor
import torch.nn.functional as F
from typing import Dict
import random
# 秩约束损失
def rank_loss(model, threshold=1.0):
    """
    Encourage Koopman A matrix to be full rank.
    L2 = Σ relu(1 - σ_i), where σ_i are singular values of A
    """
    A = model.lA.weight           # [N, N]
    s = torch.linalg.svdvals(A)  # 所有奇异值
    L2 = torch.sum(torch.relu(threshold - s))
    return L2

def spectral_radius_loss(model):
    A = model.lA.weight
    eigvals = torch.linalg.eigvals(A)
    return torch.sum(torch.clamp(torch.abs(eigvals) - 1.0, min=0.0))

# 稀疏正则项损失 
def sparsity_loss(model):
    """
    L3 = Σ_i ‖H_i‖₁
    """
    if hasattr(model, "H"):
        # H_blocks = model.get_Hi_list()
        # return sum(torch.norm(H_j, p=1) for H_j in H_blocks)
        return torch.norm(model.H.weight, p=1)
    else:
        return torch.tensor(0.0)
class BaseLoss(nn.Module):
    def __init__(self, loss_name: str = "mse", device='cpu'):
        super(BaseLoss, self).__init__()
        self.loss_name = loss_name
        self.weight = torch.tensor([1.0]*3 + [10.0]*7, device=device).view(1, -1)
    def forward(self, preds: Tensor, labels: Tensor):
        if self.loss_name == "mse":
            loss = F.mse_loss(preds, labels)
        elif self.loss_name == "mae":
            loss = F.l1_loss(preds, labels)
        elif self.loss_name == "nmse":
            loss = (F.mse_loss(preds, labels)) / (torch.square(labels).mean())
        elif self.loss_name == "weighted_mse":
            loss = (F.mse_loss(preds[:,:3], labels[:,:3]) + 9 * F.mse_loss(preds[:,3:], labels[:,3:]))/10
            # loss = torch.mean(self.weight * (preds - labels) ** 2)      
        else:
            raise ValueError(f"Loss name {self.loss_name} not implemented!")
        return loss
    
def k_linear_loss(
        epoch,
        batch_data: Dict[str, Tensor],
        net: KoopmanNet,
        loss_name: str = "mse",
        gamma: float = 0.99,
        pre_length: int = 5,
        device='cpu',
):
    recon_bool = 0
    if type(net).__name__ == 'KoopmanAutoencoder' or 'KoopmanBAutoencoder':
        recon_bool = 1
    x = batch_data["x"]
    u = batch_data["u"]
    _, steps, _ = x.shape
    start_idx = random.randint(0 , steps - pre_length - 1)
    x0 = x[:, start_idx , :]    # 输入长度5

    base_loss_fn = BaseLoss(loss_name, device)
    weighted_loss_fn = BaseLoss(loss_name, device)

    koopman_loss = 0.0
    pred_loss = 0.0
    recon_loss = 0.0
    total_loss = 0.0
    dis_loss = 0.0
    angle_loss = 0.0
    beta = 1.0
    cont = 0.0

    λ_rank = 0 # 0 1e-8
    λ_spec = 1e-3 # 1e-3
    λ_sparse = 1e-9 # 1e-9
    x0_emb = net.x_encoder(x0)
    for i in range(pre_length):
        i = i + start_idx
        u0 = u[:, i, :]
        u0_emb = net.u_encoder(x0, u0)
        
        x1_emb_pred = net.koopman_operation(x0_emb, u0_emb)
        x1_pred = net.x_decoder(x1_emb_pred)
        x1 = x[:, i + 1 , :]
        x1_emb = net.x_encoder(x1)
        '''测试重构损失'''
        x1_recon = net.x_decoder(x1_emb)
        recon_loss += beta * base_loss_fn(x1_recon, x1)
        '''训练损失'''
        koopman_loss += beta * base_loss_fn(x1_emb_pred, x1_emb)
        
        dis_loss += beta * base_loss_fn(x1_pred[:,:3], x1[:,:3])
        angle_loss += beta * base_loss_fn(x1_pred[:,3:], x1[:,3:])

        pred_loss += beta * base_loss_fn(x1_pred, x1)   # 位置与角度单位不一致，角度权重大些 
        total_loss += beta * (base_loss_fn(x1_emb_pred, x1_emb) + \
                              base_loss_fn(x1_pred, x1) + \
                              recon_bool * base_loss_fn(x1_recon, x1) )

        cont += beta
        beta *= gamma
        x0_emb = x1_emb_pred

    
    stable_Loss = spectral_radius_loss(net)
    H_Loss = sparsity_loss(net)
    total_loss = total_loss / cont #+ λ_spec * stable_Loss + λ_sparse * H_Loss
        
    return dict(
        total_loss = total_loss ,
        koopman_loss=koopman_loss / cont,
        pred_loss=pred_loss / cont,
        recon_loss=recon_loss / cont,
        dis_loss = dis_loss / cont,
        angle_loss = angle_loss / cont,
        stable_Loss = stable_Loss * λ_spec,
        H_Loss = H_Loss * λ_sparse
    )


def pred_and_eval_loss_old(
        batch_data: Dict[str, Tensor],
        net: KoopmanNet,
):
    x = batch_data["x"]
    u = batch_data["u"]
    _, steps, _ = x.shape
    x0 = x[:, 0, :]
    pred = x0.clone().unsqueeze(1)

    base_loss_fn = BaseLoss('mae')
    pred_loss = 0.0
    dis_loss = 0.0
    angle_loss = 0.0

    cont = 0.0

    for i in range(steps - 1):
        u0 = u[:, i, :]
        x0_emb = net.x_encoder(x0)
        u0_emb = net.u_encoder(x0, u0)

        x1_emb_pred = net.koopman_operation(x0_emb, u0_emb) # koopman_operation,test_wigth
        x1_pred = net.x_decoder(x1_emb_pred)
        x1 = x[:, i + 1, :]

        pred_loss += base_loss_fn(x1_pred, x1)
        dis_loss += base_loss_fn(x1_pred[:,:3], x1[:,:3])
        angle_loss += base_loss_fn(x1_pred[:,3:], x1[:,3:])

        pred = torch.cat((pred, x1_pred.unsqueeze(1)), dim=1)

        x0 = x1_pred

        cont += 1.0

    return dict(
        pred=pred,
        pred_loss=pred_loss / cont,
        dis_loss=dis_loss / cont,
        angle_loss=angle_loss / cont
    )

def pred_and_eval_loss_new(
        batch_data: Dict[str, Tensor],
        net: KoopmanNet,
):
    x = batch_data["x"]
    u = batch_data["u"]
    _, steps, _ = x.shape
    x0 = x[:, 0, :]
    pred = x0.clone().unsqueeze(1)

    base_loss_fn = BaseLoss('mae')
    pred_loss = 0.0
    dis_loss = 0.0
    angle_loss = 0.0
    cont = 0.0
    x0_emb = net.x_encoder(x0)

    for i in range(steps - 1):
        u0 = u[:, i, :]
        u0_emb = net.u_encoder(x0, u0)

        x1_emb_pred = net.koopman_operation(x0_emb, u0_emb)
        x1_pred = net.x_decoder(x1_emb_pred)
        x1 = x[:, i + 1, :]

        pred_loss += base_loss_fn(x1_pred, x1)
        dis_loss += base_loss_fn(x1_pred[:,:3], x1[:,:3])
        angle_loss += base_loss_fn(x1_pred[:,3:], x1[:,3:])

        pred = torch.cat((pred, x1_pred.unsqueeze(1)), dim=1)

        x0_emb = x1_emb_pred
        x0 = x1
        cont += 1.0

    return dict(
        pred=pred,
        pred_loss=pred_loss / cont,
        dis_loss=dis_loss / cont,
        angle_loss=angle_loss / cont
    )