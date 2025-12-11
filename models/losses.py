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
    if net.use_decoder:
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

        cont += beta
        beta *= gamma
        x0_emb = x1_emb_pred

    total_loss = koopman_loss + dis_loss + 0.25 * angle_loss + recon_bool * recon_loss
    stable_Loss = spectral_radius_loss(net)
    H_Loss = sparsity_loss(net)
    total_loss = total_loss / cont #+ λ_spec * stable_Loss + λ_sparse * H_Loss
        
    return dict(
        total_loss = total_loss ,
        koopman_loss=koopman_loss / cont,
        pred_loss=pred_loss / cont,
        recon_loss=recon_loss / cont,
        dis_loss =  dis_loss / cont,
        angle_loss = 0.25 * angle_loss / cont,
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

def koopformer_loss(
    epoch,
    batch_data: Dict[str, Tensor],
    net: KoopmanNet,  
    loss_name: str = "mse",
    gamma: float = 0.99,
    pre_length: int = 5,
    device='cpu'
):
    """
    为 Koopformer 模型定制的损失函数。

    该函数执行多步开环预测 (multi-step open-loop prediction)，并计算以下损失：
    1.  pred_loss: 最终预测状态与真实状态之间的差异。
    2.  koopman_loss: 潜在空间中的线性一致性损失。

    Args:
        batch_data (Dict): 包含 'x' 和 'u' 的数据字典。
                           'x' shape: (B, total_len, x_dim)
                           'u' shape: (B, total_len, u_dim)
        net (Koopformer): 训练中的模型实例。
        loss_name (str): 使用的损失类型, 如 "mse" 或 "l1"。
        gamma (float): 多步预测中损失的时间衰减因子。
        pre_length (int): 开环预测的步数。
    """
    recon_bool = 0
    if net.use_decoder:
        recon_bool = 1
    x = batch_data["x"]
    u = batch_data["u"]
    # B, total_len, x_dim = x.shape
    seq_len = net.seq_len # 从模型中获取历史序列长度
    
    # 2. 初始化损失计算器和累积变量
    base_loss_fn = BaseLoss(loss_name, device)
    
    pred_loss = 0.0
    recon_loss = 0.0
    koopman_loss = 0.0
    # (可选) 分别计算位置和角度损失
    dis_loss = 0.0
    angle_loss = 0.0
    
    total_weight = 0.0
    beta = 1.0  # 当前时间步的权重

    # 3. 编码初始状态
    # 准备编码器的输入：拼接历史状态(包括当前状态)
    x_history = x[:,  : seq_len, :]
    
    # 得到当前时刻 t 的潜在状态 z_t
    z_t = net.x_encoder(x_history)

    # 4. 执行多步开环预测循环
    z_current = z_t  # 用于循环迭代的潜在状态
    for i in range(pre_length):
        # 当前时间步的索引
        current_time_idx = (seq_len-1) + i
        
        # a. 获取用于预测的未来控制 u_{t+i} 和真实下一状态 x_{t+i+1}
        u_future = u[:, current_time_idx, :]
        x_true_next = x[:, current_time_idx + 1, :]
        
        # b. 在潜在空间中进行一步预测
        u_emb = net.u_encoder(None, u_future)
        z_pred_next = net.koopman_operation(z_current, u_emb) # 预测 z_{t+i+1}

        # c. 将预测的潜在状态解码回原始空间
        x_pred_next = net.x_decoder(z_pred_next)

        # d. 计算各项损失
        # d.1. 预测损失 (Prediction Loss)
        # 比较预测的 x_{t+i+1} 和真实的 x_{t+i+1}
        current_pred_loss = base_loss_fn(x_pred_next, x_true_next)
        pred_loss += beta * current_pred_loss
        
        # (可选) 分离位置和角度损失
        dis_loss += beta * base_loss_fn(x_pred_next[:, :3], x_true_next[:, :3])
        angle_loss += beta * base_loss_fn(x_pred_next[:, 3:], x_true_next[:, 3:])

        # d.2. 库普曼一致性损失 (Koopman Consistency Loss)
        # 将真实的 x_{t+i+1} 编码，得到 z_true_next
        # 注意：这里为了得到 z_{t+i+1}, 我们需要 t+i 时刻的历史数据
        x_hist_next = x[:, current_time_idx + 1 - (seq_len-1)  : current_time_idx + 2, :]
        z_true_next = net.x_encoder(x_hist_next)
        koopman_loss += beta * base_loss_fn(z_pred_next, z_true_next) 
        # d.3. 重构损失 (Reconstruction Loss) - 只在第一步计算即可
        x_recon_initial = net.x_decoder(z_true_next) # 解码初始 z_t
        recon_loss += beta * base_loss_fn(x_recon_initial, x_hist_next[:,  -1 , :])

        # e. 更新循环变量
        total_weight += beta
        beta *= gamma
        z_current = z_pred_next  # 开环：用预测值进行下一步预测

    # 5. 计算最终加权平均损失
    total_loss = koopman_loss + dis_loss + 0.25 * angle_loss + recon_bool * recon_loss
    total_loss = total_loss / total_weight
    # 6. 返回损失字典 (移除了 H_Loss 和 stable_Loss)
    return dict(
        total_loss=total_loss,
        koopman_loss=koopman_loss / total_weight,
        pred_loss=pred_loss / total_weight,
        recon_loss=recon_loss / total_weight,  
        dis_loss=dis_loss / total_weight,
        angle_loss=angle_loss / total_weight,
        stable_Loss = torch.tensor(0.0),
        H_Loss = torch.tensor(0.0)
    )

def koopformer_eval_loss_new(
    batch_data: Dict[str, Tensor],
    net: KoopmanNet  # Koopformer 模型实例
) -> Dict[str, Tensor]:
    """
    对 Koopformer 模型进行开环预测并评估损失，返回格式与旧函数保持一致。

    1. 使用初始的一段历史数据 (长度为 seq_len) 进行编码，得到初始潜在状态 z_t。
    2. 从该初始状态开始，利用未来的控制信号 u, 在潜在空间中进行连续的多步预测。
    3. 将每一步预测的潜在状态解码回物理空间。
    4. 计算并累加各项损失。
    5. 将所有预测步拼接成一个完整的轨迹张量。

    Args:
        batch_data (Dict): 包含 'x' 和 'u' 的数据字典。
                           'x' shape: (B, total_len, x_dim)
                           'u' shape: (B, total_len, u_dim)
        net (Koopformer): 待评估的模型实例。

    Returns:
        Dict[str, Tensor]: 包含以下键值对的字典：
            - 'pred': 完整的预测轨迹，形状为 (B, total_len, x_dim)。
            - 'pred_loss': 平均预测损失 (标量张量)。
            - 'dis_loss': 平均位置损失 (标量张量)。
            - 'angle_loss': 平均角度损失 (标量张量)。
    """
    net.eval()  # 确保模型处于评估模式

    x = batch_data["x"]
    u = batch_data["u"]
    B, total_len, x_dim = x.shape
    seq_len = net.seq_len  # 从模型中获取历史序列长度

    # --- 1. 使用初始历史序列编码 ---
    x_history = x[:, :seq_len, :]
    # 得到 t = seq_len-1 时刻的潜在状态 z
    z_current = net.x_encoder(x_history)

    pred_list = [x_history]
    base_loss_fn = BaseLoss('mae')
    pred_loss = 0.0
    dis_loss = 0.0
    angle_loss = 0.0
    
    # 预测的步数
    num_pred_steps = total_len - seq_len 

    # --- 3. 执行开环预测循环 ---
    for i in range(num_pred_steps):
        # 当前时间步索引 t
        current_time_idx = (seq_len - 1) + i
        
        # a. 获取用于预测的未来控制 u_t
        u_future = u[:, current_time_idx, :]
        
        # b. 在潜在空间中进行一步预测，得到 z_{t+1}
        u_emb = net.u_encoder(None, u_future) # u_encoder现在只接收u
        z_pred_next = net.koopman_operation(z_current, u_emb)
        
        # c. 将预测的潜在状态解码回物理空间，得到 x_{t+1}
        x_pred_next = net.x_decoder(z_pred_next)
        
        # d. 记录预测结果
        pred_list.append(x_pred_next.unsqueeze(1))
        # e. 计算与真实值的损失
        x_true_next = x[:, current_time_idx + 1, :]
        pred_loss += base_loss_fn(x_pred_next, x_true_next)
        
        # 分别计算位置和角度损失
        dis_loss += base_loss_fn(x_pred_next[:, :3], x_true_next[:, :3])
        angle_loss += base_loss_fn(x_pred_next[:, 3:], x_true_next[:, 3:])

        # f. 更新循环变量，为下一步预测做准备 (开环核心)
        z_current = z_pred_next

    # --- 4. 整理并返回结果 ---
    # 将历史和预测拼接成完整的轨迹
    pred_trajectory = torch.cat(pred_list, dim=1)
    # 计算平均损失
    # 增加一个保护，防止 num_pred_steps 为0

    avg_pred_loss = pred_loss / num_pred_steps
    avg_dis_loss = dis_loss / num_pred_steps
    avg_angle_loss = angle_loss / num_pred_steps

    return dict(
        pred=pred_trajectory,
        pred_loss=avg_pred_loss,
        dis_loss=avg_dis_loss,
        angle_loss=avg_angle_loss
    )

def koopformer_eval_loss_old(
    batch_data: Dict[str, Tensor],
    net: KoopmanNet  # Koopformer 模型实例
) -> Dict[str, Tensor]:
    """
    对 Koopformer 模型进行开环预测并评估损失，返回格式与旧函数保持一致。

    1. 使用初始的一段历史数据 (长度为 seq_len) 进行编码，得到初始潜在状态 z_t。
    2. 从该初始状态开始，利用未来的控制信号 u, 在潜在空间中进行连续的多步预测。
    3. 将每一步预测的潜在状态解码回物理空间。
    4. 计算并累加各项损失。
    5. 将所有预测步拼接成一个完整的轨迹张量。

    Args:
        batch_data (Dict): 包含 'x' 和 'u' 的数据字典。
                           'x' shape: (B, total_len, x_dim)
                           'u' shape: (B, total_len, u_dim)
        net (Koopformer): 待评估的模型实例。

    Returns:
        Dict[str, Tensor]: 包含以下键值对的字典：
            - 'pred': 完整的预测轨迹，形状为 (B, total_len, x_dim)。
            - 'pred_loss': 平均预测损失 (标量张量)。
            - 'dis_loss': 平均位置损失 (标量张量)。
            - 'angle_loss': 平均角度损失 (标量张量)。
    """
    net.eval()  # 确保模型处于评估模式

    x = batch_data["x"]
    u = batch_data["u"]
    B, total_len, x_dim = x.shape
    seq_len = net.seq_len  # 从模型中获取历史序列长度

    # --- 1. 使用初始历史序列编码 ---
    x_history = x[:, :seq_len, :]
    # 得到 t = seq_len-1 时刻的潜在状态 z

    pred = x_history.clone()
    base_loss_fn = BaseLoss('mae')
    pred_loss = 0.0
    dis_loss = 0.0
    angle_loss = 0.0
    
    # 预测的步数
    num_pred_steps = total_len - seq_len 

    # --- 3. 执行开环预测循环 ---
    for i in range(num_pred_steps):
        # 当前时间步索引 t
        current_time_idx = (seq_len - 1) + i
        # a. 获取用于预测的未来控制 u_t
        z_current = net.x_encoder(x_history)
        u_future = u[:, current_time_idx, :]
        
        # b. 在潜在空间中进行一步预测，得到 z_{t+1}
        u_emb = net.u_encoder(None, u_future) # u_encoder现在只接收u
        z_pred_next = net.koopman_operation(z_current, u_emb)
        
        # c. 将预测的潜在状态解码回物理空间，得到 x_{t+1}
        x_pred_next = net.x_decoder(z_pred_next)
        
        # d. 记录预测结果
        pred = torch.cat((pred, x_pred_next.unsqueeze(1)), dim=1)
        # e. 计算与真实值的损失
        x_true_next = x[:, current_time_idx + 1, :]
        pred_loss += base_loss_fn(x_pred_next, x_true_next)
        
        # 分别计算位置和角度损失
        dis_loss += base_loss_fn(x_pred_next[:, :3], x_true_next[:, :3])
        angle_loss += base_loss_fn(x_pred_next[:, 3:], x_true_next[:, 3:])

        # f. 更新循环变量，为下一步预测做准备 (开环核心)
        x_history = pred[:,-seq_len:,:].clone()


    avg_pred_loss = pred_loss / num_pred_steps
    avg_dis_loss = dis_loss / num_pred_steps
    avg_angle_loss = angle_loss / num_pred_steps

    return dict(
        pred=pred,
        pred_loss=avg_pred_loss,
        dis_loss=avg_dis_loss,
        angle_loss=avg_angle_loss
    )