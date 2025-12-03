import time
import numpy as np
import torch
from tqdm import tqdm
from models.base_model import KoopmanNet
from models.init_model import init_model
from models.losses import k_linear_loss, pred_and_eval_loss_old, koopformer_loss, koopformer_eval_loss
from args import Args
import os
import shutil
from utility import plot_predictions, plot_contour, dump_json
from torch.utils.tensorboard import SummaryWriter
from copy import copy
from SOARM101.SOARM101_DataCollection import SOARM101DataGenerator
import random

def set_seed(seed: int = 42) -> None:
    """
    为所有可能引入随机性的库设置一个固定的随机种子，以确保实验的可复现性。

    Args:
        seed (int): 要设置的随机种子值。
    """
    # 1. Python 内置的 random 模块
    random.seed(seed)
    # 2. NumPy
    np.random.seed(seed)
    # 3. PyTorch
    # 为所有GPU和CPU设置种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # 为当前GPU设置种子
        torch.cuda.manual_seed(seed)

def evaluate(
        model: KoopmanNet,
        test_loader,
        scaler
):
    scores = []
    dis_scores = []
    angle_scores = []
    all_preds = []
    all_labels = dict(
        x=[],
        u=[]
    )

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):

            if type(model).__name__ == 'Koopformer_PatchTST':
                pred_and_error = koopformer_eval_loss(
                    batch_data=batch,
                    net=model,
                )
            else:
                pred_and_error = pred_and_eval_loss_old(
                    batch_data=batch,
                    net=model,
                )
                
            pred = pred_and_error["pred"]
            error = pred_and_error["pred_loss"]
            dis_loss = pred_and_error["dis_loss"]
            angle_loss = pred_and_error["angle_loss"]

            scores.append(error.unsqueeze(0).cpu().detach())
            dis_scores.append(dis_loss.unsqueeze(0).cpu().detach())
            angle_scores.append(angle_loss.unsqueeze(0).cpu().detach())

            all_preds.append(pred.cpu().detach())
            for key in all_labels.keys():
                all_labels[key].append(batch[key].cpu().detach())

        scores = torch.cat(scores, dim=0).numpy()
        avg_error = np.mean(scores)
        dis_error = np.mean(torch.cat(dis_scores, dim=0).numpy())
        angle_error = np.mean(torch.cat(angle_scores, dim=0).numpy())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        for key in all_labels.keys():
            all_labels[key] = torch.cat(all_labels[key], dim=0).numpy()

    return dict(
        avg_error=avg_error,  # 整个测试集的平均预测误差 1
        dis_error=dis_error,
        angle_error=angle_error,
        scores=scores,        # 每个批次的误差值（数组） (16,)
        all_preds=all_preds,  # 所有预测结果  (2000,200,10)
        all_labels_x=all_labels["x"],  # 所有状态 x 的真实值 (2000,200,10)
        all_labels_u=all_labels["u"]   # 所有控制输入 u 的真实值 (2000,200,7)
    )

def train(
    model,       # 要训练的KoopmanNet模型实例
    train_loader,  # 训练数据集
    test_loader,   # 测试数据集
    output_dir: str,         # 输出目录路径
    num_epochs: int,         # 训练总轮数
    lr: float,               # 学习率
    batch_size: int,         # 训练批次大小
    eval_batch_size: int,    # 评估批次大小
    log_interval: int,       # 日志记录间隔(步数)
    eval_interval: int,      # 评估间隔(轮数)
    x_dim: int,              # 状态变量维度
    u_dim: int,              # 控制输入维度
    loss_name: str,          # 使用的损失函数名称
    gamma: float,            # 损失函数中的权重参数
    device: str,             # 计算设备('cpu'或'cuda'
    pre_length,
    scaler 
):
 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    start_time = time.time()
    global_step = 0

    # 初始化损失记录字典
    all_train_losses = dict(
        total_loss=[],      # 总损失
        koopman_loss=[],    # Koopman算子损失
        pred_loss=[],       # 预测损失
        recon_loss=[]       # 重构损失
    )
    
    writer = SummaryWriter(log_dir=output_dir)
    best_loss = 1000.0
    for epoch in tqdm(range(num_epochs)):
        epoch_start_time = time.time()
        # 初始化当前epoch的损失记录
        epoch_train_losses = dict(
            total_loss=[],
            koopman_loss=[],
            pred_loss=[],
            recon_loss=[],
            dis_loss = [],
            angle_loss = [],
            stable_Loss=[],
            H_Loss=[]
        )
        for step, batch in enumerate(train_loader):
            model.train()
            if type(model).__name__ == 'Koopformer_PatchTST':
                losses = koopformer_loss(
                    epoch,
                    batch_data=batch,
                    net=model,
                    loss_name=loss_name,
                    gamma=gamma,
                    pre_length=pre_length,
                    device=device,
                )
            else:
                losses = k_linear_loss(
                    epoch,
                    batch_data=batch,
                    net=model,
                    loss_name=loss_name,
                    gamma=gamma,
                    pre_length=pre_length,
                    device=device,
                )
            loss = losses["total_loss"]

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #  记录损失值
            for key in losses.keys():
                # 将损失值从张量转换为标量并添加到列表
                epoch_train_losses[key] = epoch_train_losses[key] + [losses[key].item()]
            global_step += 1

            # 达到日志记录间隔时打印日志
            if global_step % log_interval == 0 or step == len(train_loader)-1 :
                # 初始化平均损失字典
                avg_losses = dict(
                    total_loss=[],
                    koopman_loss=[],
                    pred_loss=[],
                    recon_loss=[],
                    dis_loss=[],
                    angle_loss=[],
                    stable_Loss=[],
                    H_Loss=[]
                )
                # 计算各项损失的平均值
                for key in avg_losses.keys():
                    avg_losses[key] = sum(epoch_train_losses[key]) / (len(epoch_train_losses[key]) + 1e-5)
                 # 构建日志信息
                log_info = {
                    "epoch": epoch,
                    "step": step,
                    "lr": f"{scheduler.get_last_lr()[0]:.3e}",
                    "total_loss": f"{avg_losses['total_loss']:.3e}",
                    "koopman_loss": f"{avg_losses['koopman_loss']:.3e}",
                    "pred_loss": f"{avg_losses['pred_loss']:.3e}",
                    "dis_loss": f"{avg_losses['dis_loss']:.3e}",
                    "angle_loss": f"{avg_losses['angle_loss']:.3e}",
                    "recon_loss": f"{avg_losses['recon_loss']:.3e}",
                    "stable_Loss": f"{avg_losses['stable_Loss']:.3e}",
                    "H_Loss": f"{avg_losses['H_Loss']:.3e}",
                    "time": round(time.time() - start_time)
                }
                print(log_info)

        writer.add_scalar('Train/total_loss',log_info["total_loss"], epoch)
        writer.add_scalar('Train/koopman_loss',log_info["koopman_loss"], epoch)
        writer.add_scalar('Train/pred_loss',log_info["pred_loss"], epoch)
        writer.add_scalar('Train/recon_loss',log_info["recon_loss"], epoch)
        writer.add_scalar('Train/stable_Loss',log_info["stable_Loss"], epoch)
        writer.add_scalar('Train/H_Loss',log_info["H_Loss"], epoch)        
        # 达到评估间隔时进行评估
        if (epoch + 1) % eval_interval == 0:
            # 在测试集上评估模型
            eval_info = evaluate(
                model = model,
                test_loader = test_loader,
                scaler = scaler
            )
            print(f"Prediction error: {eval_info['avg_error']}")
            writer.add_scalar('Eval/total_loss',eval_info['avg_error'], epoch)

            if eval_info['avg_error'] < best_loss:
                best_loss = copy(eval_info['avg_error'])
                # 创建检查点目录
                ckpt_dir = output_dir + "/ckpt"
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                # 保存模型检查点
                ckpt_path = ckpt_dir + f"/model-{epoch}.pt"
                print(f"Saving checkpoint to {ckpt_path}")
                torch.save(model.state_dict(), ckpt_path)
                torch.save(model.state_dict(), output_dir + "/best_model.pt")
                # 根据状态维度选择可视化方式
                if x_dim < 20:
                    # 低维数据使用标准预测图
                    plot_predictions(
                        all_preds=eval_info['all_preds'],       # 所有预测结果
                        all_labels_x=eval_info['all_labels_x'], # 真实状态标签
                        all_labels_u=eval_info['all_labels_u'], # 真实控制标签
                        plot_idx=np.random.randint(0, len(eval_info['all_preds']), size=8),
                        # 随机选8个样本
                        save_dir=ckpt_dir, # 保存目录
                        epoch = epoch
                    )
                else:
                    # 高维数据使用等高线图
                    plot_contour(
                        all_preds=eval_info['all_preds'],
                        all_labels_x=eval_info['all_labels_x'],
                        all_labels_u=eval_info['all_labels_u'],
                        plot_idx=np.random.randint(0, len(eval_info['all_preds']), size=8),
                        save_dir=ckpt_dir,
                        epoch = epoch
                    )

                # 保存评估指标
                epoch_scores = dict(
                    epoch=epoch,
                    train_total_loss=float(sum(epoch_train_losses['total_loss']) / (len(epoch_train_losses['total_loss']) + 1e-5)),
                    koopman_loss=float(sum(epoch_train_losses['koopman_loss']) / (len(epoch_train_losses['total_loss']) + 1e-5)),
                    pred_loss=float(sum(epoch_train_losses['pred_loss']) / (len(epoch_train_losses['pred_loss']) + 1e-5)),
                    recon_loss=float(sum(epoch_train_losses['recon_loss']) / (len(epoch_train_losses['recon_loss']) + 1e-5)),
                    evaluate_loss=float(eval_info['avg_error']),
                    stable_Loss=float(sum(epoch_train_losses['stable_Loss']) / (len(epoch_train_losses['stable_Loss']) + 1e-5)),
                    H_Loss=float(sum(epoch_train_losses['H_Loss']) / (len(epoch_train_losses['H_Loss']) + 1e-5)),
                    time=time.time() - epoch_start_time
                )
                dump_json(epoch_scores, ckpt_dir + f"/scores-{epoch}.json")
                dump_json(epoch_scores, output_dir + "/best_scores.json")

        scheduler.step()
        for key in all_train_losses.keys():
            all_train_losses[key] = all_train_losses[key] + [(float(sum(epoch_train_losses[key])/(len(epoch_train_losses[key]) + 1e-5)))] 

    all_train_losses['time'] = time.time()-start_time
    dump_json(all_train_losses, output_dir + "/train_losses.json")

def main(args):
    # Loading dataset
    # set_seed(args.seed)
    data_generate = SOARM101DataGenerator(args)
    # Model
    model = init_model(args)
    print(model)
    scaler = None
    if args.mode == "train":
        data_generate.generate_and_save_data()
        train_loader,val_loader = data_generate.get_train_loader()
        print("==== Saving args ====")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        shutil.copy('./args.py', args.output_dir + '/train_args.py')
        train(
            model=model,
            train_loader = train_loader,
            test_loader = val_loader,
            output_dir = args.output_dir,
            num_epochs = args.num_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            loss_name=args.loss_name,
            gamma=args.gamma,
            device=args.device,
            pre_length = args.pre_length,
            scaler = scaler
        )

    elif args.mode == "test":
        load_model_path = args.output_dir + "/best_model.pt"
        print(f"Loading the checkpoint from {load_model_path}")
        if not os.path.exists(load_model_path):
            raise FileNotFoundError(f"No checkpoint!")
        model.load_state_dict(torch.load(load_model_path))

        test_types = ['random', 'sin', 'chirp'] if args.test_type == 'all' else [args.test_type]
        for test_type in test_types:
            test_loader = data_generate.get_test_loader(test_type)

            eval_info = evaluate(
                model=model,
                test_loader=test_loader,
                scaler = scaler
            )
            print(f"Prediction error: {eval_info['avg_error']}")
            # Save checkpoint
            save_path = args.output_dir + f"/test_{test_type}" 
            plot_predictions(
                all_preds=eval_info['all_preds'],
                all_labels_x=eval_info['all_labels_x'],
                all_labels_u=eval_info['all_labels_u'],
                plot_idx=np.random.randint(0, len(eval_info['all_preds']), size=8),
                save_dir=save_path,
            )
            if args.x_dim > 5:
                plot_contour(
                    all_preds=eval_info['all_preds'],
                    all_labels_x=eval_info['all_labels_x'],
                    all_labels_u=eval_info['all_labels_u'],
                    plot_idx=np.random.randint(0, len(eval_info['all_preds']), size=8),
                    save_dir=save_path,
                )
            save_info = dict(
                avg_error=float(eval_info['avg_error']),
                dis_error=float(eval_info['dis_error']),
                angle_error=float(eval_info['angle_error'])
            )
            dump_json(save_info, save_path + "/scores.json")
            np.save(save_path + "/all_preds.npy", eval_info['all_preds'])
            np.save(save_path + "/all_labels_x.npy", eval_info['all_labels_x'])
            np.save(save_path + "/all_labels_u.npy", eval_info['all_labels_u'])

if __name__ == "__main__":
    args = Args()
    if args.model == 'all':
        methods = ['DKUC', 'DBKN', 'IBKN', 'IKN']
        if args.mode == 'test':
            args.device = 'cpu'
        for method in methods:  
            args.model = method
            args.output_dir = "./results/" + args.env + "/" + args.suffix + "/"+ args.model
            main(args)

    else:
        main(args)