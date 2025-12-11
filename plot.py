import numpy as np
from args import Args
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import random
from matplotlib import gridspec
def main():
    # Loading dataset
    Methods = ['Koopformer','DKUC','KoopmanLSTMlinear']
    Methods_name = ['Koopformer','DKUC','KoopmanLSTMlinear']

    Color = ['blue','black','red','green','pink','purple','orange','grey']
    args = Args()

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    # np.random.seed(98)
    for i , method in enumerate(Methods):
        output_dir = "./results/" + args.env + "/" + args.suffix + "/" + method
        if method.startswith("Koopformer") or method.startswith("KoopmanLSTMlinear"):
            start_id = args.seq_len # 序列长度
        else:
            start_id = 1
        for col, input_type in enumerate(["random", "chirp"]):
            save_path = output_dir + f"/test_{input_type}"
            horizen = 151
            all_preds = np.load(save_path + "/all_preds.npy")
            all_labels_x = np.load(save_path + "/all_labels_x.npy")
            all_labels_u = np.load(save_path + "/all_labels_u.npy")
            # print(all_preds.shape , all_labels_x.shape) # (2000, 201, 8)
            mse_per_timestep = np.mean((all_preds[:,:,:] - all_labels_x[:,:,:]) ** 2, axis=(0, 2))  # 输出形状为 (200,)
            dis_per_timestep = np.mean((all_preds[:,:,:3] - all_labels_x[:,:,:3]) ** 2, axis=(0, 2))  # 输出形状为 (200,)
            angle_per_timestep = np.mean((all_preds[:,:,3:] - all_labels_x[:,:,3:]) ** 2, axis=(0, 2))  # 输出形状为 (200,)

            _, steps, _ = all_preds.shape
            steps = np.arange(0, steps)
            # ==== 长期预测：前150步 ====
            axes[0,col].plot(steps[0:horizen], np.log10(mse_per_timestep[start_id:start_id+horizen]), color=Color[i] ,label=Methods_name[i], linewidth=1.5)
            # axes[0,col].legend()
            axes[0,col].set_xlabel('Steps',fontsize=12)
            axes[0,col].set_ylabel('log10(Error)',fontsize=12)
            if input_type == "random":
                axes[0,col].set_title('Random Input',fontsize=14)
            if input_type == "sin":
                axes[0,col].set_title('Sinusoidal Periodic Input',fontsize=14)
            if input_type == "chirp":
                axes[0,col].set_title('Chirp Input',fontsize=14)

            if method == 'KoopmanLSTMlinear':
                # ==== 预测结果 ====
                idx = random.randint(0 , len(steps))
                axes[1,col].plot(steps, all_preds[idx, :, 0], color='blue', linestyle='-.', label='Predicted x')
                axes[1,col].plot(steps, all_preds[idx, :, 1], color='purple', linestyle='-.', label='Predicted y')
                axes[1,col].plot(steps, all_preds[idx, :, 2], color='black', linestyle='-.', label='Predicted z')
                axes[1,col].plot(steps, all_labels_x[idx, :, 0], color='green',label='True x')
                axes[1,col].plot(steps, all_labels_x[idx, :, 1], color='yellow', label='True y')
                axes[1,col].plot(steps, all_labels_x[idx, :, 2], color='pink', label='True z')
                # axes[1,col].legend()
                axes[1,col].set_xlabel('Steps',fontsize=12)
                axes[1,col].set_ylabel('position(m)',fontsize=12)
                # ==== 输入结果 ====
                for j in range(5):
                    axes[2,col].plot(steps, all_labels_u[idx, :, j], color=Color[j],label=f'ω{j+1}')
                # axes[2,col].legend()
                axes[2,col].set_xlabel('Steps',fontsize=12)
                axes[2,col].set_ylabel('input(rad/s)',fontsize=12)
                
    # ====== 每行统一图例放在右侧 ======
    for row in range(3):
        handles, labels = axes[row, 0].get_legend_handles_labels()
        axes[row, -1].legend(handles, labels,
                            loc="center left",
                            bbox_to_anchor=(1.05, 0.5),
                            borderaxespad=0)
    plt.tight_layout()
    fig.savefig("./results/" + args.env + "/" + args.suffix + f'/plot.png', dpi=300)
    plt.show()
    plt.close(fig)

    loss_dir = "./results/" + args.env + "/" + args.suffix + \
                    "/" + 'KoopmanLSTMlinear'+ "/" + 'train_losses.json'
    # 读取 JSON 文件
    with open(loss_dir, "r") as f:
        data = json.load(f)
        total_loss = data["total_loss"]
        koopman_loss = data["koopman_loss"]
        pred_loss = data["pred_loss"]
        recon_loss = data["recon_loss"]

    # x轴为迭代次数
    epochs = list(range(1, len(total_loss)+1))
    # 绘图
    plt.figure(figsize=(8,5))
    plt.plot(epochs, total_loss, label="Total Loss")
    plt.plot(epochs, koopman_loss, label="Koopman Loss")
    plt.plot(epochs, pred_loss, label="Prediction Loss")
    plt.plot(epochs, recon_loss, label="Reconstruction Loss")
    plt.yscale('log')  # 换成对数尺度方便观察小数值变化
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("./results/" + args.env + "/" + args.suffix + f'/loss.png', dpi=300)
    plt.show()

    
if __name__ == "__main__":
    main()