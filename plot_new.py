import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from args import Args


def main():
    # Loading dataset
    Methods = ["DKUC", "DBKN", "IKN", "IBKN"]
    Methods_name = ["DKUC", "DBKN", "IKN", "IBKN"]
    Color = ["blue", "black", "red", "green", "pink", "purple", "orange", "grey"]
    args = Args()

    # 创建 4 行布局，使用 6 列作为最小公倍数以便平分 2 列和 3 列
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(4, 6, figure=fig)

    # 轨迹索引固定，确保输入和输出对应
    fixed_idx = random.randint(0, 2000)
    input_types = ["random", "sin"]  # 按照您的要求，展示 random 和 sin

    for row_idx, input_type in enumerate(input_types):
        # 计算当前输入类型对应的起始行 (random对应0,1行; sin对应2,3行)
        base_row = row_idx * 2

        # --- 准备数据 ---
        # 假设所有方法在同一 input_type 下的 label 是一致的，取最后一个方法的 label 即可
        plot_data = {}
        for i, method in enumerate(Methods):
            save_path = f"./results/{args.env}/{args.suffix}/{method}/test_{input_type}"
            all_preds = np.load(save_path + "/all_preds.npy")
            all_labels_x = np.load(save_path + "/all_labels_x.npy")
            all_labels_u = np.load(save_path + "/all_labels_u.npy")

            steps = np.arange(all_preds.shape[1])
            mse_per_timestep = np.mean((all_preds - all_labels_x) ** 2, axis=(0, 2))

            plot_data[method] = {
                "preds": all_preds[fixed_idx],
                "mse": mse_per_timestep,
                "labels_x": all_labels_x[fixed_idx],
                "labels_u": all_labels_u[fixed_idx],
                "steps": steps,
            }

        # --- 第一/三行：左(u) 右(Error) ---
        ax_u = fig.add_subplot(gs[base_row, 0:3])  # 占左半边
        ax_err = fig.add_subplot(gs[base_row, 3:6])  # 占右半边

        # 1. 绘制控制输入 u (取最后一个方法的 label)
        u_data = plot_data[Methods[-1]]["labels_u"]
        steps = plot_data[Methods[-1]]["steps"]
        for j in range(min(5, u_data.shape[-1])):
            ax_u.plot(steps, u_data[:, j], color=Color[j], label=f"ω{j + 1}")
        ax_u.set_title(f"{input_type.upper()} - Control Input (u)", fontsize=14)
        ax_u.set_ylabel("rad/s")
        ax_u.legend(loc="upper right", fontsize=8)

        # 2. 绘制误差对比 Error
        for i, method in enumerate(Methods):
            start_id = (
                args.seq_len if (method.startswith("Koopformer") or method.startswith("KoopmanLSTMlinear")) else 1
            )
            mse = plot_data[method]["mse"]
            ax_err.plot(
                steps[0:151],
                np.log10(mse[start_id : start_id + 151]),
                color=Color[i],
                label=Methods_name[i],
                linewidth=1.5,
            )
        ax_err.set_title(f"{input_type.upper()} - Log Error Contrast", fontsize=14)
        ax_err.set_ylabel("log10(MSE)")
        ax_err.legend(loc="upper right", fontsize=8)

        # --- 第二/四行：各轴 x, y, z 对比 ---
        ax_xyz = [
            fig.add_subplot(gs[base_row + 1, 0:2]),
            fig.add_subplot(gs[base_row + 1, 2:4]),
            fig.add_subplot(gs[base_row + 1, 4:6]),
        ]
        axis_names = ["x", "y", "z"]

        for axis_i in range(3):
            # 绘制真值
            ax_xyz[axis_i].plot(
                steps,
                plot_data[Methods[-1]]["labels_x"][:, axis_i],
                color="orange",
                linewidth=2,
                label="Ground Truth",
                zorder=0,
            )
            # 绘制各方法预测值
            for i, method in enumerate(Methods):
                ax_xyz[axis_i].plot(
                    steps,
                    plot_data[method]["preds"][:, axis_i],
                    color=Color[i],
                    linestyle="--",
                    alpha=0.8,
                    label=Methods_name[i],
                )

            ax_xyz[axis_i].set_title(f"{input_type.upper()} - {axis_names[axis_i]} axis", fontsize=12)
            ax_xyz[axis_i].set_xlabel("Steps")
            if axis_i == 0:
                ax_xyz[axis_i].set_ylabel("Position (m)")
            if axis_i == 2:
                ax_xyz[axis_i].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)

    plt.tight_layout()
    # 调整间距防止标题重叠
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    save_path_final = f"./results/{args.env}/{args.suffix}/comparison_plot.png"
    fig.savefig(save_path_final, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
