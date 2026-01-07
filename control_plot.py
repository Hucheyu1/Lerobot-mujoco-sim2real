import os

import matplotlib.pyplot as plt
import numpy as np

from args import Args
from control.config import ConfigGenerator

if __name__ == "__main__":
    args = Args()
    config_dict = {
        "suffix": "12_11",
        "env_name": args.env,
        "method": "IBKN",
        "use_KEM": False,
        "use_nosie": False,
        "traj_name": "Fig8",  # 使用Fig8\FigStar轨迹进行测试
    }
    # 定义7个输入的颜色（确保不重复，可根据喜好调整）
    Color = ["#ff0000", "#ff7f00", "#ffff00", "#00ff00", "#0000ff", "#4b0082", "#9400d3"]  # 红→紫7色

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_Fig_path = os.path.join(current_dir, "control", "FigResults", config_dict["suffix"])
    if not os.path.exists(save_Fig_path):
        os.makedirs(save_Fig_path)

    configs = ConfigGenerator(config_dict)
    traj_config = configs.traj_config
    method_config = configs.method_config
    robustness_config = configs.robustness_config

    scenario_list = ["noise_robustness", "base_performance"]
    results_dict = {}
    for scenario in scenario_list:
        results = configs.get_result(scenario)
        results_dict[scenario] = results.copy()
    configs.save_table(results_dict, save_Fig_path)

    scenario = "noise_robustness"
    results = results_dict[scenario]
    if scenario == "noise_robustness":
        """绘制噪声鲁棒性测试结果"""
        fig, axes = plt.subplots(4, 2, figsize=(18, 24))
        # fig.suptitle(f'Noise Robustness Comparison (σ=0.005)',
        #             fontsize=18, fontweight='bold')
        # 为每个轨迹类型创建图表
        for idx, traj_name in enumerate(["Fig8", "FigStar"]):
            ax0 = axes[0, idx]
            ax0.set_title(traj_config[traj_name]["title"], fontsize=15)
            ax0.set_xlabel("Y(m)", fontsize=12)
            ax0.set_ylabel("Z(m)", fontsize=12)

            # 绘制参考轨迹
            ref_traj = results[traj_name]["ref_traj"]
            ax0.plot(ref_traj[:, 1], ref_traj[:, 2], color="black", linewidth=2.0, alpha=1.0, label="Reference")
            # ====================== 第1行：预测值vs真实值（X/Y/Z三方向） ======================
            ax1 = axes[2, idx]
            ax1.set_xlabel("Steps", fontsize=12)
            ax1.set_ylabel("Position of the end-effector(m)", fontsize=12)
            # 绘制真实值（x/y/z）
            steps = np.arange(0, len(ref_traj))
            ax1.plot(steps, ref_traj[:, 0], color="green", linewidth=1.5, label="True x")
            ax1.plot(steps, ref_traj[:, 1], color="yellow", linewidth=1.5, label="True y")
            ax1.plot(steps, ref_traj[:, 2], color="pink", linewidth=1.5, label="True z")
            timestep1 = 1
            # ====================== 第2行：误差 ======================
            ax2 = axes[1, idx]
            ax2.set_xlabel("Steps", fontsize=12)
            ax2.set_ylabel("Tracking error(m)", fontsize=12)
            timestep2 = 1
            # ====================== 第3行：输入序列（ω₁~ω₇） ======================
            ax3 = axes[3, idx]
            ax3.set_xlabel("Steps", fontsize=12)
            ax3.set_ylabel("Input(rad/s)", fontsize=12)
            # 绘制每种方法和鲁棒性组合的轨迹
            for method in method_config:
                if method not in results[traj_name]:
                    continue

                model_cfg = method_config[method]

                for rob_type in ["none", "KEM"]:
                    if rob_type not in results[traj_name][method]:
                        continue

                    rob_cfg = robustness_config[rob_type]
                    pre_traj = results[traj_name][method][rob_type]["pre_traj"]
                    error = pre_traj[:, :3] - ref_traj[: len(pre_traj)]
                    # 绘制轨迹
                    ax0.plot(
                        pre_traj[::5, 1],
                        pre_traj[::5, 2],
                        color=rob_cfg["color"],
                        alpha=rob_cfg["alpha"],
                        linewidth=1.5,
                        linestyle=rob_cfg["line_style"],
                        label=rob_cfg["name"],
                    )
                    steps1 = np.arange(0, len(pre_traj), timestep1)
                    steps2 = np.arange(0, len(pre_traj), timestep2)
                    if rob_type == "none":
                        # 绘制预测值（x/y/z）
                        # ax1.plot(steps1, pre_traj[:: timestep1, 0], color='red', alpha=1,linestyle=':', linewidth=2.2, label='Pred x (no UKF)')
                        # ax1.plot(steps1, pre_traj[:: timestep1, 1], color='darkorange', alpha=1,linestyle=':', linewidth=2.2, label='Pred y (no UKF)')
                        # ax1.plot(steps1, pre_traj[:: timestep1, 2], color='brown', alpha=1,linestyle=':', linewidth=2.2, label='Pred z (no UKF)')
                        # 绘制误差（x/y/z）
                        ax2.plot(
                            steps2,
                            error[::timestep2, 0],
                            color="red",
                            alpha=1,
                            linestyle="--",
                            linewidth=1.5,
                            label="error x (no UKF)",
                        )
                        ax2.plot(
                            steps2,
                            error[::timestep2, 1],
                            color="darkorange",
                            alpha=1,
                            linestyle="--",
                            linewidth=1.5,
                            label="error y (no UKF)",
                        )
                        ax2.plot(
                            steps2,
                            error[::timestep2, 2],
                            color="brown",
                            alpha=1,
                            linestyle="--",
                            linewidth=1.5,
                            label="error z (no UKF)",
                        )
                    if rob_type == "KEM":
                        # 绘制预测值（x/y/z）
                        all_labels_u = results[traj_name][method][rob_type]["u"]
                        ax1.plot(
                            steps1,
                            pre_traj[::timestep1, 0],
                            color="blue",
                            alpha=0.8,
                            linestyle="--",
                            linewidth=1.5,
                            label="Pred x (with UKF)",
                        )
                        ax1.plot(
                            steps1,
                            pre_traj[::timestep1, 1],
                            color="purple",
                            alpha=0.8,
                            linestyle="--",
                            linewidth=1.5,
                            label="Pred y (with UKF)",
                        )
                        ax1.plot(
                            steps1,
                            pre_traj[::timestep1, 2],
                            color="black",
                            alpha=0.8,
                            linestyle="--",
                            linewidth=1.5,
                            label="Pred z (with UKF)",
                        )
                        # 绘制误差（x/y/z）
                        ax2.plot(
                            steps2,
                            error[::timestep2, 0],
                            color="blue",
                            linestyle="-",
                            linewidth=1.5,
                            label="error x (with UKF)",
                        )
                        ax2.plot(
                            steps2,
                            error[::timestep2, 1],
                            color="purple",
                            linestyle="-",
                            linewidth=1.5,
                            label="error y (with UKF)",
                        )
                        ax2.plot(
                            steps2,
                            error[::timestep2, 2],
                            color="black",
                            linestyle="-",
                            linewidth=1.5,
                            label="error z (with UKF)",
                        )
                        # 绘制7个输入量（ω₁~ω₇）
                        for j in range(args.u_dim):
                            ax3.plot(
                                steps1, all_labels_u[::timestep1, j], color=Color[j], linewidth=1.5, label=f"ω{j + 1}"
                            )
            ax0.grid(True, alpha=0.3)
            ax0.axis("equal")
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)
            ax3.grid(True, alpha=0.3)
            # ax2.axis('equal')
            # ====== 每行统一图例放在右侧 ======
        for row in range(4):
            handles, labels = axes[row, 0].get_legend_handles_labels()
            axes[row, -1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0)
        # ====================== 全局布局优化（关键！避免拥挤） ======================
        # rect=[左, 下, 右, 上]：为大标题和子图留足空间，行间距自动调整
        plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])  # 上下左右留1%边距，标题占5%
        # 调整子图之间的垂直间距（h_pad），避免行与行重叠
        plt.subplots_adjust(hspace=0.20)  # 垂直间距0.3（越大越宽松，建议0.25~0.35）
        plt.savefig(os.path.join(save_Fig_path, f"{scenario}.png"), format="png", dpi=500, bbox_inches="tight")
        plt.show()
