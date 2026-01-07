import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# 必须显式导入 3D 绘图工具
from args import Args

# 假设这些是你自己的模块
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

    Color = ["#ff0000", "#ff7f00", "#ffff00", "#00ff00", "#0000ff", "#4b0082", "#9400d3"]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_Fig_path = os.path.join(current_dir, "control", "FigResults", config_dict["suffix"])
    if not os.path.exists(save_Fig_path):
        os.makedirs(save_Fig_path)

    # 模拟数据加载（保留原逻辑）
    configs = ConfigGenerator(config_dict)
    traj_config = configs.traj_config
    method_config = configs.method_config
    robustness_config = configs.robustness_config

    scenario_list = ["noise_robustness", "base_performance"]
    results_dict = {}
    for scenario in scenario_list:
        results = configs.get_result(scenario)
        results_dict[scenario] = results.copy()

    scenario = "noise_robustness"
    results = results_dict[scenario]

    if scenario == "noise_robustness":
        # --- 初始化 Figure 1 (仅 3D 轨迹) ---
        fig1 = plt.figure(figsize=(18, 8))  # 1行高度可以小一点
        gs1 = GridSpec(1, 2, figure=fig1)
        axes1 = np.empty((1, 2), dtype=object)

        # --- 初始化 Figure 2 (2D 性能指标) ---
        fig2 = plt.figure(figsize=(18, 18))
        gs2 = GridSpec(3, 2, figure=fig2)
        axes2 = np.empty((3, 2), dtype=object)

        # 创建子图句柄
        for c in range(2):
            # Fig 1: 3D 投影
            axes1[0, c] = fig1.add_subplot(gs1[0, c], projection="3d")
            # Fig 2: 2D 投影
            axes2[0, c] = fig2.add_subplot(gs2[0, c])  # 对应原 axes[1, c] 误差
            axes2[1, c] = fig2.add_subplot(gs2[1, c])  # 对应原 axes[2, c] 位置
            axes2[2, c] = fig2.add_subplot(gs2[2, c])  # 对应原 axes[3, c] 输入

        for idx, traj_name in enumerate(["Fig8", "FigStar"]):
            # ====================== Figure 1: 3D 轨迹展示 ======================
            ax0 = axes1[0, idx]
            ax0.set_title(traj_config[traj_name]["title"], fontsize=15, pad=20)
            ax0.set_xlabel("X(m)", fontsize=10)
            ax0.set_ylabel("Y(m)", fontsize=10)
            ax0.set_zlabel("Z(m)", fontsize=10)

            ref_traj = results[traj_name]["ref_traj"]
            ax0.plot(
                ref_traj[:, 0],
                ref_traj[:, 1],
                ref_traj[:, 2],
                color="black",
                linewidth=2.5,
                alpha=1.0,
                label="Reference",
                zorder=1,
            )

            # ====================== Figure 2: 2D 曲线 ======================
            # 误差图 (第0行)
            ax_err = axes2[0, idx]
            ax_err.set_xlabel("Steps", fontsize=12)
            ax_err.set_ylabel("Tracking error (m)", fontsize=12)

            # 位置图 (第1行)
            ax_pos = axes2[1, idx]
            ax_pos.set_xlabel("Steps", fontsize=12)
            ax_pos.set_ylabel("Position of EE (m)", fontsize=12)
            steps = np.arange(len(ref_traj))
            ax_pos.plot(steps, ref_traj[:, 0], color="green", linewidth=1.2, label="True x", alpha=0.6)
            ax_pos.plot(steps, ref_traj[:, 1], color="orange", linewidth=1.2, label="True y", alpha=0.6)
            ax_pos.plot(steps, ref_traj[:, 2], color="pink", linewidth=1.2, label="True z", alpha=0.6)

            # 输入图 (第2行)
            ax_input = axes2[2, idx]
            ax_input.set_xlabel("Steps", fontsize=12)
            ax_input.set_ylabel("Input (rad/s)", fontsize=12)

            timestep1 = 5
            timestep2 = 20

            for method in method_config:
                if method not in results[traj_name]:
                    continue

                for rob_type in ["none", "KEM"]:
                    if rob_type not in results[traj_name][method]:
                        continue

                    rob_cfg = robustness_config[rob_type]
                    pre_traj = results[traj_name][method][rob_type]["pre_traj"]
                    error = pre_traj[:, :3] - ref_traj[: len(pre_traj)]
                    steps_pre = np.arange(len(pre_traj))

                    # 绘制 3D 轨迹 (Fig 1)
                    ax0.plot(
                        pre_traj[::5, 0],
                        pre_traj[::5, 1],
                        pre_traj[::5, 2],
                        color=rob_cfg["color"],
                        alpha=rob_cfg["alpha"],
                        linewidth=1.5,
                        linestyle=rob_cfg["line_style"],
                        label=rob_cfg["name"],
                        zorder=2,
                    )

                    if rob_type == "none":
                        # 绘制误差 (Fig 2 - row 0)
                        ax_err.plot(
                            steps_pre[::timestep2],
                            error[::timestep2, 0],
                            color="red",
                            linestyle="--",
                            label="error x (None)",
                        )
                        ax_err.plot(
                            steps_pre[::timestep2],
                            error[::timestep2, 1],
                            color="darkorange",
                            linestyle="--",
                            label="error y (None)",
                        )
                        ax_err.plot(
                            steps_pre[::timestep2],
                            error[::timestep2, 2],
                            color="brown",
                            linestyle="--",
                            label="error z (None)",
                        )

                    if rob_type == "KEM":
                        all_labels_u = results[traj_name][method][rob_type]["u"]
                        # 绘制位置分量 (Fig 2 - row 1)
                        ax_pos.plot(
                            steps_pre[::timestep1],
                            pre_traj[::timestep1, 0],
                            color="blue",
                            linestyle=":",
                            label="Pred x (KEM)",
                        )
                        ax_pos.plot(
                            steps_pre[::timestep1],
                            pre_traj[::timestep1, 1],
                            color="purple",
                            linestyle=":",
                            label="Pred y (KEM)",
                        )
                        ax_pos.plot(
                            steps_pre[::timestep1],
                            pre_traj[::timestep1, 2],
                            color="black",
                            linestyle=":",
                            label="Pred z (KEM)",
                        )

                        # 绘制误差 (Fig 2 - row 0)
                        ax_err.plot(steps_pre[::timestep2], error[::timestep2, 0], color="blue", label="error x (KEM)")
                        ax_err.plot(
                            steps_pre[::timestep2], error[::timestep2, 1], color="purple", label="error y (KEM)"
                        )
                        ax_err.plot(steps_pre[::timestep2], error[::timestep2, 2], color="black", label="error z (KEM)")

                        # 绘制输入 (Fig 2 - row 2)
                        for j in range(min(7, all_labels_u.shape[1])):
                            ax_input.plot(
                                steps_pre[::timestep1], all_labels_u[::timestep1, j], color=Color[j], label=f"ω{j + 1}"
                            )

            # 3D 优化
            ax0.view_init(elev=20, azim=-45)
            ax0.grid(True, alpha=0.3)
            # 2D 优化
            ax_err.grid(True, alpha=0.3)
            ax_pos.grid(True, alpha=0.3)
            ax_input.grid(True, alpha=0.3)

        # --- Figure 1 统一图例与保存 ---
        handles1, labels1 = axes1[0, 0].get_legend_handles_labels()
        by_label1 = dict(zip(labels1, handles1))
        axes1[0, -1].legend(
            by_label1.values(), by_label1.keys(), loc="center left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0
        )
        fig1.tight_layout(rect=[0.01, 0.01, 0.85, 0.95])
        fig1.savefig(os.path.join(save_Fig_path, f"{scenario}_3D_Traj.png"), dpi=500, bbox_inches="tight")

        # --- Figure 2 统一图例与保存 ---
        for r in range(3):
            handles2, labels2 = axes2[r, 0].get_legend_handles_labels()
            by_label2 = dict(zip(labels2, handles2))
            axes2[r, -1].legend(
                by_label2.values(), by_label2.keys(), loc="center left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0
            )

        fig2.tight_layout(rect=[0.01, 0.01, 0.85, 0.95])
        fig2.subplots_adjust(hspace=0.3, wspace=0.25)
        fig2.savefig(os.path.join(save_Fig_path, f"{scenario}_2D_Metrics.png"), dpi=500, bbox_inches="tight")

        plt.show()
