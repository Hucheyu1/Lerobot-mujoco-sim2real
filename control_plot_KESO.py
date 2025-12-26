import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from control.config import ConfigGenerator_DBKN
import matplotlib.patches as mpatches
from args import Args
from matplotlib import gridspec

if __name__ == "__main__":

    args = Args()
    config_dict={
        'suffix' : args.suffix,
        'env_name': args.env,
        'method' : args.model,
        'use_KEM' : False,
        'use_nosie': False,
        'use_payload': 2,  # 0,1,2
        'use_eso': False,
        'traj_name' : 'Fig8'  # 使用Fig8\FigStar轨迹进行测试
    }
    # 定义7个输入的颜色（确保不重复，可根据喜好调整）
    Color = ['#ff0000', '#ff7f00', '#ffff00', '#00ff00', '#0000ff', '#4b0082', '#9400d3']  # 红→紫7色
    labels_dict = {'KEM': 'KF', 'ESO': 'ESO', 'both': 'KESO', 'none':'None'}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_Fig_path = os.path.join(current_dir , "control","FigResults", config_dict['suffix']) 
    if not os.path.exists(save_Fig_path):
        os.makedirs(save_Fig_path)

    configs = ConfigGenerator_DBKN(config_dict)
    traj_config = configs.traj_config
    method_config = configs.method_config
    robustness_config = configs.robustness_config

    scenario_list = ['base_performance','noise_robustness','payload_robustness','both_robustness']
    results_dict = {}
    for scenario in scenario_list:
        results = configs.get_result(scenario)
        results_dict[scenario] = results.copy()
    # configs.save_table(results_dict, save_Fig_path)
    plot_error = True
    scenario = 'noise_robustness' 
    results = results_dict[scenario]
if scenario == 'noise_robustness':
    """绘制噪声鲁棒性测试结果 - 3行6列布局"""
    # 调整画布大小：高度增加以容纳3行，宽度适当增加
    fig = plt.figure(figsize=(24, 18)) 
    
    # 创建 3 行 6 列的网格
    # height_ratios=[1.5, 1, 1]：第0行(平面图)高度是下面时序图的1.5倍，突出显示
    gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[1.5, 1, 1]) 

    # ================== 1. 初始化子图对象 ==================
    # 使用字典来管理不同轨迹对应的子图，方便后续循环调用
    axes_map = {
        'Fig8': {
            '2d': fig.add_subplot(gs[0, 0:3]),          # 第0行，左半部分
            'ts': [fig.add_subplot(gs[1, 0:2]),         # 第1行，X (占2列)
                   fig.add_subplot(gs[1, 2:4]),         # 第1行，Y (占2列)
                   fig.add_subplot(gs[1, 4:6])]         # 第1行，Z (占2列)
        },
        'FigStar': {
            '2d': fig.add_subplot(gs[0, 3:6]),          # 第0行，右半部分
            'ts': [fig.add_subplot(gs[2, 0:2]),         # 第2行，X (占2列)
                   fig.add_subplot(gs[2, 2:4]),         # 第2行，Y (占2列)
                   fig.add_subplot(gs[2, 4:6])]         # 第2行，Z (占2列)
        }
    }

    # ================== 2. 循环绘制内容 ==================
    for traj_name in ['Fig8', 'FigStar']:
        # 获取当前轨迹对应的子图对象
        ax_2d = axes_map[traj_name]['2d']
        axes_ts = axes_map[traj_name]['ts'] # 包含 [ax_x, ax_y, ax_z]
        
        # --- 设置 2D 平面图样式 ---
        ax_2d.set_title(traj_config[traj_name]['title'], fontsize=18, pad=15)
        ax_2d.set_xlabel('Y (m)', fontsize=14)
        if traj_name == 'Fig8':
            ax_2d.set_ylabel('Z (m)', fontsize=14)
        ax_2d.grid(True, alpha=0.3)
        ax_2d.axis('equal')
        # === 关键修改：设置刻度字体大小 ===
        ax_2d.tick_params(axis='both', which='major', labelsize=12)

        # --- 设置 时序图 样式 ---
        axis_labels = ['X (m)', 'Y (m)', 'Z (m)']
        for dim, ax in enumerate(axes_ts):
            # 标题带上轨迹名称，防止混淆
            if plot_error:
                ax.set_title(f"{traj_name} - {axis_labels[dim]} - Error", fontsize=14)
            else:
                ax.set_title(f"{traj_name} - {axis_labels[dim]}", fontsize=14)
            ax.set_xlabel('Steps', fontsize=14)
            ax.grid(True, alpha=0.3)
            # 只有每行的第一个图显示Y轴标签，节省空间
            if dim == 0:
                ax.set_ylabel('Position (m)', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)
        # ================== 绘制参考轨迹 (Reference) ==================
        ref_traj = results[traj_name]['ref_traj']
        steps = np.arange(0, len(ref_traj))
        
        # 1. 画 2D 参考轨迹
        ax_2d.plot(ref_traj[:, 1], ref_traj[:, 2], color='black', linewidth=2.0, alpha=1.0, label='Reference')
        
        # 2. 画 时序 参考轨迹
        for dim in range(3):
            if plot_error:
                axes_ts[dim].plot(steps, np.zeros_like(steps), color='black', linewidth=2.0, label='Reference')
            else:
                axes_ts[dim].plot(steps, ref_traj[:, dim], color='black', linewidth=1.5, label='Reference')

        # ================== 绘制预测轨迹 (Prediction) ==================
        timestep = 1 
        
        for method in method_config:
            if method not in results[traj_name]: continue
            
            for rob_type in ['KEM', 'ESO', 'both']: # 或 'none',
                if rob_type not in results[traj_name][method]: continue
                
                rob_cfg = robustness_config[rob_type]
                # 预测值
                pre_traj = results[traj_name][method][rob_type]['pre_traj']
                # 误差
                error = pre_traj[:,:3]-ref_traj[:len(pre_traj)]
                # --- 画 2D 预测 ---
                ax_2d.plot(
                    pre_traj[::5, 1], pre_traj[::5, 2], 
                    color=rob_cfg['color'], alpha=rob_cfg['alpha'],
                    linewidth=1.5, linestyle=rob_cfg['line_style'],
                    label=rob_cfg['name']
                )

                # --- 画 时序 预测 ---
                pred_steps = np.arange(0, len(pre_traj))
                
                # 定义样式
                for dim in range(3):
                    axes_ts[dim].plot(
                        pred_steps[::timestep], error[::timestep, dim] if plot_error else pre_traj[::timestep, dim],
                        color=rob_cfg['color'], linestyle=rob_cfg['line_style'], linewidth=1.0, alpha=0.8,
                        label=f'Pred with {labels_dict[rob_type]}'
                    )

        # ================== 图例设置 ==================
        # 1. 2D平面图图例 (右上角)
        if traj_name == 'FigStar':
            # ax_2d.legend(loc="upper right", fontsize=14, framealpha=0.9)
            handles, labels = ax_2d.get_legend_handles_labels()
            by_label = dict(zip(labels, handles)) # 去重技巧
            
            ax_2d.legend(
                by_label.values(), by_label.keys(),
                loc='center left',          # 对齐参考点
                bbox_to_anchor=(1.02, 0.5), # (x, y) 坐标，1.02 表示在轴宽度的 1.02 倍处（即右侧外面）
                fontsize=14,                # 字体大小保持一致
                borderaxespad=0
            )
        
        # 2. 时序图图例 (只在每行的最右侧 Z轴图 显示，避免冗余)
        # 获取 Z轴图的句柄
        z_ax = axes_ts[2]
        handles, labels = z_ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles)) # 去重
        # 将图例放在 Z轴图的右侧外面
        z_ax.legend(by_label.values(), by_label.keys(), 
                    loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=14)

    # ================== 全局布局优化 ==================
    # rect 参数留出四周空白，特别是右侧留给图例
    plt.tight_layout(rect=[0, 0.02, 0.95, 0.98]) 
    
    # hspace 控制行间距，wspace 控制列间距
    plt.subplots_adjust(hspace=0.20, wspace=0.35) 
    
    # 保存与显示
    save_path = os.path.join(save_Fig_path, f"{scenario}_3rows.png")
    plt.savefig(save_path, format="png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Figure saved to {save_path}")