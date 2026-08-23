# 实验日志

## 2026-08-23：SOARM101 架构对齐后的 UR5e 链路验证

### 验证目的

本轮只验证修正后的数据—模型—损失—MPC 链路能够运行，并检查力矩和安全约束。使用的是 8 条训练轨迹、2 个 epoch 的 `--smoke` 配置，结果不得作为论文中 IBKN 优势的证据。

### 软件环境

- Python 3.10.20
- PyTorch 2.10.0+cu126，CUDA 可用
- MuJoCo 3.4.0
- Gymnasium 1.2.3
- NumPy 2.2.6
- SciPy 1.15.3
- CasADi 3.7.2；当前 Conda 包缺少可加载的 Ipopt DLL，因此 MPC 自动使用 SciPy/SLSQP 后端

### 数据配置

- 状态：`[p_ee(3),q(6),dq(6)]`，15 维。
- 动作：6 维残差关节力矩，文件中单位为 N·m，加载时除以额定力矩。
- 数据行：`[u(6),x(15)]`，21 维。
- 控制周期：0.02 s。
- 训练/验证/三类测试数组：`[8,13,21]` / `[4,13,21]` / 各 `[4,13,21]`。
- 随机种子：42。

### 执行命令

```powershell
python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-ur5e-final
python train.py --mode collect --smoke --force-data
python train.py --model IBKN --mode train --smoke
python train.py --model IBKN --mode test --smoke
python -m control.run_torque_control --steps 300
python -m control.run_koopman_mpc --model IBKN --steps 5 --horizon 2 --MPC-type delta_mpc
```

### 回归测试

- 结果：11 passed。
- 覆盖：15 维环境状态和末端坐标顺序、参考状态不改变仿真本体、21 维数据布局、动作归一化、四模型前向和多步损失、`L_ee+4L_q+L_dq` 权重、两种 Kronecker 排列的双线性矩阵等价性、PD/MPC 力矩限幅、闭环直接力矩运行。

### IBKN 冒烟训练

| epoch | train total loss | validation total RMSE | validation EE RMSE (m) | validation q RMSE (rad) | validation dq RMSE (rad/s) |
|---:|---:|---:|---:|---:|---:|
| 0 | 11.516428 | — | — | — | — |
| 1 | 11.066405 | 1.122203 | 0.547108 | 1.486885 | 0.887615 |

测试集开环指标：

| 信号 | total RMSE | EE RMSE (m) | q RMSE (rad) | dq RMSE (rad/s) |
|---|---:|---:|---:|---:|
| random | 1.089326 | 0.562584 | 1.471631 | 0.801642 |
| sin | 1.113820 | 0.568274 | 1.473336 | 0.877097 |
| chirp | 1.100888 | 0.582714 | 1.465461 | 0.844118 |

这些误差很大，符合“极小数据 + 2 epoch 只能检查接口”的预期。它们说明流程贯通，但不说明模型已经收敛。

### 闭环控制

| 控制器 | 步数 | EE RMSE (m) | q RMSE (rad) | 最大残差力矩 (N·m) | 最大执行力矩 (N·m) | 后端 |
|---|---:|---:|---:|---:|---:|---|
| 重力前馈 + PD | 300 | 未记录 | 0.021828 | 3.707884 | 21.953128 | 解析控制律 |
| 冒烟 IBKN 增量 MPC | 5 | 0.019925 | 0.033949 | 7.500000 | 未记录 | SciPy/SLSQP |

MPC 达到了第一关节的残差力矩上界 7.5 N·m，这是未充分训练模型导致的饱和迹象。因此本结果只能作为约束有效和闭环未安全终止的验证，不能与 PD 作性能排名。

### 本轮结论

1. 末端坐标已经进入环境、数据、网络、损失、参考轨迹和控制评估的统一接口。
2. 损失重新采用 SOARM101 的分区权重，并针对力矩动力学增加速度项。
3. DBKN/IBKN 的双线性矩阵展开与网络前向计算数值一致，排除了 `H` 的 Kronecker 排列错误。
4. MPC 已恢复原来的矩阵冻结与标准/增量控制逻辑；当前机器的 Ipopt 依赖问题由等价 SLSQP 回退处理。
5. 下一轮必须重新采集正式数据、完成多随机种子四模型训练和正式控制实验，不能沿用本冒烟 checkpoint。
