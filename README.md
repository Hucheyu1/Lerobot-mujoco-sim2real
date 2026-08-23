# UR5e Torque Koopman

这是从提交 `b34bba601887ee5424af942e65ac315972049873` 重新整理的 UR5e 分支。项目保留原来的“MuJoCo 环境 → 轨迹数据采集 → Koopman 模型训练/测试 → 闭环控制”逻辑，但已完全移除 SOARM101 模型、舵机/串口、ZMQ 真机桥接、旧权重、历史结果和视频。

## 控制语义

- 机器人：MuJoCo Menagerie UR5e 简化动力学模型。
- 状态/网络输入：`x=[p_ee,q,dq]∈R^15`，其中末端笛卡尔坐标 3 维、关节角 6 维、关节速度 6 维，单位分别为 m、rad、rad/s。前三维布局与原 SOARM101 保持一致；保留 `dq` 是因为力矩控制系统需要速度才能构成 Markov 状态。
- 模型输入：`u∈R^6`，表示物理单位为 N·m 的残差关节力矩。
- 执行器：六个 MuJoCo `motor`，额定限制为 `[150,150,150,28,28,28]` N·m。
- 实际执行：`tau_applied = clip(0.9*tau_gravity + u)`；残差、重力项和最终执行力矩分别记录。
- 仿真频率：物理步长 0.002 s，每次控制执行 10 个物理步，即 50 Hz。

注意：这是 dynamics-enabled 的简化仿真模型，不是 Universal Robots 官方仿真器，也不是经硬件辨识验证的数字孪生。模型来源与许可证见 `assets/ur5e/PROVENANCE.md`。

## 目录

```text
UR5e/
  UR5e_Env.py               6 维直接力矩 Gymnasium 环境
  UR5e_DataCollection.py    物理单位轨迹采集和 DataLoader
assets/ur5e/                Menagerie 来源文件和直接 motor 派生 MJCF
models/                     DKUC、DBKN、IKN、IBKN
control/
  TorqueController.py       重力前馈之外的残差力矩 PD
  MPC_Controler.py          基于 A/B/H/C 的 CasADi 残差力矩 MPC
  TrajectoryGenerator.py    安全关节参考轨迹
  run_torque_control.py     闭环力矩控制入口
tests/                      环境、数据、模型和控制语义测试
args.py                     UR5e 固定维度与运行参数
train.py                    collect/train/test 统一入口
```

## 安装与验证

推荐 Python 3.10：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

运行无训练的直接力矩闭环：

```powershell
python -m control.run_torque_control --steps 300
```

结果保存到 `runs/ur5e_torque/control_demo.npz`，包含状态、参考、残差力矩、实际电机力矩和 RMSE。

训练模型后运行 Koopman 预测控制：

```powershell
python -m control.run_koopman_mpc --model IBKN --checkpoint runs/ur5e_torque/IBKN/best_model.pt
```

MPC 沿用原 SOARM101 控制器的矩阵逻辑：从网络读取 `A/B/H/C`，在当前升维状态处冻结双线性项得到 `B_total(z0)=B+sum(z0_j H_j)`，再由 CasADi/Ipopt 求解标准 MPC 或增量 MPC。若本机 CasADi 缺少 Ipopt 运行库，则自动用 SciPy/SLSQP 求解同一目标与约束。默认预测步长为 10，并在求解器内同时约束归一化残差力矩和力矩增量；输出再乘额定力矩恢复为 N·m。当前实现用于仿真验证，不宣称已达到 50 Hz 实时性能。

## 数据与模型

先做小规模端到端检查：

```powershell
python train.py --mode collect --smoke --force-data
python train.py --model IBKN --mode train --smoke
python train.py --model IBKN --mode test --smoke
```

完整数据和四模型运行：

```powershell
python train.py --mode collect --force-data
python train.py --model all --mode train --device cuda
python train.py --model all --mode test --device cuda
```

保存的 `.npy` 每行 21 维，保持物理单位 `[u_Nm(6),p_ee_m(3),q_rad(6),dq_rad_s(6)]`；输入 DataLoader 后只对动作按额定力矩归一化。动作 `u_t` 作用于状态 `x_t` 并产生 `x_{t+1}`。训练仍采用旧项目的多步开环形式，物理预测损失为 `L_ee + 4 L_q + L_dq`，并叠加 Koopman 潜空间一致性损失与原有稳定性正则项。

## 与旧仓库的差异

1. 删除 SOARM101 的 5 维速度伺服环境，但保留其“末端位置优先”的状态布局、分区加权损失以及矩阵 MPC 架构。
2. 删除 SOARM101 真机串口、ZMQ、位置命令回放及所有旧平台结果。
3. 将数据状态统一为 `[p_ee,q,dq]`，动作统一为 6 维残差关节力矩。
4. 只保留论文核心模型 DKUC/DBKN/IKN/IBKN，删除未接入新链路的 KAN、LSTM 和 Koopformer 实验代码。
5. 增加执行器类型、力矩限幅、重力补偿、数据时序和闭环控制测试。
