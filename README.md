# UR5e Torque Koopman

这是从提交 `b34bba601887ee5424af942e65ac315972049873` 重新整理的 UR5e 分支。项目保留原来的“MuJoCo 环境 → 轨迹数据采集 → Koopman 模型训练/测试 → 闭环控制”逻辑，但已完全移除 SOARM101 模型、舵机/串口、ZMQ 真机桥接、旧权重、历史结果和视频。

## 控制语义

- 机器人：MuJoCo Menagerie UR5e 简化动力学模型。
- 状态：`x=[q,dq]∈R^12`，单位分别为 rad 和 rad/s。
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

保存的 `.npy` 数据保持物理单位 `[u_Nm,q_rad,dq_rad_s]`；输入 DataLoader 后才按额定力矩归一化。动作 `u_t` 作用于状态 `x_t` 并产生 `x_{t+1}`，避免旧代码中目标速度和力矩语义混用。

## 与旧仓库的差异

1. 删除 SOARM101 的 5 维速度伺服环境和 `[EE位置,q]` 非 Markov 状态。
2. 删除 SOARM101 真机串口、ZMQ、位置命令回放及所有旧平台结果。
3. 将数据状态统一为动力学状态 `[q,dq]`，动作统一为 6 维残差关节力矩。
4. 只保留论文核心模型 DKUC/DBKN/IKN/IBKN，删除未接入新链路的 KAN、LSTM 和 Koopformer 实验代码。
5. 增加执行器类型、力矩限幅、重力补偿、数据时序和闭环控制测试。
