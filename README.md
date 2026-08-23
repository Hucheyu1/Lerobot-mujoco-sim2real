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

## 实验运行步骤

以下命令均在项目根目录执行，示例使用 PowerShell。推荐 Python 3.10。完整论文实验设计、风险和消融矩阵见 [`plan.md`](plan.md)，已经执行过的验证结果见 [`experiment_log.md`](experiment_log.md)。

### 1. 安装依赖并检查环境

```powershell
python -m pip install -r requirements.txt
python --version
python -c "import torch, mujoco, casadi; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('mujoco:', mujoco.__version__); print('casadi:', casadi.__version__)"
```

如果 `cuda: False`，将后续训练命令中的 `--device cuda` 改为 `--device cpu`。MPC 首选 CasADi/Ipopt；如果本机 CasADi 缺少 Ipopt 动态库，程序会自动改用 SciPy/SLSQP 求解相同的目标和约束。

### 2. 运行代码回归测试

```powershell
python -m pytest -q -p no:cacheprovider --basetemp runs/pytest_tmp
```

测试覆盖环境状态顺序、末端坐标、21 维数据布局、四种网络、损失权重、双线性矩阵展开、PD 和标准/增量 MPC 力矩约束。测试必须全部通过后才能开始正式数据采集。

### 3. 冒烟实验：先验证完整链路

冒烟模式只生成 8 条训练轨迹并训练 2 个 epoch，用来检查程序能否贯通，不得作为论文结果。

```powershell
python train.py --mode collect --smoke --seed 42 --force-data
python train.py --model IBKN --mode train --smoke --seed 42 --device cpu
python train.py --model IBKN --mode test --smoke --seed 42 --device cpu
python -m control.run_koopman_mpc --model IBKN --checkpoint runs/ur5e_torque/IBKN/best_model.pt --steps 5 --horizon 2 --MPC-type delta_mpc --device cpu --output runs/ur5e_torque/IBKN/mpc_smoke.npz
```

注意：冒烟数据和正式数据使用相同的 `datasets/ur5e_torque` 目录。运行过冒烟实验后，正式实验必须执行下一步带 `--force-data` 的采集命令，不能直接开始正式训练。

### 4. 采集正式训练、验证和测试数据

下面显式写出当前正式默认配置，便于复现实验：训练集 512 条、验证集 96 条、每类测试集 96 条；每条轨迹包含 100 次状态转移。测试集分别使用 random、sin 和 chirp 力矩信号。

```powershell
python train.py --mode collect --seed 42 --train-samples 512 --train-steps 100 --val-samples 96 --test-samples 96 --test-steps 100 --force-data
```

生成文件：

```text
datasets/ur5e_torque/
  train.npy
  val.npy
  test_random.npy
  test_sin.npy
  test_chirp.npy
  manifest.json
```

可以用以下命令确认数据维度。每行应为 21 维：`[u(6),p_ee(3),q(6),dq(6)]`。

```powershell
python -c "import numpy as np; from pathlib import Path; p=Path('datasets/ur5e_torque'); print({f.name: np.load(f).shape for f in p.glob('*.npy')})"
Get-Content datasets/ur5e_torque/manifest.json
```

### 5. 训练四种 Koopman 模型

四种模型使用同一数据、训练轮数、学习率、batch size、多步预测长度和随机种子：

```powershell
python train.py --model all --mode train --seed 42 --device cuda --num-epochs 140 --lr 0.0005 --batch-size 128 --eval-batch-size 128 --pre-length 25 --gamma 0.98 --loss-name mse
```

也可以单独训练某一个模型：

```powershell
python train.py --model DKUC --mode train --seed 42 --device cuda
python train.py --model DBKN --mode train --seed 42 --device cuda
python train.py --model IKN  --mode train --seed 42 --device cuda
python train.py --model IBKN --mode train --seed 42 --device cuda
```

每个模型的结果保存在 `runs/ur5e_torque/<MODEL>/`：

```text
best_model.pt       验证集 RMSE 最低的权重
history.json        每轮训练损失和定期验证指标
```

不要在不同模型之间改变数据、seed、epoch 或调参预算，否则无法公平判断可逆结构和双线性结构的贡献。

### 6. 测试四种模型的开环预测

对 random、sin、chirp 三类独立测试轨迹统一评估：

```powershell
python train.py --model all --mode test --seed 42 --device cuda --test-type all
```

只测试某个模型或某类信号：

```powershell
python train.py --model IBKN --mode test --seed 42 --device cuda --test-type all
python train.py --model IBKN --mode test --seed 42 --device cuda --test-type chirp
```

测试结果保存在 `runs/ur5e_torque/<MODEL>/test_metrics.json`，包含总体、EE、关节角和关节速度 RMSE：

```powershell
Get-Content runs/ur5e_torque/IBKN/test_metrics.json
```

### 7. 运行无学习模型的 PD 控制基线

```powershell
python -m control.run_torque_control --steps 300 --seed 7 --output runs/ur5e_torque/pd_control.npz
```

如需打开 MuJoCo 可视化窗口：

```powershell
python -m control.run_torque_control --steps 300 --seed 7 --render --output runs/ur5e_torque/pd_control_rendered.npz
```

输出包含真实状态、参考状态、残差力矩、最终执行力矩和控制指标。无图形界面的服务器不要使用 `--render`。

### 8. 运行 Koopman MPC 控制实验

IBKN 增量 MPC：

```powershell
python -m control.run_koopman_mpc --model IBKN --checkpoint runs/ur5e_torque/IBKN/best_model.pt --steps 300 --seed 7 --horizon 10 --MPC-type delta_mpc --device cpu --output runs/ur5e_torque/IBKN/mpc_delta.npz
```

IBKN 标准 MPC：

```powershell
python -m control.run_koopman_mpc --model IBKN --checkpoint runs/ur5e_torque/IBKN/best_model.pt --steps 300 --seed 7 --horizon 10 --MPC-type mpc --device cpu --output runs/ur5e_torque/IBKN/mpc_standard.npz
```

依次运行四个模型和两种 MPC：

```powershell
$models = 'DKUC','DBKN','IKN','IBKN'
$controllers = 'mpc','delta_mpc'
foreach ($model in $models) {
    foreach ($controller in $controllers) {
        python -m control.run_koopman_mpc --model $model --checkpoint "runs/ur5e_torque/$model/best_model.pt" --steps 300 --seed 7 --horizon 10 --MPC-type $controller --device cpu --output "runs/ur5e_torque/$model/${controller}_control.npz"
        if ($LASTEXITCODE -ne 0) { throw "$model $controller control failed" }
    }
}
```

MPC 的输入是当前 `x_t=[p_ee,q,dq]` 和未来 `H` 步参考轨迹，输出是 6 维残差力矩。程序从网络读取 `A/B/H/C`，在当前状态处冻结 `B_total(z0)=B+sum(z0_j H_j)` 后求解有限时域控制问题。

### 9. 查看控制结果

```powershell
python -c "import numpy as np; d=np.load('runs/ur5e_torque/IBKN/mpc_delta.npz', allow_pickle=True); print(d['metrics'].item()); print('states:', d['states'].shape); print('references:', d['references'].shape); print('torques:', d['residual_torques'].shape)"
```

控制实验至少检查：

- `ee_rmse_m` 和 `q_rmse_rad`；
- 最大残差力矩是否超过 `[7.5,7.5,7.5,1.4,1.4,1.4]` N·m；
- 是否出现 safety termination；
- `solver_backend` 实际使用 `casadi-ipopt` 还是 `scipy-slsqp`；
- MPC 是否长期饱和。未充分训练的模型即使没有违反约束，也不能据此认定控制有效。

### 10. 多随机种子正式实验

论文结果至少使用 5 个随机种子。当前程序的固定输出目录会被下一次训练覆盖，因此每个 seed 测试完成后必须立即归档模型、指标和对应数据：

```powershell
$seeds = 42,43,44,45,46
foreach ($seed in $seeds) {
    python train.py --mode collect --seed $seed --train-samples 512 --train-steps 100 --val-samples 96 --test-samples 96 --test-steps 100 --force-data
    if ($LASTEXITCODE -ne 0) { throw "seed $seed collection failed" }

    python train.py --model all --mode train --seed $seed --device cuda --num-epochs 140 --pre-length 25
    if ($LASTEXITCODE -ne 0) { throw "seed $seed training failed" }

    python train.py --model all --mode test --seed $seed --device cuda --test-type all
    if ($LASTEXITCODE -ne 0) { throw "seed $seed testing failed" }

    python -m control.run_torque_control --steps 300 --seed $seed --output "runs/ur5e_torque/pd_seed_${seed}.npz"
    if ($LASTEXITCODE -ne 0) { throw "seed $seed PD control failed" }

    $models = 'DKUC','DBKN','IKN','IBKN'
    $controllers = 'mpc','delta_mpc'
    foreach ($model in $models) {
        foreach ($controller in $controllers) {
            python -m control.run_koopman_mpc --model $model --checkpoint "runs/ur5e_torque/$model/best_model.pt" --steps 300 --seed $seed --horizon 10 --MPC-type $controller --device cpu --output "runs/ur5e_torque/$model/${controller}_seed_${seed}.npz"
            if ($LASTEXITCODE -ne 0) { throw "seed $seed $model $controller control failed" }
        }
    }

    $resultDir = "runs/ur5e_torque_seeded/seed_$seed"
    New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
    Copy-Item -Path runs/ur5e_torque/* -Destination $resultDir -Recurse -Force
    New-Item -ItemType Directory -Force -Path "$resultDir/dataset" | Out-Null
    Copy-Item -Path datasets/ur5e_torque/* -Destination "$resultDir/dataset" -Recurse -Force
}
```

完成后应基于五个 seed 的原始 JSON/NPZ 计算均值、标准差和 95% 置信区间。不能只选择 IBKN 表现最好的一次运行，也不能为了符合预设结论而删除失败 seed。

## 数据、损失与结果说明

保存的 `.npy` 每行 21 维，保持物理单位 `[u_Nm(6),p_ee_m(3),q_rad(6),dq_rad_s(6)]`；输入 DataLoader 后只对动作按额定力矩归一化。动作 `u_t` 作用于状态 `x_t` 并产生 `x_{t+1}`。训练采用原项目的多步开环形式，物理预测损失为 `L_ee + 4 L_q + L_dq`，并叠加 Koopman 潜空间一致性损失和稳定性正则项。

`--smoke` 结果、单 seed 结果和短时域 MPC 结果只用于调试。正式论文结论应来自 `plan.md` 中规定的多 seed、等训练预算、结构消融、扰动控制和不完美观测实验。

## 与旧仓库的差异

1. 删除 SOARM101 的 5 维速度伺服环境，但保留其“末端位置优先”的状态布局、分区加权损失以及矩阵 MPC 架构。
2. 删除 SOARM101 真机串口、ZMQ、位置命令回放及所有旧平台结果。
3. 将数据状态统一为 `[p_ee,q,dq]`，动作统一为 6 维残差关节力矩。
4. 只保留论文核心模型 DKUC/DBKN/IKN/IBKN，删除未接入新链路的 KAN、LSTM 和 Koopformer 实验代码。
5. 增加执行器类型、力矩限幅、重力补偿、数据时序和闭环控制测试。
