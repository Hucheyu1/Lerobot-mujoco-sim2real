# UR5e Torque Koopman

这是从提交 `b34bba601887ee5424af942e65ac315972049873` 重新整理的 UR5e 分支。项目保留原来的“MuJoCo 环境 → 轨迹数据采集 → Koopman 模型训练/测试 → 闭环控制”逻辑，但已完全移除 SOARM101 模型、舵机/串口、ZMQ 真机桥接、旧权重、历史结果和视频。

## 控制语义

- 机器人：MuJoCo Menagerie UR5e 简化动力学模型。
- 状态/网络输入：`x=[p_ee,q,dq]∈R^15`，其中末端笛卡尔坐标 3 维、关节角 6 维、关节速度 6 维，单位分别为 m、rad、rad/s。前三维布局与原 SOARM101 保持一致；保留 `dq` 是因为力矩控制系统需要速度才能构成 Markov 状态。
- 模型控制输入：`u≡τ_applied∈R^6`，表示六个电机实际施加的完整关节力矩，文件中单位为 N·m，进入网络后按各关节额定力矩归一化。
- 执行器：六个 MuJoCo `motor`，额定限制为 `[150,150,150,28,28,28]` N·m。
- 实际执行：`τ_applied=clip(τ_requested,-τ_rated,τ_rated)`。环境不再暗中叠加重力、位置或速度控制项；重力补偿必须显式包含在控制器输出的完整力矩中。
- 仿真频率：物理步长 0.002 s，每次控制执行 10 个物理步，即 50 Hz。

注意：这是 dynamics-enabled 的简化仿真模型，不是 Universal Robots 官方仿真器，也不是经硬件辨识验证的数字孪生。模型来源与许可证见 `assets/ur5e/PROVENANCE.md`。

## 目录

```text
UR5e/
  UR5e_Env.py               6 维直接力矩 Gymnasium 环境
  UR5e_DataCollection.py    闭环计算力矩轨迹采集和 DataLoader
assets/ur5e/                Menagerie 来源文件和直接 motor 派生 MJCF
models/                     DKUC、DBKN、IKN、IBKN
control/
  TorqueController.py       逆动力学前馈 + PD 完整力矩基线
  MPC_Controler.py          基于 A/B/H/C 的完整力矩增量/标准 MPC
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
New-Item -ItemType Directory -Force -Path runs | Out-Null
python -m pytest -q -p no:cacheprovider --basetemp runs/pytest_tmp
```

测试覆盖环境状态顺序、末端坐标、逆动力学、21 维数据布局、旧数据版本隔离、长轨迹安全性、四种网络、损失权重、双线性矩阵展开、PD 和标准/增量 MPC 力矩约束。测试必须全部通过后才能开始正式数据采集。

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

正式默认配置与原 SOARM101 数据布局对齐：训练集 50,000 条短轨迹，每条 20 次状态转移；验证集 2,000 条、每类测试集 2,000 条，每条 200 次状态转移。训练集共包含 1,000,000 次状态转移，强调独立初始状态覆盖。验证和测试的 4 s 长轨迹不再由开环力矩直接驱动，而是参考 `Adaptive-koopman-main` 的机械臂采集方法，采用“随机关节路点 → 三次样条 → 计算力矩跟踪”的闭环采集。`random`、`sin` 和 `chirp` 只表示叠加在跟踪力矩上的小幅辨识激励类型。

正式采集默认参数如下：

- 初始关节角从 `HOME±0.50 rad` 均匀采样，初始关节速度从 `±0.05 rad/s` 采样。
- 每条轨迹使用 10 个路点；相邻路点由 `±0.07 rad/s` 的随机路点速度积分得到，并限制在关节安全范围内。
- 三次样条同时生成 `q_ref`、`dq_ref` 和 `ddq_ref`。
- 计算力矩控制采用 `Kp=16`、`Kd=8`，期望加速度逐关节限制为 `±4 rad/s²`。
- MuJoCo 逆动力学直接计算完整力矩 `τ_id=M(q)ddq_cmd+bias(q,dq)-τ_passive`；叠加辨识激励后按额定力矩裁剪，不再减去任何隐藏的重力前馈。
- 辨识激励最大幅值为额定力矩的 1%，即前三关节不超过 `±1.5 N·m`、后三关节不超过 `±0.28 N·m`。random 默认每个 0.02 s 控制步更新一次；sin/chirp 的初始频率为 0.15–0.75 Hz，chirp 在单条轨迹内额外扫频 0.9 Hz。
- `.npy` 保存的是裁剪后实际施加的完整关节力矩 `τ_applied`，不是未裁剪控制器输出、力矩增量或残差力矩。

```powershell
python train.py --mode collect --seed 42 --train-samples 50000 --train-steps 20 --val-samples 2000 --test-samples 2000 --test-steps 200 --force-data
```

也可以显式覆盖采集器参数，例如：

```powershell
python train.py --mode collect --seed 42 --force-data --initial-position-span 0.50 --initial-velocity-span 0.05 --waypoint-count 10 --waypoint-velocity-limit 0.07 --tracking-kp 16 --tracking-kd 8 --tracking-acceleration-limit 4 --excitation-fraction 0.01 --random-hold-steps 1
```

数据采集算法和动作定义都已改变，新数据集版本为 `ur5e_full_joint_torque_v1`。此前使用开环力矩或残差力矩生成的 `datasets/ur5e_torque/train.npy` 等文件不能继续使用；首次运行必须带 `--force-data`。如果数据没有 manifest、生成中断、参数不一致或 shape 不符，程序会明确拒绝复用，避免混合不同输入语义的数据。

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

可以用以下命令确认数据维度。每行应为 21 维：`[τ_applied(6),p_ee(3),q(6),dq(6)]`。

```powershell
python -c "import numpy as np; from pathlib import Path; p=Path('datasets/ur5e_torque'); print({f.name: np.load(f).shape for f in p.glob('*.npy')})"
Get-Content datasets/ur5e_torque/manifest.json
```

`manifest.json` 的 `status` 必须为 `complete`，并记录每个 split 的接受率、拒绝原因、力矩饱和率、关节参考跟踪 RMSE/最大误差以及动作/状态逐维最小值和最大值。若采集中断，manifest 会保留为 `generating`，必须使用 `--force-data` 从头重建全部 split。

### 5. 训练四种 Koopman 模型

四种模型使用同一数据、训练轮数、学习率、batch size、多步预测长度和随机种子：

```powershell
python train.py --model all --mode train --seed 42 --device cuda --num-epochs 500 --lr 0.0005 --batch-size 256 --eval-batch-size 256 --pre-length 10 --gamma 0.98 --loss-name mse
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
model_manifest.json 数据版本、完整力矩输入定义和额定力矩
history.json        每轮训练损失和定期验证指标
```

测试和 Koopman MPC 都会校验 `model_manifest.json`。没有该文件或仍标记为残差力矩输入的旧 checkpoint 会被拒绝，不能因为网络张量形状相同而直接复用。

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

输出包含真实状态、参考状态、实际完整关节力矩和控制指标。无图形界面的服务器不要使用 `--render`。

### 8. 运行 Koopman MPC 控制实验

IBKN 增量 MPC（正式默认方式，因此可以省略 `--MPC-type delta_mpc`）：

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

MPC 的输入是当前 `x_t=[p_ee,q,dq]` 和未来 `H` 步参考轨迹，输出是 6 维完整关节力矩。默认增量 MPC 优化归一化力矩增量 `Δτ_k`，递推 `τ_k=τ_(k-1)+Δτ_k`；初始 `τ_(k-1)` 由当前状态的逆动力学平衡力矩给出。程序从网络读取 `A/B/H/C`，在当前状态处冻结 `B_total(z0)=B+sum(z0_j H_j)` 后求解有限时域控制问题。

### 9. 查看控制结果

```powershell
python -c "import numpy as np; d=np.load('runs/ur5e_torque/IBKN/mpc_delta.npz', allow_pickle=True); print(d['metrics'].item()); print('states:', d['states'].shape); print('references:', d['references'].shape); print('torques:', d['joint_torques'].shape)"
```

控制实验至少检查：

- `ee_rmse_m` 和 `q_rmse_rad`；
- 完整关节力矩是否超过 `[150,150,150,28,28,28]` N·m；
- 增量 MPC 相邻控制步变化是否超过额定力矩的 `0.015`，即 `[2.25,2.25,2.25,0.42,0.42,0.42]` N·m/控制步；
- 是否出现 safety termination；
- `solver_backend` 实际使用 `casadi-ipopt` 还是 `scipy-slsqp`；
- MPC 是否长期饱和。未充分训练的模型即使没有违反约束，也不能据此认定控制有效。

### 10. 多随机种子正式实验

论文结果至少使用 5 个随机种子。当前程序的固定输出目录会被下一次训练覆盖，因此每个 seed 测试完成后必须立即归档模型、指标和对应数据：

```powershell
$seeds = 42,43,44,45,46
foreach ($seed in $seeds) {
    python train.py --mode collect --seed $seed --train-samples 50000 --train-steps 20 --val-samples 2000 --test-samples 2000 --test-steps 200 --force-data
    if ($LASTEXITCODE -ne 0) { throw "seed $seed collection failed" }

    python train.py --model all --mode train --seed $seed --device cuda --num-epochs 500 --batch-size 256 --eval-batch-size 256 --pre-length 10
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

保存的 `.npy` 每行 21 维，保持物理单位 `[τ_applied_Nm(6),p_ee_m(3),q_rad(6),dq_rad_s(6)]`；输入 DataLoader 后只将完整关节力矩除以各关节额定力矩，网络接口名称 `u` 保留用于兼容，但物理含义固定为 `τ_applied/τ_rated`。力矩 `τ_t` 作用于同一行状态 `x_t` 并产生 `x_{t+1}`。参考轨迹只用于安全地生成有激励的数据，不作为网络输入。训练采用原项目的多步开环形式，物理预测损失为 `L_ee + 4 L_q + L_dq`，并叠加 Koopman 潜空间一致性损失和稳定性正则项。

`--smoke` 结果、单 seed 结果和短时域 MPC 结果只用于调试。正式论文结论应来自 `plan.md` 中规定的多 seed、等训练预算、结构消融、扰动控制和不完美观测实验。

## 与旧仓库的差异

1. 删除 SOARM101 的 5 维速度伺服环境，但保留其“末端位置优先”的状态布局、分区加权损失以及矩阵 MPC 架构。
2. 删除 SOARM101 真机串口、ZMQ、位置命令回放及所有旧平台结果。
3. 将数据状态统一为 `[p_ee,q,dq]`，动作统一为六个电机实际施加的完整关节力矩 `τ_applied`。
4. 只保留论文核心模型 DKUC/DBKN/IKN/IBKN，删除未接入新链路的 KAN、LSTM 和 Koopformer 实验代码。
5. 增加执行器类型、完整力矩限幅、显式逆动力学补偿、数据时序和增量 MPC 控制测试。
