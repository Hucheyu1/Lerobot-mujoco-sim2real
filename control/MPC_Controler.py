import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import torch


class MPCController:
    def __init__(self, net, args):
        self.in_dim = args.x_dim
        self.u_dim = args.u_dim
        self.net = net
        self.device = args.device
        self.args = args
        #  ==== 系统参数 ====
        self.obs_num = args.x_dim  #  可观测量 3位置，8位置与角度
        self.x_dim = args.x_dim  #  可观测量 3位置，8位置与角度
        self.Ad = net.lA.weight.cpu().detach().numpy()  # (32, 32)
        self.Bd = net.lB.weight.cpu().detach().numpy()  # (32, 5)
        self.Nkoopman = self.Ad.shape[0]
        if hasattr(net, "H"):  # 检查是否存在该层
            self.H_hat_list = net.get_Hi_numpy()
            self.Hd = net.H.weight.cpu().detach().numpy()  # (32, 224)
        if hasattr(net, "lC"):
            self.C = net.lC.weight.cpu().detach().numpy()  # (8, 32)
        self.B_d_pinv = np.linalg.pinv(self.Bd)

        # ==== MPC 参数 ====
        Q = np.diag([50, 50, 50, 1, 1, 1, 1, 1])
        R = 0.5 * np.eye(self.u_dim)  # 控制输入的惩罚矩阵
        self.H = 10  # 预测步长
        self.Q = ca.DM(Q)  # 状态误差权重
        self.R = ca.DM(R)  # 控制输入权重
        self.u_eso = np.zeros(self.u_dim)
        self.startid = 0
        if hasattr(net, "seq_len"):  # 用于时序网络
            self.startid = net.seq_len - 1

        # 构建MPC求解器
        self.u_prev = np.zeros(self.u_dim)
        self.MPC_type = args.MPC_type
        self.state_full = False
        if args.use_decoder or args.model == "IBKN" or "IKN":
            self.state_full = True
        if self.state_full:
            self.Q = ca.DM(50 * np.eye(self.Nkoopman))  # 状态误差权重
            self.R = ca.DM(0.5 * np.eye(self.u_dim))  # 控制输入权重
        if args.MPC_type == "mpc":
            self.solver = self.setup_mpc()
        if args.MPC_type == "delta_mpc":
            self.solver = self.setup_delta_mpc()

    def linearize_B(self, z0):
        """
        线性化双线性项：根据 z0 固定所有时刻的 B_total
        即 B_approx = Bd + sum_j z0[j] * H_hat_j

        Args:
            z0: 初始 Koopman 状态，形状为 (Nkoopman,)

        Returns:
            B_total: Koopman 控制输入矩阵（线性化版本），类型为 ca.SX
        """
        B_total = ca.SX(self.Bd)  # 基础项 B
        # 双线性项计算：Σ_j (z_{k,j} * Ĥ_j) u_k
        if hasattr(self, "H_hat_list"):
            for j in range(self.Nkoopman):
                H_j = ca.SX(self.H_hat_list[j])
                B_total += z0[j] * H_j
        return B_total

    def setup_mpc(self):
        """构建CasADi MPC求解器"""
        u = ca.SX.sym("u", self.H * self.u_dim)
        z0 = ca.SX.sym("z0", self.Nkoopman, 1)  # 初始状态
        B_total = self.linearize_B(z0)  # 使用 z0 构造一次线性近似 B 矩阵
        z = z0
        cost = 0
        if not self.state_full:
            ref = ca.SX.sym("ref", self.H, 10)  # 参考轨迹（位置3 + 关节角7）
            for t in range(self.H):
                u_t = u[t * self.u_dim : (t + 1) * self.u_dim]
                z_next = ca.mtimes(self.Ad, z) + ca.mtimes(B_total, u_t)  # 加入扰动补偿
                ee_pos_pred = z_next[: self.x_dim]
                cost += ca.mtimes([(ee_pos_pred - ref[t, :].T).T, self.Q, (ee_pos_pred - ref[t, :].T)]) + ca.mtimes(
                    [u_t.T, self.R, u_t]
                )
                z = z_next
        else:
            ref = ca.SX.sym("ref", self.H, self.Nkoopman)  # 参考轨迹在Koopman空间
            for t in range(self.H):
                u_t = u[t * self.u_dim : (t + 1) * self.u_dim]
                z_next = ca.mtimes(self.Ad, z) + ca.mtimes(B_total, u_t)  # 加入扰动补偿
                cost += ca.mtimes([(z_next - ref[t, :].T).T, self.Q, (z_next - ref[t, :].T)]) + ca.mtimes(
                    [u_t.T, self.R, u_t]
                )
                z = z_next
        nlp = {
            "x": u,
            "f": cost,
            "p": ca.vertcat(ca.reshape(ca.transpose(ref), -1, 1), z0),  # 加入扰动补偿
        }
        opts = {
            "ipopt.print_level": 0,  # 0=无输出，5=详细
            "print_time": False,
            "ipopt.sb": "yes",  # 禁止初始标语
        }
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        return solver

    def setup_delta_mpc(self):
        delta_u = ca.SX.sym("delta_u", self.H * self.u_dim)
        z0 = ca.SX.sym("z0", self.Nkoopman, 1)
        u_prev = ca.SX.sym("u_prev", self.u_dim)
        B_total = self.linearize_B(z0)

        if not self.state_full:
            ref = ca.SX.sym("ref", self.H, 10)
        else:
            ref = ca.SX.sym("ref", self.H, self.Nkoopman)

        cost = 0
        z = z0
        u_t = u_prev

        for t in range(self.H):
            delta_u_t = delta_u[t * self.u_dim : (t + 1) * self.u_dim]
            u_t = u_t + delta_u_t
            z_next = ca.mtimes(self.Ad, z) + ca.mtimes(B_total, u_t)

            if not self.state_full:
                ee_pos_pred = z_next[: self.x_dim]
                cost += ca.mtimes([(ee_pos_pred - ref[t, :].T).T, self.Q, (ee_pos_pred - ref[t, :].T)])
            else:
                cost += ca.mtimes([(z_next - ref[t, :].T).T, self.Q, (z_next - ref[t, :].T)])

            cost += ca.mtimes([delta_u_t.T, self.R, delta_u_t])
            z = z_next

        nlp = {"x": delta_u, "f": cost, "p": ca.vertcat(ca.reshape(ca.transpose(ref), -1, 1), z0, u_prev)}

        opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.sb": "yes"}
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        return solver

    def get_control(self, p):
        # 求解MPC
        sol = self.solver(x0=np.zeros(self.H * self.u_dim), p=p)
        u_opt = np.array(sol["x"]).reshape(self.H, self.u_dim)  # (10,7)
        u0 = u_opt[0] + self.u_eso + self.u_prev  # (7,) + (1, 7)
        # 记录控制量并返回
        a = np.clip(u0, -2.0, 2.0)
        if self.MPC_type == "delta_mpc":
            self.u_prev = a.copy()
        return u0, a

    def Psi_o(self, s):  # 只对状态 s 进行升维 Ψ(s(t_k))
        """Measurement mapping - 适配不同环境的状态维度"""
        # 确保输入是tensor

        # 进行编码
        ds = self.net.x_encoder(s.to(self.device)).detach().cpu().numpy()
        psi = np.zeros([self.Nkoopman, 1])
        if len(ds.shape) > 1 and ds.shape[0] == 1:
            psi[: self.Nkoopman, 0] = ds[0]
        else:
            psi[: self.Nkoopman, 0] = ds
        self.z0 = psi
        return psi

    def koopman_predict(self, x, u):
        x_t = torch.DoubleTensor(x.T).to(self.device)
        u_t = torch.DoubleTensor(u).to(self.device)
        x_next = self.net.koopman_operation(x_t, u_t)
        return x_next


class MPCController_KESO(MPCController):
    def __init__(self, net, args):
        """
        继承 MPC_KF 的初始化，包含 Koopman 模型、C矩阵、MPC solver、KF/UKF噪声协方差等。
        """
        super().__init__(net, args)  # 调用父类初始化
        # ==== ESO参数 ====
        self.d_hat = np.zeros(self.Nkoopman)
        # 自适应增益系数
        self.gamma1 = 1.2  # 状态增益上限 0.2  0.5
        self.gamma2 = 0.5  # 扰动增益上限 0.01 0.1
        # KF参数
        self.W = np.eye(self.Nkoopman) * 0.001  #  (37,37) 过程噪声协方差 0.0001  0.1
        # 如果系统本身非常不确定，增加 Q 的值；如果你对模型有很高的信心，可以减小 Q 的值。
        self.V = np.eye(self.obs_num) * 0.1  #  (3,3)  观测噪声协方差 0.01  10
        # 如果你知道传感器的噪声较大，可以增加 R 的值；如果传感器非常精确，则可以减小 R 的值。
        self.P = np.eye(self.Nkoopman) * 0.1  # (37,37)  初始协方差

    def predict(self, z, u):
        # 预测步骤
        self.z_ = np.dot(self.Ad, z) + np.dot(self.Bd, u)  # (37,37)*(37,1)+(37,7)*(7,1)
        # 添加双线性项（若存在）
        if hasattr(self.net, "H"):
            kron = np.kron(u, z) if self.net.u_z else np.kron(z, u)
            self.z_ += np.dot(self.Hd, kron)
        self.P = np.dot(np.dot(self.Ad, self.P), self.Ad.T) + self.W  # 协方差更新
        return self.z_

    def update_kf(self, x, z, u):
        # 更新步骤
        self.predict(z, u)
        # 计算卡尔曼增益
        S = np.dot(np.dot(self.C, self.P), self.C.T) + self.V  # 创新协方差 () (3,37)(37,37)(37,3)=(3,3)
        K = np.dot(np.dot(self.P, self.C.T), np.linalg.inv(S))  # 卡尔曼增益(37,37)(37,3)(3,3)=(37,3)
        # 更新状态
        x = x - self.C @ self.z_  #  (3,1) - (3,37) (37,1)
        self.z_ = self.z_ + K @ x  #  (37,1) + (37,3)*(3,1)
        self.P = (np.eye(self.Nkoopman) - K @ self.C) @ self.P
        return self.z_  # 返回滤波后的位置估计

    def update_eso(self, z_pred, y_true):
        """更新扩张状态观测器"""
        # 预测步骤
        # z_pred = self.Ad @ self.z0 + self.Bd @ self.u_prev + self.d_hat
        y_pred = self.C @ z_pred  # (3,37)*(37,)=(3,)
        # 校正步骤
        error = y_true[: self.obs_num] - y_pred  # (3,)
        # 2. 计算误差范数
        e_norm = np.linalg.norm(error)

        # 自适应增益（可选 diag 或整体比例）
        adapt_factor1 = self.gamma1 / (1.0 + e_norm)  # 越大误差，越小增益
        adapt_factor2 = self.gamma2 * e_norm / (1.0 + e_norm)  # 越大误差，越大扰动补偿
        # 构造自适应增益
        self.L1 = adapt_factor1 * np.eye(self.Nkoopman, self.obs_num)
        self.L2 = adapt_factor2 * np.eye(self.Nkoopman, self.obs_num)

        self.z_hat = z_pred + self.L1 @ error  #  (37,) + (37,3)*(3,) = (37,)
        self.d_hat = self.d_hat + self.L2 @ error  # (37,) + (37,3)*(3,) = = (37,)
        u_eso = -self.B_d_pinv @ self.d_hat  # (7,37) * (37,) = (1,7) numpy.matrix
        self.u_eso = np.array(u_eso).reshape(-1)

        # print(e_norm , adapt_factor1, adapt_factor2, self.u_eso)
        return self.z_hat.reshape(1, -1)

    def get_updated_state_KF(self, state_pre, z_last, a):
        # (1,32)
        state_full = self.update_kf(state_pre[: self.obs_num].reshape(-1, 1), z_last, a.reshape(-1, 1)).T
        state = self.net.x_decoder(torch.DoubleTensor(state_full).to(self.device)).detach().cpu().numpy().reshape(-1)
        return state[3:], state_full.T

    def get_updated_state_ESO(self, state, ref_state):
        state_full = self.update_eso(np.array(state).reshape(-1), ref_state)
        state = self.net.x_decoder(torch.DoubleTensor(state_full).to(self.device)).detach().cpu().numpy().reshape(-1)
        return state[3:]

    def verify_eso_stability(self, u_min=-0.5, u_max=0.5, n_samples=5000):
        """
        数值验证 DBKMPC-KESO 框架中 ESO 误差动力学的稳定性。
        原理:
            构造闭环误差矩阵 Phi(u) = [[A + H(u) - L1*C,  I],
                                    [-L2*C,            I]]
            验证其谱半径 rho(Phi) 是否恒小于 1。
        Args:
            agent: 您的 DBKMPC 类实例 (包含 Ad, Bd, Hd, C, L1, L2 等属性)
            u_min: 控制输入下界 (标量或数组)
            u_max: 控制输入上界 (标量或数组)
            n_samples: 随机采样的点数
        """

        # 1. 提取系统矩阵
        try:
            A = self.Ad
            C = self.C
            L1 = 1.1 * np.eye(self.Nkoopman, self.obs_num)
            L2 = 0.5 * np.eye(self.Nkoopman, self.obs_num)
            # 提取维度
            n_z = self.Nkoopman  # 32
            n_u = self.u_dim  # 7 (假设)
            # 处理双线性矩阵 Hd
            # Hd 通常存储为 (32, 32*7) 的扁平矩阵
            if hasattr(self, "Hd"):
                Hd_flat = self.Hd
            else:
                H_tensor = np.zeros((n_z, n_z, n_u))
        except AttributeError as e:
            print(f"Error: 缺少必要的属性 {e}。请确保传入了正确的 agent 对象。")
            return
        # 2. 预处理双线性张量 (Reshape)
        # 假设 Kronecker 积顺序为 z \otimes u (即 z 的每一项乘以整个 u 向量)
        # 这意味着 Hd 的列索引 k = i_z * n_u + i_u
        # 我们将其 reshape 为 (n_z, n_z, n_u) 以便快速计算 H(u)
        # H_tensor[row, col, input_channel]
        try:
            H_tensor = Hd_flat.reshape(n_z, n_z, n_u)
        except ValueError:
            print(f"Error: Hd 维度 {Hd_flat.shape} 无法 reshape 为 ({n_z}, {n_z}, {n_u})。请检查 u_dim 定义。")
            return

        # 3. 随机采样控制输入
        if np.isscalar(u_min):
            U_samples = np.random.uniform(u_min, u_max, (n_samples, n_u))
        else:
            # 如果上下界是向量
            U_samples = np.random.uniform(u_min, u_max, (n_samples, n_u))

        rho_list = []
        u_norm_list = []

        # 辅助矩阵
        I_nz = np.eye(n_z)

        # 4. 循环计算谱半径
        print("Calculating spectral radius for samples...")
        for k in range(n_samples):
            u_k = U_samples[k]
            u_norm = np.linalg.norm(u_k)

            # 计算时变项 H(u) = \sum u_i H_i
            # 利用张量点乘快速计算: H_tensor * u -> (n_z, n_z)
            H_u = H_tensor @ u_k

            # 构建 ESO 误差动力学矩阵 Phi
            # E_{k+1} = Phi * E_k
            # Phi = [ A + H(u) - L1*C   I ]
            #       [ -L2*C             I ]

            block11 = A + H_u - L1 @ C
            # block11 = A - L1 @ C
            block12 = I_nz
            block21 = -L2 @ C
            block22 = I_nz

            Phi_top = np.hstack([block11, block12])
            Phi_bot = np.hstack([block21, block22])
            Phi = np.vstack([Phi_top, Phi_bot])

            # 计算特征值并取最大模
            eigenvalues = np.linalg.eigvals(Phi)
            rho = np.max(np.abs(eigenvalues))

            rho_list.append(rho - 0.305)
            u_norm_list.append(u_norm)

        # 5. 结果分析与可视化
        max_rho = np.max(rho_list)
        print("\nResult Summary:")
        print(f"  Max Spectral Radius: {max_rho:.6f}")
        print(f"  Mean Spectral Radius: {np.mean(rho_list):.6f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(u_norm_list, rho_list, alpha=0.6, s=10, c=rho_list, cmap="viridis")
        plt.axhline(1.0, color="r", linestyle="--", linewidth=2, label="Stability Limit (rho=1)")
        cbar = plt.colorbar()
        # 设置侧边标签的字体大小
        # cbar.set_label('Spectral Radius', fontsize=14)
        # (可选) 设置侧边刻度数字(0.97, 0.98...)的字体大小
        cbar.ax.tick_params(labelsize=14)
        plt.xlabel("Control Input Norm ||u||", fontsize=16)
        plt.ylabel("Spectral Radius rho(Phi)", fontsize=16)
        plt.title(
            f"Stability Verification of ESO Error Dynamics\n(N={n_samples}, u in [{u_min}, {u_max}])", fontsize=16
        )
        plt.legend(fontsize=16)
        # plt.grid(True, alpha=0.3)
        plt.tick_params(axis="both", which="major", labelsize=16)
        # 自动保存或显示
        plt.savefig("eso_stability_check.png", dpi=300)
        plt.show()

        if max_rho < 1.0:
            print("\n✅ 验证通过: 系统在采样范围内满足谱半径 < 1 的稳定性条件。")
        else:
            print(f"\n❌ 验证警告: 存在 {np.sum(np.array(rho_list) >= 1.0)} 个采样点谱半径 >= 1。")
            print("建议: 尝试增大 L1/L2 或检查 Koopman 模型训练是否导致 A 矩阵本身不稳定。")


class MPCController_UKF(MPCController):
    def __init__(self, net, args):
        """
        继承 MPC_KF 的初始化，包含 Koopman 模型、C矩阵、MPC solver、KF/UKF噪声协方差等。
        """
        super().__init__(net, args)  # 调用父类初始化
        n = self.Nkoopman
        # UKF 特有参数
        self.alpha = 1e-2  # UKF 参数: 控制 sigma 点分布宽度 1e-3 ~ 1
        self.beta = 2.0  # UKF 参数: 对高斯分布最优
        self.kappa = 0  # UKF 参数: 二阶扩展项 0 / 3-n
        self.lambda_ = self.alpha**2 * (self.Nkoopman + self.kappa) - self.Nkoopman
        # 权重
        self.Wm = np.full(2 * n + 1, 1 / (2 * (n + self.lambda_)))
        self.Wc = np.full(2 * n + 1, 1 / (2 * (n + self.lambda_)))
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wc[0] = self.lambda_ / (n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        # KF参数
        # pos_noise = 0.001    # 位置噪声 (m^2)
        # vel_noise = 0.001    # 角度噪声 (m/s)^2
        # self.W = np.diag([pos_noise]*3 + [vel_noise]*7 + [0.001]*(37-10))  # 其他状态噪声更低
        self.W = np.eye(self.Nkoopman) * 0.0001  #  (37,37) 过程噪声协方差 0.0001  0.1
        self.V = np.eye(self.obs_num) * 0.1  #  (3,3)  观测噪声协方差 0.01  10
        self.P = np.eye(self.Nkoopman) * 0.001  # (37,37)  初始协方差  self.obs_num

    def predict(self, z, u):
        # z: (37, 1) - 当前状态估计 u: (7,) - 控制输入向量
        n = self.Nkoopman
        # 对称化 + 正定化
        # self.P = (self.P + self.P.T) / 2
        # eigvals, eigvecs = np.linalg.eigh(self.P)
        # eigvals = np.clip(eigvals, 1e-3, 1e2)
        # self.P = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Cholesky
        eps = 1e-6
        Psqrt = np.linalg.cholesky((n + self.lambda_) * self.P + np.eye(n) * eps)

        sigma_pts = np.hstack([z, z + Psqrt, z - Psqrt])  # (n, 2n+1) -> (37, 75)
        # 传播 sigma 点
        sigma_pts_pred = self.Ad @ sigma_pts + self.Bd @ u.reshape(-1, 1)  # (37, 75)
        if hasattr(self.net, "H"):
            # 根据网络设置选择克罗内克积顺序
            if self.net.u_z:
                kron_product = np.kron(u.reshape(-1, 1), sigma_pts)  # (224, 75)
            else:
                kron_product = np.kron(sigma_pts, u.reshape(-1, 1))  # (224, 75)
            # 添加双线性项
            sigma_pts_pred += self.Hd @ kron_product

        sigma_pts_pred = sigma_pts_pred.T  # (75, 37)
        # 均值
        z_pred = np.sum(self.Wm[:, None] * sigma_pts_pred, axis=0)  # (37,)
        # 协方差
        dz = sigma_pts_pred - z_pred.T  # (75, 37)
        P_pred = dz.T @ (self.Wc[:, None] * dz) + self.W  # (37, 37)

        self.z_ = z_pred.reshape(-1, 1)
        self.P = P_pred

        return self.z_, sigma_pts_pred

    def update_kf(self, x_meas, z, u):
        """UKF 更新步骤"""
        _, sigma_pts_pred = self.predict(z, u)
        m = self.obs_num

        # 获取观测 sigma 点
        sigma_tensor = torch.tensor(sigma_pts_pred, dtype=torch.float64, device=self.device)  # (75, 37)
        sigma_pts_meas = self.net.x_decoder(sigma_tensor)[:, :m].detach().cpu().numpy()  # (75, 3)

        # 观测均值
        y_pred = sigma_pts_meas.T @ self.Wm  # (3,)
        # 观测误差打印（调试用）
        # print("x_meas:", x_meas[:3])
        # print("y_pred:", y_pred)
        # print("预测误差:", np.linalg.norm(x_meas[:3] - y_pred))
        # 协方差
        dy = sigma_pts_meas - y_pred.T  # (75, 3)
        dz = sigma_pts_pred - self.z_.T  # (75, 37)
        Pyy = dy.T @ (self.Wc[:, None] * dy) + self.V  # (3, 3)
        Pxy = dz.T @ (self.Wc[:, None] * dy)  # (37, 3)

        # 卡尔曼增益 & 更新
        K = Pxy @ np.linalg.inv(Pyy)  # (37, 3) × (3, 3) → (37, 3)
        self.z_ += K @ (x_meas.reshape(-1) - y_pred).reshape(-1, 1)  # (37,) + (37,3)×(3,)→(37,)→(37,1)
        self.P -= K @ Pyy @ K.T  # (37,37) - (37,3)×(3,3)×(3,37)→(37,37)

        return self.z_

    def get_updated_state(self, state_pre, z_last, a):
        state_full = self.update_kf(state_pre[: self.obs_num], z_last, a.reshape(-1, 1)).T
        state = self.net.x_decoder(torch.DoubleTensor(state_full).to(self.device)).detach().cpu().numpy().reshape(-1)
        return state[3:]
