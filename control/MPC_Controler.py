import casadi as ca
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
        self.obs_num = 8  #  可观测量 3位置，8位置与角度
        self.x_dim = args.x_dim  #  可观测量 3位置，8位置与角度
        self.Ad = net.lA.weight.cpu().detach().numpy() # (32, 32)
        self.Bd = net.lB.weight.cpu().detach().numpy() # (32, 5)
        self.Nkoopman = self.Ad.shape[0]
        if hasattr(net, 'H'):  # 检查是否存在该层
            self.H_hat_list = net.get_Hi_numpy()
            self.Hd = net.H.weight.cpu().detach().numpy() # (32, 224)
        # self.Cd = net.lC.weight.cpu().detach().numpy() # (32, 5)
        self.B_d_pinv = np.linalg.pinv(self.Bd)

        # ==== MPC 参数 ====
        Q = np.diag([50, 50, 50, 1, 1, 1, 1, 1])
        R = 0.5 * np.eye(self.u_dim) # 控制输入的惩罚矩阵
        self.H = 10  # 预测步长
        self.Q = ca.DM(Q)  # 状态误差权重
        self.R = ca.DM(R)# 控制输入权重
        self.u_eso = np.zeros(self.u_dim)
        self.startid = 0
        if hasattr(net, 'seq_len'):  # 用于时序网络
            self.startid = net.seq_len - 1

        # 构建MPC求解器
        self.u_prev = np.zeros(self.u_dim)
        self.MPC_type = args.MPC_type
        self.state_full = False
        if args.use_decoder or args.model == 'IBKN' or 'IKN':
            self.state_full = True
        if self.state_full:
            self.Q = ca.DM(50* np.eye(self.Nkoopman))  # 状态误差权重
            self.R = ca.DM(0.5 * np.eye(self.u_dim)) # 控制输入权重
        if args.MPC_type == 'mpc':
            self.solver = self.setup_mpc()
        if args.MPC_type == 'delta_mpc':
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
        if hasattr(self, 'H_hat_list'):
            for j in range(self.Nkoopman):
                H_j = ca.SX(self.H_hat_list[j])
                B_total += z0[j] * H_j
        return B_total
    
    def setup_mpc(self):
        """构建CasADi MPC求解器"""
        u = ca.SX.sym("u", self.H*self.u_dim)
        z0 = ca.SX.sym("z0", self.Nkoopman, 1)     # 初始状态
        B_total = self.linearize_B(z0)  # 使用 z0 构造一次线性近似 B 矩阵
        z = z0
        cost = 0
        if not self.state_full:
            ref = ca.SX.sym("ref", self.H, 10)      # 参考轨迹（位置3 + 关节角7）
            for t in range(self.H):
                u_t = u[t*self.u_dim:(t+1)*self.u_dim]
                z_next = ca.mtimes(self.Ad, z) + ca.mtimes(B_total, u_t)  # 加入扰动补偿
                ee_pos_pred = z_next[:self.x_dim]    
                cost += ca.mtimes([(ee_pos_pred - ref[t,:].T).T, self.Q, (ee_pos_pred - ref[t,:].T)]) + ca.mtimes([u_t.T, self.R, u_t])
                z = z_next
        else:
            ref = ca.SX.sym("ref", self.H, self.Nkoopman)  # 参考轨迹在Koopman空间 
            for t in range(self.H):
                u_t = u[t*self.u_dim:(t+1)*self.u_dim]
                z_next = ca.mtimes(self.Ad, z) + ca.mtimes(B_total, u_t)  # 加入扰动补偿
                cost += ca.mtimes([(z_next - ref[t,:].T).T, self.Q, (z_next - ref[t,:].T)]) + ca.mtimes([u_t.T, self.R, u_t])
                z = z_next        
        nlp = {
            'x': u,
            'f': cost,
            'p': ca.vertcat(ca.reshape(ca.transpose(ref), -1, 1), z0) # 加入扰动补偿
        }
        opts = {
            'ipopt.print_level': 0,  # 0=无输出，5=详细
            'print_time': False,
            'ipopt.sb': 'yes'       # 禁止初始标语
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
            delta_u_t = delta_u[t * self.u_dim : (t+1) * self.u_dim]
            u_t = u_t + delta_u_t
            z_next = ca.mtimes(self.Ad, z) + ca.mtimes(B_total, u_t)
            
            if not self.state_full:
                ee_pos_pred = z_next[:self.x_dim]
                cost += ca.mtimes([(ee_pos_pred - ref[t,:].T).T, self.Q, (ee_pos_pred - ref[t,:].T)])
            else:
                cost += ca.mtimes([(z_next - ref[t,:].T).T, self.Q, (z_next - ref[t,:].T)])
            
            cost += ca.mtimes([delta_u_t.T, self.R, delta_u_t])
            z = z_next

        nlp = {
            'x': delta_u,
            'f': cost,
            'p': ca.vertcat(
                ca.reshape(ca.transpose(ref), -1, 1),
                z0,
                u_prev
            )
        }
        
        opts = {'ipopt.print_level': 0, 'print_time': False, 'ipopt.sb': 'yes'}
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        return solver
    
    def get_control(self, p):
        # 求解MPC
        sol = self.solver(x0=np.zeros(self.H * self.u_dim), p=p)
        u_opt = np.array(sol['x']).reshape(self.H, self.u_dim) # (10,7)
        u0 = u_opt[0] + self.u_eso + self.u_prev# (7,) + (1, 7)
        # 记录控制量并返回
        a = np.clip(u0, -0.5, 0.5)
        if self.MPC_type == 'delta_mpc':
            self.u_prev = a.copy()
        return u0, a
    
    def Psi_o(self, s): # 只对状态 s 进行升维 Ψ(s(t_k))
          """Measurement mapping - 适配不同环境的状态维度"""
          # 确保输入是tensor
    
          # 进行编码
          ds = self.net.x_encoder(s.to(self.device)).detach().cpu().numpy()       
          psi = np.zeros([self.Nkoopman,1])
          if len(ds.shape) > 1 and ds.shape[0] == 1:
              psi[:self.Nkoopman,0] = ds[0]
          else:
              psi[:self.Nkoopman,0] = ds
          self.z0 = psi
          return psi
    
    def koopman_predict(self, x, u):
        x_t = torch.DoubleTensor(x.T).to(self.device)
        u_t = torch.DoubleTensor(u).to(self.device)
        x_next = self.net.koopman_operation(x_t, u_t)
        return x_next
    
class MPCController_UKF(MPCController):
    def __init__(self, net, args):
        """
        继承 MPC_KF 的初始化，包含 Koopman 模型、C矩阵、MPC solver、KF/UKF噪声协方差等。
        """
        super().__init__(net, args)  # 调用父类初始化
        n = self.Nkoopman
        # UKF 特有参数
        self.alpha = 1e-2    # UKF 参数: 控制 sigma 点分布宽度 1e-3 ~ 1
        self.beta = 2.0        # UKF 参数: 对高斯分布最优
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
        self.W = np.eye(self.Nkoopman) * 0.0001 #  (37,37) 过程噪声协方差 0.0001  0.1
        self.V = np.eye(self.obs_num) * 0.1   #  (3,3)  观测噪声协方差 0.01  10
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

        sigma_pts = np.hstack([z,
                            z + Psqrt,
                            z - Psqrt])  # (n, 2n+1) -> (37, 75)
        # 传播 sigma 点
        sigma_pts_pred = self.Ad @ sigma_pts + self.Bd @ u.reshape(-1, 1)  # (37, 75)
        if hasattr(self.net, 'H'):
            # 根据网络设置选择克罗内克积顺序
            if self.net.u_z:
                kron_product = np.kron(u.reshape(-1, 1), sigma_pts) # (224, 75)
            else:
                kron_product = np.kron(sigma_pts, u.reshape(-1, 1)) # (224, 75)
            # 添加双线性项
            sigma_pts_pred += self.Hd @ kron_product

        sigma_pts_pred = sigma_pts_pred.T  # (75, 37) 
        # 均值
        z_pred = np.sum(self.Wm[:, None] * sigma_pts_pred, axis=0) # (37,)
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
        sigma_tensor = torch.tensor(sigma_pts_pred, dtype=torch.float64, device=self.device) # (75, 37)
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
        Pxy = dz.T @ (self.Wc[:, None] * dy)           # (37, 3)

        # 卡尔曼增益 & 更新
        K = Pxy @ np.linalg.inv(Pyy)  # (37, 3) × (3, 3) → (37, 3)
        self.z_ += K @ (x_meas.reshape(-1) - y_pred).reshape(-1, 1)  # (37,) + (37,3)×(3,)→(37,)→(37,1)
        self.P -= K @ Pyy @ K.T # (37,37) - (37,3)×(3,3)×(3,37)→(37,37)

        return self.z_
    
    def get_updated_state(self, state_pre, z_last, a):
        state_full = self.update_kf(state_pre[:self.obs_num], z_last, a.reshape(-1,1)).T
        state = self.net.x_decoder(torch.DoubleTensor(state_full).to(self.device)).detach().cpu().numpy().reshape(-1) 
        return state[3:]