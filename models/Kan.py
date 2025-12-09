import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        # --- 1. 基本属性设置 ---
        self.in_features = in_features         # 输入特征数
        self.out_features = out_features       # 输出特征数
        self.grid_size = grid_size             # B-样条网格的区间数
        self.spline_order = spline_order       # B-样条的阶数 (k)，k=3表示三次样条

        # --- 2. 网格 (Grid) 初始化 ---
        # 网格定义了样条函数的节点位置。
        h = (grid_range[1] - grid_range[0]) / grid_size # 每个网格区间的宽度
        # 创建网格点。为了计算k阶B-样条，我们需要在grid_range两侧各扩展k个节点。
        # 总节点数 = grid_size + 1 (内部节点) + 2 * spline_order (外部扩展节点)
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1) # 将一维网格扩展到(in_features, num_knots)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # --- 3. 可学习参数 (Parameters) 初始化 ---
        # 3.1 基函数权重 (Symbolic part)
        # 对应论文中的 b(x) 部分，提供一个全局的、类似传统MLP的组件。
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # 3.2 样条函数权重 (Numeric part)
        # 这些是B-样条基函数的系数。每个连接(in_i -> out_j)都有一组系数。
        # 形状: (输出特征, 输入特征, 每个样条的系数个数)
        # 系数个数 = grid_size + spline_order
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        # 3.3 (可选) 样条缩放因子
        # 允许为每个样条函数（每个连接）学习一个独立的缩放因子，增强模型表达能力。
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        # --- 4. 其他超参数 ---
        self.scale_noise = scale_noise # 初始化样条权重时加入的噪声大小
        self.scale_base = scale_base   # 初始化基函数权重的缩放因子
        self.scale_spline = scale_spline # 初始化样条权重的缩放因子
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation() # 基函数的激活函数，默认为SiLU
        self.grid_eps = grid_eps       # 网格更新时，自适应网格与均匀网格的混合比例
        
        # --- 5. 初始化参数 ---
        self.reset_parameters()

    def reset_parameters(self):
        # 使用Kaiming均匀初始化基函数权重，这是一种标准做法。
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        
        # 初始化样条权重，这是一个复杂的过程。
        with torch.no_grad(): # 在这个块中不追踪梯度
            # 目标：生成一些随机的曲线，然后计算出拟合这些曲线的B-样条系数。
            # 1. 生成随机噪声作为目标曲线的y值。
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 0.5 # 中心化到0
                )
                * self.scale_noise
                / self.grid_size
            )
            # 2. 调用 curve2coeff，传入网格内部的节点(x)和噪声(y)，计算出样条系数。
            #    self.grid.T[k:-k] 提取出内部的 G+1 个网格点。
            coeffs = self.curve2coeff(
                self.grid.T[self.spline_order : -self.spline_order],
                noise,
            )
            # 3. 将计算出的系数赋给样条权重。
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * coeffs
            )
            # 4. 如果启用了独立的缩放因子，也对其进行初始化。
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        # --- 前向传播 ---
        # x 形状: (batch_size, in_features)
        
        # 1. 基函数部分 (Symbolic part)
        #    - 首先对输入x应用激活函数
        #    - 然后进行标准的线性变换
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 2. 样条函数部分 (Numeric part)
        #    - 计算B-样条基函数的值
        spline_basis = self.b_splines(x) # 形状: (batch, in, coeffs)
        #    - 将其展平以便进行一次线性计算
        spline_basis = spline_basis.view(x.size(0), -1) # 形状: (batch, in * coeffs)
        
        #    - 获取缩放后的样条权重，并同样展平
        scaled_weights = self.scaled_spline_weight # 形状: (out, in, coeffs)
        scaled_weights = scaled_weights.view(self.out_features, -1) # 形状: (out, in * coeffs)
        
        #    - 进行线性变换，这等价于对每个输入特征的样条输出进行加权求和
        spline_output = F.linear(spline_basis, scaled_weights)

        # 3. 组合输出 (batch_size, out_features)
        return base_output + spline_output  

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        """根据输入数据动态更新网格, 这是KAN实现自适应性的关键"""
        # 1. 计算当前网格和权重下的样条输出值（在网格更新前）
        #    这是为了在更新网格后，能找到新的系数来尽可能保持函数形状不变。
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # 2. 生成新的自适应网格
        #    - 对每个输入特征通道的数据进行排序
        x_sorted = torch.sort(x, dim=0)[0]
        #    - 根据数据分位数确定自适应网格点。这是核心思想：在数据密集的地方放置更多网格点。
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        # 3. 生成均匀网格作为参考，并与自适应网格混合
        #    - 这可以防止网格点过分聚集，保持一定的稳定性。
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """计算正则化损失, 鼓励稀疏性(让一些样条函数变为0)"""
        # 这个实现是对原论文正则化的一种简化。
        # 原论文的L1正则化是基于每个样本的激活值，计算成本高。
        # 这里直接对样条权重的大小进行正则化。
        
        # 1. 计算每个样条函数的重要性（L1范数）
        #    l1_fake 模拟了每个激活函数 phi_ij 的幅度
        l1_fake = self.spline_weight.abs().mean(-1)
        # 2. 激活正则化 (L1 Norm)
        #    鼓励网络使用更少的样条函数（稀疏性）
        regularization_loss_activation = l1_fake.sum()
        # 3. 熵正则化
        #    鼓励激活的重要性分布更均匀。这与L1正则化目标相反，但可以防止所有激活都集中在少数几个输入上。
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
