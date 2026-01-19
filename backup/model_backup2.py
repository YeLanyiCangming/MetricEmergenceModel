"""
演化模型 - 度规涌现架构

核心范式（第一性原理）：
    数据 (x) → 抽象状态 (z) → 度规 g(z) → 联络 Γ (涌现) → 运动法则 (涌现)

三层架构：
    1. 感知层：学习流形的"本体"——度规张量
    2. 涌现层：从度规自动推导联络（Christoffel符号）
    3. 法则层：测地线加速度 + 外力 → 完整运动

不硬编码规则，但硬编码几何原理（微分几何的语言）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 基础组件
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


# =============================================================================
# 感知层：数据 → 抽象状态 → 度规张量
# =============================================================================

class StateEncoder(nn.Module):
    """
    状态编码器：将原始数据映射到抽象广义坐标 z
    
    z 是 AI 感知到的"你在哪"的内在表示
    """
    def __init__(self, input_dim, z_dim):
        super().__init__()
        self.z_dim = z_dim
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, z_dim * 2)
        
        # 序列编码（简单的因果卷积）
        self.conv = nn.Conv1d(z_dim * 2, z_dim, kernel_size=3, padding=2)
        
        # 输出投影
        self.out_proj = nn.Linear(z_dim, z_dim)
    
    def forward(self, x):
        """
        x: [B, L] 或 [B, L, input_dim] 输入序列
        返回: z [B, L, z_dim] 广义坐标
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, L, 1]
        
        # 投影
        h = self.input_proj(x)  # [B, L, z_dim*2]
        
        # 因果卷积
        h = h.transpose(1, 2)  # [B, z_dim*2, L]
        h = self.conv(h)[:, :, :x.shape[1]]  # [B, z_dim, L]
        h = h.transpose(1, 2)  # [B, L, z_dim]
        
        # 输出
        z = self.out_proj(F.silu(h))
        
        return z


class MetricEncoder(nn.Module):
    """
    度规编码器：从 z 生成度规张量 g(z)
    
    度规 g_ij 是流形上最基本的几何对象：
    - 定义距离：ds² = g_ij dz^i dz^j
    - 定义角度、体积
    - 联络、曲率都从度规推导
    
    结构性归纳偏置：
    - 输出下三角矩阵 L
    - g = L @ L.T 确保对称正定
    """
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        
        # 下三角矩阵的参数数量：z_dim * (z_dim + 1) / 2
        self.n_params = z_dim * (z_dim + 1) // 2
        
        # 生成下三角矩阵的参数
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim * 2),
            nn.SiLU(),
            nn.Linear(z_dim * 2, self.n_params),
        )
        
        # 对角线的最小值（确保正定）
        self.diag_min = 0.1
    
    def forward(self, z):
        """
        z: [B, L, z_dim] 广义坐标
        返回: g [B, L, z_dim, z_dim] 度规张量（对称正定）
        """
        B, L, D = z.shape
        
        # 生成下三角矩阵的参数
        params = self.net(z)  # [B, L, n_params]
        
        # 构造下三角矩阵 L
        L = torch.zeros(B, L, D, D, device=z.device, dtype=z.dtype)
        
        # 填充下三角
        idx = 0
        for i in range(D):
            for j in range(i + 1):
                if i == j:
                    # 对角线：使用 softplus 确保正数，加最小值确保正定
                    L[:, :, i, j] = F.softplus(params[:, :, idx]) + self.diag_min
                else:
                    L[:, :, i, j] = params[:, :, idx]
                idx += 1
        
        # g = L @ L.T（确保对称正定）
        g = torch.matmul(L, L.transpose(-2, -1))
        
        return g


# =============================================================================
# 涌现层：度规 → 联络（Christoffel 符号）
# =============================================================================

class ChristoffelComputer(nn.Module):
    """
    Christoffel 符号计算器
    
    第一性原理：在黎曼几何中，一旦定义了度规，联络就唯一确定
    
    公式：Γ^k_ij = 1/2 * g^kl * (∂g_jl/∂z^i + ∂g_il/∂z^j - ∂g_ij/∂z^l)
    
    这里我们使用自动微分计算 ∂g/∂z
    
    注意：这不是学习的，而是数学运算
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, z, g, metric_encoder):
        """
        z: [B, L, D] 广义坐标
        g: [B, L, D, D] 度规张量
        metric_encoder: 度规编码器（用于计算导数）
        
        返回: Gamma [B, L, D, D, D] Christoffel 符号 Γ^k_ij
        """
        B, L, D = z.shape
        device = z.device
        dtype = z.dtype
        
        # 计算度规的逆 g^{kl}
        g_inv = torch.linalg.inv(g)  # [B, L, D, D]
        
        # 计算 ∂g_ij/∂z^l
        # 使用有限差分近似（更稳定）
        eps = 1e-4
        dg_dz = torch.zeros(B, L, D, D, D, device=device, dtype=dtype)
        
        for l in range(D):
            # z + eps * e_l
            z_plus = z.clone()
            z_plus[:, :, l] = z_plus[:, :, l] + eps
            g_plus = metric_encoder(z_plus)
            
            # z - eps * e_l
            z_minus = z.clone()
            z_minus[:, :, l] = z_minus[:, :, l] - eps
            g_minus = metric_encoder(z_minus)
            
            # 中心差分
            dg_dz[:, :, :, :, l] = (g_plus - g_minus) / (2 * eps)
        
        # 计算 Christoffel 符号
        # Γ^k_ij = 1/2 * g^kl * (∂g_jl/∂z^i + ∂g_il/∂z^j - ∂g_ij/∂z^l)
        Gamma = torch.zeros(B, L, D, D, D, device=device, dtype=dtype)
        
        for k in range(D):
            for i in range(D):
                for j in range(D):
                    for l in range(D):
                        Gamma[:, :, k, i, j] += 0.5 * g_inv[:, :, k, l] * (
                            dg_dz[:, :, j, l, i] +  # ∂g_jl/∂z^i
                            dg_dz[:, :, i, l, j] -  # ∂g_il/∂z^j
                            dg_dz[:, :, i, j, l]    # ∂g_ij/∂z^l
                        )
        
        return Gamma


# =============================================================================
# 法则层：联络 → 运动法则
# =============================================================================

class GeodesicAcceleration(nn.Module):
    """
    测地线加速度计算
    
    第一性原理：在没有外力的弯曲空间中，粒子沿测地线运动
    
    测地线方程：d²z^k/dt² + Γ^k_ij * dz^i/dt * dz^j/dt = 0
    
    因此，纯几何导致的加速度：a_geo^k = -Γ^k_ij * dz^i * dz^j
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, dz, Gamma):
        """
        dz: [B, L, D] 速度（一阶导数）
        Gamma: [B, L, D, D, D] Christoffel 符号
        
        返回: a_geo [B, L, D] 测地线加速度
        """
        B, L, D = dz.shape
        
        # a_geo^k = -Γ^k_ij * dz^i * dz^j
        a_geo = torch.zeros(B, L, D, device=dz.device, dtype=dz.dtype)
        
        for k in range(D):
            for i in range(D):
                for j in range(D):
                    a_geo[:, :, k] -= Gamma[:, :, k, i, j] * dz[:, :, i] * dz[:, :, j]
        
        return a_geo


class ExternalForce(nn.Module):
    """
    外力网络
    
    第一性原理：如果观测到的运动偏离测地线，偏离部分就是"外力"
    
    柔软的归纳偏置：
    - 网络结构简单（奥卡姆剃刀）
    - 正则化鼓励 F_external → 0（优先用几何解释）
    """
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        
        # 简单的网络：z, dz → F
        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim * 2),
            nn.Tanh(),  # 限制输出范围
            nn.Linear(z_dim * 2, z_dim),
        )
        
        # 缩放因子（控制外力的初始大小）
        self.scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, z, dz):
        """
        z: [B, L, D] 位置
        dz: [B, L, D] 速度
        
        返回: F [B, L, D] 外力
        """
        h = torch.cat([z, dz], dim=-1)
        F = self.net(h) * self.scale
        return F


# =============================================================================
# 解码器：z 空间的加速度 → 原始空间的变化分布
# =============================================================================

class ProbabilisticDecoder(nn.Module):
    """
    概率解码器：输出分布而不是点估计
    
    第一性原理：
        - 世界不是确定性的，我们的知识也不是
        - 输出分布可以量化"无知"
        - 当模型不确定时，它应该说"我不确定"
    
    输出：
        - μ: 均值（最可能的 d²x）
        - σ: 标准差（不确定性）
    
    损失：
        - 负对数似然 (NLL)，而不是 MSE
    """
    def __init__(self, z_dim, output_dim=1):
        super().__init__()
        
        # 均值网络
        self.mu_net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.SiLU(),
            nn.Linear(z_dim, output_dim),
        )
        
        # 方差网络（输出 log_var，确保正定）
        self.logvar_net = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.SiLU(),
            nn.Linear(z_dim, output_dim),
        )
        
        # 方差的最小值（防止数值不稳定）
        self.min_var = 1e-4
    
    def forward(self, d2z):
        """
        d2z: [B, z_dim] z空间的加速度
        返回: 
            mu: [B] 均值
            sigma: [B] 标准差
        """
        mu = self.mu_net(d2z).squeeze(-1)
        log_var = self.logvar_net(d2z).squeeze(-1)
        
        # 标准差 = sqrt(exp(log_var)) = exp(log_var / 2)
        sigma = torch.exp(0.5 * log_var) + self.min_var
        
        return mu, sigma
    
    def nll_loss(self, mu, sigma, target):
        """
        负对数似然损失
        
        对于高斯分布: NLL = 0.5 * (log(2πσ²) + (x-μ)²/σ²)
        """
        var = sigma ** 2
        nll = 0.5 * (torch.log(2 * 3.14159 * var) + (target - mu) ** 2 / var)
        return nll.mean()
    
    def sample(self, mu, sigma):
        """从分布中采样"""
        eps = torch.randn_like(mu)
        return mu + sigma * eps


# =============================================================================
# 完整模型：度规涌现演化
# =============================================================================

class MetricEvolutionModel(nn.Module):
    """
    度规涌现演化模型
    
    核心链条：
        1. 感知层：x → z → g(z)
        2. 涌现层：g(z) → Γ (自动微分)
        3. 法则层：Γ, dz → a_geodesic + F_external → d²z
        4. 概率层：d²z → P(d²x | μ, σ)
    
    第一性原理：
        - 度规是流形的DNA
        - 联络从度规涌现
        - 运动法则从联络涌现
        - 外力是几何无法解释的残差
        - 不确定性是知识的一部分
    """
    
    def __init__(self, z_dim=4, input_dim=1):
        super().__init__()
        self.z_dim = z_dim
        
        # 1. 感知层
        self.state_encoder = StateEncoder(input_dim, z_dim)
        self.metric_encoder = MetricEncoder(z_dim)
        
        # 2. 涌现层
        self.christoffel = ChristoffelComputer()
        self.geodesic = GeodesicAcceleration()
        
        # 3. 法则层
        self.external_force = ExternalForce(z_dim)
        
        # 4. 概率解码层
        self.decoder = ProbabilisticDecoder(z_dim, 1)
    
    def forward(self, values, compute_christoffel=True):
        """
        values: [B, L] 输入序列
        
        返回: 完整的演化信息，包括分布参数
        """
        B, L = values.shape
        
        # 计算微分结构
        dx = values[:, 1:] - values[:, :-1]
        x_last = values[:, -1]
        dx_last = dx[:, -1]
        
        # 1. 感知层：x → z
        z = self.state_encoder(values)
        
        # 2. 感知层：z → g(z)
        g = self.metric_encoder(z)
        
        # 计算 z 的速度 dz
        dz = z[:, 1:] - z[:, :-1]
        dz_last = dz[:, -1]
        z_last = z[:, -1]
        
        # 3. 涌现层：g → Γ
        if compute_christoffel:
            z_for_gamma = z_last.unsqueeze(1)
            g_for_gamma = g[:, -1:, :, :]
            
            Gamma = self.christoffel(z_for_gamma, g_for_gamma, self.metric_encoder)
            Gamma = Gamma.squeeze(1)
            
            # 4. 法则层：测地线加速度
            dz_for_geo = dz_last.unsqueeze(1)
            a_geo = self.geodesic(dz_for_geo, Gamma.unsqueeze(1))
            a_geo = a_geo.squeeze(1)
        else:
            Gamma = None
            a_geo = torch.zeros(B, self.z_dim, device=values.device)
        
        # 5. 法则层：外力
        F_ext = self.external_force(z_last, dz_last)
        
        # 6. 总加速度
        d2z = a_geo + F_ext
        
        # 7. 概率解码：输出分布参数
        mu, sigma = self.decoder(d2z)
        
        # 8. 采样或用均值作为预测
        pred_d2x = mu  # 用均值作为点估计
        
        # 9. 重建下一个位置
        x_new = x_last + dx_last + pred_d2x
        
        return {
            'x_new': x_new,
            'pred_d2x': pred_d2x,
            'mu': mu,           # 分布均值
            'sigma': sigma,     # 分布标准差（不确定性）
            'x_last': x_last,
            'dx_last': dx_last,
            'z': z,
            'g': g,
            'Gamma': Gamma,
            'a_geodesic': a_geo,
            'F_external': F_ext,
            'd2z': d2z,
        }
    
    def compute_loss(self, values, target=None):
        """
        计算损失
        
        损失组成：
        1. NLL损失：负对数似然，而不是 MSE
        2. 外力正则化：奥卡姆剃刀
        3. 度规正则化：鼓励简洁
        """
        out = self.forward(values)
        
        if target is None:
            target = values[:, -1]
        
        # 真实的 d²x
        d2x_true = target - out['x_last'] - out['dx_last']
        
        # 1. NLL损失（负对数似然）
        nll_loss = self.decoder.nll_loss(out['mu'], out['sigma'], d2x_true)
        
        # 2. 外力正则化
        force_reg = (out['F_external'] ** 2).mean()
        
        # 3. 不约束度规！让数据决定 g(z) 的形状
        # 如果数据本身是欧几里得的，模型自然会学到 g(z) ~ I
        
        # 总损失
        total_loss = nll_loss + 0.01 * force_reg
        
        return {
            'loss': total_loss,
            'nll_loss': nll_loss,
            'force_reg': force_reg,
            'x_new': out['x_new'],
            'pred_d2x': out['pred_d2x'],
            'd2x_true': d2x_true,
            'mu': out['mu'],
            'sigma': out['sigma'],
            'a_geodesic': out['a_geodesic'],
            'F_external': out['F_external'],
        }
    
    def predict_sequence(self, values, steps=5, use_sampling=False):
        """
        预测未来序列
        
        use_sampling: 是否从分布中采样，还是用均值
        """
        result = values.clone()
        uncertainties = []  # 收集每步的不确定性
        
        for _ in range(steps):
            out = self.forward(result, compute_christoffel=True)
            
            if use_sampling:
                next_d2x = self.decoder.sample(out['mu'], out['sigma'])
            else:
                next_d2x = out['mu']
            
            next_val = out['x_last'] + out['dx_last'] + next_d2x
            next_val = next_val.unsqueeze(1)
            result = torch.cat([result, next_val], dim=1)
            uncertainties.append(out['sigma'].item())
        
        return result, uncertainties
    
    def get_uncertainty(self, values):
        """获取预测的不确定性"""
        out = self.forward(values)
        return {
            'mu': out['mu'],
            'sigma': out['sigma'],
            'confidence_interval_95': (out['mu'] - 1.96 * out['sigma'], 
                                        out['mu'] + 1.96 * out['sigma']),
        }


# =============================================================================
# 简化版本：用于快速验证
# =============================================================================

class EvolutionForSequence(nn.Module):
    """
    简化版本的度规演化模型
    
    为了快速验证，简化 Christoffel 计算
    """
    
    def __init__(self, d=32, num_layers=1, num_heads=4, input_dim=1, z_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.d = d
        
        # 使用完整的度规演化模型
        self.model = MetricEvolutionModel(z_dim=z_dim, input_dim=input_dim)
    
    def forward(self, values):
        return self.model.forward(values)
    
    def compute_loss(self, values, target=None):
        return self.model.compute_loss(values, target)
    
    def predict_sequence(self, values, steps=5):
        return self.model.predict_sequence(values, steps)


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("度规涌现演化模型测试")
    print("=" * 60)
    
    model = MetricEvolutionModel(z_dim=4)
    
    total = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total:,}")
    
    # 测试前向传播
    x = torch.rand(2, 10)
    out = model(x)
    
    print(f"\n输入: {x.shape}")
    print(f"z: {out['z'].shape}")
    print(f"g(z): {out['g'].shape}")
    print(f"Gamma: {out['Gamma'].shape if out['Gamma'] is not None else 'None'}")
    print(f"a_geodesic: {out['a_geodesic'].shape}")
    print(f"F_external: {out['F_external'].shape}")
    print(f"pred_d2x: {out['pred_d2x'].shape}")
    print(f"x_new: {out['x_new'].shape}")
    
    # 测试损失
    loss_dict = model.compute_loss(x, x[:, -1])
    print(f"\n损失:")
    print(f"  total: {loss_dict['loss'].item():.6f}")
    print(f"  pred: {loss_dict['pred_loss'].item():.6f}")
    print(f"  force_reg: {loss_dict['force_reg'].item():.6f}")
    print(f"  metric_reg: {loss_dict['metric_reg'].item():.6f}")
