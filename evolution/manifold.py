"""可微流形与神经微分方程

核心数学结构：
1. 可微流形 M - Z空间的几何结构
2. 切空间 TM - 速度/一阶变化
3. 度量张量 g - 定义距离与内积
4. Neural ODE - 连续时间演化

关键特性：
- 流形参数化：通过指数映射保持流形约束
- 连续演化：ODE求解器实现平滑轨迹
- 几何先验：度量学习捕捉数据内在结构
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Protocol
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 代数数据类型定义
# =============================================================================

@dataclass(frozen=True)
class ManifoldPoint:
    """流形上的点（不可变）"""
    position: torch.Tensor      # 位置 z ∈ M
    velocity: torch.Tensor      # 速度 v ∈ T_z M
    metric: torch.Tensor        # 局部度量 g_z
    
    def evolve(self, dt: float, acceleration: torch.Tensor) -> 'ManifoldPoint':
        """沿测地线演化"""
        new_velocity = self.velocity + dt * acceleration
        new_position = self.position + dt * new_velocity
        return ManifoldPoint(new_position, new_velocity, self.metric)


@dataclass(frozen=True)
class TangentVector:
    """切向量"""
    base_point: torch.Tensor    # 基点
    direction: torch.Tensor     # 方向


# =============================================================================
# 协议定义（接口）
# =============================================================================

class VectorField(Protocol):
    """向量场协议"""
    def __call__(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """在点z处t时刻的向量场值"""
        ...


class MetricTensor(Protocol):
    """度量张量协议"""
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """返回点z处的度量矩阵 [B, D, D]"""
        ...


# =============================================================================
# 核心组件 1: 学习的度量张量
# =============================================================================

class LearnedMetric(nn.Module):
    """可学习的黎曼度量
    
    度量张量 g(z) 定义：
    - 切空间上的内积: <u, v>_z = u^T g(z) v
    - 测地线距离
    - 曲率
    
    约束：g 必须正定对称
    """
    def __init__(self, d: int, hidden_dim: int = 64):
        super().__init__()
        self.d = d
        
        # 参数化正定矩阵：g = L L^T + εI
        # L 是下三角矩阵
        self.metric_net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d * d),
        )
        
        # 确保正定性的最小特征值
        self.eps = 1e-4
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, D] 流形上的点
        返回: [B, D, D] 正定对称度量矩阵
        """
        B, D = z.shape
        
        # 预测矩阵元素
        L_flat = self.metric_net(z)  # [B, D*D]
        L = L_flat.view(B, D, D)
        
        # 取下三角部分
        L = torch.tril(L)
        
        # g = L L^T + εI（保证正定）
        g = torch.bmm(L, L.transpose(-2, -1))
        g = g + self.eps * torch.eye(D, device=z.device).unsqueeze(0)
        
        return g
    
    def inner_product(self, z: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """计算切向量内积 <u, v>_z = u^T g(z) v"""
        g = self(z)  # [B, D, D]
        # u^T g v
        return torch.einsum('bd,bde,be->b', u, g, v)
    
    def norm(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """计算切向量范数 ||v||_z = sqrt(<v, v>_z)"""
        return torch.sqrt(self.inner_product(z, v, v) + 1e-8)


# =============================================================================
# 核心组件 2: 指数映射（流形约束）
# =============================================================================

class ExponentialMap(nn.Module):
    """指数映射: T_z M → M
    
    将切空间中的向量映射到流形上
    exp_z(v) = 沿v方向从z出发的测地线终点
    
    对于一般流形，使用神经网络近似
    """
    def __init__(self, d: int, hidden_dim: int = 64):
        super().__init__()
        self.d = d
        
        # 指数映射网络
        self.exp_net = nn.Sequential(
            nn.Linear(d * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d),
        )
        
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        z: [B, D] 基点
        v: [B, D] 切向量
        返回: [B, D] 流形上的新点 exp_z(v)
        """
        # 一阶近似：exp_z(v) ≈ z + v + correction
        combined = torch.cat([z, v], dim=-1)
        correction = self.exp_net(combined) * self.scale
        
        # 保持一阶精度
        return z + v + correction * v.norm(dim=-1, keepdim=True)


class LogarithmMap(nn.Module):
    """对数映射: M × M → T_z M
    
    log_z(w) = 从z到w的测地线初始切向量
    是指数映射的逆
    """
    def __init__(self, d: int, hidden_dim: int = 64):
        super().__init__()
        self.d = d
        
        self.log_net = nn.Sequential(
            nn.Linear(d * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d),
        )
    
    def forward(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        z: [B, D] 基点
        w: [B, D] 目标点
        返回: [B, D] 切向量 log_z(w)
        """
        # 一阶近似：log_z(w) ≈ w - z + correction
        combined = torch.cat([z, w], dim=-1)
        correction = self.log_net(combined)
        
        diff = w - z
        return diff + correction * diff.norm(dim=-1, keepdim=True)


# =============================================================================
# 核心组件 3: Neural ODE 求解器
# =============================================================================

class ODESolver:
    """ODE求解器（纯函数式）
    
    求解: dz/dt = f(z, t)
    
    支持多种方法：
    - euler: 一阶欧拉法
    - rk4: 四阶龙格-库塔
    - midpoint: 中点法
    """
    
    @staticmethod
    def euler_step(
        f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        z: torch.Tensor,
        t: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """欧拉法单步"""
        return z + dt * f(z, t)
    
    @staticmethod
    def rk4_step(
        f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        z: torch.Tensor,
        t: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """RK4单步"""
        k1 = f(z, t)
        k2 = f(z + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(z + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(z + dt * k3, t + dt)
        return z + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    @staticmethod
    def midpoint_step(
        f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        z: torch.Tensor,
        t: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """中点法单步"""
        z_mid = z + 0.5 * dt * f(z, t)
        return z + dt * f(z_mid, t + 0.5 * dt)
    
    @staticmethod
    def solve(
        f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        z0: torch.Tensor,
        t_span: Tuple[float, float],
        num_steps: int = 10,
        method: str = 'rk4'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        求解ODE
        
        f: 向量场函数
        z0: [B, D] 初始状态
        t_span: (t0, t1) 时间范围
        num_steps: 步数
        method: 求解方法
        
        返回: (轨迹 [B, T, D], 时间点 [T])
        """
        t0, t1 = t_span
        dt = (t1 - t0) / num_steps
        device = z0.device
        
        # 选择步进方法
        step_fn = {
            'euler': ODESolver.euler_step,
            'rk4': ODESolver.rk4_step,
            'midpoint': ODESolver.midpoint_step,
        }[method]
        
        # 函数式折叠
        trajectory = [z0]
        times = [t0]
        z = z0
        t = torch.tensor(t0, device=device)
        
        for i in range(num_steps):
            z = step_fn(f, z, t, dt)
            t = t + dt
            trajectory.append(z)
            times.append(t.item())
        
        return torch.stack(trajectory, dim=1), torch.tensor(times, device=device)


# =============================================================================
# 核心组件 4: 神经向量场
# =============================================================================

class NeuralVectorField(nn.Module):
    """神经网络参数化的向量场
    
    f(z, t) 定义了流形上的动力学
    
    特性：
    - 时间依赖：可学习时间调制
    - 利普希茨约束：谱归一化保证稳定性
    """
    def __init__(self, d: int, hidden_dim: int = 128, time_embed_dim: int = 16):
        super().__init__()
        self.d = d
        
        # 时间嵌入（傅里叶特征）
        self.time_embed = nn.Sequential(
            FourierFeatures(time_embed_dim),
            nn.Linear(time_embed_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        
        # 向量场网络
        self.field_net = nn.Sequential(
            nn.Linear(d + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d),
        )
        
        # 谱归一化（利普希茨约束）
        self._apply_spectral_norm()
    
    def _apply_spectral_norm(self):
        """应用谱归一化"""
        for module in self.field_net:
            if isinstance(module, nn.Linear):
                nn.utils.parametrizations.spectral_norm(module)
    
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z: [B, D] 当前状态
        t: [] 或 [B] 时间
        返回: [B, D] 向量场值
        """
        B = z.shape[0]
        
        # 时间嵌入
        if t.dim() == 0:
            t = t.expand(B)
        t_emb = self.time_embed(t.unsqueeze(-1))  # [B, hidden]
        
        # 拼接并计算向量场
        combined = torch.cat([z, t_emb], dim=-1)
        return self.field_net(combined)


class FourierFeatures(nn.Module):
    """傅里叶特征嵌入"""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # 随机频率
        self.register_buffer('freqs', torch.randn(embed_dim) * 2 * math.pi)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B, 1] → [B, 2*embed_dim]"""
        proj = t * self.freqs  # [B, embed_dim]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# =============================================================================
# 核心组件 5: 流形神经ODE
# =============================================================================

class ManifoldNeuralODE(nn.Module):
    """流形上的神经ODE
    
    结合：
    1. 学习的度量（几何结构）
    2. 神经向量场（动力学）
    3. 指数映射（流形约束）
    
    dz/dt = exp_z(f(z, t))
    """
    def __init__(
        self, 
        d: int, 
        hidden_dim: int = 128,
        use_metric: bool = True,
        solver: str = 'rk4'
    ):
        super().__init__()
        self.d = d
        self.solver = solver
        self.use_metric = use_metric
        
        # 度量张量
        self.metric = LearnedMetric(d, hidden_dim // 2) if use_metric else None
        
        # 神经向量场
        self.vector_field = NeuralVectorField(d, hidden_dim)
        
        # 指数映射
        self.exp_map = ExponentialMap(d, hidden_dim // 2)
        
        # 对数映射
        self.log_map = LogarithmMap(d, hidden_dim // 2)
    
    def dynamics(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """计算流形上的动力学"""
        # 切空间中的向量场
        v = self.vector_field(z, t)
        
        # 如果使用度量，进行归一化
        if self.use_metric and self.metric is not None:
            v_norm = self.metric.norm(z, v)
            v = v / (v_norm.unsqueeze(-1) + 1e-6)
        
        return v
    
    def forward(
        self, 
        z0: torch.Tensor, 
        t_span: Tuple[float, float] = (0.0, 1.0),
        num_steps: int = 10,
        return_trajectory: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        z0: [B, D] 初始状态
        t_span: 时间范围
        num_steps: 积分步数
        return_trajectory: 是否返回完整轨迹
        
        返回: z_final [B, D] 或 (trajectory [B, T, D], times [T])
        """
        trajectory, times = ODESolver.solve(
            self.dynamics,
            z0,
            t_span,
            num_steps,
            self.solver
        )
        
        if return_trajectory:
            return trajectory, times
        else:
            return trajectory[:, -1]  # 只返回终态
    
    def geodesic_distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """计算两点间的测地线距离（近似）"""
        v = self.log_map(z1, z2)
        if self.metric is not None:
            return self.metric.norm(z1, v)
        else:
            return v.norm(dim=-1)


# =============================================================================
# 高阶组合子
# =============================================================================

def parallel_transport(
    metric: LearnedMetric,
    v: torch.Tensor,
    path: torch.Tensor
) -> torch.Tensor:
    """沿路径平行移动切向量
    
    metric: 度量张量
    v: [B, D] 初始切向量
    path: [B, T, D] 路径
    
    返回: [B, D] 移动后的切向量
    """
    # 简化实现：使用Schild阶梯法
    B, T, D = path.shape
    
    v_transported = v
    for t in range(T - 1):
        z_curr = path[:, t]
        z_next = path[:, t + 1]
        
        # 连接向量
        diff = z_next - z_curr
        
        # 简单的平行移动近似
        # 实际实现需要考虑联络
        g_curr = metric(z_curr)
        g_next = metric(z_next)
        
        # 通过度量变换
        g_ratio = torch.linalg.solve(g_curr, g_next)
        v_transported = torch.bmm(g_ratio, v_transported.unsqueeze(-1)).squeeze(-1)
    
    return v_transported


def geodesic_interpolation(
    model: ManifoldNeuralODE,
    z1: torch.Tensor,
    z2: torch.Tensor,
    num_points: int = 10
) -> torch.Tensor:
    """测地线插值
    
    z1, z2: [B, D] 端点
    num_points: 插值点数
    
    返回: [B, num_points, D] 测地线路径
    """
    # 使用ODE求解器找到连接两点的测地线
    v = model.log_map(z1, z2)
    
    # 沿切向量方向积分
    def geodesic_field(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return v
    
    trajectory, _ = ODESolver.solve(
        geodesic_field,
        z1,
        (0.0, 1.0),
        num_points - 1,
        'rk4'
    )
    
    return trajectory


# =============================================================================
# 工具函数
# =============================================================================

def batch_trace(A: torch.Tensor) -> torch.Tensor:
    """批量矩阵迹 [B, D, D] → [B]"""
    return torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)


def batch_det(A: torch.Tensor) -> torch.Tensor:
    """批量行列式 [B, D, D] → [B]"""
    return torch.linalg.det(A)


def ricci_scalar(metric: LearnedMetric, z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """计算标量曲率（数值近似）
    
    R = g^{ij} R_{ij}
    
    使用有限差分近似
    """
    B, D = z.shape
    device = z.device
    
    g = metric(z)
    g_inv = torch.linalg.inv(g)
    
    # 计算度量的二阶导数（有限差分）
    R = torch.zeros(B, device=device)
    
    for i in range(D):
        e_i = torch.zeros(D, device=device)
        e_i[i] = eps
        
        g_plus = metric(z + e_i)
        g_minus = metric(z - e_i)
        
        # 二阶导数近似
        d2g = (g_plus - 2*g + g_minus) / (eps**2)
        
        # 简化的曲率贡献
        R = R + batch_trace(torch.bmm(g_inv, d2g))
    
    return R
