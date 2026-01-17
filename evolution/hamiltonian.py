"""守恒量与哈密顿结构

核心数学结构：
1. 哈密顿力学 - H(q, p) 能量函数
2. 拉格朗日力学 - L(q, q̇) 作用量
3. 诺特定理 - 对称性 ↔ 守恒量
4. 辛几何 - 相空间结构

关键特性：
- 能量守恒：dH/dt = 0
- 动量守恒：dp/dt = 0（平移对称）
- 角动量守恒：dL/dt = 0（旋转对称）
- 辛结构保持：保体积演化
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 代数数据类型定义
# =============================================================================

@dataclass(frozen=True)
class PhaseSpacePoint:
    """相空间中的点（不可变）
    
    (q, p) ∈ T*M
    q: 广义坐标（位置）
    p: 广义动量
    """
    q: torch.Tensor  # [B, D] 位置
    p: torch.Tensor  # [B, D] 动量
    
    def to_tensor(self) -> torch.Tensor:
        """转换为单一张量 [B, 2D]"""
        return torch.cat([self.q, self.p], dim=-1)
    
    @staticmethod
    def from_tensor(z: torch.Tensor) -> 'PhaseSpacePoint':
        """从张量构造"""
        d = z.shape[-1] // 2
        return PhaseSpacePoint(z[..., :d], z[..., d:])
    
    def kinetic_energy(self, mass: torch.Tensor = None) -> torch.Tensor:
        """动能 T = p²/2m"""
        if mass is None:
            mass = torch.ones_like(self.p)
        return 0.5 * (self.p ** 2 / mass).sum(dim=-1)


@dataclass(frozen=True)
class ConservedQuantity:
    """守恒量"""
    name: str
    value: torch.Tensor
    generator: Optional[torch.Tensor]  # 对应的对称性生成元


# =============================================================================
# 哈密顿神经网络
# =============================================================================

class HamiltonianNet(nn.Module):
    """哈密顿神经网络
    
    学习能量函数 H(q, p)
    
    动力学由哈密顿方程给出：
    dq/dt = ∂H/∂p
    dp/dt = -∂H/∂q
    
    关键特性：
    1. 自动保证能量守恒
    2. 辛结构保持
    """
    def __init__(self, d: int, hidden_dim: int = 128):
        super().__init__()
        self.d = d
        
        # 分离架构：T(p) + V(q)
        # 这保证了正则结构
        
        # 动能网络 T(p)
        self.kinetic_net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # 保证非负
        )
        
        # 势能网络 V(q)
        self.potential_net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """计算哈密顿量 H(q, p) = T(p) + V(q)"""
        T = self.kinetic_net(p).squeeze(-1)
        V = self.potential_net(q).squeeze(-1)
        return T + V
    
    def kinetic(self, p: torch.Tensor) -> torch.Tensor:
        """动能 T(p)"""
        return self.kinetic_net(p).squeeze(-1)
    
    def potential(self, q: torch.Tensor) -> torch.Tensor:
        """势能 V(q)"""
        return self.potential_net(q).squeeze(-1)
    
    def dynamics(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """哈密顿方程
        
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q
        """
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        
        H = self(q, p)
        
        # 计算梯度
        dH_dq = torch.autograd.grad(H.sum(), q, create_graph=True)[0]
        dH_dp = torch.autograd.grad(H.sum(), p, create_graph=True)[0]
        
        dq_dt = dH_dp   # 正则方程
        dp_dt = -dH_dq
        
        return dq_dt, dp_dt


class SeparableHamiltonian(nn.Module):
    """可分离哈密顿量
    
    H(q, p) = T(p) + V(q)
    
    更简单的架构，更好的物理解释
    """
    def __init__(self, d: int, hidden_dim: int = 64):
        super().__init__()
        self.d = d
        
        # 学习的质量矩阵（正定）
        self.mass_L = nn.Parameter(torch.eye(d) * 0.1 + torch.randn(d, d) * 0.01)
        
        # 势能网络
        self.V_net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    @property
    def mass_matrix(self) -> torch.Tensor:
        """正定质量矩阵 M = L L^T"""
        return self.mass_L @ self.mass_L.T + 0.01 * torch.eye(self.d, device=self.mass_L.device)
    
    @property
    def inverse_mass(self) -> torch.Tensor:
        """逆质量矩阵 M^{-1}"""
        return torch.linalg.inv(self.mass_matrix)
    
    def kinetic(self, p: torch.Tensor) -> torch.Tensor:
        """动能 T = (1/2) p^T M^{-1} p"""
        M_inv = self.inverse_mass.to(p.device)
        return 0.5 * torch.einsum('bd,de,be->b', p, M_inv, p)
    
    def potential(self, q: torch.Tensor) -> torch.Tensor:
        """势能 V(q)"""
        return self.V_net(q).squeeze(-1)
    
    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """总能量"""
        return self.kinetic(p) + self.potential(q)
    
    def dynamics(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """哈密顿方程"""
        # dq/dt = M^{-1} p
        M_inv = self.inverse_mass.to(p.device)
        dq_dt = torch.einsum('de,be->bd', M_inv, p)
        
        # dp/dt = -∂V/∂q
        q = q.requires_grad_(True)
        V = self.potential(q)
        dp_dt = -torch.autograd.grad(V.sum(), q, create_graph=True)[0]
        
        return dq_dt, dp_dt


# =============================================================================
# 拉格朗日神经网络
# =============================================================================

class LagrangianNet(nn.Module):
    """拉格朗日神经网络
    
    学习拉格朗日量 L(q, q̇) = T - V
    
    动力学由欧拉-拉格朗日方程给出：
    d/dt(∂L/∂q̇) = ∂L/∂q
    """
    def __init__(self, d: int, hidden_dim: int = 128):
        super().__init__()
        self.d = d
        
        # 拉格朗日量网络
        self.L_net = nn.Sequential(
            nn.Linear(d * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, q: torch.Tensor, q_dot: torch.Tensor) -> torch.Tensor:
        """计算拉格朗日量 L(q, q̇)"""
        combined = torch.cat([q, q_dot], dim=-1)
        return self.L_net(combined).squeeze(-1)
    
    def dynamics(self, q: torch.Tensor, q_dot: torch.Tensor) -> torch.Tensor:
        """欧拉-拉格朗日方程 → 加速度
        
        q̈ = M^{-1}(∂L/∂q - ∂²L/∂q∂q̇ · q̇)
        
        其中 M = ∂²L/∂q̇²
        """
        q = q.requires_grad_(True)
        q_dot = q_dot.requires_grad_(True)
        
        L = self(q, q_dot)
        
        # ∂L/∂q
        dL_dq = torch.autograd.grad(L.sum(), q, create_graph=True, retain_graph=True)[0]
        
        # ∂L/∂q̇
        dL_dq_dot = torch.autograd.grad(L.sum(), q_dot, create_graph=True, retain_graph=True)[0]
        
        # ∂²L/∂q̇² (质量矩阵)
        B, D = q.shape
        M = torch.zeros(B, D, D, device=q.device)
        for i in range(D):
            d2L_dqdot2_i = torch.autograd.grad(
                dL_dq_dot[:, i].sum(), q_dot, create_graph=True, retain_graph=True
            )[0]
            M[:, i] = d2L_dqdot2_i
        
        # ∂²L/∂q∂q̇ · q̇
        mixed_term = torch.zeros(B, D, device=q.device)
        for i in range(D):
            d2L_dqdqdot_i = torch.autograd.grad(
                dL_dq_dot[:, i].sum(), q, create_graph=True, retain_graph=True
            )[0]
            mixed_term += d2L_dqdqdot_i * q_dot[:, i:i+1]
        
        # 求解加速度
        rhs = dL_dq - mixed_term
        q_ddot = torch.linalg.solve(M + 1e-4 * torch.eye(D, device=M.device), rhs.unsqueeze(-1)).squeeze(-1)
        
        return q_ddot


# =============================================================================
# 辛积分器
# =============================================================================

class SymplecticIntegrator:
    """辛积分器（保辛结构）
    
    保持哈密顿系统的几何结构
    """
    
    @staticmethod
    def leapfrog_step(
        H: HamiltonianNet | SeparableHamiltonian,
        q: torch.Tensor,
        p: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """蛙跳法（二阶辛积分）
        
        p_{1/2} = p_0 - (dt/2) ∂V/∂q(q_0)
        q_1 = q_0 + dt · M^{-1} p_{1/2}
        p_1 = p_{1/2} - (dt/2) ∂V/∂q(q_1)
        """
        # 半步动量更新
        q = q.requires_grad_(True)
        V = H.potential(q)
        dV_dq = torch.autograd.grad(V.sum(), q)[0]
        p_half = p - 0.5 * dt * dV_dq
        
        # 全步位置更新
        if hasattr(H, 'inverse_mass'):
            M_inv = H.inverse_mass.to(p.device)
            q_new = q + dt * torch.einsum('de,be->bd', M_inv, p_half)
        else:
            q_new = q + dt * p_half
        
        # 半步动量更新
        q_new = q_new.requires_grad_(True)
        V_new = H.potential(q_new)
        dV_dq_new = torch.autograd.grad(V_new.sum(), q_new)[0]
        p_new = p_half - 0.5 * dt * dV_dq_new
        
        return q_new.detach(), p_new.detach()
    
    @staticmethod
    def yoshida4_step(
        H: HamiltonianNet | SeparableHamiltonian,
        q: torch.Tensor,
        p: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Yoshida四阶辛积分"""
        # Yoshida系数
        w1 = 1.0 / (2 - 2**(1/3))
        w0 = 1 - 2 * w1
        
        # 三次蛙跳步
        q, p = SymplecticIntegrator.leapfrog_step(H, q, p, w1 * dt)
        q, p = SymplecticIntegrator.leapfrog_step(H, q, p, w0 * dt)
        q, p = SymplecticIntegrator.leapfrog_step(H, q, p, w1 * dt)
        
        return q, p
    
    @staticmethod
    def integrate(
        H: HamiltonianNet | SeparableHamiltonian,
        q0: torch.Tensor,
        p0: torch.Tensor,
        t_span: Tuple[float, float],
        num_steps: int = 100,
        method: str = 'leapfrog'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """辛积分
        
        返回: (q_trajectory, p_trajectory, times)
        """
        t0, t1 = t_span
        dt = (t1 - t0) / num_steps
        
        step_fn = {
            'leapfrog': SymplecticIntegrator.leapfrog_step,
            'yoshida4': SymplecticIntegrator.yoshida4_step,
        }[method]
        
        q_traj = [q0]
        p_traj = [p0]
        
        q, p = q0, p0
        for _ in range(num_steps):
            q, p = step_fn(H, q, p, dt)
            q_traj.append(q)
            p_traj.append(p)
        
        times = torch.linspace(t0, t1, num_steps + 1, device=q0.device)
        
        return torch.stack(q_traj, dim=1), torch.stack(p_traj, dim=1), times


# =============================================================================
# 守恒量检测与监控
# =============================================================================

class ConservationMonitor(nn.Module):
    """守恒量监控器
    
    检测和跟踪守恒量：
    1. 能量 H
    2. 动量 p（如果平移对称）
    3. 角动量 L（如果旋转对称）
    4. 可学习的守恒量
    """
    def __init__(self, d: int, num_learned: int = 2):
        super().__init__()
        self.d = d
        self.num_learned = num_learned
        
        # 可学习的守恒量
        self.learned_invariants = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d * 2, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            ) for _ in range(num_learned)
        ])
    
    def energy(self, H: nn.Module, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """能量"""
        return H(q, p)
    
    def momentum(self, p: torch.Tensor) -> torch.Tensor:
        """总动量 P = Σ p_i"""
        return p.sum(dim=-1)
    
    def angular_momentum(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """角动量 L = q × p（仅对3D有意义）
        
        对于高维，返回广义角动量张量的范数
        """
        # L_ij = q_i p_j - q_j p_i
        L = torch.einsum('bi,bj->bij', q, p) - torch.einsum('bj,bi->bij', q, p)
        return L.pow(2).sum(dim=(-2, -1)).sqrt()
    
    def learned(self, q: torch.Tensor, p: torch.Tensor) -> List[torch.Tensor]:
        """可学习的守恒量"""
        z = torch.cat([q, p], dim=-1)
        return [net(z).squeeze(-1) for net in self.learned_invariants]
    
    def all_quantities(
        self, 
        H: nn.Module, 
        q: torch.Tensor, 
        p: torch.Tensor
    ) -> List[ConservedQuantity]:
        """计算所有守恒量"""
        quantities = [
            ConservedQuantity("energy", self.energy(H, q, p), None),
            ConservedQuantity("momentum", self.momentum(p), None),
            ConservedQuantity("angular_momentum", self.angular_momentum(q, p), None),
        ]
        
        for i, val in enumerate(self.learned(q, p)):
            quantities.append(ConservedQuantity(f"learned_{i}", val, None))
        
        return quantities
    
    def conservation_loss(
        self, 
        H: nn.Module,
        q_trajectory: torch.Tensor,
        p_trajectory: torch.Tensor
    ) -> torch.Tensor:
        """守恒量违反损失
        
        L_cons = Var_t[C(q(t), p(t))]
        
        如果C是守恒量，则沿轨迹应该恒定
        """
        B, T, D = q_trajectory.shape
        
        # 沿轨迹计算能量
        energies = []
        for t in range(T):
            E = H(q_trajectory[:, t], p_trajectory[:, t])
            energies.append(E)
        energies = torch.stack(energies, dim=1)  # [B, T]
        
        # 能量应该恒定 → 方差应该为零
        energy_var = energies.var(dim=1).mean()
        
        # 可学习守恒量的方差
        learned_var = 0
        for net in self.learned_invariants:
            values = []
            for t in range(T):
                z = torch.cat([q_trajectory[:, t], p_trajectory[:, t]], dim=-1)
                values.append(net(z).squeeze(-1))
            values = torch.stack(values, dim=1)
            learned_var = learned_var + values.var(dim=1).mean()
        
        return energy_var + learned_var


# =============================================================================
# 哈密顿演化模块
# =============================================================================

class HamiltonianEvolution(nn.Module):
    """哈密顿演化模块
    
    结合：
    1. 可学习的哈密顿量
    2. 辛积分器
    3. 守恒量监控
    
    保证长期预测的合理性
    """
    def __init__(
        self, 
        d: int, 
        hidden_dim: int = 128,
        integrator: str = 'leapfrog',
        hamiltonian_type: str = 'separable'
    ):
        super().__init__()
        self.d = d
        self.integrator = integrator
        
        # 哈密顿量
        if hamiltonian_type == 'separable':
            self.H = SeparableHamiltonian(d, hidden_dim)
        else:
            self.H = HamiltonianNet(d, hidden_dim)
        
        # 守恒量监控
        self.monitor = ConservationMonitor(d)
        
        # 初始条件编码器（从z空间到相空间）
        self.q_encoder = nn.Linear(d, d)
        self.p_encoder = nn.Linear(d, d)
    
    def z_to_phase_space(self, z: torch.Tensor) -> PhaseSpacePoint:
        """将z空间点转换为相空间点"""
        q = self.q_encoder(z)
        p = self.p_encoder(z)
        return PhaseSpacePoint(q, p)
    
    def phase_space_to_z(self, point: PhaseSpacePoint) -> torch.Tensor:
        """将相空间点转换回z空间"""
        return point.to_tensor()
    
    def forward(
        self, 
        z: torch.Tensor, 
        t_span: Tuple[float, float] = (0.0, 1.0),
        num_steps: int = 10,
        return_trajectory: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        z: [B, D] 输入（z空间）
        t_span: 演化时间范围
        num_steps: 积分步数
        
        返回: 演化后的z [B, 2D] 或 (z, info)
        """
        # 转换到相空间
        point = self.z_to_phase_space(z)
        
        # 辛积分
        q_traj, p_traj, times = SymplecticIntegrator.integrate(
            self.H,
            point.q,
            point.p,
            t_span,
            num_steps,
            self.integrator
        )
        
        # 最终状态
        final_point = PhaseSpacePoint(q_traj[:, -1], p_traj[:, -1])
        z_final = self.phase_space_to_z(final_point)
        
        if return_trajectory:
            # 计算守恒量
            quantities = self.monitor.all_quantities(self.H, point.q, point.p)
            
            info = {
                'q_trajectory': q_traj,
                'p_trajectory': p_traj,
                'times': times,
                'conserved_quantities': quantities,
                'energy_initial': self.H(point.q, point.p),
                'energy_final': self.H(q_traj[:, -1], p_traj[:, -1]),
            }
            return z_final, info
        
        return z_final
    
    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """计算z点的能量"""
        point = self.z_to_phase_space(z)
        return self.H(point.q, point.p)
    
    def conservation_loss(self, z_trajectory: torch.Tensor) -> torch.Tensor:
        """沿轨迹的守恒量损失"""
        B, T, D = z_trajectory.shape
        
        # 转换整个轨迹
        q_traj = self.q_encoder(z_trajectory)
        p_traj = self.p_encoder(z_trajectory)
        
        return self.monitor.conservation_loss(self.H, q_traj, p_traj)


# =============================================================================
# 诺特定理：对称性 ↔ 守恒量
# =============================================================================

class NoetherModule(nn.Module):
    """诺特模块
    
    实现诺特定理：
    连续对称性 → 守恒量
    
    给定生成元 G，对应的守恒量：
    Q = p · G(q)
    """
    def __init__(self, d: int, num_symmetries: int = 3):
        super().__init__()
        self.d = d
        self.num_symmetries = num_symmetries
        
        # 可学习的对称性生成元
        # G(q) 是向量场
        self.generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, 64),
                nn.SiLU(),
                nn.Linear(64, d),
            ) for _ in range(num_symmetries)
        ])
    
    def symmetry_generator(self, q: torch.Tensor, idx: int) -> torch.Tensor:
        """计算第idx个对称性的生成元场"""
        return self.generators[idx](q)
    
    def noether_charge(
        self, 
        q: torch.Tensor, 
        p: torch.Tensor, 
        idx: int
    ) -> torch.Tensor:
        """计算诺特荷（守恒量）
        
        Q = p · G(q)
        """
        G = self.symmetry_generator(q, idx)
        return (p * G).sum(dim=-1)
    
    def all_charges(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """计算所有诺特荷 [B, num_symmetries]"""
        charges = []
        for i in range(self.num_symmetries):
            charges.append(self.noether_charge(q, p, i))
        return torch.stack(charges, dim=-1)
    
    def symmetry_loss(
        self, 
        H: nn.Module,
        q: torch.Tensor,
        p: torch.Tensor
    ) -> torch.Tensor:
        """对称性损失
        
        如果G是H的对称性，则 {Q, H} = 0
        即 dQ/dt = 0
        """
        loss = 0
        for i in range(self.num_symmetries):
            Q = self.noether_charge(q, p, i)
            
            # 计算泊松括号 {Q, H}
            q = q.requires_grad_(True)
            p = p.requires_grad_(True)
            
            Q_val = self.noether_charge(q, p, i)
            H_val = H(q, p)
            
            dQ_dq = torch.autograd.grad(Q_val.sum(), q, create_graph=True)[0]
            dQ_dp = torch.autograd.grad(Q_val.sum(), p, create_graph=True)[0]
            dH_dq = torch.autograd.grad(H_val.sum(), q, create_graph=True)[0]
            dH_dp = torch.autograd.grad(H_val.sum(), p, create_graph=True)[0]
            
            # {Q, H} = ∂Q/∂q · ∂H/∂p - ∂Q/∂p · ∂H/∂q
            poisson_bracket = (dQ_dq * dH_dp).sum(dim=-1) - (dQ_dp * dH_dq).sum(dim=-1)
            
            loss = loss + poisson_bracket.pow(2).mean()
        
        return loss / self.num_symmetries


# =============================================================================
# 高阶组合子
# =============================================================================

def poisson_bracket(
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    g: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    p: torch.Tensor
) -> torch.Tensor:
    """泊松括号
    
    {f, g} = Σ_i (∂f/∂q_i · ∂g/∂p_i - ∂f/∂p_i · ∂g/∂q_i)
    """
    q = q.requires_grad_(True)
    p = p.requires_grad_(True)
    
    f_val = f(q, p)
    g_val = g(q, p)
    
    df_dq = torch.autograd.grad(f_val.sum(), q, create_graph=True)[0]
    df_dp = torch.autograd.grad(f_val.sum(), p, create_graph=True)[0]
    dg_dq = torch.autograd.grad(g_val.sum(), q, create_graph=True)[0]
    dg_dp = torch.autograd.grad(g_val.sum(), p, create_graph=True)[0]
    
    return (df_dq * dg_dp).sum(dim=-1) - (df_dp * dg_dq).sum(dim=-1)


def symplectic_gradient(
    H: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    p: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """辛梯度
    
    X_H = (∂H/∂p, -∂H/∂q)
    
    哈密顿向量场
    """
    q = q.requires_grad_(True)
    p = p.requires_grad_(True)
    
    H_val = H(q, p)
    
    dH_dq = torch.autograd.grad(H_val.sum(), q)[0]
    dH_dp = torch.autograd.grad(H_val.sum(), p)[0]
    
    return dH_dp, -dH_dq


def liouville_measure(q_traj: torch.Tensor, p_traj: torch.Tensor) -> torch.Tensor:
    """刘维尔测度（相空间体积）
    
    辛积分保持刘维尔测度
    """
    B, T, D = q_traj.shape
    
    # 计算相空间中的"体积"变化
    # 使用雅可比行列式近似
    volumes = []
    for t in range(1, T):
        dq = q_traj[:, t] - q_traj[:, t-1]
        dp = p_traj[:, t] - p_traj[:, t-1]
        
        # 简化的体积估计
        vol = (dq.norm(dim=-1) * dp.norm(dim=-1))
        volumes.append(vol)
    
    return torch.stack(volumes, dim=1)
