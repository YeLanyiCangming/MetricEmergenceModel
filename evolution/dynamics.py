"""二阶动力学演化器

核心模块：在Z空间中进行二阶微分方程演化

数学框架：
  d²z/dt² = F(z, dz/dt, G, t)
  
等价于一阶系统：
  dz/dt = v
  dv/dt = F(z, v, G, t)

整合：
1. 流形结构（几何）
2. 对称性（等变性）
3. 哈密顿结构（守恒）
4. 因果结构（可解释）
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .manifold import ManifoldNeuralODE, LearnedMetric, ODESolver
from .symmetry import LearnedSymmetryGenerator, EquivariantMLP, RotationGroup
from .hamiltonian import HamiltonianEvolution, SeparableHamiltonian, SymplecticIntegrator
from .causality import CausalEvolution, StructuralCausalModel


# =============================================================================
# 代数数据类型
# =============================================================================

@dataclass(frozen=True)
class DynamicsState:
    """动力学状态（不可变）
    
    相空间点 (z, v) ∈ TM
    """
    position: torch.Tensor    # z: [B, D] 位置
    velocity: torch.Tensor    # v = dz/dt: [B, D] 速度
    
    def to_tensor(self) -> torch.Tensor:
        """合并为单一张量"""
        return torch.cat([self.position, self.velocity], dim=-1)
    
    @staticmethod
    def from_tensor(state: torch.Tensor) -> 'DynamicsState':
        """从张量构造"""
        d = state.shape[-1] // 2
        return DynamicsState(state[..., :d], state[..., d:])
    
    def kinetic_energy(self) -> torch.Tensor:
        """动能 T = (1/2)||v||²"""
        return 0.5 * self.velocity.pow(2).sum(dim=-1)


@dataclass(frozen=True)
class EvolutionConfig:
    """演化配置"""
    dt: float = 0.1                    # 时间步长
    num_steps: int = 10                # 演化步数
    integrator: str = 'rk4'            # 积分器类型
    use_hamiltonian: bool = True       # 是否使用哈密顿结构
    use_symmetry: bool = True          # 是否使用对称性
    use_causality: bool = False        # 是否使用因果结构
    energy_regularization: float = 0.1 # 能量正则化强度


# =============================================================================
# 力场网络
# =============================================================================

class ForceField(nn.Module):
    """力场网络
    
    F(z, v, t) = 加速度 = d²z/dt²
    
    分解为：
    - 保守力：F_cons = -∇V(z)（来自势能）
    - 阻尼力：F_damp = -γv（耗散）
    - 驱动力：F_drive(z, t)（外部输入）
    """
    def __init__(self, d: int, hidden_dim: int = 128):
        super().__init__()
        self.d = d
        
        # 势能网络
        self.potential_net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # 阻尼系数（可学习）
        self.damping = nn.Parameter(torch.tensor(0.1))
        
        # 驱动力网络
        self.drive_net = nn.Sequential(
            nn.Linear(d + 1, hidden_dim),  # +1 for time
            nn.SiLU(),
            nn.Linear(hidden_dim, d),
        )
        
        # 非线性力（神经网络补偿）
        self.nonlinear_net = nn.Sequential(
            nn.Linear(d * 2, hidden_dim),  # z + v
            nn.SiLU(),
            nn.Linear(hidden_dim, d),
        )
        
        # 非线性力强度
        self.nonlinear_scale = nn.Parameter(torch.tensor(0.1))
    
    def potential(self, z: torch.Tensor) -> torch.Tensor:
        """势能 V(z)"""
        return self.potential_net(z).squeeze(-1)
    
    def conservative_force(self, z: torch.Tensor) -> torch.Tensor:
        """保守力 F = -∇V"""
        z = z.requires_grad_(True)
        V = self.potential(z)
        grad_V = torch.autograd.grad(V.sum(), z, create_graph=True)[0]
        return -grad_V
    
    def damping_force(self, v: torch.Tensor) -> torch.Tensor:
        """阻尼力 F = -γv"""
        return -F.softplus(self.damping) * v
    
    def driving_force(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """驱动力 F(z, t)"""
        B = z.shape[0]
        if t.dim() == 0:
            t = t.expand(B, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        
        combined = torch.cat([z, t], dim=-1)
        return self.drive_net(combined)
    
    def nonlinear_force(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """非线性力（神经网络）"""
        combined = torch.cat([z, v], dim=-1)
        return self.nonlinear_scale * self.nonlinear_net(combined)
    
    def forward(
        self, 
        z: torch.Tensor, 
        v: torch.Tensor, 
        t: torch.Tensor,
        include_nonlinear: bool = True
    ) -> torch.Tensor:
        """总力 F = F_cons + F_damp + F_drive + F_nonlinear"""
        F_cons = self.conservative_force(z)
        F_damp = self.damping_force(v)
        F_drive = self.driving_force(z, t)
        
        F_total = F_cons + F_damp + F_drive
        
        if include_nonlinear:
            F_total = F_total + self.nonlinear_force(z, v)
        
        return F_total


# =============================================================================
# 对称性增强的力场
# =============================================================================

class SymmetryAugmentedForce(nn.Module):
    """对称性增强的力场
    
    F(z, v, G, α) = Σ_i α_i G_i · f_i(z, v)
    
    其中 G_i 是对称性生成元
    """
    def __init__(self, d: int, num_generators: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.d = d
        self.num_generators = num_generators
        
        # 对称性生成元
        self.generators = LearnedSymmetryGenerator(d, num_generators, 'general')
        
        # 每个生成元对应的力场系数
        self.coef_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            ) for _ in range(num_generators)
        ])
        
        # 等变力场（基础）
        self.equivariant_force = EquivariantMLP(d, hidden_dim, RotationGroup())
    
    def forward(
        self, 
        z: torch.Tensor, 
        v: torch.Tensor,
        alpha: torch.Tensor = None
    ) -> torch.Tensor:
        """
        z: [B, D] 位置
        v: [B, D] 速度
        alpha: [B, num_gen] 生成元强度（可选）
        
        返回: [B, D] 力
        """
        B = z.shape[0]
        device = z.device
        
        # 获取生成元
        G = self.generators()  # [num_gen, D, D]
        
        # 计算系数
        combined = torch.cat([z, v], dim=-1)
        
        if alpha is None:
            alpha = torch.stack([
                net(combined).squeeze(-1) for net in self.coef_nets
            ], dim=-1)  # [B, num_gen]
        
        # 组合生成元
        combined_G = torch.einsum('bg,gij->bij', alpha, G)  # [B, D, D]
        
        # 等变基础力
        F_base = self.equivariant_force(z)
        
        # 通过生成元作用
        F_augmented = torch.bmm(combined_G, F_base.unsqueeze(-1)).squeeze(-1)
        
        return F_augmented


# =============================================================================
# 二阶ODE求解器
# =============================================================================

class SecondOrderODESolver:
    """二阶ODE求解器
    
    求解: d²z/dt² = F(z, dz/dt, t)
    
    等价于一阶系统:
    dz/dt = v
    dv/dt = F(z, v, t)
    """
    
    @staticmethod
    def euler_step(
        F: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        z: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """欧拉法"""
        a = F(z, v, t)
        v_new = v + dt * a
        z_new = z + dt * v
        return z_new, v_new
    
    @staticmethod
    def verlet_step(
        F: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        z: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Velocity Verlet（辛积分器）"""
        a = F(z, v, t)
        z_new = z + dt * v + 0.5 * dt**2 * a
        a_new = F(z_new, v, t + dt)
        v_new = v + 0.5 * dt * (a + a_new)
        return z_new, v_new
    
    @staticmethod
    def rk4_step(
        F: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        z: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RK4"""
        k1_v = v
        k1_a = F(z, v, t)
        
        k2_v = v + 0.5 * dt * k1_a
        k2_a = F(z + 0.5 * dt * k1_v, k2_v, t + 0.5 * dt)
        
        k3_v = v + 0.5 * dt * k2_a
        k3_a = F(z + 0.5 * dt * k2_v, k3_v, t + 0.5 * dt)
        
        k4_v = v + dt * k3_a
        k4_a = F(z + dt * k3_v, k4_v, t + dt)
        
        z_new = z + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        v_new = v + (dt / 6) * (k1_a + 2*k2_a + 2*k3_a + k4_a)
        
        return z_new, v_new
    
    @staticmethod
    def solve(
        F: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        z0: torch.Tensor,
        v0: torch.Tensor,
        t_span: Tuple[float, float],
        num_steps: int = 10,
        method: str = 'verlet'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        求解二阶ODE
        
        返回: (z_trajectory, v_trajectory, times)
        """
        t0, t1 = t_span
        dt = (t1 - t0) / num_steps
        device = z0.device
        
        step_fn = {
            'euler': SecondOrderODESolver.euler_step,
            'verlet': SecondOrderODESolver.verlet_step,
            'rk4': SecondOrderODESolver.rk4_step,
        }[method]
        
        z_traj = [z0]
        v_traj = [v0]
        times = [t0]
        
        z, v = z0, v0
        t = torch.tensor(t0, device=device)
        
        for i in range(num_steps):
            z, v = step_fn(F, z, v, t, dt)
            t = t + dt
            z_traj.append(z)
            v_traj.append(v)
            times.append(t.item())
        
        return (
            torch.stack(z_traj, dim=1),
            torch.stack(v_traj, dim=1),
            torch.tensor(times, device=device)
        )


# =============================================================================
# 主要模块：二阶动力学演化器
# =============================================================================

class SecondOrderDynamics(nn.Module):
    """二阶动力学演化器
    
    整合所有物理结构：
    1. 流形几何
    2. 对称性
    3. 哈密顿结构
    4. 因果结构
    
    演化方程：
    d²z/dt² = F_physics(z, v) + F_symmetry(z, v, G) + F_neural(z, v)
    
    约束：
    - 能量守恒（或受控耗散）
    - 对称性不变
    - 因果一致
    """
    def __init__(
        self,
        d: int,
        hidden_dim: int = 128,
        num_generators: int = 4,
        config: EvolutionConfig = None
    ):
        super().__init__()
        self.d = d
        self.config = config or EvolutionConfig()
        
        # 基础力场
        self.force_field = ForceField(d, hidden_dim)
        
        # 对称性增强力场
        self.symmetry_force = SymmetryAugmentedForce(d, num_generators, hidden_dim)
        
        # 度量张量（流形结构）
        self.metric = LearnedMetric(d, hidden_dim // 2)
        
        # 速度编码器（从z推断初始速度）
        self.velocity_encoder = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d),
        )
        
        # 混合权重
        self.physics_weight = nn.Parameter(torch.tensor(0.5))
        self.symmetry_weight = nn.Parameter(torch.tensor(0.3))
        self.neural_weight = nn.Parameter(torch.tensor(0.2))
    
    def total_force(
        self, 
        z: torch.Tensor, 
        v: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """计算总力"""
        # 物理力
        F_physics = self.force_field(z, v, t)
        
        # 对称性力
        F_symmetry = self.symmetry_force(z, v)
        
        # 混合
        w_p = torch.sigmoid(self.physics_weight)
        w_s = torch.sigmoid(self.symmetry_weight)
        
        F_total = w_p * F_physics + w_s * F_symmetry
        
        return F_total
    
    def forward(
        self,
        z: torch.Tensor,
        v: torch.Tensor = None,
        t_span: Tuple[float, float] = (0.0, 1.0),
        num_steps: int = None,
        return_trajectory: bool = False
    ) -> DynamicsState | Tuple[DynamicsState, Dict]:
        """
        z: [B, D] 初始位置
        v: [B, D] 初始速度（可选，如果不提供则推断）
        t_span: 演化时间范围
        num_steps: 积分步数
        
        返回: 最终状态 或 (最终状态, 轨迹信息)
        """
        num_steps = num_steps or self.config.num_steps
        
        # 推断初始速度
        if v is None:
            v = self.velocity_encoder(z)
        
        # 求解ODE
        z_traj, v_traj, times = SecondOrderODESolver.solve(
            self.total_force,
            z, v,
            t_span,
            num_steps,
            self.config.integrator
        )
        
        # 最终状态
        final_state = DynamicsState(z_traj[:, -1], v_traj[:, -1])
        
        if return_trajectory:
            # 计算能量轨迹
            energies = []
            for t_idx in range(z_traj.shape[1]):
                z_t = z_traj[:, t_idx]
                v_t = v_traj[:, t_idx]
                T = 0.5 * v_t.pow(2).sum(dim=-1)
                V = self.force_field.potential(z_t)
                energies.append(T + V)
            
            info = {
                'z_trajectory': z_traj,
                'v_trajectory': v_traj,
                'times': times,
                'energies': torch.stack(energies, dim=1),
            }
            return final_state, info
        
        return final_state
    
    def predict_dynamics(
        self, 
        z: torch.Tensor, 
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测下一时刻的动力学变化
        
        返回: (dz/dt, d²z/dt²) = (v, a)
        """
        t = torch.tensor(0.0, device=z.device)
        a = self.total_force(z, v, t)
        return v, a
    
    def energy(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """计算总能量 E = T + V"""
        T = 0.5 * v.pow(2).sum(dim=-1)
        V = self.force_field.potential(z)
        return T + V
    
    def energy_conservation_loss(
        self, 
        z_trajectory: torch.Tensor, 
        v_trajectory: torch.Tensor
    ) -> torch.Tensor:
        """能量守恒损失"""
        B, T, D = z_trajectory.shape
        
        energies = []
        for t_idx in range(T):
            E = self.energy(z_trajectory[:, t_idx], v_trajectory[:, t_idx])
            energies.append(E)
        
        energies = torch.stack(energies, dim=1)  # [B, T]
        
        # 能量应该恒定 → 方差应该为零
        return energies.var(dim=1).mean()


# =============================================================================
# 完整的演化模块
# =============================================================================

class EvolutionModule(nn.Module):
    """完整的演化模块
    
    整合所有物理结构，提供统一接口
    
    流程：
    1. 输入 z → 分解为 (position, velocity, field, modality)
    2. 在Z空间中进行二阶ODE演化
    3. 输出演化后的 z'
    
    预测目标：d²z/dt²（加速度）
    """
    def __init__(
        self,
        d: int,
        hidden_dim: int = 128,
        num_generators: int = 4,
        config: EvolutionConfig = None
    ):
        super().__init__()
        self.d = d
        self.config = config or EvolutionConfig()
        
        # 子空间维度
        self.d_position = d // 4
        self.d_velocity = d // 4
        self.d_field = d // 4
        self.d_modality = d // 4
        
        # 二阶动力学
        self.dynamics = SecondOrderDynamics(
            self.d_position,
            hidden_dim,
            num_generators,
            config
        )
        
        # 场演化（生成元演化）
        self.field_evolution = nn.GRUCell(self.d_field, self.d_field)
        
        # 模态保持层（模态信息不应该变化太大）
        self.modality_gate = nn.Sequential(
            nn.Linear(self.d_modality, self.d_modality),
            nn.Sigmoid(),
        )
        
        # 输出融合
        self.output_fusion = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d),
        )
        
        # 哈密顿演化（可选）
        if self.config.use_hamiltonian:
            self.hamiltonian = HamiltonianEvolution(
                self.d_position + self.d_velocity,
                hidden_dim
            )
        
        # 因果演化（可选）
        if self.config.use_causality:
            self.causality = CausalEvolution(d, hidden_dim)
    
    def decompose_z(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """将z分解为子空间"""
        return {
            'position': z[..., :self.d_position],
            'velocity': z[..., self.d_position:self.d_position*2],
            'field': z[..., self.d_position*2:self.d_position*3],
            'modality': z[..., self.d_position*3:],
        }
    
    def compose_z(self, components: Dict[str, torch.Tensor]) -> torch.Tensor:
        """从子空间组合z"""
        return torch.cat([
            components['position'],
            components['velocity'],
            components['field'],
            components['modality'],
        ], dim=-1)
    
    def forward(
        self,
        z: torch.Tensor,
        num_steps: int = None,
        return_trajectory: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        z: [B, D] 输入状态
        
        返回: 演化后的 z' [B, D]
        """
        num_steps = num_steps or self.config.num_steps
        
        # 分解
        components = self.decompose_z(z)
        
        # 位置-速度演化（二阶ODE）
        state, info = self.dynamics(
            components['position'],
            components['velocity'],
            num_steps=num_steps,
            return_trajectory=True
        )
        
        # 更新位置和速度
        components['position'] = state.position
        components['velocity'] = state.velocity
        
        # 场演化
        components['field'] = self.field_evolution(
            components['field'],
            components['field']
        )
        
        # 模态保持
        gate = self.modality_gate(components['modality'])
        components['modality'] = gate * components['modality']
        
        # 组合
        z_evolved = self.compose_z(components)
        
        # 输出融合
        z_out = self.output_fusion(z_evolved)
        
        if return_trajectory:
            full_info = {
                'z_trajectory': info['z_trajectory'],
                'v_trajectory': info['v_trajectory'],
                'energies': info['energies'],
                'components': components,
            }
            return z_out, full_info
        
        return z_out
    
    def predict_acceleration(self, z: torch.Tensor) -> torch.Tensor:
        """预测加速度 d²z/dt²
        
        这是核心预测目标
        """
        components = self.decompose_z(z)
        
        v, a = self.dynamics.predict_dynamics(
            components['position'],
            components['velocity']
        )
        
        return a
    
    def loss(
        self,
        z: torch.Tensor,
        z_target: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """计算损失
        
        包含：
        1. 重构损失
        2. 能量守恒损失
        3. 对称性损失
        """
        # 演化
        z_pred, info = self.forward(z, return_trajectory=True)
        
        # 重构损失
        recon_loss = F.mse_loss(z_pred, z_target)
        
        # 能量守恒损失
        energy_loss = self.dynamics.energy_conservation_loss(
            info['z_trajectory'],
            info['v_trajectory']
        )
        
        # 总损失
        total_loss = (
            recon_loss +
            self.config.energy_regularization * energy_loss
        )
        
        if return_components:
            return total_loss, {
                'recon_loss': recon_loss,
                'energy_loss': energy_loss,
                'z_pred': z_pred,
            }
        
        return total_loss


# =============================================================================
# 高阶组合子
# =============================================================================

def compose_dynamics(
    *dynamics_modules: SecondOrderDynamics
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """组合多个动力学模块
    
    F_total = F_1 + F_2 + ... + F_n
    """
    def combined_force(z: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        forces = [d.total_force(z, v, t) for d in dynamics_modules]
        return reduce(torch.add, forces)
    
    return combined_force


def time_reversal(
    dynamics: SecondOrderDynamics
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """时间反演
    
    t → -t, v → -v
    
    对于保守系统，时间反演对称
    """
    def reversed_force(z: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -dynamics.total_force(z, -v, -t)
    
    return reversed_force


def perturbation_analysis(
    dynamics: SecondOrderDynamics,
    z: torch.Tensor,
    v: torch.Tensor,
    epsilon: float = 0.01
) -> Dict[str, torch.Tensor]:
    """扰动分析
    
    分析小扰动的演化行为
    """
    B, D = z.shape
    device = z.device
    t = torch.tensor(0.0, device=device)
    
    # 基准力
    F_base = dynamics.total_force(z, v, t)
    
    # 位置扰动响应
    dF_dz = torch.zeros(B, D, D, device=device)
    for i in range(D):
        z_perturbed = z.clone()
        z_perturbed[:, i] += epsilon
        F_perturbed = dynamics.total_force(z_perturbed, v, t)
        dF_dz[:, :, i] = (F_perturbed - F_base) / epsilon
    
    # 速度扰动响应
    dF_dv = torch.zeros(B, D, D, device=device)
    for i in range(D):
        v_perturbed = v.clone()
        v_perturbed[:, i] += epsilon
        F_perturbed = dynamics.total_force(z, v_perturbed, t)
        dF_dv[:, :, i] = (F_perturbed - F_base) / epsilon
    
    # 李雅普诺夫指数估计
    jacobian = torch.cat([
        torch.cat([torch.zeros(B, D, D, device=device), torch.eye(D, device=device).unsqueeze(0).expand(B, -1, -1)], dim=2),
        torch.cat([dF_dz, dF_dv], dim=2)
    ], dim=1)
    
    eigenvalues = torch.linalg.eigvals(jacobian)
    lyapunov = eigenvalues.real.max(dim=-1)[0]
    
    return {
        'dF_dz': dF_dz,
        'dF_dv': dF_dv,
        'jacobian': jacobian,
        'lyapunov_estimate': lyapunov,
    }
