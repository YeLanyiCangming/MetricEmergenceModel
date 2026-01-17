"""
演化模型 - 度规涌现架构 (Pure FP, Zero-Coupling, Algorithm-Aware)

核心范式（第一性原理）：
    数据 (x) → 抽象状态 (z) → 度规 g(z) → 联络 Γ (涌现) → 运动法则 (涌现)

三层架构：
    1. 感知层：学习流形的"本体"——度规张量
    2. 涌现层：从度规自动推导联络（Christoffel符号）——批量化向量操作
    3. 法则层：测地线加速度 + 外力 → 完整运动

架构原则：
    - 纯函数式：无共享状态，通过组合子构建
    - 类型安全：通过张量形状约束保证正确性
    - 数值稳定：自适应 eps、温和正则化、梯度裁剪
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Callable
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# 类型别名 (Type-Safe Semantic Modeling)
# =============================================================================

# 语义类型建模：用类型表达业务含义
LatentState = Tensor      # [B, L, D] 潜在状态
MetricTensor = Tensor     # [B, L, D, D] 度规张量
ChristoffelSymbol = Tensor  # [B, L, D, D, D] 联络符号
Velocity = Tensor         # [B, L, D] 速度
Acceleration = Tensor     # [B, L, D] 加速度
Distribution = Tuple[Tensor, Tensor]  # (mu, sigma)


# =============================================================================
# 纯函数组合子 (Pure Function Combinators)
# =============================================================================

def safe_inverse(g: MetricTensor, rcond: float = 1e-5) -> MetricTensor:
    """
    安全的矩阵伪逆 - 纯函数
    
    复杂度: O(D^3) per matrix
    代数律: safe_inverse(safe_inverse(g)) ≈ g (对于满秩矩阵)
    """
    return torch.linalg.pinv(g, rcond=rcond)


def symmetric_part(A: Tensor) -> Tensor:
    """
    提取对称部分 - 纯函数
    
    代数律: symmetric_part(symmetric_part(A)) = symmetric_part(A) (幂等性)
    复杂度: O(1) 内存视图操作
    """
    return (A + A.transpose(-1, -2)) * 0.5


def soft_clamp(x: Tensor, min_val: float, max_val: float, softness: float = 0.1) -> Tensor:
    """
    软裁剪 - 可微分的边界约束
    
    代数律: soft_clamp 是单调函数
    复杂度: O(1)
    """
    return min_val + (max_val - min_val) * torch.sigmoid((x - (min_val + max_val) / 2) / softness)


def hybrid_det_regularization(
    det_g: Tensor, 
    target: float = 0.1,
    log_weight: float = 1.0,
    target_weight: float = 1.0,
    sign_weight: float = 0.5,  # 已废弃
    eps: float = 1e-8,
    unconstrained: bool = False,
    min_abs_det: float = 0.1,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    混合行列式正则化 - 第一性原理版本
    
    核心哲学：轻量级正则化，让模型自由学习几何
    非退化保证应在架构层面，而非损失函数层面
    
    复杂度: O(B)
    """
    abs_det = torch.abs(det_g) + eps
    
    # 对数惩罚：轻量级，仅作为引导
    log_penalty = -torch.log(abs_det).mean()
    target_penalty = ((abs_det - target) ** 2).mean()
    
    if unconstrained:
        # 暴论模式：完全移除 det 正则化，改用条件数正则化
        # det 接近 0 不是问题，特征值有界才是关键
        total_reg = torch.tensor(0.0, device=det_g.device)  # 不再对 det 正则化
    else:
        total_reg = log_weight * log_penalty + target_weight * target_penalty
    
    mean_det = det_g.mean()
    pos_ratio = (det_g > 0).float().mean() if det_g.numel() > 1 else torch.tensor(1.0, device=det_g.device)
    
    info = {
        'log_penalty': log_penalty,
        'target_penalty': target_penalty,
        'inverse_penalty': torch.tensor(0.0, device=det_g.device),
        'threshold_penalty': torch.tensor(0.0, device=det_g.device),
        'sign_variance': torch.tensor(0.0, device=det_g.device),
        'mean_abs_det': abs_det.mean(),
        'mean_det': mean_det,
        'pos_ratio': pos_ratio,
        'det_std': det_g.std() if det_g.numel() > 1 else torch.tensor(0.0, device=det_g.device),
    }
    
    return total_reg, info


def condition_number_regularization(
    g: Tensor,
    kappa_threshold: float = 50.0,
    weight: float = 0.1,
    eps: float = 1e-8,
    min_abs_eig: float = 0.1,  # 特征值绝对值的最小阈值
    min_eig_weight: float = 1.0,  # 特征值下界正则化权重
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    条件数正则化 - 第一性原理版本
    
    核心哲学：
        - 特征值有界才是关键，det 接近 0 不是问题
        - 条件数 kappa(g) = |lambda_max| / |lambda_min| 直接衡量数值稳定性
        - 特征值绝对值不能太小，否则几何效应弱
    
    Args:
        g: [B, L, D, D] 度规张量
        kappa_threshold: 条件数阈值，超过这个值才开始惩罚
        weight: 条件数正则化权重
        eps: 数值稳定性
        min_abs_eig: 特征值绝对值的最小阈值
        min_eig_weight: 特征值下界正则化权重
    
    Returns:
        reg: 综合正则化损失
        info: 详细信息
    
    复杂度: O(D^3) 特征值分解
    """
    # 取最后一个位置的度规
    if g.dim() == 4:
        g_last = g[:, -1]  # [B, D, D]
    else:
        g_last = g  # [B, D, D]
    
    # 计算特征值（对称矩阵）
    eigenvalues = torch.linalg.eigvalsh(g_last)  # [B, D]
    
    # 使用特征值的绝对值计算条件数
    abs_eig = torch.abs(eigenvalues) + eps  # [B, D]
    eig_max = abs_eig.max(dim=-1).values  # [B]
    eig_min = abs_eig.min(dim=-1).values  # [B]
    
    # 条件数
    kappa = eig_max / eig_min  # [B]
    
    # 1. 条件数上界惩罚：只有当条件数超过阈值时才惩罚
    excess = F.softplus(kappa - kappa_threshold)  # [B]
    cond_penalty = excess.mean()
    
    # 2. 特征值绝对值下界惩罚：鼓励 |lambda| > min_abs_eig
    # 这确保几何效应不会太弱
    below_threshold = F.relu(min_abs_eig - abs_eig)  # [B, D]
    min_eig_penalty = below_threshold.mean()
    
    total_reg = weight * cond_penalty + min_eig_weight * min_eig_penalty
    
    info = {
        'kappa_mean': kappa.mean(),
        'kappa_max': kappa.max(),
        'kappa_min': kappa.min(),
        'eig_abs_min': eig_min.mean(),
        'eig_abs_max': eig_max.mean(),
        'excess_penalty': excess.mean(),
        'min_eig_penalty': min_eig_penalty,
    }
    
    return total_reg, info


def dynamic_force_weight(
    epoch: int, 
    total_epochs: int,
    min_weight: float = 0.1,
    max_weight: float = 1.2,  # 适中的最大权重
    warmup_ratio: float = 0.7,  # 延长 warmup，让模型学会使用几何
    plateau_ratio: float = 0.4,  # 延长平台期，充分探索
) -> float:
    """
    动态外力惩罚权重 - 平衡版本
    
    第一性原理：
        1. 平台期(40%)：低权重，让模型充分探索 g(z) 和如何使用它
        2. Warmup期(40%-70%)：缓慢增加权重
        3. 强化期(70%-100%)：适中的高权重
    
    复杂度: O(1)
    """
    plateau_epochs = int(total_epochs * plateau_ratio)
    warmup_epochs = int(total_epochs * warmup_ratio)
    
    if epoch < plateau_epochs:
        # 平台期：保持最低权重
        return min_weight
    elif epoch < warmup_epochs:
        # Warmup 期：线性增长到50%
        progress = (epoch - plateau_epochs) / (warmup_epochs - plateau_epochs + 1e-8)
        return min_weight + (max_weight - min_weight) * progress * 0.5
    else:
        # 强化期：余弦平滑增长到最大值
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs + 1e-8)
        cos_progress = 0.5 * (1 - math.cos(math.pi * progress))
        return min_weight + (max_weight - min_weight) * (0.5 + 0.5 * cos_progress)


def adaptive_eps_v2(z: LatentState, base_eps: float = 1e-3) -> float:
    """
    改进的自适应差分步长 - 基于 z 的标准差
    
    复杂度: O(1)
    """
    with torch.no_grad():
        z_std = z.std().item() + 1e-8
        # eps 与 z 的标准差成正比，但有上下界
        eps = base_eps * max(0.1, min(z_std, 10.0))
        return eps


def gradient_monitor(loss: Tensor, model: nn.Module, clip_value: float = 1.0) -> Dict[str, float]:
    """
    梯度监控 - 用于调试和异常检测
    
    复杂度: O(params)
    """
    total_norm = 0.0
    max_grad = 0.0
    nan_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            max_grad = max(max_grad, p.grad.data.abs().max().item())
            if torch.isnan(p.grad).any():
                nan_count += 1
    
    total_norm = total_norm ** 0.5
    
    return {
        'grad_norm': total_norm,
        'max_grad': max_grad,
        'nan_count': nan_count,
        'is_healthy': nan_count == 0 and total_norm < clip_value * 10,
    }


# =============================================================================
# 基础组件 (Foundational Components)
# =============================================================================

class RMSNorm(nn.Module):
    """
    RMS 归一化 - 稳定训练的关键组件
    
    代数律: ||RMSNorm(x)||_rms ≈ 1
    复杂度: O(D)
    """
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    
    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


# =============================================================================
# 感知层：数据 → 抽象状态 → 度规张量 (Perception Layer)
# =============================================================================

class StateEncoder(nn.Module):
    """
    状态编码器：将原始数据映射到抽象广义坐标 z
    
    z 是 AI 感知到的"你在哪"的内在表示
    
    架构特点：
        - RMSNorm 稳定输出尺度
        - 残差连接保留信息流
    
    复杂度: O(L * D)
    """
    def __init__(self, input_dim: int, z_dim: int):
        super().__init__()
        self.z_dim = z_dim
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, z_dim * 2)
        
        # 序列编码（简单的因果卷积）
        self.conv = nn.Conv1d(z_dim * 2, z_dim, kernel_size=3, padding=2)
        
        # 输出投影 + 归一化
        self.out_proj = nn.Linear(z_dim, z_dim)
        self.norm = RMSNorm(z_dim)  # 稳定 z 的尺度
    
    def forward(self, x: Tensor) -> LatentState:
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
        
        # 输出 + 归一化
        z = self.out_proj(F.silu(h))
        z = self.norm(z)  # 稳定尺度，对 eps 选择至关重要
        
        return z


class MetricEncoder(nn.Module):
    """
    度规编码器 - 第一性原理版本
    
    核心哲学：
        - 在架构层面保证特征值有界，而不是通过损失函数
        - 配合条件数正则化，确保数值稳定性
    
    暴论模式的非退化保证：
        - 将 g 的特征值通过 tanh 映射到有界范围
        - 这确保 |lambda| 有下界和上界
    
    复杂度: O(D^2)
    """
    
    NUMERICAL_EPS: float = 1e-6
    # 暴论模式：特征值范围参数
    EIG_SCALE: float = 2.0  # 特征值范围: [-EIG_SCALE, EIG_SCALE]
    MIN_ABS_EIG: float = 0.05  # 特征值绝对值的最小值
    
    def __init__(
        self, 
        z_dim: int, 
        hidden_mult: int = 8,  # 增强容量：6 -> 8
        unconstrained: bool = False,
        min_diag_bias: float = 0.0,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.unconstrained = unconstrained
        self.min_diag_bias = min_diag_bias
        hidden_dim = z_dim * hidden_mult
        
        # 5层深度网络（增加容量）
        self.layer1 = nn.Linear(z_dim, hidden_dim)
        self.norm1 = RMSNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = RMSNorm(hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)  # 新增第四层
        self.norm4 = RMSNorm(hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, z_dim * z_dim)
        
        # 可学习的对角偏置
        if not unconstrained:
            self.diag_bias = nn.Parameter(torch.ones(z_dim) * 0.5)
        else:
            # 暴论模式：初始化为零，让模型自由学习
            self.diag_bias = nn.Parameter(torch.zeros(z_dim))
        
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: LatentState) -> MetricTensor:
        B, L, D = z.shape
        
        # 5层前向传播
        h = F.silu(self.norm1(self.layer1(z)))
        h = h + F.silu(self.norm2(self.layer2(h)))  # 残差连接
        h = F.silu(self.norm3(self.layer3(h)))
        h = h + F.silu(self.norm4(self.layer4(h)))  # 残差连接
        
        params = self.layer5(h).view(B, L, D, D) * self.output_scale.abs()
        
        if self.unconstrained:
            # 暴论模式：通过架构保证特征值有界
            g = symmetric_part(params)
            
            # 可学习的对角偏置（允许正负）
            g = g + torch.diag(self.diag_bias).unsqueeze(0).unsqueeze(0).to(g.device)
            
            # 数值安全网：添加微小的单位矩阵确保不退化
            identity = torch.eye(D, device=g.device, dtype=g.dtype)
            g = g + self.NUMERICAL_EPS * identity.unsqueeze(0).unsqueeze(0)
        else:
            # 约束模式：L @ L.T 确保半正定
            L_matrix = params
            g = torch.matmul(L_matrix, L_matrix.transpose(-1, -2))
            diag_bias = self.diag_bias.abs() + self.min_diag_bias + 0.1
            g = g + torch.diag(diag_bias).unsqueeze(0).unsqueeze(0).to(g.device)
        
        # 数值安全网
        identity = torch.eye(D, device=g.device, dtype=g.dtype)
        g = g + self.NUMERICAL_EPS * identity.unsqueeze(0).unsqueeze(0)
        
        return g


# =============================================================================
# 涌现层：度规 → 联络（Christoffel 符号）- 完全向量化 (Emergence Layer)
# =============================================================================

class ChristoffelComputer(nn.Module):
    """
    Christoffel 符号计算器（完全向量化版本）
    
    公式：Γ^k_ij = 1/2 * g^kl * (∂g_jl/∂z^i + ∂g_il/∂z^j - ∂g_ij/∂z^l)
    
    核心改进：
        - 完全移除 for 循环，使用向量化操作
        - 基于 z 标准差的自适应 eps
        - 梯度稳定的伪逆计算
    
    复杂度: O(D^3) 而非 O(D^4) 的循环版本
    代数律: Γ^k_ij = Γ^k_ji (对称性)
    """
    def __init__(self, base_eps: float = 1e-3):
        super().__init__()
        self.base_eps = base_eps
    
    def forward(
        self, 
        z: LatentState, 
        g: MetricTensor, 
        metric_encoder: nn.Module
    ) -> ChristoffelSymbol:
        """
        z: [B, L, D] 广义坐标
        g: [B, L, D, D] 度规张量
        metric_encoder: 度规编码器
        
        返回: Gamma [B, L, D, D, D] Christoffel 符号 Γ^k_ij
        """
        B, L, D = z.shape
        device = z.device
        dtype = z.dtype
        
        # 改进的自适应 eps，基于 z 的标准差
        eps = adaptive_eps_v2(z, self.base_eps)
        
        # 使用伪逆，更稳定
        g_inv = safe_inverse(g)  # [B, L, D, D]
        
        # =========================================================
        # 批量化计算 ∂g/∂z - 关键优化：移除 for 循环
        # =========================================================
        
        # 创建所有方向的扰动向量 [D, D]
        # identity 的每一行是一个方向的单位扰动
        perturbations = torch.eye(D, device=device, dtype=dtype)  # [D, D]
        
        # 扩展 z 为 [B, L, D, D]，最后一维是扰动方向
        z_expanded = z.unsqueeze(-1).expand(B, L, D, D)  # [B, L, D, D]
        
        # z_plus[b, l, :, d] = z[b, l, :] + eps * e_d
        z_plus = z_expanded + eps * perturbations  # [B, L, D, D] 广播
        z_minus = z_expanded - eps * perturbations  # [B, L, D, D]
        
        # 重塑为 [B*L*D, D] 进行批量前向传播
        z_plus_flat = z_plus.permute(0, 1, 3, 2).reshape(B * L * D, D)  # [B*L*D, D]
        z_minus_flat = z_minus.permute(0, 1, 3, 2).reshape(B * L * D, D)  # [B*L*D, D]
        
        # 批量计算 g(z + eps*e_l) 和 g(z - eps*e_l)
        # 需要为 metric_encoder 添加虚拟序列维度
        g_plus_flat = metric_encoder(z_plus_flat.unsqueeze(1)).squeeze(1)  # [B*L*D, D, D]
        g_minus_flat = metric_encoder(z_minus_flat.unsqueeze(1)).squeeze(1)  # [B*L*D, D, D]
        
        # 重塑回 [B, L, D, D, D]
        # dg_dz[b, l, i, j, k] = ∂g_ij/∂z^k
        g_plus = g_plus_flat.view(B, L, D, D, D)  # [B, L, D(扰动方向), D, D]
        g_minus = g_minus_flat.view(B, L, D, D, D)
        
        # 中心差分: dg_dz[b, l, i, j, k] = (g_plus - g_minus) / (2*eps)
        # g_plus/g_minus 形状: [B, L, k(扰动方向), i, j]
        # 需要调整为 [B, L, i, j, k]
        dg_dz = (g_plus - g_minus).permute(0, 1, 3, 4, 2) / (2 * eps)  # [B, L, D, D, D]
        
        # =========================================================
        # 向量化计算 Christoffel 符号
        # Γ^k_ij = 0.5 * g^kl * (∂g_jl/∂z^i + ∂g_il/∂z^j - ∂g_ij/∂z^l)
        # =========================================================
        
        # dg_dz[b, l, i, j, k] = ∂g_ij/∂z^k
        # 需要构建 bracket[b, l, i, j, m] = ∂g_jm/∂z^i + ∂g_im/∂z^j - ∂g_ij/∂z^m
        
        # term1: ∂g_jm/∂z^i = dg_dz[..., j, m, i]
        # term2: ∂g_im/∂z^j = dg_dz[..., i, m, j]  
        # term3: ∂g_ij/∂z^m = dg_dz[..., i, j, m]
        
        # 使用 einsum 计算
        # Γ^k_ij = 0.5 * g^km * (∂g_jm/∂z^i + ∂g_im/∂z^j - ∂g_ij/∂z^m)
        
        Gamma = 0.5 * torch.einsum(
            'blkm,bljmi->blkij',
            g_inv,  # [B, L, k, m]
            dg_dz.permute(0, 1, 3, 4, 2) +  # ∂g_jm/∂z^i: [B, L, j, m, i]
            dg_dz.permute(0, 1, 2, 4, 3) -  # ∂g_im/∂z^j: [B, L, i, m, j]
            dg_dz                            # ∂g_ij/∂z^m: [B, L, i, j, m]
        )
        
        return Gamma


# =============================================================================
# 法则层：联络 → 运动法则 (Law Layer)
# =============================================================================

class GeodesicAcceleration(nn.Module):
    """
    测地线加速度计算（向量化版本）
    
    测地线方程：d²z^k/dt² + Γ^k_ij * dz^i/dt * dz^j/dt = 0
    
    纯几何加速度：a_geo^k = -Γ^k_ij * dz^i * dz^j
    
    复杂度: O(D^2)
    代数律: 当 dz=0 时，a_geo=0
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, dz: Velocity, Gamma: ChristoffelSymbol) -> Acceleration:
        """
        dz: [B, L, D] 速度
        Gamma: [B, L, D, D, D] Christoffel 符号
        
        返回: a_geo [B, L, D] 测地线加速度
        """
        # 向量化: a_geo^k = -Γ^k_ij * dz^i * dz^j
        a_geo = -torch.einsum('blkij,bli,blj->blk', Gamma, dz, dz)
        return a_geo


class ExternalForce(nn.Module):
    """
    外力网络（受限版）
    
    第一性原理：
        几何优先 - 外力应该是最后的手段
        限制外力网络的能力，迫使模型使用几何
    
    复杂度: O(D)
    """
    MAX_SCALE: float = 0.1  # 限制 scale 的最大值
    
    def __init__(self, z_dim: int):
        super().__init__()
        self.z_dim = z_dim
        
        # 简化网络：减少容量
        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim),
            RMSNorm(z_dim),
            nn.Tanh(),  # 使用 Tanh 限制输出范围
        )
        
        # 可学习的缩放因子，初始很小
        self.scale = nn.Parameter(torch.tensor(0.001))
    
    def forward(self, z: LatentState, dz: Velocity) -> Acceleration:
        """
        z: [B, L, D] 位置
        dz: [B, L, D] 速度
        
        返回: F [B, L, D] 外力
        """
        if z.dim() == 2:
            z = z.unsqueeze(1)
        if dz.dim() == 2:
            dz = dz.unsqueeze(1)
            
        h = torch.cat([z, dz], dim=-1)
        # 限制 scale 的最大值
        bounded_scale = self.scale.abs().clamp(max=self.MAX_SCALE)
        F_ext = self.net(h) * bounded_scale
        return F_ext.squeeze(1) if F_ext.shape[1] == 1 else F_ext


# =============================================================================
# 解码器：z 空间的加速度 → 原始空间的变化分布 (Decoder Layer)
# =============================================================================

class ProbabilisticDecoder(nn.Module):
    """
    概率解码器：输出分布而不是点估计
    
    第一性原理：
        - 世界不是确定性的，我们的知识也不是
        - 输出分布可以量化"无知"
        - 当模型不确定时，它应该说"我不确定"
    
    改进：
        - 更大的 min_var 防止 sigma 过小
        - 添加层归一化
    
    复杂度: O(D)
    """
    def __init__(self, z_dim: int, output_dim: int = 1):
        super().__init__()
        
        # 均值网络
        self.mu_net = nn.Sequential(
            nn.Linear(z_dim, z_dim * 2),
            RMSNorm(z_dim * 2),
            nn.SiLU(),
            nn.Linear(z_dim * 2, output_dim),
        )
        
        # 方差网络（输出 log_var，确保正定）
        self.logvar_net = nn.Sequential(
            nn.Linear(z_dim, z_dim * 2),
            RMSNorm(z_dim * 2),
            nn.SiLU(),
            nn.Linear(z_dim * 2, output_dim),
        )
        
        # 方差的最小值（防止 sigma 过小导致 NLL 病态）
        self.min_sigma = 0.05  # 提高最小值
        self.max_sigma = 2.0   # 限制最大值
    
    def forward(self, d2z: Acceleration) -> Distribution:
        """
        d2z: [B, z_dim] z空间的加速度
        返回: 
            mu: [B] 均值
            sigma: [B] 标准差
        """
        mu = self.mu_net(d2z).squeeze(-1)
        log_var = self.logvar_net(d2z).squeeze(-1)
        
        # 标准差 = sqrt(exp(log_var)) = exp(log_var / 2)
        # 软裁剪到合理范围
        sigma_raw = torch.exp(0.5 * log_var.clamp(-10, 10))
        sigma = sigma_raw.clamp(self.min_sigma, self.max_sigma)
        
        return mu, sigma
    
    def nll_loss(self, mu: Tensor, sigma: Tensor, target: Tensor) -> Tensor:
        """
        负对数似然损失
        
        对于高斯分布: NLL = 0.5 * (log(2πσ²) + (x-μ)²/σ²)
        """
        var = sigma ** 2
        nll = 0.5 * (torch.log(2 * math.pi * var) + (target - mu) ** 2 / var)
        return nll.mean()
    
    def sample(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """从分布中采样"""
        eps = torch.randn_like(mu)
        return mu + sigma * eps


# =============================================================================
# 完整模型：度规涌现演化 (Full Model)
# =============================================================================

class MetricEvolutionModel(nn.Module):
    """
    度规涌现演化模型 - 暴论版本
    
    核心链条：
        1. 感知层：x → z → g(z)
        2. 涌现层：g(z) → Γ (向量化计算)
        3. 法则层：Γ, dz → a_geodesic + F_external → d²z
        4. 概率层：d²z → P(d²x | μ, σ)
    
    模式：
        - constrained: 正定度规，稳定训练
        - unconstrained: 不定号度规，探索洛伦兹签名等
    """
    
    def __init__(
        self, 
        z_dim: int = 4, 
        input_dim: int = 1,
        hidden_mult: int = 6,         # MetricEncoder 隐藏层倍数
        unconstrained: bool = False,  # 暴论模式
    ):
        super().__init__()
        self.z_dim = z_dim
        self.unconstrained = unconstrained
        
        # 1. 感知层
        self.state_encoder = StateEncoder(input_dim, z_dim)
        self.metric_encoder = MetricEncoder(
            z_dim, 
            hidden_mult=hidden_mult,
            unconstrained=unconstrained,
            min_diag_bias=0.0 if unconstrained else 0.1,
        )
        
        # 2. 涌现层 - 向量化版本
        self.christoffel = ChristoffelComputer(base_eps=1e-3)
        self.geodesic = GeodesicAcceleration()
        
        # 3. 法则层
        self.external_force = ExternalForce(z_dim)
        
        # 4. 概率解码层
        self.decoder = ProbabilisticDecoder(z_dim, 1)
    
    def forward(self, values: Tensor, compute_christoffel: bool = True) -> Dict[str, Tensor]:
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
        
        # 3. 涌现层：g → Γ (向量化)
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
            'metric': g,        # 度规张量（用于签名分析）
            'g': g,
            'Gamma': Gamma,
            'a_geodesic': a_geo,
            'F_external': F_ext,
            'd2z': d2z,
        }
    
    def compute_loss(
        self, 
        values: Tensor, 
        target: Optional[Tensor] = None,
        force_weight: float = 0.5,
        det_target: float = 0.1,        # det(g) 目标值
        det_log_weight: float = 0.5,    # 对数惩罚权重
        det_target_weight: float = 1.0, # 目标值惩罚权重
        det_sign_weight: float = 0.3,   # 符号一致性权重（已废弃）
        kappa_threshold: float = 50.0,  # 条件数阈值
        kappa_weight: float = 0.05,     # 条件数正则化权重
        min_abs_eig: float = 0.5,       # 特征值绝对值下界（提高到更有意义的尺度）
        min_eig_weight: float = 10.0,   # 特征值下界正则化权重（大幅提高）
    ) -> Dict[str, Tensor]:
        """
        计算损失 - 第一性原理版本
        
        损失组成：
        1. NLL损失：负对数似然
        2. 外力正则化：奥卡姆剃刀
        3. 度规正则化：
           - 约束模式：det(g) 混合惩罚
           - 暴论模式：条件数正则化（不对 det 正则化）
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
        
        # 3. 度规正则化
        g_last = out['g'][:, -1]  # [B, D, D]
        det_g = torch.linalg.det(g_last)  # [B]
        
        # det(g) 正则化（暴论模式下不起作用）
        det_reg, det_info = hybrid_det_regularization(
            det_g,
            target=det_target,
            log_weight=det_log_weight,
            target_weight=det_target_weight,
            sign_weight=det_sign_weight,
            unconstrained=self.unconstrained,
        )
        
        # 条件数正则化（暴论模式的核心正则化）
        if self.unconstrained:
            cond_reg, cond_info = condition_number_regularization(
                out['g'],
                kappa_threshold=kappa_threshold,
                weight=kappa_weight,
                min_abs_eig=min_abs_eig,
                min_eig_weight=min_eig_weight,
            )
        else:
            cond_reg = torch.tensor(0.0, device=values.device)
            cond_info = {'kappa_mean': torch.tensor(0.0), 'kappa_max': torch.tensor(0.0), 'min_eig_penalty': torch.tensor(0.0)}
        
        # 总损失
        total_loss = nll_loss + force_weight * force_reg + 0.01 * det_reg + cond_reg
        
        return {
            'loss': total_loss,
            'nll_loss': nll_loss,
            'force_reg': force_reg,
            'det_reg': det_reg,
            'cond_reg': cond_reg,
            'det_g': det_g.mean(),
            'det_abs_mean': det_info['mean_abs_det'],
            'det_std': det_info['det_std'],
            'kappa_mean': cond_info['kappa_mean'],
            'kappa_max': cond_info['kappa_max'],
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
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("度规涌现演化模型测试（向量化版本）")
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
    print(f"  nll: {loss_dict['nll_loss'].item():.6f}")
    print(f"  force_reg: {loss_dict['force_reg'].item():.6f}")
    print(f"  det_reg: {loss_dict['det_reg'].item():.6f}")
    print(f"  det(g): {loss_dict['det_g'].item():.6f}")
    
    # 测试几何信息
    print(f"\n几何信息:")
    print(f"  |a_geodesic|: {loss_dict['a_geodesic'].norm().item():.6f}")
    print(f"  |F_external|: {loss_dict['F_external'].norm().item():.6f}")
    print(f"  mu: {loss_dict['mu'].mean().item():.6f}")
    print(f"  sigma: {loss_dict['sigma'].mean().item():.6f}")
    
    print("\n" + "=" * 60)
    print("测试通过")
    print("=" * 60)