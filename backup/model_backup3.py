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
    sign_weight: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    混合行列式正则化 - 解决训练不稳定性的关键
    
    组合三种惩罚：
    1. 对数惩罚：防止 det(g) 接近 0，梯度平滑
    2. 目标值惩罚：鼓励 |det(g)| 接近目标值
    3. 符号一致性惩罚：鼓励所有 det(g) 保持同符号（正定/负定）
    
    复杂度: O(B)
    代数律: 当 det(g) = target 且符号一致时取最小值
    """
    abs_det = torch.abs(det_g) + eps
    
    # 1. 对数惩罚：防止接近 0
    log_penalty = -torch.log(abs_det).mean()
    
    # 2. 目标值惩罚：鼓励接近目标值
    target_penalty = ((abs_det - target) ** 2).mean()
    
    # 3. 符号一致性惩罚：鼓励所有 det 同符号
    # 当 batch_size > 1 时才计算方差
    if det_g.numel() > 1:
        sign_variance = torch.var(torch.sign(det_g + eps), unbiased=False)
    else:
        sign_variance = torch.tensor(0.0, device=det_g.device, dtype=det_g.dtype)
    
    # 总正则化
    total_reg = (
        log_weight * log_penalty + 
        target_weight * target_penalty + 
        sign_weight * sign_variance
    )
    
    # 返回详细信息用于调试
    info = {
        'log_penalty': log_penalty,
        'target_penalty': target_penalty,
        'sign_variance': sign_variance,
        'mean_abs_det': abs_det.mean(),
        'det_std': det_g.std() if det_g.numel() > 1 else torch.tensor(0.0, device=det_g.device),
    }
    
    return total_reg, info


def dynamic_force_weight(
    epoch: int, 
    total_epochs: int,
    min_weight: float = 0.1,
    max_weight: float = 1.0,
    warmup_ratio: float = 0.3,
) -> float:
    """
    动态外力惩罚权重 - warmup 策略
    
    训练初期：低权重，让模型探索 g(z)
    训练后期：高权重，强制几何解释
    
    复杂度: O(1)
    """
    warmup_epochs = int(total_epochs * warmup_ratio)
    
    if epoch < warmup_epochs:
        # Warmup 阶段：线性增长
        progress = epoch / warmup_epochs
        return min_weight + (max_weight - min_weight) * progress * 0.5
    else:
        # 后期阶段：余弦退火到最大值
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_weight + (max_weight - min_weight) * (0.5 + 0.5 * (1 - math.cos(math.pi * progress)) / 2)


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
    度规编码器：从 z 生成度规张量 g(z) - 增强版
    
    改进：
        - 更深的网络（3层）
        - 残差连接稳定训练
        - 更强的对角偏置确保非奇异
        - 多层归一化
    
    架构特点：
        - 输入 z → 隐藏层 → 输出 g
        - 每层后有 RMSNorm
        - 可学习的对角偏置和缩放
    
    复杂度: O(D^2)
    """
    def __init__(self, z_dim: int, hidden_mult: int = 4):
        super().__init__()
        self.z_dim = z_dim
        hidden_dim = z_dim * hidden_mult
        
        # 更深的网络，每层带归一化
        self.layer1 = nn.Linear(z_dim, hidden_dim)
        self.norm1 = RMSNorm(hidden_dim)
        
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        
        self.layer3 = nn.Linear(hidden_dim, z_dim * z_dim)
        
        # 可学习的对角偏置 - 确保非奇异，初始化为较大值
        self.diag_bias = nn.Parameter(torch.ones(z_dim) * 0.5)
        
        # 输出缩放因子，防止度规爆炸
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """特殊初始化，确保初始度规接近单位矩阵"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: LatentState) -> MetricTensor:
        """
        z: [B, L, z_dim] 广义坐标
        返回: g [B, L, z_dim, z_dim] 对称度规张量
        """
        B, L, D = z.shape
        
        # 前向传播，带归一化
        h = self.layer1(z)
        h = self.norm1(h)
        h = F.silu(h)
        
        h = self.layer2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # 生成矩阵参数
        params = self.layer3(h)  # [B, L, D*D]
        params = params.view(B, L, D, D) * self.output_scale.abs()
        
        # 构造对称矩阵
        g = symmetric_part(params)
        
        # 添加可学习的对角偏置，确保非奇异
        diag_matrix = torch.diag(self.diag_bias.abs() + 0.1)  # 至少 0.1
        g = g + diag_matrix.unsqueeze(0).unsqueeze(0)
        
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
    外力网络（改进版）
    
    改进：
        - 添加层归一化稳定输出
        - 可学习的缩放因子
        - 初始化优化，开始时输出较小
    
    复杂度: O(D)
    """
    def __init__(self, z_dim: int):
        super().__init__()
        self.z_dim = z_dim
        
        # 改进网络：添加归一化
        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim * 2),
            RMSNorm(z_dim * 2),
            nn.SiLU(),
            nn.Linear(z_dim * 2, z_dim),
        )
        
        # 可学习的缩放因子，初始较小
        self.scale = nn.Parameter(torch.tensor(0.01))
    
    def forward(self, z: LatentState, dz: Velocity) -> Acceleration:
        """
        z: [B, L, D] 位置
        dz: [B, L, D] 速度
        
        返回: F [B, L, D] 外力
        """
        # 处理维度
        if z.dim() == 2:
            z = z.unsqueeze(1)
        if dz.dim() == 2:
            dz = dz.unsqueeze(1)
            
        h = torch.cat([z, dz], dim=-1)
        F_ext = self.net(h) * self.scale.abs()  # 绝对值保证正
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
    度规涌现演化模型 (Pure FP Architecture)
    
    核心链条：
        1. 感知层：x → z → g(z)
        2. 涌现层：g(z) → Γ (向量化计算)
        3. 法则层：Γ, dz → a_geodesic + F_external → d²z
        4. 概率层：d²z → P(d²x | μ, σ)
    
    改进：
        - 向量化 Christoffel 计算（无 for 循环）
        - 温和的 det(g) 正则化（对数形式）
        - 自适应 eps
        - RMSNorm 稳定训练
    """
    
    def __init__(self, z_dim: int = 4, input_dim: int = 1):
        super().__init__()
        self.z_dim = z_dim
        
        # 1. 感知层
        self.state_encoder = StateEncoder(input_dim, z_dim)
        self.metric_encoder = MetricEncoder(z_dim)
        
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
        det_sign_weight: float = 0.3,   # 符号一致性权重
    ) -> Dict[str, Tensor]:
        """
        计算损失 - 混合正则化策略
        
        损失组成：
        1. NLL损失：负对数似然
        2. 外力正则化：奥卡姆剃刀
        3. 度规正则化：混合惩罚（对数 + 目标值 + 符号一致性）
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
        
        # 3. 混合 det(g) 正则化
        g_last = out['g'][:, -1]  # [B, D, D]
        det_g = torch.linalg.det(g_last)  # [B]
        
        det_reg, det_info = hybrid_det_regularization(
            det_g,
            target=det_target,
            log_weight=det_log_weight,
            target_weight=det_target_weight,
            sign_weight=det_sign_weight,
        )
        
        # 总损失
        total_loss = nll_loss + force_weight * force_reg + 0.01 * det_reg
        
        return {
            'loss': total_loss,
            'nll_loss': nll_loss,
            'force_reg': force_reg,
            'det_reg': det_reg,
            'det_g': det_g.mean(),
            'det_abs_mean': det_info['mean_abs_det'],
            'det_std': det_info['det_std'],
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