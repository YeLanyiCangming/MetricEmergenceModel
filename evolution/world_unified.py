"""
world_unified.py - 统一世界模型

第一性原理：
    世界动力学 → 广义坐标 → 黎曼几何 → 演化预测

完整链条：
    1. WorldDynamics: 数据 X → (N, D, K, L) 物理系统
    2. ModalTransform: K·Φ = N·Φ·Λ → z = ΦᵀNx 广义坐标
    3. EnergyGeometry: V(z) → g(z) = Hessian(V) 度规涌现
    4. GeodesicEvolution: Γ → a = -Γᵏᵢⱼvⁱvʲ 测地线演化
    5. Decoder: z, a → 预测

可学习参数（物理约束下的自由度）：
    - β_func, β_struct: 图耦合强度
    - γ: 阻尼系数
    - α: 非线性势能强度
    - F_ext: 外力（小容量）
    - decoder: 输出映射
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.linalg import eigh


# =============================================================================
# 第一层：世界动力学（从数据构建物理系统）
# =============================================================================

class WorldDynamicsBuilder(nn.Module):
    """
    从数据构建世界动力学系统
    
    N·ẍ + D·ẋ + K·x = F(t)
    K = diag(ω²) + β_func·L_func + β_struct·L_struct
    
    输入：原始数据 X (B, T, C)
    输出：物理系统 (N, D, K, L, ω)
    """
    
    def __init__(
        self,
        n_channels: int,
        beta_func_init: float = 1.0,
        beta_struct_init: float = 0.0,
        gamma_init: float = 0.1,
        dt: float = 1.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.dt = dt
        
        # 可学习的物理参数
        self.log_beta_func = nn.Parameter(torch.tensor(np.log(beta_func_init + 1e-6)))
        self.log_beta_struct = nn.Parameter(torch.tensor(np.log(beta_struct_init + 1e-6)))
        self.log_gamma = nn.Parameter(torch.tensor(np.log(gamma_init + 1e-6)))
        
        # 结构性图（先验，可选）
        self.register_buffer('L_struct', torch.zeros(n_channels, n_channels))
    
    @property
    def beta_func(self):
        return torch.exp(self.log_beta_func)
    
    @property
    def beta_struct(self):
        return torch.exp(self.log_beta_struct)
    
    @property
    def gamma(self):
        return torch.exp(self.log_gamma)
    
    def estimate_local_frequencies(self, X: torch.Tensor) -> torch.Tensor:
        """
        估计每个通道的本地频率（FFT）
        
        X: (B, T, C)
        返回: (C,) 本地频率
        """
        X_np = X.detach().cpu().numpy()
        B, T, C = X_np.shape
        
        freqs = np.zeros(C)
        for c in range(C):
            # 平均跨 batch
            power_sum = np.zeros(T // 2 + 1)
            for b in range(B):
                fft = np.fft.rfft(X_np[b, :, c])
                power_sum += np.abs(fft) ** 2
            
            freq_axis = np.fft.rfftfreq(T, self.dt)
            power_sum[0] = 0  # 排除直流
            freqs[c] = freq_axis[np.argmax(power_sum)]
        
        return torch.tensor(freqs, device=X.device, dtype=X.dtype)
    
    def build_functional_laplacian(self, X: torch.Tensor) -> torch.Tensor:
        """
        从数据协方差构建功能性图拉普拉斯
        
        X: (B, T, C)
        返回: (C, C) 归一化拉普拉斯
        """
        # 计算协方差
        X_centered = X - X.mean(dim=1, keepdim=True)
        C = torch.bmm(X_centered.transpose(1, 2), X_centered) / X.shape[1]
        C = C.mean(dim=0)  # (C, C)
        
        # 邻接矩阵
        A = torch.abs(C)
        A = A - torch.diag(torch.diag(A))
        
        # 归一化拉普拉斯
        D_vec = A.sum(dim=1) + 1e-10
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_vec))
        L = torch.eye(self.n_channels, device=X.device) - D_inv_sqrt @ A @ D_inv_sqrt
        
        return L
    
    def build_mass_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """质量矩阵 N（简化为单位矩阵）"""
        return torch.eye(self.n_channels, device=X.device, dtype=X.dtype)
    
    def build_stiffness_matrix(
        self, 
        omega_local: torch.Tensor, 
        L_func: torch.Tensor
    ) -> torch.Tensor:
        """
        刚度矩阵 K = diag(ω²) + β_func·L_func + β_struct·L_struct
        """
        omega_sq = (2 * np.pi * omega_local) ** 2
        K = torch.diag(omega_sq) + self.beta_func * L_func + self.beta_struct * self.L_struct
        return K
    
    def build_damping_matrix(self, N: torch.Tensor) -> torch.Tensor:
        """阻尼矩阵 D = γ·N"""
        return self.gamma * N
    
    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        从数据构建完整的世界动力学系统
        
        X: (B, T, C) 原始数据
        返回: 物理系统参数
        """
        # 1. 本地频率
        omega_local = self.estimate_local_frequencies(X)
        
        # 2. 功能性图
        L_func = self.build_functional_laplacian(X)
        
        # 3. 物理矩阵
        N = self.build_mass_matrix(X)
        K = self.build_stiffness_matrix(omega_local, L_func)
        D = self.build_damping_matrix(N)
        
        return {
            'N': N,
            'D': D,
            'K': K,
            'L_func': L_func,
            'omega_local': omega_local,
            'beta_func': self.beta_func,
            'beta_struct': self.beta_struct,
            'gamma': self.gamma,
        }


# =============================================================================
# 第二层：模态变换（物理定义的广义坐标）
# =============================================================================

class ModalTransform(nn.Module):
    """
    模态变换：x → z = ΦᵀNx
    
    物理原理：
        广义特征值问题 K·Φ = N·Φ·Λ
        模态坐标 z = ΦᵀNx 对角化动力学
    
    这不是学习的，是从 (N, K) 计算得到的！
    """
    
    def __init__(self, n_modes: int = None):
        super().__init__()
        self.n_modes = n_modes  # 可选：截断到前 n_modes 个模态
        
        # 缓存模态矩阵
        self.register_buffer('Phi', None)
        self.register_buffer('Lambda', None)
        self.register_buffer('N_cache', None)
    
    def compute_modes(self, K: torch.Tensor, N: torch.Tensor, use_differentiable: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        求解广义特征值问题 K·Φ = N·Φ·Λ
        
        当 N=I（单位矩阵）时，使用 PyTorch 可微分特征分解
        否则使用 scipy（会断开梯度）
        
        返回: (Λ, Φ) 特征值和特征向量
        """
        # 检查 N 是否为单位矩阵（允许梯度传播）
        is_identity = torch.allclose(N, torch.eye(N.shape[0], device=N.device, dtype=N.dtype), atol=1e-5)
        
        if is_identity and use_differentiable:
            # N=I 时，广义特征问题退化为标准特征问题：K·Φ = Φ·Λ
            # PyTorch 的 eigh 是可微分的！
            
            # 确保 K 对称（数值稳定性）
            K_sym = 0.5 * (K + K.T)
            
            # 可微分特征分解
            Lambda, Phi = torch.linalg.eigh(K_sym)
            
            # 确保正定（软约束，保持可微）
            Lambda = torch.clamp(Lambda, min=1e-6)
            
        else:
            # 广义特征分解（使用 scipy，会断开梯度）
            K_np = K.detach().cpu().numpy()
            N_np = N.detach().cpu().numpy()
            
            eigenvalues, eigenvectors = eigh(K_np, N_np)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            
            Lambda = torch.tensor(eigenvalues, device=K.device, dtype=K.dtype)
            Phi = torch.tensor(eigenvectors, device=K.device, dtype=K.dtype)
        
        return Lambda, Phi
    
    def x_to_z(self, x: torch.Tensor, Phi: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        """
        观测空间 → 广义坐标
        
        z = ΦᵀNx
        
        x: (B, T, C) 或 (B, C)
        返回: z (B, T, n_modes) 或 (B, n_modes)
        """
        # ΦᵀN
        PhiT_N = Phi.T @ N  # (C, C)
        
        if x.dim() == 3:
            # (B, T, C) @ (C, C)ᵀ → (B, T, C)
            z = torch.einsum('btc,mc->btm', x, PhiT_N.T)
        else:
            z = x @ PhiT_N.T
        
        # 可选：截断模态
        if self.n_modes is not None and self.n_modes < z.shape[-1]:
            z = z[..., :self.n_modes]
        
        return z
    
    def z_to_x(self, z: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        """
        广义坐标 → 观测空间
        
        x = Φz
        """
        if self.n_modes is not None:
            Phi = Phi[:, :self.n_modes]
        
        if z.dim() == 3:
            x = torch.einsum('btm,cm->btc', z, Phi.T)
        else:
            x = z @ Phi.T
        
        return x
    
    def forward(
        self, 
        x: torch.Tensor, 
        K: torch.Tensor, 
        N: torch.Tensor,
        return_modes: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        完整的模态变换
        
        x: (B, T, C) 原始数据
        K, N: 物理矩阵
        """
        # 计算模态
        Lambda, Phi = self.compute_modes(K, N)
        
        # 缓存
        self.Phi = Phi
        self.Lambda = Lambda
        self.N_cache = N
        
        # 变换到广义坐标
        z = self.x_to_z(x, Phi, N)
        
        result = {
            'z': z,
            'Lambda': Lambda,
            'Phi': Phi,
            'omega': torch.sqrt(Lambda) / (2 * np.pi),  # 本征频率
        }
        
        if return_modes:
            result['modes'] = Phi
        
        return result


# =============================================================================
# 第三层：能量几何（度规从能量涌现）
# =============================================================================

class EnergyGeometry(nn.Module):
    """
    从能量泛函构造度规
    
    物理原理：
        势能 V(z) = (1/2) zᵀΛz + α/4 |z|⁴
        度规 g(z) = Hessian(V) = Λ + α(|z|²I + 2z⊗z)
    
    可学习参数：
        α: 非线性势能强度（控制几何的状态依赖性）
    """
    
    def __init__(self, alpha_init: float = 0.1):
        super().__init__()
        # 可学习的非线性强度
        self.log_alpha = nn.Parameter(torch.tensor(np.log(alpha_init + 1e-6)))
    
    @property
    def alpha(self):
        return torch.exp(self.log_alpha)
    
    def compute_metric(
        self, 
        z: torch.Tensor, 
        Lambda: torch.Tensor,
        potential_type: str = "anharmonic"
    ) -> torch.Tensor:
        """
        计算度规 g(z) = Hessian(V)
        
        z: (B, T, D) 或 (B, D)
        Lambda: (D,) 特征值
        
        返回: g (B, T, D, D) 或 (B, D, D)
        """
        D = Lambda.shape[0]
        device = z.device
        dtype = z.dtype
        
        # 基础度规 = Λ（对角矩阵）
        Lambda_diag = torch.diag(Lambda)  # (D, D)
        
        if potential_type == "quadratic":
            # 线性系统：g = Λ（常数）
            if z.dim() == 3:
                B, T, _ = z.shape
                g = Lambda_diag.unsqueeze(0).unsqueeze(0).expand(B, T, D, D)
            else:
                B = z.shape[0]
                g = Lambda_diag.unsqueeze(0).expand(B, D, D)
        
        elif potential_type == "anharmonic":
            # 非线性系统：g = Λ + α(|z|²I + 2z⊗z)
            z_norm_sq = (z ** 2).sum(dim=-1, keepdim=True)  # (B, T, 1) 或 (B, 1)
            
            if z.dim() == 3:
                B, T, _ = z.shape
                # 扩展 Lambda
                g = Lambda_diag.unsqueeze(0).unsqueeze(0).expand(B, T, D, D).clone()
                
                # 非线性修正
                I = torch.eye(D, device=device, dtype=dtype)
                g = g + self.alpha * z_norm_sq.unsqueeze(-1) * I  # |z|²I
                g = g + 2 * self.alpha * torch.einsum('bti,btj->btij', z, z)  # 2z⊗z
            else:
                B = z.shape[0]
                g = Lambda_diag.unsqueeze(0).expand(B, D, D).clone()
                
                I = torch.eye(D, device=device, dtype=dtype)
                g = g + self.alpha * z_norm_sq.unsqueeze(-1) * I
                g = g + 2 * self.alpha * torch.einsum('bi,bj->bij', z, z)
        else:
            raise ValueError(f"Unknown potential_type: {potential_type}")
        
        # 确保正定（添加小的单位矩阵）
        I = torch.eye(D, device=device, dtype=dtype)
        if g.dim() == 4:
            g = g + 1e-6 * I.unsqueeze(0).unsqueeze(0)
        else:
            g = g + 1e-6 * I.unsqueeze(0)
        
        return g
    
    def compute_potential_energy(self, z: torch.Tensor, Lambda: torch.Tensor) -> torch.Tensor:
        """
        计算势能 V(z) = (1/2) zᵀΛz + α/4 |z|⁴
        """
        # 二次项
        V_quad = 0.5 * (z ** 2 * Lambda).sum(dim=-1)
        
        # 四次项
        z_norm_sq = (z ** 2).sum(dim=-1)
        V_quartic = 0.25 * self.alpha * z_norm_sq ** 2
        
        return V_quad + V_quartic
    
    def forward(
        self, 
        z: torch.Tensor, 
        Lambda: torch.Tensor,
        potential_type: str = "anharmonic"
    ) -> Dict[str, torch.Tensor]:
        """
        z: (B, T, D) 广义坐标
        Lambda: (D,) 特征值
        """
        g = self.compute_metric(z, Lambda, potential_type)
        V = self.compute_potential_energy(z, Lambda)
        
        return {
            'g': g,
            'V': V,
            'alpha': self.alpha,
        }


# =============================================================================
# 第四层：测地线演化（联络从度规涌现）
# =============================================================================

class GeodesicEvolution(nn.Module):
    """
    测地线演化
    
    物理原理：
        联络 Γᵏᵢⱼ = (1/2) gᵏˡ(∂ᵢgₗⱼ + ∂ⱼgₗᵢ - ∂ₗgᵢⱼ)
        测地线加速度 aᵏ = -Γᵏᵢⱼvⁱvʲ
    
    对于 anharmonic 势能，Γ 有解析形式！
    """
    
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
    
    def compute_christoffel_anharmonic(
        self, 
        z: torch.Tensor, 
        Lambda: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 anharmonic 势能的 Christoffel 符号（解析形式）
        
        对于 V = (1/2)zᵀΛz + α/4|z|⁴
        g = Λ + α(|z|²I + 2z⊗z)
        
        Γᵏᵢⱼ 有解析表达式（通过对 g 求导）
        """
        D = z.shape[-1]
        device = z.device
        dtype = z.dtype
        
        # 对于线性情况 (α≈0)，Γ = 0
        if alpha < 1e-8:
            if z.dim() == 3:
                B, T, _ = z.shape
                return torch.zeros(B, T, D, D, D, device=device, dtype=dtype)
            else:
                B = z.shape[0]
                return torch.zeros(B, D, D, D, device=device, dtype=dtype)
        
        # 计算度规及其逆
        z_norm_sq = (z ** 2).sum(dim=-1, keepdim=True)
        
        Lambda_diag = torch.diag(Lambda)
        I = torch.eye(D, device=device, dtype=dtype)
        
        if z.dim() == 3:
            B, T, _ = z.shape
            g = Lambda_diag.unsqueeze(0).unsqueeze(0).expand(B, T, D, D).clone()
            g = g + alpha * z_norm_sq.unsqueeze(-1) * I
            g = g + 2 * alpha * torch.einsum('bti,btj->btij', z, z)
            
            # g 的逆（数值计算）
            g_inv = torch.linalg.inv(g + 1e-6 * I)
            
            # ∂g/∂zᵏ 的解析形式
            # ∂g_ij/∂z_k = 2α z_k δ_ij + 2α (δ_ik z_j + z_i δ_jk)
            # 简化：使用数值差分
            Gamma = self._numerical_christoffel(z, Lambda, alpha)
        else:
            B = z.shape[0]
            g = Lambda_diag.unsqueeze(0).expand(B, D, D).clone()
            g = g + alpha * z_norm_sq.unsqueeze(-1) * I
            g = g + 2 * alpha * torch.einsum('bi,bj->bij', z, z)
            
            g_inv = torch.linalg.inv(g + 1e-6 * I)
            Gamma = self._numerical_christoffel(z, Lambda, alpha)
        
        return Gamma
    
    def _numerical_christoffel(
        self, 
        z: torch.Tensor, 
        Lambda: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        数值计算 Christoffel 符号
        """
        D = z.shape[-1]
        device = z.device
        dtype = z.dtype
        
        def get_metric(z_in):
            z_norm_sq = (z_in ** 2).sum(dim=-1, keepdim=True)
            Lambda_diag = torch.diag(Lambda)
            I = torch.eye(D, device=device, dtype=dtype)
            
            if z_in.dim() == 3:
                B, T, _ = z_in.shape
                g = Lambda_diag.unsqueeze(0).unsqueeze(0).expand(B, T, D, D).clone()
                g = g + alpha * z_norm_sq.unsqueeze(-1) * I
                g = g + 2 * alpha * torch.einsum('bti,btj->btij', z_in, z_in)
            else:
                B = z_in.shape[0]
                g = Lambda_diag.unsqueeze(0).expand(B, D, D).clone()
                g = g + alpha * z_norm_sq.unsqueeze(-1) * I
                g = g + 2 * alpha * torch.einsum('bi,bj->bij', z_in, z_in)
            return g
        
        # 当前度规及其逆
        g = get_metric(z)
        I = torch.eye(D, device=device, dtype=dtype)
        g_inv = torch.linalg.inv(g + 1e-6 * I)
        
        # 数值计算 ∂g/∂zᵏ
        if z.dim() == 3:
            B, T, _ = z.shape
            dg = torch.zeros(B, T, D, D, D, device=device, dtype=dtype)
            
            for k in range(D):
                z_p = z.clone()
                z_m = z.clone()
                z_p[..., k] += self.eps
                z_m[..., k] -= self.eps
                
                g_p = get_metric(z_p)
                g_m = get_metric(z_m)
                
                dg[..., k] = (g_p - g_m) / (2 * self.eps)
            
            # Γᵏᵢⱼ = (1/2) gᵏˡ(∂ᵢgₗⱼ + ∂ⱼgₗᵢ - ∂ₗgᵢⱼ)
            Gamma = 0.5 * torch.einsum(
                'btkl,btlji->btkij',
                g_inv,
                dg.permute(0, 1, 3, 4, 2) + dg.permute(0, 1, 4, 3, 2) - dg
            )
        else:
            B = z.shape[0]
            dg = torch.zeros(B, D, D, D, device=device, dtype=dtype)
            
            for k in range(D):
                z_p = z.clone()
                z_m = z.clone()
                z_p[..., k] += self.eps
                z_m[..., k] -= self.eps
                
                g_p = get_metric(z_p)
                g_m = get_metric(z_m)
                
                dg[..., k] = (g_p - g_m) / (2 * self.eps)
            
            Gamma = 0.5 * torch.einsum(
                'bkl,blji->bkij',
                g_inv,
                dg.permute(0, 2, 3, 1) + dg.permute(0, 3, 2, 1) - dg
            )
        
        return Gamma
    
    def geodesic_acceleration(
        self, 
        v: torch.Tensor, 
        Gamma: torch.Tensor
    ) -> torch.Tensor:
        """
        测地线加速度 aᵏ = -Γᵏᵢⱼvⁱvʲ
        
        v: (B, T, D) 或 (B, D) 速度
        Gamma: (B, T, D, D, D) 或 (B, D, D, D)
        """
        if v.dim() == 3:
            a = -torch.einsum('btkij,bti,btj->btk', Gamma, v, v)
        else:
            a = -torch.einsum('bkij,bi,bj->bk', Gamma, v, v)
        
        return a
    
    def forward(
        self, 
        z: torch.Tensor, 
        v: torch.Tensor,
        Lambda: torch.Tensor,
        alpha: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算测地线演化
        """
        Gamma = self.compute_christoffel_anharmonic(z, Lambda, alpha)
        a_geo = self.geodesic_acceleration(v, Gamma)
        
        return {
            'Gamma': Gamma,
            'a_geodesic': a_geo,
        }


# =============================================================================
# 第五层：外力与解码（可学习部分）
# =============================================================================

class ExternalForce(nn.Module):
    """
    外力网络（小容量，物理约束）
    
    F_ext = f(z, v)，但被限制为小扰动
    """
    
    def __init__(self, z_dim: int, max_scale: float = 0.1):
        super().__init__()
        self.z_dim = z_dim
        self.max_scale = max_scale
        
        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, z_dim),
            nn.Tanh(),
        )
        self.scale = nn.Parameter(torch.tensor(0.01))
    
    def forward(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        z, v: (B, D) 或 (B, T, D)
        """
        h = torch.cat([z, v], dim=-1)
        F = self.net(h)
        
        # 限制外力大小
        bounded_scale = torch.clamp(self.scale.abs(), max=self.max_scale)
        return F * bounded_scale


class Decoder(nn.Module):
    """
    解码器：从广义坐标预测输出
    """
    
    def __init__(self, z_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim * 2),
            nn.SiLU(),
            nn.Linear(z_dim * 2, output_dim),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# =============================================================================
# 统一世界模型
# =============================================================================

class UnifiedWorldModel(nn.Module):
    """
    统一世界模型
    
    完整流程：
        X → WorldDynamics → (N,D,K) 
          → ModalTransform → z = ΦᵀNx
          → EnergyGeometry → g(z), V(z)
          → GeodesicEvolution → Γ, a_geo
          → ExternalForce → F_ext
          → Decoder → 预测
    
    物理定义 vs 可学习：
        - z: 物理定义（模态变换）
        - g(z): 物理定义（能量 Hessian）
        - Γ: 物理定义（度规导数）
        - β, γ, α: 可学习（物理参数）
        - F_ext: 可学习（小扰动）
        - decoder: 可学习（输出映射）
    """
    
    def __init__(
        self,
        n_channels: int,
        n_modes: int = None,
        output_dim: int = None,
        beta_func_init: float = 1.0,
        gamma_init: float = 0.1,
        alpha_init: float = 0.1,
        potential_type: str = "anharmonic",
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_modes = n_modes if n_modes is not None else n_channels
        self.output_dim = output_dim if output_dim is not None else n_channels
        self.potential_type = potential_type
        
        # 五层架构
        self.world_dynamics = WorldDynamicsBuilder(
            n_channels, 
            beta_func_init=beta_func_init,
            gamma_init=gamma_init
        )
        self.modal_transform = ModalTransform(n_modes=self.n_modes)
        self.energy_geometry = EnergyGeometry(alpha_init=alpha_init)
        self.geodesic = GeodesicEvolution()
        self.external_force = ExternalForce(self.n_modes)
        self.decoder = Decoder(self.n_modes, self.output_dim)
    
    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        X: (B, T, C) 输入序列
        """
        B, T, C = X.shape
        
        # 1. 世界动力学：构建物理系统
        physics = self.world_dynamics(X)
        N, D, K = physics['N'], physics['D'], physics['K']
        
        # 2. 模态变换：x → z
        modal = self.modal_transform(X, K, N)
        z = modal['z']  # (B, T, n_modes)
        Lambda = modal['Lambda']
        
        # 3. 能量几何：z → g(z), V(z)
        geom = self.energy_geometry(z, Lambda[:self.n_modes], self.potential_type)
        g = geom['g']
        V = geom['V']
        
        # 4. 计算速度 v = dz/dt
        if T > 1:
            v = z[:, 1:] - z[:, :-1]  # (B, T-1, n_modes)
            z_for_geo = z[:, 1:]  # 对应的位置
        else:
            v = torch.zeros_like(z)
            z_for_geo = z
        
        # 5. 测地线演化：Γ → a_geo
        if T > 1:
            geo_result = self.geodesic(
                z_for_geo, v, 
                Lambda[:self.n_modes], 
                self.energy_geometry.alpha
            )
            Gamma = geo_result['Gamma']
            a_geo = geo_result['a_geodesic']
        else:
            Gamma = None
            a_geo = torch.zeros(B, 1, self.n_modes, device=X.device)
        
        # 6. 外力
        if T > 1:
            F_ext = self.external_force(z_for_geo, v)
        else:
            F_ext = torch.zeros(B, 1, self.n_modes, device=X.device)
        
        # 7. 总加速度
        a_total = a_geo + F_ext
        
        # 8. 解码（预测下一步）
        if T > 1:
            z_last = z[:, -1]
            v_last = v[:, -1]
            a_last = a_total[:, -1] if a_total.dim() == 3 else a_total
            
            # z_next = z + v + 0.5*a
            z_pred = z_last + v_last + 0.5 * a_last
        else:
            z_pred = z[:, 0]
        
        # 解码到输出空间
        output = self.decoder(z_pred)
        
        # 重建回观测空间
        x_pred = self.modal_transform.z_to_x(z_pred, modal['Phi'][:, :self.n_modes])
        
        return {
            # 输出
            'output': output,
            'x_pred': x_pred,
            'z_pred': z_pred,
            
            # 物理量
            'z': z,
            'v': v if T > 1 else None,
            'g': g,
            'V': V,
            'Gamma': Gamma,
            'a_geodesic': a_geo,
            'F_external': F_ext,
            'a_total': a_total,
            
            # 物理参数
            'N': N,
            'D': D,
            'K': K,
            'Lambda': Lambda,
            'Phi': modal['Phi'],
            'omega': modal['omega'],
            
            # 可学习参数
            'beta_func': physics['beta_func'],
            'gamma': physics['gamma'],
            'alpha': self.energy_geometry.alpha,
        }
    
    def compute_loss(
        self, 
        X: torch.Tensor, 
        target: torch.Tensor = None,
        force_weight: float = 0.5,
        energy_weight: float = 0.1,
        smoothness_weight: float = 0.1,
        stiffness_weight: float = 0.1,
        damping_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        物理一致性损失：
        1. 预测损失：预测值与目标的差
        2. 能量递减：V(t+1) < V(t)（因为有阻尼）
        3. 轨迹平滑：加速度不应该太大
        4. 外力惩罚：外力应该是小扰动
        5. 刚度矩阵正定性：确保 K 正定（让 beta 有梯度）
        6. 阻尼一致性：确保能量耗散符合物理（让 gamma 有梯度）
        """
        out = self.forward(X)
        
        # 目标
        if target is None:
            target = X[:, -1]  # 默认预测最后一步
        
        # 1. 预测损失
        pred_loss = F.mse_loss(out['x_pred'], target)
        
        # 2. 能量递减损失（物理一致性）
        V = out['V']  # (B, T)
        if V.dim() == 2 and V.shape[1] > 1:
            energy_diff = V[:, 1:] - V[:, :-1]  # 应该 < 0
            energy_loss = F.relu(energy_diff).mean()
        else:
            energy_loss = torch.tensor(0.0, device=X.device)
        
        # 3. 轨迹平滑损失
        if out['a_total'].dim() == 3:
            smoothness_loss = (out['a_total'] ** 2).mean()
        else:
            smoothness_loss = torch.tensor(0.0, device=X.device)
        
        # 4. 外力惩罚
        force_loss = (out['F_external'] ** 2).mean()
        
        # 5. 刚度矩阵正定性损失（直接使用 K，让 beta 有梯度）
        K = out['K']
        Lambda = out['Lambda']  # 特征值
        
        # 惩罚负特征值（非正定）
        stiffness_loss = F.relu(-Lambda + 1e-4).mean()
        
        # 或者直接通过 K 的迹来约束
        # stiffness_loss = F.relu(-torch.trace(K) / K.shape[0] + 0.1)  # K 的迹应该为正
        
        # 6. 阻尼一致性损失（直接使用 D，让 gamma 有梯度）
        # 能量耗散率应该与阻尼成正比：dE/dt ∝ -gamma * v²
        D = out['D']
        gamma = out['gamma']
        
        if out['v'] is not None and out['v'].dim() == 3:
            v = out['v']  # (B, T-1, D)
            # 实际能量耗散率
            if V.shape[1] > 1:
                actual_dE_dt = V[:, 1:] - V[:, :-1]  # (B, T-1)
                # 理论能量耗散率：-gamma * |v|²
                v_norm_sq = (v ** 2).sum(dim=-1)  # (B, T-1)
                theoretical_dE_dt = -gamma * v_norm_sq
                # 一致性损失
                damping_loss = F.mse_loss(actual_dE_dt, theoretical_dE_dt)
            else:
                damping_loss = torch.tensor(0.0, device=X.device)
        else:
            damping_loss = torch.tensor(0.0, device=X.device)
        
        # 总损失
        total_loss = (
            pred_loss + 
            energy_weight * energy_loss + 
            smoothness_weight * smoothness_loss + 
            force_weight * force_loss +
            stiffness_weight * stiffness_loss +
            damping_weight * damping_loss
        )
        
        return {
            'loss': total_loss,
            'pred_loss': pred_loss,
            'energy_loss': energy_loss,
            'smoothness_loss': smoothness_loss,
            'force_loss': force_loss,
            'stiffness_loss': stiffness_loss,
            'damping_loss': damping_loss,
            'V_mean': V.mean(),
            'a_geo_norm': out['a_geodesic'].norm() if out['a_geodesic'] is not None else 0,
            'F_ext_norm': out['F_external'].norm(),
        }


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("统一世界模型测试")
    print("=" * 60)
    
    # 创建模型
    model = UnifiedWorldModel(
        n_channels=8,
        n_modes=4,
        output_dim=8,
        potential_type="anharmonic"
    )
    
    print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试数据：正弦波混合
    B, T, C = 4, 32, 8
    t = torch.linspace(0, 4*np.pi, T).unsqueeze(0).unsqueeze(-1)
    freqs = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.8, 1.2]).unsqueeze(0).unsqueeze(0)
    X = torch.sin(t * freqs) + 0.1 * torch.randn(B, T, C)
    
    print(f"\n输入形状: {X.shape}")
    
    # 前向传播
    out = model(X)
    
    print(f"\n输出:")
    print(f"  z: {out['z'].shape}")
    print(f"  g: {out['g'].shape}")
    print(f"  V: {out['V'].shape}")
    print(f"  x_pred: {out['x_pred'].shape}")
    print(f"  Lambda: {out['Lambda'][:4]}")
    print(f"  omega (Hz): {out['omega'][:4]}")
    
    # 计算损失
    loss_dict = model.compute_loss(X)
    
    print(f"\n损失:")
    print(f"  total: {loss_dict['loss'].item():.6f}")
    print(f"  pred: {loss_dict['pred_loss'].item():.6f}")
    print(f"  energy: {loss_dict['energy_loss'].item():.6f}")
    print(f"  smoothness: {loss_dict['smoothness_loss'].item():.6f}")
    print(f"  force: {loss_dict['force_loss'].item():.6f}")
    
    # 物理参数
    print(f"\n物理参数:")
    print(f"  beta_func: {out['beta_func'].item():.4f}")
    print(f"  gamma: {out['gamma'].item():.4f}")
    print(f"  alpha: {out['alpha'].item():.4f}")
    
    # 训练测试
    print("\n" + "=" * 60)
    print("训练测试")
    print("=" * 60)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(50):
        loss_dict = model.compute_loss(X)
        
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss_dict['loss'].item():.4f}, "
                  f"pred={loss_dict['pred_loss'].item():.4f}, "
                  f"energy={loss_dict['energy_loss'].item():.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

