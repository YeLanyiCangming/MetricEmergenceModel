"""
world_dynamics.py - 统一世界动力学模块

第一性原理：
    N·ẍ + D·ẋ + K·x = F(t)
    
    N = 质量矩阵（惯性）
    D = 阻尼矩阵（耗散）
    K = 刚度矩阵 = diag(ω²) + β·L（本地频率 + 图耦合）
    F(t) = 外部驱动

本体论：
    世界是一个图上的耗散动力系统
    - 节点 = 模态通道
    - 边 = 耦合强度（结构性 or 功能性）
    - 频率提取 = 求解本征振动
    - 多模态对齐 = 共享 (N, D, K) 动力学
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import eigh, inv
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class WorldDynamics:
    """世界动力学系统的完整描述"""
    N: np.ndarray  # 质量矩阵（惯性）
    D: np.ndarray  # 阻尼矩阵（耗散）
    K: np.ndarray  # 刚度矩阵（恢复）
    L: np.ndarray  # 图拉普拉斯（耦合）
    omega_local: np.ndarray  # 本地频率
    
    # 参数
    beta: float = 1.0   # 图耦合强度
    gamma: float = 0.1  # 阻尼系数


# =============================================================================
# 核心函数
# =============================================================================

def estimate_local_frequencies(X: np.ndarray, dt: float) -> np.ndarray:
    """
    估计每个通道的本地频率（使用 FFT）
    
    本体论：每个节点有自己的“本征振动频率”，即时间尺度
    """
    n_channels = X.shape[1]
    freqs = np.zeros(n_channels)
    
    for i in range(n_channels):
        fft = np.fft.rfft(X[:, i])
        power = np.abs(fft)**2
        freq_axis = np.fft.rfftfreq(len(X[:, i]), dt)
        power[0] = 0  # 排除直流
        freqs[i] = freq_axis[np.argmax(power)]
    
    return freqs


def build_graph_laplacian(
    X: np.ndarray = None,
    A: np.ndarray = None,
    graph_type: str = "functional",
    normalize: bool = True
) -> np.ndarray:
    """
    构建图拉普拉斯
    
    本体论区分：
    - structural: 先验的物理/语义连接（传入 A）
    - functional: 后验的数据相关性（从 X 计算）
    """
    if graph_type == "structural":
        assert A is not None, "结构性图需要提供邻接矩阵 A"
        n = A.shape[0]
    else:
        assert X is not None, "功能性图需要提供数据 X"
        n = X.shape[1]
        C = np.cov(X.T)
        A = np.abs(C)
        np.fill_diagonal(A, 0)
    
    D_vec = A.sum(axis=1) + 1e-10
    
    if normalize:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D_vec))
        L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = np.diag(D_vec) - A
    
    return L


def build_mass_matrix(X: np.ndarray, mass_type: str = "covariance") -> np.ndarray:
    """
    构建质量矩阵 N
    
    本体论：质量 = 惯性，高方差的节点“重”，响应慢
    
    - identity: N = I（各节点等质量）
    - covariance: N = cov(X)（方差大 → 质量大）
    - diagonal: N = diag(var(X))（只取对角线）
    """
    n = X.shape[1]
    
    if mass_type == "identity":
        return np.eye(n)
    elif mass_type == "covariance":
        return np.cov(X.T)
    elif mass_type == "diagonal":
        return np.diag(np.var(X, axis=0))
    else:
        return np.eye(n)


def build_damping_matrix(N: np.ndarray, gamma: float = 0.1) -> np.ndarray:
    """
    构建阻尼矩阵 D
    
    本体论：阻尼 = 能量耗散的速率
    
    使用瑞利阻尼 (Rayleigh damping): D = γ·N
    """
    return gamma * N


def build_stiffness_matrix(
    omega_local: np.ndarray,
    L_func: np.ndarray = None,
    L_struct: np.ndarray = None,
    beta_func: float = 1.0,
    beta_struct: float = 0.0
) -> np.ndarray:
    """
    构建刚度矩阵 K
    
    K = diag(ω²) + β_func·L_func + β_struct·L_struct
    
    本体论：
    - diag(ω²): 本地恢复力（每个节点的本征频率）
    - L_func: 功能性图耦合（后验，从数据）
    - L_struct: 结构性图耦合（先验，从知识）
    
    当无先验知识时，beta_struct=0，回退为纯数据驱动
    当有先验知识时，可学习 beta_* 的权重
    """
    n = len(omega_local)
    omega_squared = (2 * np.pi * omega_local) ** 2
    K = np.diag(omega_squared)
    
    if L_func is not None:
        K = K + beta_func * L_func
    
    if L_struct is not None:
        K = K + beta_struct * L_struct
    
    return K


def build_world_dynamics(
    X: np.ndarray,
    dt: float,
    L_struct: np.ndarray = None,
    beta_func: float = 1.0,
    beta_struct: float = 0.0,
    gamma: float = 0.1,
    mass_type: str = "identity"
) -> WorldDynamics:
    """
    构建完整的世界动力学系统
    
    N·ẍ + D·ẋ + K·x = F(t)
    
    本体论：
    - L_func: 功能性图（后验，从数据协方差）
    - L_struct: 结构性图（先验，从知识输入）
    - K = diag(ω²) + β_func·L_func + β_struct·L_struct
    """
    # 1. 本地频率
    omega_local = estimate_local_frequencies(X, dt)
    
    # 2. 功能性图（始终从数据构建）
    L_func = build_graph_laplacian(X=X, graph_type="functional")
    
    # 3. 质量矩阵
    N_mat = build_mass_matrix(X, mass_type)
    
    # 4. 阻尼矩阵
    D_mat = build_damping_matrix(N_mat, gamma)
    
    # 5. 刚度矩阵（双图结构）
    K_mat = build_stiffness_matrix(omega_local, L_func, L_struct, beta_func, beta_struct)
    
    return WorldDynamics(
        N=N_mat, D=D_mat, K=K_mat, L=L_func,
        omega_local=omega_local, beta=beta_func, gamma=gamma
    )


# =============================================================================
# 本征模态求解
# =============================================================================

def solve_conservative_modes(K: np.ndarray, N: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解保守系统的本征模态
    
    无阻尼时：N·ẍ + K·x = 0
    设 x = u·exp(iωt)，得 K·u = ω²·N·u
    
    返回：集体频率 (Hz), 模态向量
    """
    if N is None:
        N = np.eye(K.shape[0])
    
    # 广义特征值问题: K·u = λ·N·u
    eigenvalues, eigenvectors = eigh(K, N)
    
    # ω = √λ
    eigenvalues = np.maximum(eigenvalues, 0)
    omega_collective = np.sqrt(eigenvalues)
    freq_collective = omega_collective / (2 * np.pi)
    
    return freq_collective, eigenvectors


def solve_damped_modes(
    K: np.ndarray,
    D: np.ndarray,
    N: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    求解有阻尼系统的本征模态
    
    N·ẍ + D·ẋ + K·x = 0
    
    转换为一阶系统: [x', v']^T = A·[x, v]^T
    A = [[0, I], [-N⁻¹K, -N⁻¹D]]
    
    返回：频率 (Hz), 衰减率 (γ), 模态
    """
    n = K.shape[0]
    if N is None:
        N = np.eye(n)
    
    # 构建状态空间矩阵
    N_inv = inv(N)
    A = np.zeros((2*n, 2*n))
    A[:n, n:] = np.eye(n)
    A[n:, :n] = -N_inv @ K
    A[n:, n:] = -N_inv @ D
    
    # 特征值分解
    eigenvalues = np.linalg.eigvals(A)
    
    # 解析：λ = -γ ± iω
    freq = np.abs(eigenvalues.imag) / (2 * np.pi)
    damping = -eigenvalues.real
    
    # 排序，取正频率部分
    idx = np.argsort(freq)[::-1][:n]
    
    return freq[idx], damping[idx], eigenvalues[idx]


def compute_frequency_response(
    K: np.ndarray,
    D: np.ndarray,
    N: np.ndarray,
    omega_range: np.ndarray
) -> np.ndarray:
    """
    计算频率响应函数
    
    H(ω) = (-ω²N + iωD + K)⁻¹
    
    本体论：描述系统对不同频率输入的响应
    """
    n = K.shape[0]
    H = np.zeros((len(omega_range), n, n), dtype=complex)
    
    for i, w in enumerate(omega_range):
        Z = -w**2 * N + 1j * w * D + K
        H[i] = inv(Z)
    
    return H


# =============================================================================
# PyTorch 版本的世界动力学层
# =============================================================================

class WorldDynamicsLayer(nn.Module):
    """
    世界动力学层 - 可学习的 (N, D, K) 系统
    
    核心方程: N·ẍ + D·ẋ + K·x = F(t)
    
    本体论：
    - K = diag(ω²) + β_func·L_func + β_struct·L_struct
    - L_func: 功能性图（后验，从数据）
    - L_struct: 结构性图（先验，可选）
    
    可学习参数:
    - beta_func, beta_struct: 图耦合强度
    - gamma: 阻尼系数
    - omega_local: 本地频率 (可选)
    """
    
    def __init__(
        self,
        n_channels: int,
        beta_func_init: float = 1.0,
        beta_struct_init: float = 0.0,
        gamma_init: float = 0.1,
        learnable_omega: bool = False,
        L_struct: torch.Tensor = None
    ):
        super().__init__()
        self.n_channels = n_channels
        
        # 可学习参数
        self.log_beta_func = nn.Parameter(torch.tensor(np.log(beta_func_init + 1e-6)))
        self.log_beta_struct = nn.Parameter(torch.tensor(np.log(beta_struct_init + 1e-6)))
        self.log_gamma = nn.Parameter(torch.tensor(np.log(gamma_init + 1e-6)))
        
        # 本地频率 (可选可学习)
        self.learnable_omega = learnable_omega
        if learnable_omega:
            self.log_omega = nn.Parameter(torch.zeros(n_channels))
        
        # 功能性图 (会在 forward 中更新)
        self.register_buffer('L_func', torch.eye(n_channels))
        
        # 结构性图 (先验输入，固定)
        if L_struct is not None:
            self.register_buffer('L_struct', L_struct)
        else:
            self.register_buffer('L_struct', torch.zeros(n_channels, n_channels))
    
    @property
    def beta_func(self) -> torch.Tensor:
        return torch.exp(self.log_beta_func)
    
    @property
    def beta_struct(self) -> torch.Tensor:
        return torch.exp(self.log_beta_struct)
    
    @property
    def gamma(self) -> torch.Tensor:
        return torch.exp(self.log_gamma)
    
    def update_graph(self, X: torch.Tensor):
        """
        从数据更新功能性图拉普拉斯 L_func
        
        X: (batch, time, channels)
        """
        # 计算协方差
        X_centered = X - X.mean(dim=1, keepdim=True)
        C = torch.bmm(X_centered.transpose(1, 2), X_centered) / X.shape[1]
        C = C.mean(dim=0)  # 平均跨 batch
        
        # 邻接矩阵
        A = torch.abs(C)
        A = A - torch.diag(torch.diag(A))  # 移除对角线
        
        # 归一化拉普拉斯
        D_vec = A.sum(dim=1) + 1e-10
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D_vec))
        L = torch.eye(self.n_channels, device=X.device) - D_inv_sqrt @ A @ D_inv_sqrt
        
        self.L_func = L
    
    def build_K(self, omega_local: torch.Tensor) -> torch.Tensor:
        """
        构建刚度矩阵 K = diag(ω²) + β_func·L_func + β_struct·L_struct
        """
        omega_sq = (2 * np.pi * omega_local) ** 2
        K = torch.diag(omega_sq) + self.beta_func * self.L_func + self.beta_struct * self.L_struct
        return K
    
    def build_N(self, X: torch.Tensor, mass_type: str = "identity") -> torch.Tensor:
        """构建质量矩阵"""
        if mass_type == "identity":
            return torch.eye(self.n_channels, device=X.device)
        elif mass_type == "covariance":
            X_centered = X - X.mean(dim=1, keepdim=True)
            C = torch.bmm(X_centered.transpose(1, 2), X_centered) / X.shape[1]
            return C.mean(dim=0)
        else:
            return torch.eye(self.n_channels, device=X.device)
    
    def build_D(self, N: torch.Tensor) -> torch.Tensor:
        """构建阻尼矩阵 D = γ·N"""
        return self.gamma * N
    
    def solve_modes(self, K: torch.Tensor, N: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        求解本征模态 (保守系统)
        
        K·u = λ·N·u
        返回: 集体频率, 模态向量
        """
        # 转换为 numpy 进行特征分解
        K_np = K.detach().cpu().numpy()
        N_np = N.detach().cpu().numpy()
        
        eigenvalues, eigenvectors = eigh(K_np, N_np)
        eigenvalues = np.maximum(eigenvalues, 0)
        
        freq = np.sqrt(eigenvalues) / (2 * np.pi)
        
        return (
            torch.tensor(freq, device=K.device, dtype=K.dtype),
            torch.tensor(eigenvectors, device=K.device, dtype=K.dtype)
        )
    
    def forward(
        self,
        X: torch.Tensor,
        omega_local: torch.Tensor = None,
        return_modes: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        X: (batch, time, channels)
        omega_local: 本地频率 (Hz)
        
        返回:
        - K: 刚度矩阵
        - N: 质量矩阵
        - D: 阻尼矩阵
        - freq_collective: 集体频率
        - modes: 模态向量 (如果 return_modes=True)
        """
        # 更新图
        self.update_graph(X)
        
        # 本地频率
        if omega_local is None:
            if self.learnable_omega:
                omega_local = torch.exp(self.log_omega)
            else:
                # 从数据估计
                omega_local = self._estimate_local_freq(X)
        
        # 构建矩阵
        N = self.build_N(X)
        D = self.build_D(N)
        K = self.build_K(omega_local)
        
        # 求解模态
        freq_collective, modes = self.solve_modes(K, N)
        
        result = {
            'K': K,
            'N': N,
            'D': D,
            'L_func': self.L_func,
            'L_struct': self.L_struct,
            'omega_local': omega_local,
            'freq_collective': freq_collective,
            'beta_func': self.beta_func,
            'beta_struct': self.beta_struct,
            'gamma': self.gamma
        }
        
        if return_modes:
            result['modes'] = modes
        
        return result
    
    def _estimate_local_freq(self, X: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """估计本地频率 (使用 FFT)"""
        # X: (batch, time, channels)
        X_np = X.detach().cpu().numpy()
        
        freqs = []
        for b in range(X_np.shape[0]):
            batch_freqs = []
            for c in range(X_np.shape[2]):
                fft = np.fft.rfft(X_np[b, :, c])
                power = np.abs(fft)**2
                freq_axis = np.fft.rfftfreq(X_np.shape[1], dt)
                power[0] = 0
                batch_freqs.append(freq_axis[np.argmax(power)])
            freqs.append(batch_freqs)
        
        freqs = np.array(freqs).mean(axis=0)  # 平均跨 batch
        return torch.tensor(freqs, device=X.device, dtype=X.dtype)


# =============================================================================
# 世界动力学 → 黎曼几何 桥接
# =============================================================================

def extract_modal_coordinates(
    x: np.ndarray,
    K: np.ndarray,
    N: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从世界动力学提取广义坐标 z
    
    物理意义：
    - K·Φ = N·Φ·Λ （广义特征值问题）
    - z = Φᵀ·N·x （模态坐标变换）
    
    返回：
    - z: 广义坐标 (模态坐标)
    - Phi: 模态矩阵 (特征向量)
    - omega_sq: 特征值 (频率²)
    """
    n = K.shape[0]
    if N is None:
        N = np.eye(n)
    
    # 广义特征分解: K·Φ = N·Φ·Λ
    omega_sq, Phi = eigh(K, N)
    omega_sq = np.maximum(omega_sq, 0)
    
    # 广义坐标变换
    if x.ndim == 1:
        z = Phi.T @ N @ x
    else:
        z = np.einsum('ij,jk,...k->...i', Phi.T, N, x)
    
    return z, Phi, omega_sq


def construct_metric_from_energy(
    z: np.ndarray,
    omega_sq: np.ndarray,
    potential_type: str = "quadratic"
) -> np.ndarray:
    """
    从能量泛函构造度规 g
    
    物理原理：
    - 势能: V(z) = (1/2) zᵀ Λ z + V_nonlinear(z)
    - 度规 = 势能的 Hessian: g_ij = ∂²V/∂z_i∂z_j
    
    参数：
    - potential_type:
      - "quadratic": V = (1/2) zᵀ Λ z (线性)
      - "anharmonic": V = (1/2) zᵀ Λ z + α|z|^4 (非线性)
    """
    dim = len(omega_sq)
    Lambda = np.diag(omega_sq)
    
    if potential_type == "quadratic":
        # Hessian = Λ (常数)
        if z.ndim == 1:
            g = Lambda.copy()
        else:
            g = np.broadcast_to(Lambda, z.shape[:-1] + (dim, dim)).copy()
    
    elif potential_type == "anharmonic":
        # V = (1/2) zᵀ Λ z + α/4 |z|^4
        # Hessian = Λ + α (|z|² I + 2 z⊗z)
        alpha = 0.1
        z_norm_sq = np.sum(z**2, axis=-1, keepdims=True)
        
        if z.ndim == 1:
            g = Lambda + alpha * (z_norm_sq * np.eye(dim) + 2 * np.outer(z, z))
        else:
            g = np.zeros(z.shape[:-1] + (dim, dim))
            g[..., :, :] = Lambda
            g += alpha * z_norm_sq[..., np.newaxis] * np.eye(dim)
            g += 2 * alpha * np.einsum('...i,...j->...ij', z, z)
    else:
        g = Lambda.copy() if z.ndim == 1 else np.broadcast_to(Lambda, z.shape[:-1] + (dim, dim)).copy()
    
    # 确保正定
    g = g + 1e-6 * np.eye(dim)
    return g


class WorldToGeometry:
    """
    世界动力学 → 黎曼几何 桥接器
    
    流程：
    1. 从 (N, K) 提取模态结构: K·Φ = N·Φ·Λ
    2. x → z = ΦᵀNx (广义坐标)
    3. g(z) = Hessian(V) (度规从能量涌现)
    4. Γ(z) = f(∂g) (联络从度规涌现)
    """
    
    def __init__(self, K: np.ndarray, N: np.ndarray = None, potential_type: str = "quadratic"):
        self.K = K
        self.N = N if N is not None else np.eye(K.shape[0])
        self.potential_type = potential_type
        self.dim = K.shape[0]
        
        # 模态分解
        self.omega_sq, self.Phi = eigh(K, self.N)
        self.omega_sq = np.maximum(self.omega_sq, 1e-6)
        self.omega = np.sqrt(self.omega_sq)
    
    def x_to_z(self, x: np.ndarray) -> np.ndarray:
        """观测空间 → 广义坐标"""
        return np.einsum('ij,jk,...k->...i', self.Phi.T, self.N, x)
    
    def z_to_x(self, z: np.ndarray) -> np.ndarray:
        """广义坐标 → 观测空间"""
        return np.einsum('ij,...j->...i', self.Phi, z)
    
    def get_metric(self, z: np.ndarray) -> np.ndarray:
        """获取度规 g(z)"""
        return construct_metric_from_energy(z, self.omega_sq, self.potential_type)
    
    def get_christoffel(self, z: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """获取 Christoffel 符号 Γ(z)"""
        if self.potential_type == "quadratic":
            # 线性系统: Γ = 0
            return np.zeros((self.dim, self.dim, self.dim))
        
        # 数值计算 dg
        g = self.get_metric(z)
        g_inv = np.linalg.inv(g)
        
        dg = np.zeros((self.dim, self.dim, self.dim))
        for k in range(self.dim):
            z_p, z_m = z.copy(), z.copy()
            z_p[k] += eps
            z_m[k] -= eps
            dg[:, :, k] = (self.get_metric(z_p) - self.get_metric(z_m)) / (2 * eps)
        
        # Γ^k_ij = (1/2) g^{kl} (∂_i g_{lj} + ∂_j g_{li} - ∂_l g_{ij})
        Gamma = np.zeros((self.dim, self.dim, self.dim))
        for k in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    for l in range(self.dim):
                        Gamma[k, i, j] += 0.5 * g_inv[k, l] * (
                            dg[l, j, i] + dg[l, i, j] - dg[i, j, l]
                        )
        return Gamma
    
    def geodesic_acceleration(self, z: np.ndarray, v: np.ndarray) -> np.ndarray:
        """a^k = -Γ^k_ij v^i v^j"""
        Gamma = self.get_christoffel(z)
        return -np.einsum('kij,i,j->k', Gamma, v, v)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 数据结构
    'WorldDynamics',
    
    # PyTorch 层
    'WorldDynamicsLayer',
    
    # 世界动力学 → 黎曼几何 桥接
    'WorldToGeometry',
    'extract_modal_coordinates',
    'construct_metric_from_energy',
    
    # NumPy 工具函数
    'estimate_local_frequencies',
    'build_graph_laplacian',
    'build_mass_matrix',
    'build_damping_matrix',
    'build_stiffness_matrix',
    'build_world_dynamics',
    'solve_conservative_modes',
    'solve_damped_modes',
    'compute_frequency_response',
]
