"""统一世界模型 - 完整动力学版本

第一性原理（完整形式）：
    N·ẍ + D·ẋ + K·x = F(t)
    
    N = 质量矩阵（惯性）: cov(X) 或 I
    D = 阻尼矩阵（耗散）: γ·N（瑞利阻尼）
    K = 刚度矩阵（恢复）: diag(ω²) + β·L
    F(t) = 外部驱动（多模态输入）

本体论：
    - 世界是一个图上的耗散动力系统
    - 节点 = 模态通道（视觉、音频、IMU...）
    - 边 = 耦合强度（结构性 or 功能性）
    - 频率提取 = 求解本征振动
    - 多模态对齐 = 共享 (N, D, K) 动力学

关键区分：
    - 结构性图（structural）: 先验的物理/语义连接
    - 功能性图（functional）: 后验的数据相关性
"""

import numpy as np
from scipy.linalg import eigh, inv
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

np.set_printoptions(precision=4, suppress=True)


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
    L: np.ndarray,
    beta: float = 1.0
) -> np.ndarray:
    """
    构建刚度矩阵 K
    
    K = diag(ω²) + β·L
    
    本体论：
    - diag(ω²): 本地恢复力（每个节点的本征频率）
    - β·L: 图耦合力（节点间的弹簧连接）
    """
    omega_squared = (2 * np.pi * omega_local) ** 2
    return np.diag(omega_squared) + beta * L


def build_world_dynamics(
    X: np.ndarray,
    dt: float,
    A_structural: np.ndarray = None,
    beta: float = 1.0,
    gamma: float = 0.1,
    mass_type: str = "identity"
) -> WorldDynamics:
    """
    构建完整的世界动力学系统
    
    N·ẍ + D·ẋ + K·x = F(t)
    """
    # 1. 本地频率
    omega_local = estimate_local_frequencies(X, dt)
    
    # 2. 图拉普拉斯
    if A_structural is not None:
        L = build_graph_laplacian(A=A_structural, graph_type="structural")
    else:
        L = build_graph_laplacian(X=X, graph_type="functional")
    
    # 3. 质量矩阵
    N_mat = build_mass_matrix(X, mass_type)
    
    # 4. 阻尼矩阵
    D_mat = build_damping_matrix(N_mat, gamma)
    
    # 5. 刚度矩阵
    K_mat = build_stiffness_matrix(omega_local, L, beta)
    
    return WorldDynamics(
        N=N_mat, D=D_mat, K=K_mat, L=L,
        omega_local=omega_local, beta=beta, gamma=gamma
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
# 实验
# =============================================================================

def generate_test_signal(T: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """生成测试信号"""
    np.random.seed(42)
    t = np.arange(0, T, dt)
    noise = 0.05
    
    # 4 个不同频率的振子
    X1 = np.sin(2*np.pi*10*t) + noise*np.random.randn(len(t))  # 10 Hz
    X2 = np.sin(2*np.pi*20*t) + noise*np.random.randn(len(t))  # 20 Hz
    X3 = np.sin(2*np.pi*15*t) + noise*np.random.randn(len(t))  # 15 Hz
    X4 = np.sin(2*np.pi*6*t) + noise*np.random.randn(len(t))   # 6 Hz
    
    return np.column_stack([X1, X2, X3, X4]), t


def run_complete_dynamics_experiment():
    """完整动力学实验"""
    print("=" * 70)
    print("统一世界模型 - 完整动力学版本")
    print("")
    print("核心方程: N·ẍ + D·ẋ + K·x = F(t)")
    print("")
    print("  N = 质量矩阵（惯性）")
    print("  D = 阻尼矩阵（耗散）")
    print("  K = 刚度矩阵 = diag(ω²) + β·L（本地频率 + 图耦合）")
    print("  F(t) = 外部驱动（多模态输入）")
    print("=" * 70)
    
    # 1. 生成数据
    T, dt = 10.0, 0.01
    X, t = generate_test_signal(T, dt)
    true_freqs = np.array([10, 20, 15, 6])
    
    print(f"\n【1】本地频率（数据注入）: {true_freqs} Hz")
    
    # 2. 构建完整动力学系统
    world = build_world_dynamics(X, dt, beta=1.0, gamma=0.1, mass_type="identity")
    
    print(f"【2】估计的本地频率: {world.omega_local} Hz")
    
    # 3. 图拉普拉斯分析
    mu_L, _ = eigh(world.L)
    print(f"\n【3】图拉普拉斯特征值: {mu_L}")
    
    # 4. 保守系统 vs 阻尼系统
    print("\n【4】模态分析:")
    print("-" * 60)
    
    # 保守系统
    freq_conservative, _ = solve_conservative_modes(world.K, world.N)
    print(f"  保守系统 (γ=0):   {np.sort(freq_conservative)} Hz")
    
    # 阻尼系统
    freq_damped, damping, _ = solve_damped_modes(world.K, world.D, world.N)
    print(f"  阻尼系统 (γ={world.gamma}): {np.sort(freq_damped)[::-1][:4]} Hz")
    print(f"  衰减率:           {np.sort(damping)[::-1][:4]}")
    
    print("-" * 60)
    
    # 5. 不同耦合强度的比较
    print("\n【5】不同耦合强度 β 的影响:")
    print("-" * 60)
    
    betas = [0, 1, 10, 100, 1000]
    results_conservative = []
    results_damped = []
    
    for beta in betas:
        # 重新构建 K
        K = build_stiffness_matrix(world.omega_local, world.L, beta)
        
        freq_c, _ = solve_conservative_modes(K, world.N)
        freq_d, damp, _ = solve_damped_modes(K, world.D, world.N)
        
        results_conservative.append(np.sort(freq_c))
        results_damped.append(np.sort(freq_d)[::-1][:4])
        
        print(f"  β={beta:4d}: 保守={np.sort(freq_c)} Hz")
    
    print("-" * 60)
    
    # 6. 不同阻尼系数的影响
    print("\n【6】不同阻尼系数 γ 的影响 (β=1):")
    print("-" * 60)
    
    K_fixed = build_stiffness_matrix(world.omega_local, world.L, beta=1.0)
    gammas = [0, 0.01, 0.1, 1.0, 10.0]
    
    for gamma in gammas:
        D_test = build_damping_matrix(world.N, gamma)
        freq_d, damp, _ = solve_damped_modes(K_fixed, D_test, world.N)
        
        # 取频率 > 0 的部分
        valid = freq_d > 0.1
        if np.any(valid):
            print(f"  γ={gamma:5.2f}: 频率={freq_d[valid][:4]} Hz, 衰减={damp[valid][:4]}")
        else:
            print(f"  γ={gamma:5.2f}: 过阻尼（无振荡）")
    
    print("-" * 60)
    
    # 7. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上: 耦合强度 vs 频率
    ax = axes[0, 0]
    results_arr = np.array(results_conservative)
    for k in range(4):
        ax.semilogx([max(b, 0.1) for b in betas], results_arr[:, k], 
                    'o-', label=f'Mode {k+1}', markersize=8)
    for f in true_freqs:
        ax.axhline(f, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Coupling strength β')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('K = diag(ω²) + β·L\nConservative system')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 右上: 频率响应函数
    ax = axes[0, 1]
    omega_range = np.linspace(0.1, 150, 500) * 2 * np.pi
    H = compute_frequency_response(K_fixed, world.D, world.N, omega_range)
    H_norm = np.linalg.norm(H, axis=(1, 2))
    ax.semilogy(omega_range / (2*np.pi), H_norm)
    for f in true_freqs:
        ax.axvline(f, color='r', ls='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|H(ω)|')
    ax.set_title('Frequency Response Function\nH(ω) = (-ω²N + iωD + K)⁻¹')
    ax.grid(True, alpha=0.3)
    
    # 左下: 本体论示意图
    ax = axes[1, 0]
    text = """
    完整世界动力学:
    
    N·ẍ + D·ẋ + K·x = F(t)
    
    ──────────────────────────────
    矩阵      意义          来源
    ──────────────────────────────
    N        质量/惯性      cov(X) or I
    D        阻尼/耗散      γ·N
    K        刚度/恢复      diag(ω²)+βL
    F(t)     外部驱动      多模态输入
    ──────────────────────────────
    
    能量: E = (1/2)ẋᵀNẋ + (1/2)xᵀKx
    """
    ax.text(0.1, 0.5, text, ha='left', va='center',
            fontsize=10, family='monospace', transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('Ontology of World Dynamics')
    
    # 右下: 多模态块结构示意
    ax = axes[1, 1]
    text_multi = """
    多模态扩展:
    
    x(t) = [x_vision, x_audio, x_imu]ᵀ
    
    M 具有块结构:
    
    M = ┌ M_v   C_va  C_vi ┐
        │ C_av  M_a   C_ai │
        └ C_iv  C_ia  M_i  ┘
    
    M_*: 模态内动力学
    C_*: 跨模态耦合 (由跨模态图决定)
    
    多模态对齐 = 共享 (N, D, K) 动力学
    """
    ax.text(0.1, 0.5, text_multi, ha='left', va='center',
            fontsize=10, family='monospace', transform=ax.transAxes)
    ax.axis('off')
    ax.set_title('Multimodal Extension')
    
    plt.tight_layout()
    
    # 8. 本体论总结
    print("\n【7】本体论总结:")
    print("=" * 70)
    print("  世界是一个图上的耗散动力系统")
    print("")
    print("  结构性图 vs 功能性图:")
    print("    - 结构性图: 先验的物理/语义连接（如脑区、分子键）")
    print("    - 功能性图: 后验的数据相关性（如协方差）")
    print("")
    print("  频率提取 = 求解本征振动:")
    print(f"    - 本地频率: {world.omega_local} Hz")
    print(f"    - 集体频率: {np.sort(freq_conservative)} Hz")
    print("")
    print("  下一步: 将 β, γ 作为可学习参数，集成到 model.py")
    print("=" * 70)
    
    plt.show()
    
    return world


if __name__ == "__main__":
    world = run_complete_dynamics_experiment()
