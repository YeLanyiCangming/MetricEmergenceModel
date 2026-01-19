"""统一世界模型 - 精简版

第一性原理核心：
    M_unified = A_dmd @ exp(-α·L)
                ↑         ↑
            时间动力学   图上热核扩散
    
    - A_dmd: 从 Hankel DMD 提取的时间演化算子
    - exp(-αL): 图上的热核，描述信息在图上如何扩散
    - α: 可学习的扩散时间参数
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from dataclasses import dataclass, field

np.set_printoptions(precision=6, suppress=True)


# =============================================================================
# 基础工具
# =============================================================================

def hermitian_part(A: np.ndarray) -> np.ndarray:
    """提取 Hermitian 部分"""
    return (A + A.conj().T) / 2


def generate_signals(T: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """生成测试信号: 10Hz, 15Hz, 20Hz, 5-7Hz chirp"""
    np.random.seed(42)
    t = np.arange(0, T, dt)
    noise = 0.05
    
    X1 = 1.0 * np.sin(2 * np.pi * 10 * t) + noise * np.random.randn(len(t))
    X2 = 0.5 * np.sin(2 * np.pi * 20 * t + np.pi/4) + noise * np.random.randn(len(t))
    X3 = 0.8 * np.sin(2 * np.pi * 15 * t + np.pi/2) + noise * np.random.randn(len(t))
    
    f0, f1 = 5, 7
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * t**2)
    X4 = 0.3 * np.sin(phase) + noise * np.random.randn(len(t))
    
    return np.column_stack([X1, X2, X3, X4]), t


# =============================================================================
# 核心：Hankel DMD
# =============================================================================

def build_hankel(X: np.ndarray, delay: int) -> np.ndarray:
    """构建多变量 Hankel 矩阵 [delay*N, cols]"""
    W, N = X.shape
    cols = W - delay + 1
    H = np.zeros((delay * N, cols))
    for i in range(delay):
        H[i*N:(i+1)*N, :] = X[i:i+cols, :].T
    return H


def hankel_dmd(X: np.ndarray, k_modes: int, dt: float, delay: int = None):
    """
    Hankel DMD - 提取时间动力学
    
    返回:
        A_dmd: DMD 算子
        lambdas: 特征值
        Phi: 特征向量
        freqs: 频率 (Hz)
        growths: 增长率
    """
    W, N = X.shape
    if delay is None:
        delay = min(W // 3, 15)
        delay = max(delay, 5)
    
    H = build_hankel(X, delay)
    X_past, X_future = H[:, :-1], H[:, 1:]
    
    U, S, Vh = np.linalg.svd(X_past, full_matrices=False)
    
    # 自适应秩: 99% 能量
    energy = np.cumsum(S**2) / np.sum(S**2)
    r = np.searchsorted(energy, 0.99) + 1
    r = max(r, k_modes)
    r = min(r, len(S))
    
    U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
    
    # DMD 算子
    A_dmd = U_r.T @ X_future @ Vh_r.T @ np.diag(1.0 / S_r)
    
    # 特征分解
    lambdas, W_eig = np.linalg.eig(A_dmd)
    
    # 频率和增长率
    freqs = np.abs(np.angle(lambdas)) / (2 * np.pi * dt)
    growths = np.log(np.abs(lambdas) + 1e-10) / dt
    
    # 按频率排序
    idx = np.argsort(freqs)[::-1]
    lambdas = lambdas[idx][:k_modes]
    freqs = freqs[idx][:k_modes]
    growths = growths[idx][:k_modes]
    Phi = U_r @ W_eig[:, idx][:, :k_modes]
    
    return A_dmd, lambdas, Phi, freqs, growths


# =============================================================================
# 核心：图拉普拉斯
# =============================================================================

def build_graph_laplacian(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从数据构建归一化图拉普拉斯
    
    返回:
        L: 归一化拉普拉斯
        lambda_L: 图频率 (特征值)
        U_L: 图傅里叶基 (特征向量)
    """
    N = X.shape[1]
    C = np.cov(X.T)
    A = np.abs(C)
    np.fill_diagonal(A, 0)
    
    D = np.diag(A.sum(axis=1) + 1e-10)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    L = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # 特征分解
    lambda_L, U_L = np.linalg.eigh(L)
    
    return L, lambda_L, U_L


# =============================================================================
# 核心：统一世界模型 M = A_dmd @ exp(-αL)
# =============================================================================

def unified_world_model(
    X: np.ndarray,
    k_modes: int,
    dt: float,
    alpha: float = 0.0,
    delay: int = None
) -> Dict:
    """
    统一世界模型
    
    第一性原理：
        M = A_dmd @ exp(-α·L)
        
        - A_dmd: 时间动力学算子 (来自 Hankel DMD)
        - exp(-αL): 图上热核 (信息在图上如何扩散)
        - α = 0: 纯 DMD (无图影响)
        - α > 0: 图参与动力学
    
    返回包含所有结果的字典
    """
    W, N = X.shape
    if delay is None:
        delay = min(W // 3, 15)
        delay = max(delay, 5)
    
    # 1. 构建图拉普拉斯
    L, lambda_L, U_L = build_graph_laplacian(X)
    
    # 2. 在图傅里叶域做 Hankel DMD
    X_gft = X @ U_L  # 图傅里叶变换
    H = build_hankel(X_gft, delay)
    d = H.shape[0]
    
    X_past, X_future = H[:, :-1], H[:, 1:]
    U, S, Vh = np.linalg.svd(X_past, full_matrices=False)
    
    energy = np.cumsum(S**2) / np.sum(S**2)
    r = np.searchsorted(energy, 0.99) + 1
    r = max(r, k_modes)
    r = min(r, len(S))
    
    U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
    A_gft = U_r.T @ X_future @ Vh_r.T @ np.diag(1.0 / S_r)
    
    # 3. 图热核约束
    if alpha > 0:
        # 扩展图频率到 Hankel 空间
        lambda_extended = np.tile(lambda_L, delay)
        Lambda_hankel = np.diag(lambda_extended)
        Lambda_r = U_r.T @ Lambda_hankel @ U_r
        
        # 热核: exp(-α·Λ)
        heat_kernel = expm(-alpha * Lambda_r)
        A_unified = A_gft @ heat_kernel
    else:
        A_unified = A_gft
    
    # 4. 特征分解
    lambdas, W_eig = np.linalg.eig(A_unified)
    
    freqs = np.abs(np.angle(lambdas)) / (2 * np.pi * dt)
    growths = np.log(np.abs(lambdas) + 1e-10) / dt
    
    # 按频率排序
    idx = np.argsort(freqs)[::-1]
    lambdas = lambdas[idx][:k_modes]
    freqs = freqs[idx][:k_modes]
    growths = growths[idx][:k_modes]
    Phi = U_r @ W_eig[:, idx][:, :k_modes]
    
    # 5. 度规 g = Φ^H Φ
    g = Phi.conj().T @ Phi
    g = hermitian_part(g)
    eigvals_g = np.linalg.eigvalsh(g)
    if np.min(eigvals_g) < 1e-10:
        g = g + (1e-6 - np.min(eigvals_g)) * np.eye(k_modes)
    
    return {
        'A_unified': A_unified,
        'lambdas': lambdas,
        'Phi': Phi,
        'freqs': freqs,
        'growths': growths,
        'g': g,
        'L': L,
        'lambda_L': lambda_L,
        'U_L': U_L,
        'alpha': alpha
    }


# =============================================================================
# 结果收集
# =============================================================================

@dataclass
class Results:
    """结果收集器"""
    times: List[float] = field(default_factory=list)
    freqs: List[np.ndarray] = field(default_factory=list)
    growths: List[np.ndarray] = field(default_factory=list)
    g_cond: List[float] = field(default_factory=list)
    
    def add(self, t, freqs, growths, g):
        self.times.append(t)
        self.freqs.append(freqs)
        self.growths.append(growths)
        eigvals = np.linalg.eigvalsh(g)
        self.g_cond.append(np.max(eigvals) / (np.min(eigvals) + 1e-10))


def run_experiment(
    method: str,
    T: float = 10.0,
    dt: float = 0.01,
    window_s: float = 0.5,
    k_modes: int = 4,
    alpha: float = 0.0
) -> Results:
    """运行实验"""
    X, t = generate_signals(T, dt)
    window = int(window_s / dt)
    
    results = Results()
    
    for i in range(len(t) - window):
        X_win = X[i:i+window]
        t_mid = t[i + window // 2]
        
        if method == 'hankel_dmd':
            _, lambdas, Phi, freqs, growths = hankel_dmd(X_win, k_modes, dt)
            g = Phi.conj().T @ Phi
            g = hermitian_part(g)
        else:
            res = unified_world_model(X_win, k_modes, dt, alpha=alpha)
            freqs, growths, g = res['freqs'], res['growths'], res['g']
        
        results.add(t_mid, freqs, growths, g)
    
    return results


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("统一世界模型 - 第一性原理")
    print("")
    print("核心公式: M = A_dmd @ exp(-α·L)")
    print("  - A_dmd: 时间动力学 (Hankel DMD)")
    print("  - exp(-αL): 图上热核 (空间扩散)")
    print("  - α: 可学习的扩散时间")
    print("="*60)
    
    true_freqs = [10, 15, 20, 6]
    
    # 实验 1: 纯 Hankel DMD (α=0)
    print("\n[1] Hankel DMD (基线)...")
    res_dmd = run_experiment('hankel_dmd', k_modes=4)
    
    # 实验 2: 统一模型 α=0 (应该和DMD一样)
    print("[2] 统一模型 α=0 (应该≈DMD)...")
    res_u0 = run_experiment('unified', k_modes=4, alpha=0.0)
    
    # 实验 3: 统一模型 α=0.1
    print("[3] 统一模型 α=0.1 (图参与)...")
    res_u1 = run_experiment('unified', k_modes=4, alpha=0.1)
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Unified World Model: M = A_dmd @ exp(-αL)', fontsize=14)
    
    methods = [
        ('Hankel DMD (α=0)', res_dmd),
        ('Unified (α=0)', res_u0),
        ('Unified (α=0.1)', res_u1)
    ]
    
    for col, (name, res) in enumerate(methods):
        t = res.times
        
        # 频率
        ax = axes[0, col]
        freqs_arr = np.array(res.freqs)
        for k in range(min(4, freqs_arr.shape[1])):
            ax.plot(t, freqs_arr[:, k], label=f'Mode {k+1}', alpha=0.7)
        for f in true_freqs:
            ax.axhline(f, color='k', ls=':', lw=1, alpha=0.5)
        ax.set_ylabel('Freq (Hz)')
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.set_ylim([0, 25])
        
        # 增长率
        ax = axes[1, col]
        growths_arr = np.array(res.growths)
        for k in range(min(4, growths_arr.shape[1])):
            ax.plot(t, growths_arr[:, k], label=f'Mode {k+1}', alpha=0.7)
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_ylabel('γ (1/s)')
        ax.set_xlabel('Time (s)')
        ax.legend(fontsize=7)
    
    plt.tight_layout()
    
    # 统计
    print("\n" + "="*60)
    print("结果统计")
    print("="*60)
    
    for name, res in methods:
        freqs_arr = np.array(res.freqs)
        print(f"\n{name}:")
        for k in range(min(4, freqs_arr.shape[1])):
            print(f"  Mode {k+1}: {np.mean(freqs_arr[:, k]):.2f} ± {np.std(freqs_arr[:, k]):.2f} Hz")
    
    print(f"\n真实频率: {true_freqs} Hz")
    
    plt.show()
