"""统一世界模型 - 本体论版本

第一性原理：
    M = diag(ω_local²) + L_graph
    
    - diag(ω²): 每个节点的本征频率（来自数据）
    - L_graph: 节点间的耦合强度（来自图结构）
    - M 的特征值: 集体模态的频率²（涌现）

物理图像：
    N 个振子通过图连接
    ẍ + M·x = 0
    解的频率由 M 的谱决定
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from typing import Tuple

np.set_printoptions(precision=4, suppress=True)


# =============================================================================
# 核心函数
# =============================================================================

def estimate_local_frequencies(X: np.ndarray, dt: float) -> np.ndarray:
    """
    估计每个通道的本地频率（使用 FFT）
    
    本体论：每个节点有自己的"本征振动频率"
    """
    N = X.shape[1]
    freqs = np.zeros(N)
    
    for i in range(N):
        fft = np.fft.rfft(X[:, i])
        power = np.abs(fft)**2
        freq_axis = np.fft.rfftfreq(len(X[:, i]), dt)
        
        # 找主频（排除直流）
        power[0] = 0
        peak_idx = np.argmax(power)
        freqs[i] = freq_axis[peak_idx]
    
    return freqs


def build_graph_laplacian(X: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    从数据构建图拉普拉斯
    
    本体论：图定义了节点间的"耦合强度"
    """
    N = X.shape[1]
    C = np.cov(X.T)
    A = np.abs(C)
    np.fill_diagonal(A, 0)
    
    D = np.diag(A.sum(axis=1) + 1e-10)
    
    if normalize:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        L = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = D - A
    
    return L


def build_unified_M(
    omega_local: np.ndarray,
    L_graph: np.ndarray,
    coupling_strength: float = 1.0
) -> np.ndarray:
    """
    构建统一世界张量
    
    M = diag(ω²) + β·L
    
    - diag(ω²): 本地动力学
    - β·L: 图耦合
    - β: 耦合强度参数
    
    本体论：M 是"本地振动 + 图耦合"的统一
    """
    omega_squared = (2 * np.pi * omega_local) ** 2
    M = np.diag(omega_squared) + coupling_strength * L_graph
    return M


def extract_collective_modes(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取集体模态
    
    ẍ + M·x = 0 的解：
    x(t) = Σ cₖ uₖ exp(i·ωₖ·t)
    
    其中 ωₖ = √μₖ（μₖ 是 M 的特征值）
    
    本体论：集体频率是本地频率与图耦合的涌现
    """
    mu, U = eigh(M)
    
    # 确保非负（数值稳定）
    mu = np.maximum(mu, 0)
    
    omega_collective = np.sqrt(mu)
    freq_collective = omega_collective / (2 * np.pi)
    
    return freq_collective, U


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


def run_ontology_experiment():
    """本体论实验"""
    print("=" * 60)
    print("统一世界模型 - 本体论版本")
    print("")
    print("核心公式: M = diag(ω_local²) + β·L_graph")
    print("")
    print("物理图像: N 个振子通过图连接")
    print("  ẍ + M·x = 0")
    print("  集体频率 = √(M的特征值)")
    print("=" * 60)
    
    # 1. 生成数据
    T, dt = 10.0, 0.01
    X, t = generate_test_signal(T, dt)
    true_freqs = np.array([10, 20, 15, 6])
    
    print(f"\n【1】本地频率（数据注入）: {true_freqs} Hz")
    
    # 2. 估计本地频率
    omega_local = estimate_local_frequencies(X, dt)
    print(f"【2】估计的本地频率: {omega_local} Hz")
    
    # 3. 构建图拉普拉斯
    L = build_graph_laplacian(X)
    mu_L, _ = eigh(L)
    print(f"\n【3】图拉普拉斯特征值: {mu_L}")
    print(f"    图的本征频率 (√μ/2π): {np.sqrt(np.maximum(mu_L, 0))/(2*np.pi)} Hz")
    
    # 4. 不同耦合强度下的集体模态
    print("\n【4】不同耦合强度下的集体频率:")
    print("-" * 50)
    
    betas = [0, 1, 10, 100, 1000]
    results = []
    
    for beta in betas:
        M = build_unified_M(omega_local, L, coupling_strength=beta)
        freq_collective, U = extract_collective_modes(M)
        results.append(freq_collective)
        print(f"  β={beta:4d}: {freq_collective} Hz")
    
    print("-" * 50)
    
    # 5. 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：集体频率 vs 耦合强度
    ax = axes[0]
    results_arr = np.array(results)
    for k in range(4):
        ax.semilogx([max(b, 0.1) for b in betas], results_arr[:, k], 
                    'o-', label=f'Mode {k+1}', markersize=8)
    
    # 添加参考线
    for f in true_freqs:
        ax.axhline(f, color='gray', ls=':', alpha=0.5)
    
    ax.set_xlabel('Coupling strength β')
    ax.set_ylabel('Collective frequency (Hz)')
    ax.set_title('M = diag(ω²) + β·L\nCollective modes vs coupling strength')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 右图：本体论示意
    ax = axes[1]
    ax.text(0.5, 0.9, 'Ontology of Unified World Model', 
            ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    text = """
    M = diag(ω_local²) + β·L_graph
    
    ω_local: Local frequencies (from data)
             Each node has its own vibration
    
    L_graph: Graph Laplacian (coupling)
             Nodes connected by "springs"
    
    β = 0:   Independent oscillators
             Collective freq = Local freq
    
    β → ∞:   Strongly coupled system
             Collective freq = Graph freq
    
    β ∈ (0,∞): Emergent collective modes
               Neither local nor graph alone!
    """
    ax.text(0.1, 0.5, text, ha='left', va='center', 
            fontsize=11, family='monospace', transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    
    # 6. 分析
    print("\n【5】本体论分析:")
    print("=" * 50)
    print(f"  β=0:    集体频率 ≈ 本地频率 {results[0]}")
    print(f"  β=1000: 集体频率 → 图主导")
    print("")
    print("  关键洞见:")
    print("  - 当 β=0 时，M = diag(ω²)，集体频率 = 本地频率")
    print("  - 当 β→∞ 时，M ≈ β·L，集体频率 ≈ √(β·μ_L)")
    print("  - 中间情况：集体模态是涌现的，不同于单独的本地或图频率")
    print("=" * 50)
    
    plt.show()
    
    return results, omega_local, L


if __name__ == "__main__":
    results, omega_local, L = run_ontology_experiment()
