"""耦合振子实验 - 验证图真正生成频率"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)


def simulate_coupled_oscillators(T=10.0, dt=0.001, coupling=100.0):
    """
    模拟 4 个耦合的弹簧振子（环形）
    
    物理方程：m·ẍᵢ + k_local·xᵢ + k_coupling·Σ(xᵢ - xⱼ) = 0
    
    这就是 N·ẍ + K·x = 0 的具体实例！
    """
    t = np.arange(0, T, dt)
    n = len(t)
    
    x = np.zeros((n, 4))
    v = np.zeros((n, 4))
    x[0] = [1, 0, 0.5, -0.5]
    
    # 本地频率（不同！）
    omega_local = np.array([10, 12, 11, 9]) * 2 * np.pi
    k_local = omega_local**2
    
    # 环形图的拉普拉斯
    L_ring = np.array([
        [ 2, -1,  0, -1],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [-1,  0, -1,  2]
    ], dtype=float)
    
    # 理论 K 矩阵
    K_theory = np.diag(k_local) + coupling * L_ring
    
    # 理论本征频率
    mu_theory = eigh(K_theory, eigvals_only=True)
    freq_theory = np.sqrt(np.maximum(mu_theory, 0)) / (2 * np.pi)
    
    # 数值模拟
    for i in range(n-1):
        for j in range(4):
            F_local = -k_local[j] * x[i, j]
            neighbors = [(j-1)%4, (j+1)%4]
            F_coupling = coupling * sum(x[i, nb] - x[i, j] for nb in neighbors)
            a = F_local + F_coupling
            v[i+1, j] = v[i, j] + a * dt
            x[i+1, j] = x[i, j] + v[i+1, j] * dt
    
    return x, t, freq_theory, omega_local/(2*np.pi), K_theory, L_ring


if __name__ == "__main__":
    X, t, freq_theory, omega_local, K_theory, L_ring = simulate_coupled_oscillators(coupling=100)
    
    print("="*70)
    print("耦合振子系统 - 图真正生成频率的验证")
    print("="*70)
    print()
    print("物理模型: N·ẍ + K·x = 0")
    print("其中: K = diag(ω_local²) + β·L_ring")
    print()
    print(f"本地频率 ω_local: {omega_local} Hz")
    print()
    print("环形图拉普拉斯 L_ring:")
    print(L_ring)
    print()
    print("="*70)
    print("理论预测 vs 数值模拟 (FFT)")
    print("="*70)
    
    # 从模拟数据提取频率
    for i in range(4):
        fft = np.fft.rfft(X[:, i])
        power = np.abs(fft)**2
        freq_axis = np.fft.rfftfreq(len(X[:, i]), 0.001)
        peaks = np.argsort(power)[::-1][:3]
        print(f"通道 {i+1} 的主频: {freq_axis[peaks]} Hz")
    
    print()
    print(f"理论集体频率: {np.sort(freq_theory)} Hz")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 左上：时域
    ax = axes[0, 0]
    for i in range(4):
        ax.plot(t[:2000], X[:2000, i], alpha=0.8, label=f"Osc {i+1} (local={omega_local[i]:.1f}Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position")
    ax.set_title("Coupled Oscillators: Time Domain")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 右上：频谱
    ax = axes[0, 1]
    for i in range(4):
        fft = np.fft.rfft(X[:, i])
        freq_axis = np.fft.rfftfreq(len(X[:, i]), 0.001)
        ax.semilogy(freq_axis[:500], np.abs(fft[:500])**2, alpha=0.7, label=f"Osc {i+1}")
    for f in freq_theory:
        ax.axvline(f, color="r", ls="--", alpha=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(f"Power Spectrum (red = theory: {freq_theory.round(1)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 左下：不同耦合强度
    ax = axes[1, 0]
    betas = [0, 10, 100, 1000, 10000]
    for beta in betas:
        K = np.diag((omega_local*2*np.pi)**2) + beta * L_ring
        mu = np.maximum(eigh(K, eigvals_only=True), 0)
        freq = np.sqrt(mu) / (2*np.pi)
        ax.semilogx([max(beta, 0.1)]*4, freq, "o", markersize=8)
    ax.set_xlabel("Coupling strength")
    ax.set_ylabel("Collective frequency (Hz)")
    ax.set_title("Collective modes vs coupling")
    ax.grid(True, alpha=0.3)
    
    # 右下：总结
    ax = axes[1, 1]
    ax.text(0.1, 0.5, 
            "Key Verification:\n\n"
            "1. Local freq: [10, 12, 11, 9] Hz\n\n"
            "2. Theory: K = diag(w^2) + beta*L\n\n"
            "3. Simulation: solve x'' + Kx = 0\n\n"
            "4. FFT: extract freq from data\n\n"
            "If Theory ~ FFT, then:\n"
            "  'Graph truly generates frequency!'",
            ha="left", va="center", fontsize=11, 
            family="monospace", transform=ax.transAxes)
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("coupled_oscillators.png", dpi=150)
    print()
    print("图已保存到 coupled_oscillators.png")
    plt.show()
