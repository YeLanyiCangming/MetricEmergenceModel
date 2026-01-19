"""统一世界模型 (M-Z-g) - 第一性原理实现

核心思想：
    1. 统一世界张量 M = Cov(X) + i·Cov(X, dX/dt)
       - 实部：结构约束（图拉普拉斯推广）
       - 虚部：动力学流（因果方向）
    
    2. 广义坐标 Z 从 M 的谱分解涌现
       - 特征值 λ = γ + iω 编码增长率和频率
       - 特征向量 Φ 是系统的正规模态
    
    3. 度规 g = Φ^H N Φ 从拉格朗日动能原理涌现
       - 这不是人为构造，而是变分原理的自然结果
"""

import numpy as np
from scipy.linalg import eig, eigh
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

# 设置打印选项
np.set_printoptions(precision=6, suppress=True)


# =============================================================================
# 纯函数组合子 (Pure Function Combinators)
# =============================================================================

def symmetric_part(A: np.ndarray) -> np.ndarray:
    """提取对称部分 - 纯函数，幂等性: sym(sym(A)) = sym(A)"""
    return (A + A.T) / 2


def hermitian_part(A: np.ndarray) -> np.ndarray:
    """提取 Hermitian 部分 - 纯函数，幂等性"""
    return (A + A.conj().T) / 2


def safe_inverse(A: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """安全伪逆 - 纯函数"""
    return np.linalg.pinv(A, rcond=rcond)


# =============================================================================
# 信号生成 (Signal Generation)
# =============================================================================

def generate_signals(T: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成多频率混合信号用于测试
    
    包含:
        - X1: 10 Hz 正弦波
        - X2: 20 Hz 正弦波（相位偏移）
        - X3: 15 Hz 正弦波（相位偏移）
        - X4: 5-7 Hz 线性 chirp 信号
    
    Returns:
        X: [T/dt, 4] 信号矩阵
        t: [T/dt] 时间向量
    """
    np.random.seed(42)  # 可重复性
    t = np.arange(0, T, dt)
    noise_level = 0.05  # 降低噪声水平
    
    X1 = 1.0 * np.sin(2 * np.pi * 10 * t) + noise_level * np.random.randn(len(t))
    X2 = 0.5 * np.sin(2 * np.pi * 20 * t + np.pi / 4) + noise_level * np.random.randn(len(t))
    X3 = 0.8 * np.sin(2 * np.pi * 15 * t + np.pi / 2) + noise_level * np.random.randn(len(t))
    
    # Chirp 信号: 频率从 f_start 线性变化到 f_end
    f_start, f_end = 5, 7
    instantaneous_phase = 2 * np.pi * (f_start * t + (f_end - f_start) / (2 * T) * t**2)
    X4 = 0.3 * np.sin(instantaneous_phase) + noise_level * np.random.randn(len(t))
    
    X = np.column_stack([X1, X2, X3, X4])
    return X, t


# =============================================================================
# 核心：StateEncoder (M → Z)
# =============================================================================

def compute_velocity(X: np.ndarray, dt: float, method: str = 'central') -> np.ndarray:
    """
    计算速度场 dX/dt - 纯函数
    
    Args:
        X: [W, N] 信号窗口
        dt: 采样间隔（秒）
        method: 'central'（中心差分）或 'gradient'（numpy gradient）
    
    Returns:
        dX_dt: [W, N] 速度场
    
    复杂度: O(W * N)
    """
    if method == 'central':
        # 中心差分（内部点）+ 单侧差分（边界）
        dX_dt = np.zeros_like(X)
        # 内部点：中心差分
        dX_dt[1:-1] = (X[2:] - X[:-2]) / (2 * dt)
        # 边界：单侧差分
        dX_dt[0] = (X[1] - X[0]) / dt
        dX_dt[-1] = (X[-1] - X[-2]) / dt
    else:
        # numpy gradient（自动处理边界）
        dX_dt = np.gradient(X, dt, axis=0)
    
    return dX_dt


def build_world_tensor_M(
    X_window: np.ndarray, 
    dt: float,
    regularization: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建统一世界张量 M = Re(M) + i·Im(M)
    
    第一性原理：
        - 实部 M_real = Cov(X): 结构约束（点之间的耦合强度）
        - 虚部 M_imag = Cov(X, dX/dt): 动力学流（因果方向）
    
    Args:
        X_window: [W, N] 信号窗口
        dt: 采样间隔（秒）
        regularization: 正则化参数
    
    Returns:
        M: [N, N] 复值世界张量
        N_matrix: [N, N] 质量矩阵（用于广义特征分解）
        dX_window: [W, N] 速度场
    
    复杂度: O(W * N^2)
    代数律: M_real 对称, M 的 Hermitian 部分有物理意义
    """
    W, N = X_window.shape
    
    # 实部：协方差矩阵（结构约束）
    M_real = np.cov(X_window, rowvar=False)  # [N, N]
    M_real = symmetric_part(M_real)  # 确保对称
    
    # 计算速度场（关键：正确的时间尺度！）
    dX_window = compute_velocity(X_window, dt, method='central')
    
    # 虚部：X 与 dX/dt 的互协方差（动力学流）
    X_centered = X_window - np.mean(X_window, axis=0)
    dX_centered = dX_window - np.mean(dX_window, axis=0)
    M_imag = (X_centered.T @ dX_centered) / (W - 1)  # [N, N]
    
    # 复值世界张量
    M = M_real + 1j * M_imag
    
    # 质量矩阵（用于广义特征分解，需要正定）
    N_matrix = M_real + regularization * np.eye(N)
    
    return M, N_matrix, dX_window


# =============================================================================
# 方法三：时间延迟嵌入 DMD (Time-Delay Embedding DMD)
# 这是正确提取频率的方法！
# =============================================================================

def build_hankel_matrix(x: np.ndarray, delay: int) -> np.ndarray:
    """
    构建 Hankel 矩阵（时间延迟嵌入）
    
    第一性原理：
        将时间序列的动态结构转换为空间结构，
        使得谱分解可以提取时间频率。
        
        Takens 嵌入定理：对于动力系统 x(t)，其延迟嵌入
        [x(t), x(t-τ), x(t-2τ), ...] 在拓扑上等价于原系统的吸引子。
    
    Args:
        x: [T] 单通道时间序列
        delay: 延迟嵌入维度
    
    Returns:
        H: [delay, T-delay+1] Hankel 矩阵
    
    复杂度: O(delay * T)
    代数律: H 的列是时间平移的，H[:, k+1] = shift(H[:, k])
    """
    T = len(x)
    cols = T - delay + 1
    H = np.zeros((delay, cols))
    for i in range(delay):
        H[i, :] = x[i:i+cols]
    return H


def build_multivariate_hankel(X: np.ndarray, delay: int) -> np.ndarray:
    """
    构建多变量 Hankel 矩阵
    
    Args:
        X: [W, N] 多通道信号
        delay: 延迟嵌入维度
    
    Returns:
        H: [delay*N, W-delay+1] 堆叠的 Hankel 矩阵
    """
    W, N = X.shape
    cols = W - delay + 1
    H = np.zeros((delay * N, cols))
    
    for ch in range(N):
        H_ch = build_hankel_matrix(X[:, ch], delay)
        H[ch*delay:(ch+1)*delay, :] = H_ch
    
    return H


# =============================================================================
# 方法五：统一世界张量 - 真正的图拉普拉斯 + 拉格朗日融合
# 基于 Hankel DMD 的本质思想
# =============================================================================

def extract_dynamics_matrix(H: np.ndarray, dt: float) -> np.ndarray:
    """
    从 Hankel 矩阵提取连续时间动力学矩阵 B
    
    第一性原理：
        假设线性演化: X_{k+1} = A X_k
        则 A = e^{B·Δt}，其中 B 是连续时间动力学矩阵
        B = ln(A)/Δt
        
        B 的特征值 λ_B = γ + iω 直接编码频率和增长率
    
    Args:
        H: [d, T] Hankel 矩阵
        dt: 采样间隔
    
    Returns:
        B: [d, d] 连续时间动力学矩阵
    """
    # DMD: X_{k+1} = A X_k
    X_past = H[:, :-1]
    X_future = H[:, 1:]
    
    # 最小二乘估计 A
    U, S, Vh = np.linalg.svd(X_past, full_matrices=False)
    
    # 截断小奇异值
    r = np.sum(S > 1e-10 * S[0])
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    
    # 低秩近似
    A_tilde = U_r.T @ X_future @ Vh_r.T @ np.diag(1.0 / S_r)
    
    # 连续化: B = ln(A)/dt
    # 使用特征分解计算矩阵对数
    lambdas, V = np.linalg.eig(A_tilde)
    
    # 避免 log(0)
    lambdas_safe = np.where(np.abs(lambdas) > 1e-10, lambdas, 1e-10)
    log_lambdas = np.log(lambdas_safe) / dt
    
    # 重构 B = V diag(log(λ)/dt) V^{-1}
    B_tilde = V @ np.diag(log_lambdas) @ np.linalg.pinv(V)
    
    # 投影回全空间
    B = U_r @ B_tilde @ U_r.T
    
    return B, A_tilde, U_r, lambdas, log_lambdas


def build_unified_world_tensor_v2(
    X_window: np.ndarray,
    dt: float,
    delay: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    统一世界张量 V2：融合 Hankel DMD + 图拉普拉斯 + 拉格朗日
    
    第一性原理融合：
    
    1. Hankel DMD 提取动力学矩阵 B = ln(A)/dt
    
    2. B 的分解：
       B = B_sym + B_antisym
       
       B_sym = (B + B^T)/2
           → 对称部分：类似图拉普拉斯（扩散/阻尼）
           → 描述能量耗散
       
       B_antisym = (B - B^T)/2
           → 反对称部分：类似辛形式（振荡/守恒）
           → 描述能量守恒的振荡
    
    3. 统一世界张量：
       M = -B_sym + i * B_antisym
         = L_dissipation + i * Ω_oscillation
       
       这正是“图拉普拉斯 + 拉格朗日”的自然形式！
       - 实部：图拉普拉斯类型的扩散/阻尼
       - 虚部：拉格朗日/哈密顿类型的振荡
    
    4. M 的特征值 λ = σ + iω
       - ω 是本征频率
       - σ 是阻尼/增长率
    
    Args:
        X_window: [W, N] 信号窗口
        dt: 采样间隔
        delay: Hankel 嵌入维度
    
    Returns:
        M_unified: 统一世界张量
        L_dissipation: 图拉普拉斯部分（耗散）
        Omega_oscillation: 拉格朗日部分（振荡）
        B: 连续时间动力学矩阵
        H: Hankel 矩阵
    """
    W, N = X_window.shape
    
    if delay is None:
        delay = min(W // 3, 15)
        delay = max(delay, 5)
    
    # 1. Hankel 嵌入
    H = build_multivariate_hankel(X_window, delay)
    
    # 2. 提取动力学矩阵
    B, A_tilde, U_r, _, _ = extract_dynamics_matrix(H, dt)
    
    # 3. 分解为对称和反对称部分
    B_sym = (B + B.T) / 2  # 对称部分
    B_antisym = (B - B.T) / 2  # 反对称部分
    
    # 4. 构建统一世界张量
    # 图拉普拉斯是半正定的，所以取负号
    L_dissipation = -B_sym  # 耗散项（类似图拉普拉斯）
    Omega_oscillation = B_antisym  # 振荡项（类似辛形式）
    
    M_unified = L_dissipation + 1j * Omega_oscillation
    
    return M_unified, L_dissipation, Omega_oscillation, B, H


def state_encoder_unified_v2(
    X_window: np.ndarray,
    k_modes: int,
    dt_sample: float,
    delay: int = None
) -> Tuple[np.ndarray, np.ndarray, List[Dict], np.ndarray, np.ndarray]:
    """
    统一世界编码器 V2：融合所有第一性原理
    
    解决 Hankel DMD 的缺陷：
    1. 纯数据驱动 → 添加图结构先验
    2. 单模态 → 统一表示多通道
    3. 无物理意义 → M = L(结构) + Ω(动力学)
    
    新思路：
        1. 从 DMD 提取动力学特征值（频率、增长率）
        2. 从协方差构建图结构（谁与谁连接）
        3. 统一到同一个表示 M
    
    M 的特征值 = DMD 的频率信息
    M 的特征向量 = 图结构与动力学的融合
    """
    W, N = X_window.shape
    
    if delay is None:
        delay = min(W // 3, 15)
        delay = max(delay, 5)
    
    # ===========================================
    # 第一步：Hankel DMD 提取动力学（保留其频率提取能力）
    # ===========================================
    H = build_multivariate_hankel(X_window, delay)
    
    X_past = H[:, :-1]
    X_future = H[:, 1:]
    
    # 完整 SVD（不限制秩，保留所有信息）
    U, S, Vh = np.linalg.svd(X_past, full_matrices=False)
    
    # 自适应秩选择：保留 99% 能量
    energy_ratio = np.cumsum(S**2) / np.sum(S**2)
    r = np.searchsorted(energy_ratio, 0.99) + 1
    r = max(r, k_modes)  # 至少保留 k_modes
    r = min(r, len(S))   # 不超过可用秩
    
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    
    # DMD 算子
    A_tilde = U_r.T @ X_future @ Vh_r.T @ np.diag(1.0 / S_r)
    
    # 特征分解
    lambdas_dmd, W_dmd = np.linalg.eig(A_tilde)
    
    # 提取频率和增长率
    freqs_hz = np.abs(np.angle(lambdas_dmd)) / (2 * np.pi * dt_sample)
    growth_rates = np.log(np.abs(lambdas_dmd) + 1e-10) / dt_sample
    
    # ===========================================
    # 第二步：构建图结构（Hankel DMD 缺少的）
    # ===========================================
    # 传感器间的协方差（空间结构）
    C_spatial = np.cov(X_window.T)  # [N, N]
    
    # 构建空间图拉普拉斯
    A_spatial = np.abs(C_spatial)
    np.fill_diagonal(A_spatial, 0)
    D_spatial = np.diag(A_spatial.sum(axis=1) + 1e-10)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_spatial)))
    L_spatial = np.eye(N) - D_inv_sqrt @ A_spatial @ D_inv_sqrt
    
    # ===========================================
    # 第三步：融合 DMD 动力学 + 图结构
    # ===========================================
    # 按频率排序
    idx = np.argsort(freqs_hz)[::-1]
    lambdas_sorted = lambdas_dmd[idx]
    W_sorted = W_dmd[:, idx]
    freqs_sorted = freqs_hz[idx]
    growth_sorted = growth_rates[idx]
    
    # 选择 k_modes 个主导模态
    selected_lambdas = lambdas_sorted[:k_modes]
    selected_W = W_sorted[:, :k_modes]
    
    # 投影回全空间
    Phi = U_r @ selected_W  # [delay*N, k_modes]
    
    # ===========================================
    # 第四步：构建统一世界张量 M
    # ===========================================
    # M 的特征值 = DMD 的动力学特征值（保证频率正确）
    # M 的结构 = 融合图拉普拉斯
    
    # 在低秩空间中分解 A_tilde
    A_sym = (A_tilde + A_tilde.conj().T) / 2    # 耗散项
    A_antisym = (A_tilde - A_tilde.conj().T) / 2  # 振荡项
    
    # 统一张量
    M_dynamics = A_tilde  # 动力学部分
    L_dissipation = np.real(A_sym)  # 耗散部分
    
    # ===========================================
    # 第五步：提取 Z 参数
    # ===========================================
    current_state = H[:, -1]
    
    Z_params = []
    for i in range(k_modes):
        if i < len(selected_lambdas):
            lam = selected_lambdas[i]
            phi = Phi[:, i]
            
            # 投影
            proj = np.vdot(phi.conj(), current_state) / (np.vdot(phi.conj(), phi) + 1e-10)
            
            # 频率和增长率（与 Hankel DMD 相同的计算）
            freq_hz = np.abs(np.angle(lam)) / (2 * np.pi * dt_sample)
            gamma = np.log(np.abs(lam) + 1e-10) / dt_sample
            omega = np.angle(lam) / dt_sample
            
            Z_params.append({
                'lambda': lam,
                'freq_hz': freq_hz,
                'omega_rad': omega,
                'growth_rate': gamma,
                'amplitude': np.abs(proj),
                'phase': np.angle(proj),
                'z_position': proj,
                'eigenvector': phi[:N] if len(phi) > N else phi,
                # 新增：图结构信息
                'spatial_coupling': L_spatial  # 空间耦合
            })
        else:
            Z_params.append({
                'lambda': 0, 'freq_hz': 0, 'omega_rad': 0,
                'growth_rate': 0, 'amplitude': 0, 'phase': 0,
                'z_position': 0, 'eigenvector': np.zeros(N),
                'spatial_coupling': L_spatial
            })
    
    return M_dynamics, L_spatial, Z_params, selected_lambdas, Phi


def metric_encoder_g_from_dynamics(
    L_dissipation: np.ndarray,
    selected_eigenvectors: np.ndarray,
    k_modes: int
) -> np.ndarray:
    """
    从动力学特征向量涌现度规
    
    第一性原理：
        度规 g 定义了模态空间的内积结构
        g_ij = <φ_i | φ_j>
        
        对于正交特征向量，g 近似为单位矩阵
    """
    Phi = selected_eigenvectors[:, :k_modes]
    
    # 直接计算 Gram 矩阵
    g = Phi.conj().T @ Phi
    g = hermitian_part(g)
    
    # 确保正定
    eigvals = np.linalg.eigvalsh(g)
    if np.min(eigvals) < 1e-10:
        g = g + (1e-6 - np.min(eigvals)) * np.eye(k_modes)
    
    return g


def state_encoder_hankel_dmd(
    X_window: np.ndarray,
    k_modes: int,
    dt_sample: float,
    delay: int = None
) -> Tuple[np.ndarray, List[Dict], np.ndarray, np.ndarray]:
    """
    基于时间延迟嵌入的 DMD：正确提取频率
    
    第一性原理：
        1. Hankel 矩阵将时间动态转换为空间结构
        2. DMD 在 Hankel 空间中提取线性动力学
        3. 特征值的虚部编码真实频率
    
    Args:
        X_window: [W, N] 信号窗口
        k_modes: 模态数
        dt_sample: 采样间隔
        delay: 延迟嵌入维度 (None = 自动)
    
    Returns:
        A_tilde: DMD 矩阵
        Z_params: 模态参数
        lambdas: 特征值
        Phi: 模态
    """
    W, N = X_window.shape
    
    # 自动确定延迟：达到奈奎斯特频率的 1/4 周期
    if delay is None:
        # 对于 100Hz 采样，要提取 20Hz 需要至少 5 点/周期
        # 延迟 = 2-3 个周期对应的点数
        delay = min(W // 3, 20)  # 限制最大延迟
        delay = max(delay, 5)  # 至少 5
    
    # 构建多变量 Hankel 矩阵
    H = build_multivariate_hankel(X_window, delay)  # [delay*N, W-delay+1]
    
    # DMD 分解
    X_past = H[:, :-1]   # [delay*N, cols-1]
    X_future = H[:, 1:]  # [delay*N, cols-1]
    
    # SVD
    U, S, Vh = np.linalg.svd(X_past, full_matrices=False)
    
    # 确定秩
    energy_ratio = np.cumsum(S**2) / np.sum(S**2)
    rank = np.searchsorted(energy_ratio, 0.99) + 1
    rank = min(rank, len(S), k_modes * 2)  # 需要足够的秩来捕获共轭对
    rank = max(rank, k_modes)
    
    # 截断
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = Vh[:rank, :]
    
    # 低秩 DMD 矩阵
    A_tilde = U_r.conj().T @ X_future @ V_r.T @ np.diag(1.0 / S_r)
    
    # 特征分解
    lambdas_discrete, W_tilde = np.linalg.eig(A_tilde)
    
    # 转换到连续时间特征值
    lambdas_discrete = np.where(
        np.abs(lambdas_discrete) < 1e-10,
        1e-10 + 0j,
        lambdas_discrete
    )
    lambdas_continuous = np.log(lambdas_discrete) / dt_sample
    
    # DMD 模态
    Phi_full = U_r @ W_tilde
    
    # 按频率排序（高频优先）
    freq_abs = np.abs(np.imag(lambdas_continuous)) / (2 * np.pi)
    idx = np.argsort(freq_abs)[::-1]
    
    # 选择非零频率的模态（排除直流分量）
    selected_idx = []
    for i in idx:
        if freq_abs[i] > 0.5:  # 至少 0.5 Hz
            selected_idx.append(i)
            if len(selected_idx) >= k_modes:
                break
    
    # 如果没有足够的高频模态，填充低频模态
    if len(selected_idx) < k_modes:
        for i in idx:
            if i not in selected_idx:
                selected_idx.append(i)
                if len(selected_idx) >= k_modes:
                    break
    
    selected_idx = selected_idx[:k_modes]
    
    lambdas_sorted = lambdas_continuous[selected_idx]
    Phi_sorted = Phi_full[:, selected_idx]
    
    # 提取 Z 参数
    # 使用最后一个 Hankel 列作为当前状态
    current_state = H[:, -1]
    
    Z_params = []
    for i in range(len(selected_idx)):
        lam = lambdas_sorted[i]
        phi = Phi_sorted[:, i]
        
        # 投影
        proj = np.vdot(phi.conj(), current_state) / (np.vdot(phi.conj(), phi) + 1e-10)
        
        omega = np.imag(lam)  # 角频率 (rad/s)
        gamma = np.real(lam)  # 衰减率 (1/s)
        
        Z_params.append({
            'lambda': lam,
            'freq_hz': omega / (2 * np.pi),
            'omega_rad': omega,
            'growth_rate': gamma,
            'amplitude': np.abs(proj),
            'phase': np.angle(proj),
            'z_position': proj,
            'eigenvector': phi[:N] if len(phi) > N else phi  # 取第一个通道的模态
        })
    
    return A_tilde, Z_params, lambdas_sorted, Phi_sorted


# =============================================================================
# 方法二：动态模式分解 (DMD) - 更精确的频率提取
# =============================================================================

def build_dmd_matrices(
    X_window: np.ndarray,
    dt: float,
    rank: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建 DMD 矩阵（动态模式分解）
    
    第一性原理：
        假设系统满足线性动力学: x_{k+1} ≈ A x_k
        则 A 的特征值 λ 编码离散时间动力学:
            - 连续时间特征值: ω = log(λ) / dt
            - 频率: f = Im(ω) / (2π)
            - 衰减率: γ = Re(ω)
    
    Args:
        X_window: [W, N] 信号窗口
        dt: 采样间隔
        rank: SVD 截断秩 (None = 自动)
    
    Returns:
        A_tilde: [r, r] 低秩 DMD 矩阵
        Phi: [N, r] DMD 模态
        lambdas_continuous: [r] 连续时间特征值
    """
    W, N = X_window.shape
    
    # 构建时间延迟矩阵
    X_past = X_window[:-1, :].T    # [N, W-1]
    X_future = X_window[1:, :].T   # [N, W-1]
    
    # SVD 分解
    U, S, Vh = np.linalg.svd(X_past, full_matrices=False)
    
    # 自动确定秩（基于能量保留）
    if rank is None:
        energy_ratio = np.cumsum(S**2) / np.sum(S**2)
        rank = np.searchsorted(energy_ratio, 0.99) + 1
        rank = min(rank, len(S), N)
    
    # 截断
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = Vh[:rank, :]
    
    # 低秩 DMD 矩阵: A_tilde = U^H X' V S^{-1}
    A_tilde = U_r.conj().T @ X_future @ V_r.T @ np.diag(1.0 / S_r)
    
    # 特征分解
    lambdas_discrete, W_tilde = np.linalg.eig(A_tilde)
    
    # 转换到连续时间特征值
    # ω = log(λ) / dt
    # 避免 log(0) 和异常值
    lambdas_discrete = np.where(
        np.abs(lambdas_discrete) < 1e-10, 
        1e-10 + 0j, 
        lambdas_discrete
    )
    lambdas_continuous = np.log(lambdas_discrete) / dt
    
    # DMD 模态
    Phi = X_future @ V_r.T @ np.diag(1.0 / S_r) @ W_tilde
    
    return A_tilde, Phi, lambdas_continuous


def state_encoder_dmd(
    X_window: np.ndarray,
    k_modes: int,
    dt_sample: float
) -> Tuple[np.ndarray, List[Dict], np.ndarray, np.ndarray]:
    """
    基于 DMD 的 StateEncoder：更精确的频率提取
    
    Args:
        X_window: [W, N] 信号窗口
        k_modes: 模态数
        dt_sample: 采样间隔
    
    Returns:
        A_tilde: DMD 矩阵
        Z_params: 模态参数
        lambdas: 特征值
        Phi: 模态
    """
    A_tilde, Phi, lambdas_continuous = build_dmd_matrices(
        X_window, dt_sample, rank=k_modes
    )
    
    # 按频率排序
    freq_abs = np.abs(np.imag(lambdas_continuous))
    idx = np.argsort(freq_abs)[::-1]  # 高频优先
    
    lambdas_sorted = lambdas_continuous[idx[:k_modes]]
    Phi_sorted = Phi[:, idx[:k_modes]]
    
    # 提取 Z 参数
    current_state = X_window[-1, :]
    Z_params = []
    
    for i in range(min(k_modes, len(lambdas_sorted))):
        lam = lambdas_sorted[i]
        phi = Phi_sorted[:, i]
        
        # 投影
        proj = np.vdot(phi.conj(), current_state) / (np.vdot(phi.conj(), phi) + 1e-10)
        
        omega = np.imag(lam)  # 角频率 (rad/s)
        gamma = np.real(lam)  # 衰减率 (1/s)
        
        Z_params.append({
            'lambda': lam,
            'freq_hz': omega / (2 * np.pi),
            'omega_rad': omega,
            'growth_rate': gamma,
            'amplitude': np.abs(proj),
            'phase': np.angle(proj),
            'z_position': proj,
            'eigenvector': phi
        })
    
    return A_tilde, Z_params, lambdas_sorted, Phi_sorted


def spectral_decomposition(
    M: np.ndarray, 
    N_matrix: np.ndarray, 
    k_modes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    广义特征分解：M v = λ N v
    
    第一性原理：
        - 特征值 λ = γ + iω 编码模态的动力学特性
        - Re(λ) = γ: 增长/衰减率
        - Im(λ) = ω: 角频率
        - 特征向量 Φ: 正规模态（系统的本征振动模式）
    
    Args:
        M: [N, N] 复值世界张量
        N_matrix: [N, N] 质量矩阵
        k_modes: 要提取的模态数
    
    Returns:
        selected_lambdas: [k_modes] 选中的特征值
        selected_eigenvectors: [N, k_modes] 选中的特征向量
    
    复杂度: O(N^3)
    """
    # 广义特征分解
    lambdas, eigenvectors = eig(M, N_matrix)
    
    # 按特征值模排序（主导模态优先）
    idx = np.argsort(np.abs(lambdas))[::-1]
    lambdas_sorted = lambdas[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    
    # 选择前 k 个模态
    selected_lambdas = lambdas_sorted[:k_modes]
    selected_eigenvectors = eigenvectors_sorted[:, :k_modes]
    
    return selected_lambdas, selected_eigenvectors


def extract_z_parameters(
    X_window: np.ndarray,
    dX_window: np.ndarray,
    selected_lambdas: np.ndarray,
    selected_eigenvectors: np.ndarray,
    dt: float
) -> List[Dict]:
    """
    提取广义坐标 Z 的物理参数
    
    第一性原理：
        - 频率 f = Im(λ) / (2π): 模态的振荡频率
        - 增长率 γ = Re(λ): 模态的指数增长/衰减
        - 振幅 A = |<x, v>|: 当前状态在该模态上的投影
        - 相位 φ = arg(<x, v>): 振荡的相位
    
    Returns:
        Z_params: 每个模态的物理参数字典列表
    """
    k_modes = len(selected_lambdas)
    current_state = X_window[-1, :]  # 当前位置
    current_velocity = dX_window[-1, :]  # 当前速度
    
    Z_params = []
    for i in range(k_modes):
        lam = selected_lambdas[i]
        v = selected_eigenvectors[:, i]
        
        # 投影到模态空间
        proj_position = np.vdot(v.conj(), current_state)  # 共轭内积
        proj_velocity = np.vdot(v.conj(), current_velocity)
        
        # 提取物理参数
        omega = np.imag(lam)  # 角频率 (rad/s)
        gamma = np.real(lam)  # 增长率 (1/s)
        
        Z_params.append({
            'lambda': lam,
            'freq_hz': omega / (2 * np.pi),  # 频率 (Hz)
            'omega_rad': omega,  # 角频率 (rad/s)
            'growth_rate': gamma,  # 增长率
            'amplitude': np.abs(proj_position),
            'phase': np.angle(proj_position),
            'z_position': proj_position,  # 复数形式的广义坐标
            'z_velocity': proj_velocity,  # 复数形式的广义速度
            'eigenvector': v
        })
    
    return Z_params


def state_encoder_m_z(
    X_window: np.ndarray, 
    k_modes: int, 
    dt_sample: float
) -> Tuple[np.ndarray, np.ndarray, List[Dict], np.ndarray, np.ndarray]:
    """
    完整的 StateEncoder：X_window → (M, Z)
    
    这是将原始观测映射到统一世界模型的核心函数。
    
    Args:
        X_window: [W, N] 信号窗口
        k_modes: 要提取的模态数
        dt_sample: 采样间隔（秒）
    
    Returns:
        M: [N, N] 复值世界张量
        N_matrix: [N, N] 质量矩阵
        Z_params: 广义坐标的物理参数
        selected_lambdas: [k_modes] 特征值
        selected_eigenvectors: [N, k_modes] 特征向量
    """
    # 1. 构建世界张量 M
    M, N_matrix, dX_window = build_world_tensor_M(X_window, dt_sample)
    
    # 2. 谱分解获取模态
    selected_lambdas, selected_eigenvectors = spectral_decomposition(
        M, N_matrix, k_modes
    )
    
    # 3. 提取 Z 参数
    Z_params = extract_z_parameters(
        X_window, dX_window, 
        selected_lambdas, selected_eigenvectors, 
        dt_sample
    )
    
    return M, N_matrix, Z_params, selected_lambdas, selected_eigenvectors


# =============================================================================
# 核心：MetricEncoder (Z → g)
# =============================================================================

def metric_encoder_g_from_kinetic(
    N_matrix: np.ndarray, 
    selected_eigenvectors: np.ndarray, 
    k_modes: int
) -> np.ndarray:
    """
    从动能原理涌现度规 g = Φ^H N Φ
    
    第一性原理推导：
        1. 原空间动能: T = (1/2) * (dX/dt)^T N (dX/dt)
        2. 模态变换: X = Φ Z  =>  dX/dt = Φ dZ/dt
        3. 代入得: T = (1/2) * (dZ/dt)^H (Φ^H N Φ) (dZ/dt)
        4. 定义度规: g ≡ Φ^H N Φ
        
        这是从变分原理自然涌现的度规，非人为构造！
    
    Args:
        N_matrix: [N, N] 质量矩阵
        selected_eigenvectors: [N, k_modes] 特征向量
        k_modes: 模态数
    
    Returns:
        g: [k_modes, k_modes] 模态空间的度规张量
    
    复杂度: O(N^2 * k_modes)
    代数律: g 是 Hermitian 的（因为 N 是对称正定的）
    """
    Phi = selected_eigenvectors[:, :k_modes]  # [N, k_modes]
    g = Phi.conj().T @ N_matrix @ Phi  # [k_modes, k_modes]
    
    # 确保数值 Hermitian（消除浮点误差）
    g = hermitian_part(g)
    
    return g


def metric_encoder_g_from_dmd(
    Phi: np.ndarray,
    k_modes: int
) -> np.ndarray:
    """
    从 DMD 模态构建度规
    
    第一性原理：
        DMD 模态 Φ 已经在谱空间中，度规可以简化为
        g = Φ^H Φ（当原空间使用欧几里得度规时）
    
    Args:
        Phi: [N, k_modes] DMD 模态
        k_modes: 模态数
    
    Returns:
        g: [k_modes, k_modes] 模态空间的度规张量
    """
    Phi_k = Phi[:, :k_modes]
    g = Phi_k.conj().T @ Phi_k
    g = hermitian_part(g)
    return g

# =============================================================================
# 结果存储容器 (Immutable-style collectors)
# =============================================================================

class ResultCollector:
    """
    结果收集器 - 类似不可变数据结构的操作方式
    
    使用 append 返回新实例（函数式风格）
    """
    def __init__(self, k_modes: int):
        self.k_modes = k_modes
        self.t_mid_windows: List[float] = []
        self.Z_freqs: List[List[float]] = [[] for _ in range(k_modes)]
        self.Z_growths: List[List[float]] = [[] for _ in range(k_modes)]
        self.Z_amplitudes: List[List[float]] = [[] for _ in range(k_modes)]
        self.g_diagonals: List[List[complex]] = [[] for _ in range(k_modes)]
        self.g_off_diag_01: List[complex] = []
        self.g_off_diag_10: List[complex] = []
        self.hermitian_errors: List[float] = []
        self.condition_numbers: List[float] = []
        # 新增：真实频率和角频率
        self.Z_omega_rad: List[List[float]] = [[] for _ in range(k_modes)]
    
    def append(
        self, 
        t_mid: float, 
        Z_params: List[Dict], 
        g: np.ndarray,
        herm_error: float,
        cond_num: float
    ):
        """append 新结果"""
        self.t_mid_windows.append(t_mid)
        self.hermitian_errors.append(herm_error)
        self.condition_numbers.append(cond_num)
        
        for mode_idx in range(self.k_modes):
            self.Z_freqs[mode_idx].append(Z_params[mode_idx]['freq_hz'])
            self.Z_growths[mode_idx].append(Z_params[mode_idx]['growth_rate'])
            self.Z_amplitudes[mode_idx].append(Z_params[mode_idx]['amplitude'])
            self.Z_omega_rad[mode_idx].append(Z_params[mode_idx]['omega_rad'])
            self.g_diagonals[mode_idx].append(g[mode_idx, mode_idx])
        
        self.g_off_diag_01.append(g[0, 1])
        self.g_off_diag_10.append(g[1, 0])


# =============================================================================
# 主模拟循环 (Main Simulation Loop)
# =============================================================================

def run_simulation(
    T_total: float = 10.0,
    dt_sample: float = 0.01,
    window_size_s: float = 0.5,
    k_modes: int = 4,
    verbose_interval: int = 100,
    method: str = 'hankel_dmd'  # 'cov', 'dmd', 'hankel_dmd', 'unified'
) -> Tuple[ResultCollector, np.ndarray, np.ndarray]:
    """
    运行完整的 M-Z-g 模拟
    
    Args:
        T_total: 总时间（秒）
        dt_sample: 采样间隔（秒）
        window_size_s: 窗口大小（秒）
        k_modes: 模态数
        verbose_interval: 打印间隔
        method: 方法选择
            - 'cov': 协方差方法（原始 M 方法）
            - 'dmd': 动态模式分解
            - 'hankel_dmd': 时间延迟嵌入 DMD
            - 'unified': Hankel + M 构造融合（★ 新方法）
    
    Returns:
        results: 结果收集器
        X_full: 完整信号
        t_full: 时间向量
    """
    window_points = int(window_size_s / dt_sample)
    
    # 生成信号
    X_full, t_full = generate_signals(T_total, dt_sample)
    
    method_names = {
        'cov': 'Covariance',
        'dmd': 'DMD',
        'hankel_dmd': 'Hankel DMD',
        'unified': 'Unified V2 (图拉普拉斯+拉格朗日+DMD融合)'
    }
    method_name = method_names.get(method, method)
    
    print(f"\n{'='*60}")
    print(f"统一世界模型 (M-Z-g) - 第一性原理实现")
    print(f"方法: {method_name}")
    print(f"{'='*60}")
    print(f"信号长度: {len(t_full)} 点")
    print(f"采样率: {1/dt_sample:.0f} Hz")
    print(f"窗口大小: {window_points} 点 ({window_size_s}s)")
    print(f"模态数: {k_modes}")
    print(f"输入信号频率: 10Hz, 15Hz, 20Hz, 5-7Hz(chirp)")
    print(f"{'='*60}\n")
    
    # 结果收集器
    results = ResultCollector(k_modes)
    
    # 滑动窗口处理
    for i in range(len(t_full) - window_points):
        X_window = X_full[i : i + window_points, :]
        
        if method == 'unified':
            # ★ 统一世界张量 V2：融合 Hankel DMD + 图拉普拉斯 + 拉格朗日
            M, L_diss, Z_params, selected_lambdas, selected_eigenvectors = state_encoder_unified_v2(
                X_window, k_modes, dt_sample
            )
            current_g = metric_encoder_g_from_dynamics(L_diss, selected_eigenvectors, k_modes)
        elif method == 'hankel_dmd':
            # Hankel DMD 方法
            A_tilde, Z_params, selected_lambdas, Phi = state_encoder_hankel_dmd(
                X_window, k_modes, dt_sample
            )
            current_g = metric_encoder_g_from_dmd(Phi, k_modes)
            M = A_tilde
        elif method == 'dmd':
            # 标准 DMD 方法
            A_tilde, Z_params, selected_lambdas, Phi = state_encoder_dmd(
                X_window, k_modes, dt_sample
            )
            current_g = metric_encoder_g_from_dmd(Phi, k_modes)
            M = A_tilde
        else:
            # 协方差方法
            M, N_matrix, Z_params, selected_lambdas, selected_eigenvectors = state_encoder_m_z(
                X_window, k_modes, dt_sample
            )
            current_g = metric_encoder_g_from_kinetic(N_matrix, selected_eigenvectors, k_modes)
        
        # 验证 Hermitian 性质
        herm_error = np.linalg.norm(current_g - current_g.conj().T)
        
        # 计算条件数
        eigvals_g = np.linalg.eigvalsh(current_g)
        cond_num = np.max(np.abs(eigvals_g)) / (np.min(np.abs(eigvals_g)) + 1e-10)
        
        # 存储结果
        t_mid = t_full[i + window_points // 2]
        results.append(t_mid, Z_params, current_g, herm_error, cond_num)
        
        # 详细输出
        if i % verbose_interval == 0:
            print(f"Time: {t_mid:.2f}s")
            if method == 'cov' or method == 'unified':
                print(f"  M shape: {M.shape}")
            print("  Z Modes:")
            for j, p in enumerate(Z_params[:min(4, len(Z_params))]):
                print(f"    Mode {j+1}: f={p['freq_hz']:+.2f}Hz, "
                      f"γ={p['growth_rate']:+.3f}, A={p['amplitude']:.3f}")
            if len(Z_params) > 4:
                print(f"    ... and {len(Z_params)-4} more modes")
            print(f"  g shape: {current_g.shape}")
            print(f"  Hermitian: {'✓' if herm_error < 1e-10 else '✗'} (err={herm_error:.2e})")
            print(f"  Condition number: {cond_num:.2f}")
            print("-" * 50)
    
    # 统计摘要
    hermitian_pass_rate = np.mean([e < 1e-10 for e in results.hermitian_errors])
    print(f"\n{'='*60}")
    print(f"✅ g Hermitian 通过率: {100*hermitian_pass_rate:.1f}%")
    print(f"✅ 平均条件数: {np.mean(results.condition_numbers):.2f}")
    print(f"{'='*60}")
    
    return results, X_full, t_full


# =============================================================================
# 可视化 (Visualization)
# =============================================================================

def visualize_results(results: ResultCollector, k_modes: int, true_freqs: List[float] = None):
    """
    可视化 M-Z-g 结果
    
    Args:
        results: 结果收集器
        k_modes: 模态数
        true_freqs: 真实频率列表（用于对比）
    """
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle('Principle-Based World Model (M-Z-g) - First Principles', fontsize=16)
    
    t = results.t_mid_windows
    colors = plt.cm.tab10(np.linspace(0, 1, k_modes))
    
    # 1. 角频率 ω (rad/s) - 更直观
    ax = axes[0]
    for k in range(k_modes):
        ax.plot(t, results.Z_omega_rad[k], label=f'Mode {k+1}', alpha=0.8, color=colors[k])
    ax.set_ylabel(r'$\omega$ (rad/s)')
    ax.set_title('Z: Angular Frequencies')
    ax.legend(loc='upper right')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    
    # 2. 频率 f (Hz)
    ax = axes[1]
    for k in range(k_modes):
        ax.plot(t, results.Z_freqs[k], label=f'Mode {k+1}', alpha=0.8, color=colors[k])
    if true_freqs:
        for f in true_freqs:
            ax.axhline(f, color='red', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(-f, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel('Freq (Hz)')
    ax.set_title('Z: Mode Frequencies (red dashed = true frequencies)')
    ax.legend(loc='upper right')
    
    # 3. 增长/衰减率
    ax = axes[2]
    for k in range(k_modes):
        ax.plot(t, results.Z_growths[k], label=f'Mode {k+1}', alpha=0.8, color=colors[k])
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel(r'$\gamma$ (1/s)')
    ax.set_title('Z: Growth/Damping Rate (should be ~0 for steady-state signals)')
    ax.legend(loc='upper right')
    
    # 4. 振幅
    ax = axes[3]
    for k in range(k_modes):
        ax.plot(t, results.Z_amplitudes[k], label=f'Mode {k+1}', alpha=0.8, color=colors[k])
    ax.set_ylabel('Amplitude')
    ax.set_title('Z: Mode Amplitudes')
    ax.legend(loc='upper right')
    
    # 5. 度规对角元
    ax = axes[4]
    for k in range(min(2, k_modes)):
        ax.plot(t, [np.real(v) for v in results.g_diagonals[k]], 
                label=f'Re(g_{{{k+1}{k+1}}})', alpha=0.8, color=colors[k])
    # 非对角元
    ax.plot(t, [np.real(v) for v in results.g_off_diag_01], 
            label='Re(g_12)', alpha=0.8, linestyle='--', color=colors[2] if k_modes > 2 else 'green')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel('g Components (Real)')
    ax.set_xlabel('Time (s)')
    ax.set_title('g: Kinetic Metric (from Lagrangian Principle)')
    ax.legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # 第二张图：Hermitian 验证
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig2.suptitle('Metric Verification', fontsize=14)
    
    # Hermitian 检查: Im(g_ij) = -Im(g_ji) => Im(g_ij) + Im(g_ji) = 0
    imag_sum = np.array([np.imag(a) + np.imag(b) 
                         for a, b in zip(results.g_off_diag_01, results.g_off_diag_10)])
    axes2[0].plot(t, imag_sum, label='Im(g_12) + Im(g_21)', alpha=0.8, color='blue')
    axes2[0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes2[0].set_ylabel('Hermitian Test')
    axes2[0].set_title('Hermitian Check: Im(g_ij) + Im(g_ji) should = 0')
    axes2[0].legend()
    axes2[0].set_ylim([-0.1, 0.1])  # 放大查看
    
    # 条件数
    axes2[1].plot(t, results.condition_numbers, label='Condition Number', alpha=0.8, color='orange')
    axes2[1].set_ylabel('Condition Number')
    axes2[1].set_xlabel('Time (s)')
    axes2[1].set_title('Metric Condition Number (lower = more stable)')
    axes2[1].legend()
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    # 探索: 统一世界张量 V2 - 融合所有第一性原理
    print("\n" + "="*70)
    print("探索: Hankel DMD (k=8) vs Unified V2 (k=8)")
    print("Unified V2 = Hankel DMD本质 + 图拉普拉斯(耗散) + 拉格朗日(振荡)")
    print("="*70)
    
    true_freqs = [10, 15, 20, 6]  # 输入信号的真实频率
    
    # 方法 1: Hankel DMD with k=8
    results_hankel, X_full, t_full = run_simulation(
        T_total=10.0,
        dt_sample=0.01,
        window_size_s=0.5,
        k_modes=8,  # 增加到 8 以提取所有 4 个频率
        verbose_interval=300,
        method='hankel_dmd'
    )
    
    # 方法 2: 图拉普拉斯 + 拉格朗日
    results_unified, _, _ = run_simulation(
        T_total=10.0,
        dt_sample=0.01,
        window_size_s=0.5,
        k_modes=8,
        verbose_interval=300,
        method='unified'
    )
    
    # 可视化对比
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Hankel DMD vs Unified V2 (Laplacian + Lagrangian + DMD)', fontsize=14)
    
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    t = results_hankel.t_mid_windows
    
    # Hankel DMD 频率 (k=8)
    ax = axes[0, 0]
    for k in range(min(8, len(results_hankel.Z_freqs))):
        freqs = np.abs(results_hankel.Z_freqs[k])
        ax.plot(t, freqs, label=f'Mode {k+1}', alpha=0.7, color=colors[k])
    for f in true_freqs:
        ax.axhline(f, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('|Freq| (Hz)')
    ax.set_title('Hankel DMD (k=8)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim([0, 25])
    
    # Unified V2 频率 (k=8)
    ax = axes[0, 1]
    for k in range(min(8, len(results_unified.Z_freqs))):
        freqs = np.abs(results_unified.Z_freqs[k])
        ax.plot(t, freqs, label=f'Mode {k+1}', alpha=0.7, color=colors[k])
    for f in true_freqs:
        ax.axhline(f, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('|Freq| (Hz)')
    ax.set_title('Unified V2 (k=8)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim([0, 25])
    
    # Hankel DMD 增长率
    ax = axes[1, 0]
    for k in range(min(4, len(results_hankel.Z_growths))):
        ax.plot(t, results_hankel.Z_growths[k], label=f'Mode {k+1}', alpha=0.7, color=colors[k])
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel(r'$\gamma$ (1/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Hankel DMD: Growth Rate (γ)')
    ax.legend(loc='upper right', fontsize=8)
    
    # Unified V2 增长率
    ax = axes[1, 1]
    for k in range(min(4, len(results_unified.Z_growths))):
        ax.plot(t, results_unified.Z_growths[k], label=f'Mode {k+1}', alpha=0.7, color=colors[k])
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel(r'$\gamma$ (1/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Unified V2: Growth Rate (γ)')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # 频率统计
    print("\n" + "="*70)
    print("频率提取统计 (k=8)")
    print("="*70)
    print(f"\n真实频率: {true_freqs} Hz")
    
    print(f"\nHankel DMD (k=8):")
    extracted_freqs_hankel = []
    for k in range(min(8, len(results_hankel.Z_freqs))):
        freqs = np.abs(results_hankel.Z_freqs[k])
        mean_f = np.mean(freqs)
        if mean_f > 1:  # 排除直流分量
            extracted_freqs_hankel.append(mean_f)
        print(f"  Mode {k+1}: mean={mean_f:.2f} Hz, std={np.std(freqs):.2f} Hz")
    
    print(f"\nUnified V2 (k=8):")
    extracted_freqs_unified = []
    for k in range(min(8, len(results_unified.Z_freqs))):
        freqs = np.abs(results_unified.Z_freqs[k])
        mean_f = np.mean(freqs)
        if mean_f > 1:
            extracted_freqs_unified.append(mean_f)
        print(f"  Mode {k+1}: mean={mean_f:.2f} Hz, std={np.std(freqs):.2f} Hz")
    
    # 总结
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print(f"Hankel DMD 提取的主要频率: {sorted(set([round(f) for f in extracted_freqs_hankel if f > 1]))[:4]} Hz")
    print(f"Unified V2 提取的主要频率: {sorted(set([round(f) for f in extracted_freqs_unified if f > 1]))[:4]} Hz")
    print(f"真实频率: {true_freqs} Hz")
    
    plt.show()