"""正弦波预测测试 - 度规涌现模型可视化

测试度规涌现模型对复杂正弦波的预测能力，并可视化：
    1. 输入波形与预测波形对比
    2. 度规张量热力图
    3. Christoffel 符号分布
    4. 隐空间轨迹
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed: int = 42):
    """固定随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_complex_sinwave(
    length: int,
    dt: float = 0.1,
    frequencies: List[float] = [1.0, 2.5, 0.5],
    amplitudes: List[float] = [1.0, 0.4, 0.6],
    phases: List[float] = [0.0, np.pi/3, np.pi/6],
    noise_std: float = 0.02,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成复杂正弦波（多频率叠加 + 噪声）
    
    y(t) = sum_i(A_i * sin(2π * f_i * t + φ_i)) + noise
    
    Args:
        length: 序列长度
        dt: 时间步长
        frequencies: 频率列表
        amplitudes: 振幅列表
        phases: 相位列表
        noise_std: 噪声标准差
        normalize: 是否归一化到 [-1, 1]
    
    Returns:
        (时间序列, 值序列)
    """
    t = np.arange(length) * dt
    y = np.zeros(length)
    
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        y += amp * np.sin(2 * np.pi * freq * t + phase)
    
    if noise_std > 0:
        y += np.random.randn(length) * noise_std
    
    if normalize:
        y_min, y_max = y.min(), y.max()
        y = 2 * (y - y_min) / (y_max - y_min + 1e-8) - 1  # 归一化到 [-1, 1]
    
    return t, y


def dynamic_fw(epoch: int, total: int, fw_min: float = 1.0, fw_max: float = 10.0) -> float:
    """动态外力惩罚权重"""
    if epoch < total * 0.2:
        return fw_min
    elif epoch < total * 0.6:
        progress = (epoch - total * 0.2) / (total * 0.4)
        return fw_min + (fw_max - fw_min) * progress
    else:
        return fw_max


def dynamic_ss(epoch: int, total: int, ss_max: float = 0.6) -> float:
    """动态 Scheduled Sampling 概率"""
    if epoch < total * 0.3:
        return 0.0
    progress = (epoch - total * 0.3) / (total * 0.7)
    return ss_max * progress


def generate_simple_sinwave(
    seq_len: int,
    freq: float = 0.5,
    amp: float = 1.0,
    phase: float = 0.0,
    dt: float = 0.1,
) -> np.ndarray:
    """
    生成简单的单频正弦波
    
    y(t) = amp * sin(2π * freq * t + phase)
    """
    t = np.arange(seq_len) * dt
    y = amp * np.sin(2 * np.pi * freq * t + phase)
    return y


def train_sinwave_model(
    epochs: int = 50,
    z_dim: int = 2,
    n_train_samples: int = 10,
    seq_len: int = 40,
    seed: int = 42,
    verbose: bool = True,
    window_size: int = 4,
) -> Tuple[torch.nn.Module, Dict]:
    """
    训练正弦波预测模型 - d²x 损失驱动自涌现版
    
    第一性原理训练范式：
        1. 核心损失：d²x 预测误差
        2. 外力缚化：F_ext = 0，迫使模型使用几何加速度
        3. z 空间自涌现：通过链式反向传播，迫使 StateEncoder 
           学习物理意义的相空间坐标
    
    参数:
        epochs: 训练轮数
        z_dim: 隐空间维度（建议 2-4，对应物理自由度）
        n_train_samples: 训练样本数
        seq_len: 序列长度
        window_size: 状态推断窗口大小
    """
    from .model import MetricEvolutionModel
    
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64
    
    if verbose:
        print(f"\n{'='*60}")
        print("动力学状态推断网络 - d²x 损失驱动训练")
        print(f"  z_dim={z_dim}, epochs={epochs}, window_size={window_size}")
        print(f"  核心：d²x 损失驱动 z 空间自涌现")
        print(f"  约束：F_ext=0，纯几何推导")
        print('='*60)
    
    # 创建模型
    model = MetricEvolutionModel(
        z_dim=z_dim, 
        input_dim=1,
        hidden_mult=4, 
        unconstrained=False,
        window_size=window_size,
    ).to(device).to(dtype)
    
    # 禁用外力：第一性原理要求，纯几何推导
    for param in model.external_force.parameters():
        param.requires_grad = False
        param.data.zero_()
    
    # 优化器：AdamW + 低学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    # 生成训练数据：简单正弦波，不同相位
    train_data = []
    freq = 0.5
    for i in range(n_train_samples):
        phase = i * 2 * np.pi / n_train_samples
        y = generate_simple_sinwave(seq_len, freq=freq, amp=1.0, phase=phase)
        seq = torch.tensor(y, device=device, dtype=dtype).unsqueeze(0)
        train_data.append(seq)
    
    history = {
        'loss': [], 
        'd2x_loss': [], 
        'det_reg': [], 
        'geo_ratio': [],
        'z_cycle_loss': [],
        'multi_step_loss': [],
    }
    
    # 周期点数：T = 1/freq = 2s = 20点 (dt=0.1)
    period_points = int(1.0 / freq / 0.1)
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        total_d2x = 0
        total_det = 0
        total_geo_ratio = 0
        total_cycle = 0
        total_multi = 0
        
        for seq in train_data:
            B = 1
            L = seq.shape[1]
            
            # ===== 1. 多步预测损失（核心！） =====
            # 从窗口开始，多步自回归预测
            n_pred_steps = min(10, L - window_size - 2)
            if n_pred_steps > 0:
                start_idx = window_size + 2  # 确保有足够的点计算 d²x
                current = seq[:, :start_idx].clone()
                multi_loss = torch.tensor(0.0, device=device, dtype=dtype)
                
                for step in range(n_pred_steps):
                    # 前向传播
                    out = model.forward_with_derivatives(current)
                    
                    # 预测下一个 d²x
                    d2x_pred = out['pred_d2x']  # [B]
                    
                    # 下一个位置
                    x_last = out['x_last']
                    dx_last = out['dx_last']
                    x_next = x_last + dx_last + d2x_pred
                    
                    # 目标
                    if start_idx + step < L:
                        true_next = seq[:, start_idx + step]
                        # 远期误差权重更高
                        step_weight = 1.0 + step * 0.2
                        multi_loss = multi_loss + step_weight * ((x_next - true_next) ** 2).mean()*10
                    
                    # 更新序列
                    current = torch.cat([current, x_next.unsqueeze(1)], dim=1)
                
                multi_loss = multi_loss / max(1, n_pred_steps)
            else:
                multi_loss = torch.tensor(0.0, device=device, dtype=dtype)
            
            # ===== 2. 单步 d²x 损失 =====
            out = model.forward_with_derivatives(seq)
            d2x_true = out['d2x_true']
            d2x_pred = out['pred_d2x']
            d2x_loss = ((d2x_pred - d2x_true) ** 2).mean()
            
            # ===== 3. 度规正则化 =====
            g_last = out['g'][:, -1]
            det_g = torch.linalg.det(g_last)
            det_reg = ((det_g.abs() - 0.1) ** 2).mean() * 0.01
            
            # ===== 4. z 空间周期轨道损失 =====
            z = out['z']
            if z.shape[1] > period_points:
                z_t = z[:, :-period_points, :]
                z_t_T = z[:, period_points:, :]
                cycle_loss = ((z_t - z_t_T) ** 2).mean() * 10
            else:
                cycle_loss = torch.tensor(0.0, device=device, dtype=dtype)
            
            # ===== 总损失 =====
            # d2x_loss 是主要驱动力，权重更高
            loss = 5.0 * d2x_loss + 1.0 * multi_loss + det_reg + cycle_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 统计几何加速度占比
            geo_norm = out['a_geodesic'].norm().item()
            ext_norm = out['F_external'].norm().item()
            geo_ratio = geo_norm / (geo_norm + ext_norm + 1e-8)
            
            total_d2x += d2x_loss.item()
            total_det += det_reg.item()
            total_geo_ratio += geo_ratio
            total_cycle += cycle_loss.item() if isinstance(cycle_loss, torch.Tensor) else cycle_loss
            total_multi += multi_loss.item() if isinstance(multi_loss, torch.Tensor) else multi_loss
        
        scheduler.step()
        
        n = n_train_samples
        history['d2x_loss'].append(total_d2x / n)
        history['det_reg'].append(total_det / n)
        history['geo_ratio'].append(total_geo_ratio / n)
        history['z_cycle_loss'].append(total_cycle / n)
        history['multi_step_loss'].append(total_multi / n)
        history['loss'].append((total_d2x + total_det + total_cycle + total_multi) / n)
        
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            print(f"E{epoch:3d}: d2x={total_d2x/n:.6f}, multi={total_multi/n:.6f}, "
                  f"geo_ratio={total_geo_ratio/n:.1%}, cycle={total_cycle/n:.6f}")
    
    if verbose:
        print(f"\n训练完成!")
        print(f"  最终 d²x 损失: {history['d2x_loss'][-1]:.6f}")
        print(f"  最终多步损失: {history['multi_step_loss'][-1]:.6f}")
        print(f"  最终几何占比: {history['geo_ratio'][-1]:.1%}")
    
    return model, history


def predict_sinwave(
    model: torch.nn.Module,
    given_seq: np.ndarray,
    n_predict: int,
    device: str = 'cpu',
) -> Dict:
    """
    使用模型预测正弦波
    
    Args:
        model: 训练好的模型
        given_seq: 给定的序列
        n_predict: 预测步数
    
    Returns:
        预测结果字典
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    
    current = torch.tensor(given_seq, device=device, dtype=dtype).unsqueeze(0)
    
    predictions = []
    metrics_data = []  # 保存度规信息
    z_trajectory = []  # 隐空间轨迹
    christoffel_data = []  # Christoffel 符号
    
    with torch.no_grad():
        for step in range(n_predict):
            out = model.forward(current)
            pred = out['x_new'].squeeze().item()
            
            # 保存度规相关数据
            metrics_data.append({
                'metric': out['metric'].cpu().numpy(),
                'a_geodesic': out['a_geodesic'].cpu().numpy(),
                'F_external': out['F_external'].cpu().numpy(),
            })
            
            # 隐空间轨迹
            z_trajectory.append(out['z'][:, -1].cpu().numpy())
            
            # Christoffel 符号
            if out.get('Gamma') is not None:
                christoffel_data.append(out['Gamma'].cpu().numpy())
            
            predictions.append(pred)
            
            # 继续外推
            new_val = torch.tensor([[pred]], device=device, dtype=dtype)
            current = torch.cat([current, new_val], dim=1)
    
    return {
        'predictions': np.array(predictions),
        'metrics': metrics_data,
        'z_trajectory': np.array(z_trajectory),
        'christoffel': christoffel_data if christoffel_data else None,
    }


def visualize_results(
    t_given: np.ndarray,
    y_given: np.ndarray,
    t_predict: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics_data: List[Dict],
    z_trajectory: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    可视化预测结果和度规信息
    
    生成四个子图：
        1. 输入波形与预测对比
        2. 度规张量热力图（最后一步）
        3. 几何加速度 vs 外力加速度
        4. 隐空间轨迹 (2D PCA)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('度规涌现模型 - 正弦波预测可视化', fontsize=14, fontweight='bold')
    
    # ========== 1. 波形对比 ==========
    ax1 = axes[0, 0]
    ax1.plot(t_given, y_given, 'b-', linewidth=2, label='输入序列', marker='o', markersize=4)
    ax1.plot(t_predict, y_true, 'g--', linewidth=2, label='真实值', marker='s', markersize=4)
    ax1.plot(t_predict, y_pred, 'r-', linewidth=2, label='预测值', marker='^', markersize=4)
    ax1.axvline(x=t_given[-1], color='gray', linestyle=':', alpha=0.7, label='预测起点')
    ax1.fill_between(t_predict, y_true, y_pred, alpha=0.2, color='red', label='预测误差')
    ax1.set_xlabel('时间 t')
    ax1.set_ylabel('幅值')
    ax1.set_title('输入波形 vs 预测波形')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. 度规张量热力图 ==========
    ax2 = axes[0, 1]
    last_metric = metrics_data[-1]['metric']
    # 取最后一个时间步的度规
    if last_metric.ndim == 4:  # [B, L, D, D]
        last_metric = last_metric[0, -1]  # 取 [D, D]
    elif last_metric.ndim == 3:  # [B, D, D]
        last_metric = last_metric[0]  # 取 [D, D]
    
    if last_metric.ndim == 2:
        im = ax2.imshow(last_metric, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax2, label='度规分量值')
        ax2.set_title(f'度规张量 g(z) [{last_metric.shape[0]}×{last_metric.shape[1]}]')
        ax2.set_xlabel('维度 j')
        ax2.set_ylabel('维度 i')
    else:
        ax2.text(0.5, 0.5, f'度规数据维度异常: {last_metric.shape}', ha='center', va='center')
        ax2.set_title('度规张量')
    
    # ========== 3. 几何 vs 外力加速度 ==========
    ax3 = axes[1, 0]
    geo_norms = [np.linalg.norm(m['a_geodesic']) for m in metrics_data]
    ext_norms = [np.linalg.norm(m['F_external']) for m in metrics_data]
    steps = np.arange(1, len(geo_norms) + 1)
    
    ax3.bar(steps - 0.2, geo_norms, width=0.4, label='几何加速度 |a_geo|', color='blue', alpha=0.7)
    ax3.bar(steps + 0.2, ext_norms, width=0.4, label='外力加速度 |F_ext|', color='orange', alpha=0.7)
    ax3.set_xlabel('预测步数')
    ax3.set_ylabel('加速度范数')
    ax3.set_title('几何加速度 vs 外力加速度')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 计算几何占比
    total_geo = sum(geo_norms)
    total_ext = sum(ext_norms)
    geo_ratio = total_geo / (total_geo + total_ext + 1e-8)
    ax3.text(0.98, 0.98, f'几何占比: {geo_ratio:.1%}', transform=ax3.transAxes,
             ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========== 4. 隐空间轨迹 ==========
    ax4 = axes[1, 1]
    z_traj = z_trajectory.squeeze()
    if z_traj.ndim == 2 and z_traj.shape[1] >= 2:
        # 使用前两个维度或 PCA
        if z_traj.shape[1] > 2:
            # 简单 PCA
            z_centered = z_traj - z_traj.mean(axis=0)
            cov = np.cov(z_centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            z_2d = z_centered @ eigvecs[:, idx[:2]]
            xlabel, ylabel = 'PC1', 'PC2'
        else:
            z_2d = z_traj
            xlabel, ylabel = 'z_0', 'z_1'
        
        # 绘制轨迹
        colors = plt.cm.viridis(np.linspace(0, 1, len(z_2d)))
        for i in range(len(z_2d) - 1):
            ax4.plot(z_2d[i:i+2, 0], z_2d[i:i+2, 1], color=colors[i], linewidth=2)
        ax4.scatter(z_2d[:, 0], z_2d[:, 1], c=np.arange(len(z_2d)), cmap='viridis', s=50, zorder=5)
        ax4.scatter(z_2d[0, 0], z_2d[0, 1], c='green', s=100, marker='o', label='起点', zorder=6)
        ax4.scatter(z_2d[-1, 0], z_2d[-1, 1], c='red', s=100, marker='*', label='终点', zorder=6)
        ax4.set_xlabel(xlabel)
        ax4.set_ylabel(ylabel)
        ax4.set_title('隐空间轨迹 z(t)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '隐空间数据维度不足', ha='center', va='center')
        ax4.set_title('隐空间轨迹')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def test_sinwave_prediction(
    n_given: int = 10,
    n_predict: int = 50,
    epochs: int = 50,
    z_dim: int = 2,
    seed: int = 42,
    save_path: Optional[str] = 'sinwave_prediction.png',
    show: bool = True,
    window_size: int = 4,
):
    """
    完整测试：训练模型 -> 预测正弦波 -> 可视化
    
    动力学状态推断版：
        - d²x 损失驱动 z 空间自涌现
        - z_dim=2 小模型（对应位置+速度）
        - 禁止外力，纯几何推导
        - 窗口大小 window_size=4（极小的观测窗口）
    
    参数:
        n_given: 给定的观测点数
        n_predict: 预测步数
        epochs: 训练轮数
        z_dim: 隐空间维度（2-4）
        window_size: 状态推断窗口大小
    """
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*60)
    print("动力学状态推断网络 - 正弦波预测测试")
    print("="*60)
    print(f"  n_given={n_given}, n_predict={n_predict}")
    print(f"  z_dim={z_dim}, window_size={window_size}")
    print(f"  epochs={epochs}")
    
    # 1. 训练模型
    model, history = train_sinwave_model(
        epochs=epochs,
        z_dim=z_dim,
        seed=seed,
        verbose=True,
        window_size=window_size,
    )
    
    # 2. 生成测试数据（让输入包含峰值，验证能否学会返回）
    total_len = n_given + n_predict
    # 相位设为 -π/2，让序列从负值开始，经过峰值
    test_phase = -np.pi / 2
    y = generate_simple_sinwave(total_len, freq=0.5, amp=1.0, phase=test_phase)
    t = np.arange(total_len) * 0.1
    
    t_given = t[:n_given]
    y_given = y[:n_given]
    t_predict = t[n_given:]
    y_true = y[n_given:]
    
    # 3. 预测
    print(f"\n外推测试: 给定 {n_given} 点 → 预测 {n_predict} 点")
    result = predict_sinwave(model, y_given, n_predict, device)
    y_pred = result['predictions']
    
    # 4. 计算误差
    errors = np.abs(y_pred - y_true)
    avg_error = errors.mean()
    max_error = errors.max()
    
    print(f"\n{'='*40}")
    print("预测结果:")
    for i in range(min(10, n_predict)):
        print(f"  Step {i+1:2d}: pred={y_pred[i]:+.4f}, true={y_true[i]:+.4f}, err={errors[i]:.4f}")
    if n_predict > 10:
        print(f"  ... (共 {n_predict} 步)")
    print(f"{'='*40}")
    print(f"平均误差: {avg_error:.4f}")
    print(f"最大误差: {max_error:.4f}")
    
    # 5. 可视化
    if show or save_path:
        print("\n生成可视化...")
        fig = visualize_results(
            t_given, y_given,
            t_predict, y_true, y_pred,
            result['metrics'],
            result['z_trajectory'],
            save_path=save_path,
            show=show,
        )
    else:
        fig = None
    
    return {
        'model': model,
        'history': history,
        'predictions': y_pred,
        'true_values': y_true,
        'errors': errors,
        'avg_error': avg_error,
        'max_error': max_error,
        'metrics': result['metrics'],
        'z_trajectory': result['z_trajectory'],
        'figure': fig,
    }


def visualize_metric_evolution(
    model: torch.nn.Module,
    seq: np.ndarray,
    save_path: Optional[str] = 'metric_evolution.png',
    show: bool = True,
):
    """
    可视化度规随时间的演化
    
    展示度规张量在序列不同位置的变化
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    seq_tensor = torch.tensor(seq, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
    
    # 收集不同窗口位置的度规
    metrics = []
    window_size = 10
    positions = range(0, len(seq) - window_size, 3)
    
    with torch.no_grad():
        for pos in positions:
            window = seq_tensor[:, pos:pos+window_size]
            out = model.forward(window)
            metrics.append(out['metric'].cpu().numpy().squeeze())
    
    # 绘制
    n_metrics = len(metrics)
    cols = min(4, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    fig.suptitle('度规张量 g(z) 随时间演化', fontsize=14, fontweight='bold')
    
    if n_metrics == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    
    for idx, (pos, metric) in enumerate(zip(positions, metrics)):
        row, col = idx // cols, idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        
        im = ax.imshow(metric, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_title(f't = {pos*0.1:.1f}s')
        ax.set_xlabel('j')
        ax.set_ylabel('i')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 隐藏多余的子图
    for idx in range(n_metrics, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"度规演化图已保存: {save_path}")
    
    if show:
        plt.show()
    
    return fig


if __name__ == '__main__':
    # 运行完整测试：动力学状态推断版
    result = test_sinwave_prediction(
        n_given=20,
        n_predict=60,
        epochs=100,
        z_dim=2,
        save_path='sinwave_prediction.png',
        show=True,
        window_size=4,
    )
    
    # 可选：可视化度规演化
    # t, y = generate_complex_sinwave(100)
    # visualize_metric_evolution(result['model'], y, save_path='metric_evolution.png')
