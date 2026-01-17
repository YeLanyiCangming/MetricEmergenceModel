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
    epochs: int = 10,
    z_dim: int = 4,
    n_train_samples: int = 10,
    seq_len: int = 40,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[torch.nn.Module, Dict]:
    """
    训练正弦波预测模型 - 纯几何版
    
    特点：
    1. 只用 10 条简单正弦波训练
    2. z_dim=4 小模型
    3. 完全禁止外力，纯几何推导
    4. 目标: 100% 几何占比
    """
    from .model import MetricEvolutionModel
    
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64
    
    if verbose:
        print(f"\n{'='*60}")
        print("正弦波预测 - 纯几何推导（外力=0）")
        print(f"  z_dim={z_dim}, epochs={epochs}")
        print('='*60)
    
    # 创建模型
    model = MetricEvolutionModel(
        z_dim=z_dim, input_dim=1,
        hidden_mult=4, unconstrained=False,
    ).to(device).to(dtype)
    
    # 禁用外力网络的梯度
    for param in model.external_force.parameters():
        param.requires_grad = False
        param.data.zero_()  # 置零
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 预生成 10 条训练数据
    train_data = []
    freq = 0.5
    for i in range(n_train_samples):
        phase = i * 2 * np.pi / n_train_samples
        y = generate_simple_sinwave(seq_len, freq=freq, amp=1.0, phase=phase)
        seq = torch.tensor(y, device=device, dtype=dtype).unsqueeze(0)
        train_data.append(seq)
    
    history = {'loss': [], 'geo_ratio': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        total_loss = 0
        total_geo = 0
        
        for seq in train_data:
            out = model.forward(seq[:, :-1])
            target = seq[:, -1]
            
            # 纯几何损失：只用测地线加速度
            pred = out['x_last'] + out['dx_last'] + model.decoder(out['a_geodesic'])[0]
            loss = ((pred - target) ** 2).mean()
            
            # 度规正则化
            g_last = out['g'][:, -1]
            det_g = torch.linalg.det(g_last)
            det_reg = ((det_g - 0.1) ** 2).mean() * 0.01
            
            total = loss + det_reg
            
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_geo += out['a_geodesic'].norm().item()
        
        history['loss'].append(total_loss / n_train_samples)
        history['geo_ratio'].append(1.0)  # 100% 几何
        
        # 每轮输出
        if verbose:
            print(f"E{epoch:2d}: loss={total_loss/n_train_samples:.4f}, geo=100.0%")
    
    if verbose:
        print(f"\n训练完成! 纯几何推导")
    
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
    n_given: int = 20,
    n_predict: int = 10,
    epochs: int = 10,
    z_dim: int = 4,
    seed: int = 42,
    save_path: Optional[str] = 'sinwave_prediction.png',
    show: bool = True,
):
    """
    完整测试：训练模型 -> 预测正弦波 -> 可视化
    
    纯几何版：
    - 10 条简单正弦波训练
    - z_dim=4 小模型
    - 禁止外力，纯几何推导
    """
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*60)
    print("正弦波预测测试 - 度规涌现模型")
    print("="*60)
    
    # 1. 训练模型
    model, history = train_sinwave_model(
        epochs=epochs,
        z_dim=z_dim,
        seed=seed,
        verbose=True,
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
    for i in range(n_predict):
        print(f"  Step {i+1:2d}: pred={y_pred[i]:+.4f}, true={y_true[i]:+.4f}, err={errors[i]:.4f}")
    print(f"{'='*40}")
    print(f"平均误差: {avg_error:.4f}")
    print(f"最大误差: {max_error:.4f}")
    
    # 5. 可视化
    print("\n生成可视化...")
    fig = visualize_results(
        t_given, y_given,
        t_predict, y_true, y_pred,
        result['metrics'],
        result['z_trajectory'],
        save_path=save_path,
        show=show,
    )
    
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
    # 运行完整测试
    result = test_sinwave_prediction(
        n_given=30,
        n_predict=20,
        epochs=300,
        z_dim=8,
        seed=42,
        save_path='sinwave_prediction.png',
        show=True,
    )
    
    # 可选：可视化度规演化
    # t, y = generate_complex_sinwave(100)
    # visualize_metric_evolution(result['model'], y, save_path='metric_evolution.png')
