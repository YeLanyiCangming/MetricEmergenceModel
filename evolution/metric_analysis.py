"""度规分析模块 - 纯函数式接口

核心功能：
    1. MetricSignature: 度规签名分析（特征值、洛伦兹检测）
    2. MetricCollector: 训练过程中的度规数据收集
    3. 可视化辅助函数（为未来扩展准备）

设计原则：
    - 纯函数：无副作用，输入决定输出
    - 组合子：可链式组合的操作
    - 类型安全：明确的输入输出类型
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import torch
from torch import Tensor


# =============================================================================
# 类型别名 (Type Aliases)
# =============================================================================

MetricTensor = Tensor     # [B, L, D, D] 度规张量
Eigenvalues = Tensor      # [D] 或 [B, D] 特征值
ZPath = Tensor            # [B, L, D] z 轨迹


# =============================================================================
# 度规签名分析 (Metric Signature Analysis) - 纯函数
# =============================================================================

@dataclass(frozen=True)
class MetricSignature:
    """度规签名数据结构 (不可变)"""
    eigenvalues: Tuple[float, ...]   # 排序后的特征值
    pos_count: int                    # 正特征值数量
    neg_count: int                    # 负特征值数量
    zero_count: int                   # 零特征值数量
    
    @property
    def signature_str(self) -> str:
        """签名字符串表示"""
        return f"({self.neg_count}-, {self.pos_count}+)"
    
    @property
    def is_positive_definite(self) -> bool:
        """是否正定"""
        return self.neg_count == 0 and self.zero_count == 0
    
    @property
    def is_lorentzian(self) -> bool:
        """是否洛伦兹签名 (-,+,+,...) 或 (+,...,+,-)"""
        return self.neg_count == 1 and self.zero_count == 0
    
    @property
    def is_degenerate(self) -> bool:
        """是否退化（有零特征值）"""
        return self.zero_count > 0


def get_eigenvalues(g: Tensor, eps: float = 1e-6) -> Tensor:
    """
    计算度规张量的特征值
    
    Args:
        g: [B, L, D, D] 或 [B, D, D] 或 [D, D] 度规张量
        eps: 零判定阈值
    
    Returns:
        特征值张量，形状与输入匹配（最后两维合并为一维）
    
    复杂度: O(D^3) 每个矩阵
    代数律: eigenvalues(g) = eigenvalues(g^T) (对称矩阵)
    """
    original_shape = g.shape[:-2]
    D = g.shape[-1]
    g_flat = g.reshape(-1, D, D)
    
    eigenvalues = torch.linalg.eigvalsh(g_flat)  # [N, D]
    
    return eigenvalues.reshape(*original_shape, D) if original_shape else eigenvalues.squeeze(0)


def get_signature(g: Tensor, eps: float = 1e-6) -> MetricSignature:
    """
    计算单个度规矩阵的签名
    
    Args:
        g: [D, D] 单个度规矩阵
        eps: 零判定阈值
    
    Returns:
        MetricSignature 数据结构
    
    复杂度: O(D^3)
    """
    eigenvalues = get_eigenvalues(g, eps)
    sorted_eig, _ = eigenvalues.sort()
    
    pos_count = (eigenvalues > eps).sum().item()
    neg_count = (eigenvalues < -eps).sum().item()
    zero_count = ((eigenvalues.abs() <= eps)).sum().item()
    
    return MetricSignature(
        eigenvalues=tuple(sorted_eig.tolist()),
        pos_count=int(pos_count),
        neg_count=int(neg_count),
        zero_count=int(zero_count),
    )


def batch_signatures(g: Tensor, eps: float = 1e-6) -> List[MetricSignature]:
    """
    批量计算度规签名
    
    Args:
        g: [B, D, D] 或 [B, L, D, D] 度规张量
        eps: 零判定阈值
    
    Returns:
        MetricSignature 列表
    
    复杂度: O(N * D^3) where N = product of batch dims
    """
    D = g.shape[-1]
    g_flat = g.reshape(-1, D, D)
    
    return [get_signature(g_flat[i], eps) for i in range(g_flat.shape[0])]


def aggregate_signatures(signatures: List[MetricSignature]) -> Dict[str, Any]:
    """
    聚合多个签名的统计信息
    
    Args:
        signatures: MetricSignature 列表
    
    Returns:
        聚合统计字典
    
    复杂度: O(N)
    """
    if not signatures:
        return {}
    
    pos_counts = [s.pos_count for s in signatures]
    neg_counts = [s.neg_count for s in signatures]
    
    # 统计签名分布
    signature_dist = {}
    for s in signatures:
        key = s.signature_str
        signature_dist[key] = signature_dist.get(key, 0) + 1
    
    # 找出最常见的签名
    most_common = max(signature_dist.items(), key=lambda x: x[1])
    
    return {
        'count': len(signatures),
        'pos_definite_ratio': sum(1 for s in signatures if s.is_positive_definite) / len(signatures),
        'lorentzian_ratio': sum(1 for s in signatures if s.is_lorentzian) / len(signatures),
        'degenerate_ratio': sum(1 for s in signatures if s.is_degenerate) / len(signatures),
        'signature_distribution': signature_dist,
        'most_common_signature': most_common[0],
        'avg_pos_count': sum(pos_counts) / len(pos_counts),
        'avg_neg_count': sum(neg_counts) / len(neg_counts),
    }


def is_lorentzian(g: Tensor, eps: float = 1e-6) -> bool:
    """
    判断度规是否为洛伦兹签名
    
    Args:
        g: [D, D] 单个度规矩阵
    
    Returns:
        True if 洛伦兹签名
    
    复杂度: O(D^3)
    """
    return get_signature(g, eps).is_lorentzian


# =============================================================================
# 度规数据收集器 (Metric Data Collector)
# =============================================================================

@dataclass
class MetricSnapshot:
    """度规快照 (不可变记录)"""
    z: Tensor                      # [L, D] z 轨迹
    g: Tensor                      # [L, D, D] 对应的度规
    rule: str                      # 规则名称
    epoch: int                     # 训练 epoch
    signature: MetricSignature     # 签名信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典"""
        return {
            'z': self.z.cpu().numpy().tolist(),
            'g': self.g.cpu().numpy().tolist(),
            'rule': self.rule,
            'epoch': self.epoch,
            'signature': self.signature.signature_str,
            'eigenvalues': self.signature.eigenvalues,
        }


@dataclass
class MetricCollector:
    """
    度规数据收集器
    
    用于在训练过程中收集和保存度规信息
    """
    snapshots: List[MetricSnapshot] = field(default_factory=list)
    
    def collect(
        self, 
        z: Tensor, 
        g: Tensor, 
        rule: str, 
        epoch: int,
        sample_indices: Optional[List[int]] = None,
    ) -> None:
        """
        收集度规快照
        
        Args:
            z: [B, L, D] z 轨迹
            g: [B, L, D, D] 度规张量
            rule: 规则名称
            epoch: 当前 epoch
            sample_indices: 要采样的 batch 索引，None 表示全部
        """
        B = z.shape[0]
        indices = sample_indices if sample_indices is not None else range(B)
        
        for i in indices:
            # 取最后一个位置的度规
            z_i = z[i].detach().cpu()         # [L, D]
            g_i = g[i].detach().cpu()         # [L, D, D]
            
            # 计算签名
            sig = get_signature(g_i[-1])       # 最后位置的签名
            
            snapshot = MetricSnapshot(
                z=z_i,
                g=g_i,
                rule=rule,
                epoch=epoch,
                signature=sig,
            )
            self.snapshots.append(snapshot)
    
    def get_by_rule(self, rule: str) -> List[MetricSnapshot]:
        """按规则筛选快照"""
        return [s for s in self.snapshots if s.rule == rule]
    
    def get_by_epoch(self, epoch: int) -> List[MetricSnapshot]:
        """按 epoch 筛选快照"""
        return [s for s in self.snapshots if s.epoch == epoch]
    
    def get_signature_stats(self) -> Dict[str, Any]:
        """获取签名统计"""
        return aggregate_signatures([s.signature for s in self.snapshots])
    
    def save(self, path: str) -> None:
        """保存到文件"""
        import json
        data = {
            'snapshots': [s.to_dict() for s in self.snapshots],
            'stats': self.get_signature_stats(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear(self) -> None:
        """清空收集器"""
        self.snapshots.clear()


# =============================================================================
# 训练历史记录 (Training History)
# =============================================================================

@dataclass
class TrainingHistory:
    """训练历史记录 - 用于可视化"""
    mode: str = ""  # "constrained" 或 "unconstrained"
    epochs: List[int] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    nll: List[float] = field(default_factory=list)
    geo_ratio: List[float] = field(default_factory=list)
    det_abs: List[float] = field(default_factory=list)
    force_weight: List[float] = field(default_factory=list)
    
    # 测试结果
    test_rules: List[str] = field(default_factory=list)
    test_errors: List[float] = field(default_factory=list)
    test_geo_ratios: List[float] = field(default_factory=list)
    test_eigenvalues: List[Tuple[float, ...]] = field(default_factory=list)
    test_signatures: List[str] = field(default_factory=list)
    
    def record_epoch(self, epoch: int, loss: float, nll: float, 
                     geo_ratio: float, det_abs: float, fw: float):
        """记录一个 epoch 的数据"""
        self.epochs.append(epoch)
        self.loss.append(loss)
        self.nll.append(nll)
        self.geo_ratio.append(geo_ratio)
        self.det_abs.append(det_abs)
        self.force_weight.append(fw)
    
    def record_test(self, rule: str, error: float, geo_ratio: float,
                    eigenvalues: Tuple[float, ...] = None, signature: str = None):
        """记录测试结果"""
        self.test_rules.append(rule)
        self.test_errors.append(error)
        self.test_geo_ratios.append(geo_ratio)
        self.test_eigenvalues.append(eigenvalues or ())
        self.test_signatures.append(signature or "")


# =============================================================================
# 可视化模块 (Visualization Module)
# =============================================================================

def plot_training_comparison(
    hist_constrained: TrainingHistory,
    hist_unconstrained: TrainingHistory,
    save_path: str = None,
):
    """
    绘制约束模式 vs 暴论模式的训练对比图
    
    包含 4 个子图：
    1. 损失曲线
    2. 几何占比曲线
    3. det(g) 变化
    4. 测试结果对比
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("警告: matplotlib 未安装，跳过可视化")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('度规涌现演化模型 - 约束模式 vs 暴论模式', fontsize=14, fontweight='bold')
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(hist_constrained.epochs, hist_constrained.loss, 'b-', label='约束模式', linewidth=2)
    ax1.plot(hist_unconstrained.epochs, hist_unconstrained.loss, 'r--', label='暴论模式', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 几何占比曲线
    ax2 = axes[0, 1]
    ax2.plot(hist_constrained.epochs, [r*100 for r in hist_constrained.geo_ratio], 
             'b-', label='约束模式', linewidth=2)
    ax2.plot(hist_unconstrained.epochs, [r*100 for r in hist_unconstrained.geo_ratio], 
             'r--', label='暴论模式', linewidth=2)
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('几何占比 (%)')
    ax2.set_title('几何加速度占比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # 3. det(g) 变化
    ax3 = axes[1, 0]
    ax3.plot(hist_constrained.epochs, hist_constrained.det_abs, 'b-', label='约束模式', linewidth=2)
    ax3.plot(hist_unconstrained.epochs, hist_unconstrained.det_abs, 'r--', label='暴论模式', linewidth=2)
    ax3.axhline(y=0.1, color='green', linestyle=':', alpha=0.5, label='目标值')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('|det(g)|')
    ax3.set_title('度规行列式绝对值')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 测试结果对比
    ax4 = axes[1, 1]
    rules = hist_constrained.test_rules
    x = range(len(rules))
    width = 0.35
    
    # 几何占比条形图
    bars1 = ax4.bar([i - width/2 for i in x], [r*100 for r in hist_constrained.test_geo_ratios], 
                    width, label='约束模式', color='steelblue', alpha=0.8)
    bars2 = ax4.bar([i + width/2 for i in x], [r*100 for r in hist_unconstrained.test_geo_ratios], 
                    width, label='暴论模式', color='indianred', alpha=0.8)
    
    ax4.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='几何主导阈值')
    ax4.set_xlabel('规则')
    ax4.set_ylabel('几何占比 (%)')
    ax4.set_title('测试时各规则的几何占比')
    ax4.set_xticks(x)
    ax4.set_xticklabels([r.upper() for r in rules])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.show()
    return fig


def plot_eigenvalue_analysis(
    hist_unconstrained: TrainingHistory,
    save_path: str = None,
):
    """
    绘制暴论模式的特征值分析图
    
    包含 2 个子图：
    1. 各规则的特征值分布
    2. 签名统计
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("警告: matplotlib 未安装，跳过可视化")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('暴论模式 - 度规特征值分析', fontsize=14, fontweight='bold')
    
    rules = hist_unconstrained.test_rules
    eigenvalues_list = hist_unconstrained.test_eigenvalues
    
    if not eigenvalues_list or not eigenvalues_list[0]:
        print("无特征值数据")
        return
    
    # 1. 特征值分布
    ax1 = axes[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (rule, eigs) in enumerate(zip(rules, eigenvalues_list)):
        if eigs:
            x_pos = [i + 0.15 * (j - len(eigs)/2) for j in range(len(eigs))]
            ax1.scatter(x_pos, eigs, c=[colors[i]]*len(eigs), s=100, 
                       label=f'{rule.upper()}', alpha=0.8, edgecolors='black')
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('规则')
    ax1.set_ylabel('特征值')
    ax1.set_title('各规则的度规特征值')
    ax1.set_xticks(range(len(rules)))
    ax1.set_xticklabels([r.upper() for r in rules])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 签名统计
    ax2 = axes[1]
    signatures = hist_unconstrained.test_signatures
    sig_counts = {}
    for s in signatures:
        if s:
            sig_counts[s] = sig_counts.get(s, 0) + 1
    
    if sig_counts:
        sigs = list(sig_counts.keys())
        counts = list(sig_counts.values())
        bars = ax2.bar(sigs, counts, color=['steelblue' if '0-' in s else 'indianred' for s in sigs],
                      alpha=0.8, edgecolor='black')
        ax2.set_xlabel('度规签名')
        ax2.set_ylabel('规则数量')
        ax2.set_title('度规签名分布')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.show()
    return fig


def plot_single_training(
    history: TrainingHistory,
    save_path: str = None,
):
    """
    绘制单次训练的详细图表
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("警告: matplotlib 未安装，跳过可视化")
        return
    
    mode_name = "暴论模式" if history.mode == "unconstrained" else "约束模式"
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'度规涌现演化模型 - {mode_name}', fontsize=14, fontweight='bold')
    
    color = 'indianred' if history.mode == "unconstrained" else 'steelblue'
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(history.epochs, history.loss, color=color, linewidth=2)
    ax1.fill_between(history.epochs, history.loss, alpha=0.3, color=color)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失')
    ax1.grid(True, alpha=0.3)
    
    # 2. 几何占比
    ax2 = axes[0, 1]
    ax2.plot(history.epochs, [r*100 for r in history.geo_ratio], color=color, linewidth=2)
    ax2.fill_between(history.epochs, [r*100 for r in history.geo_ratio], alpha=0.3, color=color)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('几何占比 (%)')
    ax2.set_title('几何加速度占比')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # 3. det(g)
    ax3 = axes[1, 0]
    ax3.plot(history.epochs, history.det_abs, color=color, linewidth=2)
    ax3.fill_between(history.epochs, history.det_abs, alpha=0.3, color=color)
    ax3.axhline(y=0.1, color='green', linestyle=':', alpha=0.7, label='目标值')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('|det(g)|')
    ax3.set_title('度规行列式')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 测试结果
    ax4 = axes[1, 1]
    rules = history.test_rules
    geo_ratios = [r*100 for r in history.test_geo_ratios]
    errors = history.test_errors
    
    x = range(len(rules))
    bars = ax4.bar(x, geo_ratios, color=color, alpha=0.8, edgecolor='black')
    ax4.axhline(y=30, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('规则')
    ax4.set_ylabel('几何占比 (%)')
    ax4.set_title('测试时各规则的几何占比')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{r.upper()}\nerr={e:.3f}" for r, e in zip(rules, errors)])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# 可视化辅助函数 (Visualization Helpers)
# =============================================================================

def eigenvalue_trajectory(
    g_sequence: Tensor,
) -> Tensor:
    """
    计算度规序列的特征值轨迹
    
    Args:
        g_sequence: [L, D, D] 度规序列
    
    Returns:
        [L, D] 特征值轨迹（每个位置的特征值）
    
    复杂度: O(L * D^3)
    """
    L, D, _ = g_sequence.shape
    eigenvalues = torch.stack([
        torch.linalg.eigvalsh(g_sequence[i])
        for i in range(L)
    ])
    return eigenvalues


def signature_color_encoding(signature: MetricSignature) -> Tuple[float, float, float]:
    """
    将签名编码为 RGB 颜色
    
    正定 -> 蓝色
    洛伦兹 -> 绿色
    退化 -> 红色
    其他不定号 -> 黄色
    """
    if signature.is_positive_definite:
        return (0.2, 0.4, 0.8)  # 蓝色
    elif signature.is_lorentzian:
        return (0.2, 0.8, 0.4)  # 绿色
    elif signature.is_degenerate:
        return (0.8, 0.2, 0.2)  # 红色
    else:
        return (0.8, 0.8, 0.2)  # 黄色（其他不定号）


def z_projection_2d(z: Tensor, method: str = 'pca') -> Tensor:
    """
    将高维 z 投影到 2D
    
    Args:
        z: [N, D] 或 [B, L, D] z 点集
        method: 'pca' 或 'first2'（取前两维）
    
    Returns:
        [N, 2] 或 [B, L, 2] 投影后的点
    
    复杂度: O(N * D^2) for PCA
    """
    original_shape = z.shape[:-1]
    D = z.shape[-1]
    z_flat = z.reshape(-1, D)
    
    if method == 'first2':
        result = z_flat[:, :2]
    elif method == 'pca':
        # 简单 PCA：中心化 -> SVD
        z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
        _, _, V = torch.linalg.svd(z_centered, full_matrices=False)
        result = z_centered @ V[:2].T
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return result.reshape(*original_shape, 2)


# =============================================================================
# 便捷接口
# =============================================================================

def analyze_metric(g: Tensor) -> Dict[str, Any]:
    """
    一站式度规分析
    
    Args:
        g: [B, L, D, D] 或 [D, D] 度规张量
    
    Returns:
        分析结果字典
    """
    if g.dim() == 2:
        sig = get_signature(g)
        return {
            'signature': sig.signature_str,
            'eigenvalues': sig.eigenvalues,
            'is_positive_definite': sig.is_positive_definite,
            'is_lorentzian': sig.is_lorentzian,
            'is_degenerate': sig.is_degenerate,
        }
    else:
        signatures = batch_signatures(g)
        stats = aggregate_signatures(signatures)
        return {
            'count': stats['count'],
            'most_common': stats['most_common_signature'],
            'pos_definite_ratio': stats['pos_definite_ratio'],
            'lorentzian_ratio': stats['lorentzian_ratio'],
            'distribution': stats['signature_distribution'],
        }


# =============================================================================
# 曲率计算 (Curvature Computation)
# =============================================================================

def compute_metric_derivative(g_func, z: Tensor, eps: float = 1e-4) -> Tensor:
    """
    计算度规对 z 的导数: ∂g_{ij}/∂z^k
    
    Args:
        g_func: z -> g 的函数
        z: [D] 单个 z 点
        eps: 有限差分步长
    
    Returns:
        [D, D, D] 张量，dg[i,j,k] = ∂g_{ij}/∂z^k
    """
    D = z.shape[0]
    dg = torch.zeros(D, D, D, device=z.device, dtype=z.dtype)
    
    for k in range(D):
        z_plus = z.clone()
        z_minus = z.clone()
        z_plus[k] += eps
        z_minus[k] -= eps
        
        g_plus = g_func(z_plus.unsqueeze(0).unsqueeze(0))[0, 0]  # [D, D]
        g_minus = g_func(z_minus.unsqueeze(0).unsqueeze(0))[0, 0]
        
        dg[:, :, k] = (g_plus - g_minus) / (2 * eps)
    
    return dg


def compute_christoffel(g: Tensor, dg: Tensor) -> Tensor:
    """
    计算 Christoffel 符号: Γ^i_{jk} = (1/2) g^{il} (∂_j g_{lk} + ∂_k g_{jl} - ∂_l g_{jk})
    
    Args:
        g: [D, D] 度规张量
        dg: [D, D, D] 度规导数，dg[i,j,k] = ∂g_{ij}/∂z^k
    
    Returns:
        [D, D, D] Christoffel 符号，Γ[i,j,k] = Γ^i_{jk}
    """
    D = g.shape[0]
    g_inv = torch.linalg.inv(g)  # [D, D]
    
    Gamma = torch.zeros(D, D, D, device=g.device, dtype=g.dtype)
    
    for i in range(D):
        for j in range(D):
            for k in range(D):
                for l in range(D):
                    # Γ^i_{jk} = (1/2) g^{il} (∂_j g_{lk} + ∂_k g_{jl} - ∂_l g_{jk})
                    Gamma[i, j, k] += 0.5 * g_inv[i, l] * (
                        dg[l, k, j] + dg[j, l, k] - dg[j, k, l]
                    )
    
    return Gamma


def compute_riemann(Gamma: Tensor, dGamma: Tensor) -> Tensor:
    """
    计算 Riemann 曲率张量: R^i_{jkl} = ∂_k Γ^i_{lj} - ∂_l Γ^i_{kj} + Γ^i_{km}Γ^m_{lj} - Γ^i_{lm}Γ^m_{kj}
    
    Args:
        Gamma: [D, D, D] Christoffel 符号
        dGamma: [D, D, D, D] Christoffel 导数，dGamma[i,j,k,l] = ∂_l Γ^i_{jk}
    
    Returns:
        [D, D, D, D] Riemann 张量，R[i,j,k,l] = R^i_{jkl}
    """
    D = Gamma.shape[0]
    R = torch.zeros(D, D, D, D, device=Gamma.device, dtype=Gamma.dtype)
    
    for i in range(D):
        for j in range(D):
            for k in range(D):
                for l in range(D):
                    # R^i_{jkl} = ∂_k Γ^i_{lj} - ∂_l Γ^i_{kj}
                    R[i, j, k, l] = dGamma[i, l, j, k] - dGamma[i, k, j, l]
                    
                    # + Γ^i_{km}Γ^m_{lj} - Γ^i_{lm}Γ^m_{kj}
                    for m in range(D):
                        R[i, j, k, l] += Gamma[i, k, m] * Gamma[m, l, j]
                        R[i, j, k, l] -= Gamma[i, l, m] * Gamma[m, k, j]
    
    return R


def compute_ricci(R: Tensor) -> Tensor:
    """
    计算 Ricci 张量: R_{ij} = R^k_{ikj}
    
    Args:
        R: [D, D, D, D] Riemann 张量
    
    Returns:
        [D, D] Ricci 张量
    """
    D = R.shape[0]
    Ric = torch.zeros(D, D, device=R.device, dtype=R.dtype)
    
    for i in range(D):
        for j in range(D):
            for k in range(D):
                Ric[i, j] += R[k, i, k, j]
    
    return Ric


def compute_scalar_curvature(g: Tensor, Ric: Tensor) -> Tensor:
    """
    计算标量曲率: R = g^{ij} R_{ij}
    
    Args:
        g: [D, D] 度规
        Ric: [D, D] Ricci 张量
    
    Returns:
        标量曲率 (scalar)
    """
    g_inv = torch.linalg.inv(g)
    return torch.einsum('ij,ij->', g_inv, Ric)


def full_curvature_analysis(g_func, z: Tensor, eps: float = 1e-4) -> Dict[str, Any]:
    """
    完整的曲率分析
    
    Args:
        g_func: z -> g 的函数 (MetricEncoder)
        z: [D] 单个 z 点
        eps: 有限差分步长
    
    Returns:
        包含各种曲率信息的字典
    """
    D = z.shape[0]
    
    # 1. 计算度规及其导数
    g = g_func(z.unsqueeze(0).unsqueeze(0))[0, 0]  # [D, D]
    dg = compute_metric_derivative(g_func, z, eps)
    
    # 2. 计算 Christoffel 符号
    Gamma = compute_christoffel(g, dg)
    
    # 3. 计算 Christoffel 导数 (用于 Riemann)
    dGamma = torch.zeros(D, D, D, D, device=z.device, dtype=z.dtype)
    for l in range(D):
        z_plus = z.clone()
        z_minus = z.clone()
        z_plus[l] += eps
        z_minus[l] -= eps
        
        g_plus = g_func(z_plus.unsqueeze(0).unsqueeze(0))[0, 0]
        g_minus = g_func(z_minus.unsqueeze(0).unsqueeze(0))[0, 0]
        dg_plus = compute_metric_derivative(g_func, z_plus, eps)
        dg_minus = compute_metric_derivative(g_func, z_minus, eps)
        
        Gamma_plus = compute_christoffel(g_plus, dg_plus)
        Gamma_minus = compute_christoffel(g_minus, dg_minus)
        
        dGamma[:, :, :, l] = (Gamma_plus - Gamma_minus) / (2 * eps)
    
    # 4. 计算 Riemann 张量
    R = compute_riemann(Gamma, dGamma)
    
    # 5. 计算 Ricci 张量
    Ric = compute_ricci(R)
    
    # 6. 计算标量曲率
    scalar = compute_scalar_curvature(g, Ric)
    
    # 7. 计算曲率统计
    riemann_norm = torch.norm(R).item()
    ricci_norm = torch.norm(Ric).item()
    
    return {
        'scalar_curvature': scalar.item(),
        'riemann_norm': riemann_norm,
        'ricci_norm': ricci_norm,
        'ricci_tensor': Ric.detach(),
        'metric': g.detach(),
        'signature': get_signature(g),
    }


def trajectory_curvature_analysis(
    g_func, 
    z_trajectory: Tensor,
    sample_points: int = 5,
) -> List[Dict[str, Any]]:
    """
    沿轨迹计算曲率
    
    Args:
        g_func: z -> g 的函数
        z_trajectory: [L, D] z 轨迹
        sample_points: 采样点数
    
    Returns:
        每个采样点的曲率分析结果列表
    """
    L = z_trajectory.shape[0]
    indices = torch.linspace(0, L-1, sample_points).long()
    
    results = []
    for idx in indices:
        z = z_trajectory[idx]
        try:
            result = full_curvature_analysis(g_func, z)
            result['trajectory_index'] = idx.item()
            results.append(result)
        except Exception as e:
            results.append({
                'trajectory_index': idx.item(),
                'error': str(e),
            })
    
    return results


# =============================================================================
# 深度分析可视化 (Deep Analysis Visualization)
# =============================================================================

def plot_trajectory_analysis(
    z_constrained: Tensor,
    z_unconstrained: Tensor,
    g_constrained: Tensor,
    g_unconstrained: Tensor,
    rule: str = 'alt',
    save_path: str = None,
):
    """
    绘制两种模式下的 z 轨迹和特征值对比
    
    Args:
        z_constrained: [L, D] 约束模式的 z 轨迹
        z_unconstrained: [L, D] 暴论模式的 z 轨迹
        g_constrained: [L, D, D] 约束模式的度规
        g_unconstrained: [L, D, D] 暴论模式的度规
        rule: 规则名称
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("警告: matplotlib 未安装")
        return
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{rule.upper()} 规则 - z 轨迹与度规分析', fontsize=14, fontweight='bold')
    
    L = z_constrained.shape[0]
    D = z_constrained.shape[1]
    
    # 1. z 轨迹 - PCA 投影到 2D
    ax1 = fig.add_subplot(2, 3, 1)
    z_c_2d = z_projection_2d(z_constrained, method='pca').numpy()
    z_u_2d = z_projection_2d(z_unconstrained, method='pca').numpy()
    
    ax1.plot(z_c_2d[:, 0], z_c_2d[:, 1], 'b-o', label='约束模式', markersize=6, alpha=0.8)
    ax1.plot(z_u_2d[:, 0], z_u_2d[:, 1], 'r--s', label='暴论模式', markersize=6, alpha=0.8)
    ax1.scatter(z_c_2d[0, 0], z_c_2d[0, 1], c='blue', s=100, marker='*', zorder=5)
    ax1.scatter(z_u_2d[0, 0], z_u_2d[0, 1], c='red', s=100, marker='*', zorder=5)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('z 轨迹 (PCA 投影)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. z 轨迹 - 3D (取前 3 维)
    if D >= 3:
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.plot(z_constrained[:, 0].numpy(), z_constrained[:, 1].numpy(), 
                z_constrained[:, 2].numpy(), 'b-o', label='约束', markersize=4)
        ax2.plot(z_unconstrained[:, 0].numpy(), z_unconstrained[:, 1].numpy(),
                z_unconstrained[:, 2].numpy(), 'r--s', label='暴论', markersize=4)
        ax2.set_xlabel('z₀')
        ax2.set_ylabel('z₁')
        ax2.set_zlabel('z₂')
        ax2.set_title('z 轨迹 (3D)')
        ax2.legend()
    
    # 3. 约束模式特征值沿轨迹变化
    ax3 = fig.add_subplot(2, 3, 3)
    eig_c = eigenvalue_trajectory(g_constrained).numpy()  # [L, D]
    for d in range(D):
        ax3.plot(range(L), eig_c[:, d], 'b-', alpha=0.7, linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('轨迹位置')
    ax3.set_ylabel('特征值')
    ax3.set_title('约束模式 - g(z) 特征值')
    ax3.grid(True, alpha=0.3)
    
    # 4. 暴论模式特征值沿轨迹变化
    ax4 = fig.add_subplot(2, 3, 4)
    eig_u = eigenvalue_trajectory(g_unconstrained).numpy()  # [L, D]
    for d in range(D):
        color = 'r' if eig_u[:, d].mean() < 0 else 'g'
        ax4.plot(range(L), eig_u[:, d], color=color, alpha=0.7, linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('轨迹位置')
    ax4.set_ylabel('特征值')
    ax4.set_title('暴论模式 - g(z) 特征值 (红=负, 绿=正)')
    ax4.grid(True, alpha=0.3)
    
    # 5. det(g) 沿轨迹变化
    ax5 = fig.add_subplot(2, 3, 5)
    det_c = torch.linalg.det(g_constrained).numpy()
    det_u = torch.linalg.det(g_unconstrained).numpy()
    ax5.plot(range(L), det_c, 'b-o', label='约束模式', markersize=4)
    ax5.plot(range(L), det_u, 'r--s', label='暴论模式', markersize=4)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('轨迹位置')
    ax5.set_ylabel('det(g)')
    ax5.set_title('度规行列式')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 签名分布
    ax6 = fig.add_subplot(2, 3, 6)
    sigs_c = [get_signature(g_constrained[i]).signature_str for i in range(L)]
    sigs_u = [get_signature(g_unconstrained[i]).signature_str for i in range(L)]
    
    # 统计签名
    from collections import Counter
    sig_c_dist = Counter(sigs_c)
    sig_u_dist = Counter(sigs_u)
    
    all_sigs = list(set(sigs_c + sigs_u))
    x = range(len(all_sigs))
    width = 0.35
    
    counts_c = [sig_c_dist.get(s, 0) for s in all_sigs]
    counts_u = [sig_u_dist.get(s, 0) for s in all_sigs]
    
    ax6.bar([i - width/2 for i in x], counts_c, width, label='约束模式', color='steelblue')
    ax6.bar([i + width/2 for i in x], counts_u, width, label='暴论模式', color='indianred')
    ax6.set_xlabel('度规签名')
    ax6.set_ylabel('出现次数')
    ax6.set_title('沿轨迹的签名分布')
    ax6.set_xticks(x)
    ax6.set_xticklabels(all_sigs)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.show()
    return fig


def plot_curvature_comparison(
    curvature_results_constrained: List[Dict],
    curvature_results_unconstrained: List[Dict],
    rule: str = 'alt',
    save_path: str = None,
):
    """
    绘制曲率对比图
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("警告: matplotlib 未安装")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{rule.upper()} 规则 - 曲率分析', fontsize=14, fontweight='bold')
    
    # 提取数据
    def extract_values(results, key):
        return [r.get(key, 0) for r in results if 'error' not in r]
    
    indices_c = [r['trajectory_index'] for r in curvature_results_constrained if 'error' not in r]
    indices_u = [r['trajectory_index'] for r in curvature_results_unconstrained if 'error' not in r]
    
    # 1. 标量曲率
    ax1 = axes[0]
    scalar_c = extract_values(curvature_results_constrained, 'scalar_curvature')
    scalar_u = extract_values(curvature_results_unconstrained, 'scalar_curvature')
    
    ax1.plot(indices_c, scalar_c, 'b-o', label='约束模式', markersize=8)
    ax1.plot(indices_u, scalar_u, 'r--s', label='暴论模式', markersize=8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('轨迹位置')
    ax1.set_ylabel('R (标量曲率)')
    ax1.set_title('标量曲率 Scalar Curvature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Riemann 范数
    ax2 = axes[1]
    riemann_c = extract_values(curvature_results_constrained, 'riemann_norm')
    riemann_u = extract_values(curvature_results_unconstrained, 'riemann_norm')
    
    ax2.plot(indices_c, riemann_c, 'b-o', label='约束模式', markersize=8)
    ax2.plot(indices_u, riemann_u, 'r--s', label='暴论模式', markersize=8)
    ax2.set_xlabel('轨迹位置')
    ax2.set_ylabel('||R|| (Riemann 范数)')
    ax2.set_title('Riemann 张量范数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Ricci 范数
    ax3 = axes[2]
    ricci_c = extract_values(curvature_results_constrained, 'ricci_norm')
    ricci_u = extract_values(curvature_results_unconstrained, 'ricci_norm')
    
    ax3.plot(indices_c, ricci_c, 'b-o', label='约束模式', markersize=8)
    ax3.plot(indices_u, ricci_u, 'r--s', label='暴论模式', markersize=8)
    ax3.set_xlabel('轨迹位置')
    ax3.set_ylabel('||Ric|| (Ricci 范数)')
    ax3.set_title('Ricci 张量范数')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# 洛伦兹度规引导 (Lorentzian Guidance)
# =============================================================================

def lorentzian_regularization(g: Tensor, eps: float = 1e-6) -> Tensor:
    """
    软约束：鼓励恰好一个负特征值
    
    Loss = (n_negative - 1)^2 + (n_zero)^2
    
    Args:
        g: [B, L, D, D] 或 [D, D] 度规张量
    
    Returns:
        洛伦兹正则化损失
    """
    original_shape = g.shape[:-2]
    D = g.shape[-1]
    g_flat = g.reshape(-1, D, D)
    
    eigenvalues = torch.linalg.eigvalsh(g_flat)  # [N, D]
    
    # 计算负特征值数量（软计数）
    neg_count = torch.sigmoid(-eigenvalues / eps).sum(dim=-1)  # 软 sigmoid 计数
    zero_count = torch.exp(-(eigenvalues / eps) ** 2).sum(dim=-1)  # 零附近的计数
    
    # 目标：恰好 1 个负，0 个零
    loss = (neg_count - 1) ** 2 + zero_count ** 2
    
    return loss.mean()


def eigenvalue_guidance_loss(
    g: Tensor, 
    target_neg: int = 1,
    target_pos: int = None,
    temperature: float = 0.1,
) -> Tensor:
    """
    特征值引导损失 - 更精细的控制
    
    Args:
        g: [B, L, D, D] 度规张量
        target_neg: 目标负特征值数量
        target_pos: 目标正特征值数量 (None = D - target_neg)
        temperature: 软化温度
    
    Returns:
        引导损失
    """
    D = g.shape[-1]
    g_flat = g.reshape(-1, D, D)
    
    eigenvalues = torch.linalg.eigvalsh(g_flat)  # [N, D]
    sorted_eig, _ = eigenvalues.sort(dim=-1)
    
    if target_pos is None:
        target_pos = D - target_neg
    
    # 前 target_neg 个特征值应该为负
    neg_part = sorted_eig[:, :target_neg]
    pos_part = sorted_eig[:, target_neg:]
    
    # 鼓励负部分 < 0，正部分 > 0
    neg_loss = torch.relu(neg_part + temperature).mean()  # 应该 < -temp
    pos_loss = torch.relu(-pos_part + temperature).mean()  # 应该 > temp
    
    return neg_loss + pos_loss


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("度规分析模块测试")
    print("=" * 50)
    
    # 测试正定矩阵
    g_pos = torch.eye(4) * 0.5
    sig_pos = get_signature(g_pos)
    print(f"\n正定矩阵签名: {sig_pos.signature_str}")
    print(f"  is_positive_definite: {sig_pos.is_positive_definite}")
    print(f"  eigenvalues: {sig_pos.eigenvalues}")
    
    # 测试洛伦兹签名
    g_lor = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))
    sig_lor = get_signature(g_lor)
    print(f"\n洛伦兹矩阵签名: {sig_lor.signature_str}")
    print(f"  is_lorentzian: {sig_lor.is_lorentzian}")
    print(f"  eigenvalues: {sig_lor.eigenvalues}")
    
    # 测试不定号
    g_indef = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0]))
    sig_indef = get_signature(g_indef)
    print(f"\n不定号矩阵签名: {sig_indef.signature_str}")
    print(f"  eigenvalues: {sig_indef.eigenvalues}")
    
    # 测试批量分析
    g_batch = torch.stack([g_pos, g_lor, g_indef])
    stats = aggregate_signatures(batch_signatures(g_batch))
    print(f"\n批量签名统计:")
    print(f"  distribution: {stats['signature_distribution']}")
    print(f"  lorentzian_ratio: {stats['lorentzian_ratio']:.1%}")
    
    print("\n" + "=" * 50)
    print("测试通过")
    print("=" * 50)
