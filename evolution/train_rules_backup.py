"""演化模型 - 规则学习训练 (Pure FP, Algorithm-Aware, Stable Training)

度规涌现架构：
  数据 → z → 度规 g(z) → 联络 Γ (向量化涌现) → 测地线加速度 + 外力 → d²x

训练稳定性改进：
  - 动态外力惩罚权重（warmup 策略）
  - 混合 det(g) 正则化
  - 多次运行一致性训练
  - 梯度监控和异常检测
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from .model import (
    MetricEvolutionModel, 
    dynamic_force_weight,
    gradient_monitor,
)


# 设置随机种子以提高可重复性
def set_seed(seed: int = 42):
    """设置随机种子以提高训练一致性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_sequence(rule: str, length: int = 10, device: str = 'cpu') -> torch.Tensor:
    """生成不同规则的序列"""
    if rule == 'add':
        start = torch.rand(1).item() * 0.3
        d = torch.rand(1).item() * 0.1 + 0.02
        seq = [start + i * d for i in range(length)]
    
    elif rule == 'mul':
        start = torch.rand(1).item() * 0.1 + 0.05
        k = torch.rand(1).item() * 0.3 + 1.1
        seq = [start * (k ** i) for i in range(length)]
    
    elif rule == 'alt':
        base = torch.rand(1).item() * 0.3 + 0.3
        amp = torch.rand(1).item() * 0.1 + 0.02
        seq = [base + amp * ((-1) ** i) for i in range(length)]
    
    elif rule == 'acc':
        start = torch.rand(1).item() * 0.1
        v = torch.rand(1).item() * 0.03 + 0.01
        a = torch.rand(1).item() * 0.02 + 0.005
        seq = []
        x = start
        for i in range(length):
            seq.append(x)
            x = x + v
            v = v + a
    
    else:
        raise ValueError(f"Unknown rule: {rule}")
    
    seq = [max(0, min(1, x)) for x in seq]
    return torch.tensor(seq, device=device).unsqueeze(0)


def train(
    epochs: int = 100,
    z_dim: int = 4,
    samples_per_rule: int = 20,
    lr: float = 3e-4,
    min_force_weight: float = 0.1,   # 外力惩罚最小权重
    max_force_weight: float = 0.8,   # 外力惩罚最大权重
    warmup_ratio: float = 0.3,       # warmup 比例
    det_target: float = 0.1,         # det(g) 目标值
    grad_clip: float = 0.5,
    seed: int = 42,
):
    """训练模型 - 稳定训练策略，动态外力权重"""
    # 设置随机种子
    set_seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = MetricEvolutionModel(
        z_dim=z_dim,
        input_dim=1,
    ).to(device)
    
    # 使用 AdamW 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    
    # 余弦退火学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    
    rules = ['add', 'mul', 'alt', 'acc']
    
    print("=" * 60)
    print("度规涌现演化模型 - 稳定训练 (Dynamic Force Weight)")
    print("=" * 60)
    print(f"z维度: {z_dim}, 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"架构: x → z → g(z) → Γ(向量化) → a_geo + F_ext → d²x")
    print(f"学习率: {lr}, 梯度裁剪: {grad_clip}")
    print(f"外力权重: {min_force_weight} → {max_force_weight} (warmup={warmup_ratio})")
    print(f"det(g)目标: {det_target}, 随机种子: {seed}")
    print("=" * 60)
    
    best_loss = float('inf')
    best_geo_ratio = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # 动态计算外力权重
        current_force_weight = dynamic_force_weight(
            epoch, epochs,
            min_weight=min_force_weight,
            max_weight=max_force_weight,
            warmup_ratio=warmup_ratio,
        )
        
        total_loss = 0
        total_nll = 0
        total_force = 0
        total_det = 0
        total_geo_norm = 0
        total_force_norm = 0
        total_det_abs = 0
        
        for rule in rules:
            for _ in range(samples_per_rule):
                # 生成序列
                seq = generate_sequence(rule, length=10, device=device)
                
                # 用前 n-1 个点预测最后一个点
                input_seq = seq[:, :-1]
                target = seq[:, -1]
                
                # 计算损失，使用动态外力权重
                loss_dict = model.compute_loss(
                    input_seq, target,
                    force_weight=current_force_weight,
                    det_target=det_target,
                )
                loss = loss_dict['loss']
                
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
                
                total_loss += loss.item()
                total_nll += loss_dict['nll_loss'].item()
                total_force += loss_dict['force_reg'].item()
                total_det += loss_dict['det_reg'].item()
                total_geo_norm += loss_dict['a_geodesic'].norm().item()
                total_force_norm += loss_dict['F_external'].norm().item()
                total_det_abs += loss_dict['det_abs_mean'].item()
        
        # 更新学习率
        scheduler.step()
        
        n = len(rules) * samples_per_rule
        avg_loss = total_loss / n
        avg_geo = total_geo_norm / n
        avg_force = total_force_norm / n
        geo_ratio = avg_geo / (avg_geo + avg_force + 1e-8)
        
        # 记录最佳
        if avg_loss < best_loss:
            best_loss = avg_loss
        if geo_ratio > best_geo_ratio:
            best_geo_ratio = geo_ratio
        
        if epoch % 10 == 0 or epoch == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}, nll={total_nll/n:.4f}, "
                  f"force_w={current_force_weight:.2f}")
            print(f"           |a_geo|={avg_geo:.4f}, |F_ext|={avg_force:.4f}, "
                  f"几何占比={geo_ratio:.1%}")
            print(f"           det_abs={total_det_abs/n:.4f}, lr={current_lr:.2e}")
    
    # 测试
    print("\n" + "=" * 60)
    print("规则学习测试")
    print("=" * 60)
    
    model.eval()
    
    test_cases = [
        ('add', '加法 x_{n+1} = x_n + d'),
        ('mul', '乘法 x_{n+1} = x_n * k'),
        ('alt', '交替振荡'),
        ('acc', '加速 x_{n+1} = x_n + v_n + a'),
    ]
    
    correct = 0
    total = 0
    geo_dominant_count = 0
    
    with torch.no_grad():
        for rule, desc in test_cases:
            print(f"\n【{rule.upper()}】{desc}")
            
            seq = generate_sequence(rule, length=10, device=device)
            
            input_seq = seq[:, :-1]
            target = seq[:, -1]
            
            out = model.forward(input_seq)
            
            # 几何信息
            a_geo_norm = out['a_geodesic'].norm().item()
            f_ext_norm = out['F_external'].norm().item()
            geo_ratio = a_geo_norm / (a_geo_norm + f_ext_norm + 1e-8)
            
            if geo_ratio > 0.3:  # 几何占比超过 30% 认为几何主导
                geo_dominant_count += 1
            
            print(f"  |a_geodesic|={a_geo_norm:.4f}, |F_external|={f_ext_norm:.4f}, "
                  f"几何占比={geo_ratio:.1%}")
            
            # 分布参数
            mu = out['mu'].item()
            sigma = out['sigma'].item()
            print(f"  分布: μ={mu:.6f}, σ={sigma:.6f} "
                  f"(95%CI: [{mu-1.96*sigma:.4f}, {mu+1.96*sigma:.4f}])")
            print(f"  x_new={out['x_new'].item():.4f}, target={target.item():.4f}")
            
            # 真实 d2x
            d2x_true = target - out['x_last'] - out['dx_last']
            print(f"  d2x_true={d2x_true.item():.6f}")
            
            # 置信区间检查
            in_ci = abs(d2x_true.item() - mu) < 1.96 * sigma
            
            error = abs(out['x_new'].item() - target.item())
            status = "✓" if error < 0.05 else "✗"
            ci_status = "✓" if in_ci else "✗"
            print(f"  误差={error:.4f} {status}, 真值在CI内: {ci_status}")
            
            if error < 0.05:
                correct += 1
            total += 1
    
    print("\n" + "=" * 60)
    print(f"单步预测准确率: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"几何主导规则数: {geo_dominant_count}/{total}")
    print(f"最佳训练损失: {best_loss:.6f}")
    print(f"最佳几何占比: {best_geo_ratio:.1%}")
    print("=" * 60)
    
    return model, {
        'accuracy': correct / total,
        'geo_dominant': geo_dominant_count / total,
        'best_loss': best_loss,
        'best_geo_ratio': best_geo_ratio,
    }


def train_multiple_runs(
    num_runs: int = 3,
    **train_kwargs,
):
    """多次运行训练，检查稳定性"""
    print("\n" + "#" * 60)
    print(f"多次运行稳定性测试 ({num_runs} 次)")
    print("#" * 60)
    
    results = []
    
    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"运行 {run + 1}/{num_runs}")
        print("=" * 60)
        
        _, metrics = train(seed=42 + run, **train_kwargs)
        results.append(metrics)
    
    # 统计结果
    print("\n" + "#" * 60)
    print("多次运行统计结果")
    print("#" * 60)
    
    accuracies = [r['accuracy'] for r in results]
    geo_ratios = [r['best_geo_ratio'] for r in results]
    geo_dominants = [r['geo_dominant'] for r in results]
    
    import statistics
    
    print(f"准确率: {statistics.mean(accuracies):.1%} ± {statistics.stdev(accuracies) if len(accuracies) > 1 else 0:.1%}")
    print(f"几何占比: {statistics.mean(geo_ratios):.1%} ± {statistics.stdev(geo_ratios) if len(geo_ratios) > 1 else 0:.1%}")
    print(f"几何主导: {statistics.mean(geo_dominants):.1%} ± {statistics.stdev(geo_dominants) if len(geo_dominants) > 1 else 0:.1%}")
    
    return results


if __name__ == "__main__":
    # 单次训练
    train(
        epochs=50,
        z_dim=4,
        samples_per_rule=20,
        lr=3e-4,
        min_force_weight=0.1,
        max_force_weight=0.8,
        warmup_ratio=0.3,
        det_target=0.1,
        grad_clip=0.5,
        seed=42,
    )
