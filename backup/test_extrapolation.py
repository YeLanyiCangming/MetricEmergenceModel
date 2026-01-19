"""斐波那契外推测试 - 完整优化版

测试度规涌现模型的外推能力：给定前 N 个数，预测后 M 个数

优化措施：
    1. 激进 fw_max=15.0，强制几何主导
    2. Scheduled Sampling ss_max=0.8，模拟外推过程
    3. 500 epochs 充分训练
    4. float64 高精度
    5. CosineAnnealingWarmRestarts 学习率调度
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def set_seed(seed: int = 42):
    """固定随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def generate_fibonacci(length: int, start: tuple = (1, 1), normalize: bool = True) -> list:
    """生成斐波那契数列"""
    seq = [start[0], start[1]]
    for _ in range(length - 2):
        seq.append(seq[-1] + seq[-2])
    if normalize:
        max_val = max(seq)
        seq = [x / max_val for x in seq]
    return seq


def dynamic_fw(epoch: int, total: int, fw_min: float = 1.0, fw_max: float = 15.0) -> float:
    """动态外力惩罚权重 - 三阶段调度"""
    # 阶段1: 前20% 平台期
    if epoch < total * 0.2:
        return fw_min
    # 阶段2: 20%-60% 线性增长
    elif epoch < total * 0.6:
        progress = (epoch - total * 0.2) / (total * 0.4)
        return fw_min + (fw_max - fw_min) * progress
    # 阶段3: 60%-100% 保持最大
    else:
        return fw_max


def dynamic_ss(epoch: int, total: int, ss_max: float = 0.8) -> float:
    """动态 Scheduled Sampling 概率"""
    # 前30% 不使用 SS，先学好基础
    if epoch < total * 0.3:
        return 0.0
    # 后70% 逐渐增加到 ss_max
    progress = (epoch - total * 0.3) / (total * 0.7)
    return ss_max * progress


def train_and_test(
    n_given: int = 5,
    n_predict: int = 10,
    epochs: int = 500,
    z_dim: int = 8,
    seed: int = 42,
    fw_max: float = 15.0,
    ss_max: float = 0.8,
    use_float64: bool = True,
    verbose: bool = True,
):
    """
    训练并测试外推能力（完整优化版）
    
    Args:
        n_given: 给定的点数
        n_predict: 预测的点数
        epochs: 训练轮次
        z_dim: 隐空间维度
        seed: 随机种子
        fw_max: 外力惩罚最大权重
        ss_max: Scheduled Sampling 最大概率
        use_float64: 是否使用 float64 高精度
        verbose: 是否打印详情
    
    Returns:
        dict: 包含误差和几何占比的结果
    """
    from .model import MetricEvolutionModel
    
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64 if use_float64 else torch.float32
    total_len = n_given + n_predict + 2
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"完整优化版训练")
        print(f"  配置: z_dim={z_dim}, epochs={epochs}, seed={seed}")
        print(f"  优化: fw_max={fw_max}, ss_max={ss_max}, dtype={dtype}")
        print(f"  外推: 前{n_given} → 后{n_predict}")
        print('='*60)
    
    # 创建模型（高精度）
    model = MetricEvolutionModel(
        z_dim=z_dim, input_dim=1,
        hidden_mult=6, unconstrained=False,
    ).to(device).to(dtype)
    
    # AdamW + CosineAnnealingWarmRestarts
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-5)
    
    # ========== 训练 ==========
    for epoch in range(1, epochs + 1):
        model.train()
        fw = dynamic_fw(epoch, epochs, 1.0, fw_max)
        ss = dynamic_ss(epoch, epochs, ss_max)
        
        total_loss = 0
        total_geo = 0
        total_ext = 0
        n_batch = 30
        
        for _ in range(n_batch):
            # 生成随机起点的斐波那契
            a = torch.randint(1, 5, (1,)).item()
            b = torch.randint(1, 5, (1,)).item()
            seq = generate_fibonacci(total_len, start=(a, b), normalize=True)
            seq = torch.tensor(seq, device=device, dtype=dtype).unsqueeze(0)
            
            # 随机切片 (4 到 n_given+2 个点)
            input_len = torch.randint(4, n_given + 3, (1,)).item()
            start_idx = torch.randint(0, max(1, total_len - input_len - 1), (1,)).item()
            end_idx = start_idx + input_len + 1
            
            batch = seq[:, start_idx:end_idx].clone()
            
            # Scheduled Sampling: 用模型预测替代部分真实值
            if ss > 0 and batch.shape[1] > 3 and torch.rand(1).item() < ss:
                model.eval()
                with torch.no_grad():
                    mid = batch.shape[1] // 2
                    out = model.forward(batch[:, :mid])
                    batch[0, mid] = out['x_new'].squeeze()
                model.train()
            
            loss_dict = model.compute_loss(
                batch[:, :-1], batch[:, -1],
                force_weight=fw,
                min_abs_eig=0.2,
                min_eig_weight=3.0,
            )
            
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss_dict['loss'].item()
            total_geo += loss_dict['a_geodesic'].norm().item()
            total_ext += loss_dict['F_external'].norm().item()
        
        scheduler.step()
        geo_ratio = total_geo / (total_geo + total_ext + 1e-8)
        
        if verbose and (epoch % 100 == 0 or epoch == 1):
            print(f"E{epoch:3d}: loss={total_loss/n_batch:.4f}, "
                  f"geo={geo_ratio:.1%}, fw={fw:.1f}, ss={ss:.2f}")
    
    # ========== 测试 ==========
    model.eval()
    
    # 生成测试序列
    test_seq = generate_fibonacci(n_given + n_predict, normalize=True)
    given = torch.tensor(test_seq[:n_given], device=device, dtype=dtype).unsqueeze(0)
    true_values = test_seq[n_given:]
    
    predictions = []
    errors = []
    geo_ratios = []
    
    current = given.clone()
    
    with torch.no_grad():
        for step in range(n_predict):
            out = model.forward(current)
            pred = out['x_new'].item()
            
            true_val = true_values[step]
            err = abs(pred - true_val)
            
            geo = out['a_geodesic'].norm().item()
            ext = out['F_external'].norm().item()
            geo_ratio = geo / (geo + ext + 1e-8)
            
            predictions.append(pred)
            errors.append(err)
            geo_ratios.append(geo_ratio)
            
            if verbose:
                print(f"  Step {step+1:2d}: pred={pred:.4f}, true={true_val:.4f}, "
                      f"err={err:.4f}, geo={geo_ratio:.1%}")
            
            # 加入预测值继续外推
            new_val = torch.tensor([[pred]], device=device, dtype=dtype)
            current = torch.cat([current, new_val], dim=1)
    
    avg_err = sum(errors) / len(errors)
    avg_geo = sum(geo_ratios) / len(geo_ratios)
    
    if verbose:
        print(f"\n统计: 平均误差={avg_err:.4f}, 平均geo={avg_geo:.1%}")
    
    return {
        'predictions': predictions,
        'true_values': true_values,
        'errors': errors,
        'avg_error': avg_err,
        'avg_geo': avg_geo,
        'model': model,
    }


def multi_run(
    n_runs: int = 5,
    n_given: int = 5,
    n_predict: int = 10,
    epochs: int = 300,
    z_dim: int = 8,
):
    """
    多次运行取最佳结果
    
    Args:
        n_runs: 运行次数
        其他参数同 train_and_test
    
    Returns:
        最佳结果
    """
    print(f"\n{'#'*60}")
    print(f"多次运行测试 (n_runs={n_runs})")
    print(f"配置: n_given={n_given}, n_predict={n_predict}, epochs={epochs}")
    print('#'*60)
    
    results = []
    
    for run in range(n_runs):
        print(f"\n>>> Run {run+1}/{n_runs}")
        result = train_and_test(
            n_given=n_given,
            n_predict=n_predict,
            epochs=epochs,
            z_dim=z_dim,
            seed=42 + run,
            verbose=True,
        )
        results.append(result)
        print(f"    结果: avg_err={result['avg_error']:.4f}, avg_geo={result['avg_geo']:.1%}")
    
    # 找最佳
    best = min(results, key=lambda x: x['avg_error'])
    best_idx = results.index(best)
    
    print(f"\n{'='*60}")
    print(f"最佳结果: Run {best_idx+1}")
    print(f"  平均误差: {best['avg_error']:.4f}")
    print(f"  平均geo:  {best['avg_geo']:.1%}")
    print(f"  各步误差: {[f'{e:.4f}' for e in best['errors']]}")
    print('='*60)
    
    # 统计
    avg_errors = [r['avg_error'] for r in results]
    print(f"\n所有运行平均误差: {sum(avg_errors)/len(avg_errors):.4f}")
    print(f"误差范围: [{min(avg_errors):.4f}, {max(avg_errors):.4f}]")
    
    return best


if __name__ == '__main__':
    # 单次运行
    # result = train_and_test(n_given=5, n_predict=10, epochs=300, seed=42)
    
    # 多次运行取最佳
    best = multi_run(n_runs=3, n_given=5, n_predict=10, epochs=300)
