"""泛化能力测试 - 斐波那契数列

测试度规涌现模型在未见过的规律上的表现
"""

import torch
from .model import MetricEvolutionModel, dynamic_force_weight
from .train_rules import train, generate_sequence, set_seed
from .metric_analysis import get_signature


def generate_fibonacci(length: int = 10, normalize: bool = True, device: str = 'cpu', start: tuple = (1, 1)) -> torch.Tensor:
    """生成斐波那契数列
    
    Args:
        length: 序列长度
        normalize: 是否归一化到 [0, 1]
        device: 设备
        start: 起始值 (a, b)
    
    Returns:
        [1, length] 的张量
    """
    a, b = start
    fib = [a, b]
    for i in range(2, length):
        fib.append(fib[-1] + fib[-2])
    
    seq = torch.tensor(fib[:length], dtype=torch.float32, device=device)
    
    if normalize:
        seq = seq / seq.max()
    
    return seq.unsqueeze(0)  # [1, length]


def generate_fibonacci_batch(batch_size: int, length: int = 10, device: str = 'cpu') -> torch.Tensor:
    """生成一批不同起点的斐波那契数列"""
    seqs = []
    for _ in range(batch_size):
        # 随机起始值
        a = torch.randint(1, 5, (1,)).item()
        b = torch.randint(1, 5, (1,)).item()
        seq = generate_fibonacci(length, normalize=True, device=device, start=(a, b))
        seqs.append(seq)
    return torch.cat(seqs, dim=0)  # [B, length]


def test_on_fibonacci(model, device: str = 'cpu', verbose: bool = True):
    """在斐波那契数列上测试模型
    
    Returns:
        dict: 测试结果
    """
    model.eval()
    
    # 确保模型在正确的设备上
    model = model.to(device)
    
    # 生成斐波那契序列
    fib_seq = generate_fibonacci(length=10, normalize=True, device=device)
    
    if verbose:
        print("\n" + "=" * 60)
        print("斐波那契数列泛化测试")
        print("=" * 60)
        print(f"原始序列: {[f'{x:.4f}' for x in fib_seq[0].tolist()]}")
    
    with torch.no_grad():
        # 用前9个预测第10个
        input_seq = fib_seq[:, :-1]
        target = fib_seq[0, -1].item()
        
        out = model.forward(input_seq)
        
        # 几何与外力分析
        geo = out['a_geodesic'].norm().item()
        ext = out['F_external'].norm().item()
        geo_ratio = geo / (geo + ext + 1e-8)
        
        pred = out['x_new'].item()
        error = abs(pred - target)
        
        # 度规分析
        g = out['metric'][0, -1].cpu()
        sig = get_signature(g)
        det_g = torch.linalg.det(g).item()
        eigs = torch.linalg.eigvalsh(g)
        
        if verbose:
            print(f"\n输入: {[f'{x:.4f}' for x in input_seq[0].tolist()]}")
            print(f"真实下一个: {target:.4f}")
            print(f"预测下一个: {pred:.4f}")
            print(f"误差: {error:.4f}")
            print(f"\n--- 几何分析 ---")
            print(f"几何加速度 |a_geo|: {geo:.6f}")
            print(f"外力 |F_ext|: {ext:.6f}")
            print(f"几何占比: {geo_ratio:.1%}")
            print(f"\n--- 度规分析 ---")
            print(f"签名: {sig.signature_str}")
            print(f"det(g): {det_g:.6f}")
            print(f"特征值: [{eigs.min().item():.4f}, {eigs.max().item():.4f}]")
    
    return {
        'target': target,
        'pred': pred,
        'error': error,
        'geo_ratio': geo_ratio,
        'geo': geo,
        'ext': ext,
        'signature': sig.signature_str,
        'det': det_g,
        'eig_min': eigs.min().item(),
        'eig_max': eigs.max().item(),
    }


def train_add_only(
    epochs: int = 100,  # 增加轮次
    z_dim: int = 4,
    samples_per_epoch: int = 20,
    lr: float = 3e-4,
    unconstrained: bool = False,
    verbose: bool = True,
):
    """只训练加法法则，返回几何占比最高的模型"""
    from .model import MetricEvolutionModel, dynamic_force_weight
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = MetricEvolutionModel(
        z_dim=z_dim, input_dim=1,
        hidden_mult=4,  # 减小网络
        unconstrained=unconstrained,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    
    mode_str = "暴论" if unconstrained else "约束"
    if verbose:
        print(f"\n>>> 训练加法法则 [{mode_str}模式]")
        print(f"z_dim={z_dim}, epochs={epochs}")
    
    best_geo = 0.0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        # 更严厉的 force_weight：从一开始就重惩外力
        fw = dynamic_force_weight(epoch, epochs, 1.0, 5.0, 0.4, 0.1)
        
        total_loss, total_geo, total_ext = 0, 0, 0
        
        for _ in range(samples_per_epoch):
            # 只生成加法序列
            seq = generate_sequence('add', 10, device, deterministic=False)
            loss_dict = model.compute_loss(
                seq[:, :-1], seq[:, -1],
                force_weight=fw,  # 更强的惩罚
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
        
        # 保存几何占比最高的模型
        if geo_ratio > best_geo:
            best_geo = geo_ratio
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"E{epoch:3d}: loss={total_loss/samples_per_epoch:.4f} geo={geo_ratio:.1%} fw={fw:.2f}")
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    if verbose:
        print(f"\n最佳几何占比: {best_geo:.1%}")
    
    return model, best_geo


def test_add_to_fibonacci(epochs: int = 100, n_runs: int = 2, unconstrained: bool = False):
    """训练加法 → 测试斐波那契"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "#" * 70)
    print("泛化测试: 加法训练 → 斐波那契预测")
    print("#" * 70)
    
    results = []
    
    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{n_runs}")
        print("="*60)
        
        # 训练只学加法
        model, best_geo = train_add_only(
            epochs=epochs,
            unconstrained=unconstrained,
            verbose=True,
        )
        
        # 测试加法
        model.eval()
        seq_add = generate_sequence('add', 10, device, deterministic=True)
        with torch.no_grad():
            out = model.forward(seq_add[:, :-1])
            err_add = abs(out['x_new'].item() - seq_add[0, -1].item())
            geo_add = out['a_geodesic'].norm().item()
            ext_add = out['F_external'].norm().item()
            ratio_add = geo_add / (geo_add + ext_add + 1e-8)
        
        print(f"\n[加法测试] err={err_add:.4f}, geo={ratio_add:.1%}")
        
        # 测试斐波那契
        res = test_on_fibonacci(model, device, verbose=True)
        
        results.append({
            'best_geo': best_geo,
            'add_err': err_add,
            'add_geo': ratio_add,
            'fib_err': res['error'],
            'fib_geo': res['geo_ratio'],
            'fib_sig': res['signature'],
        })
    
    # 统计
    print("\n" + "=" * 70)
    print("统计结果")
    print("=" * 70)
    
    fib_errs = torch.tensor([r['fib_err'] for r in results])
    fib_geos = torch.tensor([r['fib_geo'] for r in results])
    add_errs = torch.tensor([r['add_err'] for r in results])
    
    print(f"\n加法训练集:")
    print(f"  误差: mean={add_errs.mean():.4f}, range=[{add_errs.min():.4f}, {add_errs.max():.4f}]")
    
    print(f"\n斐波那契泛化:")
    print(f"  误差: mean={fib_errs.mean():.4f}, range=[{fib_errs.min():.4f}, {fib_errs.max():.4f}]")
    print(f"  几何占比: mean={fib_geos.mean():.1%}, range=[{fib_geos.min():.1%}, {fib_geos.max():.1%}]")
    
    # 签名分布
    from collections import Counter
    sigs = [r['fib_sig'] for r in results]
    print(f"  签名分布: {dict(Counter(sigs))}")
    
    success = sum(1 for e in fib_errs if e < 0.1)
    print(f"\n泛化成功率 (err<0.1): {success}/{n_runs}")
    
    return results


def train_fibonacci(
    epochs: int = 100,
    z_dim: int = 4,
    samples_per_epoch: int = 20,
    lr: float = 3e-4,
    n_runs: int = 2,
    verbose: bool = True,
):
    """直接训练斐波那契数列"""
    from .model import MetricEvolutionModel, dynamic_force_weight
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "#" * 70)
    print("直接训练斐波那契数列")
    print("#" * 70)
    
    results = []
    
    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{n_runs}")
        print("="*60)
        
        model = MetricEvolutionModel(
            z_dim=z_dim, input_dim=1,
            hidden_mult=4,
            unconstrained=False,
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
        
        print(f">>> 训练斐波那契 [z_dim={z_dim}, epochs={epochs}]")
        
        best_geo = 0.0
        best_err = float('inf')
        best_model_state = None
        
        for epoch in range(1, epochs + 1):
            model.train()
            fw = dynamic_force_weight(epoch, epochs, 1.0, 5.0, 0.4, 0.1)
            
            total_loss, total_geo, total_ext, total_err = 0, 0, 0, 0
            
            for _ in range(samples_per_epoch):
                # 生成斐波那契序列
                seq = generate_fibonacci_batch(1, 10, device)
                loss_dict = model.compute_loss(
                    seq[:, :-1], seq[:, -1],
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
                total_err += abs(loss_dict['x_new'].item() - seq[0, -1].item())
            
            scheduler.step()
            
            geo_ratio = total_geo / (total_geo + total_ext + 1e-8)
            avg_err = total_err / samples_per_epoch
            
            # 保存最佳模型（基于误差）
            if avg_err < best_err:
                best_err = avg_err
                best_geo = geo_ratio
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"E{epoch:3d}: loss={total_loss/samples_per_epoch:.4f} "
                      f"err={avg_err:.4f} geo={geo_ratio:.1%} fw={fw:.2f}")
        
        # 加载最佳模型
        if best_model_state:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        
        print(f"\n最佳: err={best_err:.4f}, geo={best_geo:.1%}")
        
        # 测试
        res = test_on_fibonacci(model, device, verbose=True)
        
        results.append({
            'train_err': best_err,
            'train_geo': best_geo,
            'test_err': res['error'],
            'test_geo': res['geo_ratio'],
            'signature': res['signature'],
        })
    
    # 统计
    print("\n" + "=" * 70)
    print("统计结果")
    print("=" * 70)
    
    test_errs = torch.tensor([r['test_err'] for r in results])
    test_geos = torch.tensor([r['test_geo'] for r in results])
    
    print(f"\n测试集:")
    print(f"  误差: mean={test_errs.mean():.4f}, range=[{test_errs.min():.4f}, {test_errs.max():.4f}]")
    print(f"  几何占比: mean={test_geos.mean():.1%}")
    
    success = sum(1 for e in test_errs if e < 0.1)
    print(f"\n成功率 (err<0.1): {success}/{n_runs}")
    
    return results


def test_extrapolation(model, device: str = 'cpu', n_given: int = 5, n_predict: int = 5, use_int: bool = False):
    """外推测试：给定前 n_given 个，预测后 n_predict 个"""
    model.eval()
    model = model.to(device)
    
    # 获取模型的 dtype
    dtype = next(model.parameters()).dtype
    
    # 生成完整序列
    full_seq = generate_fibonacci(n_given + n_predict, normalize=not use_int, device=device)
    full_seq = full_seq.to(dtype)
    
    print("\n" + "=" * 60)
    mode = "整数模式" if use_int else "归一化模式"
    print(f"外推测试 ({mode}): 给定 {n_given} 个 → 预测 {n_predict} 个")
    print("=" * 60)
    
    if use_int:
        print(f"完整序列: {[int(x) for x in full_seq[0].tolist()]}")
        print(f"给定: {[int(x) for x in full_seq[0, :n_given].tolist()]}")
    else:
        print(f"完整序列: {[f'{x:.4f}' for x in full_seq[0].tolist()]}")
        print(f"给定: {[f'{x:.4f}' for x in full_seq[0, :n_given].tolist()]}")
    
    # 逐步预测
    current_seq = full_seq[:, :n_given].clone()
    predictions = []
    errors = []
    geo_ratios = []
    
    with torch.no_grad():
        for step in range(n_predict):
            out = model.forward(current_seq)
            pred = out['x_new'].item()
            
            true_val = full_seq[0, n_given + step].item()
            err = abs(pred - true_val)
            geo = out['a_geodesic'].norm().item()
            ext = out['F_external'].norm().item()
            geo_ratio = geo / (geo + ext + 1e-8)
            
            predictions.append(pred)
            errors.append(err)
            geo_ratios.append(geo_ratio)
            
            if use_int:
                print(f"  Step {step+1}: pred={pred:.1f}, true={int(true_val)}, "
                      f"err={err:.1f}, geo={geo_ratio:.1%}")
            else:
                print(f"  Step {step+1}: pred={pred:.4f}, true={true_val:.4f}, "
                      f"err={err:.4f}, geo={geo_ratio:.1%}")
            
            new_val = torch.tensor([[pred]], device=device, dtype=dtype)
            current_seq = torch.cat([current_seq, new_val], dim=1)
    
    avg_err = sum(errors) / len(errors)
    avg_geo = sum(geo_ratios) / len(geo_ratios)
    
    print(f"\n统计:")
    print(f"  平均误差: {avg_err:.2f}")
    print(f"  平均几何占比: {avg_geo:.1%}")
    
    if use_int:
        print(f"  预测序列: {[f'{x:.1f}' for x in predictions]}")
        print(f"  真实序列: {[int(x) for x in full_seq[0, n_given:].tolist()]}")
    else:
        print(f"  预测序列: {[f'{x:.4f}' for x in predictions]}")
        print(f"  真实序列: {[f'{x:.4f}' for x in full_seq[0, n_given:].tolist()]}")
    
    return {
        'predictions': predictions,
        'true_values': full_seq[0, n_given:].tolist(),
        'errors': errors,
        'avg_error': avg_err,
        'avg_geo': avg_geo,
    }


def train_and_extrapolate(epochs: int = 100, n_given: int = 5, n_predict: int = 5, use_int: bool = False):
    """训练斐波那契并测试外推"""
    from .model import MetricEvolutionModel, dynamic_force_weight
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z_dim = 4
    
    mode = "整数" if use_int else "归一化"
    print("\n" + "#" * 70)
    print(f"训练斐波那契 ({mode}) + 外推测试 (前{n_given}→后{n_predict})")
    print("#" * 70)
    
    model = MetricEvolutionModel(
        z_dim=z_dim, input_dim=1,
        hidden_mult=4, unconstrained=False,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-5)
    
    print(f">>> 训练 [z_dim={z_dim}, epochs={epochs}]")
    print(f">>> 使用短序列训练 (4-7个点预测下一个)")
    
    for epoch in range(1, epochs + 1):
        model.train()
        fw = dynamic_force_weight(epoch, epochs, 1.0, 5.0, 0.4, 0.1)
        
        total_loss, total_geo, total_ext = 0, 0, 0
        
        for _ in range(20):
            # 生成斐波那契序列（整数或归一化）
            a = torch.randint(1, 5, (1,)).item()
            b = torch.randint(1, 5, (1,)).item()
            full_seq = generate_fibonacci(10, normalize=not use_int, device=device, start=(a, b))
            
            # 随机选择输入长度 (4-7)
            input_len = torch.randint(4, 8, (1,)).item()
            seq = full_seq[:, :input_len + 1]
            
            loss_dict = model.compute_loss(
                seq[:, :-1], seq[:, -1],
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
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"E{epoch:3d}: loss={total_loss/20:.4f} geo={geo_ratio:.1%} fw={fw:.2f}")
    
    print(f"\n训练完成, geo={geo_ratio:.1%}")
    
    # 外推测试
    result = test_extrapolation(model, device, n_given, n_predict, use_int=use_int)
    
    return model, result


def train_extrapolate_v2(
    epochs: int = 200,
    n_given: int = 5,
    n_predict: int = 5,
    z_dim: int = 8,
    use_scheduled_sampling: bool = True,
    train_seq_len: int = None,
    max_ss_prob: float = 0.5,  # SS 最大概率
    use_float64: bool = False,  # 使用更高精度
):
    """
    改进版训练 - 包含 Scheduled Sampling
    
    改进：
        1. Scheduled Sampling: 逐渐使用模型自己的预测作为输入
        2. 更大的模型容量 (z_dim=8)
        3. 更多训练轮次 (200)
        4. 始终使用归一化
        5. 训练序列长度匹配测试序列
        6. 可配置 SS 最大概率
        7. 可选 float64 高精度
    """
    from .model import MetricEvolutionModel, dynamic_force_weight
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64 if use_float64 else torch.float32
    
    # 训练序列长度至少要覆盖测试序列
    if train_seq_len is None:
        train_seq_len = n_given + n_predict + 2
    
    print("\n" + "#" * 70)
    print(f"改进版训练 (SS→0.{int(max_ss_prob*100):02d} + z_dim={z_dim})")
    print(f"外推测试: 前{n_given} → 后{n_predict}, 训练序列长度={train_seq_len}")
    if use_float64:
        print(f"精度: float64")
    print("#" * 70)
    
    model = MetricEvolutionModel(
        z_dim=z_dim, input_dim=1,
        hidden_mult=6,
        unconstrained=False,
    ).to(device).to(dtype)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    print(f">>> 训练 [z_dim={z_dim}, epochs={epochs}]")
    if use_scheduled_sampling:
        print(f">>> Scheduled Sampling: 0 → {max_ss_prob}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        fw = dynamic_force_weight(epoch, epochs, 1.0, 5.0, 0.4, 0.1)
        
        # Scheduled Sampling: 分阶段策略
        # 前 50% 轮次不使用 SS，先学好基础
        # 后 50% 轮次逐渐增加 SS
        if use_scheduled_sampling:
            if epoch < epochs * 0.5:
                sample_prob = 0.0
            else:
                progress = (epoch - epochs * 0.5) / (epochs * 0.5)
                sample_prob = min(max_ss_prob, max_ss_prob * progress)
        else:
            sample_prob = 0.0
        
        total_loss, total_geo, total_ext = 0, 0, 0
        
        for _ in range(30):
            # 生成足够长的归一化斐波那契序列
            a = torch.randint(1, 5, (1,)).item()
            b = torch.randint(1, 5, (1,)).item()
            full_seq = generate_fibonacci(train_seq_len, normalize=True, device=device, start=(a, b))
            full_seq = full_seq.to(dtype)
            
            # 随机选择序列起点和长度 (4 到 n_given+2)
            max_start = max(1, train_seq_len - n_given - 2)
            start_idx = torch.randint(0, max_start, (1,)).item()
            seq_len = torch.randint(4, n_given + 3, (1,)).item()
            end_idx = min(start_idx + seq_len + 1, train_seq_len)
            
            seq = full_seq[:, start_idx:end_idx].clone()
            
            # Scheduled Sampling: 有概率用模型预测替代真实值
            if sample_prob > 0 and seq.shape[1] > 3 and torch.rand(1).item() < sample_prob:
                model.eval()
                with torch.no_grad():
                    mid = seq.shape[1] // 2
                    out = model.forward(seq[:, :mid])
                    seq = seq.clone()
                    seq[0, mid] = out['x_new'].squeeze()
                model.train()
            
            loss_dict = model.compute_loss(
                seq[:, :-1], seq[:, -1],
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
        
        if epoch % 40 == 0 or epoch == 1:
            print(f"E{epoch:3d}: loss={total_loss/30:.4f} geo={geo_ratio:.1%} "
                  f"fw={fw:.2f} ss={sample_prob:.2f}")
    
    print(f"\n训练完成, geo={geo_ratio:.1%}")
    
    # 外推测试
    result = test_extrapolation(model, device, n_given, n_predict, use_int=False)
    
    return model, result


def multi_run_fibonacci_test(n_runs: int = 5, epochs: int = 30, z_dim: int = 5):
    """多次运行斐波那契泛化测试"""
    
    print("\n" + "#" * 70)
    print(f"斐波那契泛化测试: {n_runs} 次运行")
    print("#" * 70)
    
    results_c = []  # 约束模式
    results_u = []  # 暴论模式
    
    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{n_runs}")
        print("="*60)
        
        # 约束模式
        print("\n>>> 约束模式")
        model_c, _, _ = train(epochs, z_dim, unconstrained=False, seed=None, verbose=False)
        res_c = test_on_fibonacci(model_c, verbose=False)
        results_c.append(res_c)
        print(f"  err={res_c['error']:.4f}, geo={res_c['geo_ratio']:.1%}, sig=正定")
        
        # 暴论模式
        print(">>> 暴论模式")
        model_u, _, _ = train(epochs, z_dim, unconstrained=True, seed=None, verbose=False)
        res_u = test_on_fibonacci(model_u, verbose=False)
        results_u.append(res_u)
        print(f"  err={res_u['error']:.4f}, geo={res_u['geo_ratio']:.1%}, sig={res_u['signature']}")
    
    # 统计
    print("\n" + "=" * 70)
    print("斐波那契泛化统计")
    print("=" * 70)
    
    def stats(results, name):
        errors = [r['error'] for r in results]
        geos = [r['geo_ratio'] for r in results]
        err_t = torch.tensor(errors)
        geo_t = torch.tensor(geos)
        print(f"\n{name}:")
        print(f"  误差: mean={err_t.mean():.4f}, std={err_t.std():.4f}, "
              f"range=[{err_t.min():.4f}, {err_t.max():.4f}]")
        print(f"  几何占比: mean={geo_t.mean():.1%}, std={geo_t.std():.4f}, "
              f"range=[{geo_t.min():.1%}, {geo_t.max():.1%}]")
        success = sum(1 for e in errors if e < 0.1)
        print(f"  泛化成功率 (err<0.1): {success}/{len(errors)}")
    
    stats(results_c, "约束模式")
    stats(results_u, "暴论模式")
    
    # 签名分布
    from collections import Counter
    sigs = [r['signature'] for r in results_u]
    print(f"\n暴论模式签名分布: {dict(Counter(sigs))}")
    
    return results_c, results_u


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="斐波那契泛化测试")
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--runs', '-r', type=int, default=3)
    parser.add_argument('--unconstrained', '-u', action='store_true', default=False)
    args = parser.parse_args()
    
    test_add_to_fibonacci(
        epochs=args.epochs,
        n_runs=args.runs,
        unconstrained=args.unconstrained,
    )
