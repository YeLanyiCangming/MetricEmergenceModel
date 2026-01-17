"""演化模型 - 规则学习训练

度规涌现架构：数据 → z → g(z) → Γ → a_geo + F_ext → d²x
"""

from __future__ import annotations
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from .model import MetricEvolutionModel, dynamic_force_weight
from .metric_analysis import (
    get_signature, MetricCollector, TrainingHistory,
    plot_training_comparison, plot_eigenvalue_analysis, plot_single_training,
    eigenvalue_trajectory, z_projection_2d, plot_trajectory_analysis,
)


def set_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_sequence(rule: str, length: int = 10, device: str = 'cpu', deterministic: bool = False) -> torch.Tensor:
    """生成规则序列 (ADD/MUL/ALT/ACC)
    
    deterministic=True 时使用固定参数，用于测试
    """
    if deterministic:
        # 固定参数，确保测试可重复
        if rule == 'add':
            start, d = 0.1, 0.08
            seq = [start + i * d for i in range(length)]
        elif rule == 'mul':
            start, k = 0.1, 1.2
            seq = [start * (k ** i) for i in range(length)]
        elif rule == 'alt':
            base, amp = 0.5, 0.1
            seq = [base + amp * ((-1) ** i) for i in range(length)]
        elif rule == 'acc':
            x, v, a = 0.05, 0.02, 0.01
            seq = []
            for _ in range(length):
                seq.append(x)
                x, v = x + v, v + a
        else:
            raise ValueError(f"Unknown rule: {rule}")
    else:
        # 随机参数，用于训练
        if rule == 'add':
            start, d = torch.rand(1).item() * 0.3, torch.rand(1).item() * 0.1 + 0.02
            seq = [start + i * d for i in range(length)]
        elif rule == 'mul':
            start, k = torch.rand(1).item() * 0.1 + 0.05, torch.rand(1).item() * 0.3 + 1.1
            seq = [start * (k ** i) for i in range(length)]
        elif rule == 'alt':
            base, amp = torch.rand(1).item() * 0.3 + 0.3, torch.rand(1).item() * 0.1 + 0.02
            seq = [base + amp * ((-1) ** i) for i in range(length)]
        elif rule == 'acc':
            x, v = torch.rand(1).item() * 0.1, torch.rand(1).item() * 0.03 + 0.01
            a, seq = torch.rand(1).item() * 0.02 + 0.005, []
            for _ in range(length):
                seq.append(x)
                x, v = x + v, v + a
        else:
            raise ValueError(f"Unknown rule: {rule}")
    
    return torch.tensor([max(0, min(1, x)) for x in seq], device=device).unsqueeze(0)


def train(
    epochs: int = 50,
    z_dim: int = 5,
    samples_per_rule: int = 20,
    lr: float = 2e-4,
    min_force_weight: float = 0.1,
    max_force_weight: float = 1.2,   # 适中的最大权重
    warmup_ratio: float = 0.7,      # 延长 warmup
    det_target: float = 0.1,
    grad_clip: float = 0.5,
    seed: int = None,
    unconstrained: bool = False,
    hidden_mult: int = 8,
    verbose: bool = True,
    kappa_threshold: float = 50.0,
    kappa_weight: float = 0.05,
    min_abs_eig: float = 0.5,
    min_eig_weight: float = 10.0,
) -> tuple:
    """训练模型，返回 (model, metrics, history)"""
    if seed is not None:
        set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rules = ['add', 'mul', 'alt', 'acc']
    
    model = MetricEvolutionModel(
        z_dim=z_dim, input_dim=1,
        hidden_mult=hidden_mult, unconstrained=unconstrained,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    
    # 训练历史记录
    history = TrainingHistory(mode="unconstrained" if unconstrained else "constrained")
    
    mode_str = "暴论模式" if unconstrained else "约束模式"
    
    if verbose:
        print("=" * 60)
        print(f"度规涌现演化模型 - {mode_str}")
        print("=" * 60)
        print(f"z_dim={z_dim}, params={sum(p.numel() for p in model.parameters()):,}")
        print(f"lr={lr}, grad_clip={grad_clip}, hidden_mult={hidden_mult}")
        print("=" * 60)
    
    best_loss, best_geo = float('inf'), 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        fw = dynamic_force_weight(epoch, epochs, min_force_weight, max_force_weight, warmup_ratio)
        
        stats = {'loss': 0, 'nll': 0, 'geo': 0, 'force': 0, 'det': 0, 'kappa': 0}
        n = len(rules) * samples_per_rule
        
        for rule in rules:
            for _ in range(samples_per_rule):
                seq = generate_sequence(rule, 10, device)
                loss_dict = model.compute_loss(
                    seq[:, :-1], seq[:, -1], 
                    force_weight=fw, 
                    det_target=det_target,
                    kappa_threshold=kappa_threshold,
                    kappa_weight=kappa_weight,
                    min_abs_eig=min_abs_eig,
                    min_eig_weight=min_eig_weight,
                )
                
                optimizer.zero_grad()
                loss_dict['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                
                stats['loss'] += loss_dict['loss'].item()
                stats['nll'] += loss_dict['nll_loss'].item()
                stats['geo'] += loss_dict['a_geodesic'].norm().item()
                stats['force'] += loss_dict['F_external'].norm().item()
                stats['det'] += loss_dict['det_abs_mean'].item()
                if unconstrained:
                    stats['kappa'] += loss_dict['kappa_mean'].item()
        
        scheduler.step()
        
        # 计算均值
        for k in stats:
            stats[k] /= n
        geo_ratio = stats['geo'] / (stats['geo'] + stats['force'] + 1e-8)
        
        # 记录历史
        history.record_epoch(epoch, stats['loss'], stats['nll'], geo_ratio, stats['det'], fw)
        
        best_loss = min(best_loss, stats['loss'])
        best_geo = max(best_geo, geo_ratio)
        
        if verbose and (epoch % 10 == 0 or epoch == 1):
            kappa_str = f" kappa={stats['kappa']:.1f}" if unconstrained else ""
            print(f"E{epoch:3d}: loss={stats['loss']:.4f} nll={stats['nll']:.4f} "
                  f"geo={geo_ratio:.1%} det={stats['det']:.4f} fw={fw:.2f}{kappa_str}")
    
    # 测试
    if verbose:
        print("\n" + "=" * 60)
        print(f"测试 [{mode_str}]")
        print("=" * 60)
    
    model.eval()
    correct, geo_dom = 0, 0
    
    with torch.no_grad():
        for rule in rules:
            # 测试使用固定序列，确保可重复
            seq = generate_sequence(rule, 10, device, deterministic=True)
            out = model.forward(seq[:, :-1])
            
            geo = out['a_geodesic'].norm().item()
            ext = out['F_external'].norm().item()
            ratio = geo / (geo + ext + 1e-8)
            error = abs(out['x_new'].item() - seq[0, -1].item())
            
            # 记录测试结果
            sig_str, eigs = "", ()
            if unconstrained:
                sig = get_signature(out['metric'][0, -1])
                sig_str = sig.signature_str
                eigs = sig.eigenvalues
            
            history.record_test(rule, error, ratio, eigs, sig_str)
            
            if ratio > 0.3: geo_dom += 1
            if error < 0.05: correct += 1
            
            if verbose:
                extra = f" sig={sig_str}" if sig_str else ""
                print(f"[{rule.upper()}] err={error:.4f} geo={ratio:.1%}{extra}")
    
    if verbose:
        print(f"\n准确率: {correct}/4  几何主导: {geo_dom}/4  最佳geo: {best_geo:.1%}")
    
    return model, {
        'accuracy': correct / 4,
        'geo_dominant': geo_dom / 4,
        'best_loss': best_loss,
        'best_geo': best_geo,
    }, history


def train_and_compare(epochs: int = 50, z_dim: int = 5, seed: int = None, plot: bool = True):
    """训练两种模式并对比可视化"""
    print("\n" + "#" * 70)
    print("模式对比: 约束模式 vs 暴论模式")
    print("#" * 70)
    
    print("\n>>> 约束模式")
    m1, r1, h1 = train(epochs, z_dim, unconstrained=False, seed=seed)
    
    print("\n>>> 暴论模式")
    m2, r2, h2 = train(epochs, z_dim, unconstrained=True, seed=seed)
    
    print("\n" + "#" * 70)
    print("对比结果")
    print("#" * 70)
    print(f"           约束模式    暴论模式")
    print(f"准确率:    {r1['accuracy']:.0%}          {r2['accuracy']:.0%}")
    print(f"几何占比:  {r1['best_geo']:.0%}          {r2['best_geo']:.0%}")
    print(f"几何主导:  {r1['geo_dominant']:.0%}          {r2['geo_dominant']:.0%}")
    
    if plot:
        print("\n生成可视化图表...")
        plot_training_comparison(h1, h2, save_path='training_comparison.png')
        if h2.test_eigenvalues and h2.test_eigenvalues[0]:
            plot_eigenvalue_analysis(h2, save_path='eigenvalue_analysis.png')
    
    return m1, m2, h1, h2


def diagnose_stability(n_runs: int = 5, epochs: int = 10, z_dim: int = 5):
    """诊断 g(z) 稳定性 - 多次运行统计"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "#" * 70)
    print(f"g(z) 稳定性诊断: {n_runs} 次运行, {epochs} 轮")
    print("#" * 70)
    
    # 收集统计数据
    stats_c = {'geo': [], 'det': [], 'eig_min': [], 'eig_max': [], 'cond': []}
    stats_u = {'geo': [], 'det': [], 'eig_min': [], 'eig_max': [], 'cond': [], 'sig': [], 'kappa': [], 'min_eig_pen': []}
    
    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")
        
        # 训练
        model_c, _, _ = train(epochs, z_dim, unconstrained=False, seed=None, verbose=False)
        model_u, _, _ = train(epochs, z_dim, unconstrained=True, seed=None, verbose=False)
        
        model_c.eval()
        model_u.eval()
        
        # 用固定序列测试
        seq = generate_sequence('alt', 10, device, deterministic=True)
        
        with torch.no_grad():
            out_c = model_c.forward(seq[:, :-1])
            out_u = model_u.forward(seq[:, :-1])
            
            g_c = out_c['metric'][0, -1].cpu()  # 最后一个位置的度规
            g_u = out_u['metric'][0, -1].cpu()
            
            # 约束模式统计
            eig_c = torch.linalg.eigvalsh(g_c)
            geo_c = out_c['a_geodesic'].norm().item()
            ext_c = out_c['F_external'].norm().item()
            
            stats_c['geo'].append(geo_c / (geo_c + ext_c + 1e-8))
            stats_c['det'].append(torch.linalg.det(g_c).item())
            stats_c['eig_min'].append(eig_c.min().item())
            stats_c['eig_max'].append(eig_c.max().item())
            stats_c['cond'].append((eig_c.max() / (eig_c.min() + 1e-8)).item())
            
            # 暴论模式统计
            eig_u = torch.linalg.eigvalsh(g_u)
            geo_u = out_u['a_geodesic'].norm().item()
            ext_u = out_u['F_external'].norm().item()
            sig = get_signature(g_u)
            
            # 计算条件数
            abs_eig_u = torch.abs(eig_u)
            kappa_u = (abs_eig_u.max() / (abs_eig_u.min() + 1e-8)).item()
            
            # 计算 min_eig_penalty（使用默认 min_abs_eig=0.5）
            min_abs_eig_target = 0.5
            below_threshold = torch.relu(min_abs_eig_target - abs_eig_u)
            min_eig_pen = below_threshold.mean().item()
            
            stats_u['geo'].append(geo_u / (geo_u + ext_u + 1e-8))
            stats_u['det'].append(torch.linalg.det(g_u).item())
            stats_u['eig_min'].append(eig_u.min().item())
            stats_u['eig_max'].append(eig_u.max().item())
            stats_u['cond'].append((eig_c.max() / (eig_c.min().abs() + 1e-8)).item())
            stats_u['sig'].append(sig.signature_str)
            stats_u['kappa'].append(kappa_u)
            stats_u['min_eig_pen'].append(min_eig_pen)
        
        print(f"  约束: geo={stats_c['geo'][-1]:.1%}, det={stats_c['det'][-1]:.4f}, "
              f"eig=[{stats_c['eig_min'][-1]:.3f}, {stats_c['eig_max'][-1]:.3f}]")
        print(f"  暴论: geo={stats_u['geo'][-1]:.1%}, det={stats_u['det'][-1]:.4f}, "
              f"eig=[{stats_u['eig_min'][-1]:.3f}, {stats_u['eig_max'][-1]:.3f}], "
              f"sig={stats_u['sig'][-1]}, kappa={stats_u['kappa'][-1]:.1f}, pen={stats_u['min_eig_pen'][-1]:.3f}")
    
    # 统计分析
    print("\n" + "=" * 70)
    print("稳定性统计 (ALT 规则)")
    print("=" * 70)
    
    def print_stats(name, values):
        t = torch.tensor(values)
        print(f"  {name}: mean={t.mean():.4f}, std={t.std():.4f}, "
              f"range=[{t.min():.4f}, {t.max():.4f}]")
    
    print("\n约束模式:")
    print_stats("几何占比", stats_c['geo'])
    print_stats("det(g)", stats_c['det'])
    print_stats("最小特征值", stats_c['eig_min'])
    print_stats("最大特征值", stats_c['eig_max'])
    print_stats("条件数", stats_c['cond'])
    
    print("\n暴论模式:")
    print_stats("几何占比", stats_u['geo'])
    print_stats("det(g)", stats_u['det'])
    print_stats("最小特征值", stats_u['eig_min'])
    print_stats("最大特征值", stats_u['eig_max'])
    print_stats("条件数 kappa", stats_u['kappa'])
    print_stats("min_eig_penalty", stats_u['min_eig_pen'])
    
    # 签名分布
    from collections import Counter
    sig_dist = Counter(stats_u['sig'])
    print(f"  签名分布: {dict(sig_dist)}")
    
    # 稳定性评估
    print("\n" + "-" * 70)
    print("稳定性评估:")
    
    geo_c_std = torch.tensor(stats_c['geo']).std().item()
    geo_u_std = torch.tensor(stats_u['geo']).std().item()
    det_c_std = torch.tensor(stats_c['det']).std().item()
    det_u_std = torch.tensor(stats_u['det']).std().item()
    kappa_u_mean = torch.tensor(stats_u['kappa']).mean().item()
    kappa_u_std = torch.tensor(stats_u['kappa']).std().item()
    
    print(f"  几何占比 std: 约束={geo_c_std:.4f}, 暴论={geo_u_std:.4f}")
    print(f"  det(g) std: 约束={det_c_std:.4f}, 暴论={det_u_std:.4f}")
    print(f"  暴论条件数: mean={kappa_u_mean:.1f}, std={kappa_u_std:.1f}")
    
    # 条件数评估
    if kappa_u_mean < 50:
        print(f"  ✔ 条件数良好: kappa={kappa_u_mean:.1f} < 50")
    elif kappa_u_mean < 100:
        print(f"  ⚠️ 条件数较高: kappa={kappa_u_mean:.1f}")
    else:
        print(f"  ❌ 条件数过高: kappa={kappa_u_mean:.1f} > 100")
    
    if len(sig_dist) > 1:
        print(f"  ⚠️ 暴论模式签名不稳定: {len(sig_dist)} 种不同签名")
    else:
        print(f"  ✓ 暴论模式签名稳定: {list(sig_dist.keys())[0]}")
    
    return stats_c, stats_u


def analyze_geometry(epochs: int = 10, z_dim: int = 5, rules: list = None, seed: int = None):
    """分析几何变化"""
    rules = rules or ['alt', 'add']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "#" * 70)
    print("几何分析: z 轨迹与度规演化")
    print("#" * 70)
    
    # 训练两种模式
    print("\n>>> 训练约束模式")
    model_c, _, _ = train(epochs, z_dim, unconstrained=False, seed=seed, verbose=True)
    
    print("\n>>> 训练暴论模式")
    model_u, _, _ = train(epochs, z_dim, unconstrained=True, seed=seed, verbose=True)
    
    # 分析每个规则
    print("\n" + "=" * 70)
    print("规则分析")
    print("=" * 70)
    
    model_c.eval()
    model_u.eval()
    
    for rule in rules:
        print(f"\n【{rule.upper()}】规则")
        print("-" * 50)
        
        seq = generate_sequence(rule, 10, device, deterministic=True)
        
        with torch.no_grad():
            out_c = model_c.forward(seq[:, :-1])
            out_u = model_u.forward(seq[:, :-1])
            
            z_c = out_c['z'][0].cpu()  # [L, D]
            z_u = out_u['z'][0].cpu()
            g_c = out_c['metric'][0].cpu()  # [L, D, D]
            g_u = out_u['metric'][0].cpu()
            
            # 几何占比
            geo_c = out_c['a_geodesic'].norm().item()
            ext_c = out_c['F_external'].norm().item()
            geo_u = out_u['a_geodesic'].norm().item()
            ext_u = out_u['F_external'].norm().item()
            ratio_c = geo_c / (geo_c + ext_c + 1e-8)
            ratio_u = geo_u / (geo_u + ext_u + 1e-8)
            
            print(f"约束模式: geo={ratio_c:.1%}, sig={get_signature(g_c[-1]).signature_str}")
            print(f"暴论模式: geo={ratio_u:.1%}, sig={get_signature(g_u[-1]).signature_str}")
            
            # z 轨迹分析
            z_diff_c = (z_c[1:] - z_c[:-1]).norm(dim=-1)
            z_diff_u = (z_u[1:] - z_u[:-1]).norm(dim=-1)
            
            print(f"\nz 轨迹步长:")
            print(f"  约束: mean={z_diff_c.mean():.4f}, std={z_diff_c.std():.4f}")
            print(f"  暴论: mean={z_diff_u.mean():.4f}, std={z_diff_u.std():.4f}")
            
            # 特征值分析
            eig_c = eigenvalue_trajectory(g_c)  # [L, D]
            eig_u = eigenvalue_trajectory(g_u)
            
            print(f"\n特征值范围:")
            print(f"  约束: [{eig_c.min():.4f}, {eig_c.max():.4f}]")
            print(f"  暴论: [{eig_u.min():.4f}, {eig_u.max():.4f}]")
            
            # 绘制该规则的轨迹分析图
            plot_trajectory_analysis(z_c, z_u, g_c, g_u, rule, 
                                    save_path=f'trajectory_{rule}.png')
    
    return model_c, model_u


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="度规涌现演化模型训练")
    parser.add_argument('--mode', '-m', choices=['c', 'u', 'compare', 'analyze', 'diagnose'], default='c')
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--z_dim', '-z', type=int, default=5)
    parser.add_argument('--seed', '-s', type=int, default=None)
    parser.add_argument('--runs', '-r', type=int, default=5, help='诊断模式运行次数')
    parser.add_argument('--plot', '-p', action='store_true', help='生成可视化图表')
    args = parser.parse_args()
    
    if args.mode == 'compare':
        train_and_compare(args.epochs, args.z_dim, args.seed, plot=args.plot)
    elif args.mode == 'analyze':
        analyze_geometry(epochs=args.epochs, z_dim=args.z_dim, rules=['alt', 'add'], seed=args.seed)
    elif args.mode == 'diagnose':
        # 稳定性诊断模式
        diagnose_stability(n_runs=args.runs, epochs=args.epochs, z_dim=args.z_dim)
    else:
        unconstrained = (args.mode == 'u')
        _, result, history = train(
            epochs=args.epochs,
            z_dim=args.z_dim,
            seed=args.seed,
            unconstrained=unconstrained,
        )
        if args.plot:
            plot_single_training(history, save_path=f'training_{"unconstrained" if unconstrained else "constrained"}.png')
