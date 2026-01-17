"""Evolution Package 测试脚本"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from evolution import (
    EvolutionForSequence, 
    SecondOrderDynamics, 
    EvolutionConfig,
    LearnedMetric,
    LearnedSymmetryGenerator,
    SeparableHamiltonian,
    SymplecticIntegrator,
)


def test_sequence_model():
    """测试序列演化模型"""
    print("\n1. 测试 EvolutionForSequence...")
    model = EvolutionForSequence(d=64, hidden_dim=128, num_generators=4)
    params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {params:,}")
    
    values = torch.rand(2, 10)
    pred, acc = model(values)
    print(f"   输入形状: {values.shape}")
    print(f"   预测形状: {pred.shape}")
    print(f"   加速度形状: {acc.shape}")
    
    # 生成测试
    print("\n2. 测试自回归生成...")
    generated = model.generate(values[:, :3], max_new=5)
    print(f"   生成序列长度: {generated.shape[1]}")
    
    return True


def test_dynamics():
    """测试二阶动力学"""
    print("\n3. 测试 SecondOrderDynamics...")
    config = EvolutionConfig(dt=0.1, num_steps=10, integrator='verlet')
    dynamics = SecondOrderDynamics(d=32, hidden_dim=64, config=config)
    
    z = torch.randn(2, 32)
    v = torch.randn(2, 32)
    state, info = dynamics(z, v, return_trajectory=True)
    
    print(f"   轨迹形状: {info['z_trajectory'].shape}")
    print(f"   能量轨迹形状: {info['energies'].shape}")
    
    # 能量守恒检验
    print("\n4. 能量守恒检验...")
    energies = info['energies'][0].detach()
    energy_var = energies.var().item()
    print(f"   能量方差: {energy_var:.6f} (越小越好)")
    
    return True


def test_metric():
    """测试度量张量"""
    print("\n5. 测试 LearnedMetric...")
    metric = LearnedMetric(d=16, hidden_dim=32)
    z = torch.randn(2, 16)
    
    g = metric(z)
    print(f"   度量矩阵形状: {g.shape}")
    
    # 检查正定性
    eigenvalues = torch.linalg.eigvalsh(g)
    min_eig = eigenvalues.min().item()
    print(f"   最小特征值: {min_eig:.6f} (应为正)")
    
    return min_eig > 0


def test_symmetry():
    """测试对称性生成元"""
    print("\n6. 测试 LearnedSymmetryGenerator...")
    gen = LearnedSymmetryGenerator(d=16, num_generators=4, generator_type='antisymmetric')
    G = gen()
    print(f"   生成元形状: {G.shape}")
    
    # 检查反对称性
    antisym_error = (G + G.transpose(-2, -1)).abs().max().item()
    print(f"   反对称误差: {antisym_error:.6f} (应接近0)")
    
    # 测试指数映射
    x = torch.randn(2, 16)
    alpha = torch.randn(2, 4)
    x_transformed = gen.exp_action(x, alpha)
    print(f"   变换后形状: {x_transformed.shape}")
    
    return antisym_error < 1e-5


def test_hamiltonian():
    """测试哈密顿结构"""
    print("\n7. 测试 SeparableHamiltonian...")
    H = SeparableHamiltonian(d=8, hidden_dim=32)
    
    q = torch.randn(2, 8)
    p = torch.randn(2, 8)
    
    energy = H(q, p)
    print(f"   能量形状: {energy.shape}")
    
    # 辛积分
    print("\n8. 测试辛积分...")
    q_traj, p_traj, times = SymplecticIntegrator.integrate(
        H, q, p, (0, 1), num_steps=20, method='leapfrog'
    )
    
    # 能量守恒检验
    E_init = H(q_traj[:, 0], p_traj[:, 0])
    E_final = H(q_traj[:, -1], p_traj[:, -1])
    energy_drift = (E_final - E_init).abs().mean().item()
    print(f"   能量漂移: {energy_drift:.6f}")
    
    return True


def main():
    print("=" * 60)
    print("Evolution Package 完整测试")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("序列模型", test_sequence_model()))
    except Exception as e:
        print(f"   错误: {e}")
        results.append(("序列模型", False))
    
    try:
        results.append(("二阶动力学", test_dynamics()))
    except Exception as e:
        print(f"   错误: {e}")
        results.append(("二阶动力学", False))
    
    try:
        results.append(("度量张量", test_metric()))
    except Exception as e:
        print(f"   错误: {e}")
        results.append(("度量张量", False))
    
    try:
        results.append(("对称性", test_symmetry()))
    except Exception as e:
        print(f"   错误: {e}")
        results.append(("对称性", False))
    
    try:
        results.append(("哈密顿", test_hamiltonian()))
    except Exception as e:
        print(f"   错误: {e}")
        results.append(("哈密顿", False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 通过")
    
    if passed == len(results):
        print("\n所有测试通过!")
    else:
        print("\n部分测试失败，请检查。")


if __name__ == "__main__":
    main()
