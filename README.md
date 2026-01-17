# 度规涌现模型 (Metric Emergence Model)

一个基于微分几何的神经网络架构，从数据中自动涌现黎曼度规，用几何结构预测时序演化。

## 文件说明

> **核心模型**: `evolution/model.py`  
> **外推测试**: `evolution/test_extrapolation.py`  
> **泛化测试**: `evolution/test_generalization.py`  
> **训练规则**: `evolution/train_rules.py`

## 核心思想

```
data → z → g(z) → Γ → a_geodesic + F_external → prediction
```

- **状态编码**: 将输入数据映射到隐空间 z
- **度规涌现**: 从 z 学习黎曼度规 g(z)
- **测地线加速度**: 通过 Christoffel 符号计算几何加速度
- **外力补偿**: 小容量外力网络处理非几何效应
- **几何优先**: 通过惩罚机制迫使模型优先使用几何

## 项目结构

```
evolution/
├── model.py                # 核心模型实现 ⭐
├── test_extrapolation.py   # 外推测试（完整优化版）
├── test_generalization.py  # 泛化测试
├── train_rules.py          # 训练规则
├── train_evolution.py      # 训练脚本
└── ...
```

## 快速开始

### 斐波那契外推测试

```python
from evolution.test_extrapolation import train_and_test

# 训练并测试：给定前5个数，预测后10个数
result = train_and_test(
    n_given=5,
    n_predict=10,
    epochs=500,
    z_dim=8,
    fw_max=15.0,    # 外力惩罚权重
    ss_max=0.8,     # Scheduled Sampling
    use_float64=True,
)
```

### 多次运行取最佳

```python
from evolution.test_extrapolation import multi_run

best = multi_run(n_runs=3, epochs=500)
```

## 实验结果

### 斐波那契数列外推 (前5 → 后10)

| 指标 | 值 |
|------|-----|
| 平均误差 | ~0.01 |
| Step 10 误差 | ~3.5% |

**示例输出:**

```
这个是简单预测结果（代码随时可以复现，修改这些）：PS C:\Users\YeLanyi\Documents\document\Qwen3-0.6B> cd c:\Users\YeLanyi\Documents\document\Qwen3-0.6B; python -c "from evolution.test_generalization import train_extrapolate_v2; train_extrapolate_v2(epochs=200, z_dim=8, n_given=5, n_predict=10)"

######################################################################
改进版训练 (Scheduled Sampling + z_dim=8)
外推测试: 前5 → 后10, 训练序列长度=17
######################################################################
>>> 训练 [z_dim=8, epochs=200]
>>> 启用 Scheduled Sampling
E  1: loss=0.8985 geo=0.0% fw=1.00 ss=0.01
E 40: loss=-1.4988 geo=9.4% fw=1.67 ss=0.20
E 80: loss=-1.9158 geo=24.7% fw=3.00 ss=0.40
E120: loss=-1.9449 geo=25.5% fw=3.50 ss=0.50
E160: loss=-2.0609 geo=34.6% fw=4.50 ss=0.50
E200: loss=-2.0215 geo=31.9% fw=5.00 ss=0.50

训练完成, geo=31.9%

============================================================
外推测试 (归一化模式): 给定 5 个 → 预测 10 个
============================================================
完整序列: ['0.0016', '0.0016', '0.0033', '0.0049', '0.0082', '0.0131', '0.0213', '0.0344', '0.0557', '0.0902', '0.1459', '0.2361', '0.3820', '0.6180', '1.0000']
给定: ['0.0016', '0.0016', '0.0033', '0.0049', '0.0082']
  Step 1: pred=0.0136, true=0.0131, err=0.0004, geo=0.0%
  Step 2: pred=0.0224, true=0.0213, err=0.0011, geo=0.1%
  Step 3: pred=0.0368, true=0.0344, err=0.0024, geo=0.2%
  Step 4: pred=0.0600, true=0.0557, err=0.0042, geo=0.4%
  Step 5: pred=0.0969, true=0.0902, err=0.0068, geo=1.0%
  Step 6: pred=0.1556, true=0.1459, err=0.0097, geo=2.5%
  Step 7: pred=0.2486, true=0.2361, err=0.0125, geo=4.9%
  Step 8: pred=0.3979, true=0.3820, err=0.0160, geo=7.6%
  Step 9: pred=0.6421, true=0.6180, err=0.0240, geo=10.9%
  Step 10: pred=1.0350, true=1.0000, err=0.0350, geo=16.6%

统计:
  平均误差: 0.01
  平均几何占比: 4.4%
  预测序列: ['0.0136', '0.0224', '0.0368', '0.0600', '0.0969', '0.1556', '0.2486', '0.3979', '0.6421', '1.0350']
  真实序列: ['0.0131', '0.0213', '0.0344', '0.0557', '0.0902', '0.1459', '0.2361', '0.3820', '0.6180', '1.0000']
PS C:\Users\YeLanyi\Docume
```

## 核心组件

### MetricEvolutionModel (`evolution/model.py`)

主模型类，包含：
- `StateEncoder`: 序列 → 隐状态
- `MetricEncoder`: 隐状态 → 度规张量 g(z)
- `ChristoffelComputer`: 度规 → Christoffel 符号
- `ExternalForce`: 受限外力网络
- `Decoder`: 加速度 → 预测

### 训练策略

1. **动态外力惩罚 (fw)**: 逐渐增加对外力使用的惩罚
2. **Scheduled Sampling (ss)**: 训练时使用模型自己的预测
3. **特征值正则化**: 保证度规正定性
4. **高精度 float64**: 减少累积误差

## 技术细节

### 约束模式 vs 暴论模式

- **约束模式** (`unconstrained=False`): 度规正定，稳定
- **暴论模式** (`unconstrained=True`): 不定号度规，灵活但不稳定

### 度规涌现原理

模型学习一个从隐空间到度规张量的映射：

```
g_ij(z) = L(z) @ L(z)^T + ε * I
```

其中 L(z) 是 Cholesky 分解的下三角矩阵，保证正定性。

## License

Apache-2.0
