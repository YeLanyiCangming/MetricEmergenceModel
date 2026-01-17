"""Evolution Package - 度规涌现演化框架

核心范式（第一性原理）：
  数据 (x) → 抽象状态 (z) → 度规 g(z) → 联络 Γ (涌现) → 运动法则 (涌现)

三层架构：
  1. 感知层：学习流形的"本体"——度规张量
  2. 涌现层：从度规自动推导联络（Christoffel符号）
  3. 法则层：测地线加速度 + 外力 → 完整运动

不硬编码规则，但硬编码几何原理（微分几何的语言）
"""

__version__ = "0.4.0"

from .model import (
    MetricEvolutionModel,
    StateEncoder,
    MetricEncoder,
    ChristoffelComputer,
    GeodesicAcceleration,
    ExternalForce,
    ProbabilisticDecoder,
    RMSNorm,
)

__all__ = [
    '__version__',
    'MetricEvolutionModel',
    'StateEncoder',
    'MetricEncoder',
    'ChristoffelComputer',
    'GeodesicAcceleration',
    'ExternalForce',
    'ProbabilisticDecoder',
    'RMSNorm',
]
