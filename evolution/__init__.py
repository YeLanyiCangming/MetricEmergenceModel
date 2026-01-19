"""Evolution Package - 度规涌现演化框架

核心范式（第一性原理）：
  数据 (x) → 抽象状态 (z) → 度规 g(z) → 联络 Γ (涌现) → 运动法则 (涌现)

世界动力学 (N·ẍ + D·ẋ + K·x = F(t))：
  - N = 质量矩阵（惯性）
  - D = 阻尼矩阵（耗散）
  - K = 刚度矩阵 = diag(ω²) + β·L（本地频率 + 图耦合）
  - F(t) = 外部驱动
"""

__version__ = "0.5.0"

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

from .world_dynamics import (
    WorldDynamics,
    WorldDynamicsLayer,
    build_world_dynamics,
    solve_conservative_modes,
    solve_damped_modes,
)

__all__ = [
    '__version__',
    # 度规演化模型
    'MetricEvolutionModel',
    'StateEncoder',
    'MetricEncoder',
    'ChristoffelComputer',
    'GeodesicAcceleration',
    'ExternalForce',
    'ProbabilisticDecoder',
    'RMSNorm',
    # 世界动力学
    'WorldDynamics',
    'WorldDynamicsLayer',
    'build_world_dynamics',
    'solve_conservative_modes',
    'solve_damped_modes',
]
