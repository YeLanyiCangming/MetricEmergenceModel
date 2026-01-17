"""对称性与等变神经网络

核心数学结构：
1. 李群 G - 对称变换群（平移、旋转、缩放）
2. 群表示 ρ - 群在向量空间上的作用
3. 等变映射 - f(g·x) = g·f(x)
4. 不变量 - f(g·x) = f(x)

关键特性：
- 等变性：保持物理定律的形式不变性
- 群卷积：利用对称性减少参数
- 诺特定理：对称性 ↔ 守恒量
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Protocol
from functools import reduce
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 代数数据类型定义
# =============================================================================

@dataclass(frozen=True)
class GroupElement:
    """群元素（不可变）
    
    表示为矩阵形式：g ∈ GL(n)
    """
    matrix: torch.Tensor  # [D, D] 变换矩阵
    
    def __matmul__(self, other: 'GroupElement') -> 'GroupElement':
        """群乘法"""
        return GroupElement(self.matrix @ other.matrix)
    
    def inverse(self) -> 'GroupElement':
        """群逆"""
        return GroupElement(torch.linalg.inv(self.matrix))
    
    def act(self, x: torch.Tensor) -> torch.Tensor:
        """作用在向量上"""
        return x @ self.matrix.T


@dataclass(frozen=True)
class LieAlgebraElement:
    """李代数元素（无穷小生成元）"""
    generator: torch.Tensor  # [D, D] 生成元矩阵
    
    def exp(self) -> GroupElement:
        """指数映射：李代数 → 李群"""
        return GroupElement(torch.matrix_exp(self.generator))
    
    def bracket(self, other: 'LieAlgebraElement') -> 'LieAlgebraElement':
        """李括号 [X, Y] = XY - YX"""
        return LieAlgebraElement(
            self.generator @ other.generator - other.generator @ self.generator
        )


# =============================================================================
# 对称群定义
# =============================================================================

class SymmetryGroup(ABC):
    """对称群抽象基类"""
    
    @abstractmethod
    def generators(self, d: int, device: torch.device) -> List[torch.Tensor]:
        """返回李代数生成元"""
        ...
    
    @abstractmethod
    def sample(self, batch_size: int, d: int, device: torch.device) -> torch.Tensor:
        """采样群元素"""
        ...


class TranslationGroup(SymmetryGroup):
    """平移群 R^n"""
    
    def generators(self, d: int, device: torch.device) -> List[torch.Tensor]:
        """平移生成元（d个方向）"""
        gens = []
        for i in range(d):
            g = torch.zeros(d + 1, d + 1, device=device)
            g[i, d] = 1.0  # 在齐次坐标下
            gens.append(g)
        return gens
    
    def sample(self, batch_size: int, d: int, device: torch.device) -> torch.Tensor:
        """采样随机平移"""
        t = torch.randn(batch_size, d, device=device)
        # 构造齐次变换矩阵
        T = torch.eye(d + 1, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        T[:, :d, d] = t
        return T


class RotationGroup(SymmetryGroup):
    """旋转群 SO(n)"""
    
    def generators(self, d: int, device: torch.device) -> List[torch.Tensor]:
        """旋转生成元（反对称矩阵）
        
        对于SO(n)，有 n(n-1)/2 个生成元
        """
        gens = []
        for i in range(d):
            for j in range(i + 1, d):
                g = torch.zeros(d, d, device=device)
                g[i, j] = 1.0
                g[j, i] = -1.0  # 反对称
                gens.append(g)
        return gens
    
    def sample(self, batch_size: int, d: int, device: torch.device) -> torch.Tensor:
        """采样随机旋转（QR分解法）"""
        # 随机矩阵
        A = torch.randn(batch_size, d, d, device=device)
        # QR分解得到正交矩阵
        Q, R = torch.linalg.qr(A)
        # 确保行列式为1（SO(n)而非O(n)）
        det = torch.linalg.det(Q)
        Q = Q * det.sign().unsqueeze(-1).unsqueeze(-1)
        return Q


class ScaleGroup(SymmetryGroup):
    """缩放群 R_+"""
    
    def generators(self, d: int, device: torch.device) -> List[torch.Tensor]:
        """缩放生成元（单位矩阵）"""
        return [torch.eye(d, device=device)]
    
    def sample(self, batch_size: int, d: int, device: torch.device) -> torch.Tensor:
        """采样随机缩放"""
        log_s = torch.randn(batch_size, 1, device=device) * 0.5
        s = torch.exp(log_s)
        return s.unsqueeze(-1) * torch.eye(d, device=device).unsqueeze(0)


class SE3Group(SymmetryGroup):
    """特殊欧几里得群 SE(3) = SO(3) ⋉ R^3
    
    刚体运动群：旋转 + 平移
    """
    
    def generators(self, d: int = 3, device: torch.device = None) -> List[torch.Tensor]:
        """SE(3)生成元：3个旋转 + 3个平移"""
        device = device or torch.device('cpu')
        gens = []
        
        # 旋转生成元（嵌入在4x4齐次矩阵中）
        for i in range(3):
            for j in range(i + 1, 3):
                g = torch.zeros(4, 4, device=device)
                g[i, j] = 1.0
                g[j, i] = -1.0
                gens.append(g)
        
        # 平移生成元
        for i in range(3):
            g = torch.zeros(4, 4, device=device)
            g[i, 3] = 1.0
            gens.append(g)
        
        return gens
    
    def sample(self, batch_size: int, d: int = 3, device: torch.device = None) -> torch.Tensor:
        """采样SE(3)元素"""
        device = device or torch.device('cpu')
        
        # 旋转部分
        R = RotationGroup().sample(batch_size, d, device)
        
        # 平移部分
        t = torch.randn(batch_size, d, device=device)
        
        # 构造齐次变换矩阵
        T = torch.eye(d + 1, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        T[:, :d, :d] = R
        T[:, :d, d] = t
        
        return T


# =============================================================================
# 等变层
# =============================================================================

class EquivariantLinear(nn.Module):
    """等变线性层
    
    保证：f(ρ_in(g)x) = ρ_out(g)f(x)
    
    通过限制权重矩阵结构实现等变性
    """
    def __init__(
        self, 
        d_in: int, 
        d_out: int, 
        group: SymmetryGroup,
        num_generators: int = None
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.group = group
        
        # 获取生成元
        device = torch.device('cpu')  # 初始化时用CPU
        self.gens = group.generators(min(d_in, d_out), device)
        self.num_gens = len(self.gens) if num_generators is None else num_generators
        
        # 等变权重：W = Σ_i α_i G_i
        # 其中 G_i 是与群生成元对易的基
        self.coefficients = nn.Parameter(torch.randn(self.num_gens) * 0.1)
        
        # 不变偏置（可选）
        self.bias = nn.Parameter(torch.zeros(d_out))
        
        # 一般线性部分（对于高维）
        if d_in != d_out or self.num_gens == 0:
            self.general_weight = nn.Parameter(torch.randn(d_out, d_in) * 0.01)
        else:
            self.general_weight = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, d_in]
        返回: [B, d_out]
        """
        device = x.device
        
        if self.general_weight is not None:
            return F.linear(x, self.general_weight, self.bias)
        
        # 构造等变权重
        d = min(self.d_in, self.d_out)
        W = torch.zeros(d, d, device=device)
        
        for i, (coef, gen) in enumerate(zip(self.coefficients, self.gens)):
            gen = gen.to(device)
            W = W + coef * gen[:d, :d]
        
        # 添加单位矩阵分量（不变部分）
        W = W + torch.eye(d, device=device)
        
        # 如果维度不匹配，进行调整
        if self.d_in != self.d_out:
            full_W = torch.zeros(self.d_out, self.d_in, device=device)
            full_W[:d, :d] = W
            W = full_W
        
        return F.linear(x, W, self.bias)


class EquivariantMLP(nn.Module):
    """等变MLP
    
    多层等变网络，使用等变激活函数
    """
    def __init__(
        self, 
        d: int, 
        hidden_dim: int,
        group: SymmetryGroup,
        num_layers: int = 2
    ):
        super().__init__()
        self.d = d
        
        layers = []
        dims = [d] + [hidden_dim] * (num_layers - 1) + [d]
        
        for i in range(len(dims) - 1):
            layers.append(EquivariantLinear(dims[i], dims[i+1], group))
            if i < len(dims) - 2:
                layers.append(EquivariantActivation())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EquivariantActivation(nn.Module):
    """等变激活函数
    
    对于向量，使用范数门控：
    φ(v) = σ(||v||) * v / ||v||
    
    这保持了旋转等变性
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) + self.eps
        gate = torch.sigmoid(norm)
        return gate * x


# =============================================================================
# 群卷积
# =============================================================================

class GroupConvolution(nn.Module):
    """群卷积层
    
    (f * κ)(g) = ∫_G f(h) κ(h^{-1}g) dh
    
    通过离散化群实现
    """
    def __init__(
        self, 
        d_in: int, 
        d_out: int, 
        group: SymmetryGroup,
        num_samples: int = 8
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.group = group
        self.num_samples = num_samples
        
        # 核函数（在群上的函数）
        self.kernel = nn.Parameter(torch.randn(num_samples, d_out, d_in) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, d_in]
        返回: [B, d_out]
        """
        B = x.shape[0]
        device = x.device
        
        # 采样群元素
        g_samples = self.group.sample(self.num_samples, self.d_in, device)
        
        # 群卷积
        output = torch.zeros(B, self.d_out, device=device)
        
        for i in range(self.num_samples):
            # 变换输入
            g = g_samples[i]
            if g.shape[0] > self.d_in:  # 齐次坐标
                g = g[:self.d_in, :self.d_in]
            x_transformed = x @ g.T
            
            # 应用核
            output = output + F.linear(x_transformed, self.kernel[i])
        
        return output / self.num_samples


# =============================================================================
# 不变量提取
# =============================================================================

class InvariantExtractor(nn.Module):
    """不变量提取器
    
    提取对群作用不变的特征
    
    对于旋转群：范数、角度、内积
    对于平移群：差值
    """
    def __init__(self, d: int, num_invariants: int = None):
        super().__init__()
        self.d = d
        self.num_invariants = num_invariants or d
        
        # 可学习的不变量基
        self.invariant_basis = nn.Parameter(torch.randn(self.num_invariants, d) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]
        返回: [B, num_invariants] 不变特征
        """
        # 范数（旋转不变）
        norm = x.norm(dim=-1, keepdim=True)
        
        # 与不变基的内积（旋转等变但平移破缺）
        inner_products = x @ self.invariant_basis.T
        
        # 组合不变特征
        invariants = torch.cat([norm, inner_products], dim=-1)
        
        return invariants[:, :self.num_invariants]


# =============================================================================
# 对称性生成元学习
# =============================================================================

class LearnedSymmetryGenerator(nn.Module):
    """可学习的对称性生成元
    
    学习数据中隐含的对称性
    
    生成元 G 满足：
    - 反对称（旋转）: G + G^T = 0
    - 迹为零（体积保持）: tr(G) = 0
    """
    def __init__(self, d: int, num_generators: int = 4, generator_type: str = 'general'):
        super().__init__()
        self.d = d
        self.num_generators = num_generators
        self.generator_type = generator_type
        
        if generator_type == 'antisymmetric':
            # 反对称生成元（旋转类）
            # 只学习上三角部分
            self.upper_params = nn.Parameter(
                torch.randn(num_generators, d * (d - 1) // 2) * 0.1
            )
        elif generator_type == 'traceless':
            # 无迹生成元（体积保持）
            self.params = nn.Parameter(torch.randn(num_generators, d, d) * 0.1)
        else:
            # 一般生成元
            self.params = nn.Parameter(torch.randn(num_generators, d, d) * 0.1)
    
    def forward(self) -> torch.Tensor:
        """返回生成元矩阵 [num_gen, D, D]"""
        if self.generator_type == 'antisymmetric':
            # 构造反对称矩阵
            G = torch.zeros(self.num_generators, self.d, self.d, device=self.upper_params.device)
            idx = 0
            for i in range(self.d):
                for j in range(i + 1, self.d):
                    G[:, i, j] = self.upper_params[:, idx]
                    G[:, j, i] = -self.upper_params[:, idx]
                    idx += 1
            return G
        
        elif self.generator_type == 'traceless':
            # 投影到无迹子空间
            G = self.params
            trace = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1, keepdim=True)
            G = G - (trace / self.d).unsqueeze(-1) * torch.eye(self.d, device=G.device)
            return G
        
        else:
            return self.params
    
    def exp_action(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """指数映射作用
        
        x: [B, D] 输入
        alpha: [B, num_gen] 各生成元的强度
        
        返回: exp(Σ α_i G_i) · x
        """
        G = self()  # [num_gen, D, D]
        
        # 组合生成元
        combined_G = torch.einsum('bg,gij->bij', alpha, G)  # [B, D, D]
        
        # 指数映射（使用泰勒展开）
        exp_G = matrix_exp_taylor(combined_G, order=4)
        
        # 作用在x上
        return torch.bmm(exp_G, x.unsqueeze(-1)).squeeze(-1)


def matrix_exp_taylor(A: torch.Tensor, order: int = 4) -> torch.Tensor:
    """矩阵指数的泰勒展开
    
    exp(A) ≈ I + A + A²/2! + A³/3! + ...
    """
    B, D, _ = A.shape
    device = A.device
    
    result = torch.eye(D, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    A_power = A.clone()
    factorial = 1.0
    
    for n in range(1, order + 1):
        factorial *= n
        result = result + A_power / factorial
        A_power = torch.bmm(A_power, A)
    
    return result


# =============================================================================
# 等变注意力
# =============================================================================

class EquivariantAttention(nn.Module):
    """等变注意力机制
    
    保持对旋转的等变性：
    - Q, K, V 都是等变映射
    - 注意力权重使用不变量计算
    """
    def __init__(self, d: int, num_heads: int = 4):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        
        # 等变Q, K, V投影
        self.q_proj = EquivariantLinear(d, d, RotationGroup())
        self.k_proj = EquivariantLinear(d, d, RotationGroup())
        self.v_proj = EquivariantLinear(d, d, RotationGroup())
        
        # 输出投影
        self.o_proj = EquivariantLinear(d, d, RotationGroup())
        
        # 不变量提取（用于注意力权重）
        self.invariant_q = InvariantExtractor(self.head_dim)
        self.invariant_k = InvariantExtractor(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, L, D]
        mask: [L, L] 因果掩码
        """
        B, L, D = x.shape
        
        # 等变投影
        Q = self.q_proj(x.view(-1, D)).view(B, L, self.num_heads, self.head_dim)
        K = self.k_proj(x.view(-1, D)).view(B, L, self.num_heads, self.head_dim)
        V = self.v_proj(x.view(-1, D)).view(B, L, self.num_heads, self.head_dim)
        
        # 转置
        Q = Q.transpose(1, 2)  # [B, H, L, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 使用不变量计算注意力（而非简单点积）
        # 这保持了旋转不变性
        Q_inv = self.invariant_q(Q.reshape(-1, self.head_dim)).view(B, self.num_heads, L, -1)
        K_inv = self.invariant_k(K.reshape(-1, self.head_dim)).view(B, self.num_heads, L, -1)
        
        # 注意力分数
        scores = torch.matmul(Q_inv, K_inv.transpose(-2, -1)) / math.sqrt(Q_inv.shape[-1])
        
        if mask is not None:
            scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        
        # 应用注意力
        output = torch.matmul(attn, V)
        
        # 合并头
        output = output.transpose(1, 2).reshape(B, L, D)
        
        return self.o_proj(output.view(-1, D)).view(B, L, D)


# =============================================================================
# 完整的等变演化模块
# =============================================================================

class EquivariantEvolution(nn.Module):
    """等变演化模块
    
    结合：
    1. 可学习的对称性生成元
    2. 等变神经网络
    3. 不变量提取
    
    保证物理定律的形式不变性
    """
    def __init__(
        self, 
        d: int, 
        hidden_dim: int = 128,
        num_generators: int = 4,
        generator_type: str = 'general'
    ):
        super().__init__()
        self.d = d
        
        # 对称性生成元
        self.generators = LearnedSymmetryGenerator(d, num_generators, generator_type)
        
        # 强度提取器（从输入中提取各生成元的强度）
        self.strength_extractor = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_generators),
        )
        
        # 等变MLP
        self.equivariant_mlp = EquivariantMLP(d, hidden_dim, RotationGroup())
        
        # 不变量提取
        self.invariant_extractor = InvariantExtractor(d)
    
    def forward(
        self, 
        z: torch.Tensor, 
        return_invariants: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [B, D] 输入状态
        
        返回: 演化后的状态 [B, D]
        """
        # 提取强度
        alpha = self.strength_extractor(z)  # [B, num_gen]
        
        # 对称性变换
        z_transformed = self.generators.exp_action(z, alpha)
        
        # 等变MLP
        z_evolved = self.equivariant_mlp(z_transformed)
        
        if return_invariants:
            invariants = self.invariant_extractor(z_evolved)
            return z_evolved, invariants
        
        return z_evolved
    
    def get_generators(self) -> torch.Tensor:
        """获取学习到的生成元"""
        return self.generators()


# =============================================================================
# 高阶组合子
# =============================================================================

def orbit(group: SymmetryGroup, x: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
    """计算点的轨道
    
    Orb(x) = {g·x | g ∈ G}
    """
    d = x.shape[-1]
    device = x.device
    
    g_samples = group.sample(num_samples, d, device)
    
    orbits = []
    for i in range(num_samples):
        g = g_samples[i]
        if g.shape[0] > d:
            g = g[:d, :d]
        orbits.append(x @ g.T)
    
    return torch.stack(orbits, dim=0)


def average_over_group(
    f: Callable[[torch.Tensor], torch.Tensor],
    group: SymmetryGroup,
    x: torch.Tensor,
    num_samples: int = 10
) -> torch.Tensor:
    """群平均
    
    〈f〉_G(x) = (1/|G|) Σ_g f(g·x)
    
    这产生不变函数
    """
    d = x.shape[-1]
    device = x.device
    
    g_samples = group.sample(num_samples, d, device)
    
    result = 0
    for i in range(num_samples):
        g = g_samples[i]
        if g.shape[0] > d:
            g = g[:d, :d]
        x_transformed = x @ g.T
        result = result + f(x_transformed)
    
    return result / num_samples


def symmetry_loss(
    f: nn.Module,
    group: SymmetryGroup,
    x: torch.Tensor,
    num_samples: int = 5
) -> torch.Tensor:
    """对称性损失
    
    L_sym = E_g ||f(g·x) - g·f(x)||²
    
    鼓励等变性
    """
    d = x.shape[-1]
    device = x.device
    
    g_samples = group.sample(num_samples, d, device)
    f_x = f(x)
    
    loss = 0
    for i in range(num_samples):
        g = g_samples[i]
        if g.shape[0] > d:
            g = g[:d, :d]
        
        # f(g·x)
        x_transformed = x @ g.T
        f_gx = f(x_transformed)
        
        # g·f(x)
        gf_x = f_x @ g.T
        
        loss = loss + (f_gx - gf_x).pow(2).mean()
    
    return loss / num_samples
