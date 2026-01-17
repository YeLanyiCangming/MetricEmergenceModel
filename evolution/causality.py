"""因果结构与干预学习

核心概念：
1. 结构因果模型（SCM）- 变量间的因果关系
2. do算子 - 干预操作
3. 反事实推理 - "如果...会怎样"
4. 因果发现 - 从数据学习因果图

关键特性：
- 区分相关与因果
- 支持干预和反事实查询
- 可解释的因果机制
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Set
from functools import reduce
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 代数数据类型定义
# =============================================================================

class EdgeType(Enum):
    """边类型"""
    CAUSAL = "causal"           # 因果边 X → Y
    CONFOUNDED = "confounded"   # 混杂边 X ←→ Y
    UNDIRECTED = "undirected"   # 无向边 X — Y


@dataclass(frozen=True)
class CausalEdge:
    """因果边（不可变）"""
    source: int
    target: int
    edge_type: EdgeType
    strength: float = 1.0


@dataclass(frozen=True)
class CausalGraph:
    """因果图（不可变）"""
    num_nodes: int
    edges: Tuple[CausalEdge, ...]
    
    def adjacency_matrix(self, device: torch.device = None) -> torch.Tensor:
        """转换为邻接矩阵"""
        device = device or torch.device('cpu')
        adj = torch.zeros(self.num_nodes, self.num_nodes, device=device)
        for edge in self.edges:
            if edge.edge_type == EdgeType.CAUSAL:
                adj[edge.source, edge.target] = edge.strength
        return adj
    
    def parents(self, node: int) -> Set[int]:
        """获取节点的父节点"""
        return {e.source for e in self.edges 
                if e.target == node and e.edge_type == EdgeType.CAUSAL}
    
    def children(self, node: int) -> Set[int]:
        """获取节点的子节点"""
        return {e.target for e in self.edges 
                if e.source == node and e.edge_type == EdgeType.CAUSAL}
    
    def ancestors(self, node: int) -> Set[int]:
        """获取所有祖先"""
        result = set()
        to_visit = list(self.parents(node))
        while to_visit:
            current = to_visit.pop()
            if current not in result:
                result.add(current)
                to_visit.extend(self.parents(current))
        return result


@dataclass(frozen=True)
class Intervention:
    """干预操作"""
    variable: int           # 被干预的变量
    value: torch.Tensor     # 干预值
    
    def apply(self, z: torch.Tensor) -> torch.Tensor:
        """应用干预"""
        z_new = z.clone()
        z_new[:, self.variable] = self.value
        return z_new


@dataclass
class CounterfactualQuery:
    """反事实查询"""
    observation: torch.Tensor      # 观察到的事实
    intervention: Intervention     # 假设的干预
    query_variable: int            # 查询的变量


# =============================================================================
# 结构因果模型
# =============================================================================

class StructuralCausalModel(nn.Module):
    """结构因果模型 (SCM)
    
    X_i = f_i(Pa(X_i), U_i)
    
    其中：
    - Pa(X_i) 是 X_i 的父节点
    - f_i 是结构方程
    - U_i 是外生噪声
    """
    def __init__(self, d: int, hidden_dim: int = 64):
        super().__init__()
        self.d = d
        
        # 每个变量的结构方程
        self.mechanisms = nn.ModuleList([
            CausalMechanism(d, hidden_dim, i) for i in range(d)
        ])
        
        # 可学习的因果图（邻接矩阵）
        # 使用 Gumbel-Softmax 使其可微
        self.adj_logits = nn.Parameter(torch.randn(d, d) * 0.1)
        
        # 掩码对角线（无自环）
        self.register_buffer('diag_mask', 1 - torch.eye(d))
    
    @property
    def adjacency_matrix(self) -> torch.Tensor:
        """可微的邻接矩阵"""
        # Sigmoid 近似二值矩阵
        adj = torch.sigmoid(self.adj_logits) * self.diag_mask
        return adj
    
    def causal_graph(self, threshold: float = 0.5) -> CausalGraph:
        """提取离散因果图"""
        adj = self.adjacency_matrix.detach()
        edges = []
        for i in range(self.d):
            for j in range(self.d):
                if adj[i, j] > threshold:
                    edges.append(CausalEdge(i, j, EdgeType.CAUSAL, adj[i, j].item()))
        return CausalGraph(self.d, tuple(edges))
    
    def forward(
        self, 
        noise: torch.Tensor = None,
        interventions: Dict[int, torch.Tensor] = None
    ) -> torch.Tensor:
        """从噪声生成样本（按拓扑序）
        
        noise: [B, D] 外生噪声
        interventions: {变量索引: 干预值}
        """
        B = noise.shape[0] if noise is not None else 1
        device = self.adj_logits.device
        
        if noise is None:
            noise = torch.randn(B, self.d, device=device)
        
        interventions = interventions or {}
        
        # 按拓扑序生成
        z = torch.zeros(B, self.d, device=device)
        adj = self.adjacency_matrix
        
        # 简单的拓扑排序（假设DAG）
        order = self._topological_sort(adj)
        
        for i in order:
            if i in interventions:
                # 干预：直接设置值
                z[:, i] = interventions[i]
            else:
                # 自然机制
                parent_values = z * adj[:, i].unsqueeze(0)  # 加权父节点
                z[:, i] = self.mechanisms[i](parent_values, noise[:, i])
        
        return z
    
    def _topological_sort(self, adj: torch.Tensor) -> List[int]:
        """拓扑排序"""
        d = adj.shape[0]
        in_degree = (adj > 0.5).sum(dim=0).tolist()
        queue = [i for i in range(d) if in_degree[i] == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            for j in range(d):
                if adj[node, j] > 0.5:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)
        
        # 如果有环，返回简单顺序
        if len(order) != d:
            return list(range(d))
        
        return order
    
    def do(self, z: torch.Tensor, interventions: Dict[int, torch.Tensor]) -> torch.Tensor:
        """do 算子
        
        do(X_i = x) 表示将 X_i 设为 x，切断所有指向 X_i 的边
        """
        # 推断噪声
        noise = self.abduction(z)
        
        # 在干预下重新生成
        return self.forward(noise, interventions)
    
    def abduction(self, z: torch.Tensor) -> torch.Tensor:
        """溯因推理：从观察推断噪声"""
        B = z.shape[0]
        device = z.device
        
        noise = torch.zeros_like(z)
        adj = self.adjacency_matrix
        
        for i in range(self.d):
            parent_values = z * adj[:, i].unsqueeze(0)
            # 逆向推断噪声
            noise[:, i] = self.mechanisms[i].invert(z[:, i], parent_values)
        
        return noise
    
    def counterfactual(
        self, 
        observation: torch.Tensor,
        intervention: Intervention,
        query_variable: int
    ) -> torch.Tensor:
        """反事实查询
        
        "给定观察 Z=z，如果我们干预 X_i=x，Y 会是什么？"
        
        步骤：
        1. 溯因：从 z 推断噪声 U
        2. 干预：切断边，设置 X_i=x
        3. 预测：用推断的 U 在修改后的模型中前向传播
        """
        # 1. 溯因
        noise = self.abduction(observation)
        
        # 2 & 3. 干预并预测
        interventions = {intervention.variable: intervention.value}
        z_cf = self.forward(noise, interventions)
        
        return z_cf[:, query_variable]


class CausalMechanism(nn.Module):
    """单个变量的因果机制
    
    X_i = f_i(Pa(X_i)) + σ_i * U_i
    """
    def __init__(self, d: int, hidden_dim: int, variable_idx: int):
        super().__init__()
        self.variable_idx = variable_idx
        
        # 结构方程网络
        self.f = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # 噪声标准差（可学习）
        self.log_sigma = nn.Parameter(torch.tensor(0.0))
    
    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)
    
    def forward(self, parent_values: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        parent_values: [B, D] 父节点值（其他维度为0）
        noise: [B] 外生噪声
        """
        deterministic = self.f(parent_values).squeeze(-1)
        return deterministic + self.sigma * noise
    
    def invert(self, value: torch.Tensor, parent_values: torch.Tensor) -> torch.Tensor:
        """逆向推断噪声"""
        deterministic = self.f(parent_values).squeeze(-1)
        return (value - deterministic) / (self.sigma + 1e-6)


# =============================================================================
# 因果发现
# =============================================================================

class CausalDiscovery(nn.Module):
    """因果发现模块
    
    从数据中学习因果图结构
    
    使用：
    1. NOTEARS 约束（DAG约束）
    2. 稀疏性正则
    3. 干预数据（如果有）
    """
    def __init__(self, d: int, hidden_dim: int = 64):
        super().__init__()
        self.d = d
        
        # 边权重参数
        self.edge_weights = nn.Parameter(torch.randn(d, d) * 0.01)
        
        # 边检测网络（预测是否存在边）
        self.edge_detector = nn.Sequential(
            nn.Linear(d * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        # 无自环
        self.register_buffer('diag_mask', 1 - torch.eye(d))
    
    def adjacency_matrix(self, z: torch.Tensor = None) -> torch.Tensor:
        """计算邻接矩阵"""
        if z is None:
            # 静态邻接矩阵
            return torch.sigmoid(self.edge_weights) * self.diag_mask
        
        # 动态邻接矩阵（基于输入）
        B, D = z.shape
        adj = torch.zeros(B, D, D, device=z.device)
        
        for i in range(D):
            for j in range(D):
                if i != j:
                    pair = torch.cat([z[:, i:i+1], z[:, j:j+1]], dim=-1)
                    adj[:, i, j] = self.edge_detector(pair).squeeze(-1)
        
        return adj
    
    def dag_constraint(self) -> torch.Tensor:
        """NOTEARS DAG 约束
        
        h(W) = tr(e^{W ⊙ W}) - d = 0
        
        当且仅当 W 是 DAG 时成立
        """
        W = torch.sigmoid(self.edge_weights) * self.diag_mask
        W_squared = W * W
        
        # 矩阵指数的迹
        # tr(e^A) = Σ_i e^{λ_i}
        expm = torch.matrix_exp(W_squared)
        h = torch.trace(expm) - self.d
        
        return h
    
    def sparsity_loss(self) -> torch.Tensor:
        """稀疏性损失"""
        W = torch.sigmoid(self.edge_weights) * self.diag_mask
        return W.abs().mean()
    
    def intervention_loss(
        self, 
        z_obs: torch.Tensor,
        z_int: torch.Tensor,
        intervention_idx: int
    ) -> torch.Tensor:
        """干预一致性损失
        
        干预 X_i 应该只影响 X_i 的后代
        """
        adj = self.adjacency_matrix()
        
        # 找到后代
        descendants = self._find_descendants(adj, intervention_idx)
        non_descendants_mask = torch.ones(self.d, device=adj.device)
        for d in descendants:
            non_descendants_mask[d] = 0
        non_descendants_mask[intervention_idx] = 0
        
        # 非后代应该保持不变
        diff = (z_obs - z_int) * non_descendants_mask
        return diff.pow(2).mean()
    
    def _find_descendants(self, adj: torch.Tensor, node: int) -> Set[int]:
        """找到所有后代"""
        descendants = set()
        to_visit = [j for j in range(self.d) if adj[node, j] > 0.5]
        
        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                for j in range(self.d):
                    if adj[current, j] > 0.5:
                        to_visit.append(j)
        
        return descendants
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回: (邻接矩阵, DAG约束值)
        """
        adj = self.adjacency_matrix(z)
        h = self.dag_constraint()
        return adj, h


# =============================================================================
# 因果注意力
# =============================================================================

class CausalAttention(nn.Module):
    """因果注意力机制
    
    使用学习的因果图约束注意力权重
    只允许因果方向的信息流动
    """
    def __init__(self, d: int, num_heads: int = 4):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        
        # 因果发现
        self.causal_discovery = CausalDiscovery(d)
    
    def forward(
        self, 
        x: torch.Tensor, 
        causal_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, D]
        causal_mask: [D, D] 因果掩码（可选）
        
        返回: (output, 学习的因果图)
        """
        B, L, D = x.shape
        
        # 投影
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用因果掩码
        if causal_mask is None:
            # 学习因果掩码
            z_mean = x.mean(dim=1)  # [B, D]
            causal_adj, dag_h = self.causal_discovery(z_mean)
            
            # 将因果邻接矩阵转换为注意力掩码
            # 只允许父节点→子节点的注意力
            causal_mask = causal_adj.mean(dim=0)  # [D, D]
        else:
            dag_h = torch.tensor(0.0, device=x.device)
        
        # 扩展掩码到序列长度
        # 这里简化处理：使用相同的因果结构对所有位置
        mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, D, D]
        
        # 转换为注意力掩码（非因果边设为 -inf）
        attn_mask = torch.where(
            mask > 0.5,
            torch.zeros_like(mask),
            torch.full_like(mask, float('-inf'))
        )
        
        # 如果 L != D，需要调整
        if L != D:
            attn_mask = torch.zeros(1, 1, L, L, device=x.device)
            attn_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), 1)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        
        scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1)
        
        # 应用注意力
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).reshape(B, L, D)
        output = self.o_proj(output)
        
        return output, causal_mask


# =============================================================================
# 因果演化模块
# =============================================================================

class CausalEvolution(nn.Module):
    """因果演化模块
    
    结合：
    1. 结构因果模型
    2. 因果发现
    3. 干预训练
    
    确保演化遵循因果结构
    """
    def __init__(self, d: int, hidden_dim: int = 128):
        super().__init__()
        self.d = d
        
        # 结构因果模型
        self.scm = StructuralCausalModel(d, hidden_dim)
        
        # 因果注意力
        self.causal_attention = CausalAttention(d)
        
        # 演化网络（受因果约束）
        self.evolution_net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d),
        )
    
    def forward(
        self, 
        z: torch.Tensor,
        intervention: Intervention = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        z: [B, D] 输入状态
        intervention: 可选的干预
        
        返回: (演化后的状态, 信息字典)
        """
        B, D = z.shape
        
        # 如果有干预，应用 do 算子
        if intervention is not None:
            z = self.scm.do(z, {intervention.variable: intervention.value})
        
        # 因果注意力
        z_attended, causal_graph = self.causal_attention(z.unsqueeze(1))
        z_attended = z_attended.squeeze(1)
        
        # 演化
        z_evolved = z + self.evolution_net(z_attended)
        
        info = {
            'causal_graph': causal_graph,
            'dag_constraint': self.scm.adjacency_matrix.sum(),
        }
        
        return z_evolved, info
    
    def counterfactual_evolution(
        self,
        z_observed: torch.Tensor,
        intervention: Intervention,
        num_steps: int = 5
    ) -> torch.Tensor:
        """反事实演化
        
        "如果在观察到z后，我们干预了X_i，系统会如何演化？"
        """
        # 反事实查询
        z_cf = self.scm.counterfactual(
            z_observed,
            intervention,
            query_variable=slice(None)  # 查询所有变量
        )
        
        # 从反事实状态开始演化
        z = z_cf
        trajectory = [z]
        
        for _ in range(num_steps):
            z, _ = self.forward(z)
            trajectory.append(z)
        
        return torch.stack(trajectory, dim=1)
    
    def intervention_training_loss(
        self,
        z: torch.Tensor,
        intervention_idx: int
    ) -> torch.Tensor:
        """干预训练损失
        
        鼓励模型学习正确的因果结构
        """
        B = z.shape[0]
        device = z.device
        
        # 观察数据
        z_obs, _ = self.forward(z)
        
        # 干预数据
        intervention_value = torch.randn(B, device=device)
        intervention = Intervention(intervention_idx, intervention_value)
        z_int, _ = self.forward(z, intervention)
        
        # 计算干预一致性损失
        causal_adj = self.scm.adjacency_matrix
        
        # 非后代应该保持不变
        descendants = self._find_descendants(causal_adj, intervention_idx)
        non_descendants_mask = torch.ones(self.d, device=device)
        for d in descendants:
            non_descendants_mask[d] = 0
        non_descendants_mask[intervention_idx] = 0
        
        diff = (z_obs - z_int) * non_descendants_mask
        return diff.pow(2).mean()
    
    def _find_descendants(self, adj: torch.Tensor, node: int) -> Set[int]:
        """找到所有后代"""
        descendants = set()
        to_visit = [j for j in range(self.d) if adj[node, j] > 0.5]
        
        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                for j in range(self.d):
                    if adj[current, j] > 0.5:
                        to_visit.append(j)
        
        return descendants


# =============================================================================
# 高阶组合子
# =============================================================================

def do_calculus_adjustment(
    scm: StructuralCausalModel,
    treatment: int,
    outcome: int,
    z: torch.Tensor
) -> torch.Tensor:
    """do-calculus 调整公式
    
    P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) P(Z=z)
    
    用于计算因果效应
    """
    # 简化实现：直接使用干预
    interventions = {treatment: z[:, treatment]}
    z_do = scm.forward(scm.abduction(z), interventions)
    return z_do[:, outcome]


def average_treatment_effect(
    scm: StructuralCausalModel,
    treatment: int,
    outcome: int,
    z: torch.Tensor,
    treatment_value_1: float = 1.0,
    treatment_value_0: float = 0.0
) -> torch.Tensor:
    """平均处理效应 (ATE)
    
    ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
    """
    B = z.shape[0]
    device = z.device
    
    # do(X=1)
    z_do_1 = scm.forward(
        scm.abduction(z),
        {treatment: torch.full((B,), treatment_value_1, device=device)}
    )
    
    # do(X=0)
    z_do_0 = scm.forward(
        scm.abduction(z),
        {treatment: torch.full((B,), treatment_value_0, device=device)}
    )
    
    ate = z_do_1[:, outcome].mean() - z_do_0[:, outcome].mean()
    return ate


def causal_mediation_analysis(
    scm: StructuralCausalModel,
    treatment: int,
    mediator: int,
    outcome: int,
    z: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """因果中介分析
    
    分解：总效应 = 直接效应 + 间接效应
    
    返回：
    - total_effect: 总效应
    - direct_effect: 直接效应 (X → Y)
    - indirect_effect: 间接效应 (X → M → Y)
    """
    B = z.shape[0]
    device = z.device
    
    noise = scm.abduction(z)
    
    # 总效应
    z_do_1 = scm.forward(noise, {treatment: torch.ones(B, device=device)})
    z_do_0 = scm.forward(noise, {treatment: torch.zeros(B, device=device)})
    total_effect = z_do_1[:, outcome].mean() - z_do_0[:, outcome].mean()
    
    # 自然直接效应 (NDE)
    # 固定中介在 do(X=0) 下的值，然后改变 X
    m_under_0 = z_do_0[:, mediator]
    z_nde = scm.forward(noise, {treatment: torch.ones(B, device=device), mediator: m_under_0})
    nde = z_nde[:, outcome].mean() - z_do_0[:, outcome].mean()
    
    # 自然间接效应 (NIE) = 总效应 - NDE
    nie = total_effect - nde
    
    return {
        'total_effect': total_effect,
        'direct_effect': nde,
        'indirect_effect': nie,
    }
