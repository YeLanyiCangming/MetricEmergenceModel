"""演化方程模型 - 联络版本 (Connection)

核心思想：联络（Connection）
- 流形上每个点有自己的切空间
- 联络定义了如何在不同点之间"平行移动"向量
- 协变导数会随着流形的几何自动调整

问题诊断：
- 当前 Encoder：输入 → 孤立的点
- [1,1] → z₁，[50,60] → z₂，它们不知道彼此关系
- 模型只在 z₁ 附近学到流形，z₂ 是"荒野"

联络的解决：
- Encoder → (z, Γ)：位置 + 联络信息
- Γ 告诉模型：在这个点，演化方向应该如何"校正"
- 即使落在陌生位置，联络也能指引正确的演化方向
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


# =============================================================================
# 核心 1: 联络编码器 — 输入 → (位置, 联络)
# =============================================================================

class ConnectionEncoder(nn.Module):
    """
    联络编码器：输入 → (z, connection)
    
    核心改进：联络基于**曲率**，而非速度
    
    - 速度 dz = z[i] - z[i-1]：依赖坐标（绝对值）
    - 曲率 d²z = dz[i] - dz[i-1]：坐标无关（几何不变量）
    
    这是规范不变性的核心：
    - [1,1,2,3,5...] 和 [50,60,110,170...] 的 dz 不同
    - 但它们的 d²z 模式相同
    """
    def __init__(self, input_dim, d):
        super().__init__()
        self.d = d
        
        # 输入编码：数值 → 向量
        self.input_proj = nn.Linear(input_dim, d)
        
        # 位置编码
        self.z_proj = nn.Linear(d, d)
        
        # 曲率编码器：提取几何不变量
        self.curvature_encoder = nn.Linear(d, d)
        
        # 联络生成：基于曲率
        self.connection_proj = nn.Linear(d * 2, d)
        
    def forward(self, x):
        """
        x: [B, L] 或 [B, L, input_dim] 输入序列
        返回: 
            z: [B, L, D] 位置
            connection: [B, L, D] 联络（基于曲率）
        """
        # 处理输入维度
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, L] → [B, L, 1]
        
        # 输入编码
        x = self.input_proj(x)  # [B, L, D]
        z = self.z_proj(x)
        
        # 计算速度：dz = z[i] - z[i-1]
        z_prev1 = torch.cat([z[:, :1], z[:, :-1]], dim=1)
        dz = z - z_prev1
        
        # 计算曲率：d²z = dz[i] - dz[i-1]
        dz_prev = torch.cat([dz[:, :1], dz[:, :-1]], dim=1)
        d2z = dz - dz_prev  # 曲率（二阶差分）
        
        # 曲率归一化：提取"方向"而非"大小"（坐标无关）
        d2z_norm = d2z / (d2z.norm(dim=-1, keepdim=True) + 1e-6)
        
        # 曲率特征
        curvature_feat = self.curvature_encoder(d2z_norm)
        
        # 联络：基于位置 + 曲率
        connection = self.connection_proj(torch.cat([z, curvature_feat], dim=-1))
        
        return z, connection


# =============================================================================
# 核心 2: 协变向量场 — dz 会被联络校正
# =============================================================================

class CovariantVectorField(nn.Module):
    """
    协变向量场：∇z = F(z) + Γ(connection, dz)
    
    普通向量场：dz = F(z)
    协变向量场：dz = F(z) + 联络校正
    
    物理意义：
    - F(z) 给出"局部"演化方向
    - 联络校正让演化"适应流形的几何"
    """
    def __init__(self, d):
        super().__init__()
        # 局部向量场
        self.f = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.SiLU(),
            nn.Linear(d * 2, d),
        )
        
        # 联络校正：connection × dz → 校正量
        # Christoffel 符号的学习版本
        self.gamma = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.Tanh(),  # 限制校正幅度
            nn.Linear(d, d),
        )
        
        self.dt = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, z, connection):
        """
        z: 位置 [B, L, D]
        connection: 联络 [B, L, D]
        返回: 协变演化方向 [B, L, D]
        """
        # 局部演化方向
        dz_local = self.f(z)
        
        # 联络校正：根据 connection 和 dz_local 计算校正
        correction = self.gamma(torch.cat([connection, dz_local], dim=-1))
        
        # 协变演化 = 局部 + 校正
        dz = dz_local + correction
        
        return self.dt * dz


# =============================================================================
# 核心 3: 协变注意力 — 在联络指导下寻找模式
# =============================================================================

class CovariantAttention(nn.Module):
    """
    协变注意力：基于联络的模式匹配
    
    普通注意力：Q(z) @ K(z)
    协变注意力：Q(z, connection) @ K(z, connection)
    
    关键：联络参与注意力计算
    这样注意力不只看"位置相似"，还看"演化方向相似"
    """
    def __init__(self, d, h=4):
        super().__init__()
        self.h, self.hd = h, d // h
        
        # Q, K 同时考虑位置和联络
        self.q = nn.Linear(d * 2, d, bias=False)
        self.k = nn.Linear(d * 2, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        
        self.temp = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, z, connection, mask=None):
        """
        z: 位置 [B, L, D]
        connection: 联络 [B, L, D]
        返回: 注意力输出 [B, L, D]
        """
        B, L, D = z.shape
        
        # Q, K 基于 (位置, 联络)
        zc = torch.cat([z, connection], dim=-1)
        Q = self.q(zc).view(B, L, self.h, self.hd).transpose(1, 2)
        K = self.k(zc).view(B, L, self.h, self.hd).transpose(1, 2)
        V = self.v(z).view(B, L, self.h, self.hd).transpose(1, 2)
        
        scores = (Q @ K.transpose(-2, -1)) * (self.hd ** -0.5) * self.temp
        
        if mask is not None:
            scores = scores + mask
        
        # Sigmoid：独立判断相关性
        relevance = torch.sigmoid(scores)
        relevance_sum = relevance.sum(-1, keepdim=True).clamp(min=1.0)
        
        out = (relevance @ V) / relevance_sum
        out = out.transpose(1, 2).reshape(B, L, D)
        
        return self.o(out)


# =============================================================================
# 核心 4: 联络更新 — 联络也演化
# =============================================================================

class ConnectionUpdate(nn.Module):
    """
    联络更新：联络也随着演化而变化
    
    物理意义：
    - 沿着流形移动时，联络会变化
    - 这是平行移动的本质
    """
    def __init__(self, d):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(d * 3, d * 2),
            nn.SiLU(),
            nn.Linear(d * 2, d),
        )
    
    def forward(self, connection, z, dz):
        """
        connection: 当前联络 [B, L, D]
        z: 当前位置 [B, L, D]
        dz: 演化方向 [B, L, D]
        返回: 更新后的联络 [B, L, D]
        """
        # 联络的变化取决于：当前联络、位置、演化方向
        delta = self.update(torch.cat([connection, z, dz], dim=-1))
        return connection + delta


# =============================================================================
# 核心 5: 预测头 — 从流形表示预测动力学变化
# =============================================================================

class DynamicsPredictor(nn.Module):
    """
    动力学预测器：从 z 预测 d²x
    
    不硬编码规则，让网络自己学习
    """
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
    
    def forward(self, z):
        """
        z: [B, D] 最后位置的表示
        返回: d2x [B] 预测的动力学变化
        """
        return self.net(z).squeeze(-1)


# =============================================================================
# 核心 6: 演化块 — 协变演化 + 守恒约束
# =============================================================================

class CovariantEvolutionBlock(nn.Module):
    """
    协变演化块
    
    流程：
    1. dz = CovariantField(z, connection)：协变向量场
    2. ctx = CovariantAttn(z, connection)：协变注意力
    3. z_new = z + dz + ctx：更新位置
    4. connection = update(connection, z, dz)：更新联络
    
    新增：返回演化前后的 z，用于守恒损失计算
    """
    def __init__(self, d, h=4):
        super().__init__()
        self.norm_z = RMSNorm(d)
        self.norm_c = RMSNorm(d)
        
        self.field = CovariantVectorField(d)
        self.attn = CovariantAttention(d, h)
        self.conn_update = ConnectionUpdate(d)
        
        self.norm_mlp = RMSNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.SiLU(),
            nn.Linear(d * 4, d),
        )
    
    def forward(self, z, connection, mask=None):
        """
        z: 位置 [B, L, D]
        connection: 联络 [B, L, D]
        返回: (z_new, connection_new, z_before)
        """
        z_before = z  # 保存演化前的位置，用于守恒损失
        
        z_norm = self.norm_z(z)
        c_norm = self.norm_c(connection)
        
        # 1. 协变向量场
        dz = self.field(z_norm, c_norm)
        
        # 2. 协变注意力
        ctx = self.attn(z_norm, c_norm, mask)
        
        # 3. 更新位置
        z = z + dz + ctx
        
        # 4. 更新联络（联络也随演化变化）
        connection = self.conn_update(connection, z, dz)
        
        # 5. MLP
        z = z + self.mlp(self.norm_mlp(z))
        
        return z, connection, z_before


# =============================================================================
# 完整模型：序列演化
# =============================================================================

class EvolutionForSequence(nn.Module):
    """
    序列演化模型
    
    核心范式：
        输入序列 → (z, 联络) → 协变演化 → 预测 d²x
    
    预测目标：
        d²x（动力学变化/加速度），而不是 x
    """
    
    def __init__(self, d=64, num_layers=2, num_heads=4, input_dim=1):
        super().__init__()
        self.d = d
        self.num_layers = num_layers
        
        # 1. 联络编码器
        self.encoder = ConnectionEncoder(input_dim, d)
        
        # 2. 演化块
        self.blocks = nn.ModuleList([
            CovariantEvolutionBlock(d, num_heads) 
            for _ in range(num_layers)
        ])
        
        # 3. 动力学预测器
        self.predictor = DynamicsPredictor(d)
        
        # 4. 归一化
        self.norm = RMSNorm(d)
        
        # 因果掩码
        self.register_buffer("mask", torch.triu(
            torch.full((2048, 2048), float('-inf')), 1
        ))
    
    def forward(self, values):
        """
        values: [B, L] 输入序列
        返回: 预测结果字典
        
        核心：预测 d²x（动力学变化）
        """
        B, L = values.shape
        
        # 计算微分结构
        dx = values[:, 1:] - values[:, :-1]  # [B, L-1]
        
        x_last = values[:, -1]
        dx_last = dx[:, -1]
        
        # 1. 编码到 (z, 联络)
        z, connection = self.encoder(values)
        
        # 2. 协变演化
        for block in self.blocks:
            z, connection, _ = block(z, connection, self.mask[:L, :L])
        
        # 3. 归一化
        z = self.norm(z)
        
        # 4. 预测 d²x
        pred_d2x = self.predictor(z[:, -1])
        
        # 5. 用 d²x 重建下一个位置
        x_new = x_last + dx_last + pred_d2x
        
        return {
            'x_new': x_new,
            'pred_d2x': pred_d2x,
            'x_last': x_last,
            'dx_last': dx_last,
            'z': z,
        }
    
    def compute_loss(self, values, target=None):
        """
        计算损失：监督 d²x 预测
        """
        out = self.forward(values)
        
        if target is None:
            target = values[:, -1]
        
        # 计算真实的 d²x
        d2x_true = target - out['x_last'] - out['dx_last']
        
        # 预测损失
        pred_loss = F.mse_loss(out['pred_d2x'], d2x_true)
        
        return {
            'loss': pred_loss,
            'pred_loss': pred_loss,
            'x_new': out['x_new'],
            'pred_d2x': out['pred_d2x'],
            'd2x_true': d2x_true,
        }
    
    def predict_sequence(self, values, steps=5):
        """预测未来序列"""
        result = values.clone()
        
        for _ in range(steps):
            out = self.forward(result)
            next_val = out['x_new'].unsqueeze(1)
            result = torch.cat([result, next_val], dim=1)
        
        return result


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    model = EvolutionForSequence(d=64, num_layers=2)
    
    total = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total:,}")
    
    # 测试
    x = torch.rand(2, 10)
    out = model(x)
    print(f"x_new: {out['x_new'].shape}")
    print(f"z: {out['z'].shape}")
    
    # 测试损失
    loss_dict = model.compute_loss(x, x[:, -1])
    print(f"Loss: {loss_dict['loss'].item():.4f}")
    print(f"  pred_loss: {loss_dict['pred_loss'].item():.4f}")
    print(f"  conserv_loss: {loss_dict['conserv_loss'].item():.4f}")
