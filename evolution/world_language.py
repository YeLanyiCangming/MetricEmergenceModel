"""
world_language.py - 语言作为扰动的世界模型

核心思想：
    语言不是要学习的对象，语言是世界的扰动。
    世界有自己的动力学 (N, D, K)。
    几何从动力学涌现，不需要单独学习。

物理框架：
    N·ẍ + D·ẋ + K·x = F(t)
    
    其中：
    - x: 世界状态（认知/意义空间）
    - N: 惯性矩阵（响应的迟缓程度）
    - D: 阻尼矩阵（信息的衰减）
    - K: 刚度矩阵（结构的稳定性）
    - F(t): 语言扰动
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class LanguageAsDisturbance(nn.Module):
    """
    语言作为扰动的世界模型
    
    核心：语言 → 扰动 → 世界响应 → 度规涌现 → 预测
    """
    
    def __init__(
        self,
        vocab_size: int = 256,      # 字节级
        d_state: int = 64,          # 世界状态维度
        dt: float = 0.1,            # 时间步长
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_state = d_state
        self.dt = dt
        
        # ========== 动力学参数（可学习）==========
        # 惯性矩阵 N（对称正定）
        self._N_raw = nn.Parameter(torch.randn(d_state, d_state) * 0.1)
        
        # 阻尼矩阵 D（对称半正定）
        self._D_raw = nn.Parameter(torch.randn(d_state, d_state) * 0.1)
        
        # 刚度矩阵 K（对称正定）
        self._K_raw = nn.Parameter(torch.randn(d_state, d_state) * 0.1)
        
        # ========== 编码器：语言 → 扰动 ==========
        self.token_embedding = nn.Embedding(vocab_size, d_state)
        
        # 扰动调制（让不同位置的扰动有不同的"强度"）
        self.disturbance_scale = nn.Parameter(torch.ones(1))
        
        # ========== 解码器：状态 → 预测 ==========
        self.decoder = nn.Linear(d_state, vocab_size)
        
        # 初始状态
        self.x0 = nn.Parameter(torch.zeros(d_state))
        self.v0 = nn.Parameter(torch.zeros(d_state))  # 初始速度
    
    @property
    def N(self) -> torch.Tensor:
        """惯性矩阵（保证对称正定）"""
        return self._N_raw @ self._N_raw.T + 0.1 * torch.eye(self.d_state, device=self._N_raw.device)
    
    @property
    def D(self) -> torch.Tensor:
        """阻尼矩阵（保证对称半正定）"""
        return self._D_raw @ self._D_raw.T
    
    @property
    def K(self) -> torch.Tensor:
        """刚度矩阵（保证对称正定）"""
        return self._K_raw @ self._K_raw.T + 0.01 * torch.eye(self.d_state, device=self._K_raw.device)
    
    def encode_disturbance(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        将语言编码为扰动序列
        
        tokens: (B, T) 整数序列
        返回: (B, T, d_state) 扰动序列
        """
        # 简单的嵌入作为扰动
        F_t = self.token_embedding(tokens)  # (B, T, d_state)
        F_t = F_t * self.disturbance_scale
        return F_t
    
    def solve_dynamics(
        self, 
        F_seq: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        v0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        求解世界动力学响应
        
        N·ẍ + D·ẋ + K·x = F(t)
        
        简化版：使用一阶欧拉（先验证概念）
        
        F_seq: (B, T, d_state) 扰动序列
        返回: 
            x_seq: (B, T, d_state) 位置序列
            v_seq: (B, T, d_state) 速度序列
        """
        B, T, d = F_seq.shape
        device = F_seq.device
        
        # 初始条件
        if x0 is None:
            x0 = self.x0.unsqueeze(0).expand(B, -1)
        if v0 is None:
            v0 = self.v0.unsqueeze(0).expand(B, -1)
        
        # 获取动力学矩阵
        N_inv = torch.linalg.inv(self.N)  # (d, d)
        D = self.D
        K = self.K
        dt = self.dt
        
        # 存储轨迹
        x_list = []
        v_list = []
        
        x = x0  # (B, d)
        v = v0  # (B, d)
        
        for t in range(T):
            F_t = F_seq[:, t, :]  # (B, d)
            
            # 加速度: a = N^{-1} (F - D·v - K·x)
            a = (N_inv @ (F_t - (D @ v.T).T - (K @ x.T).T).T).T  # (B, d)
            
            # 更新
            v_new = v + a * dt
            x_new = x + v_new * dt
            
            x_list.append(x_new)
            v_list.append(v_new)
            
            x = x_new
            v = v_new
        
        x_seq = torch.stack(x_list, dim=1)  # (B, T, d)
        v_seq = torch.stack(v_list, dim=1)  # (B, T, d)
        
        return x_seq, v_seq
    
    def get_emergent_metric(self) -> torch.Tensor:
        """
        从动力学参数涌现度规
        
        通过广义特征分解 K·Φ = N·Φ·Λ
        度规在模态空间是对角的：g = diag(ω²)
        """
        N = self.N.detach().cpu().numpy()
        K = self.K.detach().cpu().numpy()
        
        # 广义特征分解
        from scipy.linalg import eigh
        omega_sq, Phi = eigh(K, N)
        
        # 模态空间的度规是对角的
        g_modal = np.diag(np.maximum(omega_sq, 1e-6))
        
        # 物理空间的度规
        # g_physical = Φ^{-T} g_modal Φ^{-1}
        Phi_inv = np.linalg.inv(Phi)
        g_physical = Phi_inv.T @ g_modal @ Phi_inv
        
        return torch.tensor(g_physical, dtype=torch.float32)
    
    def forward(self, tokens: torch.Tensor) -> dict:
        """
        前向传播
        
        tokens: (B, T) 输入序列
        """
        # 1. 语言 → 扰动
        F_seq = self.encode_disturbance(tokens)  # (B, T, d_state)
        
        # 2. 世界响应
        x_seq, v_seq = self.solve_dynamics(F_seq)  # (B, T, d_state)
        
        # 3. 解码 → 预测
        logits = self.decoder(x_seq)  # (B, T, vocab_size)
        
        return {
            'logits': logits,
            'x_seq': x_seq,
            'v_seq': v_seq,
            'F_seq': F_seq,
        }
    
    def compute_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        计算损失（预测下一个 token）
        """
        # 输入是 tokens[:-1]，目标是 tokens[1:]
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        
        result = self.forward(input_tokens)
        logits = result['logits']  # (B, T-1, vocab_size)
        
        # 交叉熵损失
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_tokens.reshape(-1)
        )
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_bytes: int = 50,
        temperature: float = 1.0,
    ) -> str:
        """
        生成文本
        """
        self.eval()
        device = next(self.parameters()).device
        
        # 编码 prompt
        input_bytes = list(prompt.encode('utf-8'))
        
        # 初始化状态
        x = self.x0.clone()
        v = self.v0.clone()
        
        generated = []
        
        for _ in range(max_new_bytes):
            # 当前输入
            tokens = torch.tensor([input_bytes + generated], device=device)
            
            # 前向
            result = self.forward(tokens)
            logits = result['logits'][0, -1]  # 最后一个位置
            
            # 采样
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_byte = torch.multinomial(probs, 1).item()
            else:
                next_byte = logits.argmax().item()
            
            generated.append(next_byte)
            
            # 停止条件
            if next_byte == 0:  # null byte
                break
        
        return bytes(generated).decode('utf-8', errors='replace')


def demo():
    """演示语言作为扰动的世界模型"""
    
    print("=" * 60)
    print("语言作为扰动的世界模型")
    print("=" * 60)
    
    # 创建模型
    model = LanguageAsDisturbance(
        vocab_size=256,
        d_state=32,
        dt=0.1,
    )
    
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练数据
    print("\n【1】准备训练数据")
    texts = [
        "ABABABABAB",
        "ABCABCABCABC",
        "hello world hello world",
        "the cat sat on the mat",
    ] * 50
    
    # 转换为字节，统一长度
    max_len = 20
    data = []
    for text in texts:
        bytes_seq = list(text.encode('utf-8'))[:max_len]
        # 填充到统一长度
        while len(bytes_seq) < max_len:
            bytes_seq.append(0)  # 用 0 填充
        data.append(torch.tensor(bytes_seq))
    
    data = torch.stack(data)  # (N, max_len)
    print(f"  训练样本数: {len(data)}")
    print(f"  序列长度: {max_len}")
    
    # 训练
    print("\n【2】训练")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 32
    model.train()
    for epoch in range(100):
        total_loss = 0
        n_batches = 0
        
        # 打乱数据
        perm = torch.randperm(len(data))
        
        for i in range(0, len(data), batch_size):
            batch = data[perm[i:i+batch_size]]
            
            loss = model.compute_loss(batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch+1}: loss = {avg_loss:.4f}")
    
    # 测试
    print("\n【3】测试生成")
    
    prompts = ["AB", "ABC", "hello", "the cat"]
    
    for prompt in prompts:
        generated = model.generate(prompt, max_new_bytes=20, temperature=0.8)
        print(f"  '{prompt}' → '{prompt}{generated}'")
    
    # 分析涌现的度规
    print("\n【4】涌现的度规")
    g = model.get_emergent_metric()
    
    print(f"  度规形状: {g.shape}")
    print(f"  度规是否对称: {torch.allclose(g, g.T)}")
    
    # 特征值（度规的"形状"）
    eigenvalues = torch.linalg.eigvalsh(g)
    print(f"  度规特征值范围: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
    
    # 分析动力学参数
    print("\n【5】学习到的动力学参数")
    
    N_eig = torch.linalg.eigvalsh(model.N)
    D_eig = torch.linalg.eigvalsh(model.D)
    K_eig = torch.linalg.eigvalsh(model.K)
    
    print(f"  惯性 N 特征值: [{N_eig.min():.4f}, {N_eig.max():.4f}]")
    print(f"  阻尼 D 特征值: [{D_eig.min():.4f}, {D_eig.max():.4f}]")
    print(f"  刚度 K 特征值: [{K_eig.min():.4f}, {K_eig.max():.4f}]")
    
    # 物理解释
    print("\n【6】物理解释")
    print(f"  N 大 → 响应慢（系统有惯性）")
    print(f"  D 大 → 信息衰减快（容易遗忘）")
    print(f"  K 大 → 结构稳定（模式固定）")
    
    return model


if __name__ == "__main__":
    model = demo()
    
    print("\n" + "=" * 60)
    print("【总结】")
    print("=" * 60)
    print("""
语言作为扰动的世界模型：

  1. 语言 → 扰动 F(t)
     不是直接学习语言的统计规律
     而是把语言当作对世界的扰动

  2. 世界响应 → 状态 x(t)
     N·ẍ + D·ẋ + K·x = F(t)
     世界有自己的动力学

  3. 度规涌现 → g
     从 (N, K) 通过模态分解自动得到
     不需要单独学习！

  4. 预测 → decode(x)
     预测是世界状态的解码
     不是统计采样

关键区别：
  - 传统: 学习 P(next|context)
  - 我们: 学习世界动力学 (N, D, K)
  
  几何不是学习的对象，是涌现的结果。
""")
