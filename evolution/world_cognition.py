"""
world_cognition.py - 认知的动力学本质

核心思想：
    语言不是扰动，而是势能景观的重定义。
    理解 = 找到正确的平衡态。
    思考 = 在势能景观上的演化。
    生成 = 演化轨迹的解码。

物理框架：
    N·ẍ + D·ẋ + K(w)·(x - x_eq(w)) = 0
    
    语言 w 改变：
    - K(w): 刚度矩阵（结构）
    - x_eq(w): 平衡点（意义）
    
    系统自主演化到新平衡，不需要外力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class CognitiveDynamics(nn.Module):
    """
    认知的动力学模型
    
    语言重定义势能景观，系统演化到新平衡
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        d_state: int = 64,
        n_evolve_steps: int = 10,  # 演化步数
        memory_decay: float = 0.8,  # 认知阻尼 λ
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_state = d_state
        self.n_evolve_steps = n_evolve_steps
        self.memory_decay = memory_decay
        
        # ========== 默认状态（全局吸引子）==========
        self.x_default = nn.Parameter(torch.zeros(d_state))
        
        # ========== 惯性矩阵 N（固定，表示认知惯性）==========
        self.register_buffer('N', torch.eye(d_state))
        
        # ========== 语言 → 势能景观 ==========
        # 每个 token 定义一个平衡点 x_eq
        self.token_to_equilibrium = nn.Embedding(vocab_size, d_state)
        
        # 每个 token 调制刚度矩阵 K
        # K(w) = K_base + token_modulation(w)
        self._K_base = nn.Parameter(torch.randn(d_state, d_state) * 0.1)
        self.token_to_K_modulation = nn.Embedding(vocab_size, d_state)
        
        # ========== 离散跳跃函数 J ==========
        self.jump_gate = nn.Sequential(
            nn.Linear(d_state * 2 + d_state, d_state),  # x_prev, x_eq, token_emb
            nn.Tanh(),
            nn.Linear(d_state, d_state),
        )
        self.token_embedding = nn.Embedding(vocab_size, d_state)
        
        # ========== 解码器 ==========
        self.decoder = nn.Sequential(
            nn.Linear(d_state, d_state * 2),
            nn.GELU(),
            nn.Linear(d_state * 2, vocab_size),
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self._K_base)
        nn.init.normal_(self.token_to_equilibrium.weight, std=0.1)
        nn.init.normal_(self.token_to_K_modulation.weight, std=0.01)
    
    @property
    def K_base(self) -> torch.Tensor:
        """基础刚度矩阵（对称正定）"""
        K = self._K_base @ self._K_base.T
        return K + 0.1 * torch.eye(self.d_state, device=K.device)
    
    def get_K(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        获取语言调制后的刚度矩阵
        
        tokens: (B,) 当前 token
        返回: (B, d, d) 刚度矩阵
        """
        B = tokens.shape[0]
        
        # 基础刚度
        K = self.K_base.unsqueeze(0).expand(B, -1, -1)  # (B, d, d)
        
        # Token 调制（对角修正）
        modulation = self.token_to_K_modulation(tokens)  # (B, d)
        modulation = F.softplus(modulation)  # 保证正数
        
        # 对角调制
        K = K + torch.diag_embed(modulation)  # (B, d, d)
        
        return K
    
    def get_equilibrium(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        获取语言定义的平衡点
        
        tokens: (B,) 当前 token
        返回: (B, d) 平衡点
        """
        return self.token_to_equilibrium(tokens)
    
    def get_damping(self, K: torch.Tensor) -> torch.Tensor:
        """
        计算临界阻尼 D = 2√(NK) = 2√K（因为 N=I）
        
        K: (B, d, d) 刚度矩阵
        返回: (B, d, d) 阻尼矩阵
        """
        # 对 K 做特征分解
        eigenvalues, eigenvectors = torch.linalg.eigh(K)
        
        # 临界阻尼的特征值
        D_eigenvalues = 2 * torch.sqrt(torch.clamp(eigenvalues, min=1e-6))
        
        # 重建 D
        D = eigenvectors @ torch.diag_embed(D_eigenvalues) @ eigenvectors.transpose(-1, -2)
        
        return D
    
    def discrete_jump(
        self, 
        x_prev: torch.Tensor, 
        tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        离散跳跃：认知阻尼 + 状态更新
        
        x_jump = (1-λ) x_default + λ J(x_prev, w)
        
        x_prev: (B, d) 之前状态
        tokens: (B,) 当前 token
        返回: (B, d) 跳跃后状态
        """
        B = x_prev.shape[0]
        
        # 获取平衡点和 token 嵌入
        x_eq = self.get_equilibrium(tokens)  # (B, d)
        token_emb = self.token_embedding(tokens)  # (B, d)
        
        # 跳跃函数
        jump_input = torch.cat([x_prev, x_eq, token_emb], dim=-1)  # (B, 3d)
        x_jump_raw = self.jump_gate(jump_input)  # (B, d)
        
        # 认知阻尼：混合默认状态
        x_default = self.x_default.unsqueeze(0).expand(B, -1)
        x_jump = (1 - self.memory_decay) * x_default + self.memory_decay * x_jump_raw
        
        return x_jump
    
    def continuous_evolve(
        self,
        x_init: torch.Tensor,
        K: torch.Tensor,
        x_eq: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        连续演化：滑向新平衡
        
        N·ẍ + D·ẋ + K·(x - x_eq) = 0
        
        使用临界阻尼，保证快速收敛无振荡
        
        x_init: (B, d) 初始状态
        K: (B, d, d) 刚度矩阵
        x_eq: (B, d) 平衡点
        n_steps: 演化步数
        
        返回: (B, d) 最终状态
        """
        if n_steps is None:
            n_steps = self.n_evolve_steps
        
        B, d = x_init.shape
        
        # 临界阻尼
        D = self.get_damping(K)  # (B, d, d)
        
        # N = I
        N_inv = torch.eye(d, device=x_init.device).unsqueeze(0).expand(B, -1, -1)
        
        # 初始条件
        x = x_init.clone()
        v = torch.zeros_like(x)  # 初始速度为 0
        
        dt = 0.5  # 时间步长
        
        for _ in range(n_steps):
            # 相对位移
            dx = x - x_eq  # (B, d)
            
            # 加速度: a = -N^{-1}(D·v + K·dx)
            # 由于 N = I: a = -(D·v + K·dx)
            Dv = torch.bmm(D, v.unsqueeze(-1)).squeeze(-1)  # (B, d)
            Kdx = torch.bmm(K, dx.unsqueeze(-1)).squeeze(-1)  # (B, d)
            a = -(Dv + Kdx)
            
            # 更新（半隐式欧拉）
            v = v + a * dt
            x = x + v * dt
        
        return x
    
    def forward(self, tokens: torch.Tensor) -> dict:
        """
        前向传播
        
        tokens: (B, T) 输入序列
        """
        B, T = tokens.shape
        device = tokens.device
        
        # 初始状态
        x = self.x_default.unsqueeze(0).expand(B, -1).clone()  # (B, d)
        
        # 存储每个时间步的输出状态
        x_outputs = []
        
        for t in range(T):
            token_t = tokens[:, t]  # (B,)
            
            # 1. 离散跳跃
            x_jump = self.discrete_jump(x, token_t)
            
            # 2. 获取新的势能景观参数
            K = self.get_K(token_t)  # (B, d, d)
            x_eq = self.get_equilibrium(token_t)  # (B, d)
            
            # 3. 连续演化到新平衡
            x = self.continuous_evolve(x_jump, K, x_eq)
            
            x_outputs.append(x)
        
        # 堆叠
        x_seq = torch.stack(x_outputs, dim=1)  # (B, T, d)
        
        # 解码
        logits = self.decoder(x_seq)  # (B, T, vocab_size)
        
        return {
            'logits': logits,
            'x_seq': x_seq,
        }
    
    def compute_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        
        result = self.forward(input_tokens)
        logits = result['logits']
        
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
        """生成文本"""
        self.eval()
        device = next(self.parameters()).device
        
        # 编码 prompt
        input_bytes = list(prompt.encode('utf-8'))
        
        # 初始状态
        x = self.x_default.clone()
        
        # 处理 prompt
        for byte in input_bytes:
            token = torch.tensor([byte], device=device)
            
            # 离散跳跃
            x_jump = self.discrete_jump(x.unsqueeze(0), token).squeeze(0)
            
            # 获取势能景观
            K = self.get_K(token)
            x_eq = self.get_equilibrium(token).squeeze(0)
            
            # 演化
            x = self.continuous_evolve(
                x_jump.unsqueeze(0), K, x_eq.unsqueeze(0)
            ).squeeze(0)
        
        # 生成
        generated = []
        
        for _ in range(max_new_bytes):
            # 解码当前状态
            logits = self.decoder(x.unsqueeze(0)).squeeze(0)  # (vocab_size,)
            
            # 采样
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_byte = torch.multinomial(probs, 1).item()
            else:
                next_byte = logits.argmax().item()
            
            generated.append(next_byte)
            
            # 用生成的 token 更新状态
            token = torch.tensor([next_byte], device=device)
            
            x_jump = self.discrete_jump(x.unsqueeze(0), token).squeeze(0)
            K = self.get_K(token)
            x_eq = self.get_equilibrium(token).squeeze(0)
            x = self.continuous_evolve(
                x_jump.unsqueeze(0), K, x_eq.unsqueeze(0)
            ).squeeze(0)
            
            # 停止条件
            if next_byte == 0:
                break
        
        return bytes(generated).decode('utf-8', errors='replace')
    
    def analyze_equilibrium(self, text: str) -> dict:
        """分析文本对应的平衡态"""
        device = next(self.parameters()).device
        
        bytes_seq = list(text.encode('utf-8'))
        tokens = torch.tensor(bytes_seq, device=device)
        
        # 获取每个 token 的平衡点
        x_eqs = self.get_equilibrium(tokens)  # (T, d)
        
        # 计算平衡点之间的距离
        distances = []
        for i in range(len(tokens) - 1):
            dist = torch.norm(x_eqs[i+1] - x_eqs[i]).item()
            distances.append(dist)
        
        return {
            'equilibria': x_eqs.detach().cpu().numpy(),
            'distances': distances,
            'mean_distance': np.mean(distances) if distances else 0,
        }


def demo():
    """演示认知动力学模型"""
    
    print("=" * 60)
    print("认知的动力学本质")
    print("=" * 60)
    
    # 创建模型
    model = CognitiveDynamics(
        vocab_size=256,
        d_state=64,
        n_evolve_steps=10,
        memory_decay=0.8,
    )
    
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练数据
    print("\n【1】准备训练数据")
    texts = [
        "ABABABABAB",
        "ABCABCABCABC",
        "hello world hello world",
        "the cat sat on the mat",
        "dog cat dog cat dog cat",
    ] * 50
    
    max_len = 24
    data = []
    for text in texts:
        bytes_seq = list(text.encode('utf-8'))[:max_len]
        while len(bytes_seq) < max_len:
            bytes_seq.append(0)
        data.append(torch.tensor(bytes_seq))
    
    data = torch.stack(data)
    print(f"  训练样本数: {len(data)}")
    
    # 训练
    print("\n【2】训练")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 32
    model.train()
    
    for epoch in range(100):
        total_loss = 0
        n_batches = 0
        
        perm = torch.randperm(len(data))
        
        for i in range(0, len(data), batch_size):
            batch = data[perm[i:i+batch_size]]
            
            loss = model.compute_loss(batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch+1}: loss = {avg_loss:.4f}")
    
    # 测试
    print("\n【3】测试生成")
    
    prompts = ["AB", "ABC", "hello", "the cat", "dog"]
    
    for prompt in prompts:
        generated = model.generate(prompt, max_new_bytes=20, temperature=0.7)
        print(f"  '{prompt}' → '{prompt}{generated}'")
    
    # 分析平衡态
    print("\n【4】分析平衡态")
    
    for text in ["ABAB", "hello"]:
        analysis = model.analyze_equilibrium(text)
        print(f"  '{text}':")
        print(f"    平均平衡点距离: {analysis['mean_distance']:.4f}")
    
    # 物理解释
    print("\n【5】物理解释")
    print("""
  语言改变势能景观：
    - 每个 token 定义一个新的"正确状态" x_eq
    - 系统滑向这个状态
    - 演化过程就是"理解"

  临界阻尼：
    - D = 2√(NK)
    - 系统最快收敛，无振荡
    - 不会产生重复

  认知阻尼：
    - x_jump = (1-λ) x_default + λ J(x, w)
    - 防止历史累积
    - 保持认知新鲜度
""")
    
    return model


if __name__ == "__main__":
    model = demo()
    
    print("\n" + "=" * 60)
    print("【总结】")
    print("=" * 60)
    print("""
认知的动力学本质：

  1. 语言不是扰动，而是势能景观的重定义
     - K(w): 什么方向重要
     - x_eq(w): 什么是正确的状态

  2. 理解 = 找到正确的平衡态
     - 系统自主演化
     - 不需要外力推动

  3. 思考 = 在势能景观上的演化
     - 连续的"滑向平衡"
     - 不是离散的"计算步骤"

  4. 生成 = 演化轨迹的解码
     - 每个平衡态对应一个 token
     - 序列是状态演化的历史

关键改进：
  - 临界阻尼：消除振荡（重复）
  - 认知阻尼：防止累积（混乱）
  - 势能重定义：语言改变"什么是正确的"
""")
