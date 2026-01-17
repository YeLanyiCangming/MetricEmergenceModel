"""Evolution - 最简单的加法规则

目的：用最简单的规则观察本质
- 规则：a[n] = a[n-1] + d（等差数列）
- d 是常数差值
- 这比斐波那契简单得多

问题：模型能否学到"加法规则"而不是"记忆模式"？
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from evolution import EvolutionLM


class AdditionDataset:
    """最简单的加法数据集：等差数列
    
    规则：a[n] = a[n-1] + d
    例如：[3, 5, 7, 9, 11, 13...] (d=2)
    """
    def __init__(self, seq_len=14, num_samples=2, start_range=(1, 10), diff_range=(1, 5)):
        self.data = []
        self.info = []  # 记录每条数据的起点和差值
        
        for _ in range(num_samples):
            start = random.randint(*start_range)
            diff = random.randint(*diff_range)  # 差值
            seq = []
            val = start
            for _ in range(seq_len):
                seq.append(val % 100)
                val = val + diff
            self.data.append(torch.tensor(seq, dtype=torch.long))
            self.info.append((start, diff))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def test_accuracy_addition(model, device, start_range, diff_range, num_tests=20, init_len=3, pred_len=10):
    """加法规则测试（初始条件3个token）
    
    3个token就足够确定等差数列：
    - a[0], a[1] → d = a[1] - a[0]
    - a[2] 用于验证
    """
    model.eval()
    correct = total = 0
    for _ in range(num_tests):
        start = random.randint(*start_range)
        diff = random.randint(*diff_range)
        
        # 初始条件：生成前3个数
        seq = [start % 100, (start + diff) % 100, (start + 2*diff) % 100]
        
        # 期望输出
        expected = []
        val = start + 2*diff
        for _ in range(pred_len):
            val = val + diff
            expected.append(val % 100)
        
        # 自回归生成
        ids = torch.tensor([seq], device=device)
        with torch.no_grad():
            output = model.generate(ids, max_new=pred_len, temp=0.01, top_k=1)
            predicted = output[0, init_len:].tolist()
        
        for p, e in zip(predicted, expected):
            if p == e:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("\n" + "="*60)
    print("最简单的加法规则：a[n] = a[n-1] + d")
    print("="*60)
    
    # 数据集：2条等差数列
    dataset = AdditionDataset(seq_len=14, num_samples=2, start_range=(1, 10), diff_range=(2, 5))
    print(f"训练数据: 2条等差数列")
    for i, (start, diff) in enumerate(dataset.info):
        print(f"示例{i+1}: {dataset[i].tolist()}  (start={start}, d={diff})")
    
    model = EvolutionLM(V=100, d=24, L=2, h=2, maxlen=512).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params:,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch_size = min(10, len(dataset))
    epochs = 50
    
    print("\n开始训练（50轮，观察变化）...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_energy_loss = 0
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch = torch.stack([dataset[j] for j in indices[i:i+batch_size]]).to(device)
            
            opt.zero_grad()
            out = model(batch, labels=batch)
            loss = out['loss']
            loss.backward()
            opt.step()
            total_loss += out['pred_loss'].item()
            if 'energy_loss' in out:
                total_energy_loss = out['energy_loss'].item()
        
        # 每轮都打印
        avg_loss = total_loss / max(1, len(indices) // batch_size)
        # 分布内：同样的差值范围
        acc_in = test_accuracy_addition(model, device, start_range=(1, 10), diff_range=(2, 5), num_tests=10)
        # 分布外：不同的起点和差值
        acc_out = test_accuracy_addition(model, device, start_range=(50, 99), diff_range=(6, 9), num_tests=10)
        print(f"Epoch {epoch+1:2d}: pred_loss={avg_loss:.4f}, energy_loss={total_energy_loss:.4f}, 分布内={acc_in*100:.0f}%, 分布外={acc_out*100:.0f}%")
        
        dt_vals = [b.field.dt.item() for b in model.blocks]
        with torch.no_grad():
            test_ids = torch.tensor([[1, 5, 6, 11]], device=device)
            z, conn = model.encoder(test_ids)
        print(f"       dt={[f'{v:.4f}' for v in dt_vals]}, conn_norm={conn.norm().item():.4f}")
    
    # 自回归生成测试
    print("\n" + "="*60)
    print("加法规则生成测试")
    print("="*60)
    
    model.eval()
    
    def generate_seq(start, diff, init_len=3):
        """生成等差数列前缀"""
        return [start + i * diff for i in range(init_len)]
    
    def check_addition(seq):
        """检查是否满足加法规则"""
        if len(seq) < 3:
            return 0, 0
        # 推断差值
        d = seq[1] - seq[0]
        correct = 0
        for i in range(2, len(seq)):
            expected = (seq[i-1] + d) % 100
            if seq[i] == expected:
                correct += 1
        return correct, len(seq) - 2
    
    def generate(prefix, max_new=10):
        """普通生成"""
        ids = torch.tensor([prefix], device=device)
        with torch.no_grad():
            output = model.generate(ids, max_new=max_new, temp=0.01, top_k=1)
        return output[0].tolist()
    
    print("\n分布内（start=1-10, d=2-5）：")
    for start, d in [(3, 2), (5, 3), (1, 4), (8, 5), (2, 2)]:
        prefix = generate_seq(start, d, 3)
        seq = generate(prefix)
        c, t = check_addition(seq)
        expected_next = (prefix[-1] + d) % 100
        actual_next = seq[3] if len(seq) > 3 else '?'
        print(f"  {prefix} (d={d}) → {seq}")
        print(f"    期望下一个: {expected_next}, 实际: {actual_next}, 一致:{c}/{t}")
    
    print("\n分布外（start=50-99, d=6-9）：")
    for start, d in [(50, 7), (60, 8), (70, 6), (80, 9), (99, 7)]:
        prefix = generate_seq(start, d, 3)
        seq = generate(prefix)
        c, t = check_addition(seq)
        expected_next = (prefix[-1] + d) % 100
        actual_next = seq[3] if len(seq) > 3 else '?'
        print(f"  {prefix} (d={d}) → {seq}")
        print(f"    期望下一个: {expected_next}, 实际: {actual_next}, 一致:{c}/{t}")


if __name__ == "__main__":
    train()
