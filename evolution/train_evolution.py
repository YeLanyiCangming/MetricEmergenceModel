"""
演化方程模型训练

核心方程：dz/dt = F(z) + Correction(z)
这不是 Mamba，这是 Mamba2
"""

import torch
from evolution import EvolutionLM
from pathlib import Path


def train():
    p = Path(__file__).parent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    
    model = EvolutionLM(d=256, L=4).to(device)
    
    total = sum(x.numel() for x in model.parameters())
    print(f"参数量: {total:,}")
    
    # 加载数据
    print("\n加载数据...")
    import json
    data_path = p / 'data' / 'distill_r1_110k_sft.jsonl'
    
    texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            d = json.loads(line)
            text = d.get('instruction', '') + d.get('output', '')
            if len(text) > 50:
                texts.append(text[:500])
    
    print(f"文本: {len(texts)} 条")
    
    seqs = []
    for text in texts:
        ids = tok.encode(text, max_length=256, truncation=True)
        if len(ids) > 20:
            seqs.append(torch.tensor(ids))
    
    print(f"序列: {len(seqs)} 条")
    
    # 训练
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    
    batch_size = 4
    epochs = 50
    
    print("\n开始训练...")
    print("（演化方程：dz/dt = F(z) + Correction）\n")
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i+batch_size]
            maxlen = max(len(s) for s in batch_seqs)
            batch = torch.zeros(len(batch_seqs), maxlen, dtype=torch.long, device=device)
            for j, s in enumerate(batch_seqs):
                batch[j, :len(s)] = s.to(device)
            
            opt.zero_grad()
            out = model(batch, batch)
            loss = out['loss']
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            steps += 1
        
        print(f"Epoch {epoch+1:02d}/{epochs}: loss={total_loss/steps:.4f}")
    
    # 探索
    print("\n" + "="*60)
    print("演化探索")
    print("="*60)
    
    model.eval()
    
    probes = ["你好", "1+1=", "今天天气", "请介绍", "<think>"]
    
    for probe in probes:
        ids = torch.tensor([tok.encode(probe)], device=device)
        out = model.generate(ids, max_new=30, temp=0.7)
        result = tok.decode(out[0].tolist())
        print(f"\n{probe}")
        print(f"  → {result[:80]}..." if len(result) > 80 else f"  → {result}")
    
    torch.save(model.state_dict(), p / 'evolution_trained.pt')
    print(f"\n已保存 evolution_trained.pt")


if __name__ == "__main__":
    train()
