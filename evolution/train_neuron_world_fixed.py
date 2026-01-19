"""
train_neuron_world_fixed.py - 修复版神经元世界模型训练

修复内容：
1. beta 现在通过可微分特征分解获得梯度
2. gamma 现在通过阻尼一致性损失获得梯度
3. 增加更多训练数据

验证：
- 检查 beta/gamma 是否真正学习
- 检查物理参数的梯度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
import os

from world_unified import UnifiedWorldModel


# =============================================================================
# 神经元激发层
# =============================================================================

class NeuronActivation(nn.Module):
    """高斯感受野神经元层"""
    
    def __init__(self, n_neurons: int, init_type: str = "uniform"):
        super().__init__()
        self.n_neurons = n_neurons
        
        if init_type == "uniform":
            preferred = torch.linspace(0, 255, n_neurons)
        elif init_type == "random":
            preferred = torch.rand(n_neurons) * 255
        else:
            preferred = torch.linspace(0, 255, n_neurons)
        
        self.preferred = nn.Parameter(preferred)
        self.log_sigma = nn.Parameter(torch.ones(n_neurons) * np.log(32.0))
        self.gain = nn.Parameter(torch.ones(n_neurons))
    
    @property
    def sigma(self):
        return torch.exp(self.log_sigma)
    
    def forward(self, byte_values: torch.Tensor) -> torch.Tensor:
        B, T = byte_values.shape
        
        byte_expanded = byte_values.float().unsqueeze(-1)
        preferred = self.preferred.unsqueeze(0).unsqueeze(0)
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        gain = self.gain.unsqueeze(0).unsqueeze(0)
        
        diff = byte_expanded - preferred
        activation = gain * torch.exp(-diff ** 2 / (2 * sigma ** 2))
        
        return activation


class NeuronWorldModel(nn.Module):
    """神经元激发 + 世界动力学"""
    
    def __init__(
        self,
        n_neurons: int = 8,
        n_modes: int = 4,
        vocab_size: int = 256,
        init_type: str = "uniform",
        **world_kwargs
    ):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.n_modes = n_modes
        self.vocab_size = vocab_size
        
        self.neuron_layer = NeuronActivation(n_neurons, init_type)
        self.world_model = UnifiedWorldModel(
            n_channels=n_neurons,
            n_modes=n_modes,
            output_dim=n_neurons,
            **world_kwargs
        )
        
        self.token_head = nn.Sequential(
            nn.Linear(n_modes, n_modes * 4),
            nn.SiLU(),
            nn.Linear(n_modes * 4, vocab_size),
        )
    
    def forward(self, byte_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T = byte_seq.shape
        
        activation = self.neuron_layer(byte_seq)
        out = self.world_model(activation)
        token_logits = self.token_head(out['z_pred'])
        
        out['token_logits'] = token_logits
        out['activation'] = activation
        
        return out
    
    def compute_loss(
        self,
        byte_seq: torch.Tensor,
        target_byte: torch.Tensor = None,
        ce_weight: float = 1.0,
        physics_weight: float = 0.5,
        sparsity_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """计算损失"""
        out = self.forward(byte_seq)
        
        # 1. 预测损失
        if target_byte is not None:
            ce_loss = F.cross_entropy(out['token_logits'], target_byte)
        else:
            ce_loss = torch.tensor(0.0, device=byte_seq.device)
        
        # 2. 物理损失（包含 stiffness 和 damping 损失，让 beta/gamma 有梯度）
        physics_loss = self.world_model.compute_loss(
            out['activation'],
            stiffness_weight=0.2,  # 增加权重
            damping_weight=0.2,    # 增加权重
        )
        
        # 3. 稀疏性损失
        activation = out['activation']
        sparsity_loss = activation.mean()
        
        total_loss = (
            ce_weight * ce_loss +
            physics_weight * physics_loss['loss'] +
            sparsity_weight * sparsity_loss
        )
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'physics_loss': physics_loss['loss'],
            'pred_loss': physics_loss.get('pred_loss', torch.tensor(0.0)),
            'stiffness_loss': physics_loss.get('stiffness_loss', torch.tensor(0.0)),
            'damping_loss': physics_loss.get('damping_loss', torch.tensor(0.0)),
            'sparsity_loss': sparsity_loss,
        }


# =============================================================================
# 数据加载
# =============================================================================

def load_text_data(file_path: str, max_chars: int = None) -> bytes:
    """加载文本文件"""
    with open(file_path, 'rb') as f:
        data = f.read()
    
    if max_chars is not None:
        data = data[:max_chars]
    
    return data


def create_training_samples(data: bytes, seq_len: int = 16, stride: int = 4) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """创建训练样本"""
    samples = []
    
    for i in range(0, len(data) - seq_len, stride):
        input_bytes = torch.tensor(list(data[i:i+seq_len-1]), dtype=torch.long)
        target_byte = torch.tensor(data[i+seq_len-1], dtype=torch.long)
        samples.append((input_bytes, target_byte))
    
    return samples


def create_batch(samples: List[Tuple[torch.Tensor, torch.Tensor]], batch_size: int, device: torch.device):
    """创建批次"""
    indices = torch.randperm(len(samples))[:batch_size]
    
    inputs = torch.stack([samples[i][0] for i in indices]).to(device)
    targets = torch.stack([samples[i][1] for i in indices]).to(device)
    
    return inputs, targets


# =============================================================================
# 训练
# =============================================================================

def check_gradients(model: nn.Module, name_prefix: str = ""):
    """检查参数的梯度"""
    grad_info = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_info[name_prefix + name] = {
                'grad_norm': grad_norm,
                'value': param.data.mean().item() if param.numel() > 1 else param.item(),
                'has_grad': grad_norm > 0
            }
        else:
            grad_info[name_prefix + name] = {
                'grad_norm': 0,
                'value': param.data.mean().item() if param.numel() > 1 else param.item(),
                'has_grad': False
            }
    
    return grad_info


def train():
    print("=" * 70)
    print("修复版神经元世界模型训练")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 加载数据
    data_path = os.path.join(os.path.dirname(__file__), "data", "《这个师父不好惹》.txt")
    
    if os.path.exists(data_path):
        print(f"\n加载数据: {data_path}")
        data = load_text_data(data_path, max_chars=50000)  # 使用更多数据
        print(f"数据大小: {len(data):,} bytes")
    else:
        print("\n未找到数据文件，使用生成数据")
        # 生成更复杂的训练数据
        patterns = [
            b"Hello World! ",
            b"The quick brown fox jumps over the lazy dog. ",
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
            b"0123456789 ",
            b"This is a test sentence for training. ",
            b"Neural networks learn patterns from data. ",
        ]
        data = b"".join(patterns * 100)
    
    # 创建样本
    samples = create_training_samples(data, seq_len=16, stride=2)
    print(f"训练样本数: {len(samples):,}")
    
    # 创建模型
    model = NeuronWorldModel(
        n_neurons=16,
        n_modes=8,
        vocab_size=256,
        init_type="uniform",
        beta_func_init=1.0,
        gamma_init=0.1,
        alpha_init=0.1,
    ).to(device)
    
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    # 记录初始物理参数
    print("\n初始物理参数:")
    print(f"  beta_func: {model.world_model.world_dynamics.beta_func.item():.6f}")
    print(f"  gamma: {model.world_model.world_dynamics.gamma.item():.6f}")
    print(f"  alpha: {model.world_model.energy_geometry.alpha.item():.6f}")
    
    # 训练
    n_epochs = 500
    batch_size = 32
    print_every = 50
    
    best_acc = 0
    history = {
        'loss': [], 'ce_loss': [], 'physics_loss': [],
        'stiffness_loss': [], 'damping_loss': [],
        'beta': [], 'gamma': [], 'alpha': [], 'accuracy': []
    }
    
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)
    
    for epoch in range(n_epochs):
        model.train()
        
        inputs, targets = create_batch(samples, batch_size, device)
        
        optimizer.zero_grad()
        loss_dict = model.compute_loss(inputs, targets)
        loss_dict['loss'].backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 记录
        with torch.no_grad():
            out = model(inputs)
            preds = out['token_logits'].argmax(dim=-1)
            accuracy = (preds == targets).float().mean().item()
        
        history['loss'].append(loss_dict['loss'].item())
        history['ce_loss'].append(loss_dict['ce_loss'].item())
        history['physics_loss'].append(loss_dict['physics_loss'].item())
        history['stiffness_loss'].append(loss_dict['stiffness_loss'].item())
        history['damping_loss'].append(loss_dict['damping_loss'].item())
        history['beta'].append(model.world_model.world_dynamics.beta_func.item())
        history['gamma'].append(model.world_model.world_dynamics.gamma.item())
        history['alpha'].append(model.world_model.energy_geometry.alpha.item())
        history['accuracy'].append(accuracy)
        
        if (epoch + 1) % print_every == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"  Loss: {loss_dict['loss'].item():.4f}")
            print(f"  CE Loss: {loss_dict['ce_loss'].item():.4f}")
            print(f"  Physics Loss: {loss_dict['physics_loss'].item():.4f}")
            print(f"  Stiffness Loss: {loss_dict['stiffness_loss'].item():.6f}")
            print(f"  Damping Loss: {loss_dict['damping_loss'].item():.6f}")
            print(f"  Accuracy: {accuracy*100:.1f}%")
            print(f"  物理参数:")
            print(f"    beta_func: {model.world_model.world_dynamics.beta_func.item():.4f}")
            print(f"    gamma: {model.world_model.world_dynamics.gamma.item():.4f}")
            print(f"    alpha: {model.world_model.energy_geometry.alpha.item():.4f}")
            
            # 检查梯度
            grad_info = check_gradients(model)
            beta_grad = grad_info.get('world_model.world_dynamics.log_beta_func', {})
            gamma_grad = grad_info.get('world_model.world_dynamics.log_gamma', {})
            alpha_grad = grad_info.get('world_model.energy_geometry.log_alpha', {})
            
            print(f"  梯度:")
            print(f"    beta grad: {beta_grad.get('grad_norm', 0):.6f} (has_grad: {beta_grad.get('has_grad', False)})")
            print(f"    gamma grad: {gamma_grad.get('grad_norm', 0):.6f} (has_grad: {gamma_grad.get('has_grad', False)})")
            print(f"    alpha grad: {alpha_grad.get('grad_norm', 0):.6f} (has_grad: {alpha_grad.get('has_grad', False)})")
            
            if accuracy > best_acc:
                best_acc = accuracy
                print(f"  >>> 新最佳准确率: {best_acc*100:.1f}%")
    
    # 最终分析
    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    
    print(f"\n最终物理参数:")
    print(f"  beta_func: {history['beta'][0]:.4f} → {history['beta'][-1]:.4f} (变化: {history['beta'][-1] - history['beta'][0]:+.4f})")
    print(f"  gamma: {history['gamma'][0]:.4f} → {history['gamma'][-1]:.4f} (变化: {history['gamma'][-1] - history['gamma'][0]:+.4f})")
    print(f"  alpha: {history['alpha'][0]:.4f} → {history['alpha'][-1]:.4f} (变化: {history['alpha'][-1] - history['alpha'][0]:+.4f})")
    
    print(f"\n最佳准确率: {best_acc*100:.1f}%")
    
    # 生成测试
    print("\n" + "=" * 70)
    print("生成测试")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        # 随机选择一个起始序列
        start_idx = np.random.randint(0, len(data) - 50)
        seed = list(data[start_idx:start_idx+15])
        seed_text = bytes(seed).decode('utf-8', errors='replace')
        print(f"\n种子: '{seed_text}'")
        
        generated = seed.copy()
        for _ in range(50):
            input_seq = torch.tensor([generated[-15:]], dtype=torch.long, device=device)
            out = model(input_seq)
            probs = F.softmax(out['token_logits'], dim=-1)
            next_byte = torch.multinomial(probs, 1).item()
            generated.append(next_byte)
        
        result = bytes(generated).decode('utf-8', errors='replace')
        print(f"生成: '{result}'")
    
    # 神经元分析
    print("\n" + "=" * 70)
    print("神经元感受野分析")
    print("=" * 70)
    
    with torch.no_grad():
        preferred = model.neuron_layer.preferred.cpu().numpy()
        sigma = model.neuron_layer.sigma.cpu().numpy()
        gain = model.neuron_layer.gain.cpu().numpy()
        
        print(f"\n偏好位置: {preferred[:8]}")
        print(f"宽度 σ: {sigma[:8]}")
        print(f"增益: {gain[:8]}")
        
        # 抑制性神经元
        inhibitory = np.sum(gain < 0)
        print(f"\n抑制性神经元数: {inhibitory}/{len(gain)}")
    
    return model, history


if __name__ == "__main__":
    model, history = train()
