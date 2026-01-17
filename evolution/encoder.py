"""多模态编码器

将不同模态（语言、文本、图像、视频）映射到统一的Z空间

核心设计：
1. 模态特定编码器 - 处理每种模态的特有结构
2. 对齐层 - 将不同模态对齐到共享空间
3. 融合层 - 多模态信息融合
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# 模态类型定义
# =============================================================================

class Modality(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    SEQUENCE = "sequence"  # 数值序列


@dataclass(frozen=True)
class ModalityInput:
    """模态输入（不可变）"""
    modality: Modality
    data: torch.Tensor
    mask: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None


@dataclass(frozen=True)
class EncodedRepresentation:
    """编码后的表示"""
    z: torch.Tensor              # [B, D] 主表示
    z_sequence: torch.Tensor     # [B, L, D] 序列表示（如果有）
    modality: Modality
    attention_weights: Optional[torch.Tensor] = None


# =============================================================================
# 基础组件
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class FeedForward(nn.Module):
    """前馈网络（SwiGLU）"""
    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        h = d * mult
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SelfAttention(nn.Module):
    """自注意力"""
    def __init__(self, d: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        
        self.qkv = nn.Linear(d, d * 3, bias=False)
        self.out = nn.Linear(d, d, bias=False)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, L, D = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, head_dim]
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        
        return self.out(out)


class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, d: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = SelfAttention(d, num_heads)
        self.norm2 = RMSNorm(d)
        self.ff = FeedForward(d)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


# =============================================================================
# 文本编码器
# =============================================================================

class TextEncoder(nn.Module):
    """文本编码器
    
    将文本token序列编码到Z空间
    """
    def __init__(
        self, 
        vocab_size: int, 
        d: int, 
        num_layers: int = 4,
        num_heads: int = 8,
        max_len: int = 2048
    ):
        super().__init__()
        self.d = d
        
        # Token嵌入
        self.token_embed = nn.Embedding(vocab_size, d)
        
        # 位置编码（RoPE风格）
        self.register_buffer('freqs', self._precompute_freqs(d, max_len))
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(d, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d)
        
        # 池化投影
        self.pool_proj = nn.Linear(d, d)
    
    def _precompute_freqs(self, d: int, max_len: int) -> torch.Tensor:
        """预计算RoPE频率"""
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
        t = torch.arange(max_len)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)
    
    def forward(
        self, 
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> EncodedRepresentation:
        """
        token_ids: [B, L] token IDs
        attention_mask: [B, L] 1表示有效位置
        
        返回: EncodedRepresentation
        """
        B, L = token_ids.shape
        
        # 嵌入
        x = self.token_embed(token_ids)  # [B, L, D]
        
        # 构建因果掩码
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=x.device), 1
        )
        
        if attention_mask is not None:
            # 添加padding掩码
            padding_mask = (1 - attention_mask.float()) * float('-inf')
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask + padding_mask
        
        # Transformer
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        x = self.norm(x)  # [B, L, D]
        
        # 池化（最后一个有效位置）
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long() - 1
            z = x[torch.arange(B), lengths]
        else:
            z = x[:, -1]
        
        z = self.pool_proj(z)
        
        return EncodedRepresentation(
            z=z,
            z_sequence=x,
            modality=Modality.TEXT
        )


# =============================================================================
# 图像编码器
# =============================================================================

class PatchEmbed(nn.Module):
    """图像块嵌入（ViT风格）"""
    def __init__(self, img_size: int = 224, patch_size: int = 16, d: int = 256, in_channels: int = 3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, d, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        返回: [B, num_patches, D]
        """
        x = self.proj(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, D]
        return x


class ImageEncoder(nn.Module):
    """图像编码器（ViT风格）"""
    def __init__(
        self, 
        img_size: int = 224,
        patch_size: int = 16,
        d: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        in_channels: int = 3
    ):
        super().__init__()
        self.d = d
        
        # Patch嵌入
        self.patch_embed = PatchEmbed(img_size, patch_size, d, in_channels)
        num_patches = self.patch_embed.num_patches
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        
        # 位置嵌入
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d) * 0.02)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(d, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d)
        
        # 输出投影
        self.out_proj = nn.Linear(d, d)
    
    def forward(self, images: torch.Tensor) -> EncodedRepresentation:
        """
        images: [B, C, H, W]
        
        返回: EncodedRepresentation
        """
        B = images.shape[0]
        
        # Patch嵌入
        x = self.patch_embed(images)  # [B, num_patches, D]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, D]
        
        # 添加位置嵌入
        x = x + self.pos_embed
        
        # Transformer
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # CLS token作为全局表示
        z = self.out_proj(x[:, 0])
        
        return EncodedRepresentation(
            z=z,
            z_sequence=x,
            modality=Modality.IMAGE
        )


# =============================================================================
# 视频编码器
# =============================================================================

class VideoEncoder(nn.Module):
    """视频编码器
    
    处理时序图像序列
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_frames: int = 8
    ):
        super().__init__()
        self.d = d
        self.num_frames = num_frames
        
        # 空间编码器（共享）
        self.spatial_encoder = ImageEncoder(
            img_size, patch_size, d, num_layers // 2, num_heads
        )
        
        # 时间位置嵌入
        self.temporal_embed = nn.Parameter(torch.randn(1, num_frames, d) * 0.02)
        
        # 时间Transformer
        self.temporal_layers = nn.ModuleList([
            TransformerBlock(d, num_heads) for _ in range(num_layers // 2)
        ])
        
        self.norm = RMSNorm(d)
        self.out_proj = nn.Linear(d, d)
    
    def forward(self, video: torch.Tensor) -> EncodedRepresentation:
        """
        video: [B, T, C, H, W] 视频帧
        
        返回: EncodedRepresentation
        """
        B, T, C, H, W = video.shape
        
        # 空间编码每一帧
        video_flat = video.view(B * T, C, H, W)
        spatial_enc = self.spatial_encoder(video_flat)
        
        # 提取每帧的全局表示
        frame_features = spatial_enc.z.view(B, T, -1)  # [B, T, D]
        
        # 添加时间位置嵌入
        frame_features = frame_features + self.temporal_embed[:, :T]
        
        # 时间建模
        for layer in self.temporal_layers:
            frame_features = layer(frame_features)
        
        frame_features = self.norm(frame_features)
        
        # 时间池化
        z = self.out_proj(frame_features.mean(dim=1))
        
        return EncodedRepresentation(
            z=z,
            z_sequence=frame_features,
            modality=Modality.VIDEO
        )


# =============================================================================
# 序列编码器（数值序列）
# =============================================================================

class SequenceEncoder(nn.Module):
    """数值序列编码器
    
    处理时间序列数据（如您项目中的数值序列）
    """
    def __init__(self, d: int, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.d = d
        
        # 数值嵌入（保留数学结构）
        self.value_embed = nn.Sequential(
            nn.Linear(1, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        
        # 位置编码（傅里叶特征）
        self.pos_embed = FourierPositionEmbed(d)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(d, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d)
        
        # 规则推断头
        self.rule_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, d // 4),  # 规则嵌入
        )
        
        # 速度编码
        self.velocity_encoder = nn.Sequential(
            nn.Linear(1, d // 4),
            nn.SiLU(),
        )
    
    def forward(self, values: torch.Tensor) -> EncodedRepresentation:
        """
        values: [B, L] 数值序列
        
        返回: EncodedRepresentation
        """
        B, L = values.shape
        
        # 数值嵌入
        v = values.unsqueeze(-1)  # [B, L, 1]
        x = self.value_embed(v)  # [B, L, D]
        
        # 添加位置编码
        x = x + self.pos_embed(L, x.device)
        
        # 因果掩码
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=x.device), 1
        )
        
        # Transformer
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        x = self.norm(x)
        
        # 提取规则参数（从前几个数）
        if L >= 3:
            rule_emb = self.rule_head(x[:, :3].mean(dim=1))
        else:
            rule_emb = self.rule_head(x.mean(dim=1))
        
        # 计算速度（差分）
        velocity = torch.zeros_like(v)
        velocity[:, 1:] = v[:, 1:] - v[:, :-1]
        velocity_emb = self.velocity_encoder(velocity).mean(dim=1)  # [B, D//4]
        
        # 组合全局表示
        z_global = x.mean(dim=1)  # [B, D]
        
        return EncodedRepresentation(
            z=z_global,
            z_sequence=x,
            modality=Modality.SEQUENCE,
            attention_weights=None
        )


class FourierPositionEmbed(nn.Module):
    """傅里叶位置嵌入"""
    def __init__(self, d: int, max_len: int = 1024):
        super().__init__()
        self.d = d
        
        # 预计算傅里叶特征
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        
        pe = torch.zeros(max_len, d)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, length: int, device: torch.device) -> torch.Tensor:
        return self.pe[:length].to(device).unsqueeze(0)


# =============================================================================
# 统一多模态编码器
# =============================================================================

class UnifiedEncoder(nn.Module):
    """统一多模态编码器
    
    将所有模态映射到同一个Z空间
    """
    def __init__(
        self,
        d: int = 256,
        vocab_size: int = 50000,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 8,
        num_layers: int = 4,
        num_heads: int = 8
    ):
        super().__init__()
        self.d = d
        
        # 模态特定编码器
        self.text_encoder = TextEncoder(vocab_size, d, num_layers, num_heads)
        self.image_encoder = ImageEncoder(img_size, patch_size, d, num_layers, num_heads)
        self.video_encoder = VideoEncoder(img_size, patch_size, d, num_layers, num_heads, num_frames)
        self.sequence_encoder = SequenceEncoder(d, num_layers // 2, num_heads // 2)
        
        # 模态对齐层
        self.alignment = ModalityAlignment(d)
        
        # 模态标识嵌入
        self.modality_embed = nn.Embedding(len(Modality), d // 4)
    
    def forward(self, inputs: List[ModalityInput]) -> Dict[Modality, EncodedRepresentation]:
        """
        编码多个模态输入
        
        inputs: 模态输入列表
        
        返回: {模态: 编码表示}
        """
        results = {}
        
        for inp in inputs:
            if inp.modality == Modality.TEXT:
                enc = self.text_encoder(inp.data, inp.mask)
            elif inp.modality == Modality.IMAGE:
                enc = self.image_encoder(inp.data)
            elif inp.modality == Modality.VIDEO:
                enc = self.video_encoder(inp.data)
            elif inp.modality == Modality.SEQUENCE:
                enc = self.sequence_encoder(inp.data)
            else:
                raise ValueError(f"Unknown modality: {inp.modality}")
            
            # 添加模态标识
            modality_idx = list(Modality).index(inp.modality)
            modality_emb = self.modality_embed(torch.tensor([modality_idx], device=enc.z.device))
            
            # 对齐到统一空间
            enc = self.alignment.align(enc, modality_emb)
            
            results[inp.modality] = enc
        
        return results
    
    def encode_single(
        self, 
        data: torch.Tensor, 
        modality: Modality,
        mask: torch.Tensor = None
    ) -> EncodedRepresentation:
        """编码单个模态"""
        inp = ModalityInput(modality, data, mask)
        return self.forward([inp])[modality]


class ModalityAlignment(nn.Module):
    """模态对齐层
    
    将不同模态的表示对齐到共享空间
    """
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        
        # 对齐投影
        self.align_proj = nn.Sequential(
            nn.Linear(d + d // 4, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        
        # 归一化
        self.norm = RMSNorm(d)
    
    def align(
        self, 
        enc: EncodedRepresentation, 
        modality_emb: torch.Tensor
    ) -> EncodedRepresentation:
        """对齐表示"""
        B = enc.z.shape[0]
        
        # 拼接模态嵌入
        modality_emb = modality_emb.expand(B, -1)
        combined = torch.cat([enc.z, modality_emb], dim=-1)
        
        # 投影对齐
        z_aligned = self.align_proj(combined)
        z_aligned = self.norm(z_aligned)
        
        return EncodedRepresentation(
            z=z_aligned,
            z_sequence=enc.z_sequence,
            modality=enc.modality,
            attention_weights=enc.attention_weights
        )


# =============================================================================
# 对比学习对齐
# =============================================================================

class ContrastiveAlignment(nn.Module):
    """对比学习对齐
    
    通过对比学习对齐不同模态
    """
    def __init__(self, d: int, temperature: float = 0.07):
        super().__init__()
        self.d = d
        self.temperature = temperature
        
        # 投影头
        self.proj = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
    
    def forward(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算对比损失
        
        z1, z2: [B, D] 两个模态的表示（应该是配对的）
        
        返回: (损失, 相似度矩阵)
        """
        # 投影
        z1 = F.normalize(self.proj(z1), dim=-1)
        z2 = F.normalize(self.proj(z2), dim=-1)
        
        # 相似度矩阵
        sim = z1 @ z2.T / self.temperature
        
        # 对比损失（InfoNCE）
        B = z1.shape[0]
        labels = torch.arange(B, device=z1.device)
        
        loss_12 = F.cross_entropy(sim, labels)
        loss_21 = F.cross_entropy(sim.T, labels)
        
        loss = (loss_12 + loss_21) / 2
        
        return loss, sim


# =============================================================================
# Z空间结构
# =============================================================================

class ZSpaceStructure(nn.Module):
    """Z空间结构模块
    
    将编码后的表示组织成模块化结构：
    Z = Z_position ⊕ Z_velocity ⊕ Z_field ⊕ Z_modality
    """
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        
        # 子空间维度
        self.d_position = d // 4   # 位置/状态
        self.d_velocity = d // 4   # 速度/变化
        self.d_field = d // 4      # 场/规则
        self.d_modality = d // 4   # 模态信息
        
        # 子空间投影
        self.position_proj = nn.Linear(d, self.d_position)
        self.velocity_proj = nn.Linear(d, self.d_velocity)
        self.field_proj = nn.Linear(d, self.d_field)
        self.modality_proj = nn.Linear(d, self.d_modality)
    
    def decompose(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """将z分解为子空间"""
        return {
            'position': self.position_proj(z),
            'velocity': self.velocity_proj(z),
            'field': self.field_proj(z),
            'modality': self.modality_proj(z),
        }
    
    def compose(self, components: Dict[str, torch.Tensor]) -> torch.Tensor:
        """从子空间组合z"""
        return torch.cat([
            components['position'],
            components['velocity'],
            components['field'],
            components['modality'],
        ], dim=-1)
