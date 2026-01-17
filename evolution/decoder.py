"""多模态解码器

将Z空间表示解码回不同模态（语言、文本、图像、视频）

核心设计：
1. 模态特定解码器 - 生成每种模态的输出
2. 条件生成 - 基于Z空间状态生成
3. 自回归/并行解码
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .encoder import Modality, RMSNorm, FeedForward, TransformerBlock


# =============================================================================
# 解码输出类型
# =============================================================================

@dataclass
class DecodedOutput:
    """解码输出"""
    modality: Modality
    data: torch.Tensor              # 主输出
    logits: Optional[torch.Tensor] = None  # logits（如果适用）
    attention_weights: Optional[torch.Tensor] = None


# =============================================================================
# 文本解码器
# =============================================================================

class TextDecoder(nn.Module):
    """文本解码器
    
    从Z空间解码生成文本
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
        self.vocab_size = vocab_size
        
        # Token嵌入
        self.token_embed = nn.Embedding(vocab_size, d)
        
        # Z条件投影
        self.z_proj = nn.Linear(d, d)
        
        # 交叉注意力（用于条件生成）
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(d, num_heads) for _ in range(num_layers)
        ])
        
        # 自注意力层
        self.self_attn_layers = nn.ModuleList([
            TransformerBlock(d, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d)
        
        # LM头
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # 权重绑定
        
        # 因果掩码
        self.register_buffer("causal_mask", torch.triu(
            torch.full((max_len, max_len), float('-inf')), 1
        ))
    
    def forward(
        self,
        z: torch.Tensor,
        target_ids: torch.Tensor = None,
        max_len: int = 100
    ) -> DecodedOutput:
        """
        z: [B, D] Z空间表示
        target_ids: [B, L] 目标token IDs（训练时）
        max_len: 最大生成长度
        
        返回: DecodedOutput
        """
        B = z.shape[0]
        device = z.device
        
        # 条件向量
        z_cond = self.z_proj(z).unsqueeze(1)  # [B, 1, D]
        
        if target_ids is not None:
            # 训练模式：教师强制
            return self._forward_train(z_cond, target_ids)
        else:
            # 推理模式：自回归生成
            return self._forward_generate(z_cond, max_len)
    
    def _forward_train(
        self, 
        z_cond: torch.Tensor, 
        target_ids: torch.Tensor
    ) -> DecodedOutput:
        """训练前向"""
        B, L = target_ids.shape
        
        # 嵌入
        x = self.token_embed(target_ids)  # [B, L, D]
        
        # 因果掩码
        mask = self.causal_mask[:L, :L]
        
        # 交叉注意力 + 自注意力
        for cross_attn, self_attn in zip(self.cross_attn_layers, self.self_attn_layers):
            x = x + cross_attn(x, z_cond, z_cond)
            x = self_attn(x, mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)  # [B, L, V]
        
        return DecodedOutput(
            modality=Modality.TEXT,
            data=logits.argmax(dim=-1),
            logits=logits
        )
    
    @torch.no_grad()
    def _forward_generate(
        self, 
        z_cond: torch.Tensor, 
        max_len: int
    ) -> DecodedOutput:
        """自回归生成"""
        B = z_cond.shape[0]
        device = z_cond.device
        
        # 初始化：BOS token（假设为0）
        generated = torch.zeros(B, 1, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            x = self.token_embed(generated)
            L = x.shape[1]
            mask = self.causal_mask[:L, :L]
            
            for cross_attn, self_attn in zip(self.cross_attn_layers, self.self_attn_layers):
                x = x + cross_attn(x, z_cond, z_cond)
                x = self_attn(x, mask)
            
            x = self.norm(x)
            logits = self.lm_head(x[:, -1:])  # [B, 1, V]
            
            # 采样下一个token
            probs = F.softmax(logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs.squeeze(1), 1)  # [B, 1]
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # EOS检查
            if (next_token == 1).all():  # 假设1是EOS
                break
        
        return DecodedOutput(
            modality=Modality.TEXT,
            data=generated
        )


class CrossAttention(nn.Module):
    """交叉注意力"""
    def __init__(self, d: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        
        self.norm = RMSNorm(d)
    
    def forward(
        self, 
        x: torch.Tensor,      # [B, L, D] 查询
        key: torch.Tensor,    # [B, S, D] 键
        value: torch.Tensor   # [B, S, D] 值
    ) -> torch.Tensor:
        B, L, D = x.shape
        S = key.shape[1]
        
        x = self.norm(x)
        
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        out = (attn @ V).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


# =============================================================================
# 图像解码器
# =============================================================================

class ImageDecoder(nn.Module):
    """图像解码器
    
    从Z空间解码生成图像
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        d: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        out_channels: int = 3
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.d = d
        self.num_patches = (img_size // patch_size) ** 2
        
        # Z投影
        self.z_proj = nn.Linear(d, d)
        
        # 查询嵌入（可学习的图像块查询）
        self.query_embed = nn.Parameter(torch.randn(1, self.num_patches, d) * 0.02)
        
        # 位置嵌入
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d) * 0.02)
        
        # 交叉注意力层
        self.cross_layers = nn.ModuleList([
            CrossAttention(d, num_heads) for _ in range(num_layers // 2)
        ])
        
        # 自注意力层
        self.self_layers = nn.ModuleList([
            TransformerBlock(d, num_heads) for _ in range(num_layers // 2)
        ])
        
        self.norm = RMSNorm(d)
        
        # Patch解码器
        self.patch_decoder = nn.Linear(d, patch_size * patch_size * out_channels)
        
        self.out_channels = out_channels
    
    def forward(self, z: torch.Tensor) -> DecodedOutput:
        """
        z: [B, D] Z空间表示
        
        返回: DecodedOutput，data为 [B, C, H, W]
        """
        B = z.shape[0]
        
        # Z条件
        z_cond = self.z_proj(z).unsqueeze(1)  # [B, 1, D]
        
        # 查询
        queries = self.query_embed.expand(B, -1, -1) + self.pos_embed
        
        # 交叉注意力（从Z获取信息）
        for cross_layer in self.cross_layers:
            queries = queries + cross_layer(queries, z_cond, z_cond)
        
        # 自注意力（patch间交互）
        for self_layer in self.self_layers:
            queries = self_layer(queries)
        
        queries = self.norm(queries)
        
        # 解码到像素
        patches = self.patch_decoder(queries)  # [B, num_patches, P*P*C]
        
        # 重组为图像
        H = W = self.img_size // self.patch_size
        patches = patches.view(B, H, W, self.patch_size, self.patch_size, self.out_channels)
        patches = patches.permute(0, 5, 1, 3, 2, 4)  # [B, C, H, P, W, P]
        img = patches.reshape(B, self.out_channels, self.img_size, self.img_size)
        
        return DecodedOutput(
            modality=Modality.IMAGE,
            data=img
        )


# =============================================================================
# 视频解码器
# =============================================================================

class VideoDecoder(nn.Module):
    """视频解码器
    
    从Z空间解码生成视频帧序列
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
        self.num_frames = num_frames
        self.d = d
        
        # 帧解码器（共享）
        self.frame_decoder = ImageDecoder(img_size, patch_size, d, num_layers, num_heads)
        
        # 时间查询
        self.temporal_query = nn.Parameter(torch.randn(1, num_frames, d) * 0.02)
        
        # Z投影
        self.z_proj = nn.Linear(d, d)
        
        # 时间Transformer
        self.temporal_layers = nn.ModuleList([
            TransformerBlock(d, num_heads) for _ in range(num_layers // 2)
        ])
        
        self.norm = RMSNorm(d)
    
    def forward(self, z: torch.Tensor) -> DecodedOutput:
        """
        z: [B, D] Z空间表示
        
        返回: DecodedOutput，data为 [B, T, C, H, W]
        """
        B = z.shape[0]
        
        # Z条件
        z_cond = self.z_proj(z).unsqueeze(1)  # [B, 1, D]
        
        # 时间查询
        temporal_queries = self.temporal_query.expand(B, -1, -1)  # [B, T, D]
        
        # 添加Z条件
        temporal_queries = temporal_queries + z_cond
        
        # 时间建模
        for layer in self.temporal_layers:
            temporal_queries = layer(temporal_queries)
        
        temporal_queries = self.norm(temporal_queries)  # [B, T, D]
        
        # 解码每一帧
        frames = []
        for t in range(self.num_frames):
            z_t = temporal_queries[:, t]  # [B, D]
            frame = self.frame_decoder(z_t)
            frames.append(frame.data)
        
        video = torch.stack(frames, dim=1)  # [B, T, C, H, W]
        
        return DecodedOutput(
            modality=Modality.VIDEO,
            data=video
        )


# =============================================================================
# 序列解码器（数值序列）
# =============================================================================

class SequenceDecoder(nn.Module):
    """数值序列解码器
    
    从Z空间解码生成数值序列
    """
    def __init__(self, d: int, num_layers: int = 2, num_heads: int = 4, max_len: int = 100):
        super().__init__()
        self.d = d
        self.max_len = max_len
        
        # Z投影
        self.z_proj = nn.Linear(d, d)
        
        # 位置查询
        self.pos_query = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        
        # 交叉注意力
        self.cross_layers = nn.ModuleList([
            CrossAttention(d, num_heads) for _ in range(num_layers)
        ])
        
        # 自注意力
        self.self_layers = nn.ModuleList([
            TransformerBlock(d, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d)
        
        # 值预测头
        self.value_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.SiLU(),
            nn.Linear(d // 2, 1),
        )
    
    def forward(self, z: torch.Tensor, length: int = None) -> DecodedOutput:
        """
        z: [B, D] Z空间表示
        length: 生成长度
        
        返回: DecodedOutput，data为 [B, L]
        """
        B = z.shape[0]
        L = length or self.max_len
        
        # Z条件
        z_cond = self.z_proj(z).unsqueeze(1)  # [B, 1, D]
        
        # 位置查询
        queries = self.pos_query[:, :L].expand(B, -1, -1)  # [B, L, D]
        
        # 因果掩码
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=z.device), 1
        )
        
        # 交叉注意力 + 自注意力
        for cross_layer, self_layer in zip(self.cross_layers, self.self_layers):
            queries = queries + cross_layer(queries, z_cond, z_cond)
            queries = self_layer(queries, causal_mask)
        
        queries = self.norm(queries)
        
        # 预测值
        values = self.value_head(queries).squeeze(-1)  # [B, L]
        
        return DecodedOutput(
            modality=Modality.SEQUENCE,
            data=values
        )


# =============================================================================
# 统一多模态解码器
# =============================================================================

class UnifiedDecoder(nn.Module):
    """统一多模态解码器
    
    从Z空间解码到任意模态
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
        
        # 模态特定解码器
        self.text_decoder = TextDecoder(vocab_size, d, num_layers, num_heads)
        self.image_decoder = ImageDecoder(img_size, patch_size, d, num_layers, num_heads)
        self.video_decoder = VideoDecoder(img_size, patch_size, d, num_layers, num_heads, num_frames)
        self.sequence_decoder = SequenceDecoder(d, num_layers // 2, num_heads // 2)
        
        # 模态选择器
        self.modality_selector = nn.Linear(d, len(Modality))
    
    def forward(
        self, 
        z: torch.Tensor, 
        target_modality: Modality,
        **kwargs
    ) -> DecodedOutput:
        """
        z: [B, D] Z空间表示
        target_modality: 目标模态
        
        返回: DecodedOutput
        """
        if target_modality == Modality.TEXT:
            return self.text_decoder(z, **kwargs)
        elif target_modality == Modality.IMAGE:
            return self.image_decoder(z)
        elif target_modality == Modality.VIDEO:
            return self.video_decoder(z)
        elif target_modality == Modality.SEQUENCE:
            return self.sequence_decoder(z, **kwargs)
        else:
            raise ValueError(f"Unknown modality: {target_modality}")
    
    def predict_modality(self, z: torch.Tensor) -> torch.Tensor:
        """预测最适合的输出模态"""
        return F.softmax(self.modality_selector(z), dim=-1)
    
    def decode_all(self, z: torch.Tensor) -> Dict[Modality, DecodedOutput]:
        """解码到所有模态（用于多模态生成）"""
        results = {}
        for modality in [Modality.TEXT, Modality.IMAGE, Modality.SEQUENCE]:
            try:
                results[modality] = self.forward(z, modality)
            except Exception:
                pass
        return results


# =============================================================================
# 条件生成模块
# =============================================================================

class ConditionalGenerator(nn.Module):
    """条件生成器
    
    支持多种条件生成模式：
    1. 无条件生成
    2. 跨模态条件生成
    3. 属性条件生成
    """
    def __init__(self, d: int, num_conditions: int = 10):
        super().__init__()
        self.d = d
        
        # 条件嵌入
        self.condition_embed = nn.Embedding(num_conditions, d)
        
        # 条件融合
        self.fusion = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        
        # 无条件嵌入（用于 classifier-free guidance）
        self.uncond_embed = nn.Parameter(torch.randn(1, d) * 0.02)
    
    def apply_condition(
        self, 
        z: torch.Tensor, 
        condition: torch.Tensor = None,
        condition_scale: float = 1.0
    ) -> torch.Tensor:
        """应用条件
        
        z: [B, D] 输入表示
        condition: [B] 条件索引 或 [B, D] 条件向量
        condition_scale: 条件强度（用于 classifier-free guidance）
        """
        B = z.shape[0]
        
        if condition is None:
            # 无条件
            cond_emb = self.uncond_embed.expand(B, -1)
        elif condition.dim() == 1:
            # 索引条件
            cond_emb = self.condition_embed(condition)
        else:
            # 向量条件
            cond_emb = condition
        
        # 融合
        combined = torch.cat([z, cond_emb], dim=-1)
        z_cond = self.fusion(combined)
        
        # Classifier-free guidance
        if condition_scale != 1.0 and condition is not None:
            uncond_combined = torch.cat([z, self.uncond_embed.expand(B, -1)], dim=-1)
            z_uncond = self.fusion(uncond_combined)
            z_cond = z_uncond + condition_scale * (z_cond - z_uncond)
        
        return z_cond


# =============================================================================
# 自回归包装器
# =============================================================================

class AutoregressiveWrapper(nn.Module):
    """自回归包装器
    
    将任意解码器包装为自回归生成器
    """
    def __init__(self, decoder: nn.Module, d: int):
        super().__init__()
        self.decoder = decoder
        self.d = d
        
        # 状态更新网络
        self.state_update = nn.GRUCell(d, d)
    
    def forward(
        self, 
        z: torch.Tensor,
        num_steps: int,
        **kwargs
    ) -> List[DecodedOutput]:
        """
        自回归生成多步
        
        z: [B, D] 初始Z状态
        num_steps: 生成步数
        """
        outputs = []
        state = z
        
        for step in range(num_steps):
            # 解码当前步
            output = self.decoder(state, **kwargs)
            outputs.append(output)
            
            # 更新状态
            # 从输出中提取特征（这里简化处理）
            output_features = state  # 实际中应该从output.data提取
            state = self.state_update(output_features, state)
        
        return outputs
