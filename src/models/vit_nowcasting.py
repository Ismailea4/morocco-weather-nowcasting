"""
Vision Transformer-based nowcasting model.
Replace CNN with ViT encoder for spatial features; add temporal fusion.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ViTNowcasterConfig:
    """Configuration for ViT Nowcaster."""
    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 4
    out_channels: int = 4
    t_out: int = 2
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0
    temporal_fusion: str = 'attention'  # 'attention', 'avg', or 'last'
    temporal_heads: int = 4


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them."""
    
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block with multi-head attention and MLP."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (B, N, embed_dim)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalFusion(nn.Module):
    """Fuse temporal information from multiple frames."""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, fusion_type: str = 'attention'):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            self.temporal_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (B, T, N, embed_dim)
        B, T, N, D = x.shape
        
        if self.fusion_type == 'attention':
            # Flatten temporal and spatial dimensions
            x_flat = x.reshape(B, T * N, D)
            x_fused = self.temporal_attn(x_flat, x_flat, x_flat)[0]
            x_fused = self.norm(x_fused)
            x_fused = x_fused.reshape(B, T, N, D)
            return x_fused
        elif self.fusion_type == 'avg':
            return x.mean(dim=1, keepdim=True).expand(-1, T, -1, -1)
        elif self.fusion_type == 'last':
            return x[:, -1:, :, :].expand(-1, T, -1, -1)
        else:
            return x


class ViTNowcaster(nn.Module):
    """Vision Transformer for weather nowcasting."""
    
    def __init__(self, config: ViTNowcasterConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            config.image_size, config.patch_size, 
            config.in_channels, config.embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.embed_dim, config.num_heads, 
                config.mlp_ratio, config.dropout, config.attn_dropout
            ) for _ in range(config.depth)
        ])
        
        # Temporal fusion
        self.temporal_fusion = TemporalFusion(
            config.embed_dim, config.temporal_heads, config.temporal_fusion
        )
        
        # Output projection
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.out_channels * config.patch_size ** 2)
        
        self.t_out = config.t_out
        
        # Learnable query tokens for future frames
        self.future_queries = nn.Parameter(torch.randn(1, config.t_out, num_patches, config.embed_dim))
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.future_queries, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def unpatchify(self, x):
        """Convert patches back to images.
        
        Args:
            x: (B, T, num_patches, patch_size^2 * C)
        
        Returns:
            images: (B, T, C, H, W)
        """
        B, T, N, _ = x.shape
        p = self.config.patch_size
        h = w = int(N ** 0.5)
        c = self.config.out_channels
        
        x = x.reshape(B, T, h, w, p, p, c)
        x = x.permute(0, 1, 6, 2, 4, 3, 5)  # (B, T, C, h, p, w, p)
        x = x.reshape(B, T, c, h * p, w * p)
        return x
    
    def forward(self, x):
        """
        Args:
            x: Input frames of shape (B, T_in, C, H, W)
        
        Returns:
            predictions: (B, T_out, C, H, W)
        """
        B, T_in, C, H, W = x.shape
        
        # Embed each frame
        patches = []
        for t in range(T_in):
            patch_t = self.patch_embed(x[:, t])  # (B, N, embed_dim)
            patch_t = patch_t + self.pos_embed
            patches.append(patch_t)
        
        patches = torch.stack(patches, dim=1)  # (B, T_in, N, embed_dim)
        
        # Apply transformer blocks
        B, T, N, D = patches.shape
        patches_flat = patches.reshape(B * T, N, D)
        
        for block in self.blocks:
            patches_flat = block(patches_flat)
        
        patches = patches_flat.reshape(B, T, N, D)
        
        # Temporal fusion
        patches = self.temporal_fusion(patches)
        
        # Use future queries for prediction
        future_tokens = self.future_queries.expand(B, -1, -1, -1)  # (B, T_out, N, D)
        
        # Combine with encoded context (simple approach: use last frame as context)
        context = patches[:, -1:, :, :].expand(-1, self.t_out, -1, -1)
        pred_tokens = future_tokens + context
        
        # Project to output
        pred_tokens_flat = pred_tokens.reshape(B * self.t_out, N, D)
        pred_tokens_flat = self.norm(pred_tokens_flat)
        pred_patches = self.head(pred_tokens_flat)  # (B*T_out, N, p^2*C)
        pred_patches = pred_patches.reshape(B, self.t_out, N, -1)
        
        # Unpatchify
        output = self.unpatchify(pred_patches)
        
        return output
