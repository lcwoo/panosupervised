import torch
import torch.nn as nn
import numpy as np
from functools import partial
from timm.models.layers import trunc_normal_
from einops import rearrange

class ConvolutionalVisionTransformerEncoder(nn.Module):
    """
    Convolutional Vision Transformer Encoder
    
    Parameters
    ----------
    cfg : Config
        Configuration with parameters
    """
    def __init__(self, cfg):
        super().__init__()

        # CVT의 다중 스테이지를 위한 설정
        self.num_stages = cfg.num_stages  # Number of stages
        self.embed_dims = cfg.embed_dims  # Embedding dimensions for each stage
        self.num_heads = cfg.num_heads    # Number of attention heads
        self.depths = cfg.depths          # Number of blocks in each stage
        self.patch_sizes = cfg.patch_sizes
        self.patch_strides = cfg.patch_strides
        self.patch_paddings = cfg.patch_paddings
        self.mlp_ratios = cfg.mlp_ratios

        self.drop_rate = cfg.drop_rate
        self.attn_drop_rate = cfg.attn_drop_rate
        self.drop_path_rate = cfg.drop_path_rate
        self.qkv_bias = cfg.qkv_bias

        # Positional embedding and CLS token initialization
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims[-1])) if cfg.cls_token else None
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # 각 스테이지별 Vision Transformer 블록 생성
        self.blocks = nn.ModuleList()
        for i in range(self.num_stages):
            stage = VisionTransformerBlock(
                embed_dim=self.embed_dims[i],
                num_heads=self.num_heads[i],
                mlp_ratio=self.mlp_ratios[i],
                depth=self.depths[i],
                patch_size=self.patch_sizes[i],
                patch_stride=self.patch_strides[i],
                patch_padding=self.patch_paddings[i],
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                drop_path_rate=self.drop_path_rate,
                qkv_bias=self.qkv_bias,
            )
            self.blocks.append(stage)

        # 입력 이미지를 정규화하기 위한 평균 및 표준편차 버퍼
        self.register_buffer('input_mean', torch.FloatTensor(cfg.input_mean).view(1, 3, 1, 1))
        self.register_buffer('input_std', torch.FloatTensor(cfg.input_std).view(1, 3, 1, 1))

        self.reduction = np.array([4, 8, 16, 32])  # 각 스테이지별 다운샘플링 비율
        self.num_ch_enc = np.array(self.embed_dims)

    def forward(self, input_image):
        # 입력 이미지 정규화
        x_i = (input_image - self.input_mean) / self.input_std

        features = []
        for i, block in enumerate(self.blocks):
            x, cls_token = block(x_i)
            features.append(x)

        return features

class VisionTransformerBlock(nn.Module):
    """
    Single Vision Transformer block for Convolutional Vision Transformer Encoder.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio, depth, patch_size, patch_stride, patch_padding, drop_rate, attn_drop_rate, drop_path_rate, qkv_bias):
        super().__init__()

        # Conv patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=embed_dim,
            kernel_size=patch_size, stride=patch_stride, padding=patch_padding
        )

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate
                )
            )

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        for block in self.blocks:
            x = block(x, H, W)

        return x, None


class TransformerBlock(nn.Module):
    """
    Single transformer block for Convolutional Vision Transformer
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, drop_path_rate):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, bias=qkv_bias, dropout=attn_drop_rate)
        self.drop_path = nn.Dropout(drop_path_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), drop=drop_rate)

    def forward(self, x, h, w):
        # Attention
        attn_output, _ = self.attn(x, x, x)
        x = x + self.drop_path(attn_output)
        x = self.norm1(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Mlp(nn.Module):
    """
    MLP layer used inside the transformer block
    """
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
