import argparse, torch
torch.serialization.add_safe_globals([argparse.Namespace])

import torch.nn as nn
import torchvision.models as models
from .convnextv2.convnextv2 import (
    convnextv2_tiny,
    convnextv2_base,
    convnextv2_large
)
from mambavision import create_model

class EfficientNetWithTransformer(nn.Module):
    def __init__(self, num_points: int, image_size: int = 224,
                 hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 1):
        super().__init__()

        # Load EfficientNetV2-M backbone
        base = models.efficientnet_v2_m(pretrained=True)
        self.backbone = base.features  # Remove classifier
        self.input_dim = 1280
        
        self.pool = nn.AdaptiveAvgPool2d((7, 7))  # Ensure fixed size output
        self.seq_len = 7 * 7  # 49 tokens

        # Projection: EfficientNet output to Transformer hidden dim
        self.projector = nn.Linear(self.input_dim, hidden_dim)

        # Positional encoding (scale down to avoid NaN)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim) * 0.01)

        # LayerNorm to stabilize before Transformer
        self.pre_norm = nn.LayerNorm(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation='gelu',  # More stable than ReLU
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output regression head
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_dim * self.seq_len),  # stabilize final layer
            nn.Linear(hidden_dim * self.seq_len, num_points * 2)
        )

    def forward(self, x):
        x = self.backbone(x)               # [B, 1280, 7, 7]
        x = self.pool(x)                   # [B, 1280, 7, 7]
        x = x.flatten(2).transpose(1, 2)   # [B, 49, 1280]
        x = self.projector(x)             # [B, 49, hidden_dim]
        x = self.pre_norm(x)              # LayerNorm before Transformer
        x = x + self.positional_encoding  # Add position info
        x = self.transformer(x)           # [B, 49, hidden_dim]
        out = self.output_head(x)         # [B, num_points * 2]
        return out

class EfficientNetMidWithTransformer(nn.Module):
    def __init__(self, num_points: int, image_size: int = 224,
                 hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 1):
        super().__init__()

        # Load EfficientNetV2-M and extract mid-level feature block
        base = models.efficientnet_v2_m(pretrained=True)
        self.backbone = nn.Sequential(*list(base.features.children())[:6])  # Output: [B, 176, 14, 14]
        self.input_dim = 176

        self.pool = nn.AdaptiveAvgPool2d((14, 14))  # Ensure fixed size
        self.seq_len = 14 * 14  # 196 tokens

        # Project to transformer dimension
        self.projector = nn.Linear(self.input_dim, hidden_dim)

        # Learnable positional encoding (scaled to prevent NaNs)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim) * 0.01)

        # Pre-transformer normalization
        self.pre_norm = nn.LayerNorm(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression head
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_dim * self.seq_len),
            nn.Linear(hidden_dim * self.seq_len, num_points * 2)
        )

    def forward(self, x):
        x = self.backbone(x)                 # [B, 176, 14, 14]
        x = self.pool(x)                     # [B, 176, 14, 14]
        x = x.flatten(2).transpose(1, 2)     # [B, 196, 176]
        x = self.projector(x)                # [B, 196, hidden_dim]
        x = self.pre_norm(x)
        x = x + self.positional_encoding
        x = self.transformer(x)              # [B, 196, hidden_dim]
        out = self.output_head(x)            # [B, num_points * 2]
        return out

class PatchTransformer(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, nhead=4, nlayers=1, hw=None):
        super().__init__()
        assert hw is not None, "Please pass hw=(H,W) for positional enc length."
        H, W = hw
        self.seq_len = H * W
        self.proj = nn.Linear(in_channels, hidden_dim)

        # learnable pos enc（縮小初始化，避免數值暴衝）
        self.pos = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim) * 0.01)
        self.prenorm = nn.LayerNorm(hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 4, activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

    def forward(self, x):                 # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x = self.proj(x)                  # [B, HW, hidden]
        x = self.prenorm(x)
        x = x + self.pos[:, :x.size(1), :]
        x = self.encoder(x)               # [B, HW, hidden]
        x = x.mean(dim=1)                 # mean pooling over tokens → [B, hidden]
        return x

class EfficientNetMultiScaleTransformer_4scales_GAPConcat(nn.Module):
    """
    Multi-Scale (last 4 MBConv) + Transformer per scale
    融合方式：Mean Pooling + Concat
    """
    def __init__(self, num_points: int, image_size: int = 224,
                 hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 1):
        super().__init__()

        base = models.efficientnet_v2_m(pretrained=True)
        feats = base.features  # torchvision 的順序如下（對 224x224）：
        # 0: stem (Conv2dNormActivation)
        # 1: seq (24,112x112)
        # 2: seq (48,56x56)
        # 3: seq (80,28x28)
        # 4: seq (160,14x14)  ← MBConv block A
        # 5: seq (176,14x14)  ← MBConv block B
        # 6: seq (304, 7x7)   ← MBConv block C
        # 7: seq (512, 7x7)   ← MBConv block D
        # 8: Conv2dNormActivation (1280, 7x7)

        # 前導部分（跑到 block4 之前）
        self.stem_to_3 = nn.Sequential(*feats[:4])  # 到 80, 28x28

        # 四個要取出的 MBConv 區段（保持原模塊，逐段 forward）
        self.block4 = feats[4]  # ch=160, 14x14
        self.block5 = feats[5]  # ch=176, 14x14
        self.block6 = feats[6]  # ch=304,  7x7
        self.block7 = feats[7]  # ch=512,  7x7

        # 各層對應的 Transformer（固定 token 長度：14x14=196、7x7=49）
        self.trans4 = PatchTransformer(in_channels=160, hidden_dim=hidden_dim,
                                       nhead=num_heads, nlayers=num_layers, hw=(14, 14))
        self.trans5 = PatchTransformer(in_channels=176, hidden_dim=hidden_dim,
                                       nhead=num_heads, nlayers=num_layers, hw=(14, 14))
        self.trans6 = PatchTransformer(in_channels=304, hidden_dim=hidden_dim,
                                       nhead=num_heads, nlayers=num_layers, hw=(7, 7))
        self.trans7 = PatchTransformer(in_channels=512, hidden_dim=hidden_dim,
                                       nhead=num_heads, nlayers=num_layers, hw=(7, 7))

        # 融合後 head（MeanPool 後 concat → [B, 4*hidden_dim]）
        self.head = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, num_points * 2)
        )

    def forward(self, x):
        # 到 80/28x28
        x = self.stem_to_3(x)

        # 逐層跑 MBConv 並接各自的 Transformer
        f4 = self.block4(x)           # [B,160,14,14]
        t4 = self.trans4(f4)          # [B, hidden]

        f5 = self.block5(f4)          # [B,176,14,14]
        t5 = self.trans5(f5)          # [B, hidden]

        f6 = self.block6(f5)          # [B,304,7,7]
        t6 = self.trans6(f6)          # [B, hidden]

        f7 = self.block7(f6)          # [B,512,7,7]
        t7 = self.trans7(f7)          # [B, hidden]

        # Mean Pooling 後 concat
        fused = torch.cat([t4, t5, t6, t7], dim=1)  # [B, 4*hidden]
        out = self.head(fused)                      # [B, num_points*2]
        return out

class EfficientNetMultiScaleTransformer_3scales_GAPConcat(nn.Module):
    """
    Multi-Scale (last 3 MBConv) + Transformer per scale
    融合方式：Mean Pooling + Concat
    """
    def __init__(self, num_points: int, image_size: int = 224,
                 hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 1):
        super().__init__()

        base = models.efficientnet_v2_m(pretrained=True)
        feats = base.features  # torchvision 的順序如下（對 224x224）：
        # 0: stem (Conv2dNormActivation)
        # 1: seq (24,112x112)
        # 2: seq (48,56x56)
        # 3: seq (80,28x28)
        # 4: seq (160,14x14)  
        # 5: seq (176,14x14)
        # 6: seq (304, 7x7)   ← MBConv block A
        # 7: seq (512, 7x7)   ← MBConv block B
        # 8: Conv2dNormActivation (1280, 7x7) ← MBConv block C

        # 前導部分（跑到 block4 之前）
        self.stem_to_5 = nn.Sequential(*feats[:6]) # 到 176, 14x14

        # 四個要取出的 MBConv 區段（保持原模塊，逐段 forward）
        self.block6 = feats[6]  # ch=304,  7x7
        self.block7 = feats[7]  # ch=512,  7x7
        self.block8 = feats[8]  # ch=1280, 7x7

        # 各層對應的 Transformer（固定 token 長度：14x14=196、7x7=49）
        self.trans6 = PatchTransformer(in_channels=304, hidden_dim=hidden_dim,
                                       nhead=num_heads, nlayers=num_layers, hw=(7, 7))
        self.trans7 = PatchTransformer(in_channels=512, hidden_dim=hidden_dim,
                                       nhead=num_heads, nlayers=num_layers, hw=(7, 7))
        self.trans8 = PatchTransformer(in_channels=1280, hidden_dim=hidden_dim,
                                       nhead=num_heads, nlayers=num_layers, hw=(7, 7))

        # 融合後 head（MeanPool 後 concat → [B, 3*hidden_dim]）
        self.head = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, num_points * 2)
        )

    def forward(self, x):
        # 到 80/28x28
        x = self.stem_to_5(x)

        # 逐層跑 MBConv 並接各自的 Transformer
        f6 = self.block6(x)          # [B,304,7,7]
        t6 = self.trans6(f6)          # [B, hidden]

        f7 = self.block7(f6)          # [B,512,7,7]
        t7 = self.trans7(f7)          # [B, hidden]

        f8 = self.block8(f7)          # [B,1280,7,7]
        t8 = self.trans8(f8)          # [B, hidden]

        # Mean Pooling 後 concat
        fused = torch.cat([t6, t7, t8], dim=1)  # [B, 3*hidden]
        out = self.head(fused)                      # [B, num_points*2]
        return out
    
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        # Channel Attention
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.shared_mlp(self.channel_avg_pool(x))
        max_out = self.shared_mlp(self.channel_max_pool(x))
        channel_attention = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attention

        # Spatial Attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_pool, max_pool], dim=1)))
        x = x * spatial_attention
        return x

class EfficientNetWithCBAM(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        base = models.efficientnet_v2_m(pretrained=True)
        self.backbone = base.features
        self.cbam = CBAM(in_channels=1280)  # 最後一層輸出通道數
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(1280, num_points * 2)

    def forward(self, x):
        x = self.backbone(x)     # [B, 1280, H/32, W/32]
        x = self.cbam(x)         # CBAM 模組強化重要特徵
        x = self.pool(x)         # [B, 1280, 1, 1]
        x = x.view(x.size(0), -1)
        return self.head(x)

class EfficientNetMultiScaleCBAM_4scales_GAPConcat(nn.Module):
    """
    使用 EfficientNetV2-M 最後四個 MBConv 輸出 (C=160,176,304,512)，
    各自經過 CBAM 後做 Global Average Pooling，再 concat 成一個向量，
    送入 MLP head 回歸 keypoints。
    融合方式：GAP + Concat
    """
    def __init__(self, num_points: int):
        super().__init__()
        base = models.efficientnet_v2_m(pretrained=True)
        feats = base.features
        # 到 80ch, 28x28（block3）
        self.stem_to_3 = nn.Sequential(*feats[:4])
        # 最後四個 MBConv 區塊
        self.block4 = feats[4]  # 160, 14x14
        self.block5 = feats[5]  # 176, 14x14
        self.block6 = feats[6]  # 304,  7x7
        self.block7 = feats[7]  # 512,  7x7

        # 對應的 CBAM
        self.cbam4 = CBAM(160)
        self.cbam5 = CBAM(176)
        self.cbam6 = CBAM(304)
        self.cbam7 = CBAM(512)

        # 全域平均池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        fused_dim = 160 + 176 + 304 + 512  # 1152
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_points * 2)
        )

    def forward(self, x):
        x = self.stem_to_3(x)   # 80, 28x28
        f4 = self.block4(x)     # 160,14x14
        f5 = self.block5(f4)    # 176,14x14
        f6 = self.block6(f5)    # 304,7x7
        f7 = self.block7(f6)    # 512,7x7

        f4 = self.cbam4(f4)
        f5 = self.cbam5(f5)
        f6 = self.cbam6(f6)
        f7 = self.cbam7(f7)

        # GAP → flatten
        z4 = self.gap(f4).flatten(1)  # [B,160]
        z5 = self.gap(f5).flatten(1)  # [B,176]
        z6 = self.gap(f6).flatten(1)  # [B,304]
        z7 = self.gap(f7).flatten(1)  # [B,512]

        fused = torch.cat([z4, z5, z6, z7], dim=1)  # [B,1152]
        out = self.head(fused)                      # [B, num_points*2]
        return out

class EfficientNetMultiScaleCBAM_3scales_GAPConcat(nn.Module):
    """
    使用 EfficientNetV2-M 最後四個 MBConv 輸出 (C=160,176,304,512)，
    各自經過 CBAM 後做 Global Average Pooling，再 concat 成一個向量，
    送入 MLP head 回歸 keypoints。
    融合方式：GAP + Concat
    """
    def __init__(self, num_points: int):
        super().__init__()
        base = models.efficientnet_v2_m(pretrained=True)
        feats = base.features
        # 到 80ch, 28x28（block3）
        self.stem_to_5 = nn.Sequential(*feats[:6])
        # 最後三層
        self.block6 = feats[6]  # 304,  7x7
        self.block7 = feats[7]  # 512,  7x7
        self.block8 = feats[8]  # 1280,  7x7

        # 對應的 CBAM
        self.cbam6 = CBAM(304)
        self.cbam7 = CBAM(512)
        self.cbam8 = CBAM(1280)

        # 全域平均池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        fused_dim = 304 + 512 + 1280  # 1432
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_points * 2)
        )

    def forward(self, x):
        x = self.stem_to_5(x)   # 80, 28x28
        f6 = self.block6(x)    # 304,7x7
        f7 = self.block7(f6)    # 512,7x7
        f8 = self.block8(f7)    # 1280,7x7

        f6 = self.cbam6(f6)
        f7 = self.cbam7(f7)
        f8 = self.cbam8(f8)

        # GAP → flatten
        z6 = self.gap(f6).flatten(1)  # [B,304]
        z7 = self.gap(f7).flatten(1)  # [B,512]
        z8 = self.gap(f8).flatten(1)  # [B,1280]

        fused = torch.cat([z6, z7, z8], dim=1)  # [B, 2048]
        out = self.head(fused)                      # [B, num_points*2]
        return out

class ScaleHead(nn.Module):
    """\
    將各尺度特徵做：1x1降維→BN→SiLU→(殘差)CBAM→GAP→LayerNorm，
    輸出一個固定維度的向量供融合。
    """
    def __init__(self, in_ch: int, out_ch: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU()
        self.cbam = CBAM(out_ch)
        self.ln   = nn.LayerNorm(out_ch)

    def forward(self, x):                      # x: [B, C, H, W]
        x = self.proj(x)                       # [B, out, H, W]
        x = self.act(self.bn(x))               # 正規化+非線性
        y = self.cbam(x) + x                   # 殘差式 CBAM，避免過度抑制
        y = y.mean(dim=(-2, -1))               # GAP → [B, out]
        y = self.ln(y)                         # 穩定不同尺度的統計
        return y

class EfficientNetMultiScaleCBAM_3scales_GatedFusion(nn.Module):
    def __init__(self, num_points: int, embed_dim: int = 128):
        super().__init__()
        base  = models.efficientnet_v2_m(pretrained=True)
        feats = base.features
        self.stem_to_3 = nn.Sequential(*feats[:4])  # 到 80,28x28

        # 這裡要把 block4/5 也保留下來，forward 會經過它們
        self.block4 = feats[4]  # 160,14x14
        self.block5 = feats[5]  # 176,14x14
        self.block6 = feats[6]  # 304, 7x7（要用）
        self.block7 = feats[7]  # 512, 7x7（要用）
        self.head8  = feats[8]  # 1280,7x7（要用）

        self.h6 = ScaleHead(304, embed_dim)
        self.h7 = ScaleHead(512, embed_dim)
        self.h8 = ScaleHead(1280, embed_dim)

        self.gates = nn.Parameter(torch.ones(3))
        self.reg_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(4 * embed_dim, num_points * 2)
        )

    def forward(self, x):
        x  = self.stem_to_3(x)   # [B,80,28,28]
        x  = self.block4(x)      # [B,160,14,14]
        x  = self.block5(x)      # [B,176,14,14]

        f6 = self.block6(x)      # [B,304,7,7]
        f7 = self.block7(f6)     # [B,512,7,7]
        f8 = self.head8(f7)      # [B,1280,7,7]

        z6 = self.h6(f6)
        z7 = self.h7(f7)
        z8 = self.h8(f8)

        g  = torch.clamp(self.gates, 0.0, 3.0)
        fused = torch.cat([g[0]*z6, g[1]*z7, g[2]*z8], dim=1)
        return self.reg_head(fused)


class EfficientNetCBAMTransformer(nn.Module):
    """
    EfficientNetV2-M backbone → CBAM(1280ch) → 7x7 tokens → TransformerEncoder → Regression head
    流程：Backbone特徵增強(通道/空間注意力) → 全域關聯建模 → 直接回歸 keypoints
    """
    def __init__(self, num_points: int, image_size: int = 224,
                 hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 1):
        super().__init__()
        base = models.efficientnet_v2_m(pretrained=True)
        self.backbone = base.features
        self.cbam = CBAM(in_channels=1280)

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.seq_len = 7 * 7
        self.input_dim = 1280

        self.projector = nn.Linear(self.input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim) * 0.01)
        self.pre_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_dim * self.seq_len),
            nn.Linear(hidden_dim * self.seq_len, num_points * 2),
        )

    def forward(self, x):
        x = self.backbone(x)                # [B, 1280, 7, 7]
        x = self.cbam(x)                    # CBAM 強化重點特徵
        x = self.pool(x)                    # [B, 1280, 7, 7]（保險固定大小）
        x = x.flatten(2).transpose(1, 2)    # [B, 49, 1280]
        x = self.projector(x)               # [B, 49, hidden]
        x = self.pre_norm(x)
        x = x + self.positional_encoding
        x = self.transformer(x)             # [B, 49, hidden]
        out = self.output_head(x)           # [B, num_points*2]
        return out

class PosEnc(nn.Module):
    def __init__(self, seq_len, dim, scale=0.01):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, dim) * scale)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MSDecoder(nn.Module):
    """
    Query  : conv head (1280) → tokens [B,49,H]
    Memory : concat(block6(304), block7(512)) → tokens [B,98,H]
    Cross-Attn: Q attends to K,V (memory)
    """
    def __init__(self, c6=304, c7=512, c8=1280, hidden=256, nhead=4, nlayers=1):
        super().__init__()
        # 1x1 conv 將各尺度通道對齊到 hidden
        self.p6 = nn.Conv2d(c6, hidden, 1, bias=False)
        self.p7 = nn.Conv2d(c7, hidden, 1, bias=False)
        self.p8 = nn.Conv2d(c8, hidden, 1, bias=False)

        # 位置編碼（7x7 → 49 tokens；兩層 memory → 98 tokens）
        self.pos_q = PosEnc(49, hidden)
        self.pos_m = PosEnc(98, hidden)

        # Decoder layer：Self-Attn(在Q上) + Cross-Attn(看memory)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=nhead,
            dim_feedforward=hidden*4, activation="gelu",
            batch_first=True
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=nlayers)

        self.prenorm_q = nn.LayerNorm(hidden)
        self.prenorm_m = nn.LayerNorm(hidden)

    def to_tokens(self, x, proj):   # x: [B,C,7,7] → [B,49,H]
        x = proj(x)                 # [B,H,7,7]
        x = x.flatten(2).transpose(1, 2)
        return x

    def forward(self, f6, f7, f8):  # 三個都是 [B,C,7,7]
        q = self.to_tokens(f8, self.p8)     # [B,49,H]
        m6 = self.to_tokens(f6, self.p6)    # [B,49,H]
        m7 = self.to_tokens(f7, self.p7)    # [B,49,H]
        mem = torch.cat([m6, m7], dim=1)    # [B,98,H]

        q = self.prenorm_q(self.pos_q(q))
        mem = self.prenorm_m(self.pos_m(mem))

        # TransformerDecoder: output shape [B,49,H]
        out = self.dec(tgt=q, memory=mem)
        return out

class EfficientNetMultiScale_3scales_CrossAttn(nn.Module):
    def __init__(self, num_points: int, hidden: int = 256, nhead: int = 4, nlayers: int = 1):
        super().__init__()
        base  = models.efficientnet_v2_m(pretrained=True)
        feats = base.features
        self.stem_to_3 = nn.Sequential(*feats[:4])  # 到 80,28x28
        self.block4 = feats[4]                    # 80→176
        self.block5 = feats[5]                    # 176→176
        self.block6 = feats[6]                    # 176→304
        self.block7 = feats[7]                    # 304→512
        self.head8  = feats[8]                    # 512→1280

        self.ms_dec = MSDecoder(c6=304, c7=512, c8=1280, hidden=hidden, nhead=nhead, nlayers=nlayers)

        # 把 decoder 輸出的 49 個 token 做 mean pooling → [B,hidden]
        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2*hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2*hidden, num_points * 2)
        )

    def forward(self, x):
        x  = self.stem_to_3(x)
        x  = self.block4(x)
        x  = self.block5(x)
        f6 = self.block6(x)     # [B,304,7,7]
        f7 = self.block7(f6)    # [B,512,7,7]
        f8 = self.head8(f7)     # [B,1280,7,7]

        dec_out = self.ms_dec(f6, f7, f8)  # [B,49,H]
        z = dec_out.mean(dim=1)            # [B,H]
        return self.reg_head(z)

class EfficientNet(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = models.efficientnet_v2_m(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_points * 2)
        )
        self.model = model

    def forward(self, x):
        return self.model(x)
    
class EfficientNet_FC2048(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = models.efficientnet_v2_m(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_points * 2)
        )
        self.model = model

    def forward(self, x):
        return self.model(x)

class ConvNeXtWithCBAM(nn.Module):
    """
    ConvNeXt-Small backbone，最後一個 stage 輸出 [B, 768, H/32, W/32]
    -> 接 CBAM
    -> GAP (1x1)
    -> LayerNorm2d（沿用官方）
    -> Flatten + Linear(num_points*2)
    """
    def __init__(self, num_points: int, pretrained: bool = True):
        super().__init__()
        base = models.convnext_small(pretrained=pretrained)

        self.backbone = base.features                     # 到最後 stage 輸出 C=768
        self.cbam = CBAM(in_channels=768)                 # 接在最後 stage 之後
        self.pool = nn.AdaptiveAvgPool2d((1, 1))          # [B, 768, 1, 1]

        # 直接沿用官方 classifier 的 LayerNorm2d 作為歸一化
        self.norm = base.classifier[0]                    # LayerNorm2d(768, eps=1e-6)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(768, num_points * 2)

    def forward(self, x):
        x = self.backbone(x)      # [B, 768, h, w]
        x = self.cbam(x)          # CBAM 強化
        x = self.pool(x)          # [B, 768, 1, 1]
        x = self.norm(x)          # LayerNorm2d
        x = self.flatten(x)       # [B, 768]
        x = self.head(x)          # [B, 2*num_points]
        return x

class ConvNeXtWithTransformer(nn.Module):
    def __init__(self, num_points: int,
                 image_size: int = 224,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 1,
                 pretrained: bool = True):
        super().__init__()

        # 1. ConvNeXt Small Backbone
        base = models.convnext_small(pretrained=pretrained)
        self.backbone = base.features  # [B, 768, H/32, W/32]
        self.input_dim = 768

        # 2. Fix output size to 7x7 like EfficientNet
        self.pool = nn.AdaptiveAvgPool2d((7, 7))  # [B, 768, 7, 7]
        self.seq_len = 7 * 7

        # 3. Project to hidden dim
        self.projector = nn.Linear(self.input_dim, hidden_dim)

        # 4. Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.seq_len, hidden_dim) * 0.01
        )

        # 5. Normalize before Transformer
        self.pre_norm = nn.LayerNorm(hidden_dim)

        # 6. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 7. Output regression head
        self.output_head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_dim * self.seq_len),
            nn.Linear(hidden_dim * self.seq_len, num_points * 2)
        )

    def forward(self, x):
        x = self.backbone(x)                # [B, 768, H/32, W/32]
        x = self.pool(x)                    # [B, 768, 7, 7]
        x = x.flatten(2).transpose(1, 2)    # [B, 49, 768]
        x = self.projector(x)               # [B, 49, hidden_dim]
        x = self.pre_norm(x)
        x = x + self.positional_encoding    # [B, 49, hidden_dim]
        x = self.transformer(x)             # [B, 49, hidden_dim]
        out = self.output_head(x)           # [B, 2*num_points]
        return out

class ConvNeXtSmall(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = models.convnext_small(pretrained=True)
        in_features = model.classifier[2].in_features  # ConvNeXt classifier: [LayerNorm2d, Flatten, Linear]
        model.classifier[2] = nn.Linear(in_features, num_points * 2)
        self.model = model

    def forward(self, x):
        return self.model(x)
    
class ConvNeXtTiny(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = models.convnext_tiny(pretrained=True)
        in_features = model.classifier[2].in_features  # ConvNeXt classifier: [LayerNorm2d, Flatten, Linear]
        model.classifier[2] = nn.Linear(in_features, num_points * 2)
        self.model = model

    def forward(self, x):
        return self.model(x)    

class ConvNeXtBase(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = models.convnext_base(pretrained=True)
        in_features = model.classifier[2].in_features  # ConvNeXt classifier: [LayerNorm2d, Flatten, Linear]
        model.classifier[2] = nn.Linear(in_features, num_points * 2)
        self.model = model

    def forward(self, x):
        return self.model(x)

class ConvNeXtLarge(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = models.convnext_large(pretrained=True)
        in_features = model.classifier[2].in_features  # ConvNeXt classifier: [LayerNorm2d, Flatten, Linear]
        model.classifier[2] = nn.Linear(in_features, num_points * 2)
        self.model = model

    def forward(self, x):
        return self.model(x)

class ConvNeXtV2Tiny(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = convnextv2_tiny(num_classes=1000)
        pretrained_path = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt"
        state_dict = torch.hub.load_state_dict_from_url(pretrained_path, progress=True)
        model.load_state_dict(state_dict['model'])

        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_points * 2)
        self.model = model

    def forward(self, x):
        return self.model(x)

class ConvNeXtV2Base(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = convnextv2_base(num_classes=1000)
        pretrained_path = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt"
        state_dict = torch.hub.load_state_dict_from_url(pretrained_path, progress=True)
        model.load_state_dict(state_dict['model'])

        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_points * 2)
        self.model = model

    def forward(self, x):
        return self.model(x)

class ConvNeXtV2Large(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = convnextv2_large(num_classes=1000)
        pretrained_path = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt"
        state_dict = torch.hub.load_state_dict_from_url(pretrained_path, progress=True)
        model.load_state_dict(state_dict['model'])

        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_points * 2)
        self.model = model

    def forward(self, x):
        return self.model(x)

class MambaVisionSmall(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = create_model('mamba_vision_S', pretrained=True, model_path="/tmp/mambavision_small_1k.pth.tar")
        
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_points * 2)
        self.model = model
    
    def forward(self, x):
        return self.model(x)

class MambaVisionBase(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = create_model('mamba_vision_B', pretrained=True, model_path="/tmp/mambavision_base_1k.pth.tar")
        
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_points * 2)
        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
class MambaVisionLarge(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = create_model('mamba_vision_L', pretrained=True, model_path="/tmp/mambavision_large_1k.pth.tar")
        
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_points * 2)
        self.model = model
    
    def forward(self, x):
        return self.model(x)

class ResNet50(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_points * 2)
        )
        self.model = model

    def forward(self, x):
        return self.model(x)

class VGG19(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_points * 2)
        self.model = model

    def forward(self, x):
        return self.model(x)

MODEL = {
    "efficientnet": EfficientNet,
    "efficientnet_FC2048": EfficientNet_FC2048,
    "efficientnet_transformer": EfficientNetWithTransformer,
    "efficientnet_mid_transformer": EfficientNetMidWithTransformer,
    "efficientnet_ms_transformer_4scales_gapconcat": EfficientNetMultiScaleTransformer_4scales_GAPConcat,
    "efficientnet_ms_transformer_3scales_gapconcat": EfficientNetMultiScaleTransformer_3scales_GAPConcat,
    "efficientnet_cbam": EfficientNetWithCBAM,
    "efficientnet_cbam_transformer": EfficientNetCBAMTransformer,
    "efficientnet_ms_cbam_4scales_gapconcat": EfficientNetMultiScaleCBAM_4scales_GAPConcat,
    "efficientnet_ms_cbam_3scales_gated": EfficientNetMultiScaleCBAM_3scales_GatedFusion,
    "efficientnet_ms_cbam_3scales_gapconcat": EfficientNetMultiScaleCBAM_3scales_GAPConcat,
    "efficientnet_ms_3scales_cross_attn": EfficientNetMultiScale_3scales_CrossAttn,
    "convnext_tiny": ConvNeXtTiny,
    "convnext": ConvNeXtSmall,
    "convnext_base": ConvNeXtBase,
    "convnext_large": ConvNeXtLarge,
    "convnext_cbam": ConvNeXtWithCBAM,
    "convnext_transformer": ConvNeXtWithTransformer,
    "convnext_v2_tiny": ConvNeXtV2Tiny,
    "convnext_v2_base": ConvNeXtV2Base,
    "convnext_v2_large": ConvNeXtV2Large,
    "mambavision_small": MambaVisionSmall,
    "mambavision_base": MambaVisionBase,
    "mambavision_large": MambaVisionLarge,
    "resnet": ResNet50,
    "vgg": VGG19
}

def initialize_model(model_name, num_points, **kwargs):
    model_name = model_name.lower()
    if model_name not in MODEL:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL[model_name](num_points, **kwargs)