import torch
import torch.nn as nn
import torchvision.models as models
from models.attention.CBAM import CBAM
from models.head import HeadAdapter

class EfficientNetWithTransformer(nn.Module):
    def __init__(self, num_points: int, head_type: str,
                 image_size: int = 224, hidden_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 1,
                 Nx: int = None, Ny: int = None):
        super().__init__()
        base = models.efficientnet_v2_m(pretrained=True)
        self.backbone = base.features
        self.input_dim = 1280
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.seq_len = 7 * 7
        self.projector = nn.Linear(self.input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim) * 0.01)
        self.pre_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 只負責把 token 攤平成單一向量 z
        self.readout = nn.Sequential(
            nn.Flatten(),                           # [B, 49*hidden_dim]
            nn.LayerNorm(hidden_dim * self.seq_len)
        )
        self.kp_head = HeadAdapter(head_type, in_features=hidden_dim * self.seq_len,
                                   num_points=num_points, Nx=Nx, Ny=Ny)

    def forward(self, x):
        x = self.backbone(x)                # [B,1280,7,7]
        x = self.pool(x)                    # [B,1280,7,7]
        x = x.flatten(2).transpose(1, 2)    # [B,49,1280]
        x = self.projector(x)               # [B,49,H]
        x = self.pre_norm(x)
        x = x + self.positional_encoding
        x = self.transformer(x)             # [B,49,H]
        z = self.readout(x)                 # [B, 49*H]
        return self.kp_head(z)              # direct: [B,2K] / simcc: tuple

class EfficientNetMidWithTransformer(nn.Module):
    def __init__(self, num_points: int, head_type: str,
                 image_size: int = 224, hidden_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 1,
                 Nx: int = None, Ny: int = None):
        super().__init__()
        base = models.efficientnet_v2_m(pretrained=True)
        self.backbone = nn.Sequential(*list(base.features.children())[:6])  # [B,176,14,14]
        self.input_dim = 176
        self.pool = nn.AdaptiveAvgPool2d((14, 14))
        self.seq_len = 14 * 14
        self.projector = nn.Linear(self.input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim) * 0.01)
        self.pre_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.readout = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_dim * self.seq_len)
        )
        self.kp_head = HeadAdapter(head_type, in_features=hidden_dim * self.seq_len,
                                   num_points=num_points, Nx=Nx, Ny=Ny)

    def forward(self, x):
        x = self.backbone(x)                # [B,176,14,14]
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)    # [B,196,176]
        x = self.projector(x)               # [B,196,H]
        x = self.pre_norm(x)
        x = x + self.positional_encoding
        x = self.transformer(x)             # [B,196,H]
        z = self.readout(x)                 # [B, 196*H]
        return self.kp_head(z)

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

# ---- 多尺度 + Transformer (4 scales) ----
class EfficientNetMultiScaleTransformer_4scales_GAPConcat(nn.Module):
    def __init__(self, num_points: int, head_type: str,
                 image_size: int = 224, hidden_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 1,
                 Nx: int = None, Ny: int = None):
        super().__init__()
        base = models.efficientnet_v2_m(pretrained=True)
        feats = base.features
        self.stem_to_3 = nn.Sequential(*feats[:4])  # 80,28x28
        self.block4 = feats[4]  # 160,14x14
        self.block5 = feats[5]  # 176,14x14
        self.block6 = feats[6]  # 304, 7x7
        self.block7 = feats[7]  # 512, 7x7

        self.trans4 = PatchTransformer(160, hidden_dim, num_heads, num_layers, hw=(14,14))
        self.trans5 = PatchTransformer(176, hidden_dim, num_heads, num_layers, hw=(14,14))
        self.trans6 = PatchTransformer(304, hidden_dim, num_heads, num_layers, hw=(7,7))
        self.trans7 = PatchTransformer(512, hidden_dim, num_heads, num_layers, hw=(7,7))

        fused_dim = 4 * hidden_dim
        # （原本直接回歸）→ 現在只做「融合到 z」
        self.fuse = nn.LayerNorm(fused_dim)
        self.kp_head = HeadAdapter(head_type, in_features=fused_dim,
                                   num_points=num_points, Nx=Nx, Ny=Ny)

    def forward(self, x):
        x  = self.stem_to_3(x)
        f4 = self.block4(x); t4 = self.trans4(f4)
        f5 = self.block5(f4); t5 = self.trans5(f5)
        f6 = self.block6(f5); t6 = self.trans6(f6)
        f7 = self.block7(f6); t7 = self.trans7(f7)
        z = torch.cat([t4,t5,t6,t7], dim=1)
        z = self.fuse(z)
        return self.kp_head(z)

# ---- 多尺度 + Transformer (3 scales) ----
class EfficientNetMultiScaleTransformer_3scales_GAPConcat(nn.Module):
    def __init__(self, num_points: int, head_type: str,
                 image_size: int = 224, hidden_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 1,
                 Nx: int = None, Ny: int = None):
        super().__init__()
        base = models.efficientnet_v2_m(pretrained=True)
        feats = base.features
        self.stem_to_5 = nn.Sequential(*feats[:6]) # 到 176,14x14
        self.block6 = feats[6]  # 304,7x7
        self.block7 = feats[7]  # 512,7x7
        self.block8 = feats[8]  # 1280,7x7
        self.trans6 = PatchTransformer(304, hidden_dim, num_heads, num_layers, hw=(7,7))
        self.trans7 = PatchTransformer(512, hidden_dim, num_heads, num_layers, hw=(7,7))
        self.trans8 = PatchTransformer(1280, hidden_dim, num_heads, num_layers, hw=(7,7))

        fused_dim = 3 * hidden_dim
        self.fuse = nn.LayerNorm(fused_dim)
        self.kp_head = HeadAdapter(head_type, in_features=fused_dim,
                                   num_points=num_points, Nx=Nx, Ny=Ny)

    def forward(self, x):
        x  = self.stem_to_5(x)
        f6 = self.block6(x);  t6 = self.trans6(f6)
        f7 = self.block7(f6); t7 = self.trans7(f7)
        f8 = self.block8(f7); t8 = self.trans8(f8)
        z = torch.cat([t6,t7,t8], dim=1)
        z = self.fuse(z)
        return self.kp_head(z)

class EfficientNetWithCBAM(nn.Module):
    def __init__(self, num_points: int, head_type: str, Nx: int = None, Ny: int = None):
        super().__init__()
        base = models.efficientnet_v2_m(pretrained=True)
        self.backbone = base.features
        self.cbam = CBAM(in_channels=1280)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        z_dim = 1280
        self.norm = nn.LayerNorm(z_dim)
        self.kp_head = HeadAdapter(head_type, in_features=z_dim,
                                   num_points=num_points, Nx=Nx, Ny=Ny)

    def forward(self, x):
        x = self.backbone(x)     # [B,1280,H/32,W/32]
        x = self.cbam(x)
        x = self.pool(x).flatten(1)  # [B,1280]
        z = self.norm(x)
        return self.kp_head(z)

class EfficientNetMultiScaleCBAM_4scales_GAPConcat(nn.Module):
    def __init__(self, num_points: int, head_type: str, Nx: int = None, Ny: int = None):
        super().__init__()
        base = models.efficientnet_v2_m(pretrained=True)
        feats = base.features
        self.stem_to_3 = nn.Sequential(*feats[:4])
        self.block4, self.block5, self.block6, self.block7 = feats[4], feats[5], feats[6], feats[7]
        self.cbam4, self.cbam5, self.cbam6, self.cbam7 = CBAM(160), CBAM(176), CBAM(304), CBAM(512)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        fused_dim = 160 + 176 + 304 + 512
        self.fuse_norm = nn.LayerNorm(fused_dim)   # 讓 fused 向量穩定
        self.kp_head = HeadAdapter(head_type, in_features=fused_dim,
                                   num_points=num_points, Nx=Nx, Ny=Ny)

    def forward(self, x):
        x  = self.stem_to_3(x)
        f4 = self.block4(x); f5 = self.block5(f4); f6 = self.block6(f5); f7 = self.block7(f6)
        f4, f5, f6, f7 = self.cbam4(f4), self.cbam5(f5), self.cbam6(f6), self.cbam7(f7)
        z4 = self.gap(f4).flatten(1)
        z5 = self.gap(f5).flatten(1)
        z6 = self.gap(f6).flatten(1)
        z7 = self.gap(f7).flatten(1)
        z = torch.cat([z4,z5,z6,z7], dim=1)
        z = self.fuse_norm(z)
        return self.kp_head(z)

class EfficientNetMultiScaleCBAM_3scales_GAPConcat(nn.Module):
    def __init__(self, num_points: int, head_type: str, Nx: int = None, Ny: int = None):
        super().__init__()
        base = models.efficientnet_v2_m(pretrained=True)
        feats = base.features
        self.stem_to_5 = nn.Sequential(*feats[:6])
        self.block6, self.block7, self.block8 = feats[6], feats[7], feats[8]
        self.cbam6, self.cbam7, self.cbam8 = CBAM(304), CBAM(512), CBAM(1280)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        fused_dim = 304 + 512 + 1280
        self.fuse_norm = nn.LayerNorm(fused_dim)   # 讓 fused 向量穩定
        self.kp_head = HeadAdapter(head_type, in_features=fused_dim,
                                   num_points=num_points, Nx=Nx, Ny=Ny)

    def forward(self, x):
        x  = self.stem_to_5(x)
        f6 = self.block6(x); f7 = self.block7(f6); f8 = self.block8(f7)
        f6, f7, f8 = self.cbam6(f6), self.cbam7(f7), self.cbam8(f8)
        z6 = self.gap(f6).flatten(1)
        z7 = self.gap(f7).flatten(1)
        z8 = self.gap(f8).flatten(1)
        z = torch.cat([z6,z7,z8], dim=1)
        z = self.fuse_norm(z)
        return self.kp_head(z)

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
    def __init__(self, num_points: int, head_type: str, embed_dim: int = 128, Nx: int = None, Ny: int = None):
        super().__init__()
        base  = models.efficientnet_v2_m(pretrained=True)
        feats = base.features
        self.stem_to_3 = nn.Sequential(*feats[:4])
        self.block4, self.block5, self.block6, self.block7, self.head8 = feats[4], feats[5], feats[6], feats[7], feats[8]
        self.h6, self.h7, self.h8 = ScaleHead(304, embed_dim), ScaleHead(512, embed_dim), ScaleHead(1280, embed_dim)
        self.gates = nn.Parameter(torch.ones(3))
        # 只做前置投影，最後仍交給 HeadAdapter
        self.pre = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.kp_head = HeadAdapter(head_type, in_features=4*embed_dim,
                                   num_points=num_points, Nx=Nx, Ny=Ny)

    def forward(self, x):
        x  = self.stem_to_3(x)
        x  = self.block4(x); x = self.block5(x)
        f6 = self.block6(x); f7 = self.block7(f6); f8 = self.head8(f7)
        z6, z7, z8 = self.h6(f6), self.h7(f7), self.h8(f8)
        g = torch.clamp(self.gates, 0.0, 3.0)
        fused = torch.cat([g[0]*z6, g[1]*z7, g[2]*z8], dim=1)
        z = self.pre(fused)             # [B, 4*embed_dim]
        return self.kp_head(z)

class EfficientNetCBAMTransformer(nn.Module):
    def __init__(self, num_points: int, head_type: str,
                 image_size: int = 224, hidden_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 1,
                 Nx: int = None, Ny: int = None):
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
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.readout = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_dim * self.seq_len),
        )
        self.kp_head = HeadAdapter(head_type, in_features=hidden_dim * self.seq_len,
                                   num_points=num_points, Nx=Nx, Ny=Ny)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)                    # [B,1280,7,7]
        x = x.flatten(2).transpose(1, 2)    # [B,49,1280]
        x = self.projector(x)
        x = self.pre_norm(x)
        x = x + self.positional_encoding
        x = self.transformer(x)             # [B,49,H]
        z = self.readout(x)                 # [B,49*H]
        return self.kp_head(z)

# ===== Cross-Attn 版本 =====
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
    def __init__(self, num_points: int, head_type: str,
                 hidden: int = 256, nhead: int = 4, nlayers: int = 1,
                 Nx: int = None, Ny: int = None):
        super().__init__()
        base  = models.efficientnet_v2_m(pretrained=True)
        feats = base.features
        self.stem_to_3 = nn.Sequential(*feats[:4])
        self.block4, self.block5, self.block6, self.block7, self.head8 = feats[4], feats[5], feats[6], feats[7], feats[8]
        self.ms_dec = MSDecoder(c6=304, c7=512, c8=1280, hidden=hidden, nhead=nhead, nlayers=nlayers)
        self.readout = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),        # 將 49 tokens → 均值/池化成單一 token
        )
        self.norm = nn.LayerNorm(hidden)
        self.kp_head = HeadAdapter(head_type, in_features=hidden,
                                   num_points=num_points, Nx=Nx, Ny=Ny)

    def forward(self, x):
        x  = self.stem_to_3(x)
        x  = self.block4(x); x = self.block5(x)
        f6 = self.block6(x); f7 = self.block7(f6); f8 = self.head8(f7)
        dec_out = self.ms_dec(f6, f7, f8)    # [B,49,H]
        # 以 mean 更直觀；若堅持使用 pool1d 也可：
        z = dec_out.mean(dim=1)              # [B,H]
        z = self.norm(z)
        return self.kp_head(z)

# ===== 兩個簡化版 EfficientNet =====
class EfficientNet(nn.Module):
    """
    取代原 classifier：輸出展平成向量 z，再交給 HeadAdapter
    """
    def __init__(self, num_points: int, head_type: str, Nx: int = None, Ny: int = None):
        super().__init__()
        model = models.efficientnet_v2_m(pretrained=True)
        in_features = model.classifier[1].in_features
        # 讓 forward 輸出的是 avgpool 後的向量
        model.classifier = nn.Identity()
        self.model = model
        self.norm = nn.LayerNorm(in_features)
        self.kp_head = HeadAdapter(head_type, in_features=in_features,
                                   num_points=num_points, Nx=Nx, Ny=Ny)

    def forward(self, x):
        z = self.model(x)         # [B, in_features]
        z = self.norm(z)
        return self.kp_head(z)

class EfficientNet_FC2048(nn.Module):
    """
    保留你原本的中介 MLP(到 2048)，但最後輸出 z 再交給 HeadAdapter
    """
    def __init__(self, num_points: int, head_type: str, Nx: int = None, Ny: int = None):
        super().__init__()
        model = models.efficientnet_v2_m(pretrained=True)
        in_features = model.classifier[1].in_features
        # 取代成「到 2048 的嵌入」，不直接輸出 2K
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
        )
        self.model = model
        self.norm = nn.LayerNorm(2048)
        self.kp_head = HeadAdapter(head_type, in_features=2048,
                                   num_points=num_points, Nx=Nx, Ny=Ny)

    def forward(self, x):
        z = self.model(x)         # [B,2048]
        z = self.norm(z)
        return self.kp_head(z)