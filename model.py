import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetWithTransformer(nn.Module):
    def __init__(self, num_points: int, image_size: int = 224,
                 hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 1):
        super().__init__()

        # Load EfficientNetV2-M backbone
        base = models.efficientnet_v2_m(pretrained=True)
        self.backbone = base.features  # Remove classifier
        self.input_dim = 1280
        
        self.pool = nn.AdaptiveAvgPool2d((14, 14))  # Ensure fixed size output
        self.seq_len = 14 * 14  # 196 tokens
        
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
        x = self.backbone(x)               # [B, 1280, 14, 14]
        x = self.pool(x)                   # [B, 1280, 14, 14]
        x = x.flatten(2).transpose(1, 2)   # [B, 196, 1280]
        x = self.projector(x)             # [B, 196, hidden_dim]
        x = self.pre_norm(x)              # LayerNorm before Transformer
        x = x + self.positional_encoding  # Add position info
        x = self.transformer(x)           # [B, 196, hidden_dim]
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
        x = self.backbone(x)                 # [B, 672, 14, 14]
        x = self.pool(x)                     # [B, 672, 14, 14]
        x = x.flatten(2).transpose(1, 2)     # [B, 196, 672]
        x = self.projector(x)                # [B, 196, hidden_dim]
        x = self.pre_norm(x)
        x = x + self.positional_encoding
        x = self.transformer(x)              # [B, 196, hidden_dim]
        out = self.output_head(x)            # [B, num_points * 2]
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

def initialize_model(model_name, num_points):
    if model_name == "efficientnet":
        return EfficientNet(num_points)
    elif model_name == "efficientnet_FC2048":
        return EfficientNet_FC2048(num_points)
    elif model_name == "efficientnet_transformer":
        return EfficientNetWithTransformer(num_points)
    elif model_name == "efficientnet_mid_transformer":
        return EfficientNetMidWithTransformer(num_points)
    elif model_name == "efficientnet_cbam":
        return EfficientNetWithCBAM(num_points)
    elif model_name == "resnet":
        return ResNet50(num_points)
    elif model_name == "vgg":
        return VGG19(num_points)
    else:
        raise ValueError("Model must be 'efficientnet', 'efficientnet_FC2048', 'efficientnet_transformer', 'efficientnet_mid_transformer', 'efficientnet_cbam', 'resnet', or 'vgg'.")
