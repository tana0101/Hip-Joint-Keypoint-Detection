import torch
import torch.nn as nn
import torchvision.models as models
from models.attention.CBAM import CBAM
from models.head import HeadAdapter

class ConvNeXtWithCBAM(nn.Module):
    def __init__(self, num_points: int,
                 head_type="direct_regression",
                 Nx=None, Ny=None,
                 pretrained=True):
        super().__init__()

        base = models.convnext_small(pretrained=pretrained)

        self.backbone = base.features         # [B, 768, H/32, W/32]
        self.cbam = CBAM(in_channels=768)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ConvNeXt classifier 的 LayerNorm2d
        self.norm = base.classifier[0]
        self.flatten = nn.Flatten(1)

        # 使用你寫的 HeadAdapter
        self.head = HeadAdapter(
            head_type=head_type,
            in_features=768,
            num_points=num_points,
            Nx=Nx, Ny=Ny
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        return self.head(x)
    
class ConvNeXtWithTransformer(nn.Module):
    def __init__(self, num_points: int,
                 head_type="direct_regression",
                 Nx=None, Ny=None,
                 image_size=224,
                 hidden_dim=256,
                 num_heads=4,
                 num_layers=1,
                 pretrained=True):
        super().__init__()

        base = models.convnext_small(pretrained=pretrained)
        self.backbone = base.features
        self.input_dim = 768

        # Fix output size
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.seq_len = 7 * 7

        self.projector = nn.Linear(self.input_dim, hidden_dim)

        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.seq_len, hidden_dim) * 0.01
        )

        self.pre_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # 💡 最終輸出 dim = hidden_dim * seq_len
        self.final_dim = hidden_dim * self.seq_len
        self.flatten = nn.Flatten()
        self.norm = nn.LayerNorm(self.final_dim)

        # 插拔 head
        self.head = HeadAdapter(
            head_type=head_type,
            in_features=self.final_dim,
            num_points=num_points,
            Nx=Nx, Ny=Ny
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.projector(x)
        x = self.pre_norm(x)
        x = x + self.positional_encoding
        x = self.transformer(x)
        x = self.flatten(x)
        x = self.norm(x)
        return self.head(x)
    
class ConvNeXtTiny(nn.Module):
    def __init__(self, num_points,
                 head_type="direct_regression", Nx=None, Ny=None,
                 pretrained=True):
        super().__init__()
        model = models.convnext_tiny(pretrained=pretrained)
        in_features = model.classifier[2].in_features  # ConvNeXt classifier: [LayerNorm2d, Flatten, Linear]
        model.classifier[2] = HeadAdapter(
            head_type, in_features,
            num_points, Nx, Ny
        )
        self.model = model

    def forward(self, x):
        return self.model(x)    
    
class ConvNeXtSmall(nn.Module):
    def __init__(self, num_points,
                 head_type="direct_regression", Nx=None, Ny=None,
                 pretrained=True):
        super().__init__()
        model = models.convnext_small(pretrained=pretrained)
        in_features = model.classifier[2].in_features  # ConvNeXt classifier: [LayerNorm2d, Flatten, Linear]
        model.classifier[2] = HeadAdapter(
            head_type, in_features,
            num_points, Nx, Ny
        )
        self.model = model

    def forward(self, x):
        return self.model(x)
    
class ConvNeXtBase(nn.Module):
    def __init__(self, num_points,
                 head_type="direct_regression", Nx=None, Ny=None,
                 pretrained=True):
        super().__init__()
        model = models.convnext_base(pretrained=pretrained)
        in_features = model.classifier[2].in_features  # ConvNeXt classifier: [LayerNorm2d, Flatten, Linear]
        model.classifier[2] = HeadAdapter(
            head_type, in_features,
            num_points, Nx, Ny
        )
        self.model = model

    def forward(self, x):
        return self.model(x)
    
class ConvNeXtLarge(nn.Module):
    def __init__(self, num_points,
                 head_type="direct_regression", Nx=None, Ny=None,
                 pretrained=True):
        super().__init__()
        model = models.convnext_large(pretrained=pretrained)
        in_features = model.classifier[2].in_features  # ConvNeXt classifier: [LayerNorm2d, Flatten, Linear]
        model.classifier[2] = HeadAdapter(
            head_type, in_features,
            num_points, Nx, Ny
        )
        self.model = model

    def forward(self, x):
        return self.model(x)