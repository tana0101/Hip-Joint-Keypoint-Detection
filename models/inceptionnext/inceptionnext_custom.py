import torch
import torch.nn as nn
from models.head import HeadAdapter
from models.inceptionnext.inceptionnext import (
    inceptionnext_small
)

class KeypointFeatureExtractor(nn.Module):
    """
    負責：
      GAP → squeeze → LayerNorm → Dropout
    不做最後 Linear（要交給 HeadAdapter）
    """
    def __init__(self, dim, drop=0.0):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.gap(x)                 # [B, C, 1, 1]
        x = x.squeeze(-1).squeeze(-1)   # [B, C]
        x = self.norm(x)
        x = self.drop(x)
        return x                        # 回傳 feature vector

class InceptionNextSmall(nn.Module):
    def __init__(self, num_points,
                 head_type="direct_regression",
                 input_size: tuple[int, int] = (224, 224),
                 Nx=None, Ny=None,
                 drop=0.0):
        super().__init__()

        # 1. Backbone
        model = inceptionnext_small(pretrained=True)
        in_features = model.num_features

        # 2. Feature extractor（原本 KeypointHead 去掉最後 Linear）
        self.feat = KeypointFeatureExtractor(in_features, drop=drop)

        # 3. 可插拔 head（direct / simcc）
        self.head = HeadAdapter(
            head_type=head_type,
            in_features=in_features,
            num_points=num_points,
            Nx=Nx,
            Ny=Ny,
        )

        # 4. Backbone
        self.backbone = model

    def forward(self, x):
        x = self.backbone.forward_features(x)  # InceptionNext feature map
        x = self.feat(x)                       # GAP, LN, Dropout
        return self.head(x)                    # Direct 或 SimCC