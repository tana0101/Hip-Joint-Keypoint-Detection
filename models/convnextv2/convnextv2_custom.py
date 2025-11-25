import torch
import torch.nn as nn
from models.head import HeadAdapter
from models.convnextv2.convnextv2 import (
    convnextv2_tiny,
    convnextv2_base,
    convnextv2_large
)

class ConvNeXtV2Tiny(nn.Module):
    def __init__(self, num_points,
                 head_type="direct_regression",
                 Nx=None, Ny=None):
        super().__init__()

        model = convnextv2_tiny(num_classes=1000)
        pretrained_path = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt"
        state_dict = torch.hub.load_state_dict_from_url(pretrained_path, progress=True)
        model.load_state_dict(state_dict["model"])

        in_features = model.head.in_features

        # ✨ 插拔式 head
        model.head = HeadAdapter(
            head_type=head_type,
            in_features=in_features,
            num_points=num_points,
            Nx=Nx, Ny=Ny
        )

        self.model = model

    def forward(self, x):
        return self.model(x)

class ConvNeXtV2Base(nn.Module):
    def __init__(self, num_points,
                 head_type="direct_regression",
                 Nx=None, Ny=None):
        super().__init__()

        model = convnextv2_base(num_classes=1000)
        pretrained_path = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt"
        state_dict = torch.hub.load_state_dict_from_url(pretrained_path, progress=True)
        model.load_state_dict(state_dict["model"])

        in_features = model.head.in_features

        model.head = HeadAdapter(
            head_type=head_type,
            in_features=in_features,
            num_points=num_points,
            Nx=Nx, Ny=Ny
        )

        self.model = model

    def forward(self, x):
        return self.model(x)

class ConvNeXtV2Large(nn.Module):
    def __init__(self, num_points,
                 head_type="direct_regression",
                 Nx=None, Ny=None):
        super().__init__()

        model = convnextv2_large(num_classes=1000)
        pretrained_path = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt"
        state_dict = torch.hub.load_state_dict_from_url(pretrained_path, progress=True)
        model.load_state_dict(state_dict["model"])

        in_features = model.head.in_features

        model.head = HeadAdapter(
            head_type=head_type,
            in_features=in_features,
            num_points=num_points,
            Nx=Nx, Ny=Ny
        )

        self.model = model

    def forward(self, x):
        return self.model(x)