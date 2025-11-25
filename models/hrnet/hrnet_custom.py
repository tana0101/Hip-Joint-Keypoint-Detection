from types import SimpleNamespace as CN
import torch
import torch.nn as nn

from models.hrnet import hrnet
from models.head import HeadAdapter


def get_hrnet_w32_cfg():
    # STAGE2
    stage2 = CN(
        NUM_MODULES=1,
        NUM_BRANCHES=2,
        BLOCK="BASIC",
        NUM_BLOCKS=[4, 4],
        NUM_CHANNELS=[32, 64],
        FUSE_METHOD="SUM",
    )

    # STAGE3
    stage3 = CN(
        NUM_MODULES=4,
        NUM_BRANCHES=3,
        BLOCK="BASIC",
        NUM_BLOCKS=[4, 4, 4],
        NUM_CHANNELS=[32, 64, 128],
        FUSE_METHOD="SUM",
    )

    # STAGE4
    stage4 = CN(
        NUM_MODULES=3,
        NUM_BRANCHES=4,
        BLOCK="BASIC",
        NUM_BLOCKS=[4, 4, 4, 4],
        NUM_CHANNELS=[32, 64, 128, 256],
        FUSE_METHOD="SUM",
    )

    extra = CN(
        PRETRAINED_LAYERS=[
            "conv1",
            "bn1",
            "conv2",
            "bn2",
            "layer1",
            "transition1",
            "stage2",
            "transition2",
            "stage3",
            "transition3",
            "stage4",
        ],
        FINAL_CONV_KERNEL=1,
        STAGE2=stage2,
        STAGE3=stage3,
        STAGE4=stage4,
    )

    model = CN(
        STEM_INPLANES=64,
        EXTRA=extra,
    )

    cfg = CN(MODEL=model)
    return cfg


def get_hrnet_w48_cfg():
    stage2 = CN(
        NUM_MODULES=1,
        NUM_BRANCHES=2,
        BLOCK="BASIC",
        NUM_BLOCKS=[4, 4],
        NUM_CHANNELS=[48, 96],
        FUSE_METHOD="SUM",
    )

    stage3 = CN(
        NUM_MODULES=4,
        NUM_BRANCHES=3,
        BLOCK="BASIC",
        NUM_BLOCKS=[4, 4, 4],
        NUM_CHANNELS=[48, 96, 192],
        FUSE_METHOD="SUM",
    )

    stage4 = CN(
        NUM_MODULES=3,
        NUM_BRANCHES=4,
        BLOCK="BASIC",
        NUM_BLOCKS=[4, 4, 4, 4],
        NUM_CHANNELS=[48, 96, 192, 384],
        FUSE_METHOD="SUM",
    )

    extra = CN(
        PRETRAINED_LAYERS=[
            "conv1",
            "bn1",
            "conv2",
            "bn2",
            "layer1",
            "transition1",
            "stage2",
            "transition2",
            "stage3",
            "transition3",
            "stage4",
        ],
        FINAL_CONV_KERNEL=1,
        STAGE2=stage2,
        STAGE3=stage3,
        STAGE4=stage4,
    )

    model = CN(
        STEM_INPLANES=64,
        EXTRA=extra,
    )

    cfg = CN(MODEL=model)
    return cfg


class _HRNetWithHead(nn.Module):
    """
    共用 wrapper：HRNetBackbone -> GAP -> (optional proj) -> HeadAdapter
    """

    def __init__(
        self,
        backbone: hrnet.HRNetBackbone,
        num_points: int,
        head_type: str = "direct_regression",
        Nx: int | None = None,
        Ny: int | None = None,
        proj_dim: int | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.gap = nn.AdaptiveAvgPool2d(1)

        in_features = backbone.backbone_out_channels  # w32=32, w48=48

        if proj_dim is not None:
            self.proj = nn.Linear(in_features, proj_dim)
            in_features = proj_dim
        else:
            self.proj = None

        self.head = HeadAdapter(
            head_type=head_type,
            in_features=in_features,
            num_points=num_points,
            Nx=Nx,
            Ny=Ny,
        )

    def forward(self, x):
        feats = self.backbone(x)   # [B,C,H,W]
        feats = self.gap(feats)    # [B,C,1,1]
        feats = feats.flatten(1)   # [B,C]
        if self.proj is not None:
            feats = self.proj(feats)
        return self.head(feats)


class HRNetW32Custom(_HRNetWithHead):
    """
    HRNet-W32 backbone + 可插拔 head。
    initialize_model("hrnet_w32", ...) 會回傳這個類別的 instance。
    """

    def __init__(
        self,
        num_points: int,
        head_type: str = "direct_regression",
        Nx: int | None = None,
        Ny: int | None = None,
        proj_dim: int | None = None,
        pretrained_path: str | None = None,
        **kwargs,
    ):
        cfg = get_hrnet_w32_cfg()
        backbone = hrnet.HRNetBackbone(cfg)

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")
            backbone.load_state_dict(state, strict=False)

        super().__init__(
            backbone=backbone,
            num_points=num_points,
            head_type=head_type,
            Nx=Nx,
            Ny=Ny,
            proj_dim=proj_dim,
        )


class HRNetW48Custom(_HRNetWithHead):
    """
    HRNet-W48 backbone + 可插拔 head。
    initialize_model("hrnet_w48", ...) 會回傳這個類別的 instance。
    """

    def __init__(
        self,
        num_points: int,
        head_type: str = "direct_regression",
        Nx: int | None = None,
        Ny: int | None = None,
        proj_dim: int | None = None,
        pretrained_path: str | None = None,
        **kwargs,
    ):
        cfg = get_hrnet_w48_cfg()
        backbone = hrnet.HRNetBackbone(cfg)

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")
            backbone.load_state_dict(state, strict=False)

        super().__init__(
            backbone=backbone,
            num_points=num_points,
            head_type=head_type,
            Nx=Nx,
            Ny=Ny,
            proj_dim=proj_dim,
        )