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


class HRNetW32Custom(nn.Module):
    """
    HRNet-W32 backbone + 可插拔 head。
    initialize_model("hrnet_w32", ...) 會回傳這個類別的 instance。
    """
    def __init__(
        self,
        num_points: int,
        head_type: str = "direct_regression",    # "direct_regression", "simcc_1d", "simcc_2d", "simcc_2d_deconv"
        input_size: tuple[int, int] = (256, 256),
        Nx: int | None = None,
        Ny: int | None = None,
        pretrained_path: str | None = None,
        **head_kwargs,   # 給 HeadAdapter 的額外參數：比如 use_gap_norm, norm_type, num_deconv_layers...
    ):
        super().__init__()

        # 1) 建 HRNet-W32 backbone
        cfg = get_hrnet_w32_cfg()
        self.backbone = hrnet.HRNetBackbone(cfg)

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)

        # 2) 用 dummy 推出 HRNet backbone 輸出 feature map 的 shape
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size[0], input_size[1])
            feat = self.backbone(dummy)          # 預期: [1, C_out, H, W]
            _, C_out, H, W = feat.shape

        # 3) 建立通用 HeadAdapter
        self.head = HeadAdapter(
            head_type=head_type,
            in_channels=C_out,          # HRNet 輸出 channels
            map_size=(H, W),            # HRNet 輸出 H, W
            num_points=num_points,
            Nx=Nx,
            Ny=Ny,
            **head_kwargs,              # 例如 use_gap_norm=True, norm_type="layernorm"
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)   # [B, C_out, H, W]
        out = self.head(feat)     # HeadAdapter 裡決定 GAP/Conv2d/deconv/SimCC/Regression
        return out


class HRNetW48Custom(nn.Module):
    """
    HRNet-W48 backbone + 可插拔 head。
    initialize_model("hrnet_w48", ...) 會回傳這個類別的 instance。
    """
    def __init__(
        self,
        num_points: int,
        head_type: str = "direct_regression",
        input_size: tuple[int, int] = (256, 256),
        Nx: int | None = None,
        Ny: int | None = None,
        pretrained_path: str | None = None,
        **head_kwargs,
    ):
        super().__init__()

        # 1) 建 HRNet-W48 backbone
        cfg = get_hrnet_w48_cfg()
        self.backbone = hrnet.HRNetBackbone(cfg)

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)

        # 2) 用 dummy 推 feature map shape
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size[0], input_size[1])
            feat = self.backbone(dummy)          # [1, C_out, H, W]
            _, C_out, H, W = feat.shape

        # 3) HeadAdapter
        self.head = HeadAdapter(
            head_type=head_type,
            in_channels=C_out,
            map_size=(H, W),
            num_points=num_points,
            Nx=Nx,
            Ny=Ny,
            **head_kwargs,
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        out = self.head(feat)
        return out