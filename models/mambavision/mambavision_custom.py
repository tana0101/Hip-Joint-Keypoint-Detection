import torch
import torch.nn as nn
from mambavision import create_model
from models.convnextv1.convnext import Block as ConvNeXtBlock, LayerNorm as ConvNeXtLayerNorm
from models.head import HeadAdapter

class MambaVisionSmall(nn.Module):
    def __init__(self, num_points,
                 head_type: str = "direct_regression",
                 Nx: int = None, Ny: int = None):
        super().__init__()

        model = create_model(
            'mamba_vision_S',
            pretrained=True,
            model_path="/tmp/mambavision_small_1k.pth.tar"
        )

        in_features = model.head.in_features

        # ✨ 使用 HeadAdapter 取代原本 Linear
        model.head = HeadAdapter(
            head_type=head_type,
            in_features=in_features,
            num_points=num_points,
            Nx=Nx,
            Ny=Ny,
        )

        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
class MambaVisionBase(nn.Module):
    def __init__(self, num_points,
                 head_type: str = "direct_regression",
                 Nx: int = None, Ny: int = None):
        super().__init__()

        model = create_model(
            'mamba_vision_B',
            pretrained=True,
            model_path="/tmp/mambavision_base_1k.pth.tar"
        )
        
        in_features = model.head.in_features

        model.head = HeadAdapter(
            head_type=head_type,
            in_features=in_features,
            num_points=num_points,
            Nx=Nx,
            Ny=Ny,
        )

        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
class MambaVisionLarge(nn.Module):
    def __init__(self, num_points,
                 head_type: str = "direct_regression",
                 Nx: int = None, Ny: int = None):
        super().__init__()

        model = create_model(
            'mamba_vision_L',
            pretrained=True,
            model_path="/tmp/mambavision_large_1k.pth.tar"
        )
        
        in_features = model.head.in_features

        model.head = HeadAdapter(
            head_type=head_type,
            in_features=in_features,
            num_points=num_points,
            Nx=Nx,
            Ny=Ny,
        )

        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
class ConvNeXtV1StemStage0(nn.Module):
    """
    使用 convnextv1/convnext.py 的 Block & LayerNorm
    生成 [B, 128, 112, 112] 的低階特徵圖。
    修改點：
      - stem: stride=4  (保留更高解析度)
      - dim: 128        (增加通道容量)
    """
    def __init__(self, in_chans=3, dim=128, depth=3, layer_scale_init_value=1e-6):
        super().__init__()
        # ===== Stem =====
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=4, stride=4), # input_size/4
            ConvNeXtLayerNorm(dim, eps=1e-6, data_format="channels_first")
        )

        # ===== Stage0 (ConvNeXt Blocks) =====
        blocks = []
        for _ in range(depth):
            blocks.append(
                ConvNeXtBlock(dim=dim, layer_scale_init_value=layer_scale_init_value)
            )
        self.stage0 = nn.Sequential(*blocks)

    def forward(self, x):  # x: [B,3,224,224] → [B,128,112,112]
        x = self.stem(x)
        x = self.stage0(x)
        return x

def _infer_patch_embed_dim(backbone, in_chans=3, probe_size=224):
    """推斷 backbone patch_embed 的輸出維度"""
    dummy = torch.randn(1, in_chans, probe_size, probe_size)
    with torch.no_grad():
        out = backbone.patch_embed(dummy)
    return out.shape[1]

class MambaVisionSmallWithConvNeXt(nn.Module):
    def __init__(self, num_points,
                 head_type: str = "direct_regression",
                 Nx: int = None, Ny: int = None):
        super().__init__()

        backbone = create_model(
            'mamba_vision_S',
            pretrained=False,  # 你原本就是 False
            model_path="/tmp/mambavision_small_1k.pth.tar"
        )
        
        # ConvNeXtV1 的 Stem + Stage0 取代原本 patch_embed
        dim = _infer_patch_embed_dim(backbone, in_chans=3, probe_size=224)
        convnext_front = ConvNeXtV1StemStage0(
            in_chans=3,
            dim=dim,
            depth=3,
            layer_scale_init_value=0
        )
        backbone.patch_embed = convnext_front
        
        in_features = backbone.head.in_features

        backbone.head = HeadAdapter(
            head_type=head_type,
            in_features=in_features,
            num_points=num_points,
            Nx=Nx,
            Ny=Ny,
        )

        self.model = backbone

    def forward(self, x):
        return self.model(x)
    
class MambaVisionBaseWithConvNeXt(nn.Module):
    def __init__(self, num_points,
                 head_type: str = "direct_regression",
                 Nx: int = None, Ny: int = None):
        super().__init__()

        backbone = create_model(
            'mamba_vision_B',
            pretrained=True,
            model_path="/tmp/mambavision_base_1k.pth.tar"
        )
        
        # ConvNeXtV1 的 Stem + Stage0 取代原本 patch_embed
        dim = _infer_patch_embed_dim(backbone, in_chans=3, probe_size=224)
        convnext_front = ConvNeXtV1StemStage0(
            in_chans=3,
            dim=dim,
            depth=3,
            layer_scale_init_value=0
        )
        backbone.patch_embed = convnext_front
        
        in_features = backbone.head.in_features

        backbone.head = HeadAdapter(
            head_type=head_type,
            in_features=in_features,
            num_points=num_points,
            Nx=Nx,
            Ny=Ny,
        )

        self.model = backbone

    def forward(self, x):
        return self.model(x)