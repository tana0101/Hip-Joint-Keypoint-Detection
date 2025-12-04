import torch
import torch.nn as nn
from models.head import HeadAdapter
from models.convnextv1 import convnext_block_custom

class ConvNeXtSmallCustom(nn.Module):
    def __init__(self, num_points,
                 head_type="direct_regression",
                 input_size: tuple[int, int] = (224, 224),
                 Nx=None, Ny=None):
        super().__init__()

        # 原本 backbone
        model = convnext_block_custom.ConvNeXt(
            depths=[3, 3, 27, 3],
            dims=[96, 192, 384, 768],
            # block_cls=convnext_block_custom.Block
        )

        url = convnext_block_custom.model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        
        # 假設 model.forward_features(x) 輸出 [B, C_in, H, W]
        self.backbone = model

        # 用 dummy 算出 backbone feature map 大小
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size[0], input_size[1])
            feat = self._extract_features(dummy)   # [1, C_in, H, W]
            _, C_in, H, W = feat.shape
        
        # HeadAdapter
        self.head = HeadAdapter(
            head_type=head_type,
            in_channels=C_in,
            map_size=(H, W),
            num_points=num_points,
            Nx=Nx,
            Ny=Ny,
        )
        
    def _extract_features(self, x):
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)
        else:
            raise RuntimeError("backbone 沒有 forward_features，請依實作修改")

    def forward(self, x):
        feat = self._extract_features(x)  # [B, C_in, H, W]
        out = self.head(feat)
        return out
    
class ConvNeXtSmallMS(nn.Module):
    def __init__(self, num_points,
                 head_type="direct_regression",
                 input_size: tuple[int, int] = (224, 224),
                 Nx=None, Ny=None):
        super().__init__()

        backbone = convnext_block_custom.convnext_small(pretrained=True)

        self.model = convnext_block_custom.ConvNeXtFlexible(
            backbone=backbone,
            mode="fpn",              # fpn, multi_gap, cls 三種模式
            fpn_levels=(0, 1, 2, 3),    # 使用 stage1, stage2, stage3, stage4 的特徵圖
            fpn_out_channels=256,    # FPN 中間層維度
            fpn_fuse_type="concat",  # FPN 特徵融合方式：concat 或 sum(最上層的map)
        )

        # 2) 用 dummy 推出 body 輸出的 feature map size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size[0], input_size[1])
            feat_map = self.model.get_feature_map(dummy)   # [1, C_out, H, W]
            _, C_out, H, W = feat_map.shape
        
        # 3) 建 HeadAdapter，專門吃 feature map
        self.head = HeadAdapter(
            head_type=head_type,
            in_channels=C_out,
            map_size=(H, W),
            num_points=num_points,
            Nx=Nx,
            Ny=Ny,
        )

    def forward(self, x):
        feat = self.model.get_feature_map(x)
        out = self.head(feat)
        return out