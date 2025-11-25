import torch
import torch.nn as nn
from models.head import HeadAdapter
from models.convnextv1 import convnext_block_custom

class ConvNeXtSmallCustom(nn.Module):
    def __init__(self, num_points,
                 head_type="direct_regression",
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

        # convnext head 輸出維度
        in_features = model.head.in_features

        # 🔥 代換成可插拔 head
        model.head = HeadAdapter(
            head_type=head_type,
            in_features=in_features,
            num_points=num_points,
            Nx=Nx,
            Ny=Ny
        )

        self.model = model

    def forward(self, x):
        return self.model(x)
    
class ConvNeXtSmallMS(nn.Module):
    def __init__(self, num_points,
                 head_type="direct_regression",
                 Nx=None, Ny=None):
        super().__init__()

        backbone = convnext_block_custom.convnext_small(pretrained=True)

        self.model = convnext_block_custom.ConvNeXtFlexible(
            backbone=backbone,
            mode="multi_gap",
            fpn_levels=(1, 2, 3),    # 使用 satge2, stage3, stage4 的特徵圖
            fpn_out_channels=256,
            fpn_fuse_type="concat",
            num_classes=1000,       
        )

        # FPN + GAP 之後的向量維度
        in_features = self.model.out_dim

        # 可插拔 head：DirectRegression 或 SimCC
        self.head = HeadAdapter(
            head_type=head_type,
            in_features=in_features,
            num_points=num_points,
            Nx=Nx,
            Ny=Ny,
        )

    def forward(self, x):
        feat = self.model.get_feature_vector(x)
        return self.head(feat)