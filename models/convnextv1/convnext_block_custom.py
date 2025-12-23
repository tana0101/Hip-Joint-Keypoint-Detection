# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from models.attention.SE import SEBlock

class SE(nn.Module):
    def __init__(self, dim, r=4):
        super().__init__()
        hidden = max(dim // r, 8)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):  # x: (N,H,W,C)
        s = x.mean(dim=(1, 2))              # GAP -> (N,C)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = s.unsqueeze(1).unsqueeze(1)     # (N,1,1,C)
        return x * torch.sigmoid(s)

class BlockD(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        # self.ffn_drop = nn.Dropout(p=0.1) # Feed-Forward Network Dropout
        self.se = SE(4 * dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.se(x)
        # x = self.ffn_drop(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.4, # 模仿 PyTorch 的設定，將 drop_path_rate 設為 0.4 有更好的效果
                 layer_scale_init_value=1e-6, head_init_scale=1., block_cls=Block, # 使用自定義的 Block 類別
                 ):
        super().__init__()

        self.dims = dims  # 儲存 dims 以便外部使用
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                # *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                # layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                *[
                    block_cls(                           # ⭐ 這裡改用 block_cls
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_stages(self, x):
        """回傳每一個 stage 的 feature maps: [C2, C3, C4, C5]"""
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features
    
    def forward_features(self, x):
        """一般的 forward features，只回傳最後一個 stage 的 feature map"""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x
    
    def forward_head(self, x):
        x = self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
        x = self.head(x)
        return x # (B, C_out)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class FPN(nn.Module):
    def __init__(
        self,
        in_channels_list,     # 例如 [96, 192, 384, 768]
        out_channels=256,
        fuse_type="sum",      # "sum" 或 "concat"
    ):
        super().__init__()
        assert fuse_type in ["sum", "concat"]
        self.fuse_type = fuse_type

        # lateral conv 把每一層先變成 out_channels
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_c in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    # 1. 降維
                    nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False),
                    
                    # 2. 使用 ConvNeXt 的 LayerNorm
                    # 關鍵：必須指定 data_format="channels_first" 才能處理 (N, C, H, W)
                    LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
                    
                    # 3. 激活函數
                    nn.GELU(),
                    
                    # 4. SE Attention (特徵篩選)
                    # 放在 LN 和 GELU 之後效果通常最好，因為特徵已經被正規化且非線性化
                    SEBlock(out_channels, reduction=4) 
                )
            )
            # 可以再接一個 3x3 conv 平滑一下
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

        if self.fuse_type == "concat":
            # concat 之後再壓回 out_channels
            self.fuse_conv = nn.Conv2d(
                out_channels * len(in_channels_list), out_channels, kernel_size=1
            )

    def forward(self, feats):
        """
        feats: list of feature maps, 假設順序 [C2, C3, C4, C5] (淺 -> 深)
        回傳: fused feature map [B, out_channels, H, W]
        """
        # 先把順序改成深 -> 淺，比較好做 top-down
        feats = feats[::-1]  # [C5, C4, C3, C2]
        results = []

        prev = None
        for idx, x in enumerate(feats):
            # 注意：這裡用 len(feats)-1-idx 是對應原本順序
            lateral = self.lateral_convs[len(feats) - 1 - idx](x)
            if prev is None:
                y = lateral
            else:
                # 上一層 upsample 到現在這層大小
                prev_upsampled = F.interpolate(prev, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
                y = lateral + prev_upsampled
            y = self.output_convs[len(feats) - 1 - idx](y)
            results.append(y)
            prev = y

        # 把順序改回淺 -> 深
        results = results[::-1]  # [P2, P3, P4, P5]

        if self.fuse_type == "sum":
            fuse = results[0]
            return fuse  # 直接回最淺的 feature map
        elif self.fuse_type == "concat":
            # 先把全部 resize 成同一個空間，再 concat
            target_size = results[0].shape[-2:]
            resized = [
                F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                for f in results
            ]
            fused = torch.cat(resized, dim=1)  # [B, C*out_levels, H, W]
            fused = self.fuse_conv(fused)      # [B, out_channels, H, W]
            return fused                        # 直接回 fused feature

class ConvNeXtFlexible(nn.Module):
    def __init__(
        self,
        backbone: ConvNeXt,
        mode: str = "cls",          # "cls", "fpn", "multi_gap"
        fpn_levels=(1, 2, 3),       
        fpn_out_channels=256,
        fpn_fuse_type="sum",
    ):
        super().__init__()
        assert mode in ["cls", "fpn", "multi_gap"]
        self.mode = mode
        self.backbone = backbone
        self.fpn_levels = fpn_levels
        
        if mode == "cls":
            self.out_channels = backbone.dims[-1]
        
        elif mode == "multi_gap":
            in_dims = [backbone.dims[i] for i in fpn_levels]
            self.in_dims = in_dims
            self.ln_each = nn.ModuleList([nn.LayerNorm(d) for d in in_dims])
            self.concat_dim = sum(in_dims)
            self.ln_final = nn.LayerNorm(self.concat_dim)
            self.out_channels = self.concat_dim  # map = [B, C_concat, 1, 1]
            
        elif mode == "fpn":
            in_channels_list = [backbone.dims[i] for i in fpn_levels]
            self.neck = FPN(
                in_channels_list=in_channels_list,
                out_channels=fpn_out_channels,
                fuse_type=fpn_fuse_type,
            )
            self.out_channels = fpn_out_channels

    def get_feature_map(self, x):
        """
        根據 mode，回傳 [B, C_out, H, W] 的 feature map
        """
        # 1) CLS 模式：用最後一層 feature 當 feature map
        if self.mode == "cls":
            feat = self.backbone.forward_features(x)  # [B, C, H, W]
            return feat

        # 2) 多 stage features
        stage_feats = self.backbone.forward_stages(x)   # list: [C2, C3, C4, C5]，每個 [B, Ci, Hi, Wi]
        selected_feats = [stage_feats[i] for i in self.fpn_levels]

        if self.mode == "fpn":
            fused = self.neck(selected_feats)           # [B, fpn_out_channels, H, W]
            return fused
        
        elif self.mode == "multi_gap":
            # GAP + LN per level → concat → LN 
            pooled = []
            for idx, f in enumerate(selected_feats):
                g = f.mean(dim=[2, 3])           # [B, Ci]
                g = self.ln_each[idx](g)
                pooled.append(g)
            
            feat_vec = torch.cat(pooled, dim=1)  # [B, sum(Ci)]
            feat_vec = self.ln_final(feat_vec)   # [B, C_concat]
            feat_map = feat_vec.unsqueeze(-1).unsqueeze(-1)  # ⭐ [B, C_concat, 1, 1]
            return feat_map

    def forward(self, x):
        feat_map = self.get_feature_vector(x)   # [B, C, H, W]
        out = self.head(feat_map)               # HeadAdapter 自己決定 1D/2D 處理
        return out

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model