import argparse, torch
torch.serialization.add_safe_globals([argparse.Namespace])

import torch.nn as nn
import torchvision.models as models

from models.head import HeadAdapter

from models.efficientnet.efficientnetv2_torch_custom import (
    EfficientNetWithTransformer,
    EfficientNetMidWithTransformer,
    EfficientNetMultiScaleTransformer_4scales_GAPConcat,
    EfficientNetMultiScaleTransformer_3scales_GAPConcat,
    EfficientNetWithCBAM,
    EfficientNetMultiScaleCBAM_4scales_GAPConcat,
    EfficientNetMultiScaleCBAM_3scales_GAPConcat,
    EfficientNetMultiScaleCBAM_3scales_GatedFusion,
    EfficientNetCBAMTransformer,
    EfficientNetMultiScale_3scales_CrossAttn,
    EfficientNet,
    EfficientNet_FC2048,
)

from models.convnextv1.convnext_custom import (
    ConvNeXtSmallCustom,
    ConvNeXtSmallMS,
)

from models.convnextv1.convnextv1_torch_custom import (
    ConvNeXtTiny,
    ConvNeXtSmall,
    ConvNeXtBase,
    ConvNeXtLarge,
    ConvNeXtWithCBAM,
    ConvNeXtWithTransformer,
)

from models.convnextv2.convnextv2_custom import (
    ConvNeXtV2Tiny,
    ConvNeXtV2Base,
    ConvNeXtV2Large
)

from models.mambavision.mambavision_custom import (
    MambaVisionSmall,
    MambaVisionBase,
    MambaVisionLarge,
    MambaVisionSmallWithConvNeXt,
    MambaVisionBaseWithConvNeXt
)

from functools import partial
from models.inceptionnext.inceptionnext_custom import (
    InceptionNextSmall,
)

from models.hrnet.hrnet_custom import (
    HRNetW32Custom,
    HRNetW48Custom,
)

MODEL = {
    
    # 基於 pyTorch 實作的 EfficientNetV2 系列模型。
    
    "efficientnet": EfficientNet,
    "efficientnet_FC2048": EfficientNet_FC2048,
    "efficientnet_transformer": EfficientNetWithTransformer,
    "efficientnet_mid_transformer": EfficientNetMidWithTransformer,
    "efficientnet_ms_transformer_4scales_gapconcat": EfficientNetMultiScaleTransformer_4scales_GAPConcat,
    "efficientnet_ms_transformer_3scales_gapconcat": EfficientNetMultiScaleTransformer_3scales_GAPConcat,
    "efficientnet_cbam": EfficientNetWithCBAM,
    "efficientnet_cbam_transformer": EfficientNetCBAMTransformer,
    "efficientnet_ms_cbam_4scales_gapconcat": EfficientNetMultiScaleCBAM_4scales_GAPConcat,
    "efficientnet_ms_cbam_3scales_gated": EfficientNetMultiScaleCBAM_3scales_GatedFusion,
    "efficientnet_ms_cbam_3scales_gapconcat": EfficientNetMultiScaleCBAM_3scales_GAPConcat,
    "efficientnet_ms_3scales_cross_attn": EfficientNetMultiScale_3scales_CrossAttn,

    # 基於 pyTorch 實作的 ConvNeXt 系列模型。
    
    "convnext_tiny": ConvNeXtTiny,
    "convnext": ConvNeXtSmall,
    "convnext_base": ConvNeXtBase,
    "convnext_large": ConvNeXtLarge,
    "convnext_cbam": ConvNeXtWithCBAM,
    "convnext_transformer": ConvNeXtWithTransformer,
    
    # 基於官方實作的 ConvNeXt 系列模型。
    
    "convnext_small_mg1234": ConvNeXtSmallMS, # 基於官方庫實作的 Multi-Scale ConvNeXt-Small
    "convnext_small_custom": ConvNeXtSmallCustom,
    "convnext_v2_tiny": ConvNeXtV2Tiny,
    "convnext_v2_base": ConvNeXtV2Base,
    "convnext_v2_large": ConvNeXtV2Large,
    
    # MambaVision 系列模型。
    
    "mambavision_small": MambaVisionSmall,
    "mambavision_base": MambaVisionBase,
    "mambavision_large": MambaVisionLarge,
    "mambavision_small_convnext": MambaVisionSmallWithConvNeXt,
    "mambavision_base_convnext": MambaVisionBaseWithConvNeXt,
    
    # InceptionNext 系列模型。
    
    "inceptionnext_small": InceptionNextSmall,
    
    # HRNet 系列模型。
    "hrnet_w32": HRNetW32Custom,
    "hrnet_w48": HRNetW48Custom,
}

def initialize_model(model_name, num_points, head_type="direct_regression", input_size=None, Nx=None, Ny=None, **kwargs):
    model_name = model_name.lower()

    if model_name not in MODEL:
        raise ValueError(f"Unknown model: {model_name}")

    return MODEL[model_name](
        num_points=num_points,
        head_type=head_type,
        input_size=input_size,
        Nx=Nx,
        Ny=Ny,
        **kwargs
    )