import argparse, torch
torch.serialization.add_safe_globals([argparse.Namespace])

from mambavision import create_model

import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

from models.model import initialize_model

from models.inceptionnext.inceptionnext import (
    inceptionnext_tiny,
    inceptionnext_small,
    inceptionnext_base,
)

from models.convnextv2.convnextv2 import (
    convnextv2_tiny,
    convnextv2_base,
    convnextv2_large
)

if __name__ == "__main__":
    # model = initialize_model("convnext_small_custom", num_points=6)
    model = initialize_model("convnext_v2_base", num_points=6)
    # model = convnextv2_base(num_classes=1000)
    # model = models.convnext_small(pretrained=Tr)
    # model = inceptionnext_small(pretrained=True)
    # model = models.convnext_small(pretrained=True)
    summary(model, input_size=(1, 3, 224, 224))