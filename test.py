import argparse, torch
torch.serialization.add_safe_globals([argparse.Namespace])

from mambavision import create_model

import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

class MambaVisionSmall(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = create_model('mamba_vision_S', pretrained=True, model_path="/tmp/mambavision_small_1k.pth.tar")
        in_dim = model.head.in_features
        model.head = nn.Linear(in_dim, num_points * 2)

if __name__ == "__main__":
    model = MambaVisionSmall(num_points=10)
    
    