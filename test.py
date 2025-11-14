import argparse, torch
torch.serialization.add_safe_globals([argparse.Namespace])

from mambavision import create_model

import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

from models.model import initialize_model

class MambaVisionSmall(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        model = create_model('mamba_vision_S', pretrained=True, model_path="/tmp/mambavision_small_1k.pth.tar")
        
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_points * 2)
        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    model = MambaVisionSmall(num_points=6)
    # model = initialize_model("mambavision_small_convnext", num_points=6)
    summary(model, input_size=(1, 3, 224, 224))