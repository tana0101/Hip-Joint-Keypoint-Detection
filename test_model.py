import argparse, torch
torch.serialization.add_safe_globals([argparse.Namespace])

from torchinfo import summary
from models.model import initialize_model

if __name__ == "__main__":
    model = initialize_model("convnext_small_mg1234", num_points=6, head_type="simcc_2d", input_size=(224, 224), Nx=448, Ny=448)
    summary(model, input_size=(1, 3, 224, 224))