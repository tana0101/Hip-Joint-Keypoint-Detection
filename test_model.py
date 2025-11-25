import argparse, torch
torch.serialization.add_safe_globals([argparse.Namespace])

from torchinfo import summary
from models.model import initialize_model

if __name__ == "__main__":
    model = initialize_model("hrnet_w32", num_points=6, head_type="simcc", Nx=448, Ny=448)
    summary(model, input_size=(1, 3, 224, 224))