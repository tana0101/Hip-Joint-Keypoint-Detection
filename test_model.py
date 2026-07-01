import argparse, torch
torch.serialization.add_safe_globals([argparse.Namespace])

from torchinfo import summary
from models.model import initialize_model
from ultralytics import YOLO

if __name__ == "__main__":
    # model = initialize_model("convnext_tiny_fpn1234concat", num_points=4, head_type="simcc_2d", input_size=(224, 224), Nx=672, Ny=672)
    # summary(model, input_size=(1, 3, 224, 224))
    
    model = YOLO('weights/yolo26s_mtddh_set.pt')
    model.info(detailed=True)
    
    # model = YOLO('hip-yolo-xray-seg.pt')
    # model.info(imgsz=800)