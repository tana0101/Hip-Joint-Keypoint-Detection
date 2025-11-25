import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import json
import math

from utils.simcc import (
    compute_loss_simcc,
    simcc_label_encoder,
    decode_simcc_to_xy,
    simcc_loss_fn,
    SimCCLoss
)

from utils.regression import compute_loss_direct_regression

from models.model import initialize_model

LOGS_DIR = "logs"
MODELS_DIR = "weights"
POINTS_COUNT = 6  # 每側6個關鍵點

SIDE_LABELS = {"left": "LeftHip", "right": "RightHip"}
SIDE_INDEX = {"left": (0, 6), "right": (6, 12)}  # 12 點中的 slice
REORDER_6 = [2, 1, 0, 5, 4, 3] # 鏡像翻轉後的點重排序

def read_detection_for_image(detections_dir: str, img_name: str):
    """
    detections_dir: e.g., .../train/detections
    img_name:      e.g., 19435927-050.jpg  -> 對應 19435927-050.json

    回傳: dets[label] = (x1, y1, x2, y2) ；兩點會自動取 min/max 變成 bbox
    支援你資料裡「兩點是任意兩個角」的情況（不必一定是左上/右下）。
    """
    stem = os.path.splitext(img_name)[0]
    json_path = os.path.join(detections_dir, stem + ".json")
    with open(json_path, "r") as f:
        data = json.load(f)  # 結構如 { "image": "...", "objects": [ {label, points:[[x,y],[x,y]]}, ...] }

    dets = {}
    for obj in data["objects"]:
        (xa, ya), (xb, yb) = obj["points"][0], obj["points"][1]
        x1, x2 = (xa, xb) if xa <= xb else (xb, xa)
        y1, y2 = (ya, yb) if ya <= yb else (yb, ya)
        dets[obj["label"]] = (float(x1), float(y1), float(x2), float(y2))
    return dets

def parse_12pt_csv_to_np(annotation_csv_path):
    """
    讀 12 點 CSV（你原本的格式），回傳 shape=(12,2) 的 numpy。
    支援每列像 "(x,y)" 或 "x,y"。若你的 CSV 不同，這裡再微調。
    """
    row = pd.read_csv(annotation_csv_path, header=None).values.flatten()
    pts = []
    for token in row:
        token = str(token).strip().strip('"').strip("'").strip()
        token = token.replace("(", "").replace(")", "")
        x_str, y_str = token.split(",")
        pts.append([float(x_str), float(y_str)])
    pts = np.array(pts, dtype=np.float32)  # (12,2)
    assert pts.shape == (12, 2), f"Expect 12 points, got {pts.shape}"
    return pts


# Custom dataset class
class HipCropKeypointDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, detections_dir, side="left",
                 transform=None, crop_expand=0.10, keep_square=True, input_size=224):
        """
        Args:
            img_dir:      train/images
            annotation_dir: train/annotations
            detections_dir: train/detections
            side:         "left" 或 "right"
            transform:    會於「裁切後」影像才做（例：Equalize, Resize, ToTensor）
            crop_expand:  bbox 邊界外擴比例（相對於 bbox 邊長）
            keep_square:  True 則在裁切時先變成近似正方，利於之後等比縮放
        """
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.detections_dir = detections_dir
        self.transform = transform
        self.side = side.lower()
        assert self.side in ("left", "right")
        self.side_label = SIDE_LABELS[self.side]
        self.side_start, self.side_end = SIDE_INDEX[self.side]
        self.crop_expand = crop_expand
        self.keep_square = keep_square
        self.input_size = input_size

        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        self.annotations = sorted([f for f in os.listdir(annotation_dir) if f.endswith(".csv")])

        # 簡單對齊檔名（假設排序一致：同名 .jpg 對應同名 .csv）
        # 若你的檔名不同步，建議用字典：image_name -> annotation_name
        assert len(self.images) == len(self.annotations), "Images/annotations count mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, self.annotations[idx])

        img = Image.open(img_path).convert("L")  # Grayscale PIL
        W, H = img.size

        # 1) 讀這張圖的 bbox（每張一檔）
        dets = read_detection_for_image(self.detections_dir, img_name)
        if self.side_label not in dets:
            raise KeyError(f"No detection for {img_name} / {self.side_label}")
        x1, y1, x2, y2 = dets[self.side_label]

        # 可能高度=0（兩點同 y），修成接近正方
        if y2 - y1 < 2:
            width = max(2.0, x2 - x1)
            # 讓 bbox 高度 = 寬度，往下延伸（如果超界再往上調整）
            y2 = min(H, y1 + width)
            if y2 - y1 < width:  # 超界了往上撐
                y1 = max(0.0, y2 - width)

        # keep_square + 外擴
        bw, bh = (x2 - x1), (y2 - y1)
        if self.keep_square:
            side_len = max(bw, bh)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            x1, x2 = cx - side_len / 2.0, cx + side_len / 2.0
            y1, y2 = cy - side_len / 2.0, cy + side_len / 2.0
            bw = bh = side_len

        pad = self.crop_expand
        x1 -= bw * pad;  x2 += bw * pad
        y1 -= bh * pad;  y2 += bh * pad

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # 2) 讀 12 點 → 取單側 6 點
        pts12 = parse_12pt_csv_to_np(ann_path)      # (12,2)
        pts6  = pts12[self.side_start:self.side_end]  # (6,2)

        # 映射到裁切座標（以裁切左上為原點）
        crop_w, crop_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        pts6_crop = np.empty_like(pts6)
        pts6_crop[:, 0] = pts6[:, 0] - x1
        pts6_crop[:, 1] = pts6[:, 1] - y1

        # 取得裁切圖
        img_crop = img.crop((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))

        # 3) 裁切後再 transform（等同你原來）
        img_out = self.transform(img_crop)  # Tensor [3, 224, 224]

        # 裁切→縮放比例（到 224×224）
        sx = self.input_size / crop_w
        sy = self.input_size / crop_h
        pts6_resized = np.empty_like(pts6_crop)
        pts6_resized[:, 0] = pts6_crop[:, 0] * sx
        pts6_resized[:, 1] = pts6_crop[:, 1] * sy

        keypoints = torch.tensor(pts6_resized.reshape(-1), dtype=torch.float32)  # (12,)
        
        crop_size = (x2 - x1, y2 - y1)
        return img_out, keypoints, crop_size

class AugmentedKeypointDataset(Dataset):
    def __init__(self, original_dataset, max_angle=0, max_translate_x=0, max_translate_y=0):
        """
        Args:
            original_dataset: The original dataset to augment.
            max_angle: Maximum rotation angle (degrees) for augmentation.
            max_translate_x: Maximum horizontal translation (pixels).
            max_translate_y: Maximum vertical translation (pixels).
        """
        self.original_dataset = original_dataset
        self.max_angle = max_angle
        self.max_translate_x = max_translate_x
        self.max_translate_y = max_translate_y

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Retrieve the original data
        image, keypoints, crop_size = self.original_dataset[idx]
        crop_width, crop_height = crop_size

        # Generate random augmentation parameters
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        translate_x = np.random.uniform(-self.max_translate_x, self.max_translate_x)
        translate_y = np.random.uniform(-self.max_translate_y, self.max_translate_y)

        # Apply rotation augmentation
        rotated_image = transforms.functional.rotate(image, angle)

        # Calculate rotation matrix for keypoints
        angle_rad = np.deg2rad(-angle)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        # Center of rotation (image center)
        img_width, img_height = image.shape[1], image.shape[2]
        center_x, center_y = img_width / 2, img_height / 2

        # Apply rotation to keypoints
        keypoints_rotated = []
        for i in range(0, len(keypoints), 2):
            x = keypoints[i]
            y = keypoints[i + 1]
            x_new = cos_theta * (x - center_x) - sin_theta * (y - center_y) + center_x
            y_new = sin_theta * (x - center_x) + cos_theta * (y - center_y) + center_y
            keypoints_rotated.extend([x_new, y_new])

        # Apply translation to keypoints
        keypoints_translated = [
            coord + translate_x if i % 2 == 0 else coord + translate_y
            for i, coord in enumerate(keypoints_rotated)
        ]

        # Apply translation to the image
        translated_image = transforms.functional.affine(
            rotated_image, angle=0, translate=(translate_x, translate_y), scale=1, shear=0
        )

        return translated_image, torch.tensor(keypoints_translated, dtype=torch.float32), (crop_width, crop_height)

class MirroredToSideDataset(Dataset):
    """
    將「對側」資料集（left 或 right）在 224×224 空間水平鏡像，並依 REORDER_6 重新排序 6 個點，
    使其幾何語義符合 target_side（'left' 或 'right'），點的語義不變。
    """
    def __init__(self, source_dataset, target_side: str):
        """
        source_dataset: HipCropKeypointDataset(side='left' 或 'right')
        target_side: 'left' 或 'right' （轉成這一側）
        """
        assert target_side in ("left", "right")
        self.ds = source_dataset
        self.target_side = target_side

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, keypoints, crop_size = self.ds[idx]  # image: Tensor [3,224,224]；keypoints: Tensor 長度=12(6點xy)
        # 1) 水平鏡像影像
        image_flipped = TF.hflip(image)
        # 2) x 座標鏡像（在 224×224）
        _, H, W = image.shape
        k = keypoints.clone()
        for i in range(0, len(k), 2):
            k[i] = (W - 1) - k[i]  # 只改 x，y 不變
        # 3) 索引重排（6點）：[0..5] -> [2,1,0,5,4,3]
        k_xy = k.view(-1, 2)                      # (6,2)
        k_xy = k_xy[REORDER_6, :]                 # 重排
        k_out = k_xy.reshape(-1)                  # 還原為 (12,)
        return image_flipped, k_out, crop_size

def calculate_nme(preds, targets, img_size):
    """
    NME in crop space.
    img_size: (crop_width, crop_height)
    """
    preds = preds.reshape(POINTS_COUNT, 2)  # Reshape to (12, 2)
    targets = targets.reshape(POINTS_COUNT, 2)  # Reshape to (12, 2)

    # Calculate the Euclidean distance for each keypoint
    pixel_distances = np.linalg.norm(preds - targets, axis=1)  # Shape: (12,)

    # Calculate the diagonal of the image
    img_diag = np.sqrt(img_size[0]**2 + img_size[1]**2)  # Scalar

    # Normalize distances by the image diagonal
    norm_distances = pixel_distances / img_diag  # Shape: (12,)

    # Return the mean of the normalized distances
    return np.mean(norm_distances)

def calculate_pixel_error(preds, targets, img_size, input_size):
    """
    Pixel error in crop pixel units.
    img_size: (crop_width, crop_height)
    """
    preds = preds.reshape(POINTS_COUNT, 2)  # Reshape to (12, 2)
    targets = targets.reshape(POINTS_COUNT, 2)  # Reshape to (12, 2)

    # Unpack the original image dimensions
    original_width, original_height = img_size

    # Calculate the scaling factors for width and height
    scale_x = original_width / input_size  # Assuming input_size is the model input size
    scale_y = original_height / input_size

    # Scale the predictions and targets
    preds_scaled = preds * np.array([scale_x, scale_y])  # Shape: (12, 2)
    targets_scaled = targets * np.array([scale_x, scale_y])  # Shape: (12, 2)

    # Calculate the Euclidean distance for each keypoint
    pixel_distances = np.linalg.norm(preds_scaled - targets_scaled, axis=1)  # Shape: (12,)

    # Return the mean of the pixel distances
    return np.mean(pixel_distances)

def display_image(dataset, index):
    # Get the nth image and keypoints
    image, keypoints, original_size = dataset[index]
    print(f"Displaying image {index}")
    print("Original size:", original_size)
    print("Image shape:", image.shape)
    
    # Convert the image to a NumPy array
    image_np = image.permute(1, 2, 0).numpy()

    # Display the image
    plt.imshow(image_np, cmap='gray')
    plt.title(f"Image {index} with Keypoints")
    
    # Plot the keypoints with numbering
    for i in range(0, len(keypoints), 2):
        x = keypoints[i].item()  
        y = keypoints[i + 1].item()  
        plt.scatter(x, y, c='red', s=20)
        plt.text(x, y, f'{i//2 + 1}', color='yellow', fontsize=12)  # Add number next to each point

    plt.show()

def plot_training_progress(epochs_range, epoch_losses, val_losses, epoch_nmes, val_nmes, epoch_pixel_errors, val_pixel_errors, 
                           title_suffix="", start_epoch=1, loss_ylim=None, nme_ylim=None, pixel_error_ylim=None):
    """
    Function to plot training and validation progress for Loss, NME, and Pixel Error.
    Args:
        epochs_range: Range of epochs to plot
        epoch_losses: Training losses for each epoch
        val_losses: Validation losses for each epoch
        epoch_nmes: Training NME for each epoch
        val_nmes: Validation NME for each epoch
        epoch_pixel_errors: Training Pixel Error for each epoch
        val_pixel_errors: Validation Pixel Error for each epoch
        title_suffix: Optional suffix for the plot titles (e.g., " (Epoch 20 onwards)")
        start_epoch: Epoch to start plotting from (default is 1, to plot from the start)
        loss_ylim: Tuple for Loss y-axis limits (e.g., (0, 1))
        nme_ylim: Tuple for NME y-axis limits (e.g., (0, 0.1))
        pixel_error_ylim: Tuple for Pixel Error y-axis limits (e.g., (0, 5))
    """
    # Extract data from start_epoch
    if start_epoch > 1:
        epochs_range = range(start_epoch, len(epoch_losses) + 1)
        epoch_losses = epoch_losses[start_epoch - 1:]
        epoch_nmes = epoch_nmes[start_epoch - 1:]
        epoch_pixel_errors = epoch_pixel_errors[start_epoch - 1:]
        val_losses = val_losses[start_epoch - 1:]
        val_nmes = val_nmes[start_epoch - 1:]
        val_pixel_errors = val_pixel_errors[start_epoch - 1:]

    plt.figure(figsize=(12, 6))

    # Plot Loss with log scale
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, epoch_losses, label="Training Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.title(f"Loss{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log)")
    plt.yscale('log')  # Log scale
    if loss_ylim:
        plt.ylim(loss_ylim)  # Set y-axis limits for Loss
    plt.legend()

    # Plot NME
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, epoch_nmes, label="Training NME")
    plt.plot(epochs_range, val_nmes, label="Validation NME")
    plt.title(f"NME{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("NME (log)")
    plt.yscale('log')  # Log scale
    if nme_ylim:
        plt.ylim(nme_ylim)  # Set y-axis limits for NME
    plt.legend()

    # Plot Pixel Error
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, epoch_pixel_errors, label="Training Pixel Error")
    plt.plot(epochs_range, val_pixel_errors, label="Validation Pixel Error")
    plt.title(f"Pixel Error{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Pixel Error (log)")
    plt.yscale('log')  # Log scale
    if pixel_error_ylim:
        plt.ylim(pixel_error_ylim)  # Set y-axis limits for Pixel Error
    plt.legend()

    plt.tight_layout()

def build_experiment_name(
    model_name: str,
    head_type: str,
    side: str,
    mirror: bool,
    input_size: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    split_ratio: float | None = None,
    sigma: float | None = None,
) -> str:
    """
    統一產生實驗 / 檔名用的 base name，不含副檔名。
    規則：
      simcc: model_simcc_sr2.0_sigma6.0_cropleft[_mirror]_448_300_0.0001_32
      direct_regression: model_direct_regression_cropleft[_mirror]_448_300_0.0001_32
    """
    base = f"{model_name}_{head_type}"

    # 只有 simcc 才會加 sr 跟 sigma
    if head_type == "simcc" and split_ratio is not None:
        base += f"_sr{split_ratio}"
    if head_type == "simcc" and sigma is not None:
        base += f"_sigma{sigma}"

    base += f"_crop{side}"
    if mirror:
        base += "_mirror"

    # 統一後面這四個
    base += f"_{input_size}_{epochs}_{learning_rate}_{batch_size}"
    return base

def main(data_dir, model_name, input_size, epochs, learning_rate, batch_size, side, mirror, head_type="direct_regression", split_ratio=2, sigma=6.0):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Lambda(lambda img: ImageOps.equalize(img)),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    # 單側資料集
    train_dataset = HipCropKeypointDataset(
        img_dir=os.path.join(data_dir, 'train/images'),
        annotation_dir=os.path.join(data_dir, 'train/annotations'),
        detections_dir=os.path.join(data_dir, 'train/detections'),
        side=side,
        transform=transform,
        crop_expand=0.10,
        keep_square=True,
        input_size=input_size
    )
    val_dataset = HipCropKeypointDataset(
        img_dir=os.path.join(data_dir, 'val/images'),
        annotation_dir=os.path.join(data_dir, 'val/annotations'),
        detections_dir=os.path.join(data_dir, 'val/detections'),
        side=side,
        transform=transform,
        crop_expand=0.10,
        keep_square=True,
        input_size=input_size
    )
    
    # 使用鏡像資料擴增
    if mirror:
        opposite_side = "right" if side == "left" else "left"
        print(f"Preparing mirrored data from {opposite_side} side...")
        opposite_train_dataset = HipCropKeypointDataset(
            img_dir=os.path.join(data_dir, 'train/images'),
            annotation_dir=os.path.join(data_dir, 'train/annotations'),
            detections_dir=os.path.join(data_dir, 'train/detections'),
            side=opposite_side,
            transform=transform,
            crop_expand=0.10,
            keep_square=True,
            input_size=input_size
        )
        mirrored_dataset = MirroredToSideDataset(opposite_train_dataset, target_side=side)
        # 合併原本的單側資料集與鏡像資料集
        train_dataset = ConcatDataset([train_dataset, mirrored_dataset])
        print(f"using mirrored data from {opposite_side} side, total training samples: {len(train_dataset)}")
        
    # 資料增強：可以直接沿用你原本的 AugmentedKeypointDataset
    augmented_dataset = AugmentedKeypointDataset(train_dataset, max_translate_x=20, max_translate_y=20)
    augmented_dataset2 = AugmentedKeypointDataset(train_dataset, max_angle=10)
    combined_dataset = ConcatDataset([train_dataset, augmented_dataset, augmented_dataset2])
    
    # # To visualize the dataset
    # display_image(train_dataset, 0)
    # display_image(mirrored_dataset, 0)
    # for i in range(0, 3):
    #     display_image(augmented_dataset, i)
    #     display_image(augmented_dataset2, i)
    
    train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=12, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=12, pin_memory=True, prefetch_factor=4)

    print(f"[{side.upper()}] Training samples: {len(combined_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize the model, loss function, and optimizer
    if head_type == "simcc":
        Nx = int(input_size * split_ratio)
        Ny = int(input_size * split_ratio)
    else:
        Nx = None
        Ny = None
    model = initialize_model(model_name, POINTS_COUNT, head_type=head_type, Nx=Nx, Ny=Ny)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if head_type == "direct_regression":
        def loss_fn(outputs, keypoints):
            criterion = nn.MSELoss()
            return compute_loss_direct_regression(outputs, keypoints, criterion)
    elif head_type == "simcc":
        def loss_fn(outputs, keypoints):
            return compute_loss_simcc(
                outputs,
                keypoints,
                Nx=Nx,
                Ny=Ny,
                input_size=input_size,
                sigma=sigma,
                simcc_label_encoder=simcc_label_encoder,
                simcc_loss_fn=simcc_loss_fn,
            )
    else:
        raise ValueError(f"Unknown head_type: {head_type}")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # Scheduler: Warm-up + Cosine
    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(0.1 * total_steps))   # 前 10% steps 線性升溫
    base_lr = learning_rate
    min_lr = 1e-6
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / float(warmup_steps)                     # 線性 warm-up: 0 -> 1
        # Cosine decay: 1 -> (min_lr/base_lr)
        t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * t))               # 1 -> 0
        return cosine * (1 - min_lr / base_lr) + (min_lr / base_lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Save the model's training progress
    epoch_losses, epoch_nmes, epoch_pixel_errors = [], [], []
    val_losses, val_nmes, val_pixel_errors = [], [], []
    best_val_pixel_error, best_model_state = float('inf'), None

    # TensorBoard writer
    exp_name = build_experiment_name(
        model_name=model_name,
        head_type=head_type,
        side=side,
        mirror=mirror,
        input_size=input_size,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        split_ratio=split_ratio,
        sigma=sigma,
    )
    run_name = exp_name
    tb_dir = os.path.join(LOGS_DIR, "tb", run_name)
    writer = SummaryWriter(log_dir=tb_dir)
    try:
        dummy = torch.randn(1, 3, input_size, input_size, device=device)
        writer.add_graph(model, dummy)
    except Exception as e:
        print(f"[TB] Skip add_graph: {e}")

    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        nme_list, pixel_error_list = [], []

        # Training Loop
        for images, keypoints, crop_sizes in train_loader:
            images, keypoints = images.to(device), keypoints.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # (B, 12) for 6 points

            # Calculate loss
            loss = loss_fn(outputs, keypoints)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()

            if head_type == "direct_regression":
                preds = outputs.detach().cpu().numpy()
            elif head_type == "simcc":
                pred_x, pred_y = outputs
                preds = decode_simcc_to_xy(pred_x, pred_y, Nx=Nx, Ny=Ny, input_size=input_size)
                preds = preds.detach().cpu().numpy()
            targets = keypoints.detach().cpu().numpy()
            
            # Unpack the original image sizes
            widths, heights = crop_sizes
            widths = widths.cpu().numpy()  
            heights = heights.cpu().numpy() 

            crop_sizes = [(w, h) for w, h in zip(widths, heights)]

            # Calculate NME and Pixel Error for each sample
            for i in range(len(crop_sizes)):
                img_size = crop_sizes[i]

                nme = calculate_nme(preds[i], targets[i], img_size)
                pixel_error = calculate_pixel_error(preds[i], targets[i], img_size, input_size)

                nme_list.append(nme)
                pixel_error_list.append(pixel_error)
            
            # Log step loss to TensorBoard
            writer.add_scalar("train/step_loss", loss.item(), global_step)
            global_step += 1

        epoch_loss = running_loss / len(train_loader)
        epoch_nme = float(np.mean(nme_list)) if nme_list else 0.0
        epoch_pixel_error = float(np.mean(pixel_error_list)) if pixel_error_list else 0.0
        epoch_losses.append(epoch_loss); epoch_nmes.append(epoch_nme); epoch_pixel_errors.append(epoch_pixel_error)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} | NME: {epoch_nme:.4f} | Pixel: {epoch_pixel_error:.4f}")
        
        # Validation Loop
        model.eval()  # Set the model to evaluation mode
        val_loss, val_nmes_list, val_pixel_error_list = 0.0, [], []

        with torch.no_grad():  # No need to track gradients during validation
            for images, keypoints, crop_sizes in val_loader:
                images, keypoints = images.to(device), keypoints.to(device)

                outputs = model(images)

                # Calculate loss
                loss = loss_fn(outputs, keypoints)
                val_loss += loss.item()

                if head_type == "direct_regression":
                    preds = outputs.detach().cpu().numpy()
                elif head_type == "simcc":
                    pred_x, pred_y = outputs
                    preds = decode_simcc_to_xy(pred_x, pred_y, Nx=Nx, Ny=Ny, input_size=input_size)
                    preds = preds.detach().cpu().numpy()
                targets = keypoints.detach().cpu().numpy()
                
                # Unpack the original image sizes
                widths, heights = crop_sizes
                widths = widths.cpu().numpy()  
                heights = heights.cpu().numpy()

                crop_sizes = [(w, h) for w, h in zip(widths, heights)]

                # Calculate NME and Pixel Error for each sample
                for i in range(len(crop_sizes)):
                    img_size = crop_sizes[i]

                    nme = calculate_nme(preds[i], targets[i], img_size)
                    pixel_error = calculate_pixel_error(preds[i], targets[i], img_size, input_size)

                    val_nmes_list.append(nme)
                    val_pixel_error_list.append(pixel_error)

        val_loss /= max(1, len(val_loader))
        val_nme = float(np.mean(val_nmes_list)) if val_nmes_list else 0.0
        val_pixel_error = float(np.mean(val_pixel_error_list)) if val_pixel_error_list else 0.0
        val_losses.append(val_loss); val_nmes.append(val_nme); val_pixel_errors.append(val_pixel_error)
        print(f"Validation Loss: {val_loss:.4f} | NME: {val_nme:.4f} | Pixel: {val_pixel_error:.4f}")

        # Save the model with the best validation loss
        if val_pixel_error < best_val_pixel_error:
            best_val_pixel_error = val_pixel_error
            best_model_state = model.state_dict()  # Save the model state at the best point
            print(f"---------------- Validation pixel error improved to {best_val_pixel_error:.4f}, saving model. ----------------")
             
        # Log metrics to TensorBoard
        writer.add_scalars("loss/epoch", {"train": epoch_loss, "val": val_loss}, epoch)
        writer.add_scalars("nme/epoch",  {"train": epoch_nme,  "val": val_nme},  epoch)
        writer.add_scalars("pixel/epoch",{"train": epoch_pixel_error, "val": val_pixel_error}, epoch)
        writer.add_scalar("opt/lr", optimizer.param_groups[0]['lr'], epoch)
        
    # Save the training and validation progress
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save the best model (with the lowest validation loss)
    if best_model_state:
        model_path = os.path.join(MODELS_DIR, f"{exp_name}_best.pth")
        torch.save(best_model_state, model_path)
        print(f"Best model saved to: {model_path}")

    # Log hyperparameters and metrics to TensorBoard
    hparam_dict = {
        "model": model_name,
        "head_type": head_type,
        "sr": split_ratio if head_type == "simcc" else 1.0,
        "sigma": sigma if head_type == "simcc" else None,
        "side": side,
        "mirror": int(mirror),
        "epochs": epochs,
        "lr": learning_rate,
        "batch_size": batch_size,
        "input_size": input_size,
    }
    metric_dict = {"hparam/best_val_pixel_error": best_val_pixel_error,
                   "hparam/final_val_loss": val_losses[-1] if len(val_losses) else float('inf'),
                   "hparam/final_val_nme": val_nmes[-1] if len(val_nmes) else 0.0}
    writer.add_hparams(hparam_dict, metric_dict)

    writer.flush()
    writer.close()

    # --------------------------------------------------------------plotting--------------------------------------------------------------
    # Save the training progress plot
    epochs_range = range(1, epochs + 1)
    plot_training_progress(
        epochs_range, epoch_losses, val_losses, epoch_nmes, val_nmes, epoch_pixel_errors, val_pixel_errors,
        loss_ylim=(0.01, 50),
        nme_ylim=(0.0001, 0.02),
        pixel_error_ylim=(0.01, 50),
    )

    # Save the training plot
    training_plot_path = os.path.join(LOGS_DIR, f"{exp_name}_training_plot.png")
    plt.savefig(training_plot_path)
    print(f"Training plot saved to: {training_plot_path}")
    plt.show()

    # Save the Loss, NME, and Pixel Error to a text file
    training_log_path = os.path.join(LOGS_DIR, f"{exp_name}_training_log.txt")
    with open(training_log_path, "w") as f:
        for epoch, (loss, nme, pixel_error, val_loss, val_nme, val_pixel_error) in enumerate(
                zip(epoch_losses, epoch_nmes, epoch_pixel_errors, val_losses, val_nmes, val_pixel_errors), 1):
            f.write(f"Epoch {epoch}: Loss = {loss:.4f}, NME = {nme:.4f}, Pixel Error = {pixel_error:.4f}, "
                    f"Val Loss = {val_loss:.4f}, Val NME = {val_nme:.4f}, Val Pixel Error = {val_pixel_error:.4f}\n")
    print(f"Training log saved to: {training_log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name: 'efficientnet', 'resnet', or 'vgg'")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size for the model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of samples per batch")
    parser.add_argument("--side", type=str, default="left", choices=["left", "right"], help="Side to train on: 'left' or 'right'")
    parser.add_argument("--mirror", action="store_true", help="Whether to include mirrored data from the opposite side")
    parser.add_argument("--head_type", type=str, default="direct_regression", choices=["direct_regression", "simcc"], help="Which keypoint head to use")
    parser.add_argument("--split_ratio", type=float, default=2, help="Validation split ratio")
    parser.add_argument("--sigma", type=float, default=6.0, help="Sigma for SimCC label encoding")
    
    args = parser.parse_args()

    main(args.data_dir, args.model_name, args.input_size, args.epochs, args.learning_rate,
         args.batch_size, args.side, args.mirror, head_type=args.head_type, split_ratio=args.split_ratio, sigma=args.sigma)

    # python3 train_hip_crop_keypoints.py --data_dir data --model_name convnext --input_size 224 --epochs 300 --learning_rate 0.0001 --batch_size 32 --side left --mirror --head_type simcc --split_ratio 2
    # python3 train_hip_crop_keypoints.py --data_dir data --model_name convnext --input_size 224 --epochs 300 --learning_rate 0.0001 --batch_size 32 --side left --mirror --head_type direct_regression