import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import math

from datasets.transforms import get_hip_base_transform
from datasets.hip_crop_keypoints import HipCropKeypointDataset, MirroredToSideDataset
from datasets.augment import AugmentedKeypointDataset
from utils.keypoint_metrics import calculate_nme, calculate_pixel_error
from utils.keypoints import get_preds_and_targets
from utils.experiment import build_experiment_name
from utils.train_vis import GraphWrapper, plot_training_progress
from utils.simcc import (
    compute_loss_simcc,
    simcc_label_encoder,
    simcc_loss_fn,
)
from utils.regression import compute_loss_direct_regression
from models.model import initialize_model
from pathlib import Path

LOGS_DIR = "logs"
MODELS_DIR = "weights"

def train(data_dir, model_name, input_size, epochs, learning_rate, batch_size, side, mirror, head_type="direct_regression", split_ratio=2, sigma=6.0, fold_index=None):
    
    data_dir = Path(data_dir)

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()
    
    transform = get_hip_base_transform(input_size)

    # 單側資料集
    train_dataset = HipCropKeypointDataset(
        img_dir = data_dir / 'train' / 'images',
        annotation_dir = data_dir / 'train' / 'annotations',
        detections_dir = data_dir / 'train' / 'detections',
        side=side,
        transform=transform,
        crop_expand=0.10,
        keep_square=True,
        input_size=input_size
    )
    val_dataset = HipCropKeypointDataset(
        img_dir = data_dir / 'val' / 'images',
        annotation_dir = data_dir / 'val' / 'annotations',
        detections_dir = data_dir / 'val' / 'detections',
        side=side,
        transform=transform,
        crop_expand=0.10,
        keep_square=True,
        input_size=input_size
    )
    
    points_count = train_dataset.num_keypoints
    print(f"Auto-detected dataset points per side: {points_count}")
    
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
    if head_type in ["simcc_1d", "simcc_2d", "simcc_2d_deconv"]:
        Nx = int(input_size * split_ratio)
        Ny = int(input_size * split_ratio)
    else:
        Nx = None
        Ny = None
    model = initialize_model(model_name, num_points=points_count, head_type=head_type, input_size=(input_size, input_size), Nx=Nx, Ny=Ny)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if head_type == "direct_regression":
        criterion = nn.MSELoss()
        def loss_fn(outputs, keypoints):
            return compute_loss_direct_regression(outputs, keypoints, points_count, criterion)
    elif head_type in ["simcc_1d", "simcc_2d", "simcc_2d_deconv"]:
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
    if fold_index is not None:
        exp_name += f"_fold{fold_index}"
    
    run_name = exp_name
    tb_dir = os.path.join(LOGS_DIR, "tb", run_name)
    writer = SummaryWriter(log_dir=tb_dir)
    try:
        dummy = torch.randn(1, 3, input_size, input_size, device=device)
        graph_model = GraphWrapper(model, head_type)
        writer.add_graph(graph_model, dummy)
    except Exception as e:
        print(f"[TB] Skip add_graph: {e}")
        
    global_step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        nme_list, pixel_error_list = [], []

        # Training Loop
        for images, keypoints, crop_sizes, img_names in train_loader:
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

            
            preds, targets = get_preds_and_targets(
                outputs=outputs,
                keypoints=keypoints,
                head_type=head_type,
                Nx=Nx,
                Ny=Ny,
                input_size=input_size,
            )
            
            # Unpack the original image sizes
            widths, heights = crop_sizes
            widths = widths.cpu().numpy()  
            heights = heights.cpu().numpy() 

            crop_sizes = [(w, h) for w, h in zip(widths, heights)]

            # Calculate NME and Pixel Error for each sample
            for i in range(len(crop_sizes)):
                img_size = crop_sizes[i]

                nme = calculate_nme(preds[i], targets[i], points_count, img_size)
                pixel_error = calculate_pixel_error(preds[i], targets[i], points_count, img_size, input_size)

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
            for images, keypoints, crop_sizes, img_names in val_loader:
                images, keypoints = images.to(device), keypoints.to(device)

                outputs = model(images)

                # Calculate loss
                loss = loss_fn(outputs, keypoints)
                val_loss += loss.item()

                preds, targets = get_preds_and_targets(
                    outputs=outputs,
                    keypoints=keypoints,
                    head_type=head_type,
                    Nx=Nx,
                    Ny=Ny,
                    input_size=input_size,
                )

                # Unpack the original image sizes
                widths, heights = crop_sizes
                widths = widths.cpu().numpy()  
                heights = heights.cpu().numpy()

                crop_sizes = [(w, h) for w, h in zip(widths, heights)]

                # Calculate NME and Pixel Error for each sample
                for i in range(len(crop_sizes)):
                    img_size = crop_sizes[i]

                    nme = calculate_nme(preds[i], targets[i], points_count, img_size)
                    pixel_error = calculate_pixel_error(preds[i], targets[i], points_count, img_size, input_size)

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
            best_epoch_index = epoch
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

    # 找出 best epoch 的 train/val metrics（方便 K-fold 平均）
    if best_epoch_index >= 0:
        best_train_loss = epoch_losses[best_epoch_index]
        best_train_nme = epoch_nmes[best_epoch_index]
        best_train_pixel = epoch_pixel_errors[best_epoch_index]
        best_val_loss = val_losses[best_epoch_index]
        best_val_nme = val_nmes[best_epoch_index]
        best_val_pixel = val_pixel_errors[best_epoch_index]
    else:
        best_train_loss = best_train_nme = best_train_pixel = None
        best_val_loss = best_val_nme = best_val_pixel = None
    
    # Log hyperparameters and metrics to TensorBoard
    hparam_dict = {
        "model": model_name,
        "head_type": head_type,
        "sr": split_ratio if head_type in ["simcc_1d", "simcc_2d", "simcc_2d_deconv"] else 1.0,
        "sigma": sigma if head_type in ["simcc_1d", "simcc_2d", "simcc_2d_deconv"] else None,
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
    
    return {
        "epoch_losses": epoch_losses,
        "val_losses": val_losses,
        "epoch_nmes": epoch_nmes,
        "val_nmes": val_nmes,
        "epoch_pixel_errors": epoch_pixel_errors,
        "val_pixel_errors": val_pixel_errors,
        "best_epoch_index": best_epoch_index,
        "best_val_pixel_error": best_val_pixel_error,
        "best_train_loss": best_train_loss,
        "best_train_nme": best_train_nme,
        "best_train_pixel": best_train_pixel,
        "best_val_loss": best_val_loss,
        "best_val_nme": best_val_nme,
        "best_val_pixel": best_val_pixel,
        "exp_name": exp_name,
    }

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
    parser.add_argument("--head_type", type=str, default="direct_regression", choices=["direct_regression", "simcc_1d", "simcc_2d", "simcc_2d_deconv"], help="Type of model head to use")
    parser.add_argument("--split_ratio", type=float, default=2, help="SimCC split ratio for label encoding")
    parser.add_argument("--sigma", type=float, default=6.0, help="Sigma for SimCC label encoding")
    
    args = parser.parse_args()

    train(args.data_dir, args.model_name, args.input_size, args.epochs, args.learning_rate,
         args.batch_size, args.side, args.mirror, head_type=args.head_type, split_ratio=args.split_ratio, sigma=args.sigma)

    # python3 train_hip_crop_keypoints.py --data_dir data --model_name convnext_small_mg1234 --input_size 224 --epochs 200 --learning_rate 0.0001 --batch_size 32 --side left --mirror --head_type simcc_2d --split_ratio 2.0 --sigma 2.0