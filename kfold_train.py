import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.model_selection import KFold
from itertools import cycle

IMAGE_SIZE = 224 # Image size for the model
LOGS_DIR = "logs"
MODELS_DIR = "models"
POINTS_COUNT = 12  # 12 points (x, y) + 2 center points (x, y)

# Custom dataset class
class KeypointDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.annotations = sorted([f for f in os.listdir(annotation_dir) if f.endswith('.csv')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        original_width, original_height = image.size  # Get original dimensions

        annotation_path = os.path.join(self.annotation_dir, self.annotations[idx])
        keypoints = pd.read_csv(annotation_path, header=None).values.flatten()
        keypoints = [float(coord) for point in keypoints for coord in point.strip('"()').split(",")]

        if self.transform:
            image = self.transform(image)

            # After transform, get new dimensions
            new_width = image.shape[1]  # For Tensors, the width is the second dimension
            new_height = image.shape[2]  # For Tensors, the height is the third dimension
            scale_x = new_width / original_width
            scale_y = new_height / original_height
            
            # Scale the keypoints
            keypoints = [coord * scale_x if i % 2 == 0 else coord * scale_y for i, coord in enumerate(keypoints)]

        return image, torch.tensor(keypoints, dtype=torch.float32), (original_width, original_height)

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
        image, keypoints, original_size = self.original_dataset[idx]
        original_width, original_height = original_size

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

        return translated_image, torch.tensor(keypoints_translated, dtype=torch.float32), (original_width, original_height)

# Initialize model
def initialize_model(model_name):
    if model_name == "efficientnet":
        model = models.efficientnet_v2_m(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, POINTS_COUNT * 2)
        )
    elif model_name == "resnet":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, POINTS_COUNT * 2)
        )
    elif model_name == "vgg":
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, POINTS_COUNT * 2)
    else:
        raise ValueError("Model must be 'efficientnet', 'resnet', or 'vgg'.")
    
    return model

def calculate_nme(preds, targets, img_size):
    """
    Calculate the Normalized Mean Error (NME) for a single sample.

    Args:
        preds: Model predictions, shape (12, 2), representing the (x, y) coordinates of 12 keypoints.
        targets: Ground truth values, shape (12, 2).
        img_size: Original image size as a tuple (original_width, original_height).

    Returns:
        numpy.ndarray: The normalized distances for each keypoint.
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


def calculate_pixel_error(preds, targets, img_size):
    """
    Calculate the Pixel Error for a single sample.

    Args:
        preds: Model predictions, shape (8, 2), representing the (x, y) coordinates of 8 keypoints.
        targets: Ground truth values, shape (8, 2).
        img_size: Original image size as a tuple (original_width, original_height).

    Returns:
        numpy.ndarray: The pixel distances for each keypoint.
    """
    preds = preds.reshape(POINTS_COUNT, 2)  # Reshape to (12, 2)
    targets = targets.reshape(POINTS_COUNT, 2)  # Reshape to (12, 2)

    # Unpack the original image dimensions
    original_width, original_height = img_size

    # Calculate the scaling factors for width and height
    scale_x = original_width / IMAGE_SIZE  # Assuming IMAGE_SIZE is the model input size
    scale_y = original_height / IMAGE_SIZE

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


def extend_with_center_points(outputs, keypoints):
    # Reshape outputs and keypoints for easier indexing
    outputs = outputs.view(-1, POINTS_COUNT, 2)
    keypoints = keypoints.view(-1, POINTS_COUNT, 2)

    # Split outputs and keypoints into two groups (front and back)
    half_count = POINTS_COUNT // 2  
    outputs_front = outputs[:, :half_count, :]  # 前6點
    outputs_back = outputs[:, half_count:, :]  # 後6點
    
    keypoints_front = keypoints[:, :half_count, :]  # 前6點
    keypoints_back = keypoints[:, half_count:, :]  # 後6點

    # Calculate the center points for both groups
    center_output_front = torch.mean(outputs_front, dim=1)
    center_output_back = torch.mean(outputs_back, dim=1)

    center_keypoints_front = torch.mean(keypoints_front, dim=1)
    center_keypoints_back = torch.mean(keypoints_back, dim=1)

    # Combine the center points with the original outputs and keypoints
    outputs_extended = torch.cat([outputs, center_output_front.unsqueeze(1), center_output_back.unsqueeze(1)], dim=1)
    keypoints_extended = torch.cat([keypoints, center_keypoints_front.unsqueeze(1), center_keypoints_back.unsqueeze(1)], dim=1)
    
    return outputs_extended, keypoints_extended

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
    plt.title(f"Loss(MSE){title_suffix}")
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

def plot_kfold_results(epochs_range, all_train_losses, all_val_losses, 
                       all_train_nmes, all_val_nmes, 
                       all_train_pixel_errors, all_val_pixel_errors):

    metrics = {
        "MSE": (all_train_losses, all_val_losses),
        "NME": (all_train_nmes, all_val_nmes),
        "Pixel Error": (all_train_pixel_errors, all_val_pixel_errors)
    }

    for metric_name, (train_values, val_values) in metrics.items():
        plt.figure(figsize=(8, 5))

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        num_folds = len(train_values)

        for i in range(num_folds):
            color = color_cycle[i % len(color_cycle)]  # Get color for fold i
            plt.plot(epochs_range, train_values[i], linestyle="--", color=color, label=f"Train Fold {i+1}")
            plt.plot(epochs_range, val_values[i], linestyle="-",  color=color, label=f"Val Fold {i+1}")

        plt.title(f"K-Fold {metric_name}")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{LOGS_DIR}/kfold_{metric_name.replace(' ', '_')}.png")
        plt.show()

def kfold_train(data_dir, model_name, epochs, learning_rate, batch_size, k_folds=5):
    # Prepare transforms and dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.Lambda(lambda img: ImageOps.equalize(img)),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    full_dataset = KeypointDataset(
        img_dir=os.path.join(data_dir, 'images'), 
        annotation_dir=os.path.join(data_dir, 'annotations'), 
        transform=transform
    )
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Store training logs for plotting
    all_fold_train_losses = []
    all_fold_val_losses = []
    all_fold_train_nmes = []
    all_fold_val_nmes = []
    all_fold_train_pixel_errors = []
    all_fold_val_pixel_errors = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---")

        # Create train/val subsets
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        # Apply augmentations to training data
        aug1 = AugmentedKeypointDataset(train_subset, max_translate_x=20, max_translate_y=20)
        aug2 = AugmentedKeypointDataset(train_subset, max_angle=10)
        train_combined = ConcatDataset([train_subset, aug1, aug2])

        train_loader = DataLoader(
            train_combined, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, prefetch_factor=4
        )

        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True, prefetch_factor=4
        )

        model = initialize_model(model_name)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        fold_train_losses, fold_val_losses = [], []
        fold_train_nmes, fold_val_nmes = [], []
        fold_train_pixel_errors, fold_val_pixel_errors = [], []

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            train_loss, train_nme_list, train_pixel_list = 0, [], []

            for images, keypoints, original_sizes in train_loader:
                images, keypoints = images.to(device), keypoints.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                outputs_ext, keypoints_ext = extend_with_center_points(outputs, keypoints)
                loss = criterion(outputs_ext, keypoints_ext)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                preds = outputs.cpu().detach().numpy()
                targets = keypoints.cpu().numpy()
                widths, heights = original_sizes
                sizes = [(w, h) for w, h in zip(widths.numpy(), heights.numpy())]
                for i in range(len(sizes)):
                    train_nme_list.append(calculate_nme(preds[i], targets[i], sizes[i]))
                    train_pixel_list.append(calculate_pixel_error(preds[i], targets[i], sizes[i]))

            train_loss /= len(train_loader)
            fold_train_losses.append(train_loss)
            fold_train_nmes.append(np.mean(train_nme_list))
            fold_train_pixel_errors.append(np.mean(train_pixel_list))

            # ---- Validation
            model.eval()
            val_loss, val_nme_list, val_pixel_list = 0, [], []
            with torch.no_grad():
                for images, keypoints, original_sizes in val_loader:
                    images, keypoints = images.to(device), keypoints.to(device)
                    outputs = model(images)
                    outputs_ext, keypoints_ext = extend_with_center_points(outputs, keypoints)
                    loss = criterion(outputs_ext, keypoints_ext)
                    val_loss += loss.item()

                    preds = outputs.cpu().detach().numpy()
                    targets = keypoints.cpu().numpy()
                    widths, heights = original_sizes
                    sizes = [(w, h) for w, h in zip(widths.numpy(), heights.numpy())]
                    for i in range(len(sizes)):
                        val_nme_list.append(calculate_nme(preds[i], targets[i], sizes[i]))
                        val_pixel_list.append(calculate_pixel_error(preds[i], targets[i], sizes[i]))

            val_loss /= len(val_loader)
            fold_val_losses.append(val_loss)
            fold_val_nmes.append(np.mean(val_nme_list))
            fold_val_pixel_errors.append(np.mean(val_pixel_list))

            print(f"[Fold {fold+1}] Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

        # Save best model per fold
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(best_model_state, f"{MODELS_DIR}/fold_{fold+1}_best.pth")

        # Save results of this fold
        all_fold_train_losses.append(fold_train_losses)
        all_fold_val_losses.append(fold_val_losses)
        all_fold_train_nmes.append(fold_train_nmes)
        all_fold_val_nmes.append(fold_val_nmes)
        all_fold_train_pixel_errors.append(fold_train_pixel_errors)
        all_fold_val_pixel_errors.append(fold_val_pixel_errors)

    # Plot all fold curves
    epochs_range = range(1, epochs + 1)
    plot_kfold_results(
        epochs_range,
        all_fold_train_losses, all_fold_val_losses,
        all_fold_train_nmes, all_fold_val_nmes,
        all_fold_train_pixel_errors, all_fold_val_pixel_errors
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--k_folds", type=int, default=5)
    args = parser.parse_args()

    kfold_train(args.data_dir, args.model_name, args.epochs, args.learning_rate, args.batch_size, args.k_folds)

