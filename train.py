import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

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

def main(data_dir, model_name, epochs, learning_rate, batch_size):
    # Transform for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.Lambda(lambda img: ImageOps.equalize(img)),  # Apply histogram equalization
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Load the dataset and create DataLoaders
    train_dataset = KeypointDataset(img_dir=os.path.join(data_dir, 'train/images'), 
                                     annotation_dir=os.path.join(data_dir, 'train/annotations'), 
                                     transform=transform)
    val_dataset = KeypointDataset(img_dir=os.path.join(data_dir, 'val/images'), 
                                   annotation_dir=os.path.join(data_dir, 'val/annotations'), 
                                   transform=transform)
    augmented_dataset = AugmentedKeypointDataset(train_dataset, max_translate_x=20, max_translate_y=20)
    augmented_dataset2 = AugmentedKeypointDataset(train_dataset, max_angle=10)
    combined_dataset = ConcatDataset([train_dataset, augmented_dataset])
    combined_dataset = ConcatDataset([combined_dataset, augmented_dataset2])
    
    # To visualize the dataset
    display_image(train_dataset, 0)
    for i in range(0, 3):
        display_image(augmented_dataset, i)
        display_image(augmented_dataset2, i)
    
    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=4
    )

    print(f"Training samples: {len(combined_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize the model, loss function, and optimizer
    model = initialize_model(model_name)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Save the model's training progress
    epoch_losses = []
    epoch_accuracies = []
    epoch_nmes = []
    epoch_pixel_errors = []
    val_losses = []
    val_nmes = []
    val_pixel_errors = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')  # Track the best validation loss
    best_model_state = None  # Variable to store the best model

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        nme_values = []
        pixel_error_values = []

        # Training Loop
        for images, keypoints, original_sizes in train_loader:
            images, keypoints = images.to(device), keypoints.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Combine the center points with the original outputs and keypoints
            outputs_extended, keypoints_extended = extend_with_center_points(outputs, keypoints)

            # Calculate loss using the extended tensors
            loss = criterion(outputs_extended, keypoints_extended)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = outputs.cpu().detach().numpy()
            targets = keypoints.cpu().numpy()

            # Unpack the original image sizes
            widths, heights = original_sizes
            widths = widths.cpu().numpy()  
            heights = heights.cpu().numpy() 

            original_sizes = [(w, h) for w, h in zip(widths, heights)]

            # Calculate NME and Pixel Error for each sample
            for i in range(len(original_sizes)):
                img_size = original_sizes[i]

                nme = calculate_nme(preds[i], targets[i], img_size)
                pixel_error = calculate_pixel_error(preds[i], targets[i], img_size)

                nme_values.append(nme)
                pixel_error_values.append(pixel_error)

        epoch_loss = running_loss / len(train_loader)
        epoch_nme = np.mean(nme_values)
        epoch_pixel_error = np.mean(pixel_error_values)

        epoch_losses.append(epoch_loss)
        epoch_nmes.append(epoch_nme)
        epoch_pixel_errors.append(epoch_pixel_error)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, NME: {epoch_nme:.4f}, Pixel Error: {epoch_pixel_error:.4f}")
        
        # Validation Loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_nme_values = []
        val_pixel_error_values = []

        with torch.no_grad():  # No need to track gradients during validation
            for images, keypoints, original_sizes in val_loader:
                images, keypoints = images.to(device), keypoints.to(device)

                outputs = model(images)

                # Combine the center points with the original outputs and keypoints
                outputs_extended, keypoints_extended = extend_with_center_points(outputs, keypoints)

                # Calculate loss using the extended tensors
                loss = criterion(outputs_extended, keypoints_extended)
                val_loss += loss.item()

                preds = outputs.cpu().detach().numpy()
                targets = keypoints.cpu().numpy()

                # Unpack the original image sizes
                widths, heights = original_sizes
                widths = widths.cpu().numpy()  
                heights = heights.cpu().numpy()

                original_sizes = [(w, h) for w, h in zip(widths, heights)]

                # Calculate NME and Pixel Error for each sample
                for i in range(len(original_sizes)):
                    img_size = original_sizes[i]

                    nme = calculate_nme(preds[i], targets[i], img_size)
                    pixel_error = calculate_pixel_error(preds[i], targets[i], img_size)

                    val_nme_values.append(nme)
                    val_pixel_error_values.append(pixel_error)

        val_loss = val_loss / len(val_loader)
        val_nme = np.mean(val_nme_values)
        val_pixel_error = np.mean(val_pixel_error_values)

        val_losses.append(val_loss)
        val_nmes.append(val_nme)
        val_pixel_errors.append(val_pixel_error)

        print(f"Validation Loss: {val_loss:.4f}, NME: {val_nme:.4f}, Pixel Error: {val_pixel_error:.4f}")

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the model state at the best point
            print(f"Validation loss improved, saving model.")

    # Save the best model (with the lowest validation loss)
    if best_model_state:
        model_path = f"{MODELS_DIR}/{model_name}_keypoint_{epochs}_{learning_rate}_{batch_size}_best.pth"
        torch.save(best_model_state, model_path)
        print(f"Best model saved to: {model_path}")

    # Save the training and validation progress
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --------------------------------------------------------------plotting--------------------------------------------------------------
    # Save the training progress plot
    epochs_range = range(1, epochs + 1)
    plot_training_progress(
        epochs_range, epoch_losses, val_losses, epoch_nmes, val_nmes, epoch_pixel_errors, val_pixel_errors,
        loss_ylim=(0, 300),
        nme_ylim=(0, 0.02),
        pixel_error_ylim=(0, 200),
    )

    # Save the training plot
    training_plot_path = f"{LOGS_DIR}/{model_name}_training_plot_{epochs}_{learning_rate}_{batch_size}.png"
    plt.savefig(training_plot_path)
    print(f"Training plot saved to: {training_plot_path}")
    plt.show()

    # Save the Loss, NME, and Pixel Error to a text file
    training_log_path = f"{LOGS_DIR}/{model_name}_training_log_{epochs}_{learning_rate}_{batch_size}.txt"
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
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of samples per batch")
    args = parser.parse_args()

    main(args.data_dir, args.model_name, args.epochs, args.learning_rate, args.batch_size)
