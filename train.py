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

IMAGE_SIZE = 224 # Image size for the model
LOGS_DIR = "logs"
MODELS_DIR = "models"

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
    def __init__(self, original_dataset, angle=0, translate_x=0, translate_y=0):
        self.original_dataset = original_dataset
        self.angle = angle
        self.translate_x = translate_x
        self.translate_y = translate_y

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, keypoints, original_size = self.original_dataset[idx]
        original_width, original_height = original_size

        # Apply rotation augmentation
        rotated_image = transforms.functional.rotate(image, self.angle)

        # Calculate rotation matrix for keypoints
        angle_rad = np.deg2rad(-self.angle)
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
            coord + self.translate_x if i % 2 == 0 else coord + self.translate_y
            for i, coord in enumerate(keypoints_rotated)
        ]

        # Apply translation to the image
        translated_image = transforms.functional.affine(
            rotated_image, angle=0, translate=(self.translate_x, self.translate_y), scale=1, shear=0
        )

        return translated_image, torch.tensor(keypoints_translated, dtype=torch.float32), (original_width, original_height)

# Initialize model
def initialize_model(model_name):
    if model_name == "efficientnet":
        model = models.efficientnet_v2_m(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 16)  # 8 points (x, y)
        )
    elif model_name == "resnet":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 16)
        )
    elif model_name == "vgg":
        model = models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 16)
    else:
        raise ValueError("Model must be 'efficientnet', 'resnet', or 'vgg'.")
    
    return model

def calculate_nme(preds, targets, img_size):
    """
    Calculate the Normalized Mean Error (NME) for a single sample.

    Args:
        preds: Model predictions, shape (8, 2), representing the (x, y) coordinates of 8 keypoints.
        targets: Ground truth values, shape (8, 2).
        img_size: Original image size as a tuple (original_width, original_height).

    Returns:
        numpy.ndarray: The normalized distances for each keypoint.
    """
    preds = preds.reshape(8, 2)  # Reshape to (8, 2)
    targets = targets.reshape(8, 2)  # Reshape to (8, 2)

    # Calculate the Euclidean distance for each keypoint
    pixel_distances = np.linalg.norm(preds - targets, axis=1)  # Shape: (8,)

    # Calculate the diagonal of the image
    img_diag = np.sqrt(img_size[0]**2 + img_size[1]**2)  # Scalar

    # Normalize distances by the image diagonal
    norm_distances = pixel_distances / img_diag  # Shape: (8,)

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
    preds = preds.reshape(8, 2)  # Reshape to (8, 2)
    targets = targets.reshape(8, 2)  # Reshape to (8, 2)

    # Unpack the original image dimensions
    original_width, original_height = img_size

    # Calculate the scaling factors for width and height
    scale_x = original_width / IMAGE_SIZE  # Assuming IMAGE_SIZE is the model input size
    scale_y = original_height / IMAGE_SIZE

    # Scale the predictions and targets
    preds_scaled = preds * np.array([scale_x, scale_y])  # Shape: (8, 2)
    targets_scaled = targets * np.array([scale_x, scale_y])  # Shape: (8, 2)

    # Calculate the Euclidean distance for each keypoint
    pixel_distances = np.linalg.norm(preds_scaled - targets_scaled, axis=1)  # Shape: (8,)

    # Return the mean of the pixel distances
    return np.mean(pixel_distances)

def display_image(dataset):
    # Get the first image and keypoints
    image, keypoints, original_size = dataset[0]
    print("original_size:", original_size)
    print("image shape:", image.shape)
    
    # Convert the image to a NumPy array
    image_np = image.permute(1, 2, 0).numpy()

    plt.imshow(image_np, cmap='gray')
    plt.title("Image with Keypoints")
    
    # Plot the keypoints with numbering
    for i in range(0, len(keypoints), 2):
        x = keypoints[i].item()  
        y = keypoints[i + 1].item()  
        plt.scatter(x, y, c='red', s=20)
        plt.text(x, y, f'{i//2 + 1}', color='yellow', fontsize=12)  # Add number next to each point
    
    plt.show()

def extend_with_center_points(outputs, keypoints):
    # Reshape outputs and keypoints for easier indexing
    outputs = outputs.view(-1, 8, 2)
    keypoints = keypoints.view(-1, 8, 2)

    # Split outputs and keypoints into two groups (front and back)
    outputs_front = outputs[:, :4, :]
    outputs_back = outputs[:, 4:, :]
    
    keypoints_front = keypoints[:, :4, :]
    keypoints_back = keypoints[:, 4:, :]

    # Calculate the center points for both groups
    center_output_front = torch.mean(outputs_front, dim=1)
    center_output_back = torch.mean(outputs_back, dim=1)

    center_keypoints_front = torch.mean(keypoints_front, dim=1)
    center_keypoints_back = torch.mean(keypoints_back, dim=1)

    # Combine the center points with the original outputs and keypoints
    outputs_extended = torch.cat([outputs, center_output_front.unsqueeze(1), center_output_back.unsqueeze(1)], dim=1)
    keypoints_extended = torch.cat([keypoints, center_keypoints_front.unsqueeze(1), center_keypoints_back.unsqueeze(1)], dim=1)

    return outputs_extended, keypoints_extended

def main(data_dir, model_name, epochs, learning_rate, batch_size):
    # Transform for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),
    ])

    # Load the dataset and create DataLoaders
    train_dataset = KeypointDataset(img_dir=os.path.join(data_dir, 'train/images'), 
                                     annotation_dir=os.path.join(data_dir, 'train/annotations'), 
                                     transform=transform)
    augmented_dataset = AugmentedKeypointDataset(train_dataset, translate_x=20, translate_y=20)
    augmented_dataset_2 = AugmentedKeypointDataset(train_dataset, angle=10)
    # val_dataset = KeypointDataset(img_dir=os.path.join(data_dir, 'val/images'), 
    #                                annotation_dir=os.path.join(data_dir, 'val/annotations'), 
    #                                transform=transform)
    
    # To visualize the dataset
    display_image(train_dataset)
    display_image(augmented_dataset)
    
    # Combine the original and augmented datasets
    combined_train_dataset = ConcatDataset([train_dataset, augmented_dataset])
    combined_train_dataset = ConcatDataset([combined_train_dataset, augmented_dataset_2])
    print(f"Combined Train Dataset: {len(combined_train_dataset)} samples")
    
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = initialize_model(model_name)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Save the model's training progress
    epoch_losses = []
    epoch_accuracies = []
    epoch_nmes = []
    epoch_pixel_errors = []
    val_nmes = []
    val_pixel_errors = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        
        # if epoch == 200:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1  # Reduce learning rate by 10x
        
        model.train()
        running_loss = 0.0
        nme_values = []
        pixel_error_values = []

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
            widths, heights = original_sizes  # original_sizes is a tuple of two tensors
            widths = widths.cpu().numpy()  
            heights = heights.cpu().numpy() 

            # Convert the original sizes to a list of tuples
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

    # Ensure directories exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Plot and save the training progress
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 6))

    # Plot Loss with log scale
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, epoch_losses, label="Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss(log)")
    plt.yscale('log')  # Log scale
    plt.legend()

    # Plot NME
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, epoch_nmes, label="NME")
    plt.title("Training NME")
    plt.xlabel("Epoch")
    plt.ylabel("NME(log)")
    plt.yscale('log')  # Log scale
    plt.legend()

    # Plot Pixel Error
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, epoch_pixel_errors, label="Pixel Error")
    plt.title("Training Pixel Error")
    plt.xlabel("Epoch")
    plt.ylabel("Pixel Error(log)")
    plt.yscale('log')  # Log scale
    plt.legend()

    plt.tight_layout()

    # Save the training plot
    training_plot_path = f"{LOGS_DIR}/{model_name}_training_plot_{epochs}_{learning_rate}_{batch_size}.png"
    plt.savefig(training_plot_path)
    print(f"Training plot saved to: {training_plot_path}")
    plt.show()

    # Save the Loss, NME, and Pixel Error to a text file
    training_log_path = f"{LOGS_DIR}/{model_name}_training_log_{epochs}_{learning_rate}_{batch_size}.txt"
    with open(training_log_path, "w") as f:
        for epoch, (loss, nme, pixel_error) in enumerate(zip(epoch_losses, epoch_nmes, epoch_pixel_errors), 1):
            f.write(f"Epoch {epoch}: Loss = {loss:.4f}, NME = {nme:.4f}, Pixel Error = {pixel_error:.4f}\n")
    print(f"Training log saved to: {training_log_path}")

    # Save the model
    model_path = f"{MODELS_DIR}/{model_name}_keypoint_{epochs}_{learning_rate}_{batch_size}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"{model_name} model trained and saved successfully to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name: 'efficientnet', 'resnet', or 'vgg'")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of samples per batch")
    args = parser.parse_args()

    main(args.data_dir, args.model_name, args.epochs, args.learning_rate, args.batch_size)
