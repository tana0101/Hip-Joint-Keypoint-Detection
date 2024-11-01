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

        return image, torch.tensor(keypoints, dtype=torch.float32)

class AugmentedKeypointDataset(Dataset):
    def __init__(self, original_dataset, angle, translate_x=0, translate_y=0):
        self.original_dataset = original_dataset
        self.angle = angle
        self.translate_x = translate_x
        self.translate_y = translate_y

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, keypoints = self.original_dataset[idx]

        # Apply rotation augmentation
        rotated_image = transforms.functional.rotate(image, self.angle)

        # Calculate rotation matrix for keypoints
        angle_rad = np.deg2rad(-self.angle)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        # Center of rotation (image center)
        original_width, original_height = image.shape[1], image.shape[2]
        center_x, center_y = original_width / 2, original_height / 2

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

        return translated_image, torch.tensor(keypoints_translated, dtype=torch.float32)

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
    
    preds = preds.reshape(-1, 8, 2)
    targets = targets.reshape(-1, 8, 2)

    pixel_distances = np.linalg.norm(preds - targets, axis=2)
    img_diag = np.sqrt(img_size[0]**2 + img_size[1]**2) # image diagonal
    norm_distances = pixel_distances / img_diag  # normalize by image diagonal
    nme_per_sample = np.mean(norm_distances, axis=1)  # mean NME for each sample

    return np.mean(nme_per_sample)  # mean NME for all samples

def calculate_pixel_error(preds, targets, img_size):
    
    preds = preds.reshape(-1, 8, 2)  # Reshape predictions to (batch_size, 8, 2)
    targets = targets.reshape(-1, 8, 2)  # Reshape targets to (batch_size, 8, 2)

    original_width, original_height = img_size
    scale_x = original_width / IMAGE_SIZE
    scale_y = original_height / IMAGE_SIZE

    preds_scaled = preds * np.array([scale_x, scale_y]) 
    targets_scaled = targets * np.array([scale_x, scale_y]) 

    pixel_distances = np.linalg.norm(preds_scaled - targets_scaled, axis=2)  
    pixel_error_per_sample = np.mean(pixel_distances, axis=1)  
    return np.mean(pixel_error_per_sample)  

def display_image(dataset):
    # Get the first image and keypoints
    image, keypoints = dataset[0]
    
    # Convert the image to a NumPy array
    image_np = image.permute(1, 2, 0).numpy()

    plt.imshow(image_np, cmap='gray')
    plt.title("Image with Keypoints")
    
    # Plot the keypoints
    for i in range(0, len(keypoints), 2):
        x = keypoints[i].item()  
        y = keypoints[i + 1].item()  
        plt.scatter(x, y, c='red', s=20) 
    
    plt.show()

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
    augmented_dataset = AugmentedKeypointDataset(train_dataset, angle=45)
    # val_dataset = KeypointDataset(img_dir=os.path.join(data_dir, 'val/images'), 
    #                                annotation_dir=os.path.join(data_dir, 'val/annotations'), 
    #                                transform=transform)
    
    # To visualize the dataset
    display_image(train_dataset)
    display_image(augmented_dataset)
    
    # Combine the original and augmented datasets
    combined_train_dataset = ConcatDataset([train_dataset, augmented_dataset])
    
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

        for images, keypoints in train_loader:
            images, keypoints = images.to(device), keypoints.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = outputs.cpu().detach().numpy()
            targets = keypoints.cpu().numpy()

            img_size = (images.shape[2], images.shape[3])

            # NME
            nme = calculate_nme(preds, targets, img_size)
            nme_values.append(nme)

            # Pixel mean distance error
            pixel_error = calculate_pixel_error(preds, targets, img_size)
            pixel_error_values.append(pixel_error)

        epoch_loss = running_loss / len(train_loader)
        epoch_nme = np.mean(nme_values)
        epoch_pixel_error = np.mean(pixel_error_values)

        epoch_losses.append(epoch_loss)
        epoch_nmes.append(epoch_nme)
        epoch_pixel_errors.append(epoch_pixel_error)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, NME: {epoch_nme:.4f}, Pixel Error: {epoch_pixel_error:.4f}")

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
    plt.savefig(f"logs/{model_name}_training_plot_{epochs}_{learning_rate}_{batch_size}.png")
    plt.show()

    # Save the Loss, NME, and Pixel Error to a text file
    with open(f"logs/{model_name}_training_log.txt_{epochs}_{learning_rate}_{batch_size}.txt", "w") as f:
        for epoch, (loss, nme, pixel_error) in enumerate(zip(epoch_losses, epoch_nmes, epoch_pixel_errors), 1):
            f.write(f"Epoch {epoch}: Loss = {loss:.4f}, NME = {nme:.4f}, Pixel Error = {pixel_error:.4f}\n")

    # Save the model
    torch.save(model.state_dict(), f'models/{model_name}_keypoint_{epochs}_{learning_rate}_{batch_size}.pth')
    print(f"{model_name} model trained and saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--model_name", type=str, required=True, help="Model name: 'efficientnet', 'resnet', or 'vgg'")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of samples per batch")
    args = parser.parse_args()

    main(args.data_dir, args.model_name, args.epochs, args.learning_rate, args.batch_size)
