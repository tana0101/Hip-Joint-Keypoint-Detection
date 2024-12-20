import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, transforms
from torch import nn
import pandas as pd
import re

IMAGE_SIZE = 224 # Image size for the model
POINTS_COUNT = 12

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
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 16)
    else:
        raise ValueError("'efficientnet', 'resnet', or 'vgg'.")
    
    return model

def load_annotations(annotation_path):
    keypoints = pd.read_csv(annotation_path, header=None).values.flatten()
    keypoints = [float(coord) for point in keypoints for coord in point.strip('"()').split(",")]
    return np.array(keypoints).reshape(-1, 2)

# Calculate average distance between predicted and original keypoints
def calculate_avg_distance(predicted_keypoints, original_keypoints):
    distances = np.linalg.norm(predicted_keypoints - original_keypoints, axis=1)
    avg_distance = np.mean(distances)
    return avg_distance

def extract_info_from_model_path(model_path):
    # Extract epochs, learning_rate, and batch_size from the model_path
    match = re.search(r'_(\d+)_(\d+\.\d+)_(\d+)_best\.pth', model_path)
    if match:
        epochs = int(match.group(1))
        learning_rate = float(match.group(2))
        batch_size = int(match.group(3))
        return epochs, learning_rate, batch_size
    else:
        raise ValueError("Model path format is invalid. Expected format: model_name_epochs_lr_batchsize.pth")

def predict(model_name, model_path, data_dir, output_dir):
    # Extract training information from model path
    epochs, learning_rate, batch_size = extract_info_from_model_path(model_path)
    
    model = initialize_model(model_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create main output directory
    result_dir = os.path.join(output_dir, f"{model_name}_{epochs}_{learning_rate}_{batch_size}")
    os.makedirs(result_dir, exist_ok=True)

    # Create subdirectories for distance ranges
    distance_ranges = {
        "0-30": os.path.join(result_dir, "0-30"),
        "31-60": os.path.join(result_dir, "31-60"),
        "61-90": os.path.join(result_dir, "61-90"),
        "91+": os.path.join(result_dir, "91+"),
    }
    for path in distance_ranges.values():
        os.makedirs(path, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),
    ])

    all_avg_distances = []  # To store all distances
    image_labels = []  # To store image indices (1, 2, 3, ...)
    image_counter = 1  # To keep track of the image index

    for image_file in os.listdir(os.path.join(data_dir, 'images')):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(data_dir, 'images', image_file)
            image = Image.open(image_path).convert("L")  
            original_width, original_height = image.size  # Get original dimensions
            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                keypoints = model(image_tensor).cpu().numpy().reshape(-1, 2)

            # Load original keypoints
            annotation_path = os.path.join(data_dir, 'annotations', f"{os.path.splitext(image_file)[0]}.csv")
            original_keypoints = load_annotations(annotation_path)

            # Scale keypoints based on original size
            scale_x = original_width / IMAGE_SIZE
            scale_y = original_height / IMAGE_SIZE
            scaled_keypoints = keypoints * np.array([scale_x, scale_y])

            # Calculate avg distance
            avg_distance = calculate_avg_distance(scaled_keypoints, original_keypoints)
            all_avg_distances.append(avg_distance)
            image_labels.append(image_counter)

            # Determine the subdirectory based on avg_distance
            if avg_distance <= 30:
                subfolder = distance_ranges["0-30"]
            elif avg_distance <= 60:
                subfolder = distance_ranges["31-60"]
            elif avg_distance <= 90:
                subfolder = distance_ranges["61-90"]
            else:
                subfolder = distance_ranges["91+"]

            # Save image with predicted keypoints
            plt.imshow(image, cmap='gray')
            plt.scatter(scaled_keypoints[:, 0], scaled_keypoints[:, 1], c='yellow', marker='o', label='Predicted Keypoints', s=10)
            plt.scatter(original_keypoints[:, 0], original_keypoints[:, 1], c='red', marker='o', label='Original Keypoints', s=10)
            plt.text(10, original_height - 10, f'Avg Distance: {avg_distance:.2f}', color='red', fontsize=12, ha='left', va='center')

            plt.title(f'Predicted Keypoints for {image_file}')
            plt.axis('off')
            plt.legend()
            plt.savefig(os.path.join(subfolder, f"{os.path.splitext(image_file)[0]}_prediction.png"))
            plt.close()

            # Save predicted keypoints
            np.savetxt(os.path.join(subfolder, f"{os.path.splitext(image_file)[0]}_keypoints.txt"), scaled_keypoints, fmt="%.2f", delimiter=",")

            image_counter += 1  # Increment the image index

    # Plot bar chart for avg distances
    plt.figure(figsize=(12, 6))
    plt.bar(image_labels, all_avg_distances, color='blue', label='Avg Distance per Image')

    # Overall average distance
    overall_avg_distance = np.mean(all_avg_distances)
    plt.axhline(overall_avg_distance, color='red', linestyle='--', label='Overall Avg Distance')

    # Add text for the overall average distance
    plt.text(len(image_labels) - 1, overall_avg_distance + 0.01, f'Avg: {overall_avg_distance:.2f}', 
             color='red', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    plt.xlabel('Image Index')
    plt.ylabel('Avg Distance')
    plt.title('Average Distance per Image')
    plt.xticks(image_labels)  # Set x-ticks to image indices (1, 2, 3, ...)

    plt.legend()

    # Save avg distances chart
    avg_distances_path = os.path.join(result_dir, f"{model_name}_avg_distances.png")
    plt.savefig(avg_distances_path)
    plt.show()
    
    print(f"Overall average distance: {overall_avg_distance:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name: 'efficientnet', 'resnet', or 'vgg'")
    parser.add_argument("--model_path", type=str, required=True, help="path to the trained model")
    parser.add_argument("--data", type=str, required=True, help="data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="output directory for predictions")
    args = parser.parse_args()

    predict(args.model_name, args.model_path, args.data, args.output_dir)
