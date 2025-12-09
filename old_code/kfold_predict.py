import os
import argparse
import torch
from torchvision import transforms, models
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import Subset

IMAGE_SIZE = 224
POINTS_COUNT = 12

def initialize_model(model_name):
    if model_name == "efficientnet":
        model = models.efficientnet_v2_m(pretrained=False)
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(model.classifier[1].in_features, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, POINTS_COUNT * 2)
        )
    elif model_name == "resnet":
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, POINTS_COUNT * 2)
        )
    elif model_name == "vgg":
        model = models.vgg19(pretrained=False)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 16)
    else:
        raise ValueError("Model must be 'efficientnet', 'resnet', or 'vgg'")
    return model

def load_annotations(annotation_path):
    keypoints = pd.read_csv(annotation_path, header=None).values.flatten()
    keypoints = [float(coord) for point in keypoints for coord in point.strip('"()').split(",")]
    return np.array(keypoints).reshape(-1, 2)

def calculate_avg_distance(predicted, original):
    distances = np.linalg.norm(predicted - original, axis=1)
    return np.mean(distances)

class KeypointDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transform):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_file = self.images[idx]
        img_path = os.path.join(self.image_dir, image_file)
        anno_path = os.path.join(self.annotation_dir, image_file.replace(".jpg", ".csv"))

        image = Image.open(img_path).convert("L")
        original_width, original_height = image.size
        image_tensor = self.transform(image)
        keypoints = load_annotations(anno_path)

        return image_tensor, keypoints, image_file, (original_width, original_height)

def kfold_predict(models_dir, model_name, val_data_dir, output_dir, k_folds=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare folders
    os.makedirs(output_dir, exist_ok=True)
    distance_groups = {
        "0-9": os.path.join(output_dir, "0-9"),
        "10-19": os.path.join(output_dir, "10-19"),
        "20-29": os.path.join(output_dir, "20-29"),
        "30-39": os.path.join(output_dir, "30-39"),
        "40+": os.path.join(output_dir, "40+")
    }
    for d in distance_groups.values():
        os.makedirs(d, exist_ok=True)

    filenames_per_group = {key: [] for key in distance_groups.keys()}

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Lambda(lambda img: ImageOps.equalize(img)),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    full_dataset = KeypointDataset(
        image_dir=os.path.join(val_data_dir, "images"),
        annotation_dir=os.path.join(val_data_dir, "annotations"),
        transform=transform
    )

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_avg_distances = []
    image_indices = []
    seen_images = set()

    for fold, (_, val_idx) in enumerate(kfold.split(full_dataset)):
        model_path = os.path.join(models_dir, f"fold_{fold + 1}_best.pth")
        model = initialize_model(model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        val_subset = Subset(full_dataset, val_idx)

        print(f"Fold {fold + 1} - Predicting {len(val_subset)} images...")
        
        for i, (image_tensor, original_keypoints, image_file, (original_width, original_height)) in enumerate(val_subset):
            if image_file in seen_images:
                continue  # Skip already predicted image (each val image should appear only once)
            seen_images.add(image_file)

            image_tensor = image_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(image_tensor).cpu().numpy().reshape(-1, 2)

            scale_x = original_width / IMAGE_SIZE
            scale_y = original_height / IMAGE_SIZE
            scaled_preds = preds * np.array([scale_x, scale_y])

            avg_distance = calculate_avg_distance(scaled_preds, original_keypoints)
            all_avg_distances.append(avg_distance)
            image_indices.append(len(image_indices) + 1)

            # 分群
            if avg_distance <= 9:
                group = "0-9"
            elif avg_distance <= 19:
                group = "10-19"
            elif avg_distance <= 29:
                group = "20-29"
            elif avg_distance <= 39:
                group = "30-39"
            else:
                group = "40+"

            filenames_per_group[group].append(image_file)

            # Save prediction image
            image_pil = Image.open(os.path.join(val_data_dir, 'images', image_file)).convert("L")
            plt.imshow(image_pil, cmap='gray')
            plt.scatter(scaled_preds[:, 0], scaled_preds[:, 1], c='yellow', label='Predicted', s=10)
            plt.scatter(original_keypoints[:, 0], original_keypoints[:, 1], c='red', label='Ground Truth', s=10)
            plt.title(f"{image_file} | AvgDist={avg_distance:.2f}")
            plt.axis('off')
            plt.legend()
            save_name = os.path.splitext(image_file)[0]
            plt.savefig(os.path.join(distance_groups[group], f"{save_name}_prediction.png"))
            plt.close()

            # Save predicted keypoints
            np.savetxt(os.path.join(distance_groups[group], f"{save_name}_keypoints.txt"), scaled_preds, fmt="%.2f", delimiter=",")

    # Save filename lists
    for group, filenames in filenames_per_group.items():
        txt_path = os.path.join(distance_groups[group], "filenames.txt")
        with open(txt_path, 'w') as f:
            for name in filenames:
                f.write(name + "\n")

    # Plot bar chart
    plt.figure(figsize=(14, 6))
    plt.bar(image_indices, all_avg_distances, color='blue')
    overall_mean = np.mean(all_avg_distances)
    plt.axhline(overall_mean, color='red', linestyle='--', label=f'Overall Avg: {overall_mean:.2f}')
    plt.xlabel("Image Index")
    plt.ylabel("Average Distance")
    plt.title("Average Distance per Test Image")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "avg_distances_chart.png"))
    plt.show()

    print(f"Overall average distance: {overall_mean:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True, help="Path to folder with fold_1_best.pth ...")
    parser.add_argument("--model_name", type=str, required=True, help="Model type: efficientnet, resnet, vgg")
    parser.add_argument("--val_data_dir", type=str, required=True, help="Validation folder with 'images' and 'annotations'")
    parser.add_argument("--output_dir", type=str, default="kfold_val_predict", help="Output dir")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds")
    args = parser.parse_args()

    kfold_predict(args.models_dir, args.model_name, args.val_data_dir, args.output_dir, args.k_folds)
