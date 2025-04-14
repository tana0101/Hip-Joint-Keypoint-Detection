import os
import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageOps
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

def draw_hilgenreiner_line(ax, p3, p7):
    if np.allclose(p3[0], p7[0]):
        ax.axvline(x=p3[0], color='cyan', linewidth=1, label='Hilgenreiner Line')
    else:
        a = (p7[1] - p3[1]) / (p7[0] - p3[0])
        b = p3[1] - a * p3[0]
        x_min, x_max = ax.get_xlim()
        x_vals = np.array([x_min, x_max])
        y_vals = a * x_vals + b
        ax.plot(x_vals, y_vals, color='cyan', linewidth=1, label='Hilgenreiner Line')

def calculate_acetabular_index_angles(points):
    p1 = points[0]
    p3 = points[2]
    p7 = points[6]
    p9 = points[8]
    v_h = p7 - p3
    v_left = p3 - p1
    v_right = p7 - p9
    def angle(v1, v2):
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return angle(v_h, v_left), angle(-v_h, v_right)

def draw_comparison_figure(image, pred_kpts, gt_kpts, ai_pred, ai_gt, avg_distance, save_path, image_file):
    """
    建立左右對照圖：左圖使用預測點畫線，右圖使用 ground truth 畫線

    image: 原圖 (PIL or np.ndarray)
    pred_kpts: 預測的 scaled keypoints
    gt_kpts: ground truth keypoints
    ai_pred: (left, right) 使用預測點計算出來的 AI angle
    ai_gt: (left, right) 使用 ground truth 計算出來的 AI angle
    avg_distance: 平均距離
    save_path: 要儲存的路徑
    image_file: 圖片名稱（用來命名）
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for i, (kpts, title, ai) in enumerate([
        (pred_kpts, "Using Predicted Keypoints", ai_pred),
        (gt_kpts, "Using Ground Truth Keypoints", ai_gt)
    ]):
        ax = axes[i]
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        ax.scatter(pred_kpts[:, 0], pred_kpts[:, 1], c='yellow', s=4, label='Predicted')
        ax.scatter(gt_kpts[:, 0], gt_kpts[:, 1], c='red', s=4, label='Ground Truth')
        
        # # 畫關鍵點編號
        # for j, (x, y) in enumerate(kpts):
        #     ax.text(x + 2, y - 2, str(j + 1), color='white', fontsize=8, weight='bold')

        p1 = kpts[0]
        p3 = kpts[2]
        p7 = kpts[6]
        p9 = kpts[8]

        draw_hilgenreiner_line(ax, p3, p7)
        ax.plot([p7[0], p9[0]], [p7[1], p9[1]], color='magenta', linewidth=1, label='Left Roof Line')
        ax.plot([p3[0], p1[0]], [p3[1], p1[1]], color='magenta', linewidth=1, label='Right Roof Line')
        
        ax.text(10, image.size[1] + 35, f'AI Left: {ai[0]:.1f}°', color='magenta', fontsize=11)
        ax.text(10, image.size[1] + 85, f'AI Right: {ai[1]:.1f}°', color='magenta', fontsize=11)

    # 計算角度誤差
    ai_error_left = abs(ai_pred[0] - ai_gt[0])
    ai_error_right = abs(ai_pred[1] - ai_gt[1])

    # 百分比誤差（避免除以 0）
    def format_error(pred, gt):
        abs_error = abs(pred - gt)
        if gt != 0:
            rel_error = abs_error / gt * 100
            return f"{abs_error:.1f}° ({rel_error:.1f}%)"
        else:
            return f"{abs_error:.1f}° (--%)"

    # 計算圖片對角線長度（像素）
    diag_len = (image.size[0] ** 2 + image.size[1] ** 2) ** 0.5
    avg_dist_percent = avg_distance / diag_len * 100

    # 輸出資訊到圖下方
    fig.text(0.5, -0.05,
             f"Avg Distance: {avg_distance:.2f} px ({avg_dist_percent:.2f}% of image diagonal)    "
             f"AI Error (L/R): {format_error(ai_pred[0], ai_gt[0])} / {format_error(ai_pred[1], ai_gt[1])}",
             ha='center', fontsize=12, color='blue')

    axes[0].legend(loc='lower left')
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, f"{os.path.splitext(image_file)[0]}_compare.png"), bbox_inches='tight')
    plt.close()

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
        "0-7.5": os.path.join(result_dir, "0-7.5"),
        "7.5-10": os.path.join(result_dir, "7.5-10"),
        "10-12.5": os.path.join(result_dir, "10-12.5"),
        "12.5-15": os.path.join(result_dir, "12.5-15"),
        "15-17.5": os.path.join(result_dir, "15-17.5"),
        "17.5-20": os.path.join(result_dir, "17.5-20"),
        "20-30": os.path.join(result_dir, "20-30"),
        "30-40": os.path.join(result_dir, "30-40"),
        "40+": os.path.join(result_dir, "40+"),
    }
    
    for path in distance_ranges.values():
        os.makedirs(path, exist_ok=True)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.Lambda(lambda img: ImageOps.equalize(img)),  # Apply histogram equalization
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    all_avg_distances = []  # To store all distances
    image_labels = []  # To store image indices (1, 2, 3, ...)
    image_counter = 1  # To keep track of the image index
    ai_errors_left = [] # To store AI angle errors (left)
    ai_errors_right = [] # To store AI angle errors (right)


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
            
            # Calculate AI angle（預測）
            ai_left_pred, ai_right_pred = calculate_acetabular_index_angles(scaled_keypoints)

            # Calculate AI angle（ground truth）
            ai_left_gt, ai_right_gt = calculate_acetabular_index_angles(original_keypoints)
            
            ai_errors_left.append(abs(ai_left_pred - ai_left_gt))
            ai_errors_right.append(abs(ai_right_pred - ai_right_gt))
            
            # Determine the subdirectory based on avg_distance
            if avg_distance <= 7.5:
                subfolder = distance_ranges["0-7.5"]
            elif avg_distance <= 10:
                subfolder = distance_ranges["7.5-10"]
            elif avg_distance <= 12.5:
                subfolder = distance_ranges["10-12.5"]
            elif avg_distance <= 15:
                subfolder = distance_ranges["12.5-15"]
            elif avg_distance <= 17.5:
                subfolder = distance_ranges["15-17.5"]
            elif avg_distance <= 20:
                subfolder = distance_ranges["17.5-20"]
            elif avg_distance <= 30:
                subfolder = distance_ranges["20-30"]
            elif avg_distance <= 40:
                subfolder = distance_ranges["30-40"]
            else:
                subfolder = distance_ranges["40+"]

            # 畫出左右對照圖
            draw_comparison_figure(
                image=image,
                pred_kpts=scaled_keypoints,
                gt_kpts=original_keypoints,
                ai_pred=(ai_left_pred, ai_right_pred),
                ai_gt=(ai_left_gt, ai_right_gt),
                avg_distance=avg_distance,
                save_path=subfolder,
                image_file=image_file
            )

            # Save predicted keypoints
            # np.savetxt(os.path.join(subfolder, f"{os.path.splitext(image_file)[0]}_keypoints.txt"), scaled_keypoints, fmt="%.2f", delimiter=",")

            image_counter += 1  # Increment the image index

    # -------------------------------------------------------------- Plotting the average distances and AI angle errors --------------------------------------------------------------
    # Plot bar chart for avg distances
    plt.figure(figsize=(16, 6))
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
    
    # Bar chart for AI angle errors
    plt.figure(figsize=(16, 6))
    indices = np.arange(len(image_labels))

    bar_width = 0.4
    plt.bar(indices - bar_width/2, ai_errors_left, width=bar_width, label='Left AI Error', color='magenta')
    plt.bar(indices + bar_width/2, ai_errors_right, width=bar_width, label='Right AI Error', color='crimson')

    # 平均誤差
    avg_error_left = np.mean(ai_errors_left)
    avg_error_right = np.mean(ai_errors_right)
    avg_error = np.mean([avg_error_left, avg_error_right])

    plt.axhline(avg_error_left, color='magenta', linestyle='--', label=f'Avg Left Error: {avg_error_left:.2f}°')
    plt.axhline(avg_error_right, color='crimson', linestyle='--', label=f'Avg Right Error: {avg_error_right:.2f}°')
    plt.axhline(avg_error, color='blue', linestyle='--', label=f'Overall Avg Error: {avg_error:.2f}°')

    plt.xlabel('Image Index')
    plt.ylabel('AI Angle Error (°)')
    plt.title('Acetabular Index Angle Errors (Predicted vs. Ground Truth)')
    plt.xticks(indices, image_labels)
    plt.legend()

    # 儲存圖表
    ai_error_chart_path = os.path.join(result_dir, f"{model_name}_AI_angle_errors.png")
    plt.savefig(ai_error_chart_path)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name: 'efficientnet', 'resnet', or 'vgg'")
    parser.add_argument("--model_path", type=str, required=True, help="path to the trained model")
    parser.add_argument("--data", type=str, required=True, help="data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="output directory for predictions")
    args = parser.parse_args()

    predict(args.model_name, args.model_path, args.data, args.output_dir)
