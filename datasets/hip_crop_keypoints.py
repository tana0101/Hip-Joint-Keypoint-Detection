import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from PIL import Image
from tqdm import tqdm

from .augment import AugmentedKeypointDataset, ProbAugmentedKeypointDataset

DATASET_CONFIGS_BY_COUNT = {
    12: {
        "name": "IHDI_12pt",
        # IHDI 的左右定義鏡像後需要交叉重排: [0,1,2] -> [2,1,0]
        "mirror_reorder": [2, 1, 0, 5, 4, 3] 
    },
    8: {
        "name": "MTDDH_8pt",
        # MTDDH 假設是對稱定義 (1外/2內)，鏡像後通常順序不變，或根據實際情況調整
        "mirror_reorder": [0, 1, 2, 3] 
    }
}

SIDE_LABELS = {"left": "LeftHip", "right": "RightHip"}

def parse_csv_get_points(csv_path):
    """
    讀取 CSV 並回傳 numpy array，不預設點數，由資料決定
    """
    row = pd.read_csv(csv_path, header=None).values.flatten()
    pts = []
    for token in row:
        token = str(token).strip().strip('"').strip("'").strip()
        token = token.replace("(", "").replace(")", "")
        if not token: continue # 跳過空值
        x_str, y_str = token.split(",")
        pts.append([float(x_str), float(y_str)])
    return np.array(pts, dtype=np.float32)

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

def display_image(dataset, index, save_path=None):
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

    if save_path:
        plt.savefig(save_path)
        print(f"Saved image with keypoints to {save_path}")
    
    plt.show()
    plt.close()

def save_all_visualizations(dataset, output_dir="output_debug_crops", tick_spacing=10):
    """
    tick_spacing: 設定每隔多少 pixel 畫一條線 (數值越小越密集)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Start processing {len(dataset)} images...")

    for i in tqdm(range(len(dataset))):
        image_tensor, keypoints, original_crop_size, img_name = dataset[i]
        
        # 轉 numpy
        image_np = image_tensor.permute(1, 2, 0).numpy()
        
        # 建立畫布
        fig, ax = plt.subplots(figsize=(6, 6)) # 稍微大一點比較好看清楚
        ax.imshow(image_np, cmap='gray')
        
        title_str = f"{img_name}\nOrg: {int(original_crop_size[0])}x{int(original_crop_size[1])}"
        ax.set_title(title_str, fontsize=10)

        ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))
        
        ax.grid(True, which='both', color='cyan', linestyle='--', linewidth=0.5, alpha=0.8)

        # 移除刻度標籤
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        
        # ---------------------------------------------------------
        # 畫關鍵點
        kps = keypoints.numpy()
        for k in range(0, len(kps), 2):
            x = kps[k]
            y = kps[k+1]
            ax.scatter(x, y, c='red', s=40, marker='.', zorder=5) # zorder=5 確保點在網格之上
            ax.text(x + 2, y + 2, f'{k//2 + 1}', color='yellow', fontsize=12, weight='bold', zorder=5)

        # 存檔
        save_path = os.path.join(output_dir, img_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=100) # dpi 可以調高讓文字更清晰
        plt.close(fig)

    print(f"\nDone! All images saved to: {output_dir}")

# Custom dataset class
class HipCropKeypointDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, detections_dir, side="left",
                 transform=None, crop_expand=0.10, keep_square=True, input_size=224,
                 bbox_jitter=False, jitter_center=0.05, jitter_scale=0.10, jitter_prob=1.0):
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
        self.crop_expand = crop_expand
        self.keep_square = keep_square
        self.input_size = input_size
        self.side_label = SIDE_LABELS[self.side]
        self.bbox_jitter = bbox_jitter
        self.jitter_center = jitter_center
        self.jitter_scale = jitter_scale
        self.jitter_prob = jitter_prob

        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
        self.annotations = sorted([f for f in os.listdir(annotation_dir) if f.endswith(".csv")])
        print(f"Found {len(self.images)} images and {len(self.annotations)} annotations.")
        assert len(self.images) == len(self.annotations), "Images/annotations count mismatch"

        # ==========================================
        # 自動偵測邏輯
        # ==========================================
        if len(self.annotations) > 0:
            # 讀取第一個標註檔來偵測點數
            sample_path = os.path.join(self.annotation_dir, self.annotations[0])
            sample_pts = parse_csv_get_points(sample_path)
            self.total_points = sample_pts.shape[0]
            
            # 檢查是否有對應的 Config
            if self.total_points not in DATASET_CONFIGS_BY_COUNT:
                raise ValueError(f"Detected {self.total_points} points, but no config found in DATASET_CONFIGS_BY_COUNT.")
            
            self.config = DATASET_CONFIGS_BY_COUNT[self.total_points]
            print(f"[{side.upper()}] Detected Dataset: {self.config['name']} ({self.total_points} points total)")
        else:
            raise ValueError("Annotation directory is empty, cannot detect dataset format.")
        
        # ==========================================
        # 自動計算切分 (Symmetry Logic)
        # ==========================================
        self.points_per_side = self.total_points // 2
        
        # 定義左右側的 Slice 範圍
        # Left: 前半段 [0 : mid], Right: 後半段 [mid : total]
        if self.side == "left":
            self.slice_idx = (0, self.points_per_side)
        else:
            self.slice_idx = (self.points_per_side, self.total_points)
        
    def __len__(self):
        return len(self.images)

    # 讓外部程式 (如 train.py) 可以知道這份資料集單側有幾個點
    @property
    def num_keypoints(self):
        return self.points_per_side
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, self.annotations[idx])

        img = Image.open(img_path).convert("L")  # Grayscale PIL
        W, H = img.size

        # 1) 讀這張圖的 bbox（每張一檔）
        dets = read_detection_for_image(self.detections_dir, img_name)
        x1, y1, x2, y2 = dets[self.side_label]

        # 可能高度=0（兩點同 y），修成接近正方
        if y2 - y1 < 2:
            width = max(2.0, x2 - x1)
            # 讓 bbox 高度 = 寬度，往下延伸（如果超界再往上調整）
            y2 = min(H, y1 + width)
            if y2 - y1 < width:  # 超界了往上撐
                y1 = max(0.0, y2 - width)

        # ---- bbox jitter: train only ----
        if self.bbox_jitter and np.random.rand() < self.jitter_prob:
            bw = max(2.0, x2 - x1)
            bh = max(2.0, y2 - y1)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # center jitter (relative to bbox size)
            dx = np.random.uniform(-self.jitter_center, self.jitter_center) * bw
            dy = np.random.uniform(-self.jitter_center, self.jitter_center) * bh

            # size jitter
            sw = np.random.uniform(1 - self.jitter_scale, 1 + self.jitter_scale)
            sh = np.random.uniform(1 - self.jitter_scale, 1 + self.jitter_scale)

            bw2 = bw * sw
            bh2 = bh * sh

            cx2 = cx + dx
            cy2 = cy + dy

            x1 = cx2 - bw2 / 2.0
            x2 = cx2 + bw2 / 2.0
            y1 = cy2 - bh2 / 2.0
            y2 = cy2 + bh2 / 2.0

            # clamp to image bounds early
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            x2 = min(float(W), x2); y2 = min(float(H), y2)
        
         # ---- keep_square ----
        bw, bh = (x2 - x1), (y2 - y1)
        if self.keep_square:
            bw = max(2.0, x2 - x1)
            bh = max(2.0, y2 - y1)
            side_len = max(bw, bh)

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            x1, x2 = cx - side_len / 2.0, cx + side_len / 2.0
            y1, y2 = cy - side_len / 2.0, cy + side_len / 2.0

        # ---- crop_expand (use CURRENT bbox size) ----
        bw = max(2.0, x2 - x1)
        bh = max(2.0, y2 - y1)

        pad = self.crop_expand
        x1 -= bw * pad; x2 += bw * pad
        y1 -= bh * pad; y2 += bh * pad

        x1 = max(0.0, x1); y1 = max(0.0, y1)
        x2 = min(float(W), x2); y2 = min(float(H), y2)

        # 2) 讀取點並切分
        pts_all = parse_csv_get_points(ann_path) # (Total, 2)
        
        # 利用自動計算的 slice_idx 取出該側的點
        start, end = self.slice_idx
        pts_crop = pts_all[start:end] # (Points_Per_Side, 2)

        # 映射到裁切座標（以裁切左上為原點）
        crop_w, crop_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        pts_local = np.empty_like(pts_crop)
        pts_local[:, 0] = pts_crop[:, 0] - x1
        pts_local[:, 1] = pts_crop[:, 1] - y1

        # 取得裁切圖
        img_crop = img.crop((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))

        # 3) 裁切後再 transform（等同你原來）
        img_out = self.transform(img_crop)  # Tensor [3, 224, 224]

        # 裁切→縮放比例（到 224×224）
        sx = self.input_size / crop_w
        sy = self.input_size / crop_h
        pts_resized = np.empty_like(pts_local)
        pts_resized[:, 0] = pts_local[:, 0] * sx
        pts_resized[:, 1] = pts_local[:, 1] * sy

        keypoints = torch.tensor(pts_resized.reshape(-1), dtype=torch.float32)
        crop_size = (x2 - x1, y2 - y1)
        
        return img_out, keypoints, crop_size, img_name
    
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
        self.reorder_indices = self.ds.config.get("mirror_reorder", None) # 讀取關鍵點的重排規則
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, keypoints, crop_size, img_name = self.ds[idx]
        # 1) 鏡像影像
        image_flipped = TF.hflip(image) 
        _, H, W = image.shape
        k = keypoints.clone()
        for i in range(0, len(k), 2):
            k[i] = (W - 1) - k[i]
        
        # 2) 重新排序關鍵點
        if self.reorder_indices is not None:
            k_xy = k.view(-1, 2)
            # 確保 reorder 的長度與目前的點數一致 (防呆)
            if len(self.reorder_indices) == k_xy.shape[0]:
                k_xy = k_xy[self.reorder_indices, :]
                k_out = k_xy.reshape(-1)
            else:
                # 如果配置檔的 reorder 數量跟實際點數不合，退回不重排並印警告
                print(f"Warning: Mirror reorder size mismatch. Config: {len(self.reorder_indices)}, Data: {k_xy.shape[0]}")
                k_out = k
        else:
            k_out = k
        return image_flipped, k_out, crop_size, img_name
    
if __name__ == "__main__":
    from .transforms import get_hip_base_transform

    transform = get_hip_base_transform(input_size=224)

    dataset = HipCropKeypointDataset(
        # img_dir="../dataset/xray_IHDI_6/images",
        # annotation_dir="../dataset/xray_IHDI_6/annotations",
        # detections_dir="../dataset/xray_IHDI_6/detections",
        img_dir="dataset/mtddh_xray_2d/images",
        annotation_dir="dataset/mtddh_xray_2d/annotations",
        detections_dir="dataset/mtddh_xray_2d/detections",
        side="right",
        transform=transform,
        crop_expand=0.0,
        keep_square=True,
        input_size=224,
        bbox_jitter=True, jitter_center=0.05, jitter_scale=0.10, jitter_prob=0.7
    )
    
    augmented_dataset = ProbAugmentedKeypointDataset(dataset, p=0.7, max_translate_x=10, max_translate_y=10, max_angle=5, clamp=True)

    save_all_visualizations(augmented_dataset, output_dir="check_left_hip_crops")