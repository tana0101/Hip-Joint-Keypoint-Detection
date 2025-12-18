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

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, keypoints, crop_size, img_name = self.ds[idx]  # image: Tensor [3,224,224]；keypoints: Tensor 長度=12(6點xy)
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
        return image_flipped, k_out, crop_size, img_name
    
if __name__ == "__main__":
    from transforms import get_hip_base_transform

    transform = get_hip_base_transform(input_size=224)

    dataset = HipCropKeypointDataset(
        img_dir="../dataset/xray_IHDI_6/images",
        annotation_dir="../dataset/xray_IHDI_6/annotations",
        detections_dir="../dataset/xray_IHDI_6/detections",
        side="left",
        transform=transform,
        crop_expand=0.10,
        keep_square=True,
        input_size=224
    )

    save_all_visualizations(dataset, output_dir="check_left_hip_crops")