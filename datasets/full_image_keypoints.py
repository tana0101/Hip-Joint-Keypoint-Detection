import os
import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from utils.csv_parser import parse_csv_get_points

class FullImageKeypointDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None, input_size=512):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.input_size = input_size

        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
        self.annotations = sorted([f for f in os.listdir(annotation_dir) if f.endswith(".csv")])
        print(f"Found {len(self.images)} images and {len(self.annotations)} annotations.")
        assert len(self.images) == len(self.annotations), "Images/annotations count mismatch"

        # 自動偵測所有點數
        if len(self.annotations) > 0:
            sample_path = os.path.join(self.annotation_dir, self.annotations[0])
            sample_pts = parse_csv_get_points(sample_path)
            self.total_points = sample_pts.shape[0]
            print(f"[One-Stage] Detected {self.total_points} points total.")
        else:
            raise ValueError("Annotation directory is empty.")

    def __len__(self):
        return len(self.images)

    @property
    def num_keypoints(self):
        return self.total_points

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, self.annotations[idx])

        # 1. 讀取影像為灰階
        img = Image.open(img_path).convert("L")  

        # ==========================================
        # 對齊雙階段的前處理：在補黑邊前做 Equalize 與轉 3 通道
        # 這是不太好的寫法，將前處理綁定在 Dataset 裡，但為了跟雙階段的前處理保持一致，暫時先這樣做，希望我以後有時間改
        # ==========================================
        img = ImageOps.equalize(img)
        img = img.convert("RGB") # 等同於 Grayscale(num_output_channels=3)

        W, H = img.size

        # 讀取所有點
        pts = parse_csv_get_points(ann_path) # (Total, 2)

        # 2. Letterbox Padding (補黑邊變成正方形，此時補的是 RGB 黑邊 (0,0,0))
        max_side = max(W, H)
        pad_left = (max_side - W) // 2
        pad_top = (max_side - H) // 2
        
        img_padded = Image.new("RGB", (max_side, max_side), color=(0, 0, 0))
        img_padded.paste(img, (pad_left, pad_top))

        # 點座標跟著平移
        pts[:, 0] += pad_left
        pts[:, 1] += pad_top

        # 3. 等比例縮放至 input_size
        img_resized = img_padded.resize((self.input_size, self.input_size), Image.BILINEAR)
        scale_factor = self.input_size / float(max_side)
        
        pts_resized = pts * scale_factor
        keypoints = torch.tensor(pts_resized.reshape(-1), dtype=torch.float32)

        # 4. 最後只做 ToTensor()，或者加上 Normalize 等純數值操作
        if self.transform:
            img_out = self.transform(img_resized)
        else:
            from torchvision.transforms import functional as TF
            img_out = TF.to_tensor(img_resized)

        original_square_size = (max_side, max_side)
        
        return img_out, keypoints, original_square_size, img_name