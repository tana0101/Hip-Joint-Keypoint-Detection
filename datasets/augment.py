import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class AugmentedKeypointDataset(Dataset):
    def __init__(self, original_dataset, max_angle=0, max_translate_x=0, max_translate_y=0):
        self.original_dataset = original_dataset
        self.max_angle = max_angle
        self.max_translate_x = max_translate_x
        self.max_translate_y = max_translate_y

    def __len__(self):
        return len(self.original_dataset)

    @staticmethod
    def _affine_keypoints_xy(k_xy: torch.Tensor, angle_deg: float, tx: float, ty: float, W: int, H: int):
        """
        k_xy: (N,2) in pixel coords on the image tensor (same space as TF.affine)
        angle_deg: same sign as TF.affine(angle=...)
        tx, ty: translation in pixels (same as TF.affine translate)
        W, H: image width/height
        """
        # Use (W-1)/2 to match pixel center convention more stably
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Move origin to center
        x = k_xy[:, 0] - cx
        y = k_xy[:, 1] - cy

        # Rotate (same direction as TF.affine angle)
        x_new = cos_t * x - sin_t * y
        y_new = sin_t * x + cos_t * y

        # Move back + translate
        x_new = x_new + cx + tx
        y_new = y_new + cy + ty

        out = torch.stack([x_new, y_new], dim=1)
        return out

    def __getitem__(self, idx):
        image, keypoints, crop_size, img_name = self.original_dataset[idx]

        # image: (C,H,W)
        _, H, W = image.shape

        # 1) sample params (推薦用 int translation，避免影像與點 rounding 不一致)
        angle = float(np.random.uniform(-self.max_angle, self.max_angle))
        tx = int(np.random.uniform(-self.max_translate_x, self.max_translate_x))
        ty = int(np.random.uniform(-self.max_translate_y, self.max_translate_y))

        # 2) apply ONE affine to image
        #    注意：TF.affine 的 angle 正負方向，就用同一個 angle 去轉 keypoints
        image_aug = TF.affine(
            image,
            angle=angle,
            translate=[tx, ty],
            scale=1.0,
            shear=[0.0, 0.0],
        )

        # 3) apply SAME affine to keypoints
        k = keypoints.view(-1, 2).clone()
        k_aug = self._affine_keypoints_xy(k, angle_deg=angle, tx=tx, ty=ty, W=W, H=H)

        # 4) (可選但建議) clamp 到影像範圍，避免超出造成 loss 爆炸或學到怪東西
        k_aug[:, 0] = k_aug[:, 0].clamp(0, W - 1)
        k_aug[:, 1] = k_aug[:, 1].clamp(0, H - 1)

        return image_aug, k_aug.reshape(-1).to(torch.float32), crop_size, img_name

# 一階段使用的增強版本，包含水平翻轉 + 幾何變換
class FullImageAugmentedDataset(Dataset):
    def __init__(self, original_dataset, p=0.7, max_angle=10, trans_ratio=0.05, scale_range=0.15, clamp=True):
        self.original_dataset = original_dataset
        self.p = p
        self.max_angle = max_angle
        self.trans_ratio = trans_ratio # 平移改用比例 (如 0.05 代表 5%)
        self.scale_range = scale_range # 縮放範圍 (如 0.15 代表 0.85 ~ 1.15)
        self.clamp = clamp 
        
        self.total_points = original_dataset.total_points
        self.points_per_side = self.total_points // 2
        
        # 定義左右翻轉時，哪些點的 Index 要互換 (例如左邊第0點換到右邊第6點)
        self.flip_pairs = [(i, i + self.points_per_side) for i in range(self.points_per_side)]

    def __len__(self):
        return len(self.original_dataset)

    @staticmethod
    def _affine_keypoints_xy(k_xy: torch.Tensor, angle_deg: float, tx: float, ty: float, scale: float, W: int, H: int):
        cx = W * 0.5
        cy = H * 0.5

        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x = k_xy[:, 0] - cx
        y = k_xy[:, 1] - cy

        # 包含 Scale 的旋轉矩陣
        x_new = scale * (cos_t * x - sin_t * y)
        y_new = scale * (sin_t * x + cos_t * y)

        x_new = x_new + cx + tx
        y_new = y_new + cy + ty

        return torch.stack([x_new, y_new], dim=1)

    def __getitem__(self, idx):
        image, keypoints, crop_size, img_name = self.original_dataset[idx]
        _, H, W = image.shape
        
        # 轉回 (N, 2) 形狀方便操作
        k_aug = keypoints.view(-1, 2).clone()

        # 1. 隨機水平翻轉 (50% 機率，這對齊了你的 MirroredDataset)
        if np.random.rand() < 0.5:
            image = TF.hflip(image)
            k_aug[:, 0] = W - k_aug[:, 0]
            # 交換左右點的 Index
            for i, j in self.flip_pairs:
                k_aug[[i, j]] = k_aug[[j, i]]

        # 2. 幾何變換 Affine (按機率觸發)
        if np.random.rand() < self.p:
            angle = float(np.random.uniform(-self.max_angle, self.max_angle))
            tx = int(np.random.uniform(-self.trans_ratio, self.trans_ratio) * W)
            ty = int(np.random.uniform(-self.trans_ratio, self.trans_ratio) * H)
            scale = float(np.random.uniform(1.0 - self.scale_range, 1.0 + self.scale_range))

            # 對影像執行 Affine
            image = TF.affine(
                image,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
            )

            # 對關鍵點執行相同的 Affine
            k_aug = self._affine_keypoints_xy(k_aug, angle_deg=angle, tx=tx, ty=ty, scale=scale, W=W, H=H)

        if self.clamp:
            k_aug[:, 0] = k_aug[:, 0].clamp(0, W - 1)
            k_aug[:, 1] = k_aug[:, 1].clamp(0, H - 1)

        return image, k_aug.reshape(-1).to(torch.float32), crop_size, img_name

# 二階段使用的增強版本
class ProbAugmentedKeypointDataset(Dataset):
    def __init__(self, original_dataset, p=0.7, max_angle=0, max_translate_x=0, max_translate_y=0,
                 clamp=True, seed=None):
        """
        Args:
            original_dataset: base dataset
            p: probability to apply augmentation
            max_angle: degrees
            max_translate_x/y: pixels
            clamp: clamp keypoints into image bounds
            seed: optional, for reproducibility per-worker you should rely on worker_init_fn instead
        """
        self.original_dataset = original_dataset
        self.p = p
        self.max_angle = max_angle
        self.max_translate_x = max_translate_x
        self.max_translate_y = max_translate_y
        self.clamp = clamp 
        self.rng = np.random.RandomState(seed) if seed is not None else None

    def __len__(self):
        return len(self.original_dataset)

    @staticmethod
    def _affine_keypoints_xy(k_xy: torch.Tensor, angle_deg: float, tx: float, ty: float, W: int, H: int):
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x = k_xy[:, 0] - cx
        y = k_xy[:, 1] - cy

        x_new = cos_t * x - sin_t * y
        y_new = sin_t * x + cos_t * y

        x_new = x_new + cx + tx
        y_new = y_new + cy + ty

        return torch.stack([x_new, y_new], dim=1)

    def __getitem__(self, idx):
        image, keypoints, crop_size, img_name = self.original_dataset[idx]
        _, H, W = image.shape

        # choose RNG
        rng = self.rng if self.rng is not None else np.random

        # coin flip: do augmentation or not
        if rng.rand() >= self.p or (self.max_angle == 0 and self.max_translate_x == 0 and self.max_translate_y == 0):
            return image, keypoints.to(torch.float32), crop_size, img_name

        # sample params (use int translation to match image transform)
        angle = float(rng.uniform(-self.max_angle, self.max_angle))
        tx = int(rng.uniform(-self.max_translate_x, self.max_translate_x))
        ty = int(rng.uniform(-self.max_translate_y, self.max_translate_y))

        # apply ONE affine to image
        image_aug = TF.affine(
            image,
            angle=angle,
            translate=[tx, ty],
            scale=1.0,
            shear=[0.0, 0.0],
        )

        # apply SAME affine to keypoints
        k = keypoints.view(-1, 2).clone()
        k_aug = self._affine_keypoints_xy(k, angle_deg=angle, tx=tx, ty=ty, W=W, H=H)

        if self.clamp:
            k_aug[:, 0] = k_aug[:, 0].clamp(0, W - 1)
            k_aug[:, 1] = k_aug[:, 1].clamp(0, H - 1)

        return image_aug, k_aug.reshape(-1).to(torch.float32), crop_size, img_name