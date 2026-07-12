import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn

def generate_target_heatmap_vectorized(keypoints, heatmap_size, sigma=2.0, target_weight=None):
    """
    零迴圈純向量化高斯熱點圖生成
    """
    B, J, _ = keypoints.shape
    H, W = heatmap_size
    device = keypoints.device

    # 1. 建立向量化網格 [1, 1, H, 1] 與 [1, 1, 1, W]
    y_grid = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    x_grid = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)

    # 2. 提取座標並調整形狀以觸發廣播 (Broadcasting)
    x = keypoints[..., 0].view(B, J, 1, 1)
    y = keypoints[..., 1].view(B, J, 1, 1)

    # 3. 全批次平行計算高斯分佈
    dist_sq = (x_grid - x) ** 2 + (y_grid - y) ** 2
    heatmaps = torch.exp(-dist_sq / (2 * sigma ** 2))

    # 4. 嚴格邊界與可見度遮罩 (排除 (0,0) 補零陷阱)
    valid_mask = (x > 0) & (y > 0) & (x < W) & (y < H)
    if target_weight is not None:
        valid_mask = valid_mask & (target_weight.view(B, J, 1, 1) > 0)

    return torch.where(valid_mask, heatmaps, torch.zeros_like(heatmaps))


def compute_loss_heatmap(outputs, keypoints, heatmap_size, input_size, sigma, criterion, target_weight=None):
    """
    接收外部傳入的 criterion，兼顧執行效率與 SOTA 梯度放大
    """
    pred_heatmaps = outputs["heatmaps"]  # [B, J, H, W]
    B, J, _, _ = pred_heatmaps.shape
    
    kpts = keypoints.view(B, J, 2).clone()

    # --- 座標自動防禦縮放 ---
    if kpts.max() <= 1.0:
        kpts[..., 0] *= heatmap_size[1]
        kpts[..., 1] *= heatmap_size[0]
    else:
        scale_x = heatmap_size[1] / float(input_size[1] if isinstance(input_size, (list, tuple)) else input_size)
        scale_y = heatmap_size[0] / float(input_size[0] if isinstance(input_size, (list, tuple)) else input_size)
        kpts[..., 0] *= scale_x
        kpts[..., 1] *= scale_y

    # 生成乾淨的 GT
    target_heatmaps = generate_target_heatmap_vectorized(kpts, heatmap_size, sigma, target_weight)

    # --- 使用您外部傳入的 criterion 計算 ---
    loss = criterion(pred_heatmaps, target_heatmaps)

    # --- 解決均值 MSE 導致的梯度稀釋 (關鍵修正) ---
    # 如果您的 criterion 是 MSELoss(reduction='mean')，此係數能喚醒被背景稀釋的梯度
    loss = loss * (0.5 * J * 100.0)

    return loss