import torch
import torch.nn as nn
import numpy as np

def generate_target_heatmap(keypoints, heatmap_size, sigma=2.0):
    """
    keypoints: [B, J, 2] (相對於原圖的絕對座標)
    heatmap_size: (H, W) 模型輸出的特徵圖大小
    sigma: 高斯核的標準差
    """
    B, J, _ = keypoints.shape
    H, W = heatmap_size
    device = keypoints.device

    # 建立網格座標 (確保不帶梯度)
    y_grid = torch.arange(H, device=device).view(1, 1, H, 1)
    x_grid = torch.arange(W, device=device).view(1, 1, 1, W)

    target_heatmaps = torch.zeros((B, J, H, W), dtype=torch.float32, device=device)

    kpts_detached = keypoints.detach()

    for b in range(B):
        for j in range(J):
            x, y = kpts_detached[b, j, 0], kpts_detached[b, j, 1]
            
            if x >= 0 and y >= 0:
                # 純數值計算高斯
                target_heatmaps[b, j] = torch.exp(
                    -((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2)
                ).squeeze()

    return target_heatmaps

def compute_loss_heatmap(outputs, keypoints, heatmap_size, input_size, sigma, criterion):
    """
    計算預測 Heatmap 與 GT Heatmap 的 MSE Loss
    """
    pred_heatmaps = outputs["heatmaps"] # [B, J, H, W]
    
    # 將座標依據 heatmap_size 與 input_size 的比例進行縮放
    B, J = keypoints.shape[0], keypoints.shape[1] // 2
    kpts = keypoints.view(B, J, 2)
    
    scale_x = heatmap_size[1] / input_size
    scale_y = heatmap_size[0] / input_size
    
    scaled_kpts = kpts.clone()
    scaled_kpts[..., 0] *= scale_x
    scaled_kpts[..., 1] *= scale_y

    # 生成 GT
    target_heatmaps = generate_target_heatmap(scaled_kpts, heatmap_size, sigma)
    
    # 計算 MSE
    loss = criterion(pred_heatmaps, target_heatmaps)
    return loss