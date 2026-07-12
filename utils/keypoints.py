from utils.simcc import decode_simcc_expectation_to_xy as decode_simcc_to_xy
import torch

# predicted coordinates from model outputs
def get_pred_coords(outputs, head_type, Nx=None, Ny=None, input_size=None):
    """
    把模型輸出統一轉成 [B, K, 2] 的 tensor (還在 model device 上)。

    支援情況：
      - direct_regression:
          outputs = {"type": "direct_regression", "coords": [B, K, 2] or [B, 2K]}
      - simcc 系列:
          outputs = {"type": "...", "logits_x": [B, K, Nx], "logits_y": [B, K, Ny]}
      - heatmap:
          outputs = {"type": "heatmap", "heatmaps": [B, K, H, W]}
    """
    # ---- direct regression ----
    if head_type == "direct_regression":
        coords = outputs["coords"]      # [B, K, 2]
        return coords

    # ---- SimCC 系列 ----
    elif head_type in ["simcc", "simcc_1d", "simcc_2d", "simcc_2d_deconv"]:
        pred_x = outputs["logits_x"]
        pred_y = outputs["logits_y"]
        coords = decode_simcc_to_xy(
            pred_x,
            pred_y,
            Nx=Nx,
            Ny=Ny,
            input_size=input_size,
        )   # [B, K, 2]
        return coords

    # ---- 2D Heatmap ----
    elif head_type == "heatmap":
        heatmaps = outputs["heatmaps"]
        coords = decode_heatmap(heatmaps, input_size)
        return coords

    else:
        raise ValueError(f"Unknown head_type={head_type} in get_pred_coords()")

# get preds and targets as numpy arrays
def get_preds_and_targets(outputs, keypoints, head_type, Nx, Ny, input_size):
    """
    根據 head_type，把模型輸出轉成 xy 座標，並回傳 numpy 版的 preds / targets
    """
    preds = get_pred_coords(
        outputs,
        head_type=head_type,
        Nx=Nx,
        Ny=Ny,
        input_size=input_size,
    )  # [B, J, 2]
    
    preds_np   = preds.detach().cpu().numpy()
    targets_np = keypoints.detach().cpu().numpy()
    return preds_np, targets_np

def decode_heatmap(heatmaps, input_size):
    """
    遵循原版 HRNet 操作的向量化 0.25 像素偏移解碼函數
    heatmaps: [B, K, H_out, W_out]
    input_size: 224 或 (224, 224)
    回傳: [B, K, 2] 在 input_size 空間的亞像素座標
    """
    B, K, H_out, W_out = heatmaps.shape
    device = heatmaps.device
    
    # 1. 壓平找最大值的 index (與您原本邏輯相同)
    heatmaps_flat = heatmaps.view(B, K, -1)
    _, max_indices = torch.max(heatmaps_flat, dim=-1)
    
    preds_x = (max_indices % W_out).float()
    preds_y = (max_indices // W_out).float()
    
    # --- 【原版 HRNet 關鍵新增步驟】0.25 方向性像素偏移 ---
    # 轉為 Long 型態以利索引
    px = preds_x.long()
    py = preds_y.long()
    
    # 建立安全邊界遮罩：只有不在圖片最邊緣 (0 或 55) 的點才能去抓上下左右鄰居
    valid_mask = (px > 0) & (px < W_out - 1) & (py > 0) & (py < H_out - 1)
    
    # 將座標限縮在安全範圍內，避免張量索引 (Indexing) 時發生 Out of Bounds 報錯
    px_safe = px.clamp(1, W_out - 2)
    py_safe = py.clamp(1, H_out - 2)
    
    # 建立對齊形狀的批次索引 [B, K]
    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, K)
    k_idx = torch.arange(K, device=device).view(1, K).expand(B, K)
    
    # 向量化直接取出上下左右相鄰像素的權重數值
    val_right = heatmaps[b_idx, k_idx, py_safe, px_safe + 1]
    val_left  = heatmaps[b_idx, k_idx, py_safe, px_safe - 1]
    val_down  = heatmaps[b_idx, k_idx, py_safe + 1, px_safe]
    val_up    = heatmaps[b_idx, k_idx, py_safe - 1, px_safe]
    
    # 計算左右與上下差值的正負號 (-1.0, 0.0, 或 +1.0)
    diff_x = val_right - val_left
    diff_y = val_down - val_up
    
    # 只有在 valid_mask 為 True 時，才乘上 0.25 偏移量
    shift_x = torch.sign(diff_x) * 0.25 * valid_mask.float()
    shift_y = torch.sign(diff_y) * 0.25 * valid_mask.float()
    
    # 將 0.25 亞像素偏移量加回原本的座標
    preds_x = preds_x + shift_x
    preds_y = preds_y + shift_y
    # --------------------------------------------------
    
    # 2. 縮放回 input_size 空間 (與您原本邏輯相同)
    scale_x = float(input_size[1] if isinstance(input_size, (list, tuple)) else input_size) / float(W_out)
    scale_y = float(input_size[0] if isinstance(input_size, (list, tuple)) else input_size) / float(H_out)
    
    preds_x = preds_x * scale_x
    preds_y = preds_y * scale_y
    
    coords = torch.stack([preds_x, preds_y], dim=-1)
    return coords