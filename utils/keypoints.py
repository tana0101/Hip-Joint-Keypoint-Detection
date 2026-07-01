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
    將預測的 2D Heatmap 轉回 (x, y) 座標
    heatmaps: [B, K, H_out, W_out]
    回傳: [B, K, 2] 在 input_size 空間的座標
    """
    B, K, H_out, W_out = heatmaps.shape
    
    # 把 H_out * W_out 壓平找最大值的 index
    heatmaps_flat = heatmaps.view(B, K, -1) # [B, K, H_out*W_out]
    _, max_indices = torch.max(heatmaps_flat, dim=-1) # [B, K]
    
    # 將 1D index 轉回 2D 的 (x, y) 座標
    preds_x = (max_indices % W_out).float()
    preds_y = (max_indices // W_out).float()
    
    # 縮放回 input_size 空間
    scale_x = input_size / float(W_out)
    scale_y = input_size / float(H_out)
    
    preds_x = preds_x * scale_x
    preds_y = preds_y * scale_y
    
    coords = torch.stack([preds_x, preds_y], dim=-1) # [B, K, 2]
    return coords