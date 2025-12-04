from utils.simcc import decode_simcc_to_xy

# predicted coordinates from model outputs
def get_pred_coords(outputs, head_type, Nx=None, Ny=None, input_size=None):
    """
    把模型輸出統一轉成 [B, K, 2] 的 tensor (還在 model device 上)。

    支援情況：
      - direct_regression:
          outputs = {"type": "direct_regression", "coords": [B, K, 2] or [B, 2K]}
          或 outputs 直接是 tensor [B, K, 2] / [B, 2K]

      - simcc_1d / simcc_2d / simcc_2d_deconv:
          outputs = {
              "type": "...",
              "logits_x": [B, K, Nx],
              "logits_y": [B, K, Ny],
          }
          → decode_simcc_to_xy() → [B, K, 2]
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