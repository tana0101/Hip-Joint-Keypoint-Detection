import torch

def extend_with_center_points_side(outputs, keypoints, points_count):
    """
    outputs/keypoints:
      - 如果是 tensor: [B, P*2] 或 [B, P, 2]
      - 我們會統一 reshape 成 [B, POINTS_COUNT, 2]
    回傳:
      outputs_ext/keypoints_ext: [B, P+1, 2] (多一個中心點)
    """
    # 如果 outputs 是 dict，就先拿出 coords
    if isinstance(outputs, dict):
        outputs = outputs["coords"]   # 例如 [B, P, 2]

    # keypoints 應該是 tensor (GT)
    B = keypoints.shape[0]

    # 如果是 [B, P*2] 先 reshape
    if outputs.dim() == 2:
        outputs = outputs.view(B, points_count, 2)
    elif outputs.dim() == 3:
        # 假設已經是 [B, P, 2]
        pass
    else:
        raise ValueError(f"Unexpected outputs shape: {outputs.shape}")

    if keypoints.dim() == 2:
        keypoints = keypoints.view(B, points_count, 2)
    elif keypoints.dim() == 3:
        pass
    else:
        raise ValueError(f"Unexpected keypoints shape: {keypoints.shape}")

    center_output = torch.mean(outputs, dim=1, keepdim=True)   # (B,1,2)
    center_keypts = torch.mean(keypoints, dim=1, keepdim=True) # (B,1,2)

    outputs_extended = torch.cat([outputs, center_output], dim=1)   # (B,P+1,2)
    keypoints_extended = torch.cat([keypoints, center_keypts], dim=1)

    return outputs_extended, keypoints_extended

def compute_loss_direct_regression(outputs, keypoints, points_count, criterion):
    # Combine the center points with the original outputs and keypoints
    outputs_extended, keypoints_extended = extend_with_center_points_side(outputs, keypoints, points_count)
    loss = criterion(outputs_extended, keypoints_extended)
    return loss