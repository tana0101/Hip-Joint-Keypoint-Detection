import torch
POINTS_COUNT = 6

def extend_with_center_points_side(outputs, keypoints):
    """
    outputs/keypoints: [B, 12] 對應 6 點 (x,y)
    回傳:
      outputs_ext/keypoints_ext: [B, 7, 2]
    """
    outputs = outputs.view(-1, POINTS_COUNT, 2)    # (B,6,2)
    keypoints = keypoints.view(-1, POINTS_COUNT, 2)

    center_output = torch.mean(outputs, dim=1, keepdim=True)     # (B,1,2)
    center_keypts = torch.mean(keypoints, dim=1, keepdim=True)   # (B,1,2)

    outputs_extended = torch.cat([outputs, center_output], dim=1)    # (B,7,2)
    keypoints_extended = torch.cat([keypoints, center_keypts], dim=1)

    return outputs_extended, keypoints_extended

def compute_loss_direct_regression(outputs, keypoints, criterion):
    # Combine the center points with the original outputs and keypoints
    outputs_extended, keypoints_extended = extend_with_center_points_side(outputs, keypoints)
    loss = criterion(outputs_extended, keypoints_extended)
    return loss