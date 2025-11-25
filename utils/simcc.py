import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

def compute_loss_simcc(
    outputs,              # 例如 (pred_x, pred_y) 或 dict
    keypoints,            # [B, 2 * num_points] 原始座標
    Nx, Ny,
    input_size,
    sigma,
    simcc_label_encoder,  # encoder
    simcc_loss_fn,        # SimCCLoss
):
    pred_x, pred_y = outputs  # 假設 SimCCHead 回傳 (pred_x, pred_y)

    # 對 GT keypoints 做 1D label 編碼（x / y 各一條）
    target_x, target_y, target_weight = simcc_label_encoder(
        keypoints, Nx=Nx, Ny=Ny, input_size=input_size, sigma=sigma
    )

    loss = simcc_loss_fn(
        pred_x, pred_y,
        target_x, target_y,
        target_weight=target_weight
    )
    return loss

def simcc_label_encoder(
    keypoints: torch.Tensor,
    Nx: int,
    Ny: int,
    input_size: int,
    sigma: float = 6,
):
    """
    將 [B, 2*K] 的連續 keypoints 轉成 SimCC 需要的 1D label:
      - target_x: [B, K, Nx]
      - target_y: [B, K, Ny]
      - target_weight: [B, K] (目前全部設為 1，可之後支援不可見點)

    keypoints:  [B, 2*K]  (x1, y1, x2, y2, ...)
    Nx, Ny:     x / y 軸的 bin 數
    input_size: 影像邊長 (假設 x, y ∈ [0, input_size])
    sigma:      高斯在離散 bin 空間的標準差
    """
    device = keypoints.device
    B, twoK = keypoints.shape
    K = twoK // 2

    # 轉成 [B, K] 的 x, y
    x = keypoints[:, 0::2]  # [B, K]
    y = keypoints[:, 1::2]  # [B, K]

    # 將座標 scale 到 [0, Nx) / [0, Ny)
    # 注意這裡假設 keypoint 已經在 [0, input_size) 範圍
    # 若有出界可額外做 clamp
    x_idx = x / float(input_size) * float(Nx)
    y_idx = y / float(input_size) * float(Ny)

    # clamp 到合法範圍
    x_idx = x_idx.clamp(0, Nx - 1 - 1e-6)
    y_idx = y_idx.clamp(0, Ny - 1 - 1e-6)

    # 離散 index 當作 Gaussian 中心 (仍用 float 做 expectation 比較平滑)
    # 建立每個軸的坐標 [0, 1, 2, ..., Nx-1]
    grid_x = torch.arange(Nx, device=device).view(1, 1, Nx)  # [1,1,Nx]
    grid_y = torch.arange(Ny, device=device).view(1, 1, Ny)  # [1,1,Ny]

    # 擴充中心 index 形狀為 [B, K, 1]
    x_idx_exp = x_idx.unsqueeze(-1)  # [B, K, 1]
    y_idx_exp = y_idx.unsqueeze(-1)  # [B, K, 1]

    # 高斯分布：exp(-(i - mu)^2 / (2*sigma^2))
    # target_x: [B, K, Nx], target_y: [B, K, Ny]
    target_x = torch.exp(- (grid_x - x_idx_exp) ** 2 / (2 * sigma ** 2))
    target_y = torch.exp(- (grid_y - y_idx_exp) ** 2 / (2 * sigma ** 2))

    # normalize 成機率分布
    target_x = target_x / (target_x.sum(dim=-1, keepdim=True) + 1e-8)
    target_y = target_y / (target_y.sum(dim=-1, keepdim=True) + 1e-8)

    # 目前先全部視為可見點 (weight = 1)
    target_weight = torch.ones(B, K, device=device, dtype=keypoints.dtype)

    return target_x, target_y, target_weight

class SimCCLoss(torch.nn.Module):
    """
    KLDivLoss 版本的 SimCC loss:
      - 先對 pred_x, pred_y 做 log_softmax
      - 再和 target_x, target_y 做 KL divergence
      - 對 joint 做加權平均
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        # 這裡用 reduction='none' 自己處理加權平均
        self.criterion = torch.nn.KLDivLoss(reduction="none")
        self.reduction = reduction

    def forward(
        self,
        pred_x: torch.Tensor,      # [B, K, Nx] logits
        pred_y: torch.Tensor,      # [B, K, Ny] logits
        target_x: torch.Tensor,    # [B, K, Nx] prob
        target_y: torch.Tensor,    # [B, K, Ny] prob
        target_weight: torch.Tensor,   # [B, K]
    ):
        # log_softmax -> log p
        log_p_x = F.log_softmax(pred_x, dim=-1)
        log_p_y = F.log_softmax(pred_y, dim=-1)

        # KLDivLoss(log_p, q)
        loss_x = self.criterion(log_p_x, target_x)  # [B, K, Nx]
        loss_y = self.criterion(log_p_y, target_y)  # [B, K, Ny]

        # 對每個軸的分布 sum over bins
        loss_x = loss_x.sum(dim=-1)  # [B, K]
        loss_y = loss_y.sum(dim=-1)  # [B, K]

        # 加上關節權重
        # target_weight: [B, K]
        loss_x = loss_x * target_weight
        loss_y = loss_y * target_weight

        # 合併 x, y 兩個方向
        loss = loss_x + loss_y  # [B, K]

        if self.reduction == "mean":
            # 平均到「可見關節數 * 2 個軸」
            denom = target_weight.sum() * 2.0 + 1e-6
            loss = loss.sum() / denom
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            # 'none' -> 回傳 [B, K]
            pass

        return loss
    
def simcc_loss_fn(
    pred_x: torch.Tensor,
    pred_y: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    target_weight: torch.Tensor,
    simcc_loss_module: SimCCLoss = None,
):
    """
    方便在訓練時呼叫的包裝函式.
    如果沒有傳 simcc_loss_module 就建一個預設的 SimCCLoss。
    """
    if simcc_loss_module is None:
        simcc_loss_module = SimCCLoss(reduction="mean")
    return simcc_loss_module(pred_x, pred_y, target_x, target_y, target_weight)


def decode_simcc_to_xy(
    pred_x: torch.Tensor,  # [B, K, Nx] logits
    pred_y: torch.Tensor,  # [B, K, Ny] logits
    Nx: int,
    Ny: int,
    input_size: int,
) -> torch.Tensor:
    """
    將 SimCC 的 logits 轉回 [B, 2*K] 的連續座標 (x1,y1,x2,y2,...)

    做法：
      1. 先 softmax 得到每個 bin 的機率
      2. 用 expectation E[i] 當作離散坐標 (在 [0, Nx) / [0, Ny))
      3. 再 scale 回 pixel 空間: coord = idx / N_axis * input_size
    """
    device = pred_x.device
    B, K, _ = pred_x.shape

    # 機率分布
    prob_x = F.softmax(pred_x, dim=-1)  # [B, K, Nx]
    prob_y = F.softmax(pred_y, dim=-1)  # [B, K, Ny]

    # 離散 index 0..Nx-1, 0..Ny-1
    grid_x = torch.arange(Nx, device=device, dtype=prob_x.dtype).view(1, 1, Nx)
    grid_y = torch.arange(Ny, device=device, dtype=prob_y.dtype).view(1, 1, Ny)

    # expectation: sum_i p(i) * i
    x_idx = (prob_x * grid_x).sum(dim=-1)  # [B, K]
    y_idx = (prob_y * grid_y).sum(dim=-1)  # [B, K]

    # scale 回 pixel 空間
    x = x_idx / float(Nx) * float(input_size)  # [B, K]
    y = y_idx / float(Ny) * float(input_size)  # [B, K]

    # 拼成 [B, K, 2] -> [B, 2*K] (x1,y1,x2,y2,...)
    coords = torch.stack([x, y], dim=-1)  # [B, K, 2]
    coords = coords.view(B, -1)           # [B, 2*K]

    return coords

def visualize_simcc_distributions(
    keypoint, 
    Nx, 
    Ny, 
    input_size, 
    params_list, 
    save_path='simcc_comparison.png'
):
    """
    輸入:
        keypoint: tuple (x, y) 目標座標
        params_list: list of dict, 例如 [{'sigma': 6.0, 'label_smoothing': 0.1}, ...]
    輸出:
        儲存比較圖檔
    """
    # 轉成 Tensor [1, 2]
    kp_tensor = torch.tensor([[float(keypoint[0]), float(keypoint[1])]])
    
    plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(1, 2, 1) # X軸
    ax2 = plt.subplot(1, 2, 2) # Y軸
    
    # 自動生成顏色
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(params_list)))
    
    for idx, params in enumerate(params_list):
        sigma = params.get('sigma', 2.0)
        
        # 呼叫 SimCC Encoder
        target_x, target_y, _ = simcc_label_encoder(
            kp_tensor, Nx, Ny, input_size, sigma=sigma
        )
        
        # 轉成 Numpy 畫圖
        dist_x = target_x[0, 0].numpy()
        dist_y = target_y[0, 0].numpy()
        
        label_str = f"σ={sigma}"
        
        # 畫圖
        ax1.plot(dist_x, label=label_str, color=colors[idx], linewidth=2.5, alpha=0.8)
        ax2.plot(dist_y, label=label_str, color=colors[idx], linewidth=2.5, alpha=0.8)

    # 設定 X 軸圖表細節 (Zoom in 到目標點附近 +/- 40 bins)
    center_x = int(keypoint[0] / input_size * Nx)
    ax1.set_xlim(max(0, center_x - 40), min(Nx, center_x + 40))
    ax1.set_title(f"X-axis Target @ {keypoint[0]}")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # 設定 Y 軸圖表細節
    center_y = int(keypoint[1] / input_size * Ny)
    ax2.set_xlim(max(0, center_y - 40), min(Ny, center_y + 40))
    ax2.set_title(f"Y-axis Target @ {keypoint[1]}")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"圖片已儲存至: {save_path}")
    plt.show()

# 用於測試 label encoder
if __name__ == "__main__":

    params_to_test = [ 
        {'sigma': 2.0},  
        {'sigma': 3.0},  
        {'sigma': 4.0},
        {'sigma': 5.0},
        {'sigma': 6.0},
    ]

    visualize_simcc_distributions(
        keypoint=(100, 150),
        Nx=448, Ny=448, input_size=224,
        params_list=params_to_test,
        save_path='simcc_comparison.png'
    )