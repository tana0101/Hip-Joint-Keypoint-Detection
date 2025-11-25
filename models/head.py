
import torch
import torch.nn as nn

# ===== 共用的小工具：可插拔 head 包裝器 =====
class HeadAdapter(nn.Module):
    """
    輸入: 各 backbone/transformer/融合模組輸出的單一向量 z: [B, in_features]
    依 head_type 產生：
      - direct_regression: [B, num_points*2]
      - simcc            : ([B, num_points, Nx], [B, num_points, Ny])
    """
    def __init__(self, head_type: str, in_features: int, num_points: int, Nx: int = None, Ny: int = None):
        super().__init__()
        head_type = head_type.lower()
        if head_type not in HEAD_MODULES:
            raise ValueError(f"Unknown head_type: {head_type}. Choices = {list(HEAD_MODULES.keys())}")
        self.head_type = head_type
        if head_type == "direct_regression":
            self.head = DirectRegressionHead(in_features, num_points)
        elif head_type == "simcc":
            assert Nx is not None and Ny is not None, "SimCC 需要 Nx, Ny"
            self.head = SimCCHead(in_features, num_points, Nx, Ny)

    def forward(self, z):
        return self.head(z)

# ===== 具體的 Head 實作 =====
class DirectRegressionHead(nn.Module):
    """
    原始的直接座標迴歸 Head (例如用於 SimpleBaseline 或 DirectPose)
    輸出: [B, num_points * 2] (所有點的 x, y 座標)
    """
    def __init__(self, in_features, num_points):
        super().__init__()
        # 單一線性層，直接輸出所有關鍵點的 x, y 座標
        self.regressor = nn.Linear(in_features, num_points * 2)

    def forward(self, x):
        return self.regressor(x)

class SimCCHead(nn.Module):
    """
    SimCC 座標分類 Head
    輸出: 兩個獨立的 Logits: [B, num_points, Nx], [B, num_points, Ny]
    """
    def __init__(self, in_features, num_points, Nx, Ny):
        super().__init__()
        self.num_points = num_points
        self.Nx = Nx
        self.Ny = Ny
        
        # 水平座標分類器 (一個線性層)
        # 輸出 num_points * Nx 個 Logits
        self.horizontal_classifier = nn.Linear(in_features, num_points * Nx)
        
        # 垂直座標分類器 (一個線性層)
        # 輸出 num_points * Ny 個 Logits
        self.vertical_classifier = nn.Linear(in_features, num_points * Ny)

    def forward(self, x):
        # x: [B, in_features]
        out_x_raw = self.horizontal_classifier(x)
        out_y_raw = self.vertical_classifier(x)
        
        # 重新塑形為 [B, num_points, Nx] 和 [B, num_points, Ny]
        pred_x = out_x_raw.view(-1, self.num_points, self.Nx)
        pred_y = out_y_raw.view(-1, self.num_points, self.Ny)
        
        return pred_x, pred_y # SimCC 輸出兩個分類結果
    
HEAD_MODULES = {
    "direct_regression": DirectRegressionHead,
    "simcc": SimCCHead,
}