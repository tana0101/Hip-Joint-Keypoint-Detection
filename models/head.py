
import torch
import torch.nn as nn

# ===== 共用的小工具：可插拔 head 包裝器 =====
class HeadAdapter(nn.Module):
    """
    統一 head 包裝。輸入一律是 backbone feature map: feat: [B, C_in, H, W]

    head_type:
      - "direct_regression": GAP → DirectRegressionHead
      - "simcc_1d"         : GAP → SimCC1DHead
      - "simcc_2d"         : Conv2d(C_in -> num_points) → SimCC2DHead
      - "simcc_2d_deconv"  : deconv+Conv2d → SimCC2D_DeconvHead
    """
    def __init__(
        self,
        head_type: str,
        in_channels: int,
        map_size,               # (H, W) backbone 輸出
        num_points: int,
        Nx: int | None = None,
        Ny: int | None = None,
        # deconv 參數可以有預設值
        num_deconv_layers: int = 3,
        deconv_filters=(256, 256, 256),
    ):
        super().__init__()
        head_type = head_type.lower()
        self.head_type = head_type
        self.num_points = num_points

        H, W = map_size
        self.map_size = map_size

        # 共用的 GAP 跟 norm
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gap_norm = nn.LayerNorm(in_channels)

        if head_type == "direct_regression":
            # GAP → [B, C_in] → regression
            self.head_1d = DirectRegressionHead(in_features=in_channels, num_points=num_points)

        elif head_type == "simcc_1d":
            # GAP → SimCC1D
            assert Nx is not None and Ny is not None, "SimCC1D 需要 Nx, Ny"
            self.head_1d = SimCC1DHead(
                in_features=in_channels,
                num_points=num_points,
                Nx=Nx, Ny=Ny,
            )

        elif head_type == "simcc_2d":
            # 不做 deconv，只做 Conv2d(C_in → num_points) + SimCC2D
            assert Nx is not None and Ny is not None, "SimCC2D 需要 Nx, Ny"
            self.final_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_points,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.head_2d = SimCC2DHead(
                num_points=num_points,
                Nx=Nx, Ny=Ny,
                map_size=map_size,   # backbone map size
            )

        elif head_type == "simcc_2d_deconv":
            assert Nx is not None and Ny is not None, "SimCC2D_Deconv 需要 Nx, Ny"
            self.head_2d_deconv = SimCC2D_DeconvHead(
                in_channels=in_channels,
                num_points=num_points,
                Nx=Nx, Ny=Ny,
                backbone_map_size=map_size,
                num_deconv_layers=num_deconv_layers,
                deconv_filters=deconv_filters,
            )
        else:
            raise ValueError(
                f"Unknown head_type: {head_type}. "
                f"Choices = ['direct_regression', 'simcc_1d', 'simcc_2d', 'simcc_2d_deconv']"
            )
    
    # ========= helper: 處理 [B,C,H,W] / [B,C] 兩種情況 =========
    def _gap_and_norm(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, H, W]
        z = self.global_pool(feat).flatten(1)   # [B, C]
        z = self.gap_norm(z)                    # [B, C]
        return z       
    
    def forward(self, feat):
        """
        feat: backbone feature map [B, C_in, H, W] or [B, C_in]
        依 head_type 決定處理方式
        """
        if self.head_type == "direct_regression":
            # GAP → regression
            z = self._gap_and_norm(feat)
            coords = self.head_1d(z)                  # [B, num_points*2]
            coords = coords.view(coords.shape[0], self.num_points, 2)
            return {
                "type": "direct_regression",
                "coords": coords,   # [B, J, 2]
            }

        if self.head_type == "simcc_1d":
            # GAP → SimCC1D
            z = self._gap_and_norm(feat)
            logits_x, logits_y = self.head_1d(z)      # [B, J, Nx/Ny]
            return {
                "type": "simcc_1d",
                "logits_x": logits_x,
                "logits_y": logits_y,
            }

        if self.head_type == "simcc_2d":
            # Conv2d(C_in → num_points) → SimCC2D
            heatmaps = self.final_conv(feat)               # [B, J, H, W]
            logits_x, logits_y = self.head_2d(heatmaps)    # [B, J, Nx/Ny]
            return {
                "type": "simcc_2d",
                "logits_x": logits_x,
                "logits_y": logits_y,
                "heatmaps": heatmaps,
            }

        if self.head_type == "simcc_2d_deconv":
            logits_x, logits_y, heatmaps = self.head_2d_deconv(feat)
            return {
                "type": "simcc_2d_deconv",
                "logits_x": logits_x,
                "logits_y": logits_y,
                "heatmaps": heatmaps,
            }
        

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
    
class SimCC1DHead(nn.Module):
    """
    一維版 SimCC: GAP → global feature → 為每個點輸出 Nx, Ny 的 logits
    輸入:  x: [B, C_in]
    輸出: logits_x: [B, num_points, Nx]
         logits_y: [B, num_points, Ny]
    """
    def __init__(self, in_features, num_points, Nx, Ny):
        super().__init__()
        self.num_points = num_points
        self.Nx = Nx
        self.Ny = Ny
        
        # 水平座標分類器 (一個線性層)
        self.horizontal_classifier = nn.Linear(in_features, num_points * Nx)
        # 垂直座標分類器 (一個線性層)
        self.vertical_classifier = nn.Linear(in_features, num_points * Ny)

    def forward(self, x):
        # x: [B, in_features]
        out_x_raw = self.horizontal_classifier(x)
        out_y_raw = self.vertical_classifier(x)
        
        # 重新塑形為 [B, num_points, Nx] 和 [B, num_points, Ny]
        pred_x = out_x_raw.view(-1, self.num_points, self.Nx)
        pred_y = out_y_raw.view(-1, self.num_points, self.Ny)
        
        return pred_x, pred_y # SimCC 輸出兩個分類結果

class SimCC2DHead(nn.Module):
    """
    二維版 SimCC: heatmap -> flatten -> x/y MLP
    輸入: heatmaps: [B, num_points, H, W]
    輸出: logits_x: [B, num_points, Nx]
         logits_y: [B, num_points, Ny]
    """
    def __init__(self, num_points, Nx, Ny, map_size):
        super().__init__()
        self.num_points = num_points
        self.Nx = Nx
        self.Ny = Ny
        self.H, self.W = map_size
        head_input = self.H * self.W

        self.mlp_head_x = nn.Linear(head_input, Nx)
        self.mlp_head_y = nn.Linear(head_input, Ny)

    def forward(self, heatmaps):
        # heatmaps: [B, J, H, W]
        B, J, H, W = heatmaps.shape

        feat_1d = heatmaps.view(B, J, -1)       # [B, J, H*W]
        feat_1d = feat_1d.view(B * J, -1)       # [B*J, H*W]

        logits_x = self.mlp_head_x(feat_1d)     # [B*J, Nx]
        logits_y = self.mlp_head_y(feat_1d)     # [B*J, Ny]

        logits_x = logits_x.view(B, J, self.Nx) # [B, J, Nx]
        logits_y = logits_y.view(B, J, self.Ny) # [B, J, Ny]

        return logits_x, logits_y
    
class SimCC2D_DeconvHead(nn.Module):
    """
    吃 backbone feature map [B, C_in, Hf, Wf]
    deconv -> Conv2d(output=num_points) -> heatmap [B, num_points, H, W]
    然後用 SimCC2DHead 做 flatten + x/y MLP
    """
    def __init__(
        self,
        in_channels,         # C_in
        num_points,
        Nx,
        Ny,
        backbone_map_size,   # (Hf, Wf) backbone 輸出的大小
        num_deconv_layers=3,
        deconv_filters=(256, 256, 256),
    ):
        super().__init__()
        self.num_points = num_points
        self.in_channels = in_channels

        # 建 deconv 把圖拉大
        self.deconv = self._make_deconv_layers(
            in_channels=in_channels,
            num_layers=num_deconv_layers,
            num_filters=deconv_filters,
        )
        last_channels = deconv_filters[-1]

        # 先用 dummy 走一次 deconv，算出 heatmap size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, backbone_map_size[0], backbone_map_size[1])
            y = self.deconv(dummy)
            _, _, Hh, Wh = y.shape
        self.heatmap_size = (Hh, Wh)

        # final conv: 把 channel 壓成 num_points
        self.final_conv = nn.Conv2d(
            in_channels=last_channels,
            out_channels=num_points,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # heatmap -> SimCC2D
        self.simcc2d = SimCC2DHead(
            num_points=num_points,
            Nx=Nx, Ny=Ny,
            map_size=self.heatmap_size,
        )

    def _make_deconv_layers(self, in_channels, num_layers, num_filters):
        layers = []
        for i in range(num_layers):
            out_channels = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, feat):
        # feat: [B, C_in, Hf, Wf]
        y = self.deconv(feat)            # [B, F, Hh, Wh]
        heatmaps = self.final_conv(y)    # [B, num_points, Hh, Wh]
        logits_x, logits_y = self.simcc2d(heatmaps)

        return logits_x, logits_y, heatmaps

HEAD_MODULES = {
    "direct_regression": DirectRegressionHead,
    "simcc_1d": SimCC1DHead,
    "simcc_2d": SimCC2DHead,
    "simcc_2d_deconv": SimCC2D_DeconvHead,
}