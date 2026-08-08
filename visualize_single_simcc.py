import os
import numpy as np
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets.transforms import get_hip_base_transform
from models.model import initialize_model
from utils.evaluation import extract_info_from_model_path

# ==================== 使用者設定區 ====================
MODEL_PATH = "results_nckuh_kfold/convnext_tiny_fpn1234concat_simcc_2d_sr3.0_sigma4.0_cropleft_mirror_224_200_0.0001_64_fold1_best.pth"
MODEL_NAME = "convnext_tiny_fpn1234concat"  # 依你的模型架構填寫
IMAGE_PATH = "results_nckuh_kfold/convnext_tiny_fpn1234concat_simcc_2d_sr3.0_sigma4.0_left-only_224_200_0.0001_64/fold1/crops/left/22069761--20230808--Pelvis0_left.jpg"
# IMAGE_PATH = "results_nckuh_kfold/convnext_tiny_fpn1234concat_simcc_2d_sr3.0_sigma4.0_left-only_224_200_0.0001_64/fold1/crops/right/21901269--20230808--Pelvis0_right.jpg"
OUTPUT_DIR = "simcc_vis_results"
# ======================================================

def infer_simcc_with_probs(
    model, image_tensor, input_size, Nx, Ny, device="cuda"
):
    """輸入單張處理好的 image_tensor [1, C, H, W]，回傳座標與兩軸機率分布。"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)

        # 檢查是否包含 logits_x 與 logits_y
        assert (
            "logits_x" in outputs and "logits_y" in outputs
        ), "模型輸出不包含 logits_x/logits_y，請確認 head_type 為 simcc 系列"

        pred_x = outputs["logits_x"]  # [1, K, Nx]
        pred_y = outputs["logits_y"]  # [1, K, Ny]

        # 1. 計算 Softmax 機率分布
        prob_x = F.softmax(pred_x, dim=-1)[0].cpu().numpy()  # [K, Nx]
        prob_y = F.softmax(pred_y, dim=-1)[0].cpu().numpy()  # [K, Ny]

        # 2. 計算期望值座標
        grid_x = torch.arange(Nx, device=device, dtype=pred_x.dtype).view(
            1, 1, Nx
        )
        grid_y = torch.arange(Ny, device=device, dtype=pred_y.dtype).view(
            1, 1, Ny
        )

        x_idx = (F.softmax(pred_x, dim=-1) * grid_x).sum(dim=-1)  # [1, K]
        y_idx = (F.softmax(pred_y, dim=-1) * grid_y).sum(dim=-1)  # [1, K]

        x = (x_idx / float(Nx) * float(input_size))[0].cpu().numpy()  # [K]
        y = (y_idx / float(Ny) * float(input_size))[0].cpu().numpy()  # [K]

        coords = np.stack([x, y], axis=-1)  # [K, 2]

    return coords, prob_x, prob_y

def plot_simcc_distribution_aligned(
    img_crop,
    coords,
    prob_x,
    prob_y,
    input_size,
    Nx,
    Ny,
    save_path=None,
    target_kp_idx=None,
    kp_names=None,
):
    """將 SimCC 的 X 與 Y 軸機率分布以等長對齊方式畫在圖片旁。

    參數:
        img_crop: PIL.Image 或 numpy array (尺寸為 input_size x input_size)
        coords: [K, 2] 預測出的關鍵點像素座標
        prob_x: [K, Nx] X 軸 Softmax 機率
        prob_y: [K, Ny] Y 軸 Softmax 機率
        target_kp_idx: int 或 None。若指定 index，則只畫該點的分布；若為 None 則繪製所有點
        kp_names: list[str]，關鍵點名稱標籤
    """
    K = coords.shape[0]

    # 對應 SimCC bin 到實際像素空間的座標軸
    x_axis = np.arange(Nx) / float(Nx) * input_size
    y_axis = np.arange(Ny) / float(Ny) * input_size

    # 設定畫布與 GridSpec (比例配置：圖片佔 3，曲線佔 1)
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[3.5, 1],
        height_ratios=[1, 3.5],
        wspace=0.05,
        hspace=0.05,
    )

    # 建立三個子圖，並透過 sharex/sharey 強制綁定長度與縮放
    ax_img = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax_img)  # 上方：對齊 X 軸
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax_img)  # 右方：對齊 Y 軸

    # 1. 繪製中央裁切影像
    ax_img.imshow(img_crop, cmap="gray" if img_crop.mode == "L" else None)
    ax_img.set_xlim(0, input_size)
    ax_img.set_ylim(input_size, 0)  # 影像座標 Y 軸向下為正

    # 決定要繪製的關鍵點範圍
    indices_to_plot = [target_kp_idx] if target_kp_idx is not None else range(K)
    cmap = plt.get_cmap("tab10" if K <= 10 else "rainbow", K)

    # 2. 逐點畫出座標與分布曲線
    for k in indices_to_plot:
        color = cmap(k)
        label = kp_names[k] if (kp_names and k < len(kp_names)) else f"KP {k}"
        px, py = coords[k]

        # (A) 在原圖上打點
        ax_img.scatter(
            px, py, color=color, s=50, edgecolors="white", zorder=5, label=label
        )
        ax_img.axvline(x=px, color=color, linestyle="--", alpha=0.4)
        ax_img.axhline(y=py, color=color, linestyle="--", alpha=0.4)

        # (B) 上方子圖：X 軸機率分布
        ax_x.plot(x_axis, prob_x[k], color=color, lw=2)
        ax_x.fill_between(x_axis, prob_x[k], color=color, alpha=0.2)
        ax_x.axvline(x=px, color=color, linestyle="--", alpha=0.8)

        # (C) 右方子圖：Y 軸機率分布 (注意：X 軸放機率值，Y 軸放空間位置)
        ax_y.plot(prob_y[k], y_axis, color=color, lw=2)
        ax_y.fill_betweenx(y_axis, 0, prob_y[k], color=color, alpha=0.2)
        ax_y.axhline(y=py, color=color, linestyle="--", alpha=0.8)

    # 隱藏重疊的刻度標籤，讓外觀像一體成型的儀表板
    plt.setp(ax_x.get_xticklabels(), visible=False)
    plt.setp(ax_y.get_yticklabels(), visible=False)
    ax_x.set_ylabel("Prob (X)")
    ax_y.set_xlabel("Prob (Y)")

    if target_kp_idx is None and K <= 10:
        ax_img.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Save] 分布可視化圖已儲存至: {save_path}")
    else:
        plt.show()

    plt.close(fig)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 於模型名稱解析參數
    (
        head_type,
        input_size,
        epochs,
        lr,
        bs,
        split_ratio,
        sigma,
    ) = extract_info_from_model_path(MODEL_PATH)
    assert head_type in [
        "simcc",
        "simcc_1d",
        "simcc_2d",
        "simcc_2d_deconv",
    ], "此工具僅適用於 SimCC 架構模型"

    Nx = int(input_size * split_ratio)
    Ny = int(input_size * split_ratio)
    points_per_side = 6  # 依你的 Dataset 設定 (例如 12點除以2 = 6)

    # 2. 載入模型
    model = initialize_model(
        MODEL_NAME,
        points_per_side,
        head_type,
        (input_size, input_size),
        Nx,
        Ny,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.to(device).eval()

    # 3. 讀取並前處理影像
    img_crop = Image.open(IMAGE_PATH).convert("L")  # 依你的訓練維持單通道灰階
    if img_crop.size != (input_size, input_size):
        img_crop = img_crop.resize((input_size, input_size), Image.BILINEAR)

    transform = get_hip_base_transform(input_size)
    img_tensor = transform(img_crop).unsqueeze(0)  # [1, C, H, W]

    # 4. 推論並獲取分布
    coords, prob_x, prob_y = infer_simcc_with_probs(
        model, img_tensor, input_size, Nx, Ny, device
    )

    # 5. 輸出可視化
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    # 模式 A：將所有關鍵點畫在同一張圖上 (總覽)
    plot_simcc_distribution_aligned(
        img_crop=img_crop,
        coords=coords,
        prob_x=prob_x,
        prob_y=prob_y,
        input_size=input_size,
        Nx=Nx,
        Ny=Ny,
        save_path=os.path.join(OUTPUT_DIR, f"{base_name}_all_kp.png"),
    )

    # 模式 B：為每一個關鍵點單獨生成一張檢視圖 (避免多點曲線重疊干擾)
    for k in range(points_per_side):
        plot_simcc_distribution_aligned(
            img_crop=img_crop,
            coords=coords,
            prob_x=prob_x,
            prob_y=prob_y,
            input_size=input_size,
            Nx=Nx,
            Ny=Ny,
            target_kp_idx=k,
            save_path=os.path.join(OUTPUT_DIR, f"{base_name}_kp_{k}.png"),
        )


if __name__ == "__main__":
    main()