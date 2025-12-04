import os
import argparse
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau
from ultralytics import YOLO

from datasets.transforms import get_hip_base_transform
from utils.detection import _detect_one, _square_expand_clip
from utils.keypoints import get_pred_coords
from models.model import initialize_model
from utils.hip_geometry import (
    calculate_acetabular_index_angles,
    classify_quadrant_ihdi,
    draw_hilgenreiner_line,
    draw_perpendicular_line,
    draw_diagonal_line,
    draw_h_point,
)
from utils.plots import add_sigma_guides, add_zscore_right_axis

POINTS_COUNT = 6
REORDER_6 = [2, 1, 0, 5, 4, 3]  # 單側6點在左右鏡像後的索引重排（3點一組反轉）

YOLO_LEFT_CLS  = 0
YOLO_RIGHT_CLS = 1
YOLO_CONF      = 0.001
YOLO_IOU       = 0.7
BBOX_EXPAND    = 0.10

DISTANCE_BINS = [
    (0.0, 2.5,   "0-2.5"),
    (2.5, 5.0,   "2.5-5"),
    (5.0, 7.5,   "5-7.5"),
    (7.5, 10.0,  "7.5-10"),
    (10.0, 12.5, "10-12.5"),
    (12.5, 15.0, "12.5-15"),
    (15.0, np.inf,"15+"),
]

def build_distance_ranges(result_dir):
    """依據 DISTANCE_BINS 建立對應的資料夾 dict。"""
    distance_ranges = {}
    for _, _, label in DISTANCE_BINS:
        path = os.path.join(result_dir, label)
        os.makedirs(path, exist_ok=True)
        distance_ranges[label] = path
    return distance_ranges

def choose_distance_subfolder(avg_distance, distance_ranges):
    """根據 avg_distance 找到對應的區間資料夾。"""
    for lo, hi, label in DISTANCE_BINS:
        # 最後一個是 (15.0, inf, "15+")，用 >= lo 即可
        if np.isinf(hi):
            if avg_distance >= lo:
                return distance_ranges[label]
        else:
            if lo <= avg_distance < hi:
                return distance_ranges[label]
    # 理論上不會走到這裡，保險起見 fallback 到最後一個 bin
    return distance_ranges[DISTANCE_BINS[-1][2]]

def _infer_side_kp(
    kp_model,
    pil_crop,
    transform,
    crop_box,
    input_size,
    head_type="direct_regression",
    Nx=None,
    Ny=None,
):
    """
    對單側裁切圖做前處理→預測→轉回原圖座標系 (回傳 shape=(K,2) 的 numpy)。
    """
    x1, y1, x2, y2 = crop_box
    crop_w, crop_h = (x2 - x1), (y2 - y1)

    crop_tensor = transform(pil_crop).unsqueeze(0)   # [1,3,H,W]
    device = next(kp_model.parameters()).device
    crop_tensor = crop_tensor.to(device, non_blocking=True)

    with torch.inference_mode():
        outputs = kp_model(crop_tensor)
        coords = get_pred_coords(
            outputs,
            head_type=head_type,
            Nx=Nx,
            Ny=Ny,
            input_size=input_size,
        )   # [1, K, 2]
        pred = coords[0].detach().cpu().numpy()  # (K,2)

    # 反映射回原圖座標
    sx, sy = crop_w / input_size, crop_h / input_size
    pred_orig = np.empty_like(pred)
    pred_orig[:, 0] = pred[:, 0] * sx + x1
    pred_orig[:, 1] = pred[:, 1] * sy + y1
    return pred_orig

# 使用鏡像模型預測函式
def _hflip_kpts(kpts_6x2, input_size):
    """在 224x224 空間水平鏡像單側6點（numpy, shape=(6,2)）"""
    out = kpts_6x2.copy()
    out[:, 0] = (input_size - 1) - out[:, 0]
    return out

def _reorder_between_sides(kpts_6x2, from_side, to_side):
    """左↔右的單側6點索引重排；同側不動，跨側用 REORDER_6。"""
    if from_side == to_side:
        return kpts_6x2
    return kpts_6x2[REORDER_6, :]

def _infer_via_mirror(
    kp_model,
    pil_crop_src,
    transform,
    crop_box,
    model_side,
    target_side,
    input_size,
    head_type="direct_regression",
    Nx=None,
    Ny=None,
):
    """
    單模型鏡像推論：
      target_side crop → mirror → 用 model_side 模型推 → 在 input_size 空間反鏡像 →
      做 from=model_side → to=target_side 的索引重排 → 反投影回原圖。
    回傳 (K,2) numpy（target_side 順序）。
    """
    # 1) 目標側裁切 → 鏡像成模型側外觀
    pil_mirror = ImageOps.mirror(pil_crop_src)

    # 2) 模型在鏡像空間推論
    crop_tensor = transform(pil_mirror).unsqueeze(0)
    device = next(kp_model.parameters()).device
    crop_tensor = crop_tensor.to(device, non_blocking=True)

    with torch.inference_mode():
        outputs = kp_model(crop_tensor)
        coords = get_pred_coords(
            outputs,
            head_type=head_type,
            Nx=Nx,
            Ny=Ny,
            input_size=input_size,
        )   # [1, K, 2]
        pred_model_input = coords[0].detach().cpu().numpy()  # (K,2)

    # 3) input_size 空間反鏡像回未鏡像空間
    pred_unflipped = _hflip_kpts(pred_model_input, input_size)

    # 4) 索引重排：model_side → target_side
    pred_target_in = _reorder_between_sides(
        pred_unflipped,
        from_side=model_side,
        to_side=target_side,
    )

    # 5) 反投影回原圖座標
    x1, y1, x2, y2 = crop_box
    crop_w, crop_h = (x2 - x1), (y2 - y1)
    sx, sy = crop_w / input_size, crop_h / input_size

    pred_target_orig = np.empty_like(pred_target_in)
    pred_target_orig[:, 0] = pred_target_in[:, 0] * sx + x1
    pred_target_orig[:, 1] = pred_target_in[:, 1] * sy + y1
    return pred_target_orig

# Load annotations from CSV file
def load_annotations(annotation_path):
    keypoints = pd.read_csv(annotation_path, header=None).values.flatten()
    keypoints = [float(coord) for point in keypoints for coord in point.strip('"()').split(",")]
    return np.array(keypoints).reshape(-1, 2)

# Calculate average distance between predicted and original keypoints
def calculate_avg_distance(predicted_keypoints, original_keypoints):
    distances = np.linalg.norm(predicted_keypoints - original_keypoints, axis=1)
    avg_distance = np.mean(distances)
    return avg_distance

def extract_info_from_model_path(model_path):
    """
    支援的命名格式：

    SimCC 系列 (有 sr / sigma)，head_type 可以是：
      - simcc_1d
      - simcc_2d
      - simcc_2d_deconv

    例如：
      model_simcc_1d_sr2.0_sigma6.0_cropright_mirror_224_300_0.0001_32_best.pth
      model_simcc_2d_deconv_sr1.5_sigma3.0_cropleft_448_300_0.0001_32.pth

    direct_regression (沒有 sr / sigma)：
      model_direct_regression_cropleft_448_300_0.0001_32_best.pth
    """
    filename = os.path.basename(model_path)

    # 1) SimCC 系列：先抓 head_type + sr + sigma
    #   e.g. _simcc_sr2.0_..., _simcc_1d_sr2.0_..., _simcc_2d_deconv_sr2.0_...
    pattern_simcc = re.compile(
        r'_('
        r'simcc_2d_deconv|simcc_2d|simcc_1d'
        r')'                                  # group(1): head_type
        r'_sr([0-9eE\.\-]+)'                  # group(2): split_ratio (sr)
        r'_sigma([0-9eE\.\-]+)'               # group(3): sigma
        r'_crop(left|right)'                  # group(4): side
        r'(?:_mirror)?'                       # optional _mirror
        r'_(\d+)'                             # group(5): input_size
        r'_([0-9]+)'                          # group(6): epochs
        r'_([0-9eE\.\-]+)'                    # group(7): learning_rate
        r'_([0-9]+)'                          # group(8): batch_size
        r'(?:_best\.pth)?$'                   # optional _best.pth
    )

    m = pattern_simcc.search(filename)
    if m:
        head_type     = m.group(1)
        split_ratio   = float(m.group(2))
        sigma         = float(m.group(3))
        # side        = m.group(4)
        input_size    = int(m.group(5))
        epochs        = int(m.group(6))
        learning_rate = float(m.group(7))
        batch_size    = int(m.group(8))
        return head_type, input_size, epochs, learning_rate, batch_size, split_ratio, sigma

    # 2) direct_regression
    pattern_dr = re.compile(
        r'_(direct_regression)'               # group(1): head_type
        r'_crop(left|right)'                  # group(2): side
        r'(?:_mirror)?'                       # optional _mirror
        r'_(\d+)'                             # group(3): input_size
        r'_([0-9]+)'                          # group(4): epochs
        r'_([0-9eE\.\-]+)'                    # group(5): learning_rate
        r'_([0-9]+)'                          # group(6): batch_size
        r'(?:_best\.pth)?$'                   # optional _best.pth
    )

    m2 = pattern_dr.search(filename)
    if m2:
        head_type     = m2.group(1)          # "direct_regression"
        # side        = m2.group(2)
        input_size    = int(m2.group(3))
        epochs        = int(m2.group(4))
        learning_rate = float(m2.group(5))
        batch_size    = int(m2.group(6))
        split_ratio   = None
        sigma         = None
        return head_type, input_size, epochs, learning_rate, batch_size, split_ratio, sigma

    # 3) 都沒 match 就報錯
    raise ValueError(
        f"Model path format is invalid: {filename}\n"
        "Expected something like:\n"
        "  model_simcc_1d_sr2.0_sigma6.0_cropleft_448_300_0.0001_32[_best.pth]\n"
        "  model_simcc_2d_deconv_sr2.0_sigma6.0_cropleft_448_300_0.0001_32[_best.pth]\n"
        "  model_direct_regression_cropleft_448_300_0.0001_32[_best.pth]"
    )

def draw_comparison_figure(
    image, pred_kpts, gt_kpts, ai_pred, ai_gt,
    quadrants_pred, quadrants_gt,
    avg_distance, save_path, image_file):
    """
    建立左右對照圖：左圖使用預測點畫線，右圖使用 ground truth 畫線

    image: 原圖 (PIL or np.ndarray)
    pred_kpts: 預測的 scaled keypoints
    gt_kpts: ground truth keypoints
    ai_pred: (left, right) 使用預測點計算出來的 AI angle
    ai_gt: (left, right) 使用 ground truth 計算出來的 AI angle
    quadrants_pred: (left, right) 使用預測點計算出來的象限
    quadrants_gt: (left, right) 使用 ground truth 計算出來的象限
    avg_distance: 平均距離
    save_path: 要儲存的路徑
    image_file: 圖片名稱（用來命名）
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for i, (kpts, title, ai, quadrants) in enumerate([
        (pred_kpts, "Using Predicted Keypoints", ai_pred, quadrants_pred),
        (gt_kpts, "Using Ground Truth Keypoints", ai_gt, quadrants_gt)
    ]):
        ax = axes[i]
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        ax.scatter(pred_kpts[:, 0], pred_kpts[:, 1], c='yellow', s=4, label='Predicted')
        ax.scatter(gt_kpts[:, 0], gt_kpts[:, 1], c='red', s=4, label='Ground Truth')
        
        # # 畫關鍵點編號
        # for j, (x, y) in enumerate(kpts):
        #     ax.text(x + 2, y - 2, str(j + 1), color='white', fontsize=8, weight='bold')
        pts = {i: kpts[i] for i in [0, 2, 3, 5, 6, 8, 9, 11]}
        p1, p3, p4, p6, p7, p9, p10, p12 = pts[0], pts[2], pts[3], pts[5], pts[6], pts[8], pts[9], pts[11]

        # 畫 Roof Line
        ax.plot([p7[0], p9[0]], [p7[1], p9[1]], color='magenta', linewidth=1, label='Roof Line')
        ax.plot([p3[0], p1[0]], [p3[1], p1[1]], color='magenta', linewidth=1)
        
        # 畫 H-line（p3, p7）連線
        draw_hilgenreiner_line(ax, p3, p7)
        # 畫 P-line（從 p1 垂直 H-line、從 p9 垂直 H-line）
        draw_perpendicular_line(ax, p1, p3, p7, color='lime', label='P-line')
        draw_perpendicular_line(ax, p9, p3, p7, color='lime')
        # 畫 Diagonal lines from P-line/H-line intersection
        draw_diagonal_line(ax, p1, p3, p7, direction="left_down", color='orange', label='Diagonal Line')
        draw_diagonal_line(ax, p9, p3, p7, direction="right_down", color='orange')
        # 繪製 H-point
        draw_h_point(ax, kpts)
        
        # 解構象限
        left_q, right_q = quadrants
        
        ax.text(10, image.size[1] + 35, f'AI Left: {ai[0]:.1f}°  (Q{left_q})', color='magenta', fontsize=11)
        ax.text(10, image.size[1] + 85, f'AI Right: {ai[1]:.1f}°  (Q{right_q})', color='magenta', fontsize=11)

    # 百分比誤差（避免除以 0）
    def format_error(pred, gt):
        abs_error = abs(pred - gt)
        if gt != 0:
            rel_error = abs_error / gt * 100
            return f"{abs_error:.1f}° ({rel_error:.1f}%)"
        else:
            return f"{abs_error:.1f}° (--%)"

    # 計算圖片對角線長度（像素）
    diag_len = (image.size[0] ** 2 + image.size[1] ** 2) ** 0.5
    avg_dist_percent = avg_distance / diag_len * 100

    # 解構象限
    left_q_pred, right_q_pred = quadrants_pred
    left_q_gt, right_q_gt = quadrants_gt

    # 判斷是否正確
    left_match = "✓" if left_q_pred == left_q_gt else "✗"
    right_match = "✓" if right_q_pred == right_q_gt else "✗"

    # 輸出資訊到圖下方
    fig.text(
        0.5, -0.05,
        f"Avg Distance: {avg_distance:.2f} px ({avg_dist_percent:.2f}% of image diagonal)    "
        f"AI Error (L/R): {format_error(ai_pred[0], ai_gt[0])} / {format_error(ai_pred[1], ai_gt[1])}    "
        f"Quadrant (L/R): {left_q_gt} ({left_match}) / {right_q_gt} ({right_match})    ",
        ha='center', fontsize=12, color='blue'
    )

    axes[0].legend(loc='lower left')
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, f"{os.path.splitext(image_file)[0]}_compare.png"), bbox_inches='tight')
    plt.close()

# 修改後的 predict 函數片段 (整合 confusion matrix 統計)
def compute_and_save_confusion_matrices_with_accuracy(left_preds, left_gts, right_preds, right_gts, save_dir):
    labels = ['I', 'II', 'III', 'IV']

    # 左側
    cm_left = confusion_matrix(left_gts, left_preds, labels=labels)
    acc_left = accuracy_score(left_gts, left_preds)
    disp_left = ConfusionMatrixDisplay(confusion_matrix=cm_left, display_labels=labels)
    fig_left, ax_left = plt.subplots(figsize=(6, 5))
    disp_left.plot(ax=ax_left, cmap='Blues')
    ax_left.set_title(f'Left IHDI Quadrant\nAccuracy: {acc_left:.2%}')
    plt.tight_layout()
    fig_left.savefig(os.path.join(save_dir, "confusion_matrix_left.png"))
    plt.close(fig_left)

    # 右側
    cm_right = confusion_matrix(right_gts, right_preds, labels=labels)
    acc_right = accuracy_score(right_gts, right_preds)
    disp_right = ConfusionMatrixDisplay(confusion_matrix=cm_right, display_labels=labels)
    fig_right, ax_right = plt.subplots(figsize=(6, 5))
    disp_right.plot(ax=ax_right, cmap='Greens')
    ax_right.set_title(f'Right IHDI Quadrant\nAccuracy: {acc_right:.2%}')
    plt.tight_layout()
    fig_right.savefig(os.path.join(save_dir, "confusion_matrix_right.png"))
    plt.close(fig_right)

    # 全部（合併左+右）
    combined_preds = left_preds + right_preds
    combined_gts = left_gts + right_gts
    cm_all = confusion_matrix(combined_gts, combined_preds, labels=labels)
    acc_all = accuracy_score(combined_gts, combined_preds)
    disp_all = ConfusionMatrixDisplay(confusion_matrix=cm_all, display_labels=labels)
    fig_all, ax_all = plt.subplots(figsize=(6, 5))
    disp_all.plot(ax=ax_all, cmap='Purples')
    ax_all.set_title(f'IHDI Quadrant (All)\nAccuracy: {acc_all:.2%}')
    plt.tight_layout()
    fig_all.savefig(os.path.join(save_dir, "confusion_matrix_all.png"))
    plt.close(fig_all)

    return acc_left, acc_right, acc_all

def plot_ai_angle_scatter(gt_list, pred_list, side, save_path=None):
    x = np.array(gt_list)
    y = np.array(pred_list)

    # 回歸線 y = ax + b
    a, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = a * x_line + b

    # 評估指標
    pearsonr_corr, _ = pearsonr(x, y)
    spearmanr_corr, _ = spearmanr(x, y)
    kendalltau_corr, _ = kendalltau(x, y)
    r2 = r2_score(x, y)

    # 繪圖
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c='blue', alpha=0.6, label='Predicted vs. Ground Truth')

    # 畫理想線
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'g--', label='Ideal Line (y = x)')

    # 畫回歸線
    plt.plot(x_line, y_line, 'r--', label=f'Regression Line: y = {a:.2f}x + {b:.2f}')

    plt.title(f"{side} AI Angle Prediction\npearson r = {pearsonr_corr:.2f}, spearman r = {spearmanr_corr:.2f}, kendall tau = {kendalltau_corr:.2f}, R² = {r2:.2f}")
    plt.xlabel("Ground Truth AI Angle (°)")
    plt.ylabel("Predicted AI Angle (°)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_pixel_vs_angle_error(pixel_errors, ai_errors_avg, save_path=None):
    # 確保是 np.array
    x = np.array(pixel_errors)
    y = np.array(ai_errors_avg)

    # 計算統計指標
    r, _ = pearsonr(x, y)
    r2 = r2_score(x, y)

    # 線性回歸：y = a * x + b
    a, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = a * x_line + b

    # 繪圖
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color='orange', alpha=0.7, label='Avg AI Angle Error vs. Pixel Error')
    plt.plot(x_line, y_line, 'r--', label=f'Regression Line: y = {a:.2f}x + {b:.2f}')

    plt.xlabel("Avg Pixel Distance Error")
    plt.ylabel("Avg AI Angle Error (°)")
    plt.title(f"Pixel vs. Angle Error\nr = {r:.2f}, R² = {r2:.2f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def predict(model_name, kp_left_path, kp_right_path, yolo_weights, data_dir, output_dir):
    
    # 0) 以存在的模型路徑擷取紀錄資訊
    use_left  = (kp_left_path  is not None) and (str(kp_left_path).strip()  != "")
    use_right = (kp_right_path is not None) and (str(kp_right_path).strip() != "")
    assert use_left or use_right, "至少提供 --kp_left_path 或 --kp_right_path 其中之一"
    
    ref_model_path = kp_left_path if use_left else kp_right_path
    head_type, input_size, epochs, learning_rate, batch_size, split_ratio, sigma = extract_info_from_model_path(ref_model_path)
    print(f"Extracted model info:\n"
          f"  model_name   : {model_name}\n"
          f"  head_type    : {head_type}\n"
          f"  input_size   : {input_size}\n"
          f"  epochs       : {epochs}\n"
          f"  learning_rate: {learning_rate}\n"
          f"  batch_size   : {batch_size}\n"
          f"  split_ratio  : {split_ratio}\n"
          f"  sigma        : {sigma}\n"
    )
    
    if head_type in ["simcc_1d", "simcc_2d", "simcc_2d_deconv"]:
        assert split_ratio is not None and sigma is not None, "SimCC 模型需要有 split_ratio 與 sigma"
        Nx = int(input_size * split_ratio)
        Ny = int(input_size * split_ratio)
    else:
        Nx = Ny = None
    
    # 1) 載入 YOLO + KP 模型（左右可能有其一缺省）
    yolo_model = YOLO(yolo_weights)
    
    kp_left = kp_right = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points_count = POINTS_COUNT
    
    if use_left:
        kp_left = initialize_model(
            model_name,
            points_count,
            head_type=head_type,
            input_size=(input_size, input_size),
            Nx=Nx,
            Ny=Ny,
        )
        kp_left.load_state_dict(torch.load(kp_left_path, map_location="cpu"))
        kp_left.to(device)
        kp_left.eval()

    if use_right:
        kp_right = initialize_model(
            model_name,
            points_count,
            head_type=head_type,
            input_size=(input_size, input_size),
            Nx=Nx,
            Ny=Ny,
        )
        kp_right.load_state_dict(torch.load(kp_right_path, map_location="cpu"))
        kp_right.to(device)
        kp_right.eval()

    # 2) 建結果資料夾
    crop_side = "both-sides" if use_left and use_right else ("left-only" if use_left else "right-only")
    if head_type in ["simcc_1d", "simcc_2d", "simcc_2d_deconv"]:
        exp_name = f"{model_name}_{head_type}_sr{split_ratio}_sigma{sigma}_{crop_side}_{input_size}_{epochs}_{learning_rate}_{batch_size}"
    else:
        exp_name = f"{model_name}_{head_type}_{crop_side}_{input_size}_{epochs}_{learning_rate}_{batch_size}"
    result_dir = os.path.join(output_dir, exp_name)
    os.makedirs(result_dir, exist_ok=True)

    # 建立裁切輸出資料夾
    crops_dir = os.path.join(result_dir, "crops")
    crops_left_dir  = os.path.join(crops_dir, "left")
    crops_right_dir = os.path.join(crops_dir, "right")
    os.makedirs(crops_left_dir, exist_ok=True)   
    os.makedirs(crops_right_dir, exist_ok=True)

    # Create subdirectories for distance ranges
    distance_ranges = build_distance_ranges(result_dir)
    
    for path in distance_ranges.values():
        os.makedirs(path, exist_ok=True)

    transform = get_hip_base_transform(input_size)

    all_avg_distances = []  # To store all distances
    image_labels = []  # To store image indices (1, 2, 3, ...)
    image_counter = 1  # To keep track of the image index
    ai_errors_left = [] # To store AI angle errors (left)
    ai_errors_right = [] # To store AI angle errors (right)
    ai_left_gt_list = [] # To store left ground truth AI angles
    ai_left_pred_list = [] # To store left predicted AI angles
    ai_right_gt_list = [] # To store right ground truth AI angles
    ai_right_pred_list = [] # To store right predicted AI angles
    left_preds_all = [] # To store left predicted keypoints
    left_gts_all = [] # To store left ground truth keypoints
    right_preds_all = [] # To store right predicted keypoints
    right_gts_all = [] # To store right ground truth keypoints

    image_dir = os.path.join(data_dir, 'images')
    image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for image_counter, image_file in enumerate(image_files, start=1):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(data_dir, 'images', image_file)
            image_pil = Image.open(image_path).convert("RGB")  # YOLO 用 RGB 較穩
            W, H = image_pil.size
            
            # 3-1) YOLO 偵測 left/right
            box_left  = _detect_one(yolo_model, image_pil, cls_id=0, conf=0.001, iou=0.7)
            box_right = _detect_one(yolo_model, image_pil, cls_id=1, conf=0.001, iou=0.7)

            if (box_left is None) or (box_right is None):
                print(f"[WARN] {image_file}: missing detection (left={box_left is not None}, right={box_right is not None}), skip.")
                continue

            # 3-2) 調整 bbox → 裁切 → 推論單側 KPs → 還原到原圖
            # Left
            x1,y1,x2,y2 = box_left
            x1,y1,x2,y2 = _square_expand_clip(x1,y1,x2,y2, W,H, expand=0.10, keep_square=True)
            crop_left = image_pil.crop((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))).convert("L")
            left_crop_name = os.path.splitext(image_file)[0] + "_left_crop.jpg"
            crop_left.save(os.path.join(crops_left_dir, left_crop_name))

            if use_left:
                # 有左模型：直接推左
                pred_left_6 = _infer_side_kp(
                    kp_left,
                    crop_left,
                    transform,
                    (x1,y1,x2,y2),
                    input_size,
                    head_type=head_type,
                    Nx=Nx,
                    Ny=Ny,
                )
            else:
                # 沒左模型、只有右模型：把左鏡像成右 → 用右模型 → 反鏡像 + 重排回左
                pred_left_6 = _infer_via_mirror(
                    kp_right,
                    crop_left,
                    transform,
                    (x1,y1,x2,y2),
                    model_side="right",
                    target_side="left",
                    input_size=input_size,
                    head_type=head_type,
                    Nx=Nx,
                    Ny=Ny,
                )

            # Right
            x1,y1,x2,y2 = box_right
            x1,y1,x2,y2 = _square_expand_clip(x1,y1,x2,y2, W,H, expand=0.10, keep_square=True)
            crop_right = image_pil.crop((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))).convert("L")
            right_crop_name = os.path.splitext(image_file)[0] + "_right_crop.jpg"
            crop_right.save(os.path.join(crops_right_dir, right_crop_name))

            if use_right:
                # 有右模型：直接推右
                pred_right_6 = _infer_side_kp(
                    kp_right,
                    crop_right,
                    transform,
                    (x1,y1,x2,y2),
                    input_size,
                    head_type=head_type,
                    Nx=Nx,
                    Ny=Ny,
                )
            else:
                # 沒右模型、只有左模型：把右鏡像成左 → 用左模型 → 反鏡像 + 重排回右
                pred_right_6 = _infer_via_mirror(
                    kp_left,
                    crop_right,
                    transform,
                    (x1,y1,x2,y2),
                    model_side="left",
                    target_side="right",
                    input_size=input_size,
                    head_type=head_type,
                    Nx=Nx,
                    Ny=Ny,
                )

            
            # 合併成 12 點 (前6=Left, 後6=Right)
            scaled_keypoints = np.vstack([pred_left_6, pred_right_6])  # (12,2)

            # Load original keypoints
            annotation_path = os.path.join(data_dir, 'annotations', f"{os.path.splitext(image_file)[0]}.csv")
            original_keypoints = load_annotations(annotation_path)

            # Calculate avg distance
            avg_distance = calculate_avg_distance(scaled_keypoints, original_keypoints)
            all_avg_distances.append(avg_distance)
            image_labels.append(image_counter)
            
            # Calculate AI angles
            ai_left_pred, ai_right_pred = calculate_acetabular_index_angles(scaled_keypoints)
            ai_left_gt, ai_right_gt = calculate_acetabular_index_angles(original_keypoints)
            
            ai_left_pred_list.append(ai_left_pred)
            ai_right_pred_list.append(ai_right_pred)
            ai_left_gt_list.append(ai_left_gt)
            ai_right_gt_list.append(ai_right_gt)
            ai_errors_left.append(abs(ai_left_pred - ai_left_gt))
            ai_errors_right.append(abs(ai_right_pred - ai_right_gt))
            
            # Classify quadrants
            left_quadrants_pred, right_quadrants_pred = classify_quadrant_ihdi(scaled_keypoints)
            left_quadrants_gt, right_quadrants_gt = classify_quadrant_ihdi(original_keypoints)
            
            left_preds_all.append(left_quadrants_pred)
            left_gts_all.append(left_quadrants_gt)
            right_preds_all.append(right_quadrants_pred)
            right_gts_all.append(right_quadrants_gt)
            
            # Determine the subdirectory based on avg_distance
            subfolder = choose_distance_subfolder(avg_distance, distance_ranges)

            # 畫出左右對照圖
            draw_comparison_figure(
                image=image_pil.convert("L"),
                pred_kpts=scaled_keypoints,
                gt_kpts=original_keypoints,
                ai_pred=(ai_left_pred, ai_right_pred),
                ai_gt=(ai_left_gt, ai_right_gt),
                quadrants_pred=(left_quadrants_pred, right_quadrants_pred),
                quadrants_gt=(left_quadrants_gt, right_quadrants_gt),
                avg_distance=avg_distance,
                save_path=subfolder,
                image_file=image_file
            )

            # Save predicted keypoints
            # np.savetxt(os.path.join(subfolder, f"{os.path.splitext(image_file)[0]}_keypoints.txt"), scaled_keypoints, fmt="%.2f", delimiter=",")

            image_counter += 1  # Increment the image index

    # -------------------------------------------------------------- Plotting the average distances and AI angle errors --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(image_labels, all_avg_distances, label='Avg Distance per Image')
    
    # μ, σ（用不偏估計：ddof=1；若要母體標準差改成 ddof=0）
    mu_dist = float(np.mean(all_avg_distances))
    std_dist = float(np.std(all_avg_distances, ddof=1))
    # 參考線與±1σ區間
    add_sigma_guides(ax, mu=mu_dist, std=std_dist, mu_label=f'Overall Avg Dist(μ): {mu_dist:.2f}', label=f'μ ± 1σ (σ={std_dist:.2f})')
    
    # 右側 z-score 座標軸
    add_zscore_right_axis(ax, mu=mu_dist, std=std_dist)
    
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Avg Distance')
    ax.set_title('Average Distance per Image (with ±σ guides)')
    ax.set_xticks(image_labels)
    
    # 合理的圖例（去重複）
    handles, labels = ax.get_legend_handles_labels()
    # 移除重複label
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    avg_distances_path = os.path.join(result_dir, f"{model_name}_avg_distances.png")
    plt.savefig(avg_distances_path, bbox_inches='tight', dpi=150)
    plt.show()
    
    print(f"Overall average distance: {mu_dist:.2f} (±{std_dist:.2f})")

    fig, ax = plt.subplots(figsize=(16, 6))
    indices = np.arange(len(image_labels))
    bar_width = 0.4

    ax.bar(indices - bar_width/2, ai_errors_left,  width=bar_width, label='Left AI Error', color='magenta')
    ax.bar(indices + bar_width/2, ai_errors_right, width=bar_width, label='Right AI Error', color='crimson')

    # 左右平均
    avg_error_left  = float(np.mean(ai_errors_left))
    avg_error_right = float(np.mean(ai_errors_right))

    ax.axhline(avg_error_left,  linestyle='--', label=f'Avg Left Error: {avg_error_left:.2f}°', color='magenta')
    ax.axhline(avg_error_right, linestyle='--', label=f'Avg Right Error: {avg_error_right:.2f}°', color='crimson')

    # overall μ, σ（把左右合起來當同一個分佈）
    combined_errors = np.concatenate([np.asarray(ai_errors_left), np.asarray(ai_errors_right)], axis=0)
    mu_err  = float(np.mean(combined_errors))
    std_err = float(np.std(combined_errors, ddof=1))

    # overall 參考線與±1σ區間
    add_sigma_guides(ax, mu=mu_err, std=std_err, mu_label=f'Overall AI Error(μ): {mu_err:.2f}°', label=f'μ ± 1σ (σ={std_err:.2f})', mu_color='blue', color='red')

    # 右側 z-score 座標軸
    add_zscore_right_axis(ax, mu=mu_err, std=std_err)

    ax.set_xlabel('Image Index')
    ax.set_ylabel('AI Angle Error (°)')
    ax.set_title('Acetabular Index Angle Errors (with ±σ guides)')
    ax.set_xticks(indices, image_labels)

    # 圖例去重複
    handles, labels = ax.get_legend_handles_labels()
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ai_error_chart_path = os.path.join(result_dir, f"{model_name}_AI_angle_errors.png")
    plt.savefig(ai_error_chart_path, bbox_inches='tight', dpi=150)
    plt.show()

    print(f"Overall AI angle error: {mu_err:.2f}° (±{std_err:.2f}°)")

    # 計算IHDI classification的混淆矩陣跟準確度
    acc_left, acc_right, acc_all = compute_and_save_confusion_matrices_with_accuracy(
        left_preds=left_preds_all,
        left_gts=left_gts_all,
        right_preds=right_preds_all,
        right_gts=right_gts_all,
        save_dir=result_dir
    )

    print(f"Left quadrant accuracy: {acc_left:.2%}")
    print(f"Right quadrant accuracy: {acc_right:.2%}")
    print(f"Overall quadrant accuracy: {acc_all:.2%}")
    
    # 計算 AI 角度的相關性
    # 左側
    r_left, _ = pearsonr(ai_left_gt_list, ai_left_pred_list)
    r2_left = r2_score(ai_left_gt_list, ai_left_pred_list)
    plot_ai_angle_scatter(ai_left_gt_list, ai_left_pred_list, side='Left', save_path=os.path.join(result_dir, "scatter_left_ai_angle.png"))

    # 右側
    r_right, _ = pearsonr(ai_right_gt_list, ai_right_pred_list)
    r2_right = r2_score(ai_right_gt_list, ai_right_pred_list)
    plot_ai_angle_scatter(ai_right_gt_list, ai_right_pred_list, side='Right', save_path=os.path.join(result_dir, "scatter_right_ai_angle.png"))
    
    # 整體
    ai_gt_list = np.concatenate([ai_left_gt_list, ai_right_gt_list])
    ai_pred_list = np.concatenate([ai_left_pred_list, ai_right_pred_list])
    r_all, _ = pearsonr(ai_gt_list, ai_pred_list)
    r2_all = r2_score(ai_gt_list, ai_pred_list)
    plot_ai_angle_scatter(ai_gt_list, ai_pred_list, side='Overall', save_path=os.path.join(result_dir, "scatter_overall_ai_angle.png"))
    
    print(f"Left AI angle correlation: r = {r_left:.2f}, r² = {r2_left:.2f}")
    print(f"Right AI angle correlation: r = {r_right:.2f}, r² = {r2_right:.2f}")
    print(f"Overall AI angle correlation: r = {r_all:.2f}, r² = {r2_all:.2f}")
    
    # 計算平均像素距離誤差與 AI 角度誤差的相關性
    ai_errors_avg = [(l + r) / 2 for l, r in zip(ai_errors_left, ai_errors_right)]
    plot_pixel_vs_angle_error(
        pixel_errors=all_avg_distances,
        ai_errors_avg = ai_errors_avg,
        save_path=os.path.join(result_dir, "scatter_pixel_vs_angle_error.png")
    )
    r_pixel, _ = pearsonr(all_avg_distances, ai_errors_avg)
    r2_pixel = r2_score(all_avg_distances, ai_errors_avg)
    print(f"Pixel distance error vs AI angle error correlation: r = {r_pixel:.2f}, r² = {r2_pixel:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="efficientnet | resnet | vgg")
    parser.add_argument("--kp_left_path",  type=str, default="", help="left-side KP model (.pth)")
    parser.add_argument("--kp_right_path", type=str, default="", help="right-side KP model (.pth)")
    parser.add_argument("--yolo_weights", type=str, required=True, help="YOLO weights (e.g., best.pt)")
    parser.add_argument("--data", type=str, required=True, help="data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="output directory")
    args = parser.parse_args()

    predict(
        args.model_name,
        args.kp_left_path,
        args.kp_right_path,
        args.yolo_weights,
        args.data,
        args.output_dir
    )

# 單側模型預測
# python3 predict_hip_crop_keypoints.py --model_name convnext_small_fpn1234concat --kp_left_path results/25_simcc/convnext_small_fpn1234concat_simcc_2d_sr3.0_sigma7.0_cropleft_mirror_224_200_0.0001_32_best.pth --yolo_weights weights/yolo12s.pt --data "data/test" --output_dir "results"