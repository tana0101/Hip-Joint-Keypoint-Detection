import os
import argparse
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
    r2_score,
)
from scipy.stats import pearsonr, spearmanr, kendalltau
from ultralytics import YOLO

from datasets.hip_crop_keypoints import DATASET_CONFIGS_BY_COUNT # 鏡像重排
from datasets.transforms import get_hip_base_transform
from utils.detection import _detect_one, _square_expand_clip
from utils.keypoints import get_pred_coords
from utils.keypoint_metrics import calculate_icc
from models.model import initialize_model
from utils.hip_geometry import (
    calculate_acetabular_index_angles,
    classify_quadrant_ihdi,
    draw_hilgenreiner_line,
    draw_perpendicular_line,
    draw_diagonal_line,
    draw_h_point,
    unify_keypoints_format,
)
from utils.plots import add_sigma_guides, add_zscore_right_axis

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

# outlier thresholds
PIX_TH = 10.0     # pixel distance threshold
ANG_TH = 8.0      # degree threshold

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
def _hflip_kpts(kpts, input_size):
    """在 input_size 空間水平翻轉 keypoints。"""
    out = kpts.copy()
    out[:, 0] = (input_size - 1) - out[:, 0]
    return out

def _reorder_between_sides(kpts, from_side, to_side):
    """
    左↔右的單側點索引重排。
    直接查閱 DATASET_CONFIGS_BY_COUNT，實現單一維護。
    """
    if from_side == to_side:
        return kpts
    
    # 根據目前的裁切點數 (e.g., 6) 推算總點數 (e.g., 12)
    num_crop_points = kpts.shape[0]
    total_points = num_crop_points * 2
    
    if total_points in DATASET_CONFIGS_BY_COUNT:
        # 從配置中讀取重排規則
        reorder_idx = DATASET_CONFIGS_BY_COUNT[total_points]["mirror_reorder"]
        # 防呆檢查：確保長度一致
        if len(reorder_idx) == num_crop_points:
            return kpts[reorder_idx, :]
        else:
            print(f"[Warn] Reorder config length {len(reorder_idx)} != points {num_crop_points}. Skip reorder.")
            return kpts
    else:
        # 若無配置（例如自訂了非常規點數），預設不重排並警告
        print(f"[Warn] No reorder config found for {total_points} total points. Keeping original order.")
        return kpts

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

def calc_point_dists(pred_kpts: np.ndarray, gt_kpts: np.ndarray) -> np.ndarray:
    """
    pred_kpts, gt_kpts: (12, 2)
    return: per-point euclidean distance, shape (12,)
    """
    diff = pred_kpts.astype(np.float32) - gt_kpts.astype(np.float32)
    return np.sqrt((diff ** 2).sum(axis=1))

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
    avg_distance, save_path, image_file,
    raw_pred=None, raw_gt=None):
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
    
    # 確保有原始點資料，若無則退化使用 Unified keypoints
    r_pred = raw_pred if raw_pred is not None else pred_kpts
    r_gt = raw_gt if raw_gt is not None else gt_kpts
    
    # 第一張 (i=0): 標題顯示 Pred 資訊，畫 Pred 的幾何線
    # 第二張 (i=1): 標題顯示 GT 資訊，畫 GT 的幾何線
    # 但「點 (Scatter)」兩張圖都會同時畫，以便對照
    plot_configs = [
        (pred_kpts, "Predicted Geometry", ai_pred, quadrants_pred),
        (gt_kpts,   "Ground Truth Geometry", ai_gt, quadrants_gt)
    ]
    
    for i, (kpts_lines, title, ai, quadrants) in enumerate(plot_configs):
        ax = axes[i]
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        
        # -------------------------------------------------------
        # 同時繪製 GT (紅) 與 Pred (黃) 的點，並加上 Label
        # -------------------------------------------------------
        # 畫 GT 點
        ax.scatter(r_gt[:, 0], r_gt[:, 1], c='red', s=10, marker='o', label='GT')
        # 畫 Pred 點
        ax.scatter(r_pred[:, 0], r_pred[:, 1], c='yellow', s=10, marker='o', label='Pred')
        
        # -------------------------------------------------------
        # 幾何線條 (Line) 依據子圖不同而畫 Pred 或 GT
        # -------------------------------------------------------
        # 這裡使用 kpts_lines (Unified 12點格式) 來畫線
        pts = {idx: kpts_lines[idx] for idx in [0, 2, 3, 5, 6, 8, 9, 11]}
        p1, p3, p7, p9 = pts[0], pts[2], pts[6], pts[8]
        
        ax.plot([p7[0], p9[0]], [p7[1], p9[1]], color='magenta', linewidth=1, label='Roof Line')
        ax.plot([p3[0], p1[0]], [p3[1], p1[1]], color='magenta', linewidth=1)
        draw_hilgenreiner_line(ax, p3, p7)
        draw_perpendicular_line(ax, p1, p3, p7, color='lime', label='P-line')
        draw_perpendicular_line(ax, p9, p3, p7, color='lime')
        draw_diagonal_line(ax, p1, p3, p7, direction="left_down", color='orange', label='Diagonal')
        draw_diagonal_line(ax, p9, p3, p7, direction="right_down", color='orange')
        draw_h_point(ax, kpts_lines)
        
        # 顯示 AI 角度文字
        left_q, right_q = quadrants
        ax.text(10, image.size[1] + 35, f'AI Left: {ai[0]:.1f}°  (Q{left_q})', color='magenta', fontsize=11)
        ax.text(10, image.size[1] + 85, f'AI Right: {ai[1]:.1f}°  (Q{right_q})', color='magenta', fontsize=11)
        
        # 只在第一張圖顯示圖例，避免遮擋
        if i == 0:
            ax.legend(loc='lower left', fontsize=8)

    # 底部資訊欄
    diag_len = (image.size[0] ** 2 + image.size[1] ** 2) ** 0.5
    avg_dist_percent = avg_distance / diag_len * 100
    left_match = "✓" if quadrants_pred[0] == quadrants_gt[0] else "✗"
    right_match = "✓" if quadrants_pred[1] == quadrants_gt[1] else "✗"
    
    def fmt_err(p, g):
        diff = abs(p - g)
        return f"{diff:.1f}° ({diff/g*100:.1f}%)" if g!=0 else f"{diff:.1f}°"

    fig.text(0.5, -0.05, 
             f"Avg Dist: {avg_distance:.2f} px ({avg_dist_percent:.2f}%)    "
             f"AI Err(L/R): {fmt_err(ai_pred[0], ai_gt[0])} / {fmt_err(ai_pred[1], ai_gt[1])}    "
             f"Quad: {quadrants_gt[0]}({left_match}) / {quadrants_gt[1]}({right_match})", 
             ha='center', fontsize=12, color='blue')
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, f"{os.path.splitext(image_file)[0]}_compare.png"), bbox_inches='tight')
    plt.close()

def compute_and_save_confusion_matrices_with_metrics(left_preds, left_gts, right_preds, right_gts, save_dir):
    labels = ['I', 'II', 'III', 'IV']
    
    # 準備數據分組
    data_groups = {
        'left': (left_preds, left_gts, 'Left IHDI', 'Blues'),
        'right': (right_preds, right_gts, 'Right IHDI', 'Greens'),
        'all': (left_preds + right_preds, left_gts + right_gts, 'IHDI (All)', 'Purples')
    }
    
    results = {}

    for name, (preds, gts, title, cmap) in data_groups.items():
        # 1. 計算各項指標
        acc = accuracy_score(gts, preds)
        
        # 使用 weighted average 來處理多類別 (Multiclass)
        # zero_division=0 避免當某類別預測數為0時報錯
        p, r, f1, _ = precision_recall_fscore_support(gts, preds, labels=labels, average='weighted', zero_division=0)
        
        # 存入結果字典
        results[f'acc_{name}'] = acc
        results[f'prec_{name}'] = p
        results[f'rec_{name}'] = r
        results[f'f1_{name}'] = f1

        # 2. 繪製混淆矩陣 (標題增加 F1 分數展示)
        cm = confusion_matrix(gts, preds, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap=cmap)
        ax.set_title(f'{title}\nAcc: {acc:.2%} | F1: {f1:.2%} | Prec: {p:.2%} | Rec: {r:.2%}')
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"confusion_matrix_{name}.png"))
        plt.close(fig)

    return results

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
    
    # ---- 新增: 呼叫 ICC 計算 ----
    icc_val = calculate_icc(x, y)

    # 繪圖
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c='blue', alpha=0.6, label='Predicted vs. GT')

    # 畫理想線
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'g--', label='Ideal (y=x)')

    # 畫回歸線
    plt.plot(x_line, y_line, 'r--', label=f'Reg: y={a:.2f}x+{b:.2f}')

    # 更新標題，加入 ICC
    plt.title(
        f"{side} AI Angle Prediction\n"
        f"R={pearsonr_corr:.2f}, ICC={icc_val:.2f}, R²={r2:.2f}\n"
        f"(Spearman={spearmanr_corr:.2f}, Kendall={kendalltau_corr:.2f})"
    )
    plt.xlabel("Ground Truth AI Angle (°)")
    plt.ylabel("Predicted AI Angle (°)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

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

def predict(model_name, kp_left_path, kp_right_path, yolo_weights, data_dir, output_dir, fold_index=None):
    
    # 0. 自動判斷資料集格式
    ann_dir = os.path.join(data_dir, 'annotations')
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.csv')]
    if not ann_files: raise ValueError(f"No CSV annotations found in {ann_dir}")
    sample_kpts = load_annotations(os.path.join(ann_dir, ann_files[0]))
    total_points = sample_kpts.shape[0]
    points_per_side = total_points // 2
    print(f"Detected dataset: {total_points} total points ({points_per_side} per side).")
    
    # 1. 檢查並載入模型
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
    
    # 2. 載入模型
    yolo_model = YOLO(yolo_weights)
    
    kp_left = kp_right = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_left:
        kp_left = initialize_model(model_name, points_per_side, head_type, (input_size, input_size), Nx, Ny)
        kp_left.load_state_dict(torch.load(kp_left_path, map_location="cpu"))
        kp_left.to(device).eval()
    if use_right:
        kp_right = initialize_model(model_name, points_per_side, head_type, (input_size, input_size), Nx, Ny)
        kp_right.load_state_dict(torch.load(kp_right_path, map_location="cpu"))
        kp_right.to(device).eval()

    # 3. 建立輸出資料夾
    crop_side = "both" if (use_left and use_right) else ("left" if use_left else "right")
    exp_suffix = f"_{total_points}pt"
    if head_type.startswith("simcc"): exp_name = f"{model_name}_{head_type}_sr{split_ratio}_{crop_side}{exp_suffix}"
    else: exp_name = f"{model_name}_{head_type}_{crop_side}{exp_suffix}"
    
    result_dir = output_dir if fold_index is not None else os.path.join(output_dir, exp_name)
    os.makedirs(result_dir, exist_ok=True)
    for d in ["left", "right"]: os.makedirs(os.path.join(result_dir, "crops", d), exist_ok=True)
    dist_ranges = build_distance_ranges(result_dir)
    transform = get_hip_base_transform(input_size)

    # Storage Lists (for return dictionary)
    all_avg_distances = []
    image_labels = []
    
    ai_errors_left, ai_errors_right = [], []
    ai_left_gt_list, ai_left_pred_list = [], []
    ai_right_gt_list, ai_right_pred_list = [], []
    
    left_preds_all, left_gts_all = [], []
    right_preds_all, right_gts_all = [], []
    
    pixel_outlier_records = []
    angle_outlier_records = []
    all_outlier_records   = []
    all_outlier_files = []

    image_files = sorted(f for f in os.listdir(os.path.join(data_dir, 'images')) if f.lower().endswith(('.jpg', '.png')))

    # 4. 逐圖處理
    for idx, fname in enumerate(image_files, 1):
        img_path = os.path.join(data_dir, 'images', fname)
        ann_path = os.path.join(data_dir, 'annotations', os.path.splitext(fname)[0]+".csv")
        if not os.path.exists(ann_path): continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        box_l = _detect_one(yolo_model, img, YOLO_LEFT_CLS, YOLO_CONF, YOLO_IOU)
        box_r = _detect_one(yolo_model, img, YOLO_RIGHT_CLS, YOLO_CONF, YOLO_IOU)
        if not box_l or not box_r:
            print(f"[Skip] {fname} detection failed."); continue

        # Infer Left
        xl, yl, xr, yr = _square_expand_clip(*box_l, W, H, BBOX_EXPAND, True)
        crop_l = img.crop((int(xl), int(yl), int(xr), int(yr))).convert("L")
        crop_l.save(os.path.join(result_dir, "crops", "left", f"{os.path.splitext(fname)[0]}_left.jpg"))
        
        if use_left: pred_l_raw = _infer_side_kp(kp_left, crop_l, transform, (xl,yl,xr,yr), input_size, head_type, Nx, Ny)
        else: pred_l_raw = _infer_via_mirror(kp_right, crop_l, transform, (xl,yl,xr,yr), "right", "left", input_size, head_type, Nx, Ny)

        # Infer Right
        xl, yl, xr, yr = _square_expand_clip(*box_r, W, H, BBOX_EXPAND, True)
        crop_r = img.crop((int(xl), int(yl), int(xr), int(yr))).convert("L")
        crop_r.save(os.path.join(result_dir, "crops", "right", f"{os.path.splitext(fname)[0]}_right.jpg"))
        
        if use_right: pred_r_raw = _infer_side_kp(kp_right, crop_r, transform, (xl,yl,xr,yr), input_size, head_type, Nx, Ny)
        else: pred_r_raw = _infer_via_mirror(kp_left, crop_r, transform, (xl,yl,xr,yr), "left", "right", input_size, head_type, Nx, Ny)

        # --- Metric Collection ---
        kps_pred_raw = np.vstack([pred_l_raw, pred_r_raw])
        kps_gt_raw = load_annotations(ann_path)
        
        # 1. Avg Distance (Based on Raw Points)
        dist = calculate_avg_distance(kps_pred_raw, kps_gt_raw)
        all_avg_distances.append(dist)
        image_labels.append(idx)

        # 2. Geometry (Unify to 12 points first)
        kps_u_pred = unify_keypoints_format(kps_pred_raw)
        kps_u_gt = unify_keypoints_format(kps_gt_raw)
        
        ail_p, air_p = calculate_acetabular_index_angles(kps_u_pred)
        ail_g, air_g = calculate_acetabular_index_angles(kps_u_gt)
        ql_p, qr_p = classify_quadrant_ihdi(kps_u_pred)
        ql_g, qr_g = classify_quadrant_ihdi(kps_u_gt)
        
        ai_left_pred_list.append(ail_p); ai_right_pred_list.append(air_p)
        ai_left_gt_list.append(ail_g);   ai_right_gt_list.append(air_g)
        ai_errors_left.append(abs(ail_p - ail_g))
        ai_errors_right.append(abs(air_p - air_g))
        
        left_preds_all.append(ql_p); right_preds_all.append(qr_p)
        left_gts_all.append(ql_g);   right_gts_all.append(qr_g)

        # Draw Comparison
        draw_comparison_figure(
            image=img.convert("L"),
            pred_kpts=kps_u_pred,   # 傳入 12 點 (畫線用)
            gt_kpts=kps_u_gt,       # 傳入 12 點 (畫線用)
            ai_pred=(ail_p, air_p),
            ai_gt=(ail_g, air_g),
            quadrants_pred=(ql_p, qr_p),
            quadrants_gt=(ql_g, qr_g),
            avg_distance=dist,
            save_path=choose_distance_subfolder(dist, dist_ranges),
            image_file=fname,
            raw_pred=kps_pred_raw,  # 傳入原始預測 (例如 8 點)
            raw_gt=kps_gt_raw       # 傳入原始 GT (例如 8 點)
        )

        # Outlier Detection
        point_dists = calc_point_dists(kps_pred_raw, kps_gt_raw)
        mid = len(point_dists)//2
        pix_l, pix_r = np.mean(point_dists[:mid]), np.mean(point_dists[mid:])
        err_ail, err_air = abs(ail_p - ail_g), abs(air_p - air_g)
        
        is_pix = (pix_l > PIX_TH or pix_r > PIX_TH)
        is_ang = (err_ail > ANG_TH or err_air > ANG_TH)
        
        if is_pix:
            reason = f"{fname} L_pix:{pix_l:.2f} R_pix:{pix_r:.2f}"
            pixel_outlier_records.append(reason)
            all_outlier_records.append("pixel " + reason)
        if is_ang:
            reason = f"{fname} L_AI:{err_ail:.2f} R_AI:{err_air:.2f}"
            angle_outlier_records.append(reason)
            all_outlier_records.append("angle " + reason)
        if is_pix or is_ang:
            all_outlier_files.append(os.path.splitext(fname)[0])

    # -------------------------------------------------------------- Plotting the average distances and AI angle errors --------------------------------------------------------------
    # 5. 統計與繪圖

    # 設定 X 軸刻度顯示間隔
    tick_step = 50
    # 建立 X 軸座標索引
    indices = np.arange(len(image_labels))
    
    # 準備要顯示的刻度位置與標籤
    # 每隔 tick_step 取一個值
    target_ticks = indices[::tick_step]
    
    # 對應的 Labels (轉換為 list 確保相容性)
    target_labels = [str(image_labels[i]) for i in target_ticks]

    # -------------------------------------------------------------
    # 5-1. Avg Distance Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # [修改] 改用 indices 畫圖，確保與 tick 控制邏輯一致
    ax.bar(indices, all_avg_distances, label='Avg Distance per Image')
    
    # μ, σ
    mu_dist = float(np.mean(all_avg_distances))
    std_dist = float(np.std(all_avg_distances, ddof=1))
    
    # 參考線
    add_sigma_guides(ax, mu=mu_dist, std=std_dist, 
                     mu_label=f'Overall Avg Dist(μ): {mu_dist:.2f}', 
                     label=f'μ ± 1σ (σ={std_dist:.2f})')
    add_zscore_right_axis(ax, mu=mu_dist, std=std_dist)
    
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Avg Distance')
    ax.set_title(f"Average Distance per Image (mu={mu_dist:.2f}, sigma={std_dist:.2f})")
    
    # [修改] 套用間隔設定
    ax.set_xticks(target_ticks)
    ax.set_xticklabels(target_labels, rotation=0, fontsize=10) # 間隔變大後，rotation 可以改回 0 度比較好讀
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.savefig(os.path.join(result_dir, "avg_dists.png"), bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 2. AI Angle Error per Image Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 6))
    
    bar_width = 0.4
    ax.bar(indices - bar_width/2, ai_errors_left,  width=bar_width, label='Left AI Error', color='magenta')
    ax.bar(indices + bar_width/2, ai_errors_right, width=bar_width, label='Right AI Error', color='crimson')

    # 統計線
    avg_error_left  = float(np.mean(ai_errors_left))
    avg_error_right = float(np.mean(ai_errors_right))
    ax.axhline(avg_error_left,  linestyle='--', label=f'Avg Left Error: {avg_error_left:.2f}°', color='magenta')
    ax.axhline(avg_error_right, linestyle='--', label=f'Avg Right Error: {avg_error_right:.2f}°', color='crimson')

    combined_errors = np.concatenate([np.asarray(ai_errors_left), np.asarray(ai_errors_right)], axis=0)
    mu_err  = float(np.mean(combined_errors))
    std_err = float(np.std(combined_errors, ddof=1))

    add_sigma_guides(ax, mu=mu_err, std=std_err, 
                     mu_label=f'Overall AI Error(μ): {mu_err:.2f}°', 
                     label=f'μ ± 1σ (σ={std_err:.2f})', 
                     mu_color='blue', color='red')
    add_zscore_right_axis(ax, mu=mu_err, std=std_err)

    ax.set_xlabel('Image Index')
    ax.set_ylabel('AI Angle Error (°)')
    ax.set_title(f'AI Angle Errors per Image (mu={mu_err:.2f}°)')
    
    # [修改] 套用間隔設定 (與上面相同)
    ax.set_xticks(target_ticks)
    ax.set_xticklabels(target_labels, rotation=0, fontsize=10) 

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.savefig(os.path.join(result_dir, "AI_angle_errors.png"), bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------
    # 5-3. Confusion Matrices
    # -------------------------------------------------------------
    cls_metrics = compute_and_save_confusion_matrices_with_metrics(
        left_preds_all, left_gts_all, 
        right_preds_all, right_gts_all, 
        result_dir
    )
    
    # -------------------------------------------------------------
    # 5-4. AI Angle Scatter Plots (邏輯區塊)
    # -------------------------------------------------------------
    ai_gt_all = np.concatenate([ai_left_gt_list, ai_right_gt_list])
    ai_pred_all = np.concatenate([ai_left_pred_list, ai_right_pred_list])

    if len(ai_left_gt_list) > 1:
        plot_ai_angle_scatter(ai_left_gt_list, ai_left_pred_list, 'Left', os.path.join(result_dir, "scatter_ai_left.png"))
        plot_ai_angle_scatter(ai_right_gt_list, ai_right_pred_list, 'Right', os.path.join(result_dir, "scatter_ai_right.png"))
    plot_ai_angle_scatter(ai_gt_all, ai_pred_all, "Overall", os.path.join(result_dir, "scatter_ai_all.png"))
    
    # -------------------------------------------------------------
    # 5-5. Pixel vs. Angle Error Scatter Plot
    # -------------------------------------------------------------
    ai_errors_avg = [(l + r) / 2 for l, r in zip(ai_errors_left, ai_errors_right)]
    plot_pixel_vs_angle_error(all_avg_distances, ai_errors_avg, os.path.join(result_dir, "scatter_pix_vs_angle.png"))
    
    # ------------------------------------------------------------- Outliers Saving -------------------------------------------------------------
    with open(os.path.join(result_dir, "outliers_pixel.txt"), "w") as f: f.write("\n".join(pixel_outlier_records))
    with open(os.path.join(result_dir, "outliers_angle.txt"), "w") as f: f.write("\n".join(angle_outlier_records))
    with open(os.path.join(result_dir, "outliers_all.txt"), "w") as f: f.write("\n".join(all_outlier_records))
    with open(os.path.join(result_dir, "outlier_files.txt"), "w") as f: f.write("\n".join(all_outlier_files))
    
    # ------------------------------------------------------------- Final Metrics Calculation -------------------------------------------------------------
    r_left, _ = pearsonr(ai_left_gt_list, ai_left_pred_list) if len(ai_left_gt_list) > 1 else (0,0)
    r2_left = r2_score(ai_left_gt_list, ai_left_pred_list) if len(ai_left_gt_list) > 1 else 0
    icc_left = calculate_icc(ai_left_gt_list, ai_left_pred_list) # 新增

    r_right, _ = pearsonr(ai_right_gt_list, ai_right_pred_list) if len(ai_right_gt_list) > 1 else (0,0)
    r2_right = r2_score(ai_right_gt_list, ai_right_pred_list) if len(ai_right_gt_list) > 1 else 0
    icc_right = calculate_icc(ai_right_gt_list, ai_right_pred_list) # 新增

    r_all, _ = pearsonr(ai_gt_all, ai_pred_all) if len(ai_gt_all) > 1 else (0,0)
    r2_all = r2_score(ai_gt_all, ai_pred_all) if len(ai_gt_all) > 1 else 0
    icc_all = calculate_icc(ai_gt_all, ai_pred_all) # 新增

    r_pixel, _ = pearsonr(all_avg_distances, ai_errors_avg) if len(all_avg_distances) > 1 else (0,0)
    r2_pixel = r2_score(all_avg_distances, ai_errors_avg) if len(all_avg_distances) > 1 else 0

    avg_error_left = float(np.mean(ai_errors_left))
    avg_error_right = float(np.mean(ai_errors_right))
    mu_ai_err = float(np.mean(ai_errors_left + ai_errors_right))
    std_ai_err = float(np.std(ai_errors_left + ai_errors_right, ddof=1))

    print(f"Done. Avg Dist: {mu_dist:.2f} ± {std_dist:.2f}, AI Err: {mu_ai_err:.2f} ± {std_ai_err:.2f}, IHDI Acc: {cls_metrics['acc_all']:.2%}, R_all: {r_all:.2f}, ICC_all: {icc_all:.2f}")

    metrics = {
        "exp_name": exp_name,
        "num_images": len(image_labels),

        "all_avg_distances": all_avg_distances,
        "ai_errors_left": ai_errors_left,
        "ai_errors_right": ai_errors_right,
        "ai_left_gt_list": ai_left_gt_list,
        "ai_left_pred_list": ai_left_pred_list,
        "ai_right_gt_list": ai_right_gt_list,
        "ai_right_pred_list": ai_right_pred_list,
        "left_quadrants_pred": left_preds_all,
        "left_quadrants_gt": left_gts_all,
        "right_quadrants_pred": right_preds_all,
        "right_quadrants_gt": right_gts_all,

        "mu_dist": mu_dist,
        "std_dist": std_dist,

        "mu_ai_error": mu_ai_err,
        "std_ai_error": std_ai_err,
        "avg_ai_error_left": avg_error_left,
        "avg_ai_error_right": avg_error_right,

        # --- Accuracy, Precision, Recall, F1 ---
        "acc_left": cls_metrics['acc_left'],
        "prec_left": cls_metrics['prec_left'],
        "rec_left": cls_metrics['rec_left'],
        "f1_left": cls_metrics['f1_left'],

        "acc_right": cls_metrics['acc_right'],
        "prec_right": cls_metrics['prec_right'],
        "rec_right": cls_metrics['rec_right'],
        "f1_right": cls_metrics['f1_right'],

        "acc_all": cls_metrics['acc_all'],
        "prec_all": cls_metrics['prec_all'],
        "rec_all": cls_metrics['rec_all'],
        "f1_all": cls_metrics['f1_all'],
        # -----------------------------------------------------

        "r_left": r_left,
        "r2_left": r2_left,
        "icc_left": icc_left,   

        "r_right": r_right,
        "r2_right": r2_right,
        "icc_right": icc_right, 

        "r_all": r_all,
        "r2_all": r2_all,
        "icc_all": icc_all,     

        "r_pixel": r_pixel,
        "r2_pixel": r2_pixel,
    }
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="efficientnet | resnet | vgg")
    parser.add_argument("--kp_left_path",  type=str, default="", help="left-side KP model (.pth)")
    parser.add_argument("--kp_right_path", type=str, default="", help="right-side KP model (.pth)")
    parser.add_argument("--yolo_weights", type=str, required=True, help="YOLO weights (e.g., best.pt)")
    parser.add_argument("--data", type=str, required=True, help="data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="output directory")
    parser.add_argument("--fold_index", type=int, default=None, help="fold index for k-fold cross-validation (optional)")
    args = parser.parse_args()

    predict(
        args.model_name,
        args.kp_left_path,
        args.kp_right_path,
        args.yolo_weights,
        args.data,
        args.output_dir,
        args.fold_index
    )

# 單側模型預測
# python3 predict_hip_crop_keypoints.py --model_name convnext_small_fpn1234concat --kp_left_path results/25_simcc/convnext_small_fpn1234concat_simcc_2d_sr3.0_sigma7.0_cropleft_mirror_224_200_0.0001_32_best.pth --yolo_weights models/yolo12s.pt --data "data/test" --output_dir "results"