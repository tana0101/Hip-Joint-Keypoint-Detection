import os
import argparse
import torch
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import csv
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    r2_score,
    cohen_kappa_score
)
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_rel, mannwhitneyu, wilcoxon, shapiro
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
    project_to_metric6
)
from utils.plots import add_sigma_guides, add_zscore_right_axis
from utils.evaluation import extract_info_from_model_path
from collections import OrderedDict

YOLO_LEFT_CLS  = 0
YOLO_RIGHT_CLS = 1
YOLO_CONF      = 0.001
YOLO_IOU       = 0.7
BBOX_EXPAND    = 0.05

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

# 如果需要以ground truth box為基準裁切，可以用這個函式從 det json 讀取
def _load_box_from_det_json(det_path, label):
    """
    det json format:
    {
      "image": "...jpg",
      "objects": [
        {"label": "LeftHip",  "points": [[x1,y1],[x2,y2]]},
        {"label": "RightHip", "points": [[x1,y1],[x2,y2]]}
      ]
    }

    return: (x1, y1, x2, y2) as float, or None
    """
    if not os.path.exists(det_path):
        return None

    with open(det_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    for obj in d.get("objects", []):
        if obj.get("label") != label:
            continue
        pts = obj.get("points", None)
        if not pts or len(pts) < 2:
            return None
        (x1, y1), (x2, y2) = pts[0], pts[1]

        # 保險：確保是 left-top / right-bottom
        xl = float(min(x1, x2))
        yl = float(min(y1, y2))
        xr = float(max(x1, x2))
        yr = float(max(y1, y2))

        return (xl, yl, xr, yr)

    return None

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
    pred_kpts = np.array(predicted_keypoints)
    gt_kpts = np.array(original_keypoints)
    
    if pred_kpts.shape == gt_kpts.shape: # 同資料的比對
        distances = np.linalg.norm(pred_kpts - gt_kpts, axis=1)
        avg_distance = np.mean(distances)
    else: # 不同資料集將採用關鍵的六點進行比對
        common_pred = project_to_metric6(pred_kpts)
        common_gt = project_to_metric6(gt_kpts)
        distances = np.linalg.norm(common_pred - common_gt, axis=1)
        avg_distance = np.mean(distances)
    
    return avg_distance

def calc_point_dists(pred_kpts: np.ndarray, gt_kpts: np.ndarray) -> np.ndarray:
    """
    計算每個關鍵點的歐式距離。
    
    - 如果格式相同 (8vs8 或 12vs12): 回傳該長度 (8 或 12) 的距離陣列。
    - 如果格式不同: 自動轉換為 6 點共同格式，回傳長度為 6 的距離陣列。
    
    return: shape (N,) where N is 6, 8, or 12
    """
    p = np.array(pred_kpts)
    g = np.array(gt_kpts)
    
    # 1. 如果形狀一致，進行完整比對
    if p.shape == g.shape:
        diff = p - g
    
    # 2. 如果形狀不一致，進行 6 點共同比對
    else:
        p_common = project_to_metric6(p)
        g_common = project_to_metric6(g)
        diff = p_common - g_common
        
    # 計算歐式距離 (比起手寫 sqrt(sum) 更快且易讀)
    dists = np.linalg.norm(diff, axis=1)
    
    return dists

def draw_comparison_figure(
    image, pred_kpts, gt_kpts, ai_pred, ai_gt,
    quadrants_pred, quadrants_gt,
    avg_distance, save_path, image_file,
    raw_pred=None, raw_gt=None,
    label_pred="Predicted", label_gt="Ground Truth"
):
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
        (pred_kpts, f"{label_pred} Geometry", ai_pred, quadrants_pred),
        (gt_kpts,   f"{label_gt} Geometry", ai_gt, quadrants_gt)
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
        ax.scatter(r_gt[:, 0], r_gt[:, 1], c='red', s=10, marker='o', label=label_gt)
        # 畫 Pred 點
        ax.scatter(r_pred[:, 0], r_pred[:, 1], c='yellow', s=10, marker='o', label=label_pred)
        
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
             f"AI Diff(L/R): {fmt_err(ai_pred[0], ai_gt[0])} / {fmt_err(ai_pred[1], ai_gt[1])}    "
             f"Quad: {quadrants_gt[0]}({left_match}) / {quadrants_gt[1]}({right_match})", 
             ha='center', fontsize=12, color='blue')
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, f"{os.path.splitext(image_file)[0]}_compare.png"), bbox_inches='tight')
    plt.close()

def compute_and_save_confusion_matrices_with_metrics(
    left_preds, left_gts, right_preds, right_gts, save_dir,
    label_pred="Predicted", label_gt="Ground Truth"
):
    # 確保儲存目錄存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 4 分類設定
    label_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
    labels_4class = ['I', 'II', 'III', 'IV']
    
    # 二元分類設定 (0: 正常 Grade I, 1: 異常 Grade II-IV)
    labels_2class = ['Normal (I)', 'Abnormal (II-IV)']
    
    data_groups = {
        'left': (left_preds, left_gts, 'Left IHDI', 'Blues'),
        'right': (right_preds, right_gts, 'Right IHDI', 'Greens'),
        'all': (left_preds + right_preds, left_gts + right_gts, 'IHDI (All)', 'Purples')
    }
    
    results = {}

    for name, (preds, gts, title, cmap) in data_groups.items():
        # 篩選有效的標籤對
        valid_pairs = [(p, g) for p, g in zip(preds, gts) if p in label_map and g in label_map]
        
        if not valid_pairs:
            print(f"Warning: No valid labels found for group {name}. Skipping...")
            continue
            
        v_preds, v_gts = zip(*valid_pairs) 
        
        # ==========================================
        # 階段一：4 分類 (分級評估 Grading)
        # ==========================================
        preds_numeric = [label_map[p] for p in v_preds]
        gts_numeric = [label_map[g] for g in v_gts]

        # 計算等級相關性指標
        wk = cohen_kappa_score(gts_numeric, preds_numeric, weights='quadratic')
        kt, _ = kendalltau(gts_numeric, preds_numeric)
        
        # 計算 4 分類基礎指標 (改用 macro)
        acc_4 = accuracy_score(v_gts, v_preds)
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(v_gts, v_preds, labels=labels_4class, average='macro', zero_division=0)
        
        # 取得各類別獨立的 F1-score
        _, _, f1_per_class, _ = precision_recall_fscore_support(v_gts, v_preds, labels=labels_4class, average=None, zero_division=0)

        # 寫入 4 分類 results
        results[f'4cls_Acc_{name}'] = acc_4
        results[f'4cls_Precision_{name}'] = p_mac
        results[f'4cls_Recall_{name}'] = r_mac
        results[f'4cls_Macro_F1_{name}'] = f1_mac
        results[f'4cls_QWK_{name}'] = wk
        results[f'4cls_Kendall_{name}'] = kt
        for i, lbl in enumerate(labels_4class):
            results[f'4cls_F1_Grade_{lbl}_{name}'] = f1_per_class[i]

        # 繪製 4x4 混淆矩陣
        cm_4 = confusion_matrix(v_gts, v_preds, labels=labels_4class)
        cm_norm_4 = confusion_matrix(v_gts, v_preds, labels=labels_4class, normalize='true')
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{title} (4-Class)\nAcc: {acc_4:.2%} | Macro F1: {f1_mac:.2%} | QWK: {wk:.3f} | Kendall: {kt:.3f}', fontsize=14)
        
        sns.heatmap(cm_4, annot=True, fmt='d', cmap=cmap, ax=axes[0], xticklabels=labels_4class, yticklabels=labels_4class)
        axes[0].set_title('Counts')
        axes[0].set_xlabel(label_pred)
        axes[0].set_ylabel(label_gt)
        
        sns.heatmap(cm_norm_4, annot=True, fmt='.2f', cmap=cmap, ax=axes[1], xticklabels=labels_4class, yticklabels=labels_4class, vmin=0, vmax=1)
        axes[1].set_title('Normalized (by True Label)')
        axes[1].set_xlabel(label_pred)
        axes[1].set_ylabel(label_gt)
        
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"CM_4Class_{name}.png"), dpi=300)
        plt.close(fig)

        # ==========================================
        # 階段二：二元分類 (篩檢評估 Screening, I vs II-IV)
        # ==========================================
        # 轉換標籤：I -> 0 (陰性), II,III,IV -> 1 (陽性)
        bin_gts = [0 if g == 'I' else 1 for g in v_gts]
        bin_preds = [0 if p == 'I' else 1 for p in v_preds]
        
        # 取得 TP, TN, FP, FN
        tn, fp, fn, tp = confusion_matrix(bin_gts, bin_preds, labels=[0, 1]).ravel()

        # 計算基礎醫學指標
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity (Recall)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
        acc_bin = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        # --- 建議新增的進階臨床指標 ---
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # PPV (Precision)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0   # NPV
        f1_bin = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0.0 # Binary F1

        # 寫入二元分類 results
        results[f'2cls_Acc_{name}'] = acc_bin
        results[f'2cls_Sensitivity_{name}'] = sens
        results[f'2cls_Specificity_{name}'] = spec
        results[f'2cls_PPV_{name}'] = ppv
        results[f'2cls_NPV_{name}'] = npv
        results[f'2cls_F1_{name}'] = f1_bin

        # 繪製 2x2 混淆矩陣
        cm_2 = confusion_matrix(bin_gts, bin_preds, labels=[0, 1])
        cm_norm_2 = confusion_matrix(bin_gts, bin_preds, labels=[0, 1], normalize='true')
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
        fig2.suptitle(f'{title} (Screening: I vs II-IV)\nAcc: {acc_bin:.2%} | Sens: {sens:.2%} | Spec: {spec:.2%}', fontsize=14)
        
        # 為了區分，二元分類改用熱力圖顏色稍微不同的色系 (加一點透明度或選用不同 colormap，這裡沿用但標籤不同)
        sns.heatmap(cm_2, annot=True, fmt='d', cmap='Oranges', ax=axes2[0], xticklabels=labels_2class, yticklabels=labels_2class)
        axes2[0].set_title('Counts')
        axes2[0].set_xlabel(label_pred)
        axes2[0].set_ylabel(label_gt)
        
        sns.heatmap(cm_norm_2, annot=True, fmt='.2f', cmap='Oranges', ax=axes2[1], xticklabels=labels_2class, yticklabels=labels_2class, vmin=0, vmax=1)
        axes2[1].set_title('Normalized (by True Label)')
        axes2[1].set_xlabel(label_pred)
        axes2[1].set_ylabel(label_gt)
        
        plt.tight_layout()
        fig2.savefig(os.path.join(save_dir, f"CM_2Class_{name}.png"), dpi=300)
        plt.close(fig2)

    # ==========================================
    # 階段三：輸出所有指標到 CSV 檔案
    # ==========================================
    csv_file_path = os.path.join(save_dir, 'evaluation_metrics_comprehensive.csv')
    try:
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric Name', 'Value'])
            for key, value in results.items():
                formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                writer.writerow([key, formatted_value])
        print(f"Metrics successfully saved to: {csv_file_path}")
    except Exception as e:
        print(f"Failed to save metrics to CSV: {e}")

    return results

def plot_avg_distances(image_labels, all_avg_distances, result_dir, tick_step=50, dpi=100):
    """
    繪製每張影像的平均距離長條圖。
    """
    # 建立 X 軸座標索引與刻度
    indices = np.arange(len(image_labels))
    target_ticks = indices[::tick_step]
    target_labels = [str(image_labels[i]) for i in target_ticks]

    fig, ax = plt.subplots(figsize=(16, 6), dpi=dpi)
    
    # 畫長條圖
    ax.bar(indices, all_avg_distances, label='Avg Distance per Image')
    
    # 計算 μ, σ
    mu_dist = float(np.mean(all_avg_distances))
    std_dist = float(np.std(all_avg_distances, ddof=1))
    
    # 參考線
    add_sigma_guides(ax, mu=mu_dist, std=std_dist, 
                     mu_label=f'Overall Avg Dist(μ): {mu_dist:.2f}', 
                     label=f'μ ± 1σ (σ={std_dist:.2f})')
    add_zscore_right_axis(ax, mu=mu_dist, std=std_dist)
    
    # 設定圖表標題與軸標籤
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Avg Distance')
    ax.set_title(f"Average Distance per Image (mu={mu_dist:.2f}, sigma={std_dist:.2f})")
    
    # 套用間隔設定
    ax.set_xticks(target_ticks)
    ax.set_xticklabels(target_labels, rotation=0, fontsize=10)
    
    # 處理圖例 (Legend) 去除重複項目
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    # 儲存與關閉
    save_path = os.path.join(result_dir, "avg_dists.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return mu_dist, std_dist


def plot_ai_angle_errors(image_labels, ai_errors_left, ai_errors_right, result_dir, tick_step=50, bar_width=0.4, dpi=100):
    """
    繪製每張影像的 AI 角度誤差長條圖 (左右腳)。
    """
    # 建立 X 軸座標索引與刻度
    indices = np.arange(len(image_labels))
    target_ticks = indices[::tick_step]
    target_labels = [str(image_labels[i]) for i in target_ticks]

    fig, ax = plt.subplots(figsize=(16, 6), dpi=dpi)

    # 分開畫左右腳，並偏移 X 軸位置避免重疊
    ax.bar(indices - bar_width/2, ai_errors_left,  width=bar_width, label='Left AI Error', color='magenta')
    ax.bar(indices + bar_width/2, ai_errors_right, width=bar_width, label='Right AI Error', color='crimson')

    # 左右腳各自的平均統計線
    avg_error_left  = float(np.mean(ai_errors_left))
    avg_error_right = float(np.mean(ai_errors_right))
    ax.axhline(avg_error_left,  linestyle='--', label=f'Avg Left Error: {avg_error_left:.2f}°', color='magenta')
    ax.axhline(avg_error_right, linestyle='--', label=f'Avg Right Error: {avg_error_right:.2f}°', color='crimson')

    # 計算整體的 μ, σ
    combined_errors = np.concatenate([np.asarray(ai_errors_left), np.asarray(ai_errors_right)], axis=0)
    mu_err  = float(np.mean(combined_errors))
    std_err = float(np.std(combined_errors, ddof=1))

    # 參考線
    add_sigma_guides(ax, mu=mu_err, std=std_err, 
                     mu_label=f'Overall AI Error(μ): {mu_err:.2f}°', 
                     label=f'μ ± 1σ (σ={std_err:.2f})', 
                     mu_color='blue', color='red')
    add_zscore_right_axis(ax, mu=mu_err, std=std_err)

    # 設定圖表標題與軸標籤
    ax.set_xlabel('Image Index')
    ax.set_ylabel('AI Angle Error (°)')
    ax.set_title(f'AI Angle Errors per Image (mu={mu_err:.2f}°)')
    
    # 套用間隔設定
    ax.set_xticks(target_ticks)
    ax.set_xticklabels(target_labels, rotation=0, fontsize=10) 

    # 處理圖例 (Legend) 去除重複項目
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # 儲存與關閉
    save_path = os.path.join(result_dir, "AI_angle_errors.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return avg_error_left, avg_error_right, mu_err, std_err

def plot_error_histogram_with_shapiro(
    errors_left, errors_right, result_dir,
    x_label="Signed Error (Pred - GT) (°)"
):
    """
    繪製 AI 角度誤差的直方圖 (Histogram)，並執行 Shapiro-Wilk 常態分佈檢定。
    注意：傳入的 errors 必須是帶正負號的真實誤差 (Pred - GT)。
    """
    # 1. 合併左右腳的誤差
    combined_errors = np.concatenate([np.asarray(errors_left), np.asarray(errors_right)], axis=0)
    
    # 2. 進行 Shapiro-Wilk 檢定
    stat, p_shapiro = shapiro(combined_errors)
    
    # 3. 計算平均值與標準差
    mu = float(np.mean(combined_errors))
    std = float(np.std(combined_errors, ddof=1))

    # 4. 繪製直方圖
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 使用 bins=30 將誤差分組，alpha 控制透明度
    counts, bins, patches = ax.hist(combined_errors, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 畫出平均值的垂直參考線
    ax.axvline(mu, color='red', linestyle='dashed', linewidth=2, label=f'Mean (μ): {mu:.2f}°')
    
    # 5. 設定標題 (動態顯示 Shapiro-Wilk 結果)
    normality_text = "Normal" if p_shapiro > 0.05 else "Non-Normal"
    title = (
        f"AI Angle Error Distribution (Histogram)\n"
        f"μ = {mu:.2f}°, σ = {std:.2f}°\n"
        f"Shapiro-Wilk p-value = {p_shapiro:.4f} ({normality_text})"
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Frequency (Number of Images)', fontsize=11)
    
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 6. 儲存與關閉
    save_path = os.path.join(result_dir, "error_histogram_shapiro.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    # 回傳 p-value 以供後續流程自動判斷 (要用 t-test 還是 Wilcoxon)
    return p_shapiro

def plot_ai_angle_scatter(
    gt_list, pred_list, side, save_path=None,
    label_pred="Predicted", label_gt="Ground Truth"
):
    x = np.array(gt_list)
    y = np.array(pred_list)

    # 如果資料點不足，直接回傳預設值並跳過繪圖
    if len(x) <= 1:
        prefix = side.lower()
        return {f"r_{prefix}": 0, f"r2_{prefix}": 0, f"icc_{prefix}": 0, 
                f"t_pval_{prefix}": 1.0, f"mw_pval_{prefix}": 1.0}

    # 回歸線 y = ax + b
    a, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = a * x_line + b

    # 評估相關性指標
    pearsonr_corr, _ = pearsonr(x, y)
    spearmanr_corr, _ = spearmanr(x, y)
    kendalltau_corr, _ = kendalltau(x, y)
    r2 = r2_score(x, y)
    
    # 計算 ICC
    icc_val = calculate_icc(x, y)

    # ==========================================
    # 新增的差異性統計檢定
    # ==========================================
    # 1. Paired Student's t-Test (成對 t 檢定)
    t_stat, t_pval = ttest_rel(x, y)

    # 2. Mann-Whitney U test (獨立樣本非參數檢定 - 依照您的需求加入)
    u_stat, mw_pval = mannwhitneyu(x, y)

    # 3. Wilcoxon Signed-Rank test (成對樣本非參數檢定 - 統計學上更推薦)
    # 加上 try-except 是為了防止 GT 和 Pred 完全一模一樣時 (差值全為0) 導致 ValueError
    try:
        w_stat, w_pval = wilcoxon(x, y)
    except ValueError:
        w_pval = 1.0 

    # 繪圖
    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, c='blue', alpha=0.6, label=f'{label_pred} vs. {label_gt}')
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'g--', label='Ideal (y=x)')
    plt.plot(x_line, y_line, 'r--', label=f'Reg: y={a:.2f}x+{b:.2f}')

    # 更新 Title，加入 p-value 資訊 (使用 P-val 表示)
    plt.title(
        f"{side} AI Angle Comparison\n"
        f"R={pearsonr_corr:.2f}, ICC={icc_val:.2f}, R²={r2:.2f}\n"
        f"Paired t-Test p={t_pval:.3f}, Mann-Whitney p={mw_pval:.3f}, Wilcoxon p={w_pval:.3f}"
    )
    plt.xlabel(f"{label_gt} AI Angle (°)")
    plt.ylabel(f"{label_pred} AI Angle (°)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    # 將 side 轉為小寫當作 key (例如: 'Left' -> 'left', 'All' -> 'all')
    prefix = side.lower()
    return {
        f"r_{prefix}": pearsonr_corr,
        f"r2_{prefix}": r2,
        f"icc_{prefix}": icc_val,
        f"t_pval_{prefix}": t_pval,      # 紀錄 Paired t-test p-value
        f"mw_pval_{prefix}": mw_pval,    # 紀錄 Mann-Whitney p-value
        f"wilcoxon_pval_{prefix}": w_pval # 紀錄 Wilcoxon p-value
    }

def plot_pixel_vs_angle_error(
    pixel_errors, ai_errors_avg, save_path=None,
    x_label="Avg Pixel Distance Error",          
    y_label="Avg AI Angle Error (°)",            
    title_prefix="Pixel vs. Angle Error",        
    legend_label="Avg AI Angle Error vs. Pixel Error" 
):
    x = np.array(pixel_errors)
    y = np.array(ai_errors_avg)

    if len(x) <= 1:
        return {"r_pixel": 0, "r2_pixel": 0}

    # 計算統計指標
    r, _ = pearsonr(x, y)
    r2 = r2_score(x, y)

    # 線性回歸：y = a * x + b
    a, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = a * x_line + b

    # 繪圖
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color='orange', alpha=0.7, label=legend_label)
    plt.plot(x_line, y_line, 'r--', label=f'Regression Line: y = {a:.2f}x + {b:.2f}')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title_prefix}\nr = {r:.2f}, R² = {r2:.2f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    # 回傳指定指標
    return {
        "r_pixel": r,
        "r2_pixel": r2
    }

def predict(model_name, kp_left_path, kp_right_path, yolo_weights, data_dir, output_dir, fold_index=None, using_gt_box=False, model_points=None):
    
    # 0. 自動判斷資料集格式
    ann_dir = os.path.join(data_dir, 'annotations')
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.csv')]
    if not ann_files: raise ValueError(f"No CSV annotations found in {ann_dir}")
    sample_kpts = load_annotations(os.path.join(ann_dir, ann_files[0]))
    total_points = sample_kpts.shape[0] # 資料集是 8 點還是 12 點

    if model_points is None:
        num_kpts = total_points
    else:
        num_kpts = model_points

    points_per_side = num_kpts // 2 
    print(f"Initializing Model with: {num_kpts} total points ({points_per_side} per side).")
    
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
    
    if head_type.startswith("simcc"): 
        exp_name = f"{model_name}_{head_type}_sr{split_ratio}_sigma{sigma}_{crop_side}{exp_suffix}"
    elif head_type == "heatmap":
        exp_name = f"{model_name}_{head_type}_sigma{sigma}_{crop_side}{exp_suffix}"
    else: 
        exp_name = f"{model_name}_{head_type}_{crop_side}{exp_suffix}"
    
    result_dir = output_dir if fold_index is not None else os.path.join(output_dir, exp_name)
    os.makedirs(result_dir, exist_ok=True)
    for d in ["left", "right"]: os.makedirs(os.path.join(result_dir, "crops", d), exist_ok=True)
    dist_ranges = build_distance_ranges(result_dir)
    transform = get_hip_base_transform(input_size)

    # Storage Lists (for return dictionary)
    all_avg_distances = []
    image_labels = []
    
    ai_errors_left, ai_errors_right = [], [] # 絕對誤差列表
    signed_errors_left, signed_errors_right = [], [] # 帶正負號的誤差列表 (Pred - GT)
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
        det_path = os.path.join(data_dir, 'detections', os.path.splitext(fname)[0] + ".json")
        if not os.path.exists(ann_path): continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        if using_gt_box:
            box_l = _load_box_from_det_json(det_path, "LeftHip")
            box_r = _load_box_from_det_json(det_path, "RightHip")
        else:
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
        # if dist >= 50:
        #     print(f"[Skip] {fname} distance too large: {dist:.2f}"); continue
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
        
        # 絕對誤差
        ai_errors_left.append(abs(ail_p - ail_g))
        ai_errors_right.append(abs(air_p - air_g))
        # 帶正負號的誤差 (Pred - GT)
        signed_errors_left.append(ail_p - ail_g)
        signed_errors_right.append(air_p - air_g)
        
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
    tick_step_val = 50

    # 1. 繪製平均距離圖
    mu_dist, std_dist = plot_avg_distances(
        image_labels=image_labels,
        all_avg_distances=all_avg_distances,
        result_dir=result_dir,
        tick_step=tick_step_val
    )
    
    # 2. 繪製 AI 角度誤差圖
    avg_error_left, avg_error_right, mu_ai_err, std_ai_err = plot_ai_angle_errors(
        image_labels=image_labels,
        ai_errors_left=ai_errors_left,
        ai_errors_right=ai_errors_right,
        result_dir=result_dir,
        tick_step=tick_step_val
    )
    
    # 3. 繪製誤差分佈直方圖與 Shapiro-Wilk 檢定
    p_val_shapiro = plot_error_histogram_with_shapiro(
        errors_left=signed_errors_left,   
        errors_right=signed_errors_right, 
        result_dir=result_dir
    )
    
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

    ai_left_metrics = plot_ai_angle_scatter(ai_left_gt_list, ai_left_pred_list, 'Left', os.path.join(result_dir, "scatter_ai_left.png"))
    ai_right_metrics = plot_ai_angle_scatter(ai_right_gt_list, ai_right_pred_list, 'Right', os.path.join(result_dir, "scatter_ai_right.png"))
    ai_all_metrics = plot_ai_angle_scatter(ai_gt_all, ai_pred_all, 'All', os.path.join(result_dir, "scatter_ai_all.png"))
    
    # -------------------------------------------------------------
    # 5-5. Pixel vs. Angle Error Scatter Plot
    # -------------------------------------------------------------
    ai_errors_avg = [(l + r) / 2 for l, r in zip(ai_errors_left, ai_errors_right)]
    pixel_vs_angle_metrics = plot_pixel_vs_angle_error(all_avg_distances, ai_errors_avg, os.path.join(result_dir, "scatter_pix_vs_angle.png"))
    
    # ------------------------------------------------------------- Outliers Saving -------------------------------------------------------------
    with open(os.path.join(result_dir, "outliers_pixel.txt"), "w") as f: f.write("\n".join(pixel_outlier_records))
    with open(os.path.join(result_dir, "outliers_angle.txt"), "w") as f: f.write("\n".join(angle_outlier_records))
    with open(os.path.join(result_dir, "outliers_all.txt"), "w") as f: f.write("\n".join(all_outlier_records))
    with open(os.path.join(result_dir, "outlier_files.txt"), "w") as f: f.write("\n".join(all_outlier_files))
    
    # ------------------------------------------------------------- Final Metrics Calculation -------------------------------------------------------------
    print(f"Done. Avg Dist: {mu_dist:.2f} ± {std_dist:.2f}, AI Err: {mu_ai_err:.2f} ± {std_ai_err:.2f}, IHDI_4cls_Acc_All: {cls_metrics['4cls_Acc_all']:.2%}")

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
        
        "p_val_shapiro": p_val_shapiro
    }
    
    # 直接將 cls_metrics 裡所有的分類與篩檢指標無縫合併進來
    metrics.update(cls_metrics)
    metrics.update(ai_left_metrics)
    metrics.update(ai_right_metrics)
    metrics.update(ai_all_metrics)
    metrics.update(pixel_vs_angle_metrics)
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
    parser.add_argument("--use_gt_box", action='store_true', help="use ground truth bounding boxes instead of YOLO detection")
    parser.add_argument("--model_points", type=int, default=None, help="number of keypoints the model predicts (8 or 12), if not specified, infer from dataset")
    args = parser.parse_args()

    predict(
        args.model_name,
        args.kp_left_path,
        args.kp_right_path,
        args.yolo_weights,
        args.data,
        args.output_dir,
        args.fold_index,
        args.use_gt_box,
        args.model_points
    )

# 單側模型預測
# python3 predict_hip_crop_keypoints.py --model_name convnext_small_fpn1234concat --kp_left_path results/25_simcc/convnext_small_fpn1234concat_simcc_2d_sr3.0_sigma7.0_cropleft_mirror_224_200_0.0001_32_best.pth --yolo_weights models/yolo12s.pt --data "data/test" --output_dir "results"
"""
python predict_hip_crop_keypoints.py \
  --model_name convnext_tiny_fpn1234concat \
  --kp_left_path weights/convnext_tiny_fpn1234concat_simcc_2d_sr3.0_sigma7.0_cropleft_mirror_224_200_0.0001_64_best.pth \
  --yolo_weights weights/yolo26s_mtddh_set.pt \
  --data "data/test" \
  --output_dir "results_mtddh_set" \
  --model_points 8
"""