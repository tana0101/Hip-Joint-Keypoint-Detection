import os
import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, transforms
from torch import nn
import pandas as pd
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau
from ultralytics import YOLO

from models.model import initialize_model

POINTS_COUNT = 6
REORDER_6 = [2, 1, 0, 5, 4, 3]  # 單側6點在左右鏡像後的索引重排（3點一組反轉）

# Detection bbox 
def _square_expand_clip(x1, y1, x2, y2, W, H, expand=0.10, keep_square=True):
    """把 bbox 變成(可選)正方形並外擴，再裁到影像邊界內。"""
    bw, bh = x2 - x1, y2 - y1
    if keep_square:
        side = max(bw, bh, 2.0)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        x1, x2 = cx - side / 2.0, cx + side / 2.0
        y1, y2 = cy - side / 2.0, cy + side / 2.0
        bw = bh = side
    # 外擴
    x1 -= bw * expand; x2 += bw * expand
    y1 -= bh * expand; y2 += bh * expand
    # 邊界裁切
    x1 = max(0.0, x1); y1 = max(0.0, y1)
    x2 = min(float(W), x2); y2 = min(float(H), y2)
    # 避免 0 尺寸
    if x2 <= x1: x2 = min(float(W), x1 + 2.0)
    if y2 <= y1: y2 = min(float(H), y1 + 2.0)
    return x1, y1, x2, y2

def _detect_one(yolo_model, pil_img, cls_id, conf=0.05, iou=0.5):
    """回傳單一 bbox (x1,y1,x2,y2)；若無偵測，回傳 None。"""
    res = yolo_model.predict(source=pil_img, classes=[cls_id], conf=conf, iou=iou, max_det=1, verbose=False)
    if not res or len(res[0].boxes) == 0:
        return None
    xyxy = res[0].boxes.xyxy[0].cpu().numpy().tolist()  # [x1,y1,x2,y2]
    return tuple(float(v) for v in xyxy)

def _infer_side_kp(kp_model, pil_crop, transform, crop_box, input_size):
    """對單側裁切圖做前處理→預測→轉回原圖座標系 (回傳 shape=(6,2) 的 numpy)。"""
    x1, y1, x2, y2 = crop_box
    crop_w, crop_h = (x2 - x1), (y2 - y1)
    # 前處理（與訓練一致）
    crop_tensor = transform(pil_crop).unsqueeze(0)   # [1,3,224,224]
    
    # 取得模型所在裝置，並把輸入搬到同裝置
    device = next(kp_model.parameters()).device
    crop_tensor = crop_tensor.to(device, non_blocking=True)
    
    with torch.inference_mode():
        pred = kp_model(crop_tensor).detach().cpu().numpy().reshape(-1, 2)  # (6,2) in 224x224
    
    # 反映射回原圖
    sx, sy = crop_w / input_size, crop_h / input_size
    pred_orig = np.empty_like(pred)
    pred_orig[:, 0] = pred[:, 0] * sx + x1
    pred_orig[:, 1] = pred[:, 1] * sy + y1
    return pred_orig  # (6,2)

# 使用鏡像模型預測函式
def _hflip_kpts_224(kpts_6x2, input_size):
    """在 224x224 空間水平鏡像單側6點（numpy, shape=(6,2)）"""
    out = kpts_6x2.copy()
    out[:, 0] = (input_size - 1) - out[:, 0]
    return out

def _reorder_between_sides(kpts_6x2, from_side, to_side):
    """左↔右的單側6點索引重排；同側不動，跨側用 REORDER_6。"""
    if from_side == to_side:
        return kpts_6x2
    return kpts_6x2[REORDER_6, :]

def _infer_via_mirror(kp_model, pil_crop_src, transform, crop_box, model_side, target_side, input_size):
    """
    單模型模式：把 'target_side' 的裁切圖鏡像成 'model_side' 外觀 → 用該模型推論 →
    在 224 空間反鏡像回來 → 做 from=model_side → to=target_side 的索引重排 → 反投影回原圖。
    回傳 (6,2) numpy（target_side 的順序）。
    """
    # 1) 目標側裁切 → 鏡像成模型側外觀
    pil_mirror = ImageOps.mirror(pil_crop_src)  # 水平鏡像

    # 2) 模型在鏡像空間推論（得到 model_side 順序，座標=224）
    crop_tensor = transform(pil_mirror).unsqueeze(0)
    # 取得模型所在裝置，並把輸入搬到同裝置
    device = next(kp_model.parameters()).device
    crop_tensor = crop_tensor.to(device, non_blocking=True)
    
    with torch.inference_mode():
        pred_model_224 = kp_model(crop_tensor).detach().cpu().numpy().reshape(-1, 2)  # (6,2) in 224x224

    # 3) 224 空間反鏡像回未鏡像空間
    pred_unflipped_224 = _hflip_kpts_224(pred_model_224, input_size)

    # 4) 索引重排：model_side → target_side
    pred_target_224 = _reorder_between_sides(pred_unflipped_224, from_side=model_side, to_side=target_side)

    # 5) 反投影回原圖
    x1, y1, x2, y2 = crop_box
    crop_w, crop_h = (x2 - x1), (y2 - y1)
    sx, sy = crop_w / input_size, crop_h / input_size
    pred_target_orig = np.empty_like(pred_target_224)
    pred_target_orig[:, 0] = pred_target_224[:, 0] * sx + x1
    pred_target_orig[:, 1] = pred_target_224[:, 1] * sy + y1
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
    # 嘗試匹配包含 input_size 的版本: _<input_size>_<epochs>_<lr>_<batch>_best.pth
    match_with_input = re.search(r'_(\d+)_([0-9]+)_([0-9eE\.\-]+)_([0-9]+)_best\.pth', model_path)
    if match_with_input:
        input_size = int(match_with_input.group(1))
        epochs = int(match_with_input.group(2))
        learning_rate = float(match_with_input.group(3))
        batch_size = int(match_with_input.group(4))
        return input_size, epochs, learning_rate, batch_size

    # 匹配不含 input_size 的版本: _<epochs>_<lr>_<batch>_best.pth
    match_without_input = re.search(r'_(\d+)_([0-9eE\.\-]+)_(\d+)_best\.pth', model_path)
    if match_without_input:
        input_size = 224 # 預設值
        epochs = int(match_without_input.group(1))
        learning_rate = float(match_without_input.group(2))
        batch_size = int(match_without_input.group(3))
        return input_size, epochs, learning_rate, batch_size

    # 都不符合時報錯
    raise ValueError("Model path format is invalid. Expected format like: "
                     "model_name_[input_size_]epochs_lr_batchsize_best.pth")

def draw_hilgenreiner_line(ax, p3, p7):
    if np.allclose(p3[0], p7[0]):
        ax.axvline(x=p3[0], color='cyan', linewidth=1, label='H-Line')
    else:
        a = (p7[1] - p3[1]) / (p7[0] - p3[0])
        b = p3[1] - a * p3[0]
        x_min, x_max = ax.get_xlim()
        x_vals = np.array([x_min, x_max])
        y_vals = a * x_vals + b
        ax.plot(x_vals, y_vals, color='cyan', linewidth=1, label='H-Line')
        
def draw_perpendicular_line(ax, point, line_p1, line_p2, color='lime', label=None):
    # 向量方向（H-line）
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]

    # 垂直向量：旋轉90度 (-dy, dx)
    perp_dx = -dy
    perp_dy = dx

    # 將垂直向量延長一定比例來畫線
    scale = 1  # 可調整線的長度
    x0, y0 = point
    x_vals = np.array([x0 - scale * perp_dx, x0 + scale * perp_dx])
    y_vals = np.array([y0 - scale * perp_dy, y0 + scale * perp_dy])

    ax.plot(x_vals, y_vals, color=color, linewidth=1, label=label)

def draw_diagonal_line(ax, point, line_p1, line_p2, direction="left_down", color='orange', label=None):
    a = np.array(line_p1)
    b = np.array(line_p2)
    p = np.array(point)
    v = b - a

    # 計算投影點（交點）
    proj = a + v * np.dot(p - a, v) / np.dot(v, v)

    # 決定方向
    length = 100
    if direction == "left_down":
        dx, dy = -length, length   # 斜率 +1
    elif direction == "right_down":
        dx, dy = length, length    # 斜率 -1
    else:
        raise ValueError("direction must be 'left_down' or 'right_down'")

    x_vals = [proj[0], proj[0] + dx]
    y_vals = [proj[1], proj[1] + dy]
    ax.plot(x_vals, y_vals, color=color, linewidth=1, label=label)

def draw_h_point(ax, kpts):
    h_left = (kpts[9] + kpts[11]) / 2   # pt10 & pt12 中點
    h_right = (kpts[3] + kpts[5]) / 2   # pt4 & pt6 中點

    # 使用亮黃色 + 圓形
    ax.scatter(*h_left, c="blue", s=4, label='H-point')
    ax.scatter(*h_right, c="blue", s=4)

# 計算 AI 角度
def calculate_acetabular_index_angles(points):
    p1 = points[0]
    p3 = points[2]
    p7 = points[6]
    p9 = points[8]
    v_h = p7 - p3
    v_left = p3 - p1
    v_right = p7 - p9
    def angle(v1, v2):
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return angle(v_h, v_left), angle(-v_h, v_right)

# 判斷IHDI象限
def classify_quadrant_ihdi(points):
    """
    根據 IHDI 分類圖，判斷左右 H-point 各自落在哪個象限。
    
    參數
    -------
    points : array-like, shape=(12, 2)
        依序為 1~12 點的 (x, y) 影像座標（x 向右、y 向下）。
    
    回傳
    -------
    left_q , right_q : str
        左、右股骨頭中心所在象限，字元 'I' ~ 'IV'
        
    實作方式
    -------
    將股骨頭中心點（H-point）投影到以 H-line 為水平、P-line 為垂直的局部座標系中，並依其位置判斷其落在哪個象限（I~IV）。
    
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape != (12, 2):
        raise ValueError("`points` 必須是 (12, 2) 的座標陣列")

    # ---- 取出關鍵點 --------------------------------------------------------
    p1  = pts[0]      # 左 P-line 基準
    p3  = pts[2]      # H-line 起點
    p4  = pts[3]      # 左股骨頭
    p6  = pts[5]
    p7  = pts[6]      # H-line 終點
    p9  = pts[8]      # 右 P-line 基準
    p10 = pts[9]      # 右股骨頭
    p12 = pts[11]

    # ---- 基準座標系：x 軸沿 H-line，y 軸向下 ------------------------------
    v_h = p7 - p3
    u_h = v_h / np.linalg.norm(v_h)                     # 單位 H 向量

    # 兩種垂直方向：順時針 / 逆時針 90°
    v_p1 = np.array([ v_h[1], -v_h[0]])
    v_p2 = np.array([-v_h[1],  v_h[0]])
    v_p  = v_p1 if v_p1[1] > v_p2[1] else v_p2          # y 分量大的朝「下」
    u_p  = v_p / np.linalg.norm(v_p)                    # 單位「下」向量

    # H-line 與 P-line 交點（作為原點）
    def proj_on_h(pt):                                  # 投影到 H-line
        t = np.dot(pt - p3, u_h)
        return p3 + t * u_h

    o_r = proj_on_h(p9)                                 # 右側原點
    o_l = proj_on_h(p1)                                 # 左側原點

    # H-points
    h_r = (p10 + p12) / 2.0
    h_l = (p4  + p6 ) / 2.0

    # 轉成局部 (x, y)
    def to_xy(p, o):
        vec = p - o
        return np.dot(vec, u_h), np.dot(vec, u_p)

    x_r, y_r = to_xy(h_r, o_r)
    x_l, y_l = to_xy(h_l, o_l)
    x_l = -x_l            # 左側鏡射：使遠離脊椎方向為 x 正

    # ---- 象限規則（以右側為基準） -----------------------------------------
    def quad(x, y):
        # IV：在 H-line 上方／接近上方
        if y <= 0 and x >= 0:
            return 'IV'
        # I：H-line 下 + P-line 左
        if y > 0 and x < 0:
            return 'I'
        # II / III：H-line 下 + P-line 右，依對角線分界
        if y > 0 and x >= 0:
            return 'II' if y > x else 'III'
        # 其他極少見邊界情況 → 歸為 none
        return 'none'

    return quad(x_l, y_l), quad(x_r, y_r)

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

# ---------- helpers for sigma guides ----------
def add_sigma_guides(ax, mu, std, one_sigma_alpha=0.10, line_alpha=0.35, mu_label=None, label=None, mu_color='red', color='blue'):
    """
    在圖上加上 μ±1σ 淡色區間、以及 μ、μ±1σ、μ±2σ 參考線。
    """
    ymin, ymax = ax.get_ylim()

    # ±1σ淡色區間
    ax.axhspan(mu - std, mu + std, alpha=one_sigma_alpha, color=color)

    # μ與σ參考線
    ax.axhline(mu, linestyle='--', linewidth=1.5, label=mu_label, color=mu_color)
    for k in (1, 2):
        ax.axhline(mu + k*std, linestyle=':', alpha=line_alpha, linewidth=1.2, 
                   color=color, label=label if k == 1 else None)
        ax.axhline(mu - k*std, linestyle=':', alpha=line_alpha, linewidth=1.2, 
                   color=color)

    # 保持原本y範圍（避免被新元素改動）
    ax.set_ylim(ymin, ymax)

def add_zscore_right_axis(ax, mu, std):
    """
    在右側加上 z-score 軸（標準化差異）。
    """
    def y_to_z(y): return (y - mu) / std
    def z_to_y(z): return z * std + mu
    secax = ax.secondary_yaxis('right', functions=(y_to_z, z_to_y))
    secax.set_ylabel('Standardized difference (σ)')
    secax.set_yticks([-2, -1, 0, 1, 2])
    return secax
    
def predict(model_name, kp_left_path, kp_right_path, yolo_weights, data_dir, output_dir):
    
    # 1) 載入 YOLO + KP 模型（左右可能有其一缺省）
    yolo_model = YOLO(yolo_weights)
    
    use_left  = (kp_left_path  is not None) and (str(kp_left_path).strip()  != "")
    use_right = (kp_right_path is not None) and (str(kp_right_path).strip() != "")
    assert use_left or use_right, "至少提供 --kp_left_path 或 --kp_right_path 其中之一"
    
    kp_left = kp_right = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_left:
        kp_left = initialize_model(model_name, POINTS_COUNT)
        kp_left.load_state_dict(torch.load(kp_left_path, map_location="cpu"))
        kp_left.to(device)
        kp_left.eval()
    if use_right:
        kp_right = initialize_model(model_name, POINTS_COUNT)
        kp_right.load_state_dict(torch.load(kp_right_path, map_location="cpu"))
        kp_right.to(device)
        kp_right.eval()
    
    # 以存在的模型路徑擷取紀錄資訊
    input_size, epochs, learning_rate, batch_size = extract_info_from_model_path(kp_left_path if use_left else kp_right_path)

    # 2) 建結果資料夾
    crop_side = "both-sides" if use_left and use_right else ("left-only" if use_left else "right-only")
    result_dir = os.path.join(output_dir, f"{model_name}_{crop_side}_{input_size}_{epochs}_{learning_rate}_{batch_size}")
    os.makedirs(result_dir, exist_ok=True)

    # NEW: 建立裁切輸出資料夾
    crops_dir = os.path.join(result_dir, "crops")
    crops_left_dir  = os.path.join(crops_dir, "left")
    crops_right_dir = os.path.join(crops_dir, "right")
    os.makedirs(crops_left_dir, exist_ok=True)   
    os.makedirs(crops_right_dir, exist_ok=True)

    # Create subdirectories for distance ranges
    distance_ranges = {
        "0-2.5": os.path.join(result_dir, "0-2.5"),
        "2.5-5": os.path.join(result_dir, "2.5-5"),
        "5-7.5": os.path.join(result_dir, "5-7.5"),
        "7.5-10": os.path.join(result_dir, "7.5-10"),
        "10-12.5": os.path.join(result_dir, "10-12.5"),
        "12.5-15": os.path.join(result_dir, "12.5-15"),
        "15-17.5": os.path.join(result_dir, "15-17.5"),
        "17.5-20": os.path.join(result_dir, "17.5-20"),
        "20-30": os.path.join(result_dir, "20-30"),
        "30-40": os.path.join(result_dir, "30-40"),
        "40+": os.path.join(result_dir, "40+"),
    }
    
    for path in distance_ranges.values():
        os.makedirs(path, exist_ok=True)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  
        transforms.Lambda(lambda img: ImageOps.equalize(img)),  # Apply histogram equalization
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

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


    for image_file in os.listdir(os.path.join(data_dir, 'images')):
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
                pred_left_6 = _infer_side_kp(kp_left, crop_left, transform, (x1,y1,x2,y2), input_size)
            else:
                # 沒左模型、只有右模型：把左鏡像成右 → 用右模型 → 反鏡像 + 重排回左
                pred_left_6 = _infer_via_mirror(kp_right, crop_left, transform, (x1,y1,x2,y2),
                                                model_side="right", target_side="left", input_size=input_size)

            # Right
            x1,y1,x2,y2 = box_right
            x1,y1,x2,y2 = _square_expand_clip(x1,y1,x2,y2, W,H, expand=0.10, keep_square=True)
            crop_right = image_pil.crop((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))).convert("L")
            right_crop_name = os.path.splitext(image_file)[0] + "_right_crop.jpg"
            crop_right.save(os.path.join(crops_right_dir, right_crop_name))

            if use_right:
                # 有右模型：直接推右
                pred_right_6 = _infer_side_kp(kp_right, crop_right, transform, (x1,y1,x2,y2), input_size)
            else:
                # 沒右模型、只有左模型：把右鏡像成左 → 用左模型 → 反鏡像 + 重排回右
                pred_right_6 = _infer_via_mirror(kp_left, crop_right, transform, (x1,y1,x2,y2),
                                                 model_side="left", target_side="right", input_size=input_size)

            
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
            if avg_distance <= 2.5:
                subfolder = distance_ranges["0-2.5"]
            elif avg_distance <= 5:
                subfolder = distance_ranges["2.5-5"]
            elif avg_distance <= 7.5:
                subfolder = distance_ranges["5-7.5"]
            elif avg_distance <= 10:
                subfolder = distance_ranges["7.5-10"]
            elif avg_distance <= 12.5:
                subfolder = distance_ranges["10-12.5"]
            elif avg_distance <= 15:
                subfolder = distance_ranges["12.5-15"]
            elif avg_distance <= 17.5:
                subfolder = distance_ranges["15-17.5"]
            elif avg_distance <= 20:
                subfolder = distance_ranges["17.5-20"]
            elif avg_distance <= 30:
                subfolder = distance_ranges["20-30"]
            elif avg_distance <= 40:
                subfolder = distance_ranges["30-40"]
            else:
                subfolder = distance_ranges["40+"]

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

# 左右雙模型預測
# python3 predict_hip_crop_keypoints.py --model_name convnext --kp_left_path models/convnext_cropleft_750_0.0001_32_best.pth --kp_right_path models/convnext_cropright_750_0.0001_32_best.pth --yolo_weights models/yolo12s.pt --data "data/test" --output_dir "results"
# python3 predict_hip_crop_keypoints.py --model_name convnext --kp_left_path models/convnext_cropleft_mirror_750_0.0001_32_best.pth --kp_right_path models/convnext_cropright_mirror_750_0.0001_32_best.pth --yolo_weights models/yolo12s.pt --data "data/test" --output_dir "results"
# 單側模型預測
# python3 predict_hip_crop_keypoints.py --model_name convnext --kp_left_path models/convnext_cropleft_mirror_750_0.0001_32_best.pth --yolo_weights models/yolo12s.pt --data "data/test" --output_dir "results"
# python3 predict_hip_crop_keypoints.py --model_name convnext --kp_right_path models/convnext_cropright_mirror_750_0.0001_32_best.pth --yolo_weights models/yolo12s.pt --data "data/test" --output_dir "results"