import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

from models.model import initialize_model
from utils.keypoints import get_pred_coords
from utils.csv_parser import parse_csv_get_points
from utils.evaluation import extract_info_from_model_path

from predict_hip_crop_keypoints import (
    calculate_avg_distance,
    calc_point_dists,
    draw_comparison_figure,
    build_distance_ranges,
    choose_distance_subfolder,
    plot_avg_distances,
    plot_ai_angle_errors,
    plot_error_histogram_with_shapiro,
    compute_and_save_confusion_matrices_with_metrics,
    plot_ai_angle_scatter,
    plot_pixel_vs_angle_error,
    PIX_TH, ANG_TH
)

# 引入幾何計算
from utils.hip_geometry import (
    calculate_acetabular_index_angles,
    classify_quadrant_ihdi,
    unify_keypoints_format
)

def predict_onestage(model_name, model_path, data_dir, output_dir, fold_index=None, model_points=None):
    
    # 0. 判斷點數 (全圖偵測)
    ann_dir = os.path.join(data_dir, 'annotations')
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.csv')]
    if not ann_files: raise ValueError(f"No CSV annotations found in {ann_dir}")
    sample_kpts = parse_csv_get_points(os.path.join(ann_dir, ann_files[0]))
    total_points = model_points if model_points else sample_kpts.shape[0]
    
    print(f"[One-Stage] Initializing Model with: {total_points} total points.")
    
    # 1. 載入模型資訊
    head_type, input_size, epochs, learning_rate, batch_size, split_ratio, sigma = extract_info_from_model_path(model_path)
    
    if head_type in ["simcc_1d", "simcc_2d", "simcc_2d_deconv"]:
        Nx = Ny = int(input_size * split_ratio)
    else:
        Nx = Ny = None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(model_name, num_points=total_points, head_type=head_type, input_size=(input_size, input_size), Nx=Nx, Ny=Ny)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()

    # 2. 建立輸出目錄
    exp_suffix = f"_{total_points}pt"
    
    if head_type.startswith("simcc"): 
        exp_name = f"{model_name}_{head_type}_sr{split_ratio}_sigma{sigma}_onestage{exp_suffix}"
    elif head_type == "heatmap":
        exp_name = f"{model_name}_{head_type}_sigma{sigma}_onestage{exp_suffix}"
    else: 
        exp_name = f"{model_name}_{head_type}_onestage{exp_suffix}"
    
    result_dir = output_dir if fold_index is not None else os.path.join(output_dir, exp_name)
    os.makedirs(result_dir, exist_ok=True)
    dist_ranges = build_distance_ranges(result_dir)

    # 用於儲存圖表與指標的變數 (與舊版完全相同)
    all_avg_distances = []
    image_labels = []
    ai_errors_left, ai_errors_right = [], []
    signed_errors_left, signed_errors_right = [], []
    ai_left_gt_list, ai_left_pred_list = [], []
    ai_right_gt_list, ai_right_pred_list = [], []
    left_preds_all, left_gts_all = [], []
    right_preds_all, right_gts_all = [], []
    
    pixel_outlier_records, angle_outlier_records, all_outlier_records, all_outlier_files = [], [], [], []

    image_files = sorted(f for f in os.listdir(os.path.join(data_dir, 'images')) if f.lower().endswith(('.jpg', '.png')))

    # 3. 逐圖處理
    for idx, fname in enumerate(image_files, 1):
        img_path = os.path.join(data_dir, 'images', fname)
        ann_path = os.path.join(data_dir, 'annotations', os.path.splitext(fname)[0]+".csv")
        if not os.path.exists(ann_path): continue

        # ====================================================
        # 單階段前處理：Equalize -> Letterbox -> Resize
        # ====================================================
        img_raw = Image.open(img_path).convert("L")
        img_eq = ImageOps.equalize(img_raw).convert("RGB")
        W, H = img_eq.size
        
        max_side = max(W, H)
        pad_left = (max_side - W) // 2
        pad_top = (max_side - H) // 2
        
        img_padded = Image.new("RGB", (max_side, max_side), color=(0,0,0))
        img_padded.paste(img_eq, (pad_left, pad_top))
        
        img_resized = img_padded.resize((input_size, input_size), Image.BILINEAR)
        tensor_img = TF.to_tensor(img_resized).unsqueeze(0).to(device)

        # ====================================================
        # 推論與座標反向映射
        # ====================================================
        with torch.inference_mode():
            outputs = model(tensor_img)
            coords = get_pred_coords(outputs, head_type=head_type, Nx=Nx, Ny=Ny, input_size=input_size)
            pred_kpts_scaled = coords[0].cpu().numpy() # [Total_Points, 2]

        # 反推回原始長方形圖的座標
        scale_factor = max_side / input_size
        kps_pred_raw = pred_kpts_scaled * scale_factor
        kps_pred_raw[:, 0] -= pad_left
        kps_pred_raw[:, 1] -= pad_top

        # 讀取 GT 座標
        kps_gt_raw = parse_csv_get_points(ann_path)

        # ====================================================
        # 計算指標 (邏輯與兩階段完全一樣)
        # ====================================================
        dist = calculate_avg_distance(kps_pred_raw, kps_gt_raw)
        all_avg_distances.append(dist)
        image_labels.append(idx)

        # Unify formats for Angles & Quadrants
        kps_u_pred = unify_keypoints_format(kps_pred_raw)
        kps_u_gt = unify_keypoints_format(kps_gt_raw)
        
        ail_p, air_p = calculate_acetabular_index_angles(kps_u_pred)
        ail_g, air_g = calculate_acetabular_index_angles(kps_u_gt)
        ql_p, qr_p = classify_quadrant_ihdi(kps_u_pred)
        ql_g, qr_g = classify_quadrant_ihdi(kps_u_gt)
        
        ai_left_pred_list.append(ail_p); ai_right_pred_list.append(air_p)
        ai_left_gt_list.append(ail_g);   ai_right_gt_list.append(air_g)
        
        ai_errors_left.append(abs(ail_p - ail_g)); ai_errors_right.append(abs(air_p - air_g))
        signed_errors_left.append(ail_p - ail_g);  signed_errors_right.append(air_p - air_g)
        
        left_preds_all.append(ql_p); right_preds_all.append(qr_p)
        left_gts_all.append(ql_g);   right_gts_all.append(qr_g)

        # 畫圖
        draw_comparison_figure(
            image=img_raw, # 傳入最原始未補邊的圖畫線
            pred_kpts=kps_u_pred, gt_kpts=kps_u_gt,
            ai_pred=(ail_p, air_p), ai_gt=(ail_g, air_g),
            quadrants_pred=(ql_p, qr_p), quadrants_gt=(ql_g, qr_g),
            avg_distance=dist, save_path=choose_distance_subfolder(dist, dist_ranges),
            image_file=fname, raw_pred=kps_pred_raw, raw_gt=kps_gt_raw
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
            pixel_outlier_records.append(reason); all_outlier_records.append("pixel " + reason)
        if is_ang:
            reason = f"{fname} L_AI:{err_ail:.2f} R_AI:{err_air:.2f}"
            angle_outlier_records.append(reason); all_outlier_records.append("angle " + reason)
        if is_pix or is_ang:
            all_outlier_files.append(os.path.splitext(fname)[0])

    # ====================================================
    # 呼叫舊版的彙整繪圖與指標計算函式
    # ====================================================
    tick_step_val = 50

    mu_dist, std_dist = plot_avg_distances(image_labels, all_avg_distances, result_dir, tick_step_val)
    avg_error_left, avg_error_right, mu_ai_err, std_ai_err = plot_ai_angle_errors(image_labels, ai_errors_left, ai_errors_right, result_dir, tick_step_val)
    p_val_shapiro = plot_error_histogram_with_shapiro(signed_errors_left, signed_errors_right, result_dir)
    cls_metrics = compute_and_save_confusion_matrices_with_metrics(left_preds_all, left_gts_all, right_preds_all, right_gts_all, result_dir)
    
    ai_gt_all = np.concatenate([ai_left_gt_list, ai_right_gt_list])
    ai_pred_all = np.concatenate([ai_left_pred_list, ai_right_pred_list])

    ai_left_metrics = plot_ai_angle_scatter(ai_left_gt_list, ai_left_pred_list, 'Left', os.path.join(result_dir, "scatter_ai_left.png"))
    ai_right_metrics = plot_ai_angle_scatter(ai_right_gt_list, ai_right_pred_list, 'Right', os.path.join(result_dir, "scatter_ai_right.png"))
    ai_all_metrics = plot_ai_angle_scatter(ai_gt_all, ai_pred_all, 'All', os.path.join(result_dir, "scatter_ai_all.png"))
    
    ai_errors_avg = [(l + r) / 2 for l, r in zip(ai_errors_left, ai_errors_right)]
    pixel_vs_angle_metrics = plot_pixel_vs_angle_error(all_avg_distances, ai_errors_avg, os.path.join(result_dir, "scatter_pix_vs_angle.png"))
    
    # Save Outliers
    with open(os.path.join(result_dir, "outliers_pixel.txt"), "w") as f: f.write("\n".join(pixel_outlier_records))
    with open(os.path.join(result_dir, "outliers_angle.txt"), "w") as f: f.write("\n".join(angle_outlier_records))
    with open(os.path.join(result_dir, "outliers_all.txt"), "w") as f: f.write("\n".join(all_outlier_records))
    with open(os.path.join(result_dir, "outlier_files.txt"), "w") as f: f.write("\n".join(all_outlier_files))
    
    print(f"[One-Stage] Done. Avg Dist: {mu_dist:.2f} ± {std_dist:.2f}, AI Err: {mu_ai_err:.2f} ± {std_ai_err:.2f}")
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
    
    metrics.update(cls_metrics)
    metrics.update(ai_left_metrics)
    metrics.update(ai_right_metrics)
    metrics.update(ai_all_metrics)
    metrics.update(pixel_vs_angle_metrics)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to one-stage model (.pth)")
    parser.add_argument("--data", type=str, required=True, help="data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="output directory")
    parser.add_argument("--fold_index", type=int, default=None, help="Fold index for cross-validation (optional)")
    parser.add_argument("--model_points", type=int, default=None, help="Force number of keypoints (optional)")
    args = parser.parse_args()

    predict_onestage(
        args.model_name,
        args.model_path,
        args.data,
        args.output_dir,
        args.fold_index,
        args.model_points
    )

"""
python predict_full_image_keypoints.py \
  --model_name hrnet_w48 \
  --model_path weights/hrnet_w48_heatmap_sigma4.0_onestage_224_200_0.0002_8_best.pth \
  --data "data/test" \
  --output_dir "results_onestage_mtddh_set"
"""