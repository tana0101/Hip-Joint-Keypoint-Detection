import os
import argparse
import numpy as np
from PIL import Image

from predict_hip_crop_keypoints import (
    load_annotations,
    calc_point_dists,
    build_distance_ranges,
    choose_distance_subfolder,
    draw_comparison_figure,
    plot_avg_distances,
    plot_ai_angle_errors,
    plot_error_histogram_with_shapiro,
    compute_and_save_confusion_matrices_with_metrics,
    plot_ai_angle_scatter,
    plot_pixel_vs_angle_error
)
from utils.hip_geometry import (
    calculate_acetabular_index_angles,
    classify_quadrant_ihdi,
    unify_keypoints_format
)

def main(args):
    dir_a = args.dataset_a
    dir_b = args.dataset_b
    out_dir = args.output_dir
    label_a_name = args.label_a
    label_b_name = args.label_b
    
    os.makedirs(out_dir, exist_ok=True)
    dist_ranges = build_distance_ranges(out_dir)
    
    imgs_a = sorted([f for f in os.listdir(os.path.join(dir_a, 'images')) if f.endswith(('.jpg', '.png'))])
    
    # metrics 字典：新增了 signed_errors 欄位用來做常態分佈檢定
    metrics = {
        'names': [], 'avg_dists': [],
        'ai_diff_l': [], 'ai_diff_r': [],       # 絕對差異 (畫長條圖用)
        'signed_err_l': [], 'signed_err_r': [], # 帶正負號的差異 (Shapiro-Wilk 檢定用)
        'ai_a_l': [], 'ai_a_r': [],
        'ai_b_l': [], 'ai_b_r': [],
        'quad_a_l': [], 'quad_a_r': [],
        'quad_b_l': [], 'quad_b_r': []
    }

    print(f"Comparing: {label_a_name} vs {label_b_name}")

    for fname in imgs_a:
        name_no_ext = os.path.splitext(fname)[0]
        csv_name = name_no_ext + ".csv"
        
        path_img_a = os.path.join(dir_a, 'images', fname)
        path_ann_a = os.path.join(dir_a, 'annotations', csv_name)
        path_ann_b = os.path.join(dir_b, 'annotations', csv_name)
        
        if not os.path.exists(path_ann_a) or not os.path.exists(path_ann_b):
            continue

        img = Image.open(path_img_a).convert("RGB") 
        kpts_a_raw = load_annotations(path_ann_a)
        kpts_b_raw = load_annotations(path_ann_b)

        dists = calc_point_dists(kpts_a_raw, kpts_b_raw) 
        avg_dist = np.mean(dists)
        
        kpts_a_u = unify_keypoints_format(kpts_a_raw)
        kpts_b_u = unify_keypoints_format(kpts_b_raw)

        ai_l_a, ai_r_a = calculate_acetabular_index_angles(kpts_a_u)
        ai_l_b, ai_r_b = calculate_acetabular_index_angles(kpts_b_u)
        q_l_a, q_r_a = classify_quadrant_ihdi(kpts_a_u)
        q_l_b, q_r_b = classify_quadrant_ihdi(kpts_b_u)

        metrics['names'].append(name_no_ext)
        metrics['avg_dists'].append(avg_dist)
        
        # 絕對差異
        metrics['ai_diff_l'].append(abs(ai_l_a - ai_l_b))
        metrics['ai_diff_r'].append(abs(ai_r_a - ai_r_b))
        # 帶正負號差異 (A - B) -> 供統計檢定使用
        metrics['signed_err_l'].append(ai_l_a - ai_l_b)
        metrics['signed_err_r'].append(ai_r_a - ai_r_b)

        metrics['ai_a_l'].append(ai_l_a); metrics['ai_a_r'].append(ai_r_a)
        metrics['ai_b_l'].append(ai_l_b); metrics['ai_b_r'].append(ai_r_b)
        metrics['quad_a_l'].append(q_l_a); metrics['quad_a_r'].append(q_r_a)
        metrics['quad_b_l'].append(q_l_b); metrics['quad_b_r'].append(q_r_b)

        save_path = choose_distance_subfolder(avg_dist, dist_ranges)
        
        # 使用共用的 draw_comparison_figure，並傳入 Dataset Labels
        draw_comparison_figure(
            image=img.convert("L"),
            pred_kpts=kpts_a_u,    # 將 A 視為比較對象 (對應 Pred)
            gt_kpts=kpts_b_u,      # 將 B 視為基準點 (對應 GT)
            ai_pred=(ai_l_a, ai_r_a),
            ai_gt=(ai_l_b, ai_r_b),
            quadrants_pred=(q_l_a, q_r_a),
            quadrants_gt=(q_l_b, q_r_b),
            avg_distance=avg_dist,
            save_path=save_path,
            image_file=fname,
            raw_pred=kpts_a_raw,
            raw_gt=kpts_b_raw,
            label_pred=label_a_name, # 客製化標籤: 例如 "1st Reading"
            label_gt=label_b_name    # 客製化標籤: 例如 "2nd Reading"
        )

    # ---------------------------------------------------------
    # 統計報表 (全面使用重構後的共用函式)
    # ---------------------------------------------------------
    print("Generating summary plots...")
    image_indices = np.arange(1, len(metrics['names']) + 1)
    tick_step_val = max(1, len(metrics['names']) // 10)
    
    # 1. Avg Distance Bar Chart
    plot_avg_distances(
        image_labels=image_indices,
        all_avg_distances=metrics['avg_dists'],
        result_dir=out_dir,
        tick_step=tick_step_val,
    )
    
    # 2. Avg AI Angle Error Bar Chart
    plot_ai_angle_errors(
        image_labels=image_indices,
        ai_errors_left=metrics['ai_diff_l'],
        ai_errors_right=metrics['ai_diff_r'],
        result_dir=out_dir,
        tick_step=tick_step_val,
        bar_width=0.2
    )
    
    # 3. 誤差分佈直方圖與 Shapiro-Wilk 檢定 (這會取代你原本的 AI Error Histogram)
    p_val_shapiro = plot_error_histogram_with_shapiro(
        errors_left=metrics['signed_err_l'],   
        errors_right=metrics['signed_err_r'], 
        result_dir=out_dir,
        x_label=f"Signed Difference ({label_a_name} - {label_b_name}) (°)"
    )
    
    # 4. Confusion Matrices (含 QWK 與各種進階指標)
    compute_and_save_confusion_matrices_with_metrics(
        left_preds=metrics['quad_a_l'], 
        left_gts=metrics['quad_b_l'], 
        right_preds=metrics['quad_a_r'], 
        right_gts=metrics['quad_b_r'],
        save_dir=out_dir,
        label_pred=label_a_name,
        label_gt=label_b_name
    )

    # 5. Scatter Plots (包含 Paired t-Test / Wilcoxon 結果)
    all_a = metrics['ai_a_l'] + metrics['ai_a_r']
    all_b = metrics['ai_b_l'] + metrics['ai_b_r']
    plot_ai_angle_scatter(
        gt_list=all_b, 
        pred_list=all_a, 
        side="Overall", 
        save_path=os.path.join(out_dir, "scatter_ai_all.png"),
        label_pred=label_a_name,
        label_gt=label_b_name
    )

    # 6. Pixel vs Angle
    ai_err_avg = [(l+r)/2 for l, r in zip(metrics['ai_diff_l'], metrics['ai_diff_r'])]
    plot_pixel_vs_angle_error(
        pixel_errors=metrics['avg_dists'], 
        ai_errors_avg=ai_err_avg, 
        save_path=os.path.join(out_dir, "scatter_pix_vs_angle.png"),
        x_label="Avg Pixel Difference",
        y_label="Avg Angle Difference (°)",
        title_prefix="Pixel Difference vs. Angle Difference",
        legend_label=f"{label_a_name} vs {label_b_name}"
    )

    mu_dist = np.mean(metrics['avg_dists'])
    
    print(f"\nDone. Processed {len(metrics['names'])} images.")
    print(f"Avg Diff: {mu_dist:.2f} px")
    if p_val_shapiro > 0.05:
        print(f"[統計建議] 資料符合常態分佈 (p={p_val_shapiro:.4f}) -> 論文請採用 Paired t-Test 的 p-value。")
    else:
        print(f"[統計建議] 資料不符合常態分佈 (p={p_val_shapiro:.4f}) -> 論文請採用 Wilcoxon Signed-Rank Test 的 p-value。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_a", type=str, required=True)
    parser.add_argument("--dataset_b", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="comparison_result")
    parser.add_argument("--label_a", type=str, default="Dataset A")
    parser.add_argument("--label_b", type=str, default="Dataset B")
    
    args = parser.parse_args()
    main(args)
    
'''
python compare_dataset_annotations.py \
  --dataset_a "dataset/xray_IHDI_1_clean" \
  --dataset_b "dataset/xray_IHDI_2_clean" \
  --output_dir "output/Intra_Observer_Analysis(clean)" \
  --label_a "1st Reading" \
  --label_b "2nd Reading"
'''