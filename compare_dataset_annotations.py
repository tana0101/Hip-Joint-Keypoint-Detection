import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from scipy.stats import pearsonr, kendalltau

# ---------------------------------------------------------
# 1. 引用計算邏輯
# ---------------------------------------------------------
from predict_hip_crop_keypoints import (
    load_annotations,
    calc_point_dists,
    build_distance_ranges,
    choose_distance_subfolder,
    plot_pixel_vs_angle_error
)
from utils.keypoint_metrics import calculate_icc
from utils.hip_geometry import (
    calculate_acetabular_index_angles,
    classify_quadrant_ihdi,
    unify_keypoints_format,
    draw_hilgenreiner_line,
    draw_perpendicular_line,
    draw_diagonal_line,
    draw_h_point
)
# 引用原本的繪圖輔助線工具
from utils.plots import add_sigma_guides, add_zscore_right_axis

# ---------------------------------------------------------
# 2. 繪圖函式區
# ---------------------------------------------------------

def plot_avg_dist_bar_custom(distances, names, save_path):
    """
    繪製每張圖片的平均距離差異長條圖 (含 Mean & Std 輔助線)
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    indices = np.arange(len(names))
    ax.bar(indices, distances, label='Avg Pixel Difference', color='skyblue')
    
    # 計算統計值
    mu_dist = float(np.mean(distances))
    std_dist = float(np.std(distances, ddof=1)) if len(distances) > 1 else 0.0
    
    # 畫輔助線 (Mean, Sigma)
    add_sigma_guides(ax, mu=mu_dist, std=std_dist, 
                     mu_label=f'Mean Diff: {mu_dist:.2f} px', 
                     label=f'Mean ± 1σ (σ={std_dist:.2f})')
    add_zscore_right_axis(ax, mu=mu_dist, std=std_dist)
    
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Avg Distance (px)')
    ax.set_title(f"Difference per Image (Mean={mu_dist:.2f}, Std={std_dist:.2f})")
    
    # 優化 X 軸標籤顯示 (避免太擠，每隔 50 張顯示一個)
    tick_step = max(1, len(names) // 30) # 動態調整間隔，讓圖上大概顯示30個標籤
    target_ticks = indices[::tick_step]
    target_labels = [str(i) for i in target_ticks] # 顯示 Index 即可，顯示檔名會太長
    
    ax.set_xticks(target_ticks)
    ax.set_xticklabels(target_labels, rotation=0, fontsize=10)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_ai_angle_error_bar_custom(errors_l, errors_r, names, save_path):
    """
    繪製每張圖片的 AI 角度誤差長條圖 (Left/Right 分開顯示)
    - 會畫 Left/Right 平均線
    - 會畫 Overall 的 μ 與 μ±1σ 輔助線
    - 右側加上 z-score 軸
    - x 軸 tick 會自動稀疏顯示（避免太擠）
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # --- 資料準備 ---
    errors_l = np.asarray(errors_l, dtype=float)
    errors_r = np.asarray(errors_r, dtype=float)
    indices = np.arange(len(names))

    # --- Bar plot (仿你的範例) ---
    bar_width = 0.4
    ax.bar(indices - bar_width/2, errors_l, width=bar_width, label='Left AI Error',  color='magenta')
    ax.bar(indices + bar_width/2, errors_r, width=bar_width, label='Right AI Error', color='crimson')

    # --- Left/Right 平均線 ---
    avg_error_left  = float(np.mean(errors_l)) if len(errors_l) > 0 else 0.0
    avg_error_right = float(np.mean(errors_r)) if len(errors_r) > 0 else 0.0
    ax.axhline(avg_error_left,  linestyle='--', label=f'Avg Left Error: {avg_error_left:.2f}°',  color='magenta')
    ax.axhline(avg_error_right, linestyle='--', label=f'Avg Right Error: {avg_error_right:.2f}°', color='crimson')

    # --- Overall μ, σ (合併 L/R) ---
    combined_errors = np.concatenate([errors_l, errors_r], axis=0) if (len(errors_l) + len(errors_r)) > 0 else np.array([0.0])
    mu_err  = float(np.mean(combined_errors))
    std_err = float(np.std(combined_errors, ddof=1)) if len(combined_errors) > 1 else 0.0

    add_sigma_guides(
        ax, mu=mu_err, std=std_err,
        mu_label=f'Overall AI Error(μ): {mu_err:.2f}°',
        label=f'μ ± 1σ (σ={std_err:.2f})',
        mu_color='blue', color='red'
    )
    add_zscore_right_axis(ax, mu=mu_err, std=std_err)

    # --- Labels / Title ---
    ax.set_xlabel('Image Index')
    ax.set_ylabel('AI Angle Error (°)')
    ax.set_title(f'AI Angle Errors per Image (mu={mu_err:.2f}°)')

    # --- X 軸 tick（跟 plot_avg_dist_bar_custom 一致：動態稀疏顯示） ---
    tick_step = max(1, len(names) // 30)  # 目標顯示約 30 個刻度
    target_ticks = indices[::tick_step]
    target_labels = [str(i) for i in target_ticks]

    ax.set_xticks(target_ticks)
    ax.set_xticklabels(target_labels, rotation=0, fontsize=10)

    # --- Legend 去重 ---
    handles, labels = ax.get_legend_handles_labels()
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def draw_dataset_comparison(
    image, kpts_a, kpts_b, ai_a, ai_b,
    quad_a, quad_b,
    avg_distance, save_path, image_file,
    raw_a=None, raw_b=None,
    label_a="Dataset A", label_b="Dataset B"
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    r_a = raw_a if raw_a is not None else kpts_a
    r_b = raw_b if raw_b is not None else kpts_b
    
    plot_configs = [
        (kpts_a, f"{label_a} Geometry", ai_a, quad_a),
        (kpts_b, f"{label_b} Geometry", ai_b, quad_b)
    ]
    
    for i, (kpts_lines, title, ai, quadrants) in enumerate(plot_configs):
        ax = axes[i]
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        
        # 畫點：Dataset B (基準/紅), Dataset A (比較/黃)
        ax.scatter(r_b[:, 0], r_b[:, 1], c='red', s=10, marker='o', label=label_b)
        ax.scatter(r_a[:, 0], r_a[:, 1], c='yellow', s=10, marker='o', label=label_a)
        
        # 畫線
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
        
        # 顯示角度文字
        left_q, right_q = quadrants
        ax.text(10, image.size[1] + 35, f'{label_a if i==0 else label_b} L: {ai[0]:.1f}° (Q{left_q})', color='magenta', fontsize=11)
        ax.text(10, image.size[1] + 85, f'{label_a if i==0 else label_b} R: {ai[1]:.1f}° (Q{right_q})', color='magenta', fontsize=11)
        
        if i == 0:
            ax.legend(loc='lower left', fontsize=8)

    # 底部資訊
    diag_len = (image.size[0] ** 2 + image.size[1] ** 2) ** 0.5
    avg_dist_percent = avg_distance / diag_len * 100
    left_match = "✓" if quad_a[0] == quad_b[0] else "✗"
    right_match = "✓" if quad_a[1] == quad_b[1] else "✗"
    
    def fmt_diff(a, b):
        diff = abs(a - b)
        return f"{diff:.1f}°"

    fig.text(0.5, -0.05, 
             f"Avg Dist: {avg_distance:.2f} px ({avg_dist_percent:.2f}%)    "
             f"AI Diff(L/R): {fmt_diff(ai_a[0], ai_b[0])} / {fmt_diff(ai_a[1], ai_b[1])}    "
             f"Quad Match: L({left_match}) / R({right_match})", 
             ha='center', fontsize=12, color='blue')
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, f"{os.path.splitext(image_file)[0]}_compare.png"), bbox_inches='tight')
    plt.close()

def plot_ai_angle_scatter_custom(list_b, list_a, title_prefix, save_path, label_x="Dataset B", label_y="Dataset A"):
    x = np.array(list_b)
    y = np.array(list_a)

    a, b_reg = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = a * x_line + b_reg

    pearsonr_corr, _ = pearsonr(x, y)
    icc_val = calculate_icc(x, y)
    r2 = r2_score(x, y)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c='blue', alpha=0.6, label=f'{label_y} vs. {label_x}')
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'g--', label='Identity (y=x)')
    plt.plot(x_line, y_line, 'r--', label=f'Reg: y={a:.2f}x+{b_reg:.2f}')

    plt.title(
        f"{title_prefix} Angle Comparison\n"
        f"R={pearsonr_corr:.2f}, ICC={icc_val:.2f}, R²={r2:.2f}"
    )
    plt.xlabel(f"{label_x} AI Angle (°)")
    plt.ylabel(f"{label_y} AI Angle (°)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()

def save_confusion_matrix_custom(preds, gts, title, save_path, label_x="Dataset B (Ref)", label_y="Dataset A"):
    labels = ['I', 'II', 'III', 'IV']
    # 建立映射以計算 Kendall Correlation (必須是數值順序)
    label_map = {l: i for i, l in enumerate(labels)}
    preds_numeric = [label_map[p] for p in preds]
    gts_numeric = [label_map[g] for g in gts]

    # 1. 計算原有的基礎指標
    acc = accuracy_score(gts, preds)
    p, r, f1, _ = precision_recall_fscore_support(gts, preds, labels=labels, average='weighted', zero_division=0)
    
    # 2. 計算新增的權重指標
    # weights='quadratic' 最符合醫學診斷對於嚴重錯判的懲罰邏輯
    weighted_kappa = cohen_kappa_score(gts, preds, labels=labels, weights='quadratic')
    
    # Kendall's Tau 衡量等級排序的一致性
    tau, _ = kendalltau(gts_numeric, preds_numeric)
    
    # 3. 繪製混淆矩陣
    cm = confusion_matrix(gts, preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d') # values_format='d' 確保顯示整數
    
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    
    # 在標題中展示所有關鍵指標
    # WK: Weighted Kappa, KT: Kendall's Tau
    ax.set_title(f'{title}\nAcc: {acc:.2%} | F1: {f1:.2%}\nWK(Quad): {weighted_kappa:.3f} | Kendall: {tau:.3f}', 
                 fontsize=10)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300) # 提高解析度
    plt.close(fig)

# ---------------------------------------------------------
# 3. 主程式
# ---------------------------------------------------------

def main(args):
    dir_a = args.dataset_a
    dir_b = args.dataset_b
    out_dir = args.output_dir
    label_a_name = args.label_a
    label_b_name = args.label_b
    
    os.makedirs(out_dir, exist_ok=True)
    dist_ranges = build_distance_ranges(out_dir)
    
    imgs_a = sorted([f for f in os.listdir(os.path.join(dir_a, 'images')) if f.endswith(('.jpg', '.png'))])
    
    metrics = {
        'names': [], 'avg_dists': [],
        'ai_diff_l': [], 'ai_diff_r': [],
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
        metrics['ai_diff_l'].append(abs(ai_l_a - ai_l_b))
        metrics['ai_diff_r'].append(abs(ai_r_a - ai_r_b))
        metrics['ai_a_l'].append(ai_l_a); metrics['ai_a_r'].append(ai_r_a)
        metrics['ai_b_l'].append(ai_l_b); metrics['ai_b_r'].append(ai_r_b)
        metrics['quad_a_l'].append(q_l_a); metrics['quad_a_r'].append(q_r_a)
        metrics['quad_b_l'].append(q_l_b); metrics['quad_b_r'].append(q_r_b)

        save_path = choose_distance_subfolder(avg_dist, dist_ranges)
        
        draw_dataset_comparison(
            image=img.convert("L"),
            kpts_a=kpts_a_u,
            kpts_b=kpts_b_u,
            ai_a=(ai_l_a, ai_r_a),
            ai_b=(ai_l_b, ai_r_b),
            quad_a=(q_l_a, q_r_a),
            quad_b=(q_l_b, q_r_b),
            avg_distance=avg_dist,
            save_path=save_path,
            image_file=fname,
            raw_a=kpts_a_raw,
            raw_b=kpts_b_raw,
            label_a=label_a_name,
            label_b=label_b_name
        )

    # ---------------------------------------------------------
    # 統計報表
    # ---------------------------------------------------------
    print("Generating summary plots...")
    
    # 1. Avg Distance Bar Chart
    plot_avg_dist_bar_custom(
        metrics['avg_dists'], 
        metrics['names'], 
        os.path.join(out_dir, "avg_dists.png")
    )
    
    # Avg Distance Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(metrics['avg_dists'], bins=30, alpha=0.75, edgecolor="black")

    plt.xlabel("Average Pixel Distance")
    plt.ylabel("Count")
    plt.title("Histogram of Avg Distances Between Datasets")
    plt.tight_layout()
    hist_dist_path = os.path.join(out_dir, "hist_avg_dists.png")
    plt.savefig(hist_dist_path, dpi=300)
    plt.close()
    
    # 2. Avg AI Angle Error Bar Chart
    plot_ai_angle_error_bar_custom(
        metrics['ai_diff_l'],
        metrics['ai_diff_r'],
        metrics['names'],
        os.path.join(out_dir, "ai_angle_errors.png")
    )
    
    # Avg AI Angle Error Histogram
    ai_avg_errors = [ (l + r) / 2 for l, r in zip(metrics['ai_diff_l'], metrics['ai_diff_r']) ]
    plt.figure(figsize=(8, 6))
    plt.hist(ai_avg_errors, bins=30, alpha=0.75, edgecolor="black")

    plt.xlabel("Avg AI Angle Error (°)")
    plt.ylabel("Count")
    plt.title("Histogram of Avg AI Angle Error (All folds)")
    plt.tight_layout()
    hist_ai_path = os.path.join(out_dir, "hist_ai_error.png")
    plt.savefig(hist_ai_path, dpi=300)
    plt.close()
    

    # 3. Confusion Matrix
    save_confusion_matrix_custom(
        metrics['quad_a_l'] + metrics['quad_a_r'], 
        metrics['quad_b_l'] + metrics['quad_b_r'],
        "IHDI Classification Consistency (All)",
        os.path.join(out_dir, "confusion_matrix_all.png"),
        label_x=label_b_name, label_y=label_a_name
    )

    # 4. Scatter Plots
    all_a = metrics['ai_a_l'] + metrics['ai_a_r']
    all_b = metrics['ai_b_l'] + metrics['ai_b_r']
    plot_ai_angle_scatter_custom(
        all_b, all_a, "Overall", 
        os.path.join(out_dir, "scatter_ai_all.png"),
        label_x=label_b_name, label_y=label_a_name
    )

    # 5. Pixel vs Angle
    ai_err_avg = [(l+r)/2 for l, r in zip(metrics['ai_diff_l'], metrics['ai_diff_r'])]
    plot_pixel_vs_angle_error(metrics['avg_dists'], ai_err_avg, 
                              os.path.join(out_dir, "scatter_pix_vs_angle.png"))

    mu_dist = np.mean(metrics['avg_dists'])
    
    print(f"Done. Processed {len(metrics['names'])} images.")
    print(f"Avg Diff: {mu_dist:.2f} px")

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