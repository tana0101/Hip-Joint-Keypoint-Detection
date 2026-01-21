import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from predict_hip_crop_keypoints import (
    predict,
    extract_info_from_model_path,
    compute_and_save_confusion_matrices_with_metrics,
    plot_ai_angle_scatter,
    plot_pixel_vs_angle_error,
)

def main():
    parser = argparse.ArgumentParser(description="K-fold test for hip crop keypoints.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="efficientnet | resnet | vgg | convnext ...",
    )
    parser.add_argument(
        "--kp_left_tpl",
        type=str,
        default="",
        help="左側 KP model 路徑模板，例如: "
        "'weights/convnext_left_fold{fold}_best.pth'，"
        "會用 .format(fold=i) 產生每個 fold 的路徑。",
    )
    parser.add_argument(
        "--kp_right_tpl",
        type=str,
        default="",
        help="右側 KP model 路徑模板，可留空。",
    )
    parser.add_argument(
        "--yolo_weights", type=str, required=True, help="YOLO weights (e.g., weights/yolo12s_fold{fold}.pt)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="包含 fold1, fold2, ... 的資料夾，例如 data 或 dataset/xray_IHDI_5_kfold",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="fold 數量",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results_kfold_test",
        help="每個 fold 的輸出根目錄（下層會自動建立 fold1, fold2, ...）",
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    output_root = os.path.abspath(args.output_root)

    all_fold_metrics = []
    exp_name = None

    for fold_idx in range(1, args.k + 1):
        print("\n" + "=" * 80)
        print(f"[KFold-Test] Fold {fold_idx}/{args.k}")
        print("=" * 80)

        # 根據 template 產生該 fold 的 KP model 路徑
        if args.kp_left_tpl:
            kp_left_path = args.kp_left_tpl.format(fold=fold_idx)
        else:
            kp_left_path = ""

        if args.kp_right_tpl:
            kp_right_path = args.kp_right_tpl.format(fold=fold_idx)
        else:
            kp_right_path = ""

        yolo_path = args.yolo_weights.format(fold=fold_idx)

        if exp_name is None:
            use_left = (kp_left_path is not None) and (str(kp_left_path).strip() != "")
            use_right = (kp_right_path is not None) and (str(kp_right_path).strip() != "")
            head_type, input_size, epochs, learning_rate, batch_size, split_ratio, sigma = extract_info_from_model_path(
                kp_left_path if use_left else kp_right_path
            )
            crop_side = "both-sides" if use_left and use_right else ("left-only" if use_left else "right-only")
            if head_type in ["simcc_1d", "simcc_2d", "simcc_2d_deconv"]:
                exp_name = f"{args.model_name}_{head_type}_sr{split_ratio}_sigma{sigma}_{crop_side}_{input_size}_{epochs}_{learning_rate}_{batch_size}"
            else:
                exp_name = (
                    f"{args.model_name}_{head_type}_{crop_side}_{input_size}_{epochs}_{learning_rate}_{batch_size}"
                )

        # 每個 fold 的資料目錄：data_root/fold{fold}
        data_dir = os.path.join(data_root, f"fold{fold_idx}")
        if not os.path.isdir(data_dir):
            raise RuntimeError(f"[ERROR] data_dir not found: {data_dir}")

        # 每個 fold 的輸出目錄：output_root/exp_name/fold{fold}
        fold_output_dir = os.path.join(output_root, exp_name, f"fold{fold_idx}")
        os.makedirs(fold_output_dir, exist_ok=True)

        print(f"[KFold-Test] data_dir      = {data_dir}")
        print(f"[KFold-Test] output_dir    = {fold_output_dir}")
        print(f"[KFold-Test] kp_left_path  = {kp_left_path if kp_left_path else '(none)'}")
        print(f"[KFold-Test] kp_right_path = {kp_right_path if kp_right_path else '(none)'}")
        print(f"[KFold-Test] yolo_weights  = {yolo_path}")

        metrics = predict(
            args.model_name,
            kp_left_path,
            kp_right_path,
            yolo_path,
            data_dir,
            fold_output_dir,
            fold_index=fold_idx,
        )
        metrics["fold_index"] = fold_idx
        all_fold_metrics.append(metrics)

        print(f"[KFold-Test] Fold {fold_idx} done. num_images = {metrics['num_images']}")

    if not all_fold_metrics:
        print("[KFold-Test] No fold metrics collected, abort.")
        return

    print("\n" + "=" * 80)
    print("[KFold-Test] Aggregating metrics across folds...")
    print("=" * 80)

    # 找出要做 mean/std 的欄位（排除非數值）
    ignore_keys = {"exp_name", "fold_index"}
    numeric_keys = []

    example = all_fold_metrics[0]
    for key, value in example.items():
        if key in ignore_keys:
            continue
        if isinstance(value, (int, float, np.number)):
            numeric_keys.append(key)

    summary_lines = []
    summary_lines.append(f"K-fold test summary for {exp_name}")
    summary_lines.append(f"k = {len(all_fold_metrics)}")
    summary_lines.append("")

    # 先列出每個 fold 的數值
    for m in all_fold_metrics:
        fold_id = m.get("fold_index", "?")
        summary_lines.append(f"Fold {fold_id}:")
        summary_lines.append(f"  num_images        = {m.get('num_images', 'N/A')}")
        for key in sorted(numeric_keys):
            summary_lines.append(f"  {key:<18} = {m[key]:.6f}")
        summary_lines.append("")

    # 再對每個 numeric_key 做 mean ± std
    summary_lines.append("Average ± 1 std across folds:")
    for key in sorted(numeric_keys):
        vals = np.array([m[key] for m in all_fold_metrics], dtype=float)
        mean_val = float(vals.mean())
        std_val = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_lines.append(f"  {key:<18} = {mean_val:.6f} ± {std_val:.6f}")

    # 寫到檔案
    os.makedirs(output_root, exist_ok=True)
    summary_path = os.path.join(output_root, f"{exp_name}_kfold_test_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"[KFold-Test] Summary written to: {summary_path}")
    print("[KFold-Test] Done.")

    # ===================================================
    #  全部 folds 的 raw list 聚合：建立「全照片統計與圖」
    # ===================================================
    all_folds_summary_dir = os.path.join(output_root, exp_name, "summary")
    os.makedirs(all_folds_summary_dir, exist_ok=True)

    # ---- 1) 把所有 folds 的 per-image lists 接起來 ----
    all_dist = []
    all_ai_err_left = []
    all_ai_err_right = []
    all_ai_left_gt = []
    all_ai_left_pred = []
    all_ai_right_gt = []
    all_ai_right_pred = []
    all_left_pred = []
    all_left_gt = []
    all_right_pred = []
    all_right_gt = []

    for m in all_fold_metrics:
        all_dist.extend(m["all_avg_distances"])
        all_ai_err_left.extend(m["ai_errors_left"])
        all_ai_err_right.extend(m["ai_errors_right"])
        all_ai_left_gt.extend(m["ai_left_gt_list"])
        all_ai_left_pred.extend(m["ai_left_pred_list"])
        all_ai_right_gt.extend(m["ai_right_gt_list"])
        all_ai_right_pred.extend(m["ai_right_pred_list"])
        all_left_pred.extend(m["left_quadrants_pred"])
        all_left_gt.extend(m["left_quadrants_gt"])
        all_right_pred.extend(m["right_quadrants_pred"])
        all_right_gt.extend(m["right_quadrants_gt"])

    all_dist = np.asarray(all_dist, dtype=float)
    all_ai_err_left = np.asarray(all_ai_err_left, dtype=float)
    all_ai_err_right = np.asarray(all_ai_err_right, dtype=float)

    # per-image 平均 AI angle error
    all_ai_err_avg = (all_ai_err_left + all_ai_err_right) / 2.0

    # ---- 2) avg_distances 直方圖（全部 image）----
    plt.figure(figsize=(8, 6))
    plt.hist(all_dist, bins=30, alpha=0.75, edgecolor="black")

    plt.xlabel("Average Pixel Distance")
    plt.ylabel("Count")
    plt.title("Histogram of Avg Distances (All folds)")
    plt.tight_layout()
    hist_dist_path = os.path.join(all_folds_summary_dir, "all_avg_distances_hist.png")
    plt.savefig(hist_dist_path, dpi=300)
    plt.close()

    # ---- 3) AI angle error 直方圖（用 per-image avg error）----
    plt.figure(figsize=(8, 6))
    plt.hist(all_ai_err_avg, bins=30, alpha=0.75, edgecolor="black")

    plt.xlabel("Avg AI Angle Error (°)")
    plt.ylabel("Count")
    plt.title("Histogram of Avg AI Angle Error (All folds)")
    plt.tight_layout()
    hist_ai_path = os.path.join(all_folds_summary_dir, "all_ai_error_hist.png")
    plt.savefig(hist_ai_path, dpi=300)
    plt.close()

    # ---- 4) Over-all AI angle scatter（合併左+右） ----
    ai_gt_all = np.concatenate([np.asarray(all_ai_left_gt, float), np.asarray(all_ai_right_gt, float)], axis=0)
    ai_pred_all = np.concatenate([np.asarray(all_ai_left_pred, float), np.asarray(all_ai_right_pred, float)], axis=0)

    scatter_overall_path = os.path.join(all_folds_summary_dir, "scatter_overall_ai_angle.png")
    plot_ai_angle_scatter(
        ai_gt_all,
        ai_pred_all,
        side="Overall (All folds)",
        save_path=scatter_overall_path,
    )

    # ---- 5) Over-all confusion matrix（左/右/合併）----
    all_folds_cls_metrics = compute_and_save_confusion_matrices_with_metrics(
        left_preds=all_left_pred,
        left_gts=all_left_gt,
        right_preds=all_right_pred,
        right_gts=all_right_gt,
        save_dir=all_folds_summary_dir,
    )

    # ---- 6) Pixel vs Angle error (global) ----
    pixel_vs_angle_path = os.path.join(all_folds_summary_dir, "scatter_pixel_vs_angle_error_all.png")
    plot_pixel_vs_angle_error(
        pixel_errors=all_dist,
        ai_errors_avg=all_ai_err_avg,
        save_path=pixel_vs_angle_path,
    )


if __name__ == "__main__":
    main()

"""
python kfold_predict_hip_crop_keypoints.py \
  --model_name convnext_small_fpn1234concat \
  --kp_left_tpl "results_mtddh/results_kfold/convnext_small_fpn1234concat_simcc_2d_sr3.0_sigma7.0_cropleft_mirror_224_200_0.0001_16_fold{fold}_best.pth" \
  --yolo_weights weights/yolo12s_kfold_mtddh_fold{fold}.pt \
  --data_root data \
  --k 5 \
  --output_root results_kfold
"""
# 
# 
# 