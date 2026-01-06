import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from utils.hip_geometry import (
    calculate_acetabular_index_angles,
    classify_quadrant_ihdi,
    unify_keypoints_format
)

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

def load_annotations(annotation_path):
    keypoints = pd.read_csv(annotation_path, header=None).values.flatten()
    keypoints = [float(coord) for point in keypoints for coord in point.strip('"()').split(",")]
    return np.array(keypoints).reshape(-1, 2)

# ---------------------------------------------------------
# 核心分析與繪圖函式
# ---------------------------------------------------------
def analyze_and_plot_dataset(dataset_name, dataset_root):
    """
    讀取指定資料夾下的 annotations，計算 AI 與 IHDI，畫圖並儲存。
    """
    print(f"--- Analyzing Dataset: {dataset_name} ---")
    
    annotation_dir = os.path.join(dataset_root, 'annotations')
    csv_files = glob.glob(os.path.join(annotation_dir, '*.csv'))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {annotation_dir}")
        return

    # 初始化儲存列表
    ai_left_list = []
    ai_right_list = []
    ai_avg_list = []
    ihdi_quadrants_all = [] 

    # --- 批次處理資料 ---
    for csv_path in csv_files:
        try:
            # A. Load
            kps_raw = load_annotations(csv_path)
            # B. Unify (8 points -> 12 points)
            kps_unified = unify_keypoints_format(kps_raw)
            # C. AI Angle
            ai_l, ai_r = calculate_acetabular_index_angles(kps_unified)
            ai_left_list.append(ai_l)
            ai_right_list.append(ai_r)
            ai_avg_list.append((ai_l + ai_r) / 2.0)
            # D. IHDI Quadrant
            q_l, q_r = classify_quadrant_ihdi(kps_unified)
            ihdi_quadrants_all.extend([q_l, q_r]) 
            
        except Exception as e:
            print(f"Error processing {os.path.basename(csv_path)}: {e}")

    # ==========================================
    # Plot 1: AI Angle Histogram (Left, Right, Avg)
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{dataset_name} - Acetabular Index (AI) Distribution', fontsize=16)

    def plot_hist(ax, data, title, color):
        ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor='black')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Angle (Degree)')
        ax.set_ylabel('Count')
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        # 英文統計資訊
        text_str = f'Mean: {mean_val:.2f}°\nStd: {std_val:.2f}'
        ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plot_hist(axes[0], ai_avg_list, 'AI Average Angle', 'green')
    plot_hist(axes[1], ai_left_list, 'AI Left Angle', 'blue')
    plot_hist(axes[2], ai_right_list, 'AI Right Angle', 'red')

    plt.tight_layout()
    
    # --- Save Plot 1 ---
    save_path_ai = os.path.join(dataset_root, f'{dataset_name}_AI_Distribution.png')
    plt.savefig(save_path_ai, dpi=300)
    print(f"Saved AI plot to: {save_path_ai}")
    
    plt.show()

    # ==========================================
    # Plot 2: IHDI Quadrant Bar Chart
    # ==========================================
    plt.figure(figsize=(8, 6))
    
    counter = Counter(ihdi_quadrants_all)
    labels = ['I', 'II', 'III', 'IV']
    counts = [counter[label] for label in labels]
    
    bars = plt.bar(labels, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8, edgecolor='black')
    
    # 英文標題
    plt.title(f'{dataset_name} - IHDI Quadrant Classification Stats (Total Hips: {len(ihdi_quadrants_all)})', fontsize=15)
    plt.xlabel('IHDI Quadrant', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{count}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()

    # --- Save Plot 2 ---
    save_path_ihdi = os.path.join(dataset_root, f'{dataset_name}_IHDI_Distribution.png')
    plt.savefig(save_path_ihdi, dpi=300)
    print(f"Saved IHDI plot to: {save_path_ihdi}")

    plt.show()

if __name__ == "__main__":
    # 請將此處路徑修改為您實際的資料夾路徑
    # 範例結構: 
    # dataset_xray_IHDI/annotations/*.csv
    # dataset_mtddh/annotations/*.csv
    
    path_ihdi = "dataset/xray_IHDI_6"
    path_mtddh = "dataset/mtddh_xray_2d"
    
    # 檢查路徑是否存在並執行
    if os.path.exists(path_ihdi):
        analyze_and_plot_dataset("xray_IHDI", path_ihdi)
    else:
        print(f"找不到路徑: {path_ihdi}")

    if os.path.exists(path_mtddh):
        analyze_and_plot_dataset("mtddh_xray_2d", path_mtddh)
    else:
        print(f"找不到路徑: {path_mtddh}")