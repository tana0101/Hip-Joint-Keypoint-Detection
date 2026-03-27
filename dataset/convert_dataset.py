import os
import json
import cv2
import glob
import shutil

# ================= 設定區域 =================
# 基礎資料夾 (所有東西都在這裡面)
BASE_DIR = 'mtddh_xray_2d'

# 輸入資料夾 (原始資料)
INPUT_FOLDER = os.path.join(BASE_DIR, 'data')

# 輸出資料夾 (將生成在 mtddh_xray_2d/images 等等)
DIR_IMAGES = os.path.join(BASE_DIR, 'images')
DIR_ANNOTATIONS = os.path.join(BASE_DIR, 'annotations') # CSV
DIR_DETECTIONS = os.path.join(BASE_DIR, 'detections')   # JSON
DIR_YOLO = os.path.join(BASE_DIR, 'yolo_labels')        # TXT (New YOLO)

# 確保輸出資料夾存在
for folder in [DIR_IMAGES, DIR_ANNOTATIONS, DIR_DETECTIONS, DIR_YOLO]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"已建立資料夾: {folder}")

# ================= 輔助函式 =================

def to_abs_box(norm_box, img_w, img_h):
    """將 YOLO 歸一化 bbox (cx, cy, w, h) 轉為 絕對座標 (x1, y1, x2, y2)"""
    cx, cy, w, h = norm_box
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return [x1, y1, x2, y2]

def to_yolo_format(box, img_w, img_h):
    """將 絕對座標 (x1, y1, x2, y2) 轉回 YOLO 歸一化 (cx, cy, w, h)"""
    x1, y1, x2, y2 = box
    w_px = x2 - x1
    h_px = y2 - y1
    cx_px = x1 + (w_px / 2)
    cy_px = y1 + (h_px / 2)
    
    return [
        cx_px / img_w,
        cy_px / img_h,
        w_px / img_w,
        h_px / img_h
    ]

def merge_objects_data(obj_list, img_w, img_h):
    """
    合併同一邊的物件：
    1. 計算所有小框組成的大框 (min_x, min_y, max_x, max_y)
    2. 串接所有關鍵點
    """
    if not obj_list:
        return None

    # 重要：依照原始 Class ID 排序 (確保關鍵點順序正確：先0後1)
    obj_list.sort(key=lambda x: x['orig_class'])

    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    merged_kpts = []

    for obj in obj_list:
        # 處理 BBox
        x1, y1, x2, y2 = to_abs_box(obj['bbox'], img_w, img_h)
        if x1 < min_x: min_x = x1
        if y1 < min_y: min_y = y1
        if x2 > max_x: max_x = x2
        if y2 > max_y: max_y = y2

        # 處理關鍵點
        merged_kpts.extend(obj['kpts_abs'])

    return {
        'bbox_abs': [min_x, min_y, max_x, max_y],
        'keypoints': merged_kpts
    }

# ================= 主程式 =================

def main():
    # 檢查輸入資料夾
    if not os.path.exists(INPUT_FOLDER):
        print(f"錯誤: 找不到輸入資料夾 {INPUT_FOLDER}")
        print("請確認 mtddh_xray_2d 資料夾是否已存在且包含 data 子資料夾。")
        return

    # 生成 classes.txt 在 yolo_labels 資料夾中
    classes_path = os.path.join(DIR_YOLO, 'classes.txt')
    with open(classes_path, 'w', encoding='utf-8') as f_cls:
        f_cls.write("LeftHip\nRightHip\n")
    print(f"已生成類別檔: {classes_path}")
    
    # 搜尋所有 txt 檔案
    txt_files = glob.glob(os.path.join(INPUT_FOLDER, '*.txt'))
    print(f"找到 {len(txt_files)} 筆資料，目標路徑：{BASE_DIR}/...")

    count = 0
    for txt_path in txt_files:
        filename_no_ext = os.path.splitext(os.path.basename(txt_path))[0]
        
        # 1. 尋找對應圖片
        jpg_path = os.path.join(INPUT_FOLDER, filename_no_ext + ".jpg")
        png_path = os.path.join(INPUT_FOLDER, filename_no_ext + ".png")
        
        src_img_path = None
        if os.path.exists(jpg_path): src_img_path = jpg_path
        elif os.path.exists(png_path): src_img_path = png_path
        
        if not src_img_path:
            # 可能是 outliers 刪除後遺留的 txt，或是缺失圖片
            continue

        # 讀取圖片資訊
        img = cv2.imread(src_img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]
        img_filename = os.path.basename(src_img_path)

        # 2. 複製圖片到 images 資料夾
        dst_img_path = os.path.join(DIR_IMAGES, img_filename)
        shutil.copy2(src_img_path, dst_img_path)

        # 3. 讀取並解析原始 TXT
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        raw_objects = []
        for line in lines:
            parts = list(map(float, line.strip().split()))
            cls_id = int(parts[0])
            bbox_norm = parts[1:5] # cx, cy, w, h
            
            # 解析關鍵點 (轉為絕對座標)
            kpts_raw = parts[5:]
            kpts_abs = []
            for i in range(0, len(kpts_raw), 3):
                kx = int(kpts_raw[i] * img_w)
                ky = int(kpts_raw[i+1] * img_h)
                kpts_abs.append((kx, ky))
            
            raw_objects.append({
                'orig_class': cls_id,
                'bbox': bbox_norm,
                'kpts_abs': kpts_abs,
                'center_x': bbox_norm[0]
            })

        # 4. 分組與合併 (Left / Right)
        left_group = [o for o in raw_objects if o['center_x'] < 0.5]
        right_group = [o for o in raw_objects if o['center_x'] >= 0.5]

        merged_left = merge_objects_data(left_group, img_w, img_h)
        merged_right = merge_objects_data(right_group, img_w, img_h)

        # -------------------------------------------------
        # 5. 生成 CSV (Annotations)
        # -------------------------------------------------
        csv_points = []
        if merged_left:
            for pt in merged_left['keypoints']:
                csv_points.append(f'"({pt[0]}, {pt[1]})"')
        if merged_right:
            for pt in merged_right['keypoints']:
                csv_points.append(f'"({pt[0]}, {pt[1]})"')
        
        csv_path = os.path.join(DIR_ANNOTATIONS, filename_no_ext + ".csv")
        with open(csv_path, 'w', encoding='utf-8') as f_csv:
            f_csv.write(",".join(csv_points))


        # -------------------------------------------------
        # 6. 生成 JSON (Detections)
        # -------------------------------------------------
        json_objects = []
        if merged_left:
            x1, y1, x2, y2 = merged_left['bbox_abs']
            json_objects.append({
                "label": "LeftHip",
                "points": [[x1, y1], [x2, y2]]
            })
        if merged_right:
            x1, y1, x2, y2 = merged_right['bbox_abs']
            json_objects.append({
                "label": "RightHip",
                "points": [[x1, y1], [x2, y2]]
            })
            
        json_data = {
            "image": img_filename,
            "objects": json_objects
        }
        json_path = os.path.join(DIR_DETECTIONS, filename_no_ext + ".json")
        with open(json_path, 'w', encoding='utf-8') as f_json:
            json.dump(json_data, f_json, indent=4)


        # -------------------------------------------------
        # 7. 生成 TXT (YOLO Labels)
        # -------------------------------------------------
        yolo_lines = []
        if merged_left:
            cx, cy, w, h = to_yolo_format(merged_left['bbox_abs'], img_w, img_h)
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if merged_right:
            cx, cy, w, h = to_yolo_format(merged_right['bbox_abs'], img_w, img_h)
            yolo_lines.append(f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
        yolo_path = os.path.join(DIR_YOLO, filename_no_ext + ".txt")
        with open(yolo_path, 'w') as f_yolo:
            f_yolo.write("\n".join(yolo_lines))
        
        count += 1

    print(f"轉換完成！共處理 {count} 筆資料。")
    print(f"輸出位置: {DIR_IMAGES}")

if __name__ == "__main__":
    main()