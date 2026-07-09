import cv2
import os
import glob
import json
import re

# ================= 設定區域 =================
# 定義基礎路徑
BASE_DIR = 'xray_IHDI_2'

# 輸入資料夾 (指向 mtddh_xray_2d 內部)
DIR_IMAGES = os.path.join(BASE_DIR, 'images')
DIR_ANNOTATIONS = os.path.join(BASE_DIR, 'annotations') # CSV
DIR_DETECTIONS = os.path.join(BASE_DIR, 'detections')   # JSON
DIR_YOLO = os.path.join(BASE_DIR, 'yolo_labels')        # TXT

# 視覺化結果輸出位置 (指向 mtddh_xray_2d 內部)
OUTPUT_VIS_FOLDER = os.path.join(BASE_DIR, 'vis_output')

# 顏色定義 (BGR)
COLOR_LEFT = (0, 0, 255)     # 紅色 (JSON LeftHip)
COLOR_RIGHT = (255, 0, 0)    # 藍色 (JSON RightHip)
COLOR_YOLO = (0, 255, 0)     # 綠色 (YOLO 驗證框)
COLOR_POINT = (0, 255, 255)  # 黃色 (關鍵點)
COLOR_TEXT_BG = (0, 0, 0)    # 黑色 (文字邊框)
COLOR_TEXT_FG = (255, 255, 255) # 白色 (文字本體)

# ================= 工具函式 =================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_csv_points(csv_path):
    """
    解析格式如 "(453, 562)","(501, 590)" 的 CSV 內容
    回傳 [(x, y), (x, y), ...]
    """
    points = []
    if not os.path.exists(csv_path):
        return points

    with open(csv_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 使用正規表達式找出所有的數字
        nums = re.findall(r'\d+', content)
        # 將數字轉為 int 並兩兩一組
        nums = [int(n) for n in nums]
        for i in range(0, len(nums), 2):
            if i + 1 < len(nums):
                points.append((nums[i], nums[i+1]))
    return points

def draw_text_with_outline(img, text, pos, font_scale=2, thickness=1):
    """畫出有黑色邊框的白色文字，確保清晰可見"""
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT_BG, thickness + 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT_FG, thickness)

def unnormalize_yolo(yolo_line, img_w, img_h):
    """將 YOLO 歸一化數據轉回 xyxy"""
    parts = list(map(float, yolo_line.strip().split()))
    cls_id = int(parts[0])
    cx, cy, w, h = parts[1:5]
    
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return cls_id, x1, y1, x2, y2

# ================= 主程式 =================

def main():
    # 檢查輸入目錄是否存在
    if not os.path.exists(DIR_IMAGES):
        print(f"錯誤：找不到圖片資料夾 {DIR_IMAGES}")
        print(f"請確認 {BASE_DIR} 是否已正確建立並包含 images 子目錄。")
        return

    ensure_dir(OUTPUT_VIS_FOLDER)

    # 搜尋 images 資料夾中的圖片
    image_files = glob.glob(os.path.join(DIR_IMAGES, '*.*'))
    # 過濾出 jpg 或 png
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.png'))]

    print(f"在 {DIR_IMAGES} 找到 {len(image_files)} 張圖片，開始視覺化...")
    print(f"結果將輸出至: {OUTPUT_VIS_FOLDER}")

    count = 0
    for img_path in image_files:
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]

        img = cv2.imread(img_path)
        if img is None:
            print(f"無法讀取: {img_path}")
            continue
        
        h, w = img.shape[:2]

        # ------------------------------------------------
        # 1. 繪製 YOLO Labels (綠色細框)
        # ------------------------------------------------
        yolo_path = os.path.join(DIR_YOLO, basename + ".txt")
        if os.path.exists(yolo_path):
            with open(yolo_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    cls_id, x1, y1, x2, y2 = unnormalize_yolo(line, w, h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_YOLO, 1)
                    cv2.putText(img, f"YOLO:{cls_id}", (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YOLO, 1)

        # ------------------------------------------------
        # 2. 繪製 Detections JSON (紅/藍 粗框)
        # ------------------------------------------------
        json_path = os.path.join(DIR_DETECTIONS, basename + ".json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for obj in data.get('objects', []):
                label = obj['label']
                pts = obj['points'] # [[x1, y1], [x2, y2]]
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                
                color = COLOR_LEFT if label == "LeftHip" else COLOR_RIGHT
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # ------------------------------------------------
        # 3. 繪製 Annotations CSV (關鍵點與編號)
        # ------------------------------------------------
        csv_path = os.path.join(DIR_ANNOTATIONS, basename + ".csv")
        points = parse_csv_points(csv_path)
        
        for idx, (px, py) in enumerate(points):
            cv2.circle(img, (px, py), 8, COLOR_POINT, -1)
            draw_text_with_outline(img, str(idx + 1), (px + 8, py - 8))

        # ------------------------------------------------
        # 儲存結果
        # ------------------------------------------------
        output_path = os.path.join(OUTPUT_VIS_FOLDER, f"vis_{filename}")
        cv2.imwrite(output_path, img)
        count += 1

    print(f"視覺化完成！共處理 {count} 張圖片。")
    print(f"請查看資料夾: {OUTPUT_VIS_FOLDER}")

if __name__ == "__main__":
    main()