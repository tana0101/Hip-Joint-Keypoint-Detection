import cv2
import os
import glob

def visualize_yolo_keypoints(data_dir='data', output_dir='output'):
    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)

    # 取得所有的 jpg 和 png 圖片
    image_paths = glob.glob(os.path.join(data_dir, '*.jpg')) + glob.glob(os.path.join(data_dir, '*.png'))

    if not image_paths:
        print(f"在 {data_dir} 資料夾中找不到任何圖片！")
        return

    for img_path in image_paths:
        # 取得對應的 txt 檔案路徑
        base_name = os.path.splitext(img_path)[0]
        txt_path = base_name + '.txt'
        
        # 若沒有標註檔則跳過
        if not os.path.exists(txt_path):
            print(f"找不到標註檔，跳過: {img_path}")
            continue
            
        # 讀取圖片
        img = cv2.imread(img_path)
        if img is None:
            print(f"無法讀取圖片，跳過: {img_path}")
            continue
            
        h, w, _ = img.shape
        
        # 讀取標註檔
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            x_center, y_center, bbox_w, bbox_h = parts[1:5]
            
            # 1. 還原 Bounding Box 座標
            xmin = int((x_center - bbox_w / 2) * w)
            ymin = int((y_center - bbox_h / 2) * h)
            xmax = int((x_center + bbox_w / 2) * w)
            ymax = int((y_center + bbox_h / 2) * h)
            
            # 畫出綠色框框 (B, G, R) = (0, 255, 0)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # 2. 還原關鍵點 (每 3 個數值為一組: x, y, visibility)
            keypoints = parts[5:]
            for i in range(0, len(keypoints), 3):
                if i + 2 < len(keypoints):
                    kx, ky, kconf = keypoints[i], keypoints[i+1], keypoints[i+2]
                    
                    # YOLO 關鍵點可見度: 0 (未標註), 1 (遮擋但標註), 2 (可見)
                    # 只要大於 0 就畫出來
                    if int(kconf) > 0:
                        kx_pixel = int(kx * w)
                        ky_pixel = int(ky * h)
                        
                        # 畫出紅色點點 (B, G, R) = (0, 0, 255)
                        cv2.circle(img, (kx_pixel, ky_pixel), 4, (0, 0, 255), -1)
                        
        # 儲存結果
        out_filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, img)
        print(f"已儲存可視化結果: {out_path}")

if __name__ == "__main__":
    # 假設你的資料放在 'data' 資料夾內
    visualize_yolo_keypoints(data_dir='mtddh_xray_2d/data', output_dir='mtddh_xray_2d/output_visualized')