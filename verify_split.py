import argparse
from pathlib import Path

def extract_stems_from_txt(txt_path: Path) -> set:
    """從 YOLO 的 txt 檔中讀取每一行路徑，並提取檔名 (stem)"""
    if not txt_path.exists():
        raise FileNotFoundError(f"找不到 YOLO 的 txt 檔: {txt_path}")
    
    stems = set()
    with txt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                stems.add(Path(line).stem)
    return stems

def extract_stems_from_dir(dir_path: Path) -> set:
    """從關鍵點模型的資料夾中讀取所有圖片，並提取檔名 (stem)"""
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"找不到關鍵點的資料夾: {dir_path}")
        
    img_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    stems = set()
    for ext in img_exts:
        for file in dir_path.glob(f"*{ext}"):
            stems.add(file.stem)
    return stems

def verify_sets(name: str, yolo_set: set, kp_set: set):
    """比對兩個 Set 是否完全一致，並印出詳細報告"""
    print("-" * 50)
    print(f"[{name} 檢查]")
    print(f"YOLO 數量: {len(yolo_set)}")
    print(f"Keypoint 數量: {len(kp_set)}")
    
    if yolo_set == kp_set:
        print(f"✅ 完美吻合！兩邊的 {name} 資料 100% 相同。")
    else:
        print(f"❌ 警告！兩邊的 {name} 資料不一致！")
        # 找出是誰多了什麼、少了什麼
        yolo_only = yolo_set - kp_set
        kp_only = kp_set - yolo_set
        
        if yolo_only:
            print(f"  -> YOLO 多出了 {len(yolo_only)} 張圖，例如: {list(yolo_only)[:5]}")
        if kp_only:
            print(f"  -> Keypoint 多出了 {len(kp_only)} 張圖，例如: {list(kp_only)[:5]}")

def main():
    parser = argparse.ArgumentParser(description="驗證 YOLO 與 Keypoint 的資料切割是否一致")
    parser.add_argument("--yolo_train_txt", type=str, required=True, help="YOLO 的 train.txt 路徑")
    parser.add_argument("--yolo_val_txt", type=str, required=True, help="YOLO 的 val.txt 路徑")
    parser.add_argument("--kp_train_dir", type=str, required=True, help="Keypoint 的 train 資料夾路徑")
    parser.add_argument("--kp_val_dir", type=str, required=True, help="Keypoint 的 val 資料夾路徑")
    args = parser.parse_args()

    # 1. 取得 Train 的檔名集合
    yolo_train_stems = extract_stems_from_txt(Path(args.yolo_train_txt))
    kp_train_stems = extract_stems_from_dir(Path(args.kp_train_dir))
    
    # 2. 取得 Val 的檔名集合
    yolo_val_stems = extract_stems_from_txt(Path(args.yolo_val_txt))
    kp_val_stems = extract_stems_from_dir(Path(args.kp_val_dir))

    print("=" * 50)
    print("啟動資料外洩 (Data Leakage) 防禦檢查...")
    
    # 3. 執行嚴格比對
    verify_sets("訓練集 (Inner Train)", yolo_train_stems, kp_train_stems)
    verify_sets("驗證集 (Inner Val)", yolo_val_stems, kp_val_stems)
    
    # 4. 終極安全檢查：確保 Train 和 Val 彼此之間沒有交集 (以防萬一自己寫的切割邏輯有 bug)
    train_val_overlap = yolo_train_stems.intersection(yolo_val_stems)
    print("-" * 50)
    if len(train_val_overlap) == 0:
        print("✅ 安全檢查通過：Train 和 Val 之間沒有重複的圖片。")
    else:
        print(f"💀 致命錯誤！Train 和 Val 之間有 {len(train_val_overlap)} 張圖片重複發生了資料洩漏！")
        print(f"  -> 重複的圖片例如: {list(train_val_overlap)[:5]}")
    print("=" * 50)

if __name__ == "__main__":
    main()
    
'''
python verify_split.py \
  --yolo_train_txt data/train_inner_fold5.txt \
  --yolo_val_txt data/val_inner_fold5.txt \
  --kp_train_dir /home/p76134888/Hip-Joint-Keypoint-Detection/data/kfold_tmp/outer_inner/fold5/train/images \
  --kp_val_dir /home/p76134888/Hip-Joint-Keypoint-Detection/data/kfold_tmp/outer_inner/fold5/val/images
'''