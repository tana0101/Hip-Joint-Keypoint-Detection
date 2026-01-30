import json
import os

# ================= 設定 =================
JSON_FILE = 'outliers.json'  # 與腳本在同一層
DATA_FOLDER = os.path.join('mtddh_xray_2d', 'data') 

# 要刪除的副檔名列表
TARGET_EXTENSIONS = ['.jpg', '.png', '.txt']
# ========================================

def delete_outliers():
    # 1. 檢查 JSON 檔案是否存在
    if not os.path.exists(JSON_FILE):
        print(f"錯誤: 找不到 {JSON_FILE}")
        return

    # 檢查目標資料夾是否存在 (新增的安全檢查)
    if not os.path.exists(DATA_FOLDER):
        print(f"錯誤: 找不到資料資料夾 {DATA_FOLDER}")
        print("請確認是否已經將 mtddh_xray_2d clone 到此目錄下。")
        return

    # 2. 讀取 JSON
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. 收集所有要刪除的檔案名稱 (不含副檔名)
    files_to_remove = set()
    
    for category, file_list in data.items():
        for filename in file_list:
            files_to_remove.add(filename)

    print(f"從 JSON 中共找到 {len(files_to_remove)} 個不重複的檔案名稱需要刪除。")

    # 4. 執行刪除
    deleted_count = 0
    not_found_count = 0

    for filename_no_ext in files_to_remove:
        file_deleted_in_this_loop = False
        
        for ext in TARGET_EXTENSIONS:
            file_path = os.path.join(DATA_FOLDER, filename_no_ext + ext)
            
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"已刪除: {file_path}")
                    deleted_count += 1
                    file_deleted_in_this_loop = True
                except OSError as e:
                    print(f"刪除失敗: {file_path}. 原因: {e}")
            else:
                pass
        
        if not file_deleted_in_this_loop:
             not_found_count += 1

    print("-" * 30)
    print("清理完成！")
    print(f"共刪除檔案數: {deleted_count}")
    if not_found_count > 0:
        print(f"有 {not_found_count} 組檔案名稱在資料夾中未找到任何對應檔案 (可能已被刪除或路徑錯誤)。")

if __name__ == "__main__":
    delete_outliers()