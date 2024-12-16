import os
import csv
import re

# 定義用於解析點的正則表達式
point_pattern = re.compile(r"\(\d+, \d+\)")

def check_csv_points(directory):
    # 紀錄點數量不正確的檔案
    invalid_files = []

    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(directory):
        # 確保只處理 CSV 檔案
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)

            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    reader = csv.reader(file)
                    for row in reader:
                        # 將所有行連接成一個字串，然後解析點
                        row_content = ",".join(row)
                        points = point_pattern.findall(row_content)

                        # 如果點的數量不是 12，記錄檔名
                        if len(points) != 12:
                            invalid_files.append(filename)
                            break  # 無需繼續檢查此檔案

            except Exception as e:
                print(f"無法處理檔案 {filename}: {e}")

    # 列出點數量不正確的檔案
    if invalid_files:
        print("以下檔案的點數量不為 12:")
        for file in invalid_files:
            print(file)
    else:
        print("所有檔案的點數量均為 12！")

# 指定要檢查的資料夾
folder_path = "annotations"
check_csv_points(folder_path)