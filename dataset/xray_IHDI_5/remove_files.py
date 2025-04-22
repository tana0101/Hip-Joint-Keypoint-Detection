import os

# 設定檔案路徑
remove_file = 'remove.txt'
images_dir = 'images'
annotations_dir = 'annotations'

# 讀取要刪除的檔名清單
with open(remove_file, 'r') as f:
    filenames = [line.strip() for line in f if line.strip()]

# 開始刪除
for name in filenames:
    img_path = os.path.join(images_dir, name + '.jpg')
    ann_path = os.path.join(annotations_dir, name + '.csv')

    # 刪除圖像檔
    if os.path.exists(img_path):
        os.remove(img_path)
        print(f"Deleted image: {img_path}")
    else:
        print(f"Image not found: {img_path}")

    # 刪除標註檔
    if os.path.exists(ann_path):
        os.remove(ann_path)
        print(f"Deleted annotation: {ann_path}")
    else:
        print(f"Annotation not found: {ann_path}")
