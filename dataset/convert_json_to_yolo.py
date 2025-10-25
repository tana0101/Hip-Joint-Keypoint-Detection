# convert_to_yolo.py
import argparse, json
from pathlib import Path
from PIL import Image

def to_yolo(xmin, ymin, xmax, ymax, W, H):
    # clamp 防呆
    xmin = max(0, min(xmin, W-1))
    xmax = max(0, min(xmax, W-1))
    ymin = max(0, min(ymin, H-1))
    ymax = max(0, min(ymax, H-1))
    # 修正可能顛倒的座標
    if xmax < xmin: xmin, xmax = xmax, xmin
    if ymax < ymin: ymin, ymax = ymax, ymin

    cx = ((xmin + xmax) / 2.0) / W
    cy = ((ymin + ymax) / 2.0) / H
    w  = (xmax - xmin) / W
    h  = (ymax - ymin) / H
    return cx, cy, w, h

def main():
    ap = argparse.ArgumentParser(description="Convert JSON boxes (LT,RB) to YOLO format.")
    ap.add_argument("--images", default="images", help="原始圖片資料夾")
    ap.add_argument("--dets",   default="detections", help="JSON 標註資料夾")
    ap.add_argument("--out",    default="labels_yolo", help="輸出 YOLO 標籤資料夾")
    ap.add_argument("--classes", nargs="*", default=["LeftHip","RightHip"],
                    help="類別名稱依序對應成 0,1,2,...")
    args = ap.parse_args()

    img_dir = Path(args.images)
    det_dir = Path(args.dets)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 類別對應：name -> id
    name2id = {name: i for i, name in enumerate(args.classes)}

    # 寫 classes.txt（Ultralytics 風格）
    with open(out_dir / "classes.txt", "w", encoding="utf-8") as f:
        for name in args.classes:
            f.write(f"{name}\n")

    json_files = sorted(det_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] 在 {det_dir} 沒有找到 json。")
        return

    total, written = 0, 0
    for jf in json_files:
        total += 1
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[SKIP] 讀取 {jf.name} 失敗：{e}")
            continue

        img_name = data.get("image")
        if not img_name:
            print(f"[SKIP] {jf.name} 缺少 'image' 欄位")
            continue

        # 支援常見附檔名
        img_path = img_dir / img_name
        if not img_path.exists():
            # 嘗試同名不同副檔
            stem = Path(img_name).stem
            candidates = list(img_dir.glob(stem + ".*"))
            img_path = candidates[0] if candidates else img_path

        if not img_path.exists():
            print(f"[SKIP] 找不到圖片：{img_path}")
            continue

        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception as e:
            print(f"[SKIP] 無法開啟圖片 {img_path.name}：{e}")
            continue

        objects = data.get("objects", [])
        yolo_lines = []
        for obj in objects:
            label = obj.get("label")
            pts = obj.get("points", [])
            if label not in name2id:
                # 不在類別表就略過（也可選擇 raise）
                print(f"[NOTE] 未在 classes 中的標籤 '{label}'（檔案 {jf.name}）→ 略過")
                continue
            if not (isinstance(pts, list) and len(pts) == 2 and all(len(p)==2 for p in pts)):
                print(f"[NOTE] {jf.name} 標註 points 格式異常 → 略過該物件")
                continue

            (x1, y1), (x2, y2) = pts
            xmin, ymin = x1, y1
            xmax, ymax = x2, y2
            cx, cy, w, h = to_yolo(xmin, ymin, xmax, ymax, W, H)

            # 夾在 [0,1] 範圍內
            cx = max(0,min(1,cx)); cy = max(0,min(1,cy))
            w  = max(0,min(1,w));  h  = max(0,min(1,h))

            cls_id = name2id[label]
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 輸出成同檔名 .txt
        out_txt = out_dir / (img_path.stem + ".txt")
        if yolo_lines:
            out_txt.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            written += 1
        else:
            # 若沒有任何有效物件，寫空檔或選擇不寫；這裡選擇寫空檔（Ultralytics 可接受）
            out_txt.write_text("", encoding="utf-8")

    print(f"[DONE] 共處理 {total} 份 json，成功輸出 {written} 份標籤至 {out_dir}")

if __name__ == "__main__":
    main()

# python convert_json_to_yolo.py --images xray_IHDI_3/images --dets xray_IHDI_3/detections --out xray_IHDI_3/yolo_labels --classes LeftHip RightHip