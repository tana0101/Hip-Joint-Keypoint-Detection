import argparse
import shutil
import json
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split

# 改腳本用於從 JSON 分割檔（dataset_splits.json）將資料集切分成 train/val/test，並產生 Ultralytics 相容的 data.yaml。
# 依照兩階段模型訓練需求，將 JSON 的 train pool 再切分 90% (inner train) / 10% (inner val)，
# 將原本的 val 獨立作為 test (SOTA測試集)。

def list_files_by_stem(folder: Path, exts):
    """
    回傳 {stem: Path}，僅納入副檔名在 exts 中的檔案（大小寫不敏感）。
    若同名多副檔名同時存在，以任一個為主（通常只有一個）。
    """
    exts = {e.lower() for e in exts}
    m = {}
    if not folder.exists():
        return m
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            m[p.stem] = p
    return m

def make_dirs_for_split(out_base: Path, split: str, present_modalities, out_name_map):
    """為每個 split 建立存在的 modality 子資料夾（含 yolo_labels -> labels 的映射）。"""
    for modal in present_modalities:
        out_modal = out_name_map.get(modal, modal)
        (out_base / split / out_modal).mkdir(parents=True, exist_ok=True)

def read_classes_from_source(dataset_dir: Path) -> List[str]:
    """
    從 dataset/<root>/yolo_labels/classes.txt 讀取類別名稱。
    若不存在，回傳空 list（後續會用標籤掃描回推 nc 與預設名稱）。
    """
    classes_txt = dataset_dir / "yolo_labels" / "classes.txt"
    if not classes_txt.exists():
        return []
    names = []
    for line in classes_txt.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name:
            names.append(name)
    return names

def write_data_yaml(out_base: Path, names_dict: dict) -> None:
    """
    產生 Ultralytics 相容的 data.yaml：
      train/val/test：指向各自的相對路徑
      nc：len(names)
      names：列表
    """
    train_dir = "./train"
    val_dir   = "./val"
    test_dir  = "./test"

    yaml_lines = [
        f"train: {train_dir}",
        f"val: {val_dir}",
        f"test: {test_dir}",
        "",
        f"nc: {len(names_dict)}",
        "names:",
    ]
    # names dict 逐行寫入（縮排兩個空白）
    for k in sorted(names_dict.keys()):
        yaml_lines.append(f"  {k}: {names_dict[k]}")
    yaml_lines.append("")
    (out_base / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

def main(args):
    dataset_dir = Path(args.dataset)
    assert dataset_dir.exists(), f"找不到資料夾：{dataset_dir}"

    splits_path = Path(args.splits)
    assert splits_path.exists(), f"找不到 JSON 分割檔：{splits_path}"

    out_base = Path(args.out)
    if out_base.exists():
        if args.overwrite:
            print(f"[INFO] 輸出資料夾 {out_base} 已存在，將被刪除後重建。")
            shutil.rmtree(out_base)
        else:
            raise FileExistsError(f"輸出資料夾 {out_base} 已存在。請使用 --overwrite 參數來允許覆蓋。")
    
    # 讀取 JSON
    with open(splits_path, 'r', encoding='utf-8') as f:
        splits_data = json.load(f)
    
    json_train = splits_data.get("splits", {}).get("train", [])
    json_val = splits_data.get("splits", {}).get("val", [])

    # 子資料夾（若不存在就視為缺席，不強制）
    subdirs = {
        "images": dataset_dir / "images",
        "annotations": dataset_dir / "annotations",
        "detections": dataset_dir / "detections",
        "yolo_labels": dataset_dir / "yolo_labels",
    }

    # 檔案讀取規則
    exts_map = {
        "images": {".jpg", ".jpeg", ".png"},
        "annotations": {".csv"},
        "detections": {".json"},
        "yolo_labels": {".txt"},
    }

    # yolo_labels 在輸出時要改名為 labels
    out_name_map = {"yolo_labels": "labels"}

    # 逐 modality 建 stem 索引
    stem_maps = {}
    present_modalities = []
    for modal, folder in subdirs.items():
        m = list_files_by_stem(folder, exts_map[modal])
        if m:
            present_modalities.append(modal)
        stem_maps[modal] = m

    # 至少要有 images
    assert stem_maps["images"], f"{subdirs['images']} 內未找到任何影像檔（.jpg/.jpeg/.png）"
    if not present_modalities:
        raise RuntimeError("未偵測到任何可用的子資料夾與檔案。")

    # 用存在的 modalities 的 stem 做交集，確保同一筆都有對應檔
    common_stems = set(stem_maps["images"].keys())
    for modal in present_modalities:
        common_stems &= set(stem_maps[modal].keys())

    # 篩選存在於實際資料夾中的 JSON 名單 (加入 sorted 防止 Order Trap)
    train_pool_stems = sorted([s for s in json_train if s in common_stems])
    
    # 原本的 JSON val，現在正式成為獨立的 Outer Test
    test_stems = sorted([s for s in json_val if s in common_stems])
    
    # 從 train pool 中再切分出 inner train 與 inner val
    train_stems, val_stems = train_test_split(
        train_pool_stems,
        test_size=args.val_ratio,
        random_state=args.seed,
        shuffle=True
    )

    # 檢查是否有在 JSON 內但實際檔案缺失的情況
    missing_train = set(json_train) - set(train_pool_stems)
    missing_val = set(json_val) - set(test_stems)
    if missing_train:
        print(f"[WARN] JSON 中的 train 檔案遺失或資料不齊全：{len(missing_train)} 筆")
    if missing_val:
        print(f"[WARN] JSON 中的 val 檔案遺失或資料不齊全：{len(missing_val)} 筆")

    out_base = Path(args.out)
    for split in ["train", "val", "test"]:
        make_dirs_for_split(out_base, split, present_modalities, out_name_map)

    def copy_one(stem, split):
        for modal in present_modalities:
            src = stem_maps[modal][stem]
            out_modal = out_name_map.get(modal, modal)
            dst = out_base / split / out_modal / src.name
            shutil.copy2(src, dst)

    # 執行複製
    print("[INFO] 正在複製 train 檔案...")
    for stem in train_stems:
        copy_one(stem, "train")
    print("[INFO] 正在複製 val 檔案...")
    for stem in val_stems:
        copy_one(stem, "val")
    print("[INFO] 正在複製 test 檔案...")
    for stem in test_stems:
        copy_one(stem, "test")

    # 讀 classes.txt（若無則從輸出的 labels/ 推估）
    names = read_classes_from_source(dataset_dir)

    # 產生 data.yaml
    names_dict = {i: n for i, n in enumerate(names)}
    write_data_yaml(out_base, names_dict)

    # 總結
    print("\n=== Split Summary ===")
    print(f"Dataset root      : {dataset_dir}")
    print(f"Splits JSON       : {splits_path}")
    print(f"Output root       : {out_base}")
    print(f"Random Seed       : {args.seed}")
    print(f"Present modalities: {present_modalities}（輸出時 yolo_labels -> labels）")
    print("-" * 30)
    print(f"  -> Train Pool (JSON Train) : {len(train_pool_stems)}")
    print(f"       ├─ inner train ({100 - args.val_ratio * 100:.0f}%)  : {len(train_stems)}")
    print(f"       └─ inner val   ({args.val_ratio * 100:.0f}%)  : {len(val_stems)}")
    print(f"  -> Outer Test (JSON Val)   : {len(test_stems)}")
    print("-" * 30)
    print(f"[OK] 已產生 {out_base / 'data.yaml'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset using a JSON file and emit Ultralytics data.yaml with inner validation split")
    parser.add_argument("--dataset", required=True, help="Root directory of the dataset, e.g., dataset/mtddh_xray_2d")
    parser.add_argument("--splits", required=True, help="Path to the dataset_splits.json file")
    parser.add_argument("--out", default="data", help="Output root directory (default: data)")
    parser.add_argument("--overwrite", action="store_true", help="If set, removes the existing output directory before splitting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for inner split")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of inner validation split from train pool")
    args = parser.parse_args()

    main(args)
    
'''
python split_from_json.py \
  --dataset dataset/mtddh_xray_2d \
  --splits dataset/mtddh_xray_2d/dataset_splits.json \
  --out data \
  --overwrite \
  --seed 42 \
  --val_ratio 0.1
'''