# split.py
import argparse
import shutil
import random
from pathlib import Path
from typing import List

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

def main(dataset: str, out: str = "data",
         train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
         seed: int = 42):
    random.seed(seed)

    dataset_dir = Path(dataset)
    assert dataset_dir.exists(), f"找不到資料夾：{dataset_dir}"

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

    # 若有 modality 缺檔會被排除，給出提示
    total_images = len(stem_maps["images"])
    kept = len(common_stems)
    dropped = total_images - kept
    if dropped > 0:
        print(f"[WARN] 有 {dropped} 張影像無法在所有 present modalities 中找到對應檔，將被排除。")
    if kept == 0:
        raise RuntimeError("交集為空，請確認子資料夾中的檔名（不含副檔名）對得上。")

    # 依比例切分
    common_stems = sorted(common_stems)
    random.shuffle(common_stems)

    n = len(common_stems)
    # 檢查比例
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"train+val+test 必須等於 1，當前為 {s}")

    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    test_size = n - train_size - val_size

    train_stems = common_stems[:train_size]
    val_stems = common_stems[train_size:train_size + val_size]
    test_stems = common_stems[train_size + val_size:]

    out_base = Path(out)
    for split in ["train", "val", "test"]:
        make_dirs_for_split(out_base, split, present_modalities, out_name_map)

    def copy_one(stem, split):
        for modal in present_modalities:
            src = stem_maps[modal][stem]
            out_modal = out_name_map.get(modal, modal)
            dst = out_base / split / out_modal / src.name
            shutil.copy2(src, dst)

    for stem in train_stems:
        copy_one(stem, "train")
    for stem in val_stems:
        copy_one(stem, "val")
    for stem in test_stems:
        copy_one(stem, "test")

    # 讀 classes.txt（若無則從輸出的 labels/ 推估）
    names = read_classes_from_source(dataset_dir)

    # 產生 data.yaml
    names_dict = {i: n for i, n in enumerate(names)}
    write_data_yaml(out_base, names_dict)

    # 總結
    print("=== Split Summary ===")
    print(f"Dataset root    : {dataset_dir}")
    print(f"Output root     : {out_base}")
    print(f"Present modalities : {present_modalities}（輸出時 yolo_labels -> labels）")
    print(f"Total images (in images/)    : {total_images}")
    print(f"Usable samples (intersection): {kept}")
    print(f"  -> train: {len(train_stems)}")
    print(f"  -> val  : {len(val_stems)}")
    print(f"  -> test : {len(test_stems)}")
    print(f"[OK] 已產生 {out_base / 'data.yaml'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test with multiple modalities and emit Ultralytics data.yaml")
    parser.add_argument("--dataset", required=True, help="Root directory of the dataset, e.g., dataset/xray_IHDI_5")
    parser.add_argument("--out", default="data", help="Output root directory (default: data)")
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--val", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="Test split ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        out=args.out,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )

# python split.py --dataset dataset/xray_IHDI_6 --out data --train 0.8 --val 0.1 --test 0.1 --seed 42