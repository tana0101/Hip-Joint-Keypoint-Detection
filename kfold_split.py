import argparse
import shutil
from pathlib import Path
from typing import List, Optional
from sklearn.model_selection import KFold


def list_images(img_dir: Path, exts: List[str]) -> List[Path]:
    files = []
    for ext in exts:
        files.extend(img_dir.glob(f"*{ext}"))
    files = sorted(files)
    return files


def find_single_with_stem(root: Path, stem: str) -> Optional[Path]:
    """
    在指定資料夾底下找檔名 stem.* 的檔案。
    - 若 0 個 → 回傳 None
    - 若 1 個 → 回傳該 Path
    - 若 >1 個 → raise ValueError（違反「不會有多個同類型標註」假設）
    """
    if not root.is_dir():
        return None
    matches = list(root.glob(stem + ".*"))
    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"[ERROR] Multiple files found in {root} with stem '{stem}': {matches}")


def copy_if_exists(src: Optional[Path], dst_dir: Path):
    if src is None:
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)


def main(args):
    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    k = args.k
    seed = args.seed

    images_dir = src_root / "images"
    ann_dir = src_root / "annotations"
    det_dir = src_root / "detections"
    yolo_dir = src_root / "yolo_labels"

    assert images_dir.is_dir(), f"{images_dir} not found"
    if not ann_dir.is_dir():
        print(f"[WARN] {ann_dir} not found, will skip annotations")
    if not det_dir.is_dir():
        print(f"[WARN] {det_dir} not found, will skip detections")
    if not yolo_dir.is_dir():
        print(f"[WARN] {yolo_dir} not found, will skip yolo_labels")

    img_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    all_images = list_images(images_dir, img_exts)
    if len(all_images) == 0:
        raise RuntimeError(f"No images found in {images_dir}")

    print(f"[INFO] Total images found: {len(all_images)}")

    if dst_root.exists() and args.overwrite:
        print(f"[INFO] Removing existing dst_root: {dst_root}")
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    # KFold on indices
    indices = list(range(len(all_images)))
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    # fold_idx: 1..k
    folds_img_relpaths = {fold_idx: [] for fold_idx in range(1, k + 1)}

    for fold_idx, (_, test_idx) in enumerate(kf.split(indices), start=1):
        fold_name = f"fold{fold_idx}"
        fold_root = dst_root / fold_name

        fold_img_dir = fold_root / "images"
        fold_ann_dir = fold_root / "annotations"
        fold_det_dir = fold_root / "detections"
        fold_yolo_dir = fold_root / "labels"

        print(f"[INFO] Creating {fold_name}: {len(test_idx)} images")

        for i in test_idx:
            img_path = all_images[i]
            stem = img_path.stem

            # 對應標註檔
            ann_path = find_single_with_stem(ann_dir, stem)
            det_path = find_single_with_stem(det_dir, stem)
            yolo_path = find_single_with_stem(yolo_dir, stem)

            # 複製 image
            fold_img_dir.mkdir(parents=True, exist_ok=True)
            dst_img_path = fold_img_dir / img_path.name
            shutil.copy2(img_path, dst_img_path)

            # 對應標註
            copy_if_exists(ann_path, fold_ann_dir)
            copy_if_exists(det_path, fold_det_dir)
            copy_if_exists(yolo_path, fold_yolo_dir)

            # 記錄這張圖的「絕對路徑」，給 train/val txt 用
            abs_path = dst_img_path.resolve()
            folds_img_relpaths[fold_idx].append(abs_path.as_posix())

        print(f"[INFO] {fold_name} done at {fold_root}")

    # ==== 產生每一折的 train/val txt + data_fold{i}.yaml ====
    for fold_idx in range(1, k + 1):
        # val = 該 fold
        val_list = folds_img_relpaths[fold_idx]
        # train = 其它所有 fold
        train_list = []
        for f in range(1, k + 1):
            if f == fold_idx:
                continue
            train_list.extend(folds_img_relpaths[f])

        print(f"[INFO] Fold {fold_idx}: train = {len(train_list)}, val = {len(val_list)}")

        train_txt = dst_root / f"train_fold{fold_idx}.txt"
        val_txt = dst_root / f"val_fold{fold_idx}.txt"

        with train_txt.open("w") as f:
            for p in train_list:
                f.write(p + "\n")

        with val_txt.open("w") as f:
            for p in val_list:
                f.write(p + "\n")

        # YOLO data yaml
        yaml_path = dst_root / f"data_fold{fold_idx}.yaml"
        with yaml_path.open("w") as f:
            f.write(f"train: {train_txt.name}\n")
            f.write(f"val: {val_txt.name}\n")
            f.write("names:\n")
            f.write("  0: hip_left\n")
            f.write("  1: hip_right\n")

        print(f"[INFO] YOLO yaml for fold {fold_idx}: {yaml_path}")

    print("[INFO] All folds and data_fold{i}.yaml created successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-fold split for object detection and keypoint datasets")
    parser.add_argument(
        "--src", type=str, required=True,
        help="Source directory containing images/, annotations/, detections/, yolo_labels/"
    )
    parser.add_argument(
        "--dst", type=str, required=True,
        help="Destination directory where fold1..foldK and data_fold{i}.yaml will be created"
    )
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="If the destination directory exists, delete it before creating new folds"
    )

    args = parser.parse_args()
    main(args)
    
'''
python kfold_split.py \
  --src dataset/xray_IHDI_6 \
  --dst data \
  --k 5 \
  --seed 42 \
  --overwrite
'''