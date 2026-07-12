#!/usr/bin/env python3
import argparse
from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import train_test_split


def list_images(img_dir: Path, exts: list) -> list:
    """與關鍵點模型一模一樣的讀取邏輯，確保排序相同"""
    files = []
    for ext in exts:
        files.extend(img_dir.glob(f"*{ext}"))
    files = sorted(files)
    return files


def prepare_inner_yaml(data_dir: Path, k: int, fold_idx: int, seed: int = 42, inner_val_ratio: float = 0.1) -> Path:
    """
    完美複製關鍵點模型的切分邏輯：
    1. 收集 K-1 個 fold 的所有影像 (Outer Train Pool)
    2. 使用 train_test_split 切出 Inner Train / Inner Val
    3. 生成 YOLO 專屬的 txt 與 yaml
    """
    img_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    train_pool_imgs = []

    # 必須跟關鍵點模型一樣，依序 1~K 讀取並 sorted，確保丟入 split 的 list 順序完全一致
    for i in range(1, k + 1):
        if i == fold_idx:
            continue
        fold_dir = data_dir / f"fold{i}"
        images_dir = fold_dir / "images"
        if not images_dir.is_dir():
            raise RuntimeError(f"{images_dir} not found")
        
        imgs = list_images(images_dir, img_exts)
        train_pool_imgs.extend(imgs)

    # 進行 90/10 切分
    train_imgs, val_imgs = train_test_split(
        train_pool_imgs,
        test_size=inner_val_ratio,
        random_state=seed,
        shuffle=True,
    )

    # 準備寫入的路徑 (放在 data 目錄下)
    inner_train_txt = data_dir / f"train_inner_fold{fold_idx}.txt"
    inner_val_txt = data_dir / f"val_inner_fold{fold_idx}.txt"
    inner_yaml_path = data_dir / f"data_inner_fold{fold_idx}.yaml"

    # 將絕對路徑寫入 txt，供 YOLO 讀取
    with inner_train_txt.open("w") as f:
        for p in train_imgs:
            f.write(p.resolve().as_posix() + "\n")

    with inner_val_txt.open("w") as f:
        for p in val_imgs:
            f.write(p.resolve().as_posix() + "\n")

    # 生成本次訓練專用的 YAML 檔
    with inner_yaml_path.open("w") as f:
        f.write(f"train: {inner_train_txt.as_posix()}\n")
        f.write(f"val: {inner_val_txt.as_posix()}\n")
        f.write("names:\n")
        f.write("  0: hip_left\n")
        f.write("  1: hip_right\n")

    print(f"[YOLO Data Sync] Fold {fold_idx}: Inner Train={len(train_imgs)}, Inner Val={len(val_imgs)}")
    return inner_yaml_path


def train_one_fold(args, fold_idx: int):
    # 1. 取得資料集根目錄 (假設 args.data_dir 就是 "data")
    data_dir = Path(args.data_dir).resolve()
    
    # 2. 動態生成完美同步的 inner yaml
    data_yaml = prepare_inner_yaml(
        data_dir=data_dir, 
        k=args.k, 
        fold_idx=fold_idx, 
        seed=args.inner_seed,
        inner_val_ratio=args.inner_val_ratio
    )

    run_name = f"{args.name}_fold{fold_idx}"

    print("=" * 80)
    print(f"[YOLO KFold] Start training fold {fold_idx}/{args.k}")
    print(f"  model      : {args.model}")
    print(f"  data_yaml  : {data_yaml}")
    print(f"  project    : {args.project}")
    print(f"  run name   : {run_name}")
    print("=" * 80)

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=run_name,
        pretrained=args.pretrained,
        resume=args.resume,
        seed=args.seed,
        fliplr=args.fliplr,
        flipud=args.flipud,
        degrees=args.degrees,
        shear=args.shear,
        perspective=args.perspective,
        mosaic=args.mosaic,
        mixup=args.mixup,
    )

    print(f"[YOLO KFold] Fold {fold_idx} training finished.\n")


def main():
    p = argparse.ArgumentParser(description="K-fold training wrapper for YOLO with Inner Split.")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO model weights or cfg")
    # 改為傳入資料集根目錄，讓腳本自己找 fold1~foldK
    p.add_argument("--data_dir", default="data", help="Directory containing fold1..foldK")
    p.add_argument("--k", type=int, default=5, help="number of folds")
    p.add_argument("--inner_val_ratio", type=float, default=0.1, help="Inner validation ratio for train_test_split")
    p.add_argument("--inner_seed", type=int, default=42, help="Random seed for inner train/val split")
    
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="", help="CUDA device, e.g. 0 or 0,1")
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--project", default="runs/train", help="YOLO project dir")
    p.add_argument("--name", default="yolo11n_kfold", help="base run name")

    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--fliplr", type=float, default=0.0)
    p.add_argument("--flipud", type=float, default=0.0)
    p.add_argument("--degrees", type=float, default=5.0)
    p.add_argument("--shear", type=float, default=0.0)
    p.add_argument("--perspective", type=float, default=0.0)
    p.add_argument("--mosaic", type=float, default=0.0)
    p.add_argument("--mixup", type=float, default=0.0)

    args = p.parse_args()

    for fold_idx in range(1, args.k + 1):
        train_one_fold(args, fold_idx)


if __name__ == "__main__":
    main()

'''
python kfold_train_yolo.py \
  --model yolo26s.pt \
  --data_dir data \
  --k 5 --inner_val_ratio 0.1 --inner_seed 42 \
  --epochs 300 --patience 50 --imgsz 640 --batch 32 --device 0 \
  --project runs/train --name yolo26s_kfold --pretrained --seed 42 \
  --fliplr 0.0 --flipud 0.0 --degrees 5.0 \
  --shear 0.0 --perspective 0.0 --mosaic 0.0 --mixup 0.0
'''