#!/usr/bin/env python3
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_one_fold(args, fold_idx: int):
    # 依照模板產生當前 fold 的 data.yaml 路徑
    data_path_str = args.data_tpl.format(fold=fold_idx)
    data_yaml = Path(data_path_str).resolve()

    # 每個 fold 使用不同的 run name，避免覆蓋
    run_name = f"{args.name}_fold{fold_idx}"

    print("=" * 80)
    print(f"[YOLO KFold] Start training fold {fold_idx}/{args.k}")
    print(f"  model      : {args.model}")
    print(f"  data_yaml  : {data_yaml}")
    print(f"  project    : {args.project}")
    print(f"  run name   : {run_name}")
    print(f"  epochs     : {args.epochs}")
    print(f"  imgsz      : {args.imgsz}")
    print(f"  batch      : {args.batch}")
    print(f"  device     : {args.device}")
    print("=" * 80)

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
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
    p = argparse.ArgumentParser(description="K-fold training wrapper for YOLO.")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO model weights or cfg")
    # data_tpl 支援 {fold} 佔位，會用 .format(fold=i) 生成實際路徑
    p.add_argument(
        "--data_tpl",
        default="data/data_fold{fold}.yaml",
        help="data.yaml template, e.g. 'data/data_fold{fold}.yaml'",
    )
    p.add_argument("--k", type=int, default=5, help="number of folds")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="", help="CUDA device, e.g. 0 or 0,1")
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--project", default="runs/train", help="YOLO project dir")
    p.add_argument(
        "--name",
        default="yolo11n_kfold",
        help="base run name, each fold will append `_foldX`",
    )

    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # augmentation 參數維持你原本的設定
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
  --data_tpl data/data_fold{fold}.yaml \
  --k 5 \
  --epochs 100 --imgsz 640 --batch 32 --device 0 \
  --project runs/train --name yolo26s_kfold --pretrained --seed 42 \
  --fliplr 0.0 --flipud 0.0 --degrees 5.0 \
  --shear 0.0 --perspective 0.0 --mosaic 0.0 --mixup 0.0
'''