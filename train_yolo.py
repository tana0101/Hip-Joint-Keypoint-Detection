import argparse
from pathlib import Path
from ultralytics import YOLO

def main(args):
    data_yaml = Path(args.data).resolve()     # 轉絕對路徑，隔離全域設定
    model = YOLO(args.model)
    model.train(
      data=str(data_yaml),
      epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
      device=args.device, workers=args.workers,
      project=args.project, name=args.name,
      pretrained=args.pretrained, resume=args.resume, seed=args.seed,
      fliplr=args.fliplr, flipud=args.flipud, degrees=args.degrees,
      shear=args.shear, perspective=args.perspective,
      mosaic=args.mosaic, mixup=args.mixup,
    ) 

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--data",  default="data/data.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="yolo11n_run")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fliplr", type=float, default=0.0)        # 關掉水平翻轉
    p.add_argument("--flipud", type=float, default=0.0)        # 關掉上下翻轉
    p.add_argument("--degrees", type=float, default=5.0)       # 允許小角度旋轉即可
    p.add_argument("--shear", type=float, default=0.0)         # 關閉剪切避免左右語義混淆
    p.add_argument("--perspective", type=float, default=0.0)   # 關閉透視變形
    p.add_argument("--mosaic", type=float, default=0.0)        # 醫學影像關閉拼接
    p.add_argument("--mixup", type=float, default=0.0)         # 醫學影像關閉混合
    args = p.parse_args()
    main(args)

'''
python train_yolo.py \
  --model yolo12m.pt \
  --data data/data_fold1.yaml \
  --epochs 300 --imgsz 640 --batch 8 --device 0 \
  --project runs/train --name yolo12m_fold1 --pretrained --seed 42 \
  --fliplr 0.0 --flipud 0.0 --degrees 5.0 \
  --shear 0.0 --perspective 0.0 --mosaic 0.0 --mixup 0.0
'''