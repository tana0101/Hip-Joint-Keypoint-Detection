import os
import shutil
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from train_hip_crop_keypoints import train, LOGS_DIR
from utils.train_vis import plot_training_progress

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
    - 若 >1 個 → raise（依你的假設：不會有多個同類型標註）
    """
    if not root.is_dir():
        return None
    matches = list(root.glob(stem + ".*"))
    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"Multiple files in {root} with stem '{stem}': {matches}")


def copy_sample(img_path: Path, dst_root: Path, use_symlink: bool = True):
    """
    將某張影像及其對應的 annotations / detections 複製或建立 symlink 到 dst_root。
    src 結構假設為：.../foldX/images/xxx.png
    """
    fold_dir = img_path.parent.parent  # foldX
    # src_img_dir = fold_dir / "images"
    src_ann_dir = fold_dir / "annotations"
    src_det_dir = fold_dir / "detections"

    stem = img_path.stem

    dst_img_dir = dst_root / "images"
    dst_ann_dir = dst_root / "annotations"
    dst_det_dir = dst_root / "detections"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_ann_dir.mkdir(parents=True, exist_ok=True)
    dst_det_dir.mkdir(parents=True, exist_ok=True)

    # image
    dst_img = dst_img_dir / img_path.name
    if not dst_img.exists():
        if use_symlink:
            os.symlink(img_path, dst_img)
        else:
            shutil.copy2(img_path, dst_img)

    # annotation
    ann_path = find_single_with_stem(src_ann_dir, stem)
    if ann_path is not None:
        dst_ann = dst_ann_dir / ann_path.name
        if not dst_ann.exists():
            if use_symlink:
                os.symlink(ann_path, dst_ann)
            else:
                shutil.copy2(ann_path, dst_ann)

    # detection（給 crop 用的 bbox）
    det_path = find_single_with_stem(src_det_dir, stem)
    if det_path is not None:
        dst_det = dst_det_dir / det_path.name
        if not dst_det.exists():
            if use_symlink:
                os.symlink(det_path, dst_det)
            else:
                shutil.copy2(det_path, dst_det)


def build_simple_kfold_split(folds_root: Path, k: int, fold_idx: int, tmp_root: Path, use_symlink: bool = True):
    """
    mode = 'val_as_test' 的情況：
    - val = fold{fold_idx}
    - train = 其他所有 fold
    結果放在 tmp_root/train 和 tmp_root/val 底下。
    """
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    (tmp_root / "train").mkdir(parents=True, exist_ok=True)
    (tmp_root / "val").mkdir(parents=True, exist_ok=True)

    img_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    for i in range(1, k + 1):
        fold_dir = folds_root / f"fold{i}"
        images_dir = fold_dir / "images"
        if not images_dir.is_dir():
            raise RuntimeError(f"{images_dir} not found")

        imgs = list_images(images_dir, img_exts)
        if i == fold_idx:
            # 這個 fold 當 val (= 之後你也可以拿來當 test)
            dst_root = tmp_root / "val"
        else:
            dst_root = tmp_root / "train"

        for img_path in imgs:
            copy_sample(img_path, dst_root, use_symlink=use_symlink)

    print(f"[KFold simple] Fold {fold_idx}: train/val split built at {tmp_root}")


def build_outer_inner_split(folds_root: Path, k: int, fold_idx: int, tmp_root: Path,
                            inner_val_ratio: float = 0.1, inner_seed: int = 42,
                            use_symlink: bool = True):
    """
    mode = 'outer_inner' 的情況：
    - Outer Test = fold{fold_idx}（這支程式不會碰它，只是保留對應關係）
    - Outer Train Pool = 其他 fold
    - 在 Train Pool 裡再切 inner train / inner val
    結果放在 tmp_root/train 和 tmp_root/val 底下。
    """
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    (tmp_root / "train").mkdir(parents=True, exist_ok=True)
    (tmp_root / "val").mkdir(parents=True, exist_ok=True)

    img_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    # 先把 outer train pool 所有影像收集起來
    train_pool_imgs = []
    for i in range(1, k + 1):
        if i == fold_idx:
            continue  # 這個 fold 保留當 outer test
        fold_dir = folds_root / f"fold{i}"
        images_dir = fold_dir / "images"
        if not images_dir.is_dir():
            raise RuntimeError(f"{images_dir} not found")
        imgs = list_images(images_dir, img_exts)
        train_pool_imgs.extend(imgs)

    if len(train_pool_imgs) == 0:
        raise RuntimeError("No images found in outer train pool.")

    train_imgs, val_imgs = train_test_split(
        train_pool_imgs,
        test_size=inner_val_ratio,
        random_state=inner_seed,
        shuffle=True,
    )

    for img_path in train_imgs:
        copy_sample(img_path, tmp_root / "train", use_symlink=use_symlink)
    for img_path in val_imgs:
        copy_sample(img_path, tmp_root / "val", use_symlink=use_symlink)

    print(f"[KFold outer-inner] Fold {fold_idx}: "
          f"inner train = {len(train_imgs)}, inner val = {len(val_imgs)}, tmp_root = {tmp_root}")
    print(f"  Outer test fold = fold{fold_idx}")


def main():
    parser = argparse.ArgumentParser(description="K-fold training for hip crop keypoints.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="包含 fold1..foldK 的資料夾，例如 data 或 dataset/xray_IHDI_5_kfold")
    parser.add_argument("--k", type=int, default=5, help="fold 數量")
    parser.add_argument("--mode", type=str, default="outer_inner",
                        choices=["outer_inner", "val_as_test"],
                        help="outer_inner = Outer K-fold + Inner split; "
                             "val_as_test = 單層 K-fold，val 同時也是之後的 test fold")
    parser.add_argument("--inner_val_ratio", type=float, default=0.1,
                        help="在 outer_inner 模式下，Train pool 裡用多少比例做 inner validation")
    parser.add_argument("--inner_seed", type=int, default=42,
                        help="在 outer_inner 模式下，inner split 的亂數種子")
    parser.add_argument("--copy", action="store_true",
                        help="預設用 symlink 建 tmp/train,val；加這個就改成實際複製檔案（較耗空間）")

    # 把原本 train_hip_crop_keypoints.py main 裡的參數搬過來
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name: 'efficientnet', 'resnet', 'vgg', 'convnext' 等")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--head_type", type=str, default="direct_regression",
                        choices=["direct_regression", "simcc_1d", "simcc_2d", "simcc_2d_deconv"])
    parser.add_argument("--split_ratio", type=float, default=2.0,
                        help="SimCC 的 sr 參數，用在 head_type 為 simcc* 時")
    parser.add_argument("--sigma", type=float, default=6.0,
                        help="SimCC label encoding 的 sigma")

    args = parser.parse_args()

    folds_root = Path(args.data_root).resolve()
    use_symlink = not args.copy
    
    all_fold_metrics = []

    # 逐折跑訓練（不做 outer test 評估，outer test 留給 kfold_predict_hip_crop_keypoints）
    for fold_idx in range(1, args.k + 1):
        print("\n" + "=" * 80)
        print(f"[KFold] Start training for fold {fold_idx}/{args.k}, mode = {args.mode}")
        print("=" * 80)

        tmp_root = folds_root / "kfold_tmp" / args.mode / f"fold{fold_idx}"

        if args.mode == "val_as_test":
            build_simple_kfold_split(
                folds_root=folds_root,
                k=args.k,
                fold_idx=fold_idx,
                tmp_root=tmp_root,
                use_symlink=use_symlink,
            )
        else:  # outer_inner
            build_outer_inner_split(
                folds_root=folds_root,
                k=args.k,
                fold_idx=fold_idx,
                tmp_root=tmp_root,
                inner_val_ratio=args.inner_val_ratio,
                inner_seed=args.inner_seed,
                use_symlink=use_symlink,
            )

        # 呼叫單次訓練的 train()
        # 建議在 train(...) 裡面用 fold_index 來修改 exp_name / log 檔名
        metrics = train(
            data_dir=str(tmp_root),
            model_name=args.model_name,
            input_size=args.input_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            side=args.side,
            mirror=args.mirror,
            head_type=args.head_type,
            split_ratio=args.split_ratio,
            sigma=args.sigma,
            fold_index=fold_idx,  # 如果不想改 train() 的簽名，這行可以拿掉
        )
        metrics["fold_index"] = fold_idx
        all_fold_metrics.append(metrics)

        print(f"[KFold] Fold {fold_idx} training done. Best model & logs 由 train() 負責儲存。")
        
    # ==========================
    #  K-fold 平均曲線 + summary
    # ==========================
    if len(all_fold_metrics) == 0:
        print("[KFold] No folds were trained, skip aggregation.")
        return

    print("\n" + "=" * 80)
    print("[KFold] Aggregating metrics across folds...")
    print("=" * 80)

    # 假設每個 fold 都跑相同 epochs
    epochs = args.epochs
    epochs_range = range(1, epochs + 1)

    # per-epoch matrices: shape = (k, epochs)
    train_losses_mat = np.array([m["epoch_losses"] for m in all_fold_metrics], dtype=float)
    val_losses_mat = np.array([m["val_losses"] for m in all_fold_metrics], dtype=float)
    train_nme_mat = np.array([m["epoch_nmes"] for m in all_fold_metrics], dtype=float)
    val_nme_mat = np.array([m["val_nmes"] for m in all_fold_metrics], dtype=float)
    train_pixel_mat = np.array([m["epoch_pixel_errors"] for m in all_fold_metrics], dtype=float)
    val_pixel_mat = np.array([m["val_pixel_errors"] for m in all_fold_metrics], dtype=float)

    mean_train_loss = train_losses_mat.mean(axis=0)
    mean_val_loss = val_losses_mat.mean(axis=0)
    mean_train_nme = train_nme_mat.mean(axis=0)
    mean_val_nme = val_nme_mat.mean(axis=0)
    mean_train_pixel = train_pixel_mat.mean(axis=0)
    mean_val_pixel = val_pixel_mat.mean(axis=0)

    os.makedirs(LOGS_DIR, exist_ok=True)

    # k-fold 的實驗結果命名
    base_exp_name = all_fold_metrics[0]["exp_name"]
    base_exp_name = base_exp_name[: -len(f"_fold{fold_idx}")]
    
    # 平均線圖（主圖）
    plot_training_progress(
        epochs_range,
        mean_train_loss,
        mean_val_loss,
        mean_train_nme,
        mean_val_nme,
        mean_train_pixel,
        mean_val_pixel,
        loss_ylim=(0.01, 50),
        nme_ylim=(0.0001, 0.02),
        pixel_error_ylim=(0.01, 50),
    )
    summary_plot_path = os.path.join(LOGS_DIR, f"{base_exp_name}_kfold_summary.png")
    plt.savefig(summary_plot_path)
    plt.close()
    print(f"[KFold] Summary plot (mean curves) saved to: {summary_plot_path}")

    # ===== NEW: 各 fold 一條線的圖（K 條） =====
    # Val Loss per fold
    plt.figure(figsize=(10, 6))
    for i, m in enumerate(all_fold_metrics, 1):
        plt.plot(epochs_range, m["val_losses"], label=f"Fold {i}")
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.title(f"Per-fold Validation Loss Curves ({args.mode})")
    plt.legend()
    plt.grid(True)
    per_fold_val_loss_path = os.path.join(LOGS_DIR, f"{base_exp_name}_kfold_per_fold_val_loss.png")
    plt.savefig(per_fold_val_loss_path)
    plt.close()
    print(f"[KFold] Per-fold val loss plot saved to: {per_fold_val_loss_path}")

    # Val NME per fold
    plt.figure(figsize=(10, 6))
    for i, m in enumerate(all_fold_metrics, 1):
        plt.plot(epochs_range, m["val_nmes"], label=f"Fold {i}")
    plt.xlabel("Epoch")
    plt.ylabel("Val NME")
    plt.title(f"Per-fold Validation NME Curves ({args.mode})")
    plt.legend()
    plt.grid(True)
    per_fold_val_nme_path = os.path.join(LOGS_DIR, f"{base_exp_name}_kfold_per_fold_val_nme.png")
    plt.savefig(per_fold_val_nme_path)
    plt.close()
    print(f"[KFold] Per-fold val NME plot saved to: {per_fold_val_nme_path}")

    # Val Pixel Error per fold
    plt.figure(figsize=(10, 6))
    for i, m in enumerate(all_fold_metrics, 1):
        plt.plot(epochs_range, m["val_pixel_errors"], label=f"Fold {i}")
    plt.xlabel("Epoch")
    plt.ylabel("Val Pixel Error")
    plt.title(f"Per-fold Validation Pixel Error Curves ({args.mode})")
    plt.legend()
    plt.grid(True)
    per_fold_val_pixel_path = os.path.join(LOGS_DIR, f"{base_exp_name}_kfold_per_fold_val_pixel.png")
    plt.savefig(per_fold_val_pixel_path)
    plt.close()
    print(f"[KFold] Per-fold val pixel error plot saved to: {per_fold_val_pixel_path}")
    # ===== NEW END =====

    # 對 best epoch metrics 做平均（train / val）
    best_train_losses = [m["best_train_loss"] for m in all_fold_metrics]
    best_train_nmes = [m["best_train_nme"] for m in all_fold_metrics]
    best_train_pixels = [m["best_train_pixel"] for m in all_fold_metrics]

    best_val_losses = [m["best_val_loss"] for m in all_fold_metrics]
    best_val_nmes = [m["best_val_nme"] for m in all_fold_metrics]
    best_val_pixels = [m["best_val_pixel"] for m in all_fold_metrics]

    avg_best_train_loss = float(np.mean(best_train_losses))
    avg_best_train_nme = float(np.mean(best_train_nmes))
    avg_best_train_pixel = float(np.mean(best_train_pixels))

    avg_best_val_loss = float(np.mean(best_val_losses))
    avg_best_val_nme = float(np.mean(best_val_nmes))
    avg_best_val_pixel = float(np.mean(best_val_pixels))

    summary_txt = os.path.join(LOGS_DIR, f"{base_exp_name}_kfold_summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"K-fold training summary (mode = {args.mode})\n")
        f.write(f"model = {args.model_name}, side = {args.side}, head_type = {args.head_type}\n")
        f.write(f"k = {args.k}, epochs = {args.epochs}\n\n")

        for idx, m in enumerate(all_fold_metrics, 1):
            fold_id = m.get("fold_index", idx)
            be = m["best_epoch_index"]
            f.write(f"Fold {fold_id}: best_epoch = {be+1 if be is not None and be >= 0 else 'N/A'}\n")
            f.write(
                f"  train: loss = {m['best_train_loss']:.6f}, "
                f"nme = {m['best_train_nme']:.6f}, "
                f"pixel = {m['best_train_pixel']:.6f}\n"
            )
            f.write(
                f"  val  : loss = {m['best_val_loss']:.6f}, "
                f"nme = {m['best_val_nme']:.6f}, "
                f"pixel = {m['best_val_pixel']:.6f}\n\n"
            )

        f.write("Average of best-epoch metrics across folds:\n")
        f.write(
            f"  train: loss = {avg_best_train_loss:.6f}, "
            f"nme = {avg_best_train_nme:.6f}, "
            f"pixel = {avg_best_train_pixel:.6f}\n"
        )
        f.write(
            f"  val  : loss = {avg_best_val_loss:.6f}, "
            f"nme = {avg_best_val_nme:.6f}, "
            f"pixel = {avg_best_val_pixel:.6f}\n"
        )

    print(f"[KFold] Summary txt saved to: {summary_txt}")
    print("[KFold] Done.")


if __name__ == "__main__":
    main()
    
'''
python kfold_train_hip_crop_keypoints.py \
  --data_root data \
  --k 5 \
  --mode outer_inner \
  --inner_val_ratio 0.1 \
  --inner_seed 42 \
  --model_name hrnet_w48
  --input_size 224 \
  --epochs 200 \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --side left \
  --mirror \
  --head_type simcc_2d \
  --split_ratio 3.0 \
  --sigma 7.0
'''