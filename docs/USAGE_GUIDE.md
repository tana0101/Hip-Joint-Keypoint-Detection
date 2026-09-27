# 📖 Usage Guide

本文件說明 **Hip-Joint-Keypoint-Detection** 專案的環境建置、資料準備、模型訓練與評估流程。

專案採用兩階段 Top-down Pipeline：

```text
Dataset Preparation
        ↓
Dataset Split
        ↓
Stage 1: YOLO Hip Detection
        ↓
Stage 2: Keypoint Detection
        ↓
AI / IHDI Evaluation
```

目前主要支援兩種實驗流程：

1. **One-fold**：適合模型開發、快速測試與單次 Train / Validation / Test。
2. **K-fold Cross-validation**：主要研究評估流程，用於取得跨 Fold 與 OOF（Out-of-Fold）結果。

> 完整 CLI 參數可使用：
>
> ```bash
> python <script>.py --help
> ```
>
> 本文件僅列出主要使用流程與常用設定。

附註：
1. 如要使用 One Stage Pipeline（直接從原始影像進行關鍵點偵測）進行實驗，請參考 `train_full_image_keypoints.py` 與 `predict_full_image_keypoints.py`。
2. 如要使用 mtddh 公開資料集的切分方法，請參考 `split_from_json.py` 搭配 one-fold 流程。
---

# 1. Environment Setup

## 1.1 Clone Repository

```bash
git clone https://github.com/tana0101/Hip-Joint-Keypoint-Detection.git
cd Hip-Joint-Keypoint-Detection
```

## 1.2 Create Conda Environment

建議使用 Python 3.10：

```bash
conda create -n hip_joint_detection python=3.10
conda activate hip_joint_detection
```

## 1.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 2. Dataset Preparation

## 2.1 MTDDH

本專案提供 MTDDH 公開資料集作為主要重現範例。

執行：

```bash
chmod +x prepare_MTDDH_dataset.sh
./prepare_MTDDH_dataset.sh
```

腳本會進行：

```text
Download Dataset
      ↓
Data Cleaning（透過 `outliers.json` 過濾異常影像）
      ↓
Annotation Conversion
      ↓
Generate Validation Figures
```

預期資料位置：

```text
dataset/mtddh_xray_2d/
```

預期資料內容包含：

```text
dataset/mtddh_xray_2d/
├── images/
├── annotations/
├── detections/
├── vis_output/
└── yolo_labels/
```

> 實際目錄若有異動，請以目前 Repository 為準。

---

## 2.2 NCKUH_IHDI

NCKUH_IHDI 為院內研究資料，不公開提供。

預期資料內容包含：

```text
xray_IHDI_1/ -> 原始資料集（第一次標註 - 一般醫生）
xray_IHDI_2/ -> 原始資料集（第二次標註 - 資深醫生）
dataset/xray_IHDI_2_clean/ -> 清理後資料集（去除異常影像，異常影像於 `remove.txt` 中定義）
├── images/
├── annotations/
├── detections/
├── vis_output/
└── yolo_labels/
```

> 請勿將病患資料、帳號、Server 路徑或其他敏感資訊提交至公開 Repository。

---

# 3. One-fold Training & Evaluation

One-fold 適合：

- 模型開發
- Pipeline debug
- 快速比較不同模型
- 單次 Train / Validation / Test

整體流程：

```text
Dataset
   ↓
split.py
   ↓
train_yolo.py
   ↓
train_hip_crop_keypoints.py
   ↓
predict_hip_crop_keypoints.py
   ↓
results/
```

---

## 3.1 Split Dataset

使用：

```text
split.py
```

目前預設流程將 Dataset 分為：

```text
Train
Validation
Test
```

常用範例：

```bash
python split.py \
  --dataset dataset/mtddh_xray_2d \
  --out data \
  --train 0.8 \
  --val 0.1 \
  --test 0.1 \
  --seed 42
```

輸出：

```text
data/
├── train/
├── val/
├── test/
└── data.yaml
```

其中 `data.yaml` 供後續 YOLO 訓練使用。

---

## 3.2 Stage 1 — Train YOLO Detector

執行：

```text
train_yolo.py
```

YOLO 負責偵測：

```text
LeftHip
RightHip
```

並於後續 Keypoint Pipeline 中裁切單側 Hip ROI。

### Example

```bash
python train_yolo.py \
  --model yolo26s.pt \
  --data data/data.yaml \
  --epochs 300 --patience 50 --imgsz 640 --batch 32 --device 0 \
  --project runs/train --name yolo26s_mtddh --pretrained --seed 42 \
  --fliplr 0.0 --flipud 0.0 --degrees 5.0 \
  --shear 0.0 --perspective 0.0 --mosaic 0.0 --mixup 0.0
```

### Output

YOLO 訓練完成後會產生最佳權重：

```text
runs/detect/runs/yolo26s_mtddh/weights/best.pt
```

建議將最終使用的權重整理至：

```text
weights/
```

例如：

```text
weights/yolo26s.pt
```

---

## 3.3 Stage 2 — Train Keypoint Detector

執行：

```text
train_hip_crop_keypoints.py
```

目前主要研究模型為：

```text
Hip ROI
   ↓
ConvNeXt-Tiny
   ↓
FPN
   ↓
Hypercolumn
   ↓
SimCC
   ↓
Landmark Coordinates
```

目前主要設定：

```text
Input Size : 224 × 224
Side       : Left（訓練左側關節，右側可透過鏡像推論）
Mirror     : Enabled（啟用鏡像訓練，將左側影像鏡像後加入訓練）
Backbone   : ConvNeXt-Tiny
Neck       : FPN
Head       : SimCC
```

### Example

```bash
python train_hip_crop_keypoints.py \
  --data_dir Hip-Joint-Keypoint-Detection/data \
  --model_name convnext_tiny_fpn1234concat \
  --input_size 224 --epochs 100 --learning_rate 0.0001 --batch_size 64 --side left --mirror \
  --head_type simcc_2d --split_ratio 3.0 --sigma 4.0
```

### Output

模型權重：

```text
weights/
```

Training log：

```text
logs/
```

目前檔名會包含 Model / Head / Training Configuration 等資訊。

---

## 3.4 One-fold Evaluation

執行：

```text
predict_hip_crop_keypoints.py
```

Evaluation Pipeline：

```text
Test Image
   ↓
YOLO Detection
   ↓
Hip ROI
   ↓
Keypoint Prediction
   ↓
Coordinate Restoration
   ↓
AI Calculation
   ↓
IHDI Grading
   ↓
Metrics / Figures
```

### Example

```bash
python predict_hip_crop_keypoints.py \
  --model_name convnext_tiny_fpn1234concat \
  --kp_left_path weights/convnext_tiny_fpn1234concat_simcc_2d_sr3.0_sigma4.0_cropleft_mirror_224_100_0.0001_64_best.pth \
  --yolo_weights weights/yolo26s.pt \
  --data data/test \
  --output_dir results
```

輸出：

```text
results/
```

> 執行前請確認 YOLO 與 Keypoint model weights 均存在。

---

# 4. K-fold Cross-validation

K-fold 為主要研究評估流程。

整體流程：

```text
Dataset
   ↓
kfold_split.py
   ↓
kfold_train_yolo.py
   ↓
kfold_train_hip_crop_keypoints.py
   ↓
kfold_predict_hip_crop_keypoints.py
   ↓
Fold Results
   ↓
OOF Results
```

目前主要設定：

```text
K = 5
Seed = 42
```

---

## 4.1 K-fold Dataset Split

執行：

```text
kfold_split.py
```

### Example

```bash
python kfold_split.py \
  --src dataset/mtddh_xray_2d \
  --dst data \
  --k 5 \
  --seed 42 \
  --overwrite
```

輸出：

```text
data/
├── fold1/
├── fold2/
├── fold3/
├── fold4/
├── fold5/
├── data_fold1.yaml
├── data_fold2.yaml
├── ...
└── data_fold5.yaml
```

每個 Fold 在輪流作為 Test Fold 時，其餘 Fold 作為 Training Pool。

---

# 4.2 K-fold YOLO Training

執行：

```text
kfold_train_yolo.py
```

### Example

```bash
python kfold_train_yolo.py \
  --model yolo26s.pt \
  --data_dir data \
  --k 5 --inner_val_ratio 0.1 --inner_seed 42 \
  --epochs 300 --patience 50 --imgsz 640 --batch 32 --device 0 \
  --project runs/train --name yolo26s_kfold --pretrained --seed 42 \
  --fliplr 0.0 --flipud 0.0 --degrees 5.0 \
  --shear 0.0 --perspective 0.0 --mosaic 0.0 --mixup 0.0
```

建議整理權重為：

```text
weights/
├── yolo26s_fold1.pt
├── yolo26s_fold2.pt
├── yolo26s_fold3.pt
├── yolo26s_fold4.pt
└── yolo26s_fold5.pt
```

---

# 4.3 K-fold Keypoint Training

執行：

```text
kfold_train_hip_crop_keypoints.py
```

目前主要採用 **Outer K-fold + Inner Validation**：

```text
Entire Dataset
      ↓
Outer 5-fold
      ↓
┌─────────────────────┐
│ Outer Test Fold     │ → Final evaluation
└─────────────────────┘

Remaining 4 folds
      ↓
Training Pool
      ↓
┌──────────────┬──────────────┐
│ 90% Training │ 10% Inner Val│
└──────────────┴──────────────┘
```

其中：

- **Train**：模型參數更新
- **Inner Validation**：模型選擇與最佳 checkpoint
- **Outer Test**：最終 Fold 評估
- 五個 Outer Test prediction 合併為 OOF prediction

### Example

```bash
python kfold_train_hip_crop_keypoints.py \
  --data_root data \
  --k 5 \
  --mode outer_inner \
  --inner_val_ratio 0.1 \
  --inner_seed 42 \
  --model_name convnext_tiny_fpn1234concat \
  --input_size 224 \
  --epochs 200 \
  --learning_rate 0.0001 \
  --batch_size 64 \
  --side left \
  --mirror \
  --head_type simcc_2d \
  --split_ratio 3.0 \
  --sigma 4.0
```

注意：參數 k、inner_val_ratio、inner_seed 與 mode 需與 `kfold_train_yolo.py` 保持一致，以保證實驗的切分方法相同，避免因參數不一致導致結果偏差。

輸出：

```text
weights/
```

每個 Fold 應產生獨立模型權重。

---

# 4.4 K-fold Evaluation

執行：

```text
kfold_predict_hip_crop_keypoints.py
```

流程：

```text
Fold 1 Model → Fold 1 Test
Fold 2 Model → Fold 2 Test
Fold 3 Model → Fold 3 Test
Fold 4 Model → Fold 4 Test
Fold 5 Model → Fold 5 Test
                  ↓
             Merge OOF
                  ↓
        Summary / Statistics
```

### Example

```bash
python kfold_predict_hip_crop_keypoints.py \
  --model_name convnext_tiny_fpn1234concat \
  --kp_left_tpl "weights/convnext_tiny_fpn1234concat_simcc_2d_sr3.0_sigma4.0_cropleft_mirror_224_100_0.0001_64_fold{fold}_best.pth" \
  --yolo_weights weights/yolo26s_fold{fold}.pt \
  --data_root data \
  --k 5 \
  --output_root results_kfold_mtddh
```

主要輸出：

```text
results_kfold/
└── <experiment_name>/
    ├── fold1/
    ├── fold2/
    ├── fold3/
    ├── fold4/
    ├── fold5/
    └── summary/
```

`summary/` 用於保存跨 Fold 或 OOF 統計結果。

---

# 5. Experimental Split Modes

部分 Script 保留不同資料切分方式供研究實驗使用。

目前已知包括：

### `outer_inner`

```text
Outer Test
+
Inner Train / Validation
```

主要用於正式 K-fold 評估。

### `val_as_test`

```text
Training Fold
+
Validation / Test Fold
```

屬於較簡化的單層 K-fold 使用方式。

---

# 6. Experimental Models

Repository 為論文比較與消融實驗保留多種模型接口。

例如：

```text
Backbone
├── ConvNeXt
├── HRNet
├── EfficientNet
└── ...

Head
├── SimCC
├── Direct Regression
└── ...
```

目前主要推薦：

```text
ConvNeXt-Tiny
    +
FPN
    +
Hypercolumn
    +
SimCC
```

其他模型主要用途：

- Baseline comparison
- Ablation study
- Architecture experiments
- Reproducibility

詳細支援列表請以目前程式碼為準。

---

# 7. Output Directories

| Directory | Purpose |
|---|---|
| `weights/` | Model checkpoints |
| `logs/` | Training logs / curves |
| `results/` | One-fold results |
| `results_kfold/` | K-fold / OOF results |
| `runs/` | Ultralytics YOLO outputs |

---

# 8. Backend Integration

目前此 Repository 主要負責：

```text
Training
   ↓
Evaluation
   ↓
Model Weights
```

實際服務使用另一個 Backend Repository：

```text
Model Weights
      ↓
Backend
      ↓
API
      ↓
DICOM / SFTP
      ↓
Clinical System
```

Backend Repository：

```text
目前還在成大醫院測試中
```

---