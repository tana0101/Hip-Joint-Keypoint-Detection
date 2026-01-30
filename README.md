# 🦴 Deep Learning–Based Hip Joint Keypoint Detection System

<div align="center">
  <div>
    <a href="https://github.com/tana0101/Hip-Joint-Keypoint-Detection/blob/main/README_zh_TW.md">🇹🇼繁體中文</a> |
    <a href="https://github.com/tana0101/Hip-Joint-Keypoint-Detection/blob/main/README.md">🌏English</a> |
    <a href="https://deepwiki.com/tana0101/Hip-Joint-Keypoint-Detection">📚DeepWiki</a> |
    <a href="https://github.com/tana0101/Hip-Joint-Keypoint-Detection/issues">❓issues</a><!-- |
    📝Paper(尚未發表)-->
  </div>
<br>
  <img src="src/img/project_banner.png" style="width: 70%;"/>
<br>
    <a href="https://app.codacy.com/gh/tana0101/Hip-Joint-Keypoint-Detection/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/800c026fb9d1418e9cb735d1455c3383"/></a>
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/tana0101/Hip-Joint-Keypoint-Detection">
    <img alt="Using Python version" src="https://img.shields.io/badge/python-3.10-blue.svg">
    <a href="https://deepwiki.com/tana0101/Hip-Joint-Keypoint-Detection"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white"/>
    <img alt="Ultralytics YOLO" src="https://img.shields.io/badge/Ultralytics%20YOLO-%23000000.svg?style=flat&logo=ultralytics&logoColor=white"/>
</div>

## 📋 Overview

This project presents a **deep learning–based hip joint keypoint detection system** designed to assist the measurement and grading of **Developmental Dysplasia of the Hip (DDH)** in pediatric patients. The system aims to:
(1) automatically detect hip joint keypoints,
(2) compute the **Acetabular Index (AI) angle**, and
(3) perform **IHDI classification**.

The system adopts a **top-down, two-stage pipeline**. First, **YOLO** is used to detect and crop hip joint regions. Then, a keypoint model performs **unilateral (left/right) hip keypoint detection** on the cropped regions.

> **Research-only Notice**: This project is a prototype system for computer science research purposes. Its outputs must not be used directly for clinical diagnosis.

## 📑 Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage (One-fold)](#usage-one-fold-training--evaluation)
- [Usage (K-Fold)](#usage-k-fold-cross-validation)
- [Results](#results)
- [References](#references)

## Introduction

Developmental Dysplasia of the Hip (DDH) is a common but often overlooked skeletal developmental disorder. Without early diagnosis and intervention, DDH can lead to long-term impairments in gait and skeletal development.

In clinical practice, DDH diagnosis heavily relies on manual interpretation and measurement of X-ray images by physicians. This process is inherently subjective and prone to inter-observer and intra-observer variability across different clinicians and time points.

In recent years, deep learning has demonstrated outstanding performance in medical image analysis, particularly for tasks such as keypoint detection, angle measurement, and disease grading. This project leverages deep learning models to automatically detect hip joint keypoints and assist clinicians in computing DDH-related indices and classifications.

## ✨ Key Features

- 📍 **Hip Joint Keypoint Detection**: Automatic prediction of hip joint keypoint coordinates.
- 📐 **Clinical Metric Measurement**: Supports Acetabular Index (AI) angle computation and IHDI classification.
- 🏗️ **Multi-backbone Support**: Flexible selection of different backbones (e.g., ConvNeXt, HRNet, EfficientNet).
- 🎯 **Multiple Head Designs**: Supports different keypoint prediction strategies, including Direct Regression and the SimCC family.
- 🧩 **Modular Design**: Facilitates model replacement, experimental comparison, and extensible research.

## 💾 Dataset

- Data are stored in the `dataset/` directory.
- The annotation tool is located in the `Keypoint-Annotation-Tool/` directory.

🚧 **Dataset download and preparation procedures will be provided in future updates** 🚧

### 🏥 xray_IHDI (Primary Experimental Dataset)

<img src="src/img/sample_IHDI.jpg" style="width: 30%;"/>

This study uses retrospective data collected from National Cheng Kung University Hospital between **2015/01/01 and 2025/01/19**, consisting of hip X-ray images of infants and young children under 4 years old. A total of 622 images were initially collected, and 557 images were retained after outlier removal. Each image was manually annotated by clinical experts with **12 keypoints**, and LeftHip / RightHip object labels were provided for training the detection stage.

- Annotation format: Each image corresponds to one `.csv` file in the format:
```
"(x1,y1)","(x2,y2)",...,"(x12,y12)"
```
- Note: Due to medical privacy and data protection regulations, this dataset cannot be publicly released.

<hr>

### 🌍 MTDDH (Public Dataset)

<img src="src/img/sample_MTDDH.jpg" style="width: 30%;"/>

- Source: [open-hip-dysplasia](https://github.com/radoss-org/open-hip-dysplasia.git)
- Size: 1,666 hip X-ray images (after outlier removal)
- Annotations:
  - **8 keypoints**
  - LeftHip / RightHip object labels

<hr>

### 📊 Dataset Statistics

**Acetabular Index (AI) Distribution**

<img src="dataset/xray_IHDI_AI_Distribution.png" />
<img src="dataset/mtddh_xray_2d_AI_Distribution.png" />

**IHDI Classification Distribution**

<div style="display: flex; justify-content: space-between; gap: 10px;">
  <img src="dataset/xray_IHDI_IHDI_Distribution.png" style="width: 49%;" />
  <img src="dataset/mtddh_xray_2d_IHDI_Distribution.png" style="width: 49%;" />
</div>

## 🛠️ Methodology

This project adopts a **top-down, two-stage keypoint detection pipeline**:
1. **🔍 Object Detection and Unilateral Cropping**: YOLO detects LeftHip / RightHip and crops ROIs to reduce background interference.
2. **🧠 Unilateral Keypoint Detection**: Keypoints are detected on cropped unilateral hip ROIs (with comparative studies across multiple backbones and heads).

### Head Architecture

<img src="src/img/head_design.png" style="width: 99%;" />

The project supports multiple keypoint head designs to accommodate different model characteristics and experimental needs:

- **SimCC 2D / SimCC 2D Deconv**:
  Official SimCC variants that convert coordinate regression into 1D classification distributions for x and y, followed by soft-argmax to obtain coordinates.
- **SimCC 1D (Custom Variant)**:
  Uses Global Average Pooling to compress feature maps and predicts x/y distributions via fully connected layers to reduce complexity.
- **Direct Regression**:
  Directly regresses (x, y) coordinates using fully connected layers.

### Backbone Architecture

Currently supported backbone architectures include:

- ConvNeXt V1
  - `ConvNeXtSmallCustom`
- ConvNeXt V1 + Feature Pyramid Network (multi-scale features)
  - `ConvNeXtSmallMS`
- HRNet
  - `HRNetW32Custom`
  - `HRNetW48Custom`

The `Custom` suffix indicates modifications and optimizations based on official implementations to better suit the hip joint keypoint detection task.

🚧 **Additional backbones (e.g., EfficientNet, InceptionNeXt) are under active development and testing to ensure compatibility with different head architectures** 🚧

### Other Techniques

- **🔄 Data Augmentation**: Random Rotation / Random Translation
- **📉 Loss Functions**
  - **Direct Regression (MSE Loss)**
  - **SimCC Series (KL Divergence Loss)**
- **⚙️ Optimizers**: AdamW
- **📈 LR Schedulers**: Cosine Annealing + Warmup

## 📂 Project Structure

```text
Hip-Joint-Keypoint-Detection/
├── dataset/                        # 💾 Dataset storage
│   ├── xray_IHDI/                  # Primary dataset (Private)
│   └── mtddh/                      # Public dataset
├── datasets/                       # Dataset loading and processing
├── models/                         # 🧠 Model and head definitions
├── src/                            # Core resources and images
│   └── img/                        # Images used in README
├── utils/                          # 🛠️ Utility functions
├── weights/                        # 📥 Trained model weights (.pth)
├── logs/                           # 📝 Training logs and curves
├── results/                        # 📊 Statistical outputs
├── Keypoint-Annotation-Tool/       # 🖊️ Keypoint annotation tool
├── Hip-Joint-Keypoint-Detection-Tool/ # 🖥️ Clinical assistant prototype (WIP)
├── train_yolo.py                   # [Train] YOLO detector
├── train_hip_crop_keypoints.py     # [Train] Unilateral keypoint model
├── predict_hip_crop_keypoints.py   # [Inference] Full detection and evaluation
├── split.py                        # [Tool] Train/Val/Test split
├── kfold_split.py                  # [Tool] K-Fold split
├── kfold_train_yolo.py             # [K-Fold] YOLO training
├── kfold_train_hip_crop_keypoints.py # [K-Fold] Keypoint training
├── kfold_predict_hip_crop_keypoints.py # [K-Fold] Evaluation
├── requirements.txt                # 📦 Dependencies
├── README.md                       # 🇬🇧 English documentation
└── README_zh_TW.md                 # 🇹🇼 Traditional Chinese documentation
```

## 📦 Installation

Follow the steps below to set up the Python environment and install dependencies.

### Using Conda

1. Clone the repository:
```
   git clone https://github.com/tana0101/Hip-Joint-Keypoint-Detection.git
   cd Hip-Joint-Keypoint-Detection
```
2. Create and activate a Conda environment:
```
   conda create -n hip_joint_detection python=3.10
   conda activate hip_joint_detection
```
3. Install dependencies:
```
   pip install -r requirements.txt
```

### Using Standard Python Environment

1. Clone the repository:
```
   git clone https://github.com/tana0101/Hip-Joint-Keypoint-Detection.git
   cd Hip-Joint-Keypoint-Detection
```
2. Install dependencies:
```
   pip install -r requirements.txt
```

## 🚀Usage (One-fold): Training & Evaluation

### Data Preparation

We provide the MTDDH dataset for demonstration purposes. Before splitting the dataset or training, run the automation script to download, clean, and convert the raw data.

Run Preparation Script: This script will automatically download the dataset to `dataset/mtddh_xray_2d`, remove outliers, convert annotations, and generate visualization checks.
```
chmod +x prepare_MTDDH_dataset.sh
./prepare_MTDDH_dataset.sh
```

### Split Dataset

<details><summary><b>Click to expand command description</b></summary>

```
usage: split.py [-h] --dataset DATASET [--out OUT] [--train TRAIN] [--val VAL]
                [--test TEST] [--seed SEED]

Split dataset into train/val/test with multiple modalities and emit
Ultralytics data.yaml
```
</details>

Example command:
```
python split.py --dataset dataset/mtddh_xray_2d --out data --train 0.8 --val 0.1 --test 0.1 --seed 42
```

This will generate `data/train`, `data/val`, and `data/test`, along with `data/data.yaml` for object detection training.

### Training

The project uses a two-stage pipeline: YOLO-based hip detection followed by unilateral keypoint training.

#### Step 1: Train YOLO Detector

```
python train_yolo.py \
  --model yolo26s.pt \
  --data data/data.yaml \
  --epochs 300 --imgsz 640 --batch 8 --device 0 \
  --project runs/train --name yolo26s --pretrained --seed 42 \
  --fliplr 0.0 --flipud 0.0 --degrees 5.0 \
  --shear 0.0 --perspective 0.0 --mosaic 0.0 --mixup 0.0
```

⚠️ After training, move and rename the best weight `(runs/detect/runs/exp_name/weights/best.pt)` to the `weights/` directory (e.g., `yolo26s.pt`).

#### Step 2: Train Keypoint Detector

Example:
```
python train_hip_crop_keypoints.py --data_dir data --model_name convnext_small_custom --input_size 224 --epochs 200 --learning_rate 0.0001 --batch_size 32 --side left --mirror --head_type simcc_2d --split_ratio 3.0 --sigma 7.0
```

Training outputs include best weights, logs, and plots saved in `weights/` and `logs/`.

### Evaluation

Example:
```
python3 predict_hip_crop_keypoints.py --model_name convnext_small_custom --kp_left_path weights/convnext_small_custom_simcc_2d_sr3.0_sigma7.0_cropleft_mirror_224_200_0.0001_32_best.pth --yolo_weights weights/yolo26s.pt --data data/test --output_dir results
```

Results will be saved in the specified output directory.

## 🔄 Usage (K-Fold Cross Validation)

K-fold splitting, training, and evaluation are fully supported. Please refer to the original scripts and examples above for detailed usage.

### Split Dataset

Example:
```
python kfold_split.py \
  --src dataset/mtddh_xray_2d \
  --dst data \
  --k 5 \
  --seed 42 \
  --overwrite
```

### Training

#### Step 1: Train YOLO Detector

```
python kfold_train_yolo.py \
  --model yolo26s.pt \
  --data_tpl data/data_fold{fold}.yaml \
  --k 5 \
  --epochs 300 --imgsz 640 --batch 8 --device 0 \
  --project runs/train --name yolo26s_kfold --pretrained --seed 42 \
  --fliplr 0.0 --flipud 0.0 --degrees 5.0 \
  --shear 0.0 --perspective 0.0 --mosaic 0.0 --mixup 0.0
```

⚠️ Note: After training is complete, please move and rename the best weights for each fold `(runs/detect/runs/yolo26s_kfold_fold{fold}/weights/best.pt)` to the `weights/` folder (e.g., as `yolo26s_fold{i}.pt`) for subsequent inference steps.

#### Step 2: Train Keypoint Detector

```
python kfold_train_hip_crop_keypoints.py \
  --data_root data \
  --k 5 \
  --mode outer_inner \
  --inner_val_ratio 0.1 \
  --inner_seed 42 \
  --model_name convnext_small_custom \
  --input_size 224 \
  --epochs 200 \
  --learning_rate 0.0001 \
  --batch_size 16 \
  --side left \
  --mirror \
  --head_type simcc_2d \
  --split_ratio 3.0 \
  --sigma 7.0
```

### Evaluation

```
python kfold_predict_hip_crop_keypoints.py \
  --model_name convnext_small_custom \
  --kp_left_tpl "weights/convnext_small_custom_simcc_2d_sr3.0_sigma7.0_cropleft_mirror_224_200_0.0001_16_fold{fold}_best.pth" \
  --yolo_weights weights/yolo26s_fold{fold}.pt \
  --data_root data \
  --k 5 \
  --output_root results_kfold
```

## 🏆 Results

<img src="src/img/experiment.png" style="width: 99%;"/>

Five-fold cross-validation results using ConvNeXtSmallCustom + SimCC 2D on the xray_IHDI dataset are shown below.

## ⚡ Inference (WIP)

🚧 The clinical interface is under development 🚧

The code in `Hip-Joint-Keypoint-Detection-Tool/` provides a preliminary single-stage keypoint detection interface for research reference only.

## ⚠️ Disclaimer

> **For Research Use Only**  
> This system is intended solely for academic research and technical exchange and **is not a medical device**.  
> Outputs must not be used for clinical diagnosis, medical decision-making, or treatment.  
> The software is provided “as is” without any express or implied warranties.

## 📢 Project Note

Due to current research progress and collaboration considerations, this repository does not publicly release core algorithms and certain key technical details. If you are interested in full technical details or potential academic/commercial collaboration, please open an issue or contact the authors directly.

## ⚖️ License

This project is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for details.

### Dependencies & Acknowledgements

This project builds upon multiple excellent open-source projects. Users must comply with the licenses of the following components.

#### 1. 🛑 Restrictive Components

- **Ultralytics YOLO** (AGPL-3.0)
  - Impact: As YOLO is used for core training and inference, the entire project inherits AGPL-3.0. If deployed as a network service, source code disclosure is required.
- **ConvNeXt V2**
  - Code: MIT License
  - Pretrained Weights: CC-BY-NC 4.0 (Non-commercial only)
- **MambaVision** (NVIDIA Source Code License-NC)

#### 2. 🔓 Other Open Source Components

- SimCC (MIT)
- ConvNeXt V1 (MIT)
- EfficientNet (BSD-3-Clause via TorchVision)
- InceptionNeXt (Apache-2.0)
- HRNet (MIT)

