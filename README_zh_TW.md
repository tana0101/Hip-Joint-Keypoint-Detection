# 🦴 應用深度學習關鍵點偵測技術於小兒髖關節 X 光影像之發育不良檢測

<div align="center">
  <div>
    <a href="https://github.com/tana0101/Hip-Joint-Keypoint-Detection/blob/main/README_zh_TW.md">🇹🇼繁體中文</a> |
    <a href="https://github.com/tana0101/Hip-Joint-Keypoint-Detection/blob/main/README.md">🌏English</a> |
    <a href="https://deepwiki.com/tana0101/Hip-Joint-Keypoint-Detection">📚DeepWiki</a> |
    <a href="https://github.com/tana0101/Hip-Joint-Keypoint-Detection/issues">❓issues</a> |
    📝Paper(受保密條款限制)
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

## 📋Overview

小兒髖關節發育不良（Developmental Dysplasia of the Hip, **DDH**）若未能及早診斷，可能對孩童未來的行走能力造成長期影響。臨床實務上，高度仰賴醫師人工判讀 X 光影像，容易產生主觀量測差異。

本專案提出一套**由上而下（Top-down）的兩階段深度學習偵測系統**，目標為輔助 DDH 的量測與分級：
1. **自動偵測與定位**：以 YOLO 準確定位髖關節區域（ROI）。
2. **關鍵點預測**：從 ROI 中分別定位關鍵點，共標記 12 或 8 個髖關節關鍵點（依資料集而定）。
3. **臨床指標運算**：自動計算 **Acetabular Index (AI) angle**，並輸出 **IHDI 分類**結果。

> **Research-only Notice**：本專案為資訊工程研究用途之原型系統，輸出結果不得直接作為臨床診斷依據。

## ✨Key Features

- 🧩 **高度模組化架構 (Modular Design)**
  將物件偵測與關鍵點預測解耦，並設計了統一的模型接口，支援快速抽換實驗配置：
  - **Backbones**: 內建 ConvNeXt (V1/V2)、HRNet，並提供擴充介面以相容其他網路架構。
  - **Prediction Heads**: 支援官方 SimCC、改良版 SimCC、傳統 Direct Regression 與 Heatmap 方法。
- ⚙️ **微服務與後端化整合 (Backend-Ready)**
  核心推論流程已封裝為後端 Inference API 形式（目前在成大醫院測試中）。
- 📊 **嚴謹的評估機制**
  內建完整的 K-Fold 交叉驗證與數據可視化，自動產出混淆矩陣與誤差分佈圖表。

## 💾Dataset

- 資料存放於 `dataset/` 目錄中。
- 標註程式位於 `Keypoint-Annotation-Tool/` 目錄中。

### 🏥NCKUH_IHDI（成大醫院資料集）
<img src="src/img/sample_IHDI.jpg" style="width: 30%;"/>

本研究採用回溯性資料，收集來自成大醫院於 2019/06/25 至 2025/01/07 期間，影像拍攝時年齡介於 1 至 59 個月之髖部 X 光影像。原始納入 622 份影像，經排除異常值後保留 557 張。每張影像由臨床專業醫師手動標註 **12 個關鍵點**，並提供 LeftHip / RightHip 物件標籤供偵測階段訓練。

- 影像標註：每張影像對應一個 .csv，格式：
```
"(x1,y1)","(x2,y2)",...,"(x12,y12)"
```
- 備註：基於醫療隱私與資料保護規範，無法公開釋出。

<hr>

### 🌍MTDDH（公開資料集）
<img src="src/img/sample_MTDDH.jpg" style="width: 30%;"/>

- 資料來源：[open-hip-dysplasia](https://github.com/radoss-org/open-hip-dysplasia.git)
- 資料量：1666 張髖關節 X 光影像（已排除異常值）
- 標註內容：
  - **8 個關鍵點**
  - LeftHip / RightHip 物件標籤

<hr>

### 📊資料統計分佈

**Acetabular Index (AI) 分佈**
<img src="dataset/xray_IHDI_AI_Distribution.png" />
<img src="dataset/mtddh_xray_2d_AI_Distribution.png" />

**IHDI 分類分佈**
<div style="display: flex; justify-content: space-between; gap: 10px;">
  <img src="dataset/xray_IHDI_IHDI_Distribution.png" style="width: 49%;" />
  <img src="dataset/mtddh_xray_2d_IHDI_Distribution.png" style="width: 49%;" />
</div>

## 🏆 Model Performance & Results

使用 `ConvNeXtTinyMS` 搭配 `SimCC 2D` 模型，在公開資料集（MTDDH）上進行 5-fold 交叉驗證的 OOF (Out-of-Fold) 結果如下：

<p align="center">
  <img src="src/img/avg_dists.png" width="75%">
  <br>
  <b>(a) Keypoint Distance Error</b>
</p>

<p align="center">
  <img src="src/img/AI_angle_errors.png" width="75%">
  <br>
  <b>(b) AI Angle Error</b>
</p>

<p align="center">
  <img src="src/img/CM_4Class_all.png" width="75%">
  <br>
  <b>(c) IHDI 4-Class Confusion Matrix</b>
</p>

<p align="center">
  <img src="src/img/bland_altman_overall_ai_angle.png" width="53%">
  <img src="src/img/scatter_overall_ai_angle.png" width="36%">
  <br>
  <b>(d) Bland-Altman & Scatter Plot of AI Angle</b>
</p>

## 🛠️ Methodology

本專案採用由上而下（Top-down）的兩階段關鍵點偵測流程：
1. **🔍 物件偵測與單邊裁切**：以 YOLO 偵測 LeftHip / RightHip，裁切 ROI（降低背景干擾）。
2. **🧠 單側關鍵點偵測**：對裁切後單側 hip ROI 進行關鍵點偵測（多 Backbone / Head 研究比較）。

### Head Architecture

<img src="src/img/head_design.png" style="width: 99%;" />

本專案支援多種關鍵點 Head 設計，以因應不同模型特性與實驗需求：

- **SimCC 2D / SimCC 2D Deconv**：
官方 SimCC 系列，將座標回歸轉為 x/y 一維分類分佈，透過 soft-argmax 取得座標。
- **SimCC 1D（自訂變體）**：
以 Global Average Pooling 壓縮特徵圖，使用全連接層預測 x/y 分佈以降低複雜度。
- **Direct Regression**：
以全連接層直接回歸 (x, y) 座標。
- **Heatmap**：
沿用原生 HRNet 的實作方式，輸出二維熱力圖並搭配 MSE 損失函數。解碼時採用 Argmax 尋找最大值，並朝次高點方向進行 0.25 像素的次像素偏移 (Sub-pixel shift) 以補償量化誤差。

> 💡 **詳細的 Head 模型與解碼邏輯實作，請參考原始碼：[`models/head.py`](models/head.py)**

### Backbone Architecture

目前支援的 Backbone 架構如下：

- ConvNeXtV1  
  - `ConvNeXtTinyCustom`
- ConvNeXtV1 + Feature Pyramid Network（多尺度特徵）  
  - `ConvNeXtTinyMS`（convnext_tiny_fpn1234concat）
- HRNet  
  - `HRNetW32Custom`
  - `HRNetW48Custom`

其中 `Custom` 結尾代表本專案基於官方實作進行修改與優化，以更符合髖關節關鍵點偵測任務的需求。

> 💡 **所有 Backbone 模型，請參考原始碼：[`models/model.py`](models/model.py)**

🚧 **其他 Backbone（如 EfficientNet、InceptionNeXt 等）目前仍在開發與測試中，以確保可與不同 Head 架構相容** 🚧

### Other Techniques

- **🔄 Data Augmentation**：Random Rotation / Random Translation
- **📉 Loss Functions**
  - **Direct Regression (MSE Loss)**
  - **Heatmap (MSE Loss)**
  - **SimCC Series (KL Divergence Loss)**
- **⚙️ Optimizers**：AdamW
- **📈 LR Schedulers**：Cosine Annealing + Warmup
- **Decoder**：Expectation、Heuristic

  > 💡 **詳細的 Decoder 實作，請參考原始碼：[`utils/simcc.py`](utils/simcc.py)**

## 📂Project Structure

```text
Hip-Joint-Keypoint-Detection/
├── dataset/                        # 💾 資料集存放目錄
│   ├── xray_IHDI_2_clean/          # 成大醫院資料集 (Private)
│   └── mtddh_xray_2d/              # 公開資料集 (Public)
├── datasets/                       # 資料集載入與處理模組
├── models/                         # 🧠 模型與 Head 的定義與實作
├── src/                            # 系統核心資源與圖片
│   └── img/                        # README 使用之展示圖片
├── utils/                          # 🛠️ 通用工具函式庫
├── weights/                        # 📥 訓練完成的模型權重存放區 (.pth)
├── logs/                           # 📝 訓練過程日誌與損失曲線圖
├── results/                        # 📊 統計結果輸出目錄
├── experiment/                     # 所有過往實驗之記錄
├── Keypoint-Annotation-Tool/       # 🖊️ 關鍵點標註工具
├── train_yolo.py                   # [訓練] YOLO 物件偵測模型
├── train_full_image_keypoints.py   # [訓練] 雙側全圖關鍵點模型
├── train_hip_crop_keypoints.py     # [訓練] 單側髖關節關鍵點模型
├── predict_full_image_keypoints.py # [推論] 針對全圖執行完整的偵測與評估
├── predict_hip_crop_keypoints.py   # [推論] 針對單側分別執行偵測與評估
├── split.py                        # [工具] 資料集分割 (Train/Val/Test)
├── split_from_json.py              # [工具] 按照公開資料集的切分 json 進行資料集分割
├── kfold_split.py                  # [工具] K-Fold 資料集分割
├── kfold_train_yolo.py             # [K-Fold] YOLO 交叉驗證訓練
├── kfold_train_hip_crop_keypoints.py # [K-Fold] 關鍵點交叉驗證訓練
├── kfold_predict_hip_crop_keypoints.py # [K-Fold] 交叉驗證評估
├── requirements.txt                # 📦 專案依賴套件清單
├── README.md                       # 🇬🇧 英文說明文件
└── README_zh_TW.md                 # 🇹🇼 繁體中文說明文件
```

## 🚀 Workflow Pipeline

本系統的標準實驗與推論流程（Pipeline）如下：

1. 📂 **Data Preparation**：先將資料集準備好（公開資料集可使用`prepare_MTDDH_dataset` 腳本），接著執行 `split.py` 或 `kfold_split.py` 切割資料集。
2. 🔍 **Stage 1 (Detection)**：執行 `train_yolo.py` 或 `kfold_train_yolo.py` 訓練 YOLO 模型，進行髖關節 ROI 裁切，降低背景干擾。
3. 🧠 **Stage 2 (Keypoint)**：執行 `train_hip_crop_keypoints.py` 或 `kfold_train_hip_crop_keypoints.py`，載入指定的 Backbone 與 Head 進行單側關鍵點模型訓練。
4. 📈 **Evaluation**：執行 `predict_hip_crop_keypoints.py` 或 `kfold_predict_hip_crop_keypoints.py` 推論腳本進行資料集的預測，計算 AI angle 誤差與 IHDI 準確率。

> 💡 **詳細的環境建置、K-Fold 交叉驗證流程、以及完整的指令參數設定，請參閱 [📖 進階使用指南 (Usage Guide)](docs/USAGE_GUIDE.md)。**

## 🔧 Analysis & Visualization Tools

除了主要的訓練與評估流程外，本專案亦提供以下工具協助資料與模型結果分析：

- `visualize_single_simcc.py`  
  可視化單張影像的關鍵點預測結果與 SimCC 機率分布，適合用於模型預測結果檢視與異常案例分析。  
  結果輸出至 `simcc_vis_results/`。

- `analyze_hip_dataset.py`  
  統計資料集的樣本數、AI 與 IHDI 等資料分布，並產生對應統計圖表。  
  產生的結果可於 `dataset/` 對應資料集目錄中查看。

- `compare_dataset_annotations.py`  
  比較同一資料集不同標註版本之關鍵點、AI 與相關量測結果差異，可用於分析標註一致性。  
  結果輸出至 `annotation_comparison_results/`。

## ⚖️License

本專案採用 **GNU Affero General Public License v3.0 (AGPL-3.0)** 授權發布。
詳情請參閱 [LICENSE](LICENSE) 檔案。

### 核心依賴與開源鳴謝：

* **[Ultralytics YOLO](https://docs.ultralytics.com/)** (AGPL-3.0)
    * **影響**：由於本專案依賴 YOLO 進行核心訓練與推論，因此整體專案繼承 AGPL-3.0 規範。若您將本專案（或其修改版本）作為網路服務提供或公開發布，您必須公開您的原始碼。
* **[ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2)** (Meta Research)
    * **代碼**：MIT License。
    * **預訓練權重**：**CC-BY-NC 4.0 (僅限非商業用途)**。
    * **⚠️注意**：若您載入了 ConvNeXt V2 的官方預訓練權重，本專案將被限制為僅供學術研究或非商業用途。
* **[MambaVision](https://github.com/NVlabs/MambaVision)** (NVIDIA)
    * **授權**：NVIDIA Source Code License-NC (通常含有非商業用途限制，請參閱其 Repo 確認)。
* **[SimCC](https://github.com/leeyegy/SimCC)** (MIT)
* **[ConvNeXt V1](https://github.com/facebookresearch/ConvNeXt)** (MIT)
* **[EfficientNet](https://docs.pytorch.org/vision/main/models/efficientnet.html)** (BSD-3-Clause via TorchVision)
* **[InceptionNeXt](https://github.com/sail-sg/inceptionnext)** (Apache-2.0)
* **[HRNet (Bottom-Up)](https://github.com/HRNet/HRNet-Bottom-Up-Pose-Estimation)** (MIT)

感謝 SimCC、HRNet 等開源專案對深度學習社群的貢獻。若有意進行學術探討或後端串接合作，歡迎透過 Issue 聯絡。