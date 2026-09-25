# config.py
import numpy as np

class Paths:
    """檔案與路徑設定"""
    LOGS_DIR = "logs"
    MODELS_DIR = "weights"

class EMA:
    """指數移動平均 (Exponential Moving Average) 設定"""
    DECAY = 0.995
    RAMPUP_STEPS = 500
    START_DECAY = 0.90

class Augment:
    """資料增強與 Bounding Box 設定"""
    BBOX_JITTER = True
    BBOX_EXPAND = 0.05
    BBOX_JITTER_PROB = 0.7
    BBOX_JITTER_CENTER = 0.15 # 0.05
    BBOX_JITTER_SCALE = 0.20 # 0.10
    
    PROB = 0.7
    MAX_TRANSLATE_X = 5 # 10
    MAX_TRANSLATE_Y = 5 # 10
    MAX_ANGLE = 12 # 5

class FullImageAugment:
    """全圖資料增強設定"""
    PROB = 0.7
    MAX_ANGLE = 10
    TRANS_RATIO = 0.05
    SCALE_RANGE = 0.15
    CLAMP = True

class YOLOConfig:
    """YOLO 模型推論設定"""
    LEFT_CLS = 0
    RIGHT_CLS = 1
    CONF = 0.001
    IOU = 0.7
    BBOX_EXPAND = 0.05

class Eval:
    """評估與閥值設定"""
    DISTANCE_BINS = [
        (0.0, 2.5,   "0-2.5"),
        (2.5, 5.0,   "2.5-5"),
        (5.0, 7.5,   "5-7.5"),
        (7.5, 10.0,  "7.5-10"),
        (10.0, 12.5, "10-12.5"),
        (12.5, 15.0, "12.5-15"),
        (15.0, np.inf,"15+"),
    ]
    # outlier thresholds
    PIX_TH = 10.0    # pixel distance threshold
    ANG_TH = 8.0     # degree threshold
    
class Dataset:
    """資料集與標籤設定"""
    SIDE_LABELS = {"left": "LeftHip", "right": "RightHip"}
    
    CONFIGS_BY_COUNT = {
        12: {
            "name": "IHDI_12pt",
            # IHDI 的左右定義鏡像後需要交叉重排: [0,1,2] -> [2,1,0]
            "mirror_reorder": [2, 1, 0, 5, 4, 3] 
        },
        8: {
            "name": "MTDDH_8pt",
            # MTDDH 假設是對稱定義 (1外/2內)，鏡像後通常順序不變，或根據實際情況調整
            "mirror_reorder": [0, 1, 2, 3] 
        }
    }