import os
import re

def extract_info_from_model_path(model_path):
    """
    支援的命名格式：
    【雙階段】
      - model_simcc_1d_sr2.0_sigma6.0_cropright_mirror_224_300_0.0001_32_best.pth
      - model_direct_regression_cropleft_448_300_0.0001_32.pth
    【單階段】
      - model_simcc_2d_sr3.0_sigma7.0_onestage_512_300_0.0001_32_best.pth
      - model_direct_regression_onestage_512_300_0.0001_32.pth
      - model_heatmap_sigma2.0_onestage_512_300_0.0001_32_best.pth  <-- 新增支援
    """
    filename = os.path.basename(model_path)

    # 1) SimCC 與 Heatmap 系列
    pattern_simcc = re.compile(
        r'_('
        r'simcc_2d_deconv|simcc_2d|simcc_1d|heatmap'  # 加入 heatmap
        r')'                                      # group(1): head_type
        r'(?:_sr([0-9eE\.\-]+))?'                 # group(2): split_ratio (sr) -> 加上 (?:...)? 變成可選
        r'_sigma([0-9eE\.\-]+)'                   # group(3): sigma (這兩者都有)
        r'_(?:crop(?:left|right)(?:_mirror)?|onestage)' # 兼容雙階段與單階段
        r'_(\d+)'                                 # group(4): input_size
        r'_([0-9]+)'                              # group(5): epochs
        r'_([0-9eE\.\-]+)'                        # group(6): learning_rate
        r'_([0-9]+)'                              # group(7): batch_size
    )

    m = pattern_simcc.search(filename)
    if m:
        head_type     = m.group(1)
        # 因為 heatmap 沒有 sr，m.group(2) 會是 None，需要做安全轉換
        split_ratio   = float(m.group(2)) if m.group(2) is not None else None
        sigma         = float(m.group(3))
        input_size    = int(m.group(4))
        epochs        = int(m.group(5))
        learning_rate = float(m.group(6))
        batch_size    = int(m.group(7))
        return head_type, input_size, epochs, learning_rate, batch_size, split_ratio, sigma

    # 2) direct_regression (完全不變)
    pattern_dr = re.compile(
        r'_(direct_regression)'                   # group(1): head_type
        r'_(?:crop(?:left|right)(?:_mirror)?|onestage)' # 兼容雙階段與單階段
        r'_(\d+)'                                 # group(2): input_size
        r'_([0-9]+)'                              # group(3): epochs
        r'_([0-9eE\.\-]+)'                        # group(4): learning_rate
        r'_([0-9]+)'                              # group(5): batch_size
    )

    m2 = pattern_dr.search(filename)
    if m2:
        head_type     = m2.group(1)          
        input_size    = int(m2.group(2))
        epochs        = int(m2.group(3))
        learning_rate = float(m2.group(4))
        batch_size    = int(m2.group(5))
        split_ratio   = None
        sigma         = None
        return head_type, input_size, epochs, learning_rate, batch_size, split_ratio, sigma

    # 3) 都沒 match 就報錯
    raise ValueError(
        f"Model path format is invalid: {filename}\n"
        "Expected something like:\n"
        "  ..._simcc_2d_sr3.0_sigma7.0_cropleft_mirror_224_200_0.0001_32[_best.pth]\n"
        "  ..._heatmap_sigma2.0_onestage_512_200_0.0001_32[_best.pth]"
    )