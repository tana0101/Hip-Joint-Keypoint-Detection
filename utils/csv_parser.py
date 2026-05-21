import pandas as pd
import numpy as np

def parse_csv_get_points(csv_path):
    """
    讀取 CSV 並回傳 numpy array，不預設點數，由資料決定
    """
    row = pd.read_csv(csv_path, header=None).values.flatten()
    pts = []
    for token in row:
        token = str(token).strip().strip('"').strip("'").strip()
        token = token.replace("(", "").replace(")", "")
        if not token: continue # 跳過空值
        x_str, y_str = token.split(",")
        pts.append([float(x_str), float(y_str)])
    return np.array(pts, dtype=np.float32)