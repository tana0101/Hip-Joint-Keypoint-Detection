import numpy as np

# ============================================================
# 幾何計算區
# ============================================================

# 計算 AI 角度
def calculate_acetabular_index_angles(points):
    p1 = points[0]
    p3 = points[2]
    p7 = points[6]
    p9 = points[8]
    v_h = p7 - p3
    v_left = p3 - p1
    v_right = p7 - p9
    def angle(v1, v2):
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return angle(v_h, v_left), angle(-v_h, v_right)

# 判斷IHDI象限
def classify_quadrant_ihdi(points):
    """
    根據 IHDI 分類圖，判斷左右 H-point 各自落在哪個象限。
    
    參數
    -------
    points : array-like, shape=(12, 2)
        依序為 1~12 點的 (x, y) 影像座標（x 向右、y 向下）。
    
    回傳
    -------
    left_q , right_q : str
        左、右股骨頭中心所在象限，字元 'I' ~ 'IV'
        
    實作方式
    -------
    將股骨頭中心點（H-point）投影到以 H-line 為水平、P-line 為垂直的局部座標系中，並依其位置判斷其落在哪個象限（I~IV）。
    
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape != (12, 2):
        raise ValueError("`points` 必須是 (12, 2) 的座標陣列")

    # ---- 取出關鍵點 --------------------------------------------------------
    p1  = pts[0]      # 左 P-line 基準
    p3  = pts[2]      # H-line 起點
    p4  = pts[3]      # 左股骨頭
    p6  = pts[5]
    p7  = pts[6]      # H-line 終點
    p9  = pts[8]      # 右 P-line 基準
    p10 = pts[9]      # 右股骨頭
    p12 = pts[11]

    # ---- 基準座標系：x 軸沿 H-line，y 軸向下 ------------------------------
    v_h = p7 - p3
    u_h = v_h / np.linalg.norm(v_h)                     # 單位 H 向量

    # 兩種垂直方向：順時針 / 逆時針 90°
    v_p1 = np.array([ v_h[1], -v_h[0]])
    v_p2 = np.array([-v_h[1],  v_h[0]])
    v_p  = v_p1 if v_p1[1] > v_p2[1] else v_p2          # y 分量大的朝「下」
    u_p  = v_p / np.linalg.norm(v_p)                    # 單位「下」向量

    # H-line 與 P-line 交點（作為原點）
    def proj_on_h(pt):                                  # 投影到 H-line
        t = np.dot(pt - p3, u_h)
        return p3 + t * u_h

    o_r = proj_on_h(p9)                                 # 右側原點
    o_l = proj_on_h(p1)                                 # 左側原點

    # H-points
    h_r = (p10 + p12) / 2.0
    h_l = (p4  + p6 ) / 2.0

    # 轉成局部 (x, y)
    def to_xy(p, o):
        vec = p - o
        return np.dot(vec, u_h), np.dot(vec, u_p)

    x_r, y_r = to_xy(h_r, o_r)
    x_l, y_l = to_xy(h_l, o_l)
    x_l = -x_l            # 左側鏡射：使遠離脊椎方向為 x 正

    # ---- 象限規則（以右側為基準） -----------------------------------------
    def quad(x, y):
        # IV：在 H-line 上方／接近上方
        if y <= 0 and x >= 0:
            return 'IV'
        # I：H-line 下 + P-line 左
        if y > 0 and x < 0:
            return 'I'
        # II / III：H-line 下 + P-line 右，依對角線分界
        if y > 0 and x >= 0:
            return 'II' if y > x else 'III'
        # 其他極少見邊界情況 → 歸為 none
        return 'none'

    return quad(x_l, y_l), quad(x_r, y_r)

# ============================================================
# 繪圖輔助函式區
# ============================================================

def draw_hilgenreiner_line(ax, p3, p7):
    if np.allclose(p3[0], p7[0]):
        ax.axvline(x=p3[0], color='cyan', linewidth=1, label='H-Line')
    else:
        a = (p7[1] - p3[1]) / (p7[0] - p3[0])
        b = p3[1] - a * p3[0]
        x_min, x_max = ax.get_xlim()
        x_vals = np.array([x_min, x_max])
        y_vals = a * x_vals + b
        ax.plot(x_vals, y_vals, color='cyan', linewidth=1, label='H-Line')
        
def draw_perpendicular_line(ax, point, line_p1, line_p2, color='lime', label=None):
    # 向量方向（H-line）
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]

    # 垂直向量：旋轉90度 (-dy, dx)
    perp_dx = -dy
    perp_dy = dx

    # 將垂直向量延長一定比例來畫線
    scale = 1  # 可調整線的長度
    x0, y0 = point
    x_vals = np.array([x0 - scale * perp_dx, x0 + scale * perp_dx])
    y_vals = np.array([y0 - scale * perp_dy, y0 + scale * perp_dy])

    ax.plot(x_vals, y_vals, color=color, linewidth=1, label=label)
    
def draw_diagonal_line(ax, point, line_p1, line_p2, direction="left_down", color='orange', label=None):
    a = np.array(line_p1)
    b = np.array(line_p2)
    p = np.array(point)
    v = b - a

    # 計算投影點（交點）
    proj = a + v * np.dot(p - a, v) / np.dot(v, v)

    # 決定方向
    length = 100
    if direction == "left_down":
        dx, dy = -length, length   # 斜率 +1
    elif direction == "right_down":
        dx, dy = length, length    # 斜率 -1
    else:
        raise ValueError("direction must be 'left_down' or 'right_down'")

    x_vals = [proj[0], proj[0] + dx]
    y_vals = [proj[1], proj[1] + dy]
    ax.plot(x_vals, y_vals, color=color, linewidth=1, label=label)

def draw_h_point(ax, kpts):
    h_left = (kpts[9] + kpts[11]) / 2   # pt10 & pt12 中點
    h_right = (kpts[3] + kpts[5]) / 2   # pt4 & pt6 中點

    # 使用亮黃色 + 圓形
    ax.scatter(*h_left, c="blue", s=4, label='H-point')
    ax.scatter(*h_right, c="blue", s=4)