# Detection bbox 
def _square_expand_clip(x1, y1, x2, y2, W, H, expand=0.10, keep_square=True):
    """把 bbox 變成(可選)正方形並外擴，再裁到影像邊界內。"""
    bw, bh = x2 - x1, y2 - y1
    if keep_square:
        side = max(bw, bh, 2.0)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        x1, x2 = cx - side / 2.0, cx + side / 2.0
        y1, y2 = cy - side / 2.0, cy + side / 2.0
        bw = bh = side
    # 外擴
    x1 -= bw * expand; x2 += bw * expand
    y1 -= bh * expand; y2 += bh * expand
    # 邊界裁切
    x1 = max(0.0, x1); y1 = max(0.0, y1)
    x2 = min(float(W), x2); y2 = min(float(H), y2)
    # 避免 0 尺寸
    if x2 <= x1: x2 = min(float(W), x1 + 2.0)
    if y2 <= y1: y2 = min(float(H), y1 + 2.0)
    return x1, y1, x2, y2

def _detect_one(yolo_model, pil_img, cls_id, conf=0.05, iou=0.5):
    res = yolo_model.predict(source=pil_img, conf=conf, iou=iou, max_det=20, verbose=False)
    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        return None

    boxes = res[0].boxes
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy()

    keep = (cls == int(cls_id))
    if not keep.any():
        return None

    idxs = keep.nonzero()[0]
    best = idxs[confs[idxs].argmax()]
    xyxy = boxes.xyxy[best].cpu().numpy().tolist()
    return tuple(float(v) for v in xyxy)