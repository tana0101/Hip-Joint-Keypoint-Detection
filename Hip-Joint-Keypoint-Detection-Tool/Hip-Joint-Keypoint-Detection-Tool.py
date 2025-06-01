import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from PIL import Image, ImageOps
from torchvision import transforms, models
import torch.nn as nn

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QComboBox, QLabel,
    QDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont

# --------------------------- 常數設定 ---------------------------
IMAGE_SIZE = 224           # 輸入模型的影像尺寸 (training 時使用)
POINTS_COUNT = 12          # 關鍵點數量
MODEL_PATH = os.path.join('model', 'efficientnet_keypoint_2000_0.01_32_best.pth')

# --------------------------- 公用函式 ---------------------------

def calculate_avg_distance(predicted_keypoints, original_keypoints):
    distances = np.linalg.norm(predicted_keypoints - original_keypoints, axis=1)
    return np.mean(distances)

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
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    perp_dx, perp_dy = -dy, dx
    scale = 1
    x0, y0 = point
    x_vals = np.array([x0 - scale * perp_dx, x0 + scale * perp_dx])
    y_vals = np.array([y0 - scale * perp_dy, y0 + scale * perp_dy])
    ax.plot(x_vals, y_vals, color=color, linewidth=1, label=label)

def draw_diagonal_line(ax, point, line_p1, line_p2, direction="left_down", color='orange', label=None):
    a = np.array(line_p1); b = np.array(line_p2); p = np.array(point)
    v = b - a
    proj = a + v * np.dot(p - a, v) / np.dot(v, v)
    length = 100
    dx, dy = (-length, length) if direction == 'left_down' else (length, length)
    x_vals = [proj[0], proj[0] + dx]
    y_vals = [proj[1], proj[1] + dy]
    ax.plot(x_vals, y_vals, color=color, linewidth=1, label=label)

def draw_h_point(ax, kpts):
    h_left  = (kpts[9] + kpts[11]) / 2
    h_right = (kpts[3] + kpts[5]) / 2
    ax.scatter(*h_left,  c='blue', s=4, label='H-point')
    ax.scatter(*h_right, c='blue', s=4)

def calculate_acetabular_index_angles(points):
    p1, p3, p7, p9 = points[0], points[2], points[6], points[8]
    v_h = p7 - p3
    v_left  = p3 - p1
    v_right = p7 - p9
    def ang(v1, v2):
        cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1., 1.)))
    return ang(v_h, v_left), ang(-v_h, v_right)

def classify_quadrant_ihdi(points):
    pts = np.asarray(points, dtype=float)
    p1,p3,p4,p6,p7,p9,p10,p12 = pts[0],pts[2],pts[3],pts[5],pts[6],pts[8],pts[9],pts[11]
    v_h = p7 - p3
    u_h = v_h / np.linalg.norm(v_h)
    v_p1 = np.array([ v_h[1],-v_h[0]])
    v_p2 = np.array([-v_h[1], v_h[0]])
    v_p  = v_p1 if v_p1[1] > v_p2[1] else v_p2
    u_p  = v_p / np.linalg.norm(v_p)
    o_r = p3 + np.dot(p9-p3,u_h)*u_h
    o_l = p3 + np.dot(p1-p3,u_h)*u_h
    h_r = (p10+p12)/2
    h_l = (p4+p6)/2
    def to_xy(p,o):
        vec=p-o
        return np.dot(vec,u_h), np.dot(vec,u_p)
    x_r,y_r = to_xy(h_r,o_r)
    x_l,y_l = to_xy(h_l,o_l); x_l=-x_l
    def quad(x,y):
        if y<=0 and x>=0: return 'IV'
        if y>0 and x<0:  return 'I'
        if y>0 and x>=0: return 'II' if y> x else 'III'
        return 'none'
    return quad(x_l,y_l), quad(x_r,y_r)

def draw_comparison_figure(
    image, pred_kpts, gt_kpts, ai_pred, ai_gt,
    quadrants_pred, quadrants_gt,
    avg_distance, save_path, image_file):
    fig, axes = plt.subplots(1,2,figsize=(12,6))
    for i,(kpts,title,ai,quads) in enumerate([
        (pred_kpts,'Using Predicted Keypoints',ai_pred,quadrants_pred),
        (gt_kpts,'Using Ground Truth Keypoints',ai_gt,quadrants_gt)]):
        ax=axes[i]
        ax.imshow(image,cmap='gray')
        ax.set_title(title); ax.axis('off')
        ax.scatter(pred_kpts[:,0],pred_kpts[:,1],c='yellow',s=4,label='Predicted')
        ax.scatter(gt_kpts[:,0],gt_kpts[:,1],c='red',s=4,label='Ground Truth')
        p1,p3,p4,p6,p7,p9,p10,p12 = kpts[0],kpts[2],kpts[3],kpts[5],kpts[6],kpts[8],kpts[9],kpts[11]
        ax.plot([p7[0],p9[0]],[p7[1],p9[1]],color='magenta',linewidth=1,label='Roof Line')
        ax.plot([p3[0],p1[0]],[p3[1],p1[1]],color='magenta',linewidth=1)
        draw_hilgenreiner_line(ax,p3,p7)
        draw_perpendicular_line(ax,p1,p3,p7,color='lime',label='P-line')
        draw_perpendicular_line(ax,p9,p3,p7,color='lime')
        draw_diagonal_line(ax,p1,p3,p7,'left_down','orange','Diagonal')
        draw_diagonal_line(ax,p9,p3,p7,'right_down','orange')
        draw_h_point(ax,kpts)
        lq,rq = quads
        ax.text(10,image.size[1]+35,f'AI L:{ai[0]:.1f}° (Q{lq})',color='magenta')
        ax.text(10,image.size[1]+85,f'AI R:{ai[1]:.1f}° (Q{rq})',color='magenta')
    diag_len = (image.size[0]**2+image.size[1]**2)**0.5
    avg_pct = avg_distance / diag_len * 100
    fig.text(0.5,-0.05,f'Avg Dist: {avg_distance:.2f}px ({avg_pct:.2f}%)',ha='center',color='blue')
    axes[0].legend(loc='lower left'); plt.tight_layout()
    os.makedirs(save_path,exist_ok=True)
    fig.savefig(os.path.join(save_path,f'{os.path.splitext(image_file)[0]}_compare.png'),bbox_inches='tight')
    plt.close(fig)

def draw_only_prediction_figure(image, pred_kpts, save_path, image_file):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')
    ax.set_title("Prediction Only")
    ax.axis('off')
    ax.scatter(pred_kpts[:, 0], pred_kpts[:, 1], c='yellow', s=4, label='Predicted')
    p1, p3, p4, p6, p7, p9, p10, p12 = pred_kpts[0], pred_kpts[2], pred_kpts[3], pred_kpts[5], pred_kpts[6], pred_kpts[8], pred_kpts[9], pred_kpts[11]
    ax.plot([p7[0], p9[0]], [p7[1], p9[1]], color='magenta', linewidth=1, label='Roof Line')
    ax.plot([p3[0], p1[0]], [p3[1], p1[1]], color='magenta', linewidth=1)
    draw_hilgenreiner_line(ax, p3, p7)
    draw_perpendicular_line(ax, p1, p3, p7, color='lime', label='P-line')
    draw_perpendicular_line(ax, p9, p3, p7, color='lime')
    draw_diagonal_line(ax, p1, p3, p7, direction="left_down", color='orange', label='Diagonal')
    draw_diagonal_line(ax, p9, p3, p7, direction="right_down", color='orange')
    draw_h_point(ax, pred_kpts)
    ax.legend(loc='lower left')
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"{image_file}_predonly.png"), bbox_inches='tight')
    plt.close(fig)
    
# --------------------- PyQt 主應用程式 --------------------------
class KeypointAnnotationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keypoint Prediction & Annotation Tool")
        self.image_files=[]; self.current_image_idx=0; self.keypoints=[]
        self._init_model()
        self._init_ui()
    # ----------------- UI -----------------
    def _init_ui(self):
        layout = QVBoxLayout()
        # 資料夾讀取
        self.load_button = QPushButton("讀入資料夾"); self.load_button.clicked.connect(self.load_images)
        layout.addWidget(self.load_button)
        # 圖片下拉
        self.image_selector = QComboBox(); self.image_selector.currentIndexChanged.connect(self.on_image_selected)
        layout.addWidget(self.image_selector)
        # 上 / 下一張
        self.prev_button = QPushButton("上一張"); self.prev_button.clicked.connect(self.prev_image)
        self.next_button = QPushButton("下一張"); self.next_button.clicked.connect(self.next_image)
        layout.addWidget(self.prev_button); layout.addWidget(self.next_button)
        # 標註 / 清除
        self.label_button = QPushButton("標註圖片"); self.label_button.clicked.connect(self.label_image)
        self.clear_button = QPushButton("清除標記"); self.clear_button.clicked.connect(self.clear_annotations)
        layout.addWidget(self.label_button); layout.addWidget(self.clear_button)
        # 預測 & 展示
        self.predict_button = QPushButton("預測"); self.predict_button.clicked.connect(self.predict_all_images)
        self.show_pred_button = QPushButton("展示"); self.show_pred_button.clicked.connect(self.show_comparison)
        layout.addWidget(self.predict_button); layout.addWidget(self.show_pred_button)
        # 圖片顯示
        self.image_label = QLabel(); layout.addWidget(self.image_label)
        self.setLayout(layout)
    # ----------------- Model -----------------
    def _init_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base = models.efficientnet_v2_m(pretrained=True)
        base.classifier = nn.Sequential(
            nn.Linear(base.classifier[1].in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, POINTS_COUNT * 2)
        )
        # Load weights onto the same device as model
        base.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        base = base.to(self.device)  # explicitly move model to device
        base.eval()
        self.model = base
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  
            transforms.Lambda(lambda img: ImageOps.equalize(img)),  # Apply histogram equalization
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
    # ----------------- I/O -----------------
    def load_images(self):
        folder = QFileDialog.getExistingDirectory(self,"選擇資料夾")
        if folder:
            self.root_dir = folder
            exts=(".jpg",".png",".jpeg",".bmp")
            self.image_files=[os.path.join(folder,'images',f) if 'images' not in folder else os.path.join(folder,f)
                              for f in os.listdir(folder) if f.lower().endswith(exts)]
            self.image_selector.clear(); self.image_selector.addItems([os.path.basename(f) for f in self.image_files])
            if self.image_files:
                self.display_image(0)
    # ----------------- 顯示 -----------------
    def on_image_selected(self,idx):
        self.current_image_idx=idx; self.display_image(idx)
    def display_image(self,idx):
        if 0<=idx<len(self.image_files):
            img=cv2.imread(self.image_files[idx]); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            h,w,_=img.shape; scale=0.1; new_h,new_w=int(h*scale),int(w*scale)
            img_res=cv2.resize(img,(new_w,new_h))
            # 若有 GT 標註，畫點
            img_name=os.path.splitext(os.path.basename(self.image_files[idx]))[0]
            csv_path=os.path.join('annotations',f'{img_name}.csv')
            if os.path.exists(csv_path):
                with open(csv_path) as f:
                    matches=re.findall(r"\((\d+),\s*(\d+)\)",f.readline())
                for j,(x,y) in enumerate(matches,1):
                    xs,ys=int(int(x)*new_w/w),int(int(y)*new_h/h)
                    cv2.circle(img_res,(xs,ys),1,(255,0,0),-1); cv2.putText(img_res,str(j),(xs+5,ys-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
            qimg=QImage(img_res.data,new_w,new_h,3*new_w,QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qimg))
    # 上下
    def next_image(self):
        if self.current_image_idx<len(self.image_files)-1:
            self.current_image_idx+=1; self.image_selector.setCurrentIndex(self.current_image_idx)
        else: self._msg('提示','已經是最後一張')
    def prev_image(self):
        if self.current_image_idx>0:
            self.current_image_idx-=1; self.image_selector.setCurrentIndex(self.current_image_idx)
        else: self._msg('提示','已經是第一張')
    # ----------------- 標註相關 -----------------
    def label_image(self):
        self.label_window=LabelWindow(self.image_files[self.current_image_idx],self.save_keypoints)
        self.label_window.show()
    def save_keypoints(self,kpts):
        img_name=os.path.splitext(os.path.basename(self.image_files[self.current_image_idx]))[0]
        os.makedirs('annotations',exist_ok=True)
        with open(os.path.join('annotations',f'{img_name}.csv'),'w') as f:
            f.write(','.join([f'"({x}, {y})"' for x,y in kpts]))
        self._msg('提示','已儲存標註'); self.display_image(self.current_image_idx)
        self.generate_visualization(img_name)
    def clear_annotations(self):
        img_name=os.path.splitext(os.path.basename(self.image_files[self.current_image_idx]))[0]
        csv_path=os.path.join('annotations',f'{img_name}.csv')
        if os.path.exists(csv_path): os.remove(csv_path); self._msg('提示','已刪除標註')
        self.display_image(self.current_image_idx)
    # ----------------- 預測 -----------------
    def predict_all_images(self):
        if not self.image_files:
            self._msg('錯誤','請先載入圖片資料夾'); return
        os.makedirs('annotations_predicts',exist_ok=True)
        os.makedirs('predicts',exist_ok=True)
        for img_path in self.image_files:
            img = Image.open(img_path).convert('RGB')
            w,h = img.size
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = self.model(tensor).cpu().numpy().reshape(-1,2)
            # 反縮放至原尺寸
            preds_scaled = preds * np.array([w/IMAGE_SIZE, h/IMAGE_SIZE])
            # 存 CSV
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            csv_pred = os.path.join('annotations_predicts',f'{img_name}.csv')
            with open(csv_pred, 'w') as f:
                f.write(','.join([f'"({int(round(x))}, {int(round(y))})"' for x, y in preds_scaled]))
            # 若有 GT annotations，產生 compare
            self.generate_visualization(img_name)
        self._msg('完成','全部圖片已完成預測並生成對照圖')
    # ----------------- 展示比較圖 -----------------
    def show_comparison(self):
        if not self.image_files:
            return

        img_name = os.path.splitext(os.path.basename(self.image_files[self.current_image_idx]))[0]
        compare_path = os.path.join('predicts', f'{img_name}_compare.png')
        predonly_path = os.path.join('predicts', f'{img_name}_predonly.png')

        if os.path.exists(compare_path):
            image_path = compare_path
            title = f'對照圖 - {img_name}'
        elif os.path.exists(predonly_path):
            image_path = predonly_path
            title = f'預測圖（無標註） - {img_name}'
        else:
            self._msg('提示', '尚未找到對照圖或預測圖，請先執行預測')
            return

        # 顯示圖片
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        lbl = QLabel(dlg)
        lbl.setGeometry(0, 0, w, h)

        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qimg))

        dlg.resize(w, h)
        dlg.exec_()
    # ----------------- 生成對比或預測圖 -----------------
    def generate_visualization(self, image_name):
        img_path = os.path.join('images', f"{image_name}.jpg")
        gt_path = os.path.join('annotations', f"{image_name}.csv")
        pred_path = os.path.join('annotations_predicts', f"{image_name}.csv")

        if not os.path.exists(img_path):
            self._msg("錯誤", f"找不到圖片：{img_path}")
            return

        if not os.path.exists(pred_path):
            self._msg("提示", f"{image_name} 尚未有預測結果，請先執行預測")
            return

        # 讀入原圖
        image = Image.open(img_path).convert("RGB")

        # 讀入預測點
        with open(pred_path) as f:
            pred_matches = re.findall(r"\((\d+),\s*(\d+)\)", f.readline())
        pred_kpts = np.array([(int(x), int(y)) for x, y in pred_matches], dtype=float)
        
        # 兩種情況
        if os.path.exists(gt_path):
            with open(gt_path) as f:
                gt_matches = re.findall(r"\((\d+),\s*(\d+)\)", f.readline())
            gt_kpts = np.array([(int(x), int(y)) for x, y in gt_matches], dtype=float)

            avg_d = calculate_avg_distance(pred_kpts, gt_kpts)
            ai_pred = calculate_acetabular_index_angles(pred_kpts)
            ai_gt = calculate_acetabular_index_angles(gt_kpts)
            q_pred = classify_quadrant_ihdi(pred_kpts)
            q_gt = classify_quadrant_ihdi(gt_kpts)

            draw_comparison_figure(
                image=image,
                pred_kpts=pred_kpts,
                gt_kpts=gt_kpts,
                ai_pred=ai_pred,
                ai_gt=ai_gt,
                quadrants_pred=q_pred,
                quadrants_gt=q_gt,
                avg_distance=avg_d,
                save_path='predicts',
                image_file=image_name
            )
        else:
            draw_only_prediction_figure(image, pred_kpts, 'predicts', image_name)
    # ----------------- 工具 -----------------
    def _msg(self,title,msg):
        QMessageBox.information(self,title,msg)

# ----------------- Label Window -----------------
class LabelWindow(QWidget):
    def __init__(self,image_path,cb):
        super().__init__(); self.image_path=image_path; self.save_callback=cb; self.keypoints=[]; self._init()
    def _init(self):
        self.setWindowTitle('標註圖片'); self.resize(800,600)
        img=cv2.imread(self.image_path); img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.orig_h,self.orig_w,_=img.shape
        self.scale = (800/self.orig_w, 600/self.orig_h)
        self.img_disp=cv2.resize(img,(800,600)); self.back=self.img_disp.copy()
        self.canvas=QLabel(self); self.canvas.setGeometry(0,0,800,600); self._update()
    def _update(self):
        qimg=QImage(self.img_disp.data,800,600,800*3,QImage.Format_RGB888)
        self.canvas.setPixmap(QPixmap.fromImage(qimg))
    def mousePressEvent(self,e):
        if e.button()==Qt.LeftButton:
            x,y=e.x(),e.y(); ox=int(x/self.scale[0]); oy=int(y/self.scale[1])
            self.keypoints.append((ox,oy)); self._redraw()
        elif e.button()==Qt.RightButton and self.keypoints:
            self.keypoints.pop(); self._redraw()
    def _redraw(self):
        self.img_disp=self.back.copy()
        for i,(x,y) in enumerate(self.keypoints,1):
            xs,ys=int(x*self.scale[0]),int(y*self.scale[1])
            cv2.circle(self.img_disp,(xs,ys),3,(255,0,0),-1)
            cv2.putText(self.img_disp,str(i),(xs+5,ys-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        self._update()
    def closeEvent(self,e):
        self.save_callback(self.keypoints); e.accept()

# -------------------- Run --------------------
if __name__ == "__main__":
    app = QApplication([])
    app.setFont(QFont("Noto Sans CJK TC", 10))
    window = KeypointAnnotationApp()
    window.show()
    app.exec_()
