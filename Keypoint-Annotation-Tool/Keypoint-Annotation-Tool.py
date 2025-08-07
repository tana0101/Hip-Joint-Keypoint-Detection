import os
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QComboBox, QLabel, QDialog, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import re
import json

class KeypointAnnotationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keypoint Annotation Tool")
        
        self.image_files = []  # 存儲所有圖像文件的路徑
        self.current_image_idx = 0  # 當前顯示的圖像索引
        self.keypoints = []  # 存儲標註點的列表
        self.init_ui()
        self.image_selector.currentIndexChanged.connect(self.on_image_selected)

    def init_ui(self):
        layout = QVBoxLayout()

        # 讀入按鈕
        self.load_button = QPushButton("讀入資料夾")
        self.load_button.clicked.connect(self.load_images)
        layout.addWidget(self.load_button)

        # 下拉選單
        self.image_selector = QComboBox()
        layout.addWidget(self.image_selector)

        # 下一張按鈕
        self.next_button = QPushButton("下一張")
        self.next_button.clicked.connect(self.next_image)
        layout.addWidget(self.next_button)

        # 上一張按鈕
        self.prev_button = QPushButton("上一張")
        self.prev_button.clicked.connect(self.prev_image)
        layout.addWidget(self.prev_button)

        # 清除按鈕
        self.clear_button = QPushButton("清除標記")
        self.clear_button.clicked.connect(self.clear_annotations)
        layout.addWidget(self.clear_button)

        # 顯示圖片的Label
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # 標註按鈕
        self.label_button = QPushButton("關鍵點標註")
        self.label_button.clicked.connect(self.label_image)
        layout.addWidget(self.label_button)

        # 展示標記點按鈕
        self.show_button = QPushButton("展示標記點")
        self.show_button.clicked.connect(self.show_annotations)
        layout.addWidget(self.show_button)
        
        # 標註髖關節物件
        self.detect_button = QPushButton("髖關節物件標註")
        self.detect_button.clicked.connect(self.detect_objects)
        layout.addWidget(self.detect_button)

        self.setLayout(layout)

    def load_images(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if folder:
            self.image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
            self.image_selector.clear()
            self.image_selector.addItems([os.path.basename(f) for f in self.image_files])
            self.display_image(0)  # 顯示第一張圖

    def on_image_selected(self, index):
        self.current_image_idx = index
        self.display_image(index)

    def display_image(self, index):
        if 0 <= index < len(self.image_files):
            image_path = self.image_files[index]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 獲取原始圖片尺寸
            h, w, c = image.shape

            # 縮放圖片至1/10
            new_h, new_w = int(h / 5), int(w / 5)
            image_resized = cv2.resize(image, (new_w, new_h))

            # 獲取當前圖片名稱
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            # ----------------------
            # 畫關鍵點標註
            csv_file = os.path.join('annotations', f"{image_name}.csv")
            if os.path.exists(csv_file):
                with open(csv_file, "r") as f:
                    line = f.readline().strip()
                matches = re.findall(r"\((\d+),\s*(\d+)\)", line)
                keypoints = [(int(x), int(y)) for x, y in matches]

                scale_x = new_w / w
                scale_y = new_h / h

                for idx, (x, y) in enumerate(keypoints, start=1):
                    x_scaled, y_scaled = int(x * scale_x), int(y * scale_y)
                    cv2.circle(image_resized, (x_scaled, y_scaled), 1, (255, 0, 0), -1)
                    cv2.putText(image_resized, str(idx), (x_scaled + 5, y_scaled - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # ----------------------
            # 畫 Hip 物件框
            json_file = os.path.join('detections', f"{image_name}.json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data = json.load(f)

                for obj in data.get("objects", []):
                    label = obj.get("label")
                    points = obj.get("points", [])
                    if len(points) == 2:
                        (x1, y1), (x2, y2) = points
                        # 縮放
                        x1_scaled, y1_scaled = int(x1 * scale_x), int(y1 * scale_y)
                        x2_scaled, y2_scaled = int(x2 * scale_x), int(y2 * scale_y)

                        color = (0, 255, 0)  # 預設綠
                        if label == "LeftHip":
                            color = (255, 0, 0)  # 左髖用紅色
                        elif label == "RightHip":
                            color = (0, 0, 255)  # 右髖用藍色

                        cv2.rectangle(image_resized, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
                        cv2.putText(image_resized, label, (x1_scaled, y1_scaled - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # ----------------------
            # 顯示圖片
            bytes_per_line = 3 * new_w
            qimage = QImage(image_resized.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(pixmap)


    def next_image(self):
        if self.current_image_idx < len(self.image_files) - 1:
            self.current_image_idx += 1
            self.image_selector.setCurrentIndex(self.current_image_idx)
            self.display_image(self.current_image_idx)
        else:
            self.show_message("提示", "已經是最後一張圖片了")

    def prev_image(self):
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.image_selector.setCurrentIndex(self.current_image_idx)
            self.display_image(self.current_image_idx)
        else:
            self.show_message("提示", "已經是第一張圖片了")

    def clear_annotations(self):
        """清除當前圖片的標注並刪除CSV檔案"""
        self.keypoints = []  # 清空標註點
        image_name = os.path.splitext(os.path.basename(self.image_files[self.current_image_idx]))[0]
        csv_file = os.path.join('annotations', f"{image_name}.csv")

        if os.path.exists(csv_file):
            os.remove(csv_file)  # 刪除對應的CSV檔案
            self.show_message("提示", f"已刪除標註檔案: {csv_file}")
        else:
            self.show_message("提示", f"標註檔案不存在: {csv_file}")

        self.display_image(self.current_image_idx)  # 刷新顯示圖片

    def show_annotations(self):
        """彈出新視窗顯示帶有標註點和Hip物件框的圖片"""
        image_name = os.path.splitext(os.path.basename(self.image_files[self.current_image_idx]))[0]
        csv_file = os.path.join('annotations', f"{image_name}.csv")

        # 讀取原始圖片
        image_path = self.image_files[self.current_image_idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_h, original_w, c = image.shape

        # 縮放圖片至 800x600
        display_w, display_h = 800, 600
        image_resized = cv2.resize(image, (display_w, display_h))

        # 計算縮放比例
        scale_x = display_w / original_w
        scale_y = display_h / original_h

        # ------------------------
        # 繪製關鍵點
        if os.path.exists(csv_file):
            with open(csv_file, "r") as f:
                line = f.readline().strip()
            matches = re.findall(r"\((\d+),\s*(\d+)\)", line)
            self.keypoints = [(int(x), int(y)) for x, y in matches]

            for idx, (x, y) in enumerate(self.keypoints, start=1):
                x_scaled, y_scaled = int(x * scale_x), int(y * scale_y)
                cv2.circle(image_resized, (x_scaled, y_scaled), 3, (255, 0, 0), -1)  # 紅色點
                cv2.putText(
                    image_resized, 
                    str(idx), 
                    (x_scaled + 5, y_scaled - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    1, 
                    cv2.LINE_AA
                )
        else:
            self.show_message("提示", f"未找到標註文件: {csv_file}")

        # ------------------------
        # 繪製Hip物件框
        json_file = os.path.join('detections', f"{image_name}.json")
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)

            for obj in data.get("objects", []):
                label = obj.get("label")
                points = obj.get("points", [])
                if len(points) == 2:
                    (x1, y1), (x2, y2) = points
                    x1_scaled, y1_scaled = int(x1 * scale_x), int(y1 * scale_y)
                    x2_scaled, y2_scaled = int(x2 * scale_x), int(y2 * scale_y)

                    color = (0, 255, 0)  # 預設綠
                    if label == "LeftHip":
                        color = (255, 0, 0)  # 紅色
                    elif label == "RightHip":
                        color = (0, 0, 255)  # 藍色

                    cv2.rectangle(image_resized, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
                    cv2.putText(
                        image_resized,
                        label,
                        (x1_scaled, y1_scaled - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA
                    )

        # ------------------------
        # 在新視窗顯示圖片
        dialog = QDialog(self)
        dialog.setWindowTitle(f"標註 - {image_name}")
        dialog.setGeometry(100, 100, display_w, display_h)

        image_label = QLabel(dialog)
        image_label.setGeometry(0, 0, display_w, display_h)

        bytes_per_line = 3 * display_w
        qimage = QImage(image_resized.data, display_w, display_h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        image_label.setPixmap(pixmap)

        dialog.exec_()

    def label_image(self):
        self.label_window = LabelWindow(
            self.image_files[self.current_image_idx], 
            self.save_keypoints
        )
        self.label_window.show()
    
    def save_keypoints(self, keypoints):
        # 獲取圖片名稱
        image_name = os.path.splitext(os.path.basename(self.image_files[self.current_image_idx]))[0]
        # CSV 文件路徑
        csv_file = os.path.join('annotations', f"{image_name}.csv")
        
        # 確保保存標注的文件夾存在
        if not os.path.exists('annotations'):
            os.makedirs('annotations')
        
        # 將標注點轉換為字符串
        keypoints_str = ",".join([f'"({x}, {y})"' for x, y in keypoints])
        
        # 將標注點保存到CSV文件
        with open(csv_file, "w") as f:
            f.write(keypoints_str + "\n")
        
        self.show_message("提示", f"已儲存標註到: {csv_file}")
        self.display_image(self.current_image_idx)  # 刷新顯示圖片
    
    def detect_objects(self):
        self.detect_window = ObjectDetectionWindow(
            self.image_files[self.current_image_idx], 
            self.save_object_detections
        )
        self.detect_window.show()
    
    def save_object_detections(self, keypoints):
        if not os.path.exists('detections'):
            os.makedirs('detections')

        image_name = os.path.basename(self.image_files[self.current_image_idx])
        json_file = os.path.join('detections', f"{os.path.splitext(image_name)[0]}.json")

        if len(keypoints) < 4:
            self.show_message("警告", "必須標註四個點才能儲存物件偵測結果")
            return

        data = {
            "image": image_name,
            "objects": [
                {
                    "label": "LeftHip",
                    "points": [
                        [keypoints[0][0], keypoints[0][1]],
                        [keypoints[1][0], keypoints[1][1]]
                    ]
                },
                {
                    "label": "RightHip",
                    "points": [
                        [keypoints[2][0], keypoints[2][1]],
                        [keypoints[3][0], keypoints[3][1]]
                    ]
                }
            ]
        }

        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)

        self.show_message("提示", f"已儲存物件偵測到: {json_file}")
        self.display_image(self.current_image_idx)  # 刷新顯示圖片

    def show_message(self, title, message):
        """顯示通知視窗"""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

class LabelWindow(QWidget):
    def __init__(self, image_path, save_callback):
        super().__init__()
        self.image_path = image_path
        self.save_callback = save_callback
        self.keypoints = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("標註圖片")
        self.setGeometry(100, 100, 800, 600)
        
        # 顯示圖片，縮放圖片至 800x600
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.original_h, self.original_w, self.c = self.image.shape  # 原圖像的大小
        self.image = cv2.resize(self.image, (800, 600))  # 縮放到 800x600
        self.h, self.w, self.c = self.image.shape
        self.image_copy = self.image.copy()

        # 設置畫布
        self.canvas = QLabel(self)
        self.canvas.setGeometry(0, 0, self.w, self.h)
        self.update_image()

        # 顯示圖片名稱
        self.scale_x = self.original_w / self.w
        self.scale_y = self.original_h / self.h

    def update_image(self):
        qimage = QImage(self.image_copy.data, self.w, self.h, self.w * self.c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.canvas.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 左鍵點擊，添加標註點
            x, y = event.x(), event.y()
            if 0 <= x < self.w and 0 <= y < self.h:
                # 反向縮放回原圖大小
                original_x = int(x * self.scale_x)
                original_y = int(y * self.scale_y)
                self.keypoints.append((original_x, original_y))
                # 在縮放後的圖片上顯示標註點及其順序
                self.image_copy = self.image.copy()
                for idx, (px, py) in enumerate(self.keypoints, start=1):
                    # 縮放點回顯示大小
                    px_scaled = int(px / self.scale_x)
                    py_scaled = int(py / self.scale_y)
                    cv2.circle(self.image_copy, (px_scaled, py_scaled), 3, (255, 0, 0), -1)  # 紅色標註點
                    cv2.putText(
                        self.image_copy,
                        str(idx),
                        (px_scaled + 5, py_scaled - 5),  # 在點的右上方顯示序號
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),  # 綠色文字
                        1,
                        cv2.LINE_AA
                    )
                self.update_image()

        elif event.button() == Qt.RightButton:
            # 右鍵取消最近標註點
            if self.keypoints:
                self.keypoints.pop()
                self.image_copy = self.image.copy()
                for idx, (px, py) in enumerate(self.keypoints, start=1):
                    # 縮放點回顯示大小
                    px_scaled = int(px / self.scale_x)
                    py_scaled = int(py / self.scale_y)
                    cv2.circle(self.image_copy, (px_scaled, py_scaled), 3, (255, 0, 0), -1)  # 紅色標註點
                    cv2.putText(
                        self.image_copy,
                        str(idx),
                        (px_scaled + 5, py_scaled - 5),  # 在點的右上方顯示序號
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),  # 綠色文字
                        1,
                        cv2.LINE_AA
                    )
                self.update_image()

    def closeEvent(self, event):
        self.save_callback(self.keypoints)  # 儲存標註結果
        event.accept()

class ObjectDetectionWindow(QWidget):
    def __init__(self, image_path, save_object_callback):
        super().__init__()
        self.image_path = image_path
        self.save_object_callback = save_object_callback
        self.keypoints = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("物件偵測 - 點選4個點")
        self.setGeometry(100, 100, 800, 600)
        
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.original_h, self.original_w, self.c = self.image.shape
        self.image = cv2.resize(self.image, (800, 600))
        self.h, self.w, self.c = self.image.shape
        self.image_copy = self.image.copy()

        self.canvas = QLabel(self)
        self.canvas.setGeometry(0, 0, self.w, self.h)
        self.update_image()

        self.scale_x = self.original_w / self.w
        self.scale_y = self.original_h / self.h

    def update_image(self):
        qimage = QImage(self.image_copy.data, self.w, self.h, self.w * self.c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.canvas.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x, y = event.x(), event.y()
            if 0 <= x < self.w and 0 <= y < self.h:
                original_x = int(x * self.scale_x)
                original_y = int(y * self.scale_y)
                self.keypoints.append((original_x, original_y))
                self.redraw()

        elif event.button() == Qt.RightButton:
            if self.keypoints:
                self.keypoints.pop()
                self.redraw()

    def redraw(self):
        self.image_copy = self.image.copy()
        # 畫矩形或點
        if len(self.keypoints) >= 2:
            pt1 = self.keypoints[0]
            pt2 = self.keypoints[1]
            # 縮放回顯示座標
            pt1_disp = (int(pt1[0] / self.scale_x), int(pt1[1] / self.scale_y))
            pt2_disp = (int(pt2[0] / self.scale_x), int(pt2[1] / self.scale_y))
            cv2.rectangle(self.image_copy, pt1_disp, pt2_disp, (255, 0, 0), 2)  # 紅色

        if len(self.keypoints) >= 4:
            pt3 = self.keypoints[2]
            pt4 = self.keypoints[3]
            pt3_disp = (int(pt3[0] / self.scale_x), int(pt3[1] / self.scale_y))
            pt4_disp = (int(pt4[0] / self.scale_x), int(pt4[1] / self.scale_y))
            cv2.rectangle(self.image_copy, pt3_disp, pt4_disp, (0, 0, 255), 2)  # 藍色

        # 繪製所有點
        for idx, (px, py) in enumerate(self.keypoints, start=1):
            px_disp = int(px / self.scale_x)
            py_disp = int(py / self.scale_y)
            cv2.circle(self.image_copy, (px_disp, py_disp), 3, (0, 255, 0), -1)
            cv2.putText(self.image_copy, str(idx), (px_disp + 5, py_disp - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        self.update_image()

    def closeEvent(self, event):
        self.save_object_callback(self.keypoints)
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = KeypointAnnotationApp()
    window.show()
    app.exec_()
