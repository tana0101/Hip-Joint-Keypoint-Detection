import os
import cv2
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QComboBox, QLabel, QDialog, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import re

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
        self.label_button = QPushButton("標註圖片")
        self.label_button.clicked.connect(self.label_image)
        layout.addWidget(self.label_button)

        # 展示標記點按鈕
        self.show_button = QPushButton("展示標記點")
        self.show_button.clicked.connect(self.show_annotations)
        layout.addWidget(self.show_button)

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
            new_h, new_w = int(h / 10), int(w / 10)
            image_resized = cv2.resize(image, (new_w, new_h))
    
            # 獲取當前圖片的名稱
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            csv_file = os.path.join('annotations', f"{image_name}.csv")
    
            # 如果標註文件存在，讀取並繪製關鍵點
            if os.path.exists(csv_file):
                with open(csv_file, "r") as f:
                    line = f.readline().strip()
    
                # 解析關鍵點
                matches = re.findall(r"\((\d+),\s*(\d+)\)", line)
                keypoints = [(int(x), int(y)) for x, y in matches]
    
                # 計算縮放比例
                scale_x = new_w / w
                scale_y = new_h / h
    
                # 繪製關鍵點
                for idx, (x, y) in enumerate(keypoints, start=1):
                    x_scaled, y_scaled = int(x * scale_x), int(y * scale_y)
                    cv2.circle(image_resized, (x_scaled, y_scaled), 1, (255, 0, 0), -1)  # 紅色圓點
                    cv2.putText(image_resized, str(idx), (x_scaled + 5, y_scaled - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # 添加標註順序
    
            # 設置 QLabel 顯示圖片
            bytes_per_line = 3 * new_w
            qimage = QImage(image_resized.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
    
            # 顯示圖片
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
        """彈出新視窗顯示帶有標註點的圖片"""
        image_name = os.path.splitext(os.path.basename(self.image_files[self.current_image_idx]))[0]
        csv_file = os.path.join('annotations', f"{image_name}.csv")

        if os.path.exists(csv_file):
            # 讀取CSV文件的第一行
            with open(csv_file, "r") as f:
                line = f.readline().strip()  # 讀取

            # 從CSV文件中提取標註點
            matches = re.findall(r"\((\d+),\s*(\d+)\)", line)
            self.keypoints = [(int(x), int(y)) for x, y in matches]  # 转换为整数元组

            # 读取原始图片
            image_path = self.image_files[self.current_image_idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 獲取原始圖片尺寸
            original_h, original_w, c = image.shape

            # 縮放圖片至 800x600
            display_w, display_h = 800, 600
            image_resized = cv2.resize(image, (display_w, display_h))

            # 計算縮放比例
            scale_x = display_w / original_w
            scale_y = display_h / original_h

            # 繪製標註點和標註順序
            for idx, (x, y) in enumerate(self.keypoints, start=1):
                x_scaled, y_scaled = int(x * scale_x), int(y * scale_y)
                cv2.circle(image_resized, (x_scaled, y_scaled), 3, (255, 0, 0), -1)  # 紅色點
                cv2.putText(
                    image_resized, 
                    str(idx), 
                    (x_scaled + 5, y_scaled - 5),  # 偏移一點以避免與圓點重疊
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0),  # 綠色文字
                    1, 
                    cv2.LINE_AA
                )

            # 在新窗口中顯示圖片
            dialog = QDialog(self)
            dialog.setWindowTitle(f"標註 - {image_name}")
            dialog.setGeometry(100, 100, display_w, display_h)

            # 顯示圖片
            image_label = QLabel(dialog)
            image_label.setGeometry(0, 0, display_w, display_h)

            # 設置 QLabel 顯示圖片
            bytes_per_line = 3 * display_w
            qimage = QImage(image_resized.data, display_w, display_h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            image_label.setPixmap(pixmap)

            dialog.exec_()  # 顯示視窗

        else:
            self.show_message("提示", f"未找到標註文件: {csv_file}")



    def label_image(self):
        self.label_window = LabelWindow(self.image_files[self.current_image_idx], self.save_keypoints)
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


if __name__ == "__main__":
    app = QApplication([])
    window = KeypointAnnotationApp()
    window.show()
    app.exec_()
