import sys
import os
import cv2
import time
import numpy as np
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QTextEdit,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRunnable, QThreadPool, pyqtSlot, QObject
from ultralytics import YOLO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, model_path="best.pt"):
        super().__init__()
        self.model = YOLO(model_path).to("cpu")
        self._run_flag = True
        self.current_face = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.skip_frames = 5

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("摄像头打开失败")
            return

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue

            self.frame_count += 1
            if self.frame_count % self.skip_frames == 0:
                results = self.model(frame)
                if results and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        face = frame[y1:y2, x1:x2]
                        with self.lock:
                            self.current_face = face
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        break

            self.change_pixmap_signal.emit(frame)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def get_face(self):
        with self.lock:
            return self.current_face.copy() if self.current_face is not None else None


class DoubaoBot:
    def __init__(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 20)
        self.driver.get("https://doubao.com")
        self.message_history = []

    def login(self):
        try:
            login_btn = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//button[contains(text(),'登录')]"))
            )
            login_btn.click()
            input("请手动登录后按 Enter 继续...")
        except:
            pass

    def send_message(self, message):
        try:
            input_box = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "textarea[placeholder*='输入']"))
            )
            input_box.clear()
            input_box.send_keys(message)

            send_button = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "#flow-end-msg-send svg"))
            )

            prev_replies = self._get_all_reply_texts()
            send_button.click()
            return self.wait_for_new_reply(prev_replies)
        except Exception as e:
            return f"发送消息失败: {e}"

    def upload_image(self, path):
        try:
            file_input = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@type='file']"))
            )
            file_input.send_keys(path)
            time.sleep(2)
            return True
        except Exception as e:
            return False

    def _get_all_reply_texts(self):
        try:
            reply_containers = self.driver.find_elements(
                By.CSS_SELECTOR,
                'div[data-testid="receive_message"] [data-testid="message_text_content"]'
            )

            return [el.text.strip() for el in reply_containers if el.text.strip()]
        except Exception as e:
            print(f"❌ 获取所有回复失败: {e}")
            return []

    def wait_for_new_reply(self, prev_replies, timeout=60, stable_time=3):
        start_time = time.time()
        last_reply = ""
        last_change_time = time.time()

        while time.time() - start_time < timeout:
            current_replies = self._get_all_reply_texts()

            # 只关心最新回复
            if current_replies:
                new_reply = current_replies[-1]

                # 如果最新回复和上次相同，检测是否稳定
                if new_reply == last_reply:
                    if time.time() - last_change_time >= stable_time:
                        # 判断是否已在历史回复里，避免重复返回
                        if not hasattr(self, 'message_history'):
                            self.message_history = []
                        if self.message_history and self.message_history[-1] == new_reply:
                            return "⚠️ 未检测到新回复"
                        self.message_history.append(new_reply)
                        return new_reply
                else:
                    last_reply = new_reply
                    last_change_time = time.time()

            time.sleep(1)
        return "等待回复超时"

    def close(self):
        self.driver.quit()


class WorkerSignals(QObject):
    finished = pyqtSignal(str)


class SendWorker(QRunnable):
    def __init__(self, bot, user_text, face, callback):
        super().__init__()
        self.bot = bot
        self.user_text = user_text
        self.face = face
        self.signals = WorkerSignals()
        self.signals.finished.connect(callback)

    @pyqtSlot()
    def run(self):
        filename = None
        try:
            if self.face is not None:
                filename = "temp_face.jpg"
                cv2.imwrite(filename, self.face)
                self.bot.upload_image(os.path.abspath(filename))
                full_prompt = (
                    f"你是一个心理医生，请分析用户的面部表情，以及以下消息内容来提供心理支持，你的回复应包含对用户表情的分析，同时需要回复的语气应该设立在面对面交流的场景。用户说: {self.user_text}"
                )
            else:
                full_prompt = self.user_text

            reply = self.bot.send_message(full_prompt)
        except Exception as e:
            reply = f"处理失败: {e}"
        finally:
            if filename and os.path.exists(filename):
                os.remove(filename)
            self.signals.finished.emit(reply)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("情绪识别与心理对话")
        self.setGeometry(100, 100, 1000, 700)
        self.thread_pool = QThreadPool()

        self.bot = DoubaoBot()
        self.bot.login()

        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

        self.init_ui()

        self.setStyleSheet("""
            QWidget {
                font-family: '微软雅黑';
                font-size: 18px;
            }
            QMainWindow {
                background-color: #f7f9fc;
            }
        """)

    def init_ui(self):
        layout = QVBoxLayout()

        # 摄像头区域
        video_group = QGroupBox("摄像头画面")
        video_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 16px; }")
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        video_layout.addWidget(self.video_label)
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # 输入区域
        input_group = QGroupBox("输入消息")
        input_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 16px; }")
        input_layout = QHBoxLayout()
        self.input_text = QTextEdit()
        self.input_text.setFixedHeight(120)
        self.input_text.setPlaceholderText("请输入您的问题或想法...")
        input_layout.addWidget(self.input_text)

        self.send_button = QPushButton("发送")
        self.send_button.setFixedHeight(120)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.send_button.clicked.connect(self.send)
        input_layout.addWidget(self.send_button)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # 对话区域
        response_group = QGroupBox("对话")
        response_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 16px; }")
        response_layout = QVBoxLayout()
        self.response_box = QTextEdit()
        self.response_box.setReadOnly(True)
        self.response_box.setStyleSheet("""
            QTextEdit {
                background-color: #fdfdfd;
                border-radius: 8px;
                border: 1px solid #ccc;
                padding: 30px;
            }
        """)
        response_layout.addWidget(self.response_box)
        response_group.setLayout(response_layout)
        layout.addWidget(response_group)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def send(self):
        user_text = self.input_text.toPlainText().strip()
        if not user_text:
            QMessageBox.warning(self, "警告", "请输入消息")
            return

        face = self.video_thread.get_face()
        self.send_button.setEnabled(False)
        self.response_box.append(f"你: {user_text}")

        worker = SendWorker(self.bot, user_text, face, self.display_reply)
        self.thread_pool.start(worker)

    def display_reply(self, reply):
        self.response_box.append(f"模型: {reply}\n")
        self.input_text.clear()
        self.send_button.setEnabled(True)

    def closeEvent(self, event):
        self.video_thread.stop()
        self.bot.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
