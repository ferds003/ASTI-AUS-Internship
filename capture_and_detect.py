import sys
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from joblib import load
from datetime import datetime

# define constants
WINDOW_SIZE = (64, 64)
STEP_SIZE = 16
OUTPUT_DIR = "detections"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_MAPPING = {
    'chair': 1, 'computer_keyboard': 2, 'computer_mouse': 3, 'door': 4,
    'door_handle': 5, 'light_switch': 6, 'monitor': 7, 'person': 8, 'table': 9
}

# load chosen model
model = load("svm_model.joblib")

# sliding window function
def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# object detection function, how the model will interact with the captured image
def detect_objects(image_gray, model):
    detections = []
    for (x, y, window) in sliding_window(image_gray, STEP_SIZE, WINDOW_SIZE):
        if window.shape[:2] != WINDOW_SIZE:
            continue
        resized_window = cv2.resize(window, (300, 300)).flatten().reshape(1, -1)
        predicted_label = model.predict(resized_window)[0]
        numeric_label = LABEL_MAPPING.get(predicted_label, -1)
        if numeric_label > 0:
            detections.append((x, y, x + WINDOW_SIZE[0], y + WINDOW_SIZE[1], numeric_label))
    return detections

# non-maximum suppression to remove overlapping bounding boxes and ensure each object is only detected once
def non_max_suppression_fast(boxes, probs=None, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype="float")
    pick = []

    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(probs)[::-1] if probs is not None else np.argsort(y2)

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[idxs[1:]]

        idxs = idxs[np.where(overlap <= overlapThresh)[0] + 1]

    return boxes[pick].astype("int")

# main GUI using PyQt
class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Capture & Save Detection")
        self.capture = cv2.VideoCapture(1)  # Adjust camera index if needed
        self.running = False  # Flag to toggle detection

        # UI setup
        self.image_label = QLabel()
        self.detect_button = QPushButton("Capture & Detect")
        self.detect_button.clicked.connect(self.capture_and_detect)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.detect_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def capture_and_detect(self):
        ret, frame = self.capture.read()
        if not ret:
            QMessageBox.critical(self, "Error", "Failed to capture image.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detect_objects(gray, model)

        boxes = [d[:4] for d in detections]
        scores = [1] * len(detections)
        boxes_nms = non_max_suppression_fast(boxes, scores, overlapThresh=0.5)

        # Draw detections
        for i, (x1, y1, x2, y2) in enumerate(boxes_nms):
            label = detections[i][4]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"detection_{timestamp}.jpg")
        cv2.imwrite(filename, frame)

        # Show preview in GUI
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

        QMessageBox.information(self, "Saved", f"Detection saved to:\n{filename}")

    def closeEvent(self, event):
        self.capture.release()
        event.accept()

# === Run App ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec_())
