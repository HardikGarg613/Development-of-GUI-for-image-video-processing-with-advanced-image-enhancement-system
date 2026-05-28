import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import datetime

# === Fog Removal Function ===
def dehaze(frame):
    img = frame.astype(np.float32) / 255.0
    dark_channel = cv2.min(cv2.min(img[:, :, 0], img[:, :, 1]), img[:, :, 2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(dark_channel, kernel)
    atmospheric_light = np.max(dark_channel)
    transmission = 1 - 0.95 * dark_channel / (atmospheric_light + 1e-6)
    transmission = np.clip(transmission, 0.1, 1)
    result = np.empty_like(img)
    for c in range(3):
        result[:, :, c] = (img[:, :, c] - atmospheric_light) / transmission + atmospheric_light
    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)

# === Contrast Enhancement Functions ===
def histogram_equalization(frame):
    """Applies Histogram Equalization to a grayscale or color image."""
    ycrcb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

def clahe_enhancement(frame):
    """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
    return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

def gamma_correction(frame, gamma=1.0):
    """Applies Gamma Correction to adjust brightness."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

# === Image Filtering Functions ===
def apply_no_effect(frame):
    return frame

def apply_dice_effect(frame):
    height, width = frame.shape[:2]
    w, h = (width // 10, height // 10)
    temp = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

def apply_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_mauve_filter(frame):
    mauve_tint = np.array([128, 0, 128], dtype=np.uint8)
    return cv2.add(frame, mauve_tint)

# --- NEW: Additional Image Filtering Functions ---
def apply_hulk_filter(frame):
    # A simple way to create a "Hulk" effect is to shift the color channels to a green tint.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Shift hue towards green (Hue values: 0=red, 60=yellow, 120=green, 180=full circle)
    h_hulk = np.full_like(h, 60)
    final_hsv = cv2.merge([h_hulk, s, v])
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def apply_noir_blanc_filter(frame):
    # This is a simple grayscale filter, similar to black and white photography.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def apply_neon_glow_filter(frame):
    # This effect can be created by inverting the image and then blurring.
    inverted = cv2.bitwise_not(frame)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    return cv2.bitwise_not(blurred)

def apply_satin_glow_filter(frame):
    # A soft blur and color adjustment can give a "satin" look.
    blurred = cv2.bilateralFilter(frame, 9, 75, 75)
    # Increase brightness and saturation slightly
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s_satin = cv2.add(s, 20)
    v_satin = cv2.add(v, 20)
    final_hsv = cv2.merge([h, s_satin, v_satin])
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def apply_golden_glow_filter(frame):
    # A golden effect can be achieved by increasing yellow/red tones.
    # We can add a golden tint by blending.
    rows, cols, _ = frame.shape
    golden_tint = np.zeros(frame.shape, dtype=np.uint8)
    golden_tint[:, :, 2] = 255 # Red channel
    golden_tint[:, :, 1] = 215 # Green channel (less than red)
    # Blend the original frame with the golden tint
    return cv2.addWeighted(frame, 0.7, golden_tint, 0.3, 0)
# ----------------------------------

# === Video Thread Classes (Emit Raw Frames Only) ===
class VideoFileThread(QtCore.QThread):
    update_frames = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = False
        self.paused = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        self.running = True
        while self.running:
            if self.paused:
                self.msleep(100)
                continue
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self.update_frames.emit(frame, frame)
            self.msleep(33)
        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def toggle_pause(self):
        self.paused = not self.paused

class VideoCameraThread(QtCore.QThread):
    update_frames = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        self.paused = False

    def run(self):
        cap = cv2.VideoCapture(0)
        self.running = True
        while self.running:
            if self.paused:
                self.msleep(100)
                continue
            ret, frame = cap.read()
            if not ret:
                continue
            self.update_frames.emit(frame, frame)
            self.msleep(33)
        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def toggle_pause(self):
        self.paused = not self.paused

# === New Image Fusion Dialog ===
class ImageFusionDialog(QtWidgets.QDialog):
    fusion_done = QtCore.pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Fusion")
        self.setFixedSize(400, 200)
        self.img1 = None
        self.img2 = None
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()
        self.img1_btn = QtWidgets.QPushButton("Upload Image 1")
        self.img1_btn.clicked.connect(self.load_img1)
        layout.addWidget(self.img1_btn)
        self.img2_btn = QtWidgets.QPushButton("Upload Image 2")
        self.img2_btn.clicked.connect(self.load_img2)
        layout.addWidget(self.img2_btn)
        self.fuse_btn = QtWidgets.QPushButton("Fuse Images")
        self.fuse_btn.clicked.connect(self.fuse_images)
        self.fuse_btn.setEnabled(False)
        layout.addWidget(self.fuse_btn)
        self.setLayout(layout)

    def load_img1(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image 1", "", "Image Files (*.png *.jpg *.jpeg)")
        if path:
            self.img1 = cv2.imread(path)
            if self.img1 is None:
                QtWidgets.QMessageBox.critical(self, "Error", "Failed to load Image 1.")
                return
            self.check_enable_fuse()

    def load_img2(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image 2", "", "Image Files (*.png *.jpg *.jpeg)")
        if path:
            self.img2 = cv2.imread(path)
            if self.img2 is None:
                QtWidgets.QMessageBox.critical(self, "Error", "Failed to load Image 2.")
                return
            self.check_enable_fuse()

    def check_enable_fuse(self):
        if self.img1 is not None and self.img2 is not None:
            self.fuse_btn.setEnabled(True)

    def fuse_images(self):
        height = min(self.img1.shape[0], self.img2.shape[0])
        width = min(self.img1.shape[1], self.img2.shape[1])
        img1_resized = cv2.resize(self.img1, (width, height))
        img2_resized = cv2.resize(self.img2, (width, height))
        fused = cv2.addWeighted(img1_resized, 0.5, img2_resized, 0.5, 0)
        self.fusion_done.emit(img1_resized, img2_resized, fused)
        self.accept()

# === Snapshot & Filter Dialog ===
class SnapshotFilterDialog(QtWidgets.QDialog):
    def __init__(self, frame):
        super().__init__()
        self.setWindowTitle("Apply Filter & Save")
        self.setFixedSize(1200, 600)  # Increased size to accommodate more buttons
        self.original_frame = frame
        self.filtered_frame = frame
        self.current_filter_func = apply_no_effect
        self.initUI()

    def initUI(self):
        main_layout = QtWidgets.QVBoxLayout()
        image_layout = QtWidgets.QHBoxLayout()

        self.original_label = QtWidgets.QLabel()
        self.original_label.setFixedSize(560, 420)
        self.original_label.setAlignment(QtCore.Qt.AlignCenter)
        self.original_label.setText("Original")
        image_layout.addWidget(self.original_label)

        self.filtered_label = QtWidgets.QLabel()
        self.filtered_label.setFixedSize(560, 420)
        self.filtered_label.setAlignment(QtCore.Qt.AlignCenter)
        self.filtered_label.setText("Filtered")
        image_layout.addWidget(self.filtered_label)

        main_layout.addLayout(image_layout)

        # Filter buttons with new ones
        filter_layout = QtWidgets.QGridLayout()
        self.no_effect_btn = QtWidgets.QPushButton("No Effect")
        self.dice_btn = QtWidgets.QPushButton("Dice")
        self.edge_btn = QtWidgets.QPushButton("Edge")
        self.mauve_btn = QtWidgets.QPushButton("Mauve")
        self.hulk_btn = QtWidgets.QPushButton("Hulk")
        self.noir_blanc_btn = QtWidgets.QPushButton("Noir/Blanc")
        self.neon_btn = QtWidgets.QPushButton("Neon Glow")
        self.satin_btn = QtWidgets.QPushButton("Satin Glow")
        self.golden_btn = QtWidgets.QPushButton("Golden Glow")

        filter_layout.addWidget(self.no_effect_btn, 0, 0)
        filter_layout.addWidget(self.dice_btn, 0, 1)
        filter_layout.addWidget(self.edge_btn, 0, 2)
        filter_layout.addWidget(self.mauve_btn, 1, 0)
        filter_layout.addWidget(self.hulk_btn, 1, 1)
        filter_layout.addWidget(self.noir_blanc_btn, 1, 2)
        filter_layout.addWidget(self.neon_btn, 2, 0)
        filter_layout.addWidget(self.satin_btn, 2, 1)
        filter_layout.addWidget(self.golden_btn, 2, 2)

        self.no_effect_btn.clicked.connect(lambda: self.apply_filter_and_update(apply_no_effect))
        self.dice_btn.clicked.connect(lambda: self.apply_filter_and_update(apply_dice_effect))
        self.edge_btn.clicked.connect(lambda: self.apply_filter_and_update(apply_edge_detection))
        self.mauve_btn.clicked.connect(lambda: self.apply_filter_and_update(apply_mauve_filter))
        self.hulk_btn.clicked.connect(lambda: self.apply_filter_and_update(apply_hulk_filter))
        self.noir_blanc_btn.clicked.connect(lambda: self.apply_filter_and_update(apply_noir_blanc_filter))
        self.neon_btn.clicked.connect(lambda: self.apply_filter_and_update(apply_neon_glow_filter))
        self.satin_btn.clicked.connect(lambda: self.apply_filter_and_update(apply_satin_glow_filter))
        self.golden_btn.clicked.connect(lambda: self.apply_filter_and_update(apply_golden_glow_filter))

        main_layout.addLayout(filter_layout)

        save_btn = QtWidgets.QPushButton("Save Filtered Image")
        save_btn.clicked.connect(self.save_filtered_image)
        main_layout.addWidget(save_btn)

        self.setLayout(main_layout)
        self.display_frames()

    def display_frames(self):
        original_pixmap = self.to_pixmap(self.original_frame, self.original_label.size())
        self.original_label.setPixmap(original_pixmap)

        filtered_pixmap = self.to_pixmap(self.filtered_frame, self.filtered_label.size())
        self.filtered_label.setPixmap(filtered_pixmap)

    def apply_filter_and_update(self, filter_func):
        self.current_filter_func = filter_func
        self.filtered_frame = self.current_filter_func(self.original_frame.copy())
        self.display_frames()

    def save_filtered_image(self):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filter_name = self.current_filter_func.__name__.replace('apply_', '')
        filename = f"snapshot_{filter_name}_{now}.jpg"
        cv2.imwrite(filename, self.filtered_frame)
        QtWidgets.QMessageBox.information(self, "Saved", f"Filtered snapshot saved as {filename}")
        self.accept()

    def to_pixmap(self, frame, size):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimg).scaled(size, QtCore.Qt.KeepAspectRatio)

# === GUI Class ===
class SimpleVideoGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Contrast & Brightness Player")
        self.setGeometry(100, 100, 1200, 600)
        self.video_thread = None
        self.camera_thread = None
        self.recording = False
        self.video_writer = None
        self.last_output_frame = None
        self.fog_removal_enabled = False
        self.initUI()

    def initUI(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        left_layout = QtWidgets.QVBoxLayout()
        self.logo_label = QtWidgets.QLabel()
        self.logo_label.setFixedSize(150, 150)
        pixmap = QtGui.QPixmap("logo.png")
        if not pixmap.isNull():
            self.logo_label.setPixmap(pixmap.scaled(self.logo_label.size(), QtCore.Qt.KeepAspectRatio))
        self.logo_label.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(self.logo_label, alignment=QtCore.Qt.AlignCenter)
        videos_layout = QtWidgets.QHBoxLayout()
        self.input_label = QtWidgets.QLabel()
        self.input_label.setFixedSize(480, 360)
        self.input_label.setObjectName("input_label")
        videos_layout.addWidget(self.input_label)
        self.output_label = QtWidgets.QLabel()
        self.output_label.setFixedSize(480, 360)
        self.output_label.setObjectName("output_label")
        videos_layout.addWidget(self.output_label)
        left_layout.addLayout(videos_layout)
        label_layout = QtWidgets.QHBoxLayout()
        label_layout.addWidget(QtWidgets.QLabel("Input Video", alignment=QtCore.Qt.AlignCenter))
        label_layout.addWidget(QtWidgets.QLabel("Output Video", alignment=QtCore.Qt.AlignCenter))
        left_layout.addLayout(label_layout)
        main_layout.addLayout(left_layout)
        side_panel = QtWidgets.QVBoxLayout()
        upload_btn = QtWidgets.QPushButton("📁 Upload Video")
        upload_btn.clicked.connect(self.upload_video)
        side_panel.addWidget(upload_btn)
        live_btn = QtWidgets.QPushButton("📷 Live Camera")
        live_btn.clicked.connect(self.start_live_camera)
        side_panel.addWidget(live_btn)
        play_pause_btn = QtWidgets.QPushButton("⏸️ Play / Pause")
        play_pause_btn.clicked.connect(self.toggle_play_pause)
        side_panel.addWidget(play_pause_btn)
        
        snapshot_btn = QtWidgets.QPushButton("📸 Snapshot & Filter")
        snapshot_btn.clicked.connect(self.open_snapshot_dialog)
        side_panel.addWidget(snapshot_btn)
        
        self.record_btn = QtWidgets.QPushButton("🔴 Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        side_panel.addWidget(self.record_btn)
        fusion_btn = QtWidgets.QPushButton("🧪 Image Fusion")
        fusion_btn.clicked.connect(self.image_fusion)
        side_panel.addWidget(fusion_btn)
        self.fog_removal_btn = QtWidgets.QPushButton("🌫️ Toggle Fog Removal")
        self.fog_removal_btn.setCheckable(True)
        self.fog_removal_btn.clicked.connect(self.toggle_fog_removal)
        side_panel.addWidget(self.fog_removal_btn)
        side_panel.addWidget(QtWidgets.QLabel("Contrast Enhancement Technique"))
        self.contrast_group = QtWidgets.QButtonGroup(self)
        self.he_radio = QtWidgets.QRadioButton("Histogram Equalization (HE)")
        self.clahe_radio = QtWidgets.QRadioButton("CLAHE")
        self.gamma_radio = QtWidgets.QRadioButton("Gamma Correction")
        self.none_radio = QtWidgets.QRadioButton("None")
        self.none_radio.setChecked(True)
        self.contrast_group.addButton(self.he_radio)
        self.contrast_group.addButton(self.clahe_radio)
        self.contrast_group.addButton(self.gamma_radio)
        self.contrast_group.addButton(self.none_radio)
        side_panel.addWidget(self.he_radio)
        side_panel.addWidget(self.clahe_radio)
        side_panel.addWidget(self.gamma_radio)
        side_panel.addWidget(self.none_radio)
        side_panel.addWidget(QtWidgets.QLabel("Gamma Value"))
        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setRange(1, 40)
        self.gamma_slider.setValue(10)
        side_panel.addWidget(self.gamma_slider)
        side_panel.addWidget(QtWidgets.QLabel("Contrast"))
        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        side_panel.addWidget(self.contrast_slider)
        side_panel.addWidget(QtWidgets.QLabel("Brightness"))
        self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brightness_slider.setRange(0, 200)
        self.brightness_slider.setValue(100)
        side_panel.addWidget(self.brightness_slider)
        side_panel.addStretch()
        main_layout.addLayout(side_panel)

    def toggle_fog_removal(self, checked):
        self.fog_removal_enabled = checked
        self.fog_removal_btn.setText("🌫️ Fog Removal ON" if checked else "🌫️ Fog Removal OFF")

    def upload_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.avi *.mp4 *.mov)")
        if path:
            self.stop_threads()
            self.video_thread = VideoFileThread(path)
            self.video_thread.update_frames.connect(self.display_video_frames)
            self.video_thread.start()

    def start_live_camera(self):
        self.stop_threads()
        self.camera_thread = VideoCameraThread()
        self.camera_thread.update_frames.connect(self.display_video_frames)
        self.camera_thread.start()

    def toggle_play_pause(self):
        if self.video_thread:
            self.video_thread.toggle_pause()
        elif self.camera_thread:
            self.camera_thread.toggle_pause()
        
    def open_snapshot_dialog(self):
        if self.last_output_frame is not None:
            dialog = SnapshotFilterDialog(self.last_output_frame)
            dialog.exec_()
        else:
            QtWidgets.QMessageBox.warning(self, "No Frame", "No frame to capture yet.")

    def toggle_recording(self):
        if not self.recording:
            if self.last_output_frame is None:
                QtWidgets.QMessageBox.warning(self, "No Frame", "No frame available to start recording.")
                return
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{now}.avi"
            height, width, _ = self.last_output_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
            if not self.video_writer.isOpened():
                QtWidgets.QMessageBox.critical(self, "Error", "Could not start video recording.")
                return
            self.recording = True
            self.record_btn.setText("⏹️ Stop Recording")
        else:
            self.recording = False
            self.record_btn.setText("🔴 Start Recording")
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            QtWidgets.QMessageBox.information(self, "Saved", "Recording stopped and saved.")

    def display_video_frames(self, frame_in, _):
        self.input_label.setPixmap(self.to_pixmap(frame_in, self.input_label.size()))
        processed = dehaze(frame_in) if self.fog_removal_enabled else frame_in
        if self.he_radio.isChecked():
            processed = histogram_equalization(processed)
        elif self.clahe_radio.isChecked():
            processed = clahe_enhancement(processed)
        elif self.gamma_radio.isChecked():
            gamma_value = self.gamma_slider.value() / 10.0
            processed = gamma_correction(processed, gamma_value)
        contrast = self.contrast_slider.value() / 100.0
        brightness = self.brightness_slider.value() - 100
        adjusted = cv2.convertScaleAbs(processed, alpha=contrast, beta=brightness)
        self.output_label.setPixmap(self.to_pixmap(adjusted, self.output_label.size()))
        self.last_output_frame = adjusted.copy()
        if self.recording and self.video_writer:
            self.video_writer.write(adjusted)

    def image_fusion(self):
        dialog = ImageFusionDialog()
        dialog.fusion_done.connect(self.handle_fusion_result)
        dialog.exec_()

    def handle_fusion_result(self, img1, img2, fused):
        combined_input = np.hstack((img1, img2))
        self.input_label.setPixmap(self.to_pixmap(combined_input, self.input_label.size()))
        self.output_label.setPixmap(self.to_pixmap(fused, self.output_label.size()))
        self.last_output_frame = fused.copy()

    def to_pixmap(self, frame, size):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimg).scaled(size, QtCore.Qt.KeepAspectRatio)

    def stop_threads(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

    def closeEvent(self, event):
        self.stop_threads()
        if self.video_writer:
            self.video_writer.release()
        event.accept()

# === Main Function ===
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        with open("style.qss", "r") as f:
            app.setStyleSheet(f.read())
    except Exception as e:
        print(f"Warning: Could not load style.qss file: {e}")
    window = SimpleVideoGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()