"""Main application window."""

from __future__ import annotations

import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget

from app import config
from app.core.detector import DetectionMetrics, DrowsinessDetectorEngine
from app.ui.detection_page import DetectionPage
from app.ui import styles
from app.ui.welcome_page import WelcomePage
from app.workers.video_worker import VideoSignals, VideoWorker


class MainWindow(QMainWindow):
    def __init__(self, engine: DrowsinessDetectorEngine | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Driver Drowsiness Detection System")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet(styles.MAIN_WINDOW_STYLE)

        if engine is None:
            # Fallback if constructed without main(); keep Qt-after-ML order in main.py.
            try:
                engine = DrowsinessDetectorEngine()
            except Exception as exc:
                print(f"Error loading models: {exc}")
                sys.exit(1)
        self.engine = engine

        self.signals = VideoSignals()
        self.video_worker = VideoWorker(self.engine, self.signals)
        self.signals.frame_ready.connect(self._on_frame_ready)
        self.signals.metrics_ready.connect(self._on_metrics_ready)
        self.signals.camera_error.connect(self._on_camera_error)

        self.stacked = QStackedWidget()
        self.setCentralWidget(self.stacked)

        self.welcome_page = WelcomePage(on_start=self._go_to_detection)
        self.detection_page = DetectionPage(
            on_stop=self._stop_and_home,
            on_back=self._stop_and_home,
        )

        self.stacked.addWidget(self.welcome_page)
        self.stacked.addWidget(self.detection_page)
        self.stacked.setCurrentIndex(config.PAGE_WELCOME)

    def _go_to_detection(self) -> None:
        self.stacked.setCurrentIndex(config.PAGE_DETECTION)
        self.detection_page.reset_video_style()
        self.detection_page.video_label.setText("Starting camera...")
        self._start_detection()

    def _start_detection(self) -> None:
        if self.video_worker.is_running:
            return
        if not self.video_worker.start():
            self.detection_page.set_camera_error(
                "Could not open the camera. Check permissions and that no other app is using it."
            )

    def _stop_and_home(self) -> None:
        self.shutdown_detection()
        self.detection_page.set_stopped_message()
        self.detection_page.video_label.clear()
        self.detection_page.video_label.setText("Camera feed stopped.")
        self.stacked.setCurrentIndex(config.PAGE_WELCOME)

    def shutdown_detection(self) -> None:
        self.video_worker.stop()

    def shutdown(self) -> None:
        self.shutdown_detection()
        self.engine.close()

    def closeEvent(self, event) -> None:
        self.shutdown()
        event.accept()

    def _on_frame_ready(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(
            rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(image).scaled(
            self.detection_page.video_label.width(),
            self.detection_page.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.detection_page.video_label.setPixmap(pixmap)

    def _on_metrics_ready(self, metrics: DetectionMetrics) -> None:
        if self.stacked.currentIndex() != config.PAGE_DETECTION:
            return
        border_color = "#e74c3c" if metrics.alert_active else styles.PANEL_BORDER
        self.detection_page.video_label.setStyleSheet(
            f"border: 3px solid {border_color}; border-radius: 10px; "
            f"background-color: {styles.VIDEO_BG};"
        )
        html = (
            "<div style='font-family: Segoe UI, sans-serif; color: #2c3e50;'>"
            "<h2 style='text-align: center; color: #3498db;'>Driver Status</h2>"
            "<hr style='border: 1px solid #bdc3c7;'/>"
            f"{metrics.alert_html}"
            f"<p><b>Recent blinks:</b> {metrics.recent_blinks} "
            f"(last {config.BLINK_WINDOW}s)</p>"
            f"<p><b>Eyes closed:</b> {metrics.eyes_closed_duration:.2f}s</p>"
            f"<p><b>Current yawn:</b> {metrics.yawn_duration:.2f}s</p>"
            f"<p><b>Total yawns:</b> {metrics.total_yawns}</p>"
            f"<p><b>Left / right eye:</b> {metrics.left_eye_state or '—'} / "
            f"{metrics.right_eye_state or '—'}</p>"
            f"<p><b>Yawn state:</b> {metrics.yawn_state or '—'}</p>"
            f"<p style='font-size: 11px; color: #95a5a6;'>"
            f"Process ~{metrics.process_fps:.1f} fps "
            f"(target {config.TARGET_PROCESS_FPS})</p>"
            "<hr style='border: 1px solid #bdc3c7;'/>"
            "<p style='font-size: 12px; color: #7f8c8d;'>"
            f"Alerts: yawn &ge; {config.YAWN_THRESHOLD}s, "
            f"microsleep &ge; {config.MICROSLEEP_THRESHOLD}s, "
            f"{config.BLINK_THRESHOLD} blinks in {config.BLINK_WINDOW}s"
            "</p></div>"
        )
        self.detection_page.info_label.setText(html)

    def _on_camera_error(self, message: str) -> None:
        self.detection_page.set_camera_error(message)
