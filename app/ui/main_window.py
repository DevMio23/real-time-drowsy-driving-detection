"""Main application window."""

from __future__ import annotations

import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget

from app import config
from app.core.alert_manager import AlertManager
from app.core.detector import DetectionMetrics, DrowsinessDetectorEngine
from app.core.session_logger import SessionLogger
from app.ui.detection_page import DetectionPage
from app.ui import styles
from app.ui.welcome_page import WelcomePage
from app.workers.video_worker import VideoSignals, VideoWorker


class MainWindow(QMainWindow):
    def __init__(self, engine: DrowsinessDetectorEngine | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Driver Drowsiness Detection System")
        self.setGeometry(100, 100, 1100, 720)
        self.setStyleSheet(styles.MAIN_WINDOW_STYLE)

        if engine is None:
            try:
                engine = DrowsinessDetectorEngine()
            except Exception as exc:
                print(f"Error loading models: {exc}")
                sys.exit(1)
        self.engine = engine
        self.session_logger = SessionLogger()
        self.alert_manager = AlertManager()
        self.engine.set_session_logger(self.session_logger)

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
            on_debug_toggle=self._on_debug_toggle,
        )
        self.detection_page.debug_checkbox.setChecked(self.engine.show_debug_overlay)

        self.stacked.addWidget(self.welcome_page)
        self.stacked.addWidget(self.detection_page)
        self.stacked.setCurrentIndex(config.PAGE_WELCOME)

    def _on_debug_toggle(self, enabled: bool) -> None:
        self.engine.show_debug_overlay = enabled

    def _go_to_detection(self) -> None:
        self.stacked.setCurrentIndex(config.PAGE_DETECTION)
        self.detection_page.reset_video_style()
        self.detection_page.video_label.setText("Starting camera...")
        self._start_detection()

    def _start_detection(self) -> None:
        if self.video_worker.is_running:
            return
        session_id = self.session_logger.start_session()
        print(f"Session logging started: {session_id}")
        self.alert_manager.reset()
        self.detection_page.alert_history.clear_history()
        self.engine.reset_state()
        self.video_worker.start()

    def _stop_and_home(self) -> None:
        self.shutdown_detection()
        self.detection_page.set_stopped_message()
        self.detection_page.video_label.clear()
        self.detection_page.video_label.setText("Camera feed stopped.")
        self.stacked.setCurrentIndex(config.PAGE_WELCOME)

    def shutdown_detection(self) -> None:
        self.video_worker.stop()
        if self.session_logger.is_active:
            stats = self.session_logger.get_statistics()
            print(f"Session saved: {stats.csv_path}")
            self.session_logger.end_session()

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

    def _format_stats_html(self, stats) -> str:
        elapsed_min = int(stats.elapsed_s // 60)
        elapsed_sec = int(stats.elapsed_s % 60)
        return (
            f"<b>Elapsed:</b> {elapsed_min}m {elapsed_sec}s<br>"
            f"<b>Blinks:</b> {stats.blink_count} "
            f"({stats.blinks_per_min:.1f}/min)<br>"
            f"<b>Yawns:</b> {stats.yawn_count} "
            f"({stats.yawns_per_hour:.1f}/hr)<br>"
            f"<b>Alerts:</b> {stats.alert_count} | "
            f"<b>Microsleeps:</b> {stats.microsleep_episodes}"
        )

    def _on_metrics_ready(self, metrics: DetectionMetrics) -> None:
        if self.stacked.currentIndex() != config.PAGE_DETECTION:
            return
        if metrics.alert_level == "critical":
            border_color = styles.DANGER
        elif metrics.alert_level == "warning":
            border_color = styles.WARNING
        else:
            border_color = styles.PANEL_BORDER
        self.detection_page.video_label.setStyleSheet(
            f"border: 3px solid {border_color}; border-radius: 10px; "
            f"background-color: {styles.VIDEO_BG};"
        )

        if metrics.alert_triggered and metrics.alert_kind:
            record = self.alert_manager.record_alert(
                metrics.alert_kind, metrics.alert_level
            )
            self.detection_page.add_alert_to_history(record)
            if metrics.alert_level == "critical":
                self.alert_manager.play_alert_sound(
                    enabled=self.detection_page.alert_sounds_enabled()
                )
        html = (
            "<div style='font-family: Segoe UI, sans-serif; color: #2c3e50;'>"
            "<h2 style='text-align: center; color: #3498db;'>Driver Status</h2>"
            "<hr style='border: 1px solid #bdc3c7;'/>"
            f"{metrics.alert_html}"
            f"<p><b>Session blinks:</b> {metrics.session_blinks} | "
            f"<b>Recent:</b> {metrics.recent_blinks} "
            f"(last {config.BLINK_WINDOW}s)</p>"
            f"<p><b>Eyes closed:</b> {metrics.eyes_closed_duration:.2f}s</p>"
            f"<p><b>Current yawn:</b> {metrics.yawn_duration:.2f}s</p>"
            f"<p><b>Total yawns:</b> {metrics.total_yawns}</p>"
            f"<p><b>Eyes:</b> {metrics.left_eye_state or '—'} / "
            f"{metrics.right_eye_state or '—'} "
            f"({metrics.blink_method})</p>"
        )
        if config.USE_EAR_FOR_BLINKS:
            html += (
                f"<p style='font-size: 11px; color: #7f8c8d;'>"
                f"EAR L={metrics.left_ear:.2f} R={metrics.right_ear:.2f}</p>"
            )
        html += f"<p><b>Yawn state:</b> {metrics.yawn_state or '—'}</p>"
        if metrics.yawn_suppressed:
            html += (
                "<p style='font-size: 11px; color: #e67e22;'>"
                "Yawn count paused (eyes closed)</p>"
            )
        html += (
            f"<p style='font-size: 11px; color: #95a5a6;'>"
            f"Process ~{metrics.process_fps:.1f} fps</p>"
            "</div>"
        )
        self.detection_page.info_label.setText(html)

        if self.session_logger.is_active:
            stats = self.session_logger.get_statistics()
            self.detection_page.update_session_stats(
                self._format_stats_html(stats),
                stats.csv_path,
                stats,
            )

    def _on_camera_error(self, message: str) -> None:
        self.video_worker.stop()
        if self.session_logger.is_active:
            self.session_logger.end_session()
        self.detection_page.set_camera_error(message)
