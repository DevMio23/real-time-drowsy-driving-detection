"""Live detection screen layout."""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.core.session_logger import SessionStatistics
from app.ui import styles
from app.ui.stats_panel import StatsPanel


class DetectionPage(QWidget):
    def __init__(self, on_stop, on_back, on_debug_toggle=None, parent=None):
        super().__init__(parent)
        self._on_stop = on_stop
        self._on_back = on_back
        self._on_debug_toggle = on_debug_toggle

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        self.video_label = QLabel("Camera feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(700, 500)
        self.video_label.setStyleSheet(
            f"border: 2px solid {styles.PANEL_BORDER}; border-radius: 10px; "
            f"background-color: {styles.VIDEO_BG}; color: #bdc3c7; font-size: 14px;"
        )

        panel = QWidget()
        panel.setFixedWidth(320)
        panel.setStyleSheet(
            "background-color: white; border: 1px solid #bdc3c7; border-radius: 10px;"
        )
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        self.info_label = QLabel()
        self.info_label.setFont(QFont("Segoe UI", 10))
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(f"color: {styles.TEXT_DARK};")

        self.stats_title = QLabel("Session statistics")
        self.stats_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.stats_title.setStyleSheet(f"color: {styles.TEXT_DARK};")

        self.stats_summary = QLabel()
        self.stats_summary.setWordWrap(True)
        self.stats_summary.setStyleSheet("color: #7f8c8d; font-size: 10px;")

        self.log_path_label = QLabel()
        self.log_path_label.setWordWrap(True)
        self.log_path_label.setStyleSheet("color: #95a5a6; font-size: 9px;")

        self.stats_panel = StatsPanel()

        scroll_layout.addWidget(self.info_label)
        scroll_layout.addWidget(self.stats_title)
        scroll_layout.addWidget(self.stats_summary)
        scroll_layout.addWidget(self.stats_panel)
        scroll_layout.addWidget(self.log_path_label)
        scroll.setWidget(scroll_content)

        self.debug_checkbox = QCheckBox("Show debug overlay")
        self.debug_checkbox.setStyleSheet(f"color: {styles.TEXT_DARK};")
        if self._on_debug_toggle is not None:
            self.debug_checkbox.toggled.connect(self._on_debug_toggle)

        stop_btn = QPushButton("Stop Detection")
        stop_btn.setStyleSheet(styles.BTN_DANGER)
        stop_btn.setCursor(Qt.PointingHandCursor)
        stop_btn.clicked.connect(self._on_stop)

        back_btn = QPushButton("Back to Home")
        back_btn.setStyleSheet(styles.BTN_SECONDARY)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(self._on_back)

        panel_layout.addWidget(scroll, 1)
        panel_layout.addWidget(self.debug_checkbox)
        panel_layout.addWidget(stop_btn)
        panel_layout.addWidget(back_btn)

        root.addWidget(self.video_label, 3)
        root.addWidget(panel, 1)

        self.set_stopped_message()

    def set_stopped_message(self) -> None:
        self.info_label.setText(
            "<div style='font-family: Segoe UI, sans-serif;'>"
            "<h2 style='text-align: center; color: #3498db;'>Driver Status</h2>"
            "<hr style='border: 1px solid #bdc3c7;'/>"
            "<p style='color: #7f8c8d;'>Detection stopped.</p>"
            "</div>"
        )
        self.stats_summary.setText("No active session.")
        self.log_path_label.setText("")
        self.stats_panel.update_statistics(SessionStatistics())

    def update_session_stats(self, summary_html: str, log_path: str, stats) -> None:
        self.stats_summary.setText(summary_html)
        self.log_path_label.setText(
            f"CSV: {log_path}" if log_path else ""
        )
        self.stats_panel.update_statistics(stats)

    def set_camera_error(self, message: str) -> None:
        self.video_label.setText(message)
        self.video_label.setStyleSheet(
            f"border: 2px solid {styles.DANGER}; border-radius: 10px; "
            f"background-color: {styles.VIDEO_BG}; color: #e74c3c; padding: 20px;"
        )

    def reset_video_style(self) -> None:
        self.video_label.setStyleSheet(
            f"border: 2px solid {styles.PANEL_BORDER}; border-radius: 10px; "
            f"background-color: {styles.VIDEO_BG}; color: #bdc3c7;"
        )
        self.video_label.setText("Starting camera...")
