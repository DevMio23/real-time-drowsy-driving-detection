"""Live detection screen layout."""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app.ui import styles


class DetectionPage(QWidget):
    def __init__(self, on_stop, on_back, parent=None):
        super().__init__(parent)
        self._on_stop = on_stop
        self._on_back = on_back

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
        panel.setStyleSheet(
            "background-color: white; border: 1px solid #bdc3c7; border-radius: 10px;"
        )
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 16, 16, 16)

        self.info_label = QLabel()
        self.info_label.setFont(QFont("Segoe UI", 11))
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(f"color: {styles.TEXT_DARK};")
        self.set_stopped_message()

        stop_btn = QPushButton("Stop Detection")
        stop_btn.setStyleSheet(styles.BTN_DANGER)
        stop_btn.setCursor(Qt.PointingHandCursor)
        stop_btn.clicked.connect(self._on_stop)

        back_btn = QPushButton("Back to Home")
        back_btn.setStyleSheet(styles.BTN_SECONDARY)
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(self._on_back)

        panel_layout.addWidget(self.info_label)
        panel_layout.addStretch(1)
        panel_layout.addWidget(stop_btn)
        panel_layout.addWidget(back_btn)

        root.addWidget(self.video_label, 3)
        root.addWidget(panel, 1)

    def set_stopped_message(self) -> None:
        self.info_label.setText(
            "<div style='font-family: Segoe UI, sans-serif;'>"
            "<h2 style='text-align: center; color: #3498db;'>Driver Status</h2>"
            "<hr style='border: 1px solid #bdc3c7;'/>"
            "<p style='color: #7f8c8d;'>Detection stopped.</p>"
            "</div>"
        )

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
