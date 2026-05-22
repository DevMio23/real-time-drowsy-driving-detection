"""Welcome / quick-start screen."""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from app.ui import styles


class WelcomePage(QWidget):
    def __init__(self, on_start, parent=None):
        super().__init__(parent)
        self._on_start = on_start
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(24)

        title = QLabel("Driver Drowsiness Detection")
        title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        title.setStyleSheet(f"color: {styles.TEXT_DARK};")
        title.setAlignment(Qt.AlignCenter)

        desc = QLabel(
            "<p style='text-align: center; font-size: 15px; color: #7f8c8d; max-width: 520px;'>"
            "Real-time monitoring of eye closure, blinking, and yawning "
            "using computer vision. Click <b>Start Detection</b> when your webcam is ready."
            "</p>"
            "<p style='text-align: center; font-size: 13px; color: #95a5a6;'>"
            "Use <b>Stop Detection</b> or close the window when finished — "
            "do not rely on force-closing the terminal."
            "</p>"
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)

        start_btn = QPushButton("Start Detection")
        start_btn.setFont(QFont("Segoe UI", 14))
        start_btn.setStyleSheet(styles.BTN_PRIMARY)
        start_btn.setCursor(Qt.PointingHandCursor)
        start_btn.clicked.connect(self._on_start)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.setStyleSheet(f"background-color: {styles.APP_BACKGROUND};")
