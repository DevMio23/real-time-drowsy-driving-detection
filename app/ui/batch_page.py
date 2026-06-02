"""Batch video analysis page."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.ui import styles


class BatchPage(QWidget):
    def __init__(self, on_back, on_pick_video, on_start, on_stop, parent=None):
        super().__init__(parent)
        self._on_back = on_back
        self._on_pick_video = on_pick_video
        self._on_start = on_start
        self._on_stop = on_stop
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(12)

        title = QLabel("Batch Video Analysis")
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        title.setStyleSheet(f"color: {styles.TEXT_DARK};")

        self.video_path_label = QLabel("No video selected.")
        self.video_path_label.setWordWrap(True)
        self.video_path_label.setStyleSheet(f"color: {styles.TEXT_MUTED}; font-size: 12px;")

        btn_row = QHBoxLayout()
        self.pick_btn = QPushButton("Choose Video")
        self.pick_btn.setStyleSheet(styles.BTN_SECONDARY)
        self.pick_btn.clicked.connect(self._on_pick_video)

        self.start_btn = QPushButton("Start Batch Analysis")
        self.start_btn.setStyleSheet(styles.BTN_PRIMARY)
        self.start_btn.clicked.connect(self._on_start)
        self.start_btn.setEnabled(False)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(styles.BTN_DANGER)
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)

        self.back_btn = QPushButton("Back to Home")
        self.back_btn.setStyleSheet(styles.BTN_SECONDARY)
        self.back_btn.clicked.connect(self._on_back)

        btn_row.addWidget(self.pick_btn)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.back_btn)

        self.preview_label = QLabel("Preview will appear here during analysis.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(420)
        self.preview_label.setStyleSheet(
            f"border: 2px solid {styles.PANEL_BORDER}; border-radius: 10px; "
            f"background-color: {styles.VIDEO_BG}; color: #bdc3c7; font-size: 13px;"
        )

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)

        self.status_label = QLabel("Idle.")
        self.status_label.setStyleSheet(f"color: {styles.TEXT_DARK};")

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(130)
        self.summary_box.setStyleSheet("background-color: #ffffff; font-size: 12px;")

        root.addWidget(title)
        root.addWidget(self.video_path_label)
        root.addLayout(btn_row)
        root.addWidget(self.preview_label, 1)
        root.addWidget(self.progress)
        root.addWidget(self.status_label)
        root.addWidget(self.summary_box)

    def set_video_path(self, path: str) -> None:
        self.video_path_label.setText(path)
        self.start_btn.setEnabled(bool(path))

    def set_running(self, running: bool) -> None:
        self.pick_btn.setEnabled(not running)
        self.start_btn.setEnabled((not running) and self.video_path_label.text() != "No video selected.")
        self.stop_btn.setEnabled(running)
        self.back_btn.setEnabled(not running)

    def set_progress(self, processed: int, total: int) -> None:
        if total <= 0:
            self.progress.setRange(0, 0)
            self.status_label.setText(f"Processed {processed} frames...")
            return
        self.progress.setRange(0, 100)
        pct = int((processed / total) * 100)
        self.progress.setValue(max(0, min(100, pct)))
        self.status_label.setText(f"Processed {processed}/{total} frames ({pct}%).")

    def append_summary(self, text: str) -> None:
        self.summary_box.append(text)

    def clear_summary(self) -> None:
        self.summary_box.clear()
