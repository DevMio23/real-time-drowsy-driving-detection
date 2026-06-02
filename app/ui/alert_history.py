"""Scrollable alert history list for the live detection sidebar."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from app.core.alert_manager import AlertRecord
from app.ui import styles


class AlertHistoryWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel("Alert history")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        title.setStyleSheet(f"color: {styles.TEXT_DARK};")

        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(130)
        self.list_widget.setStyleSheet(
            "QListWidget { font-size: 10px; border: 1px solid #ecf0f1; "
            "border-radius: 4px; background: #fafafa; }"
        )

        layout.addWidget(title)
        layout.addWidget(self.list_widget)
        self.clear_history()

    def clear_history(self) -> None:
        self.list_widget.clear()
        placeholder = QListWidgetItem("No alerts this session.")
        placeholder.setFlags(Qt.NoItemFlags)
        placeholder.setForeground(QColor(styles.TEXT_MUTED))
        self.list_widget.addItem(placeholder)

    def add_entry(self, record: AlertRecord) -> None:
        if self.list_widget.count() == 1:
            first = self.list_widget.item(0)
            if first and not (first.flags() & Qt.ItemIsSelectable):
                self.list_widget.clear()

        color = styles.DANGER if record.level == "critical" else styles.WARNING
        item = QListWidgetItem(f"{record.time_str} — {record.label}")
        item.setForeground(QColor(color))
        item.setToolTip(record.message)
        self.list_widget.insertItem(0, item)
