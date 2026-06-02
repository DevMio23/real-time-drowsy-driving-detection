"""Session statistics chart (matplotlib embedded in PyQt)."""

from __future__ import annotations

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from app.core.session_logger import SessionStatistics

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover
    FigureCanvas = None
    Figure = None


class StatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if FigureCanvas is None:
            self._canvas = None
            return

        self._figure = Figure(figsize=(3.2, 2.2), dpi=100)
        self._figure.patch.set_facecolor("#fafafa")
        self._ax = self._figure.add_subplot(111)
        self._canvas = FigureCanvas(self._figure)
        layout.addWidget(self._canvas)
        self._clear_plot()

    def _clear_plot(self) -> None:
        if self._canvas is None:
            return
        self._ax.clear()
        self._ax.set_title("Events per minute", fontsize=9)
        self._ax.set_xlabel("Minute", fontsize=8)
        self._ax.set_ylabel("Count", fontsize=8)
        self._ax.tick_params(labelsize=7)
        self._ax.grid(True, alpha=0.3)
        self._figure.tight_layout()
        self._canvas.draw_idle()

    def update_statistics(self, stats: SessionStatistics) -> None:
        if self._canvas is None:
            return

        self._ax.clear()
        if not stats.trend_minutes:
            self._ax.text(
                0.5,
                0.5,
                "No events yet",
                ha="center",
                va="center",
                transform=self._ax.transAxes,
                fontsize=9,
                color="#7f8c8d",
            )
        else:
            x = stats.trend_minutes
            self._ax.bar(
                [i - 0.15 for i in x],
                stats.trend_blinks,
                width=0.3,
                label="Blinks",
                color="#3498db",
            )
            self._ax.bar(
                [i + 0.15 for i in x],
                stats.trend_yawns,
                width=0.3,
                label="Yawns",
                color="#e67e22",
            )
            self._ax.legend(fontsize=7, loc="upper right")

        self._ax.set_title("Events per minute", fontsize=9)
        self._ax.set_xlabel("Minute of session", fontsize=8)
        self._ax.set_ylabel("Count", fontsize=8)
        self._ax.tick_params(labelsize=7)
        self._ax.grid(True, alpha=0.3)
        self._figure.tight_layout()
        self._canvas.draw_idle()
