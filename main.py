"""Unified entry point for the drowsiness detection application."""

import sys


def main() -> int:
    # MediaPipe + YOLO must initialize before Qt on Windows (avoids segfault).
    print("Loading detection models (first run may take a few seconds)...")
    from app.core.detector import DrowsinessDetectorEngine

    engine = DrowsinessDetectorEngine()
    print("Models loaded.")

    from PyQt5.QtWidgets import QApplication

    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(engine=engine)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
