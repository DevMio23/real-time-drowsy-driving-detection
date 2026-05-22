"""Background capture and processing threads with Qt signals."""

from __future__ import annotations

import queue
import threading
import time

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from app import config
from app.core.detector import DetectionMetrics, DrowsinessDetectorEngine


class VideoSignals(QObject):
    """Thread-safe signals emitted to the main Qt thread."""

    frame_ready = pyqtSignal(object)
    metrics_ready = pyqtSignal(object)
    camera_error = pyqtSignal(str)


class VideoWorker:
    """Manages capture/process threads; does not subclass QThread."""

    def __init__(self, engine: DrowsinessDetectorEngine, signals: VideoSignals) -> None:
        self.engine = engine
        self.signals = signals
        self.stop_event = threading.Event()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
        self.cap: cv2.VideoCapture | None = None
        self.capture_thread: threading.Thread | None = None
        self.process_thread: threading.Thread | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> bool:
        if self._running:
            return True

        self.stop_event.clear()
        self.engine.reset_state()

        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self.cap.isOpened():
            self.signals.camera_error.emit(
                "Camera not found or in use. Check that a webcam is connected."
            )
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False

        self._drain_queue()
        self.capture_thread = threading.Thread(
            target=self._capture_loop, name="capture", daemon=True
        )
        self.process_thread = threading.Thread(
            target=self._process_loop, name="process", daemon=True
        )
        self.capture_thread.start()
        self.process_thread.start()
        self._running = True
        return True

    def stop(self) -> None:
        if not self._running and self.stop_event.is_set():
            return

        self.stop_event.set()
        self._join_threads()
        self._release_camera()
        self._drain_queue()
        self._running = False

    def _join_threads(self) -> None:
        for thread in (self.capture_thread, self.process_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=config.THREAD_JOIN_TIMEOUT_S)

    def _release_camera(self) -> None:
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None

    def _drain_queue(self) -> None:
        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set() and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass
            time.sleep(config.CAPTURE_SLEEP_S)

    def _process_loop(self) -> None:
        frame_time = 1.0 / config.TARGET_FPS

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=config.PROCESS_QUEUE_TIMEOUT_S)
            except queue.Empty:
                continue

            start = time.time()
            try:
                annotated, metrics = self.engine.process_frame(frame)
                self.signals.frame_ready.emit(annotated.copy())
                self.signals.metrics_ready.emit(metrics)
            except Exception as exc:
                print(f"Processing error: {exc}")

            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
