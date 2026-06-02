"""Background worker for offline batch video analysis."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from app.core.detector import DetectionMetrics, DrowsinessDetectorEngine
from app.core.session_logger import SessionLogger


@dataclass
class BatchSummary:
    session_id: str
    source_video: str
    frames_processed: int
    elapsed_s: float
    csv_path: str
    summary_path: str
    blink_count: int
    yawn_count: int
    alert_count: int
    microsleep_episodes: int


class BatchSignals(QObject):
    frame_ready = pyqtSignal(object)
    progress = pyqtSignal(int, int)
    metrics_ready = pyqtSignal(object)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class BatchVideoWorker:
    """Processes a saved video in a background thread."""

    def __init__(
        self,
        engine: DrowsinessDetectorEngine,
        session_logger: SessionLogger,
        signals: BatchSignals,
    ) -> None:
        self.engine = engine
        self.session_logger = session_logger
        self.signals = signals
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._running = False
        self._video_path = ""

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, video_path: str) -> bool:
        if self._running:
            return False
        self._video_path = video_path
        self.stop_event.clear()
        self._running = True
        self.thread = threading.Thread(target=self._run, name="batch-video", daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        if not self._running and self.stop_event.is_set():
            return
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._running = False

    def _run(self) -> None:
        cap: cv2.VideoCapture | None = None
        session_id = ""
        try:
            source = Path(self._video_path)
            if not source.exists():
                self.signals.error.emit(f"Video file not found: {source}")
                return

            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                self.signals.error.emit("Could not open selected video file.")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 240:
                fps = 25.0
            delta_s = 1.0 / fps

            session_id = self.session_logger.start_session()
            self.engine.reset_state()
            self.signals.status.emit(f"Batch session started: {session_id}")

            frame_idx = 0
            wall_start = time.perf_counter()
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                annotated, metrics = self.engine.process_frame(frame, delta_s=delta_s)
                self.signals.frame_ready.emit(annotated.copy())
                self.signals.metrics_ready.emit(metrics)
                self.signals.progress.emit(frame_idx, total_frames)

            elapsed = max(time.perf_counter() - wall_start, 0.001)
            stats = self.session_logger.get_statistics()
            summary_path = self._write_summary_json(
                session_id=session_id,
                source_video=str(source),
                frames_processed=frame_idx,
                elapsed_s=elapsed,
                stats=stats,
            )
            summary = BatchSummary(
                session_id=session_id,
                source_video=str(source),
                frames_processed=frame_idx,
                elapsed_s=elapsed,
                csv_path=stats.csv_path,
                summary_path=str(summary_path),
                blink_count=stats.blink_count,
                yawn_count=stats.yawn_count,
                alert_count=stats.alert_count,
                microsleep_episodes=stats.microsleep_episodes,
            )
            self.signals.finished.emit(summary)
        except Exception as exc:
            self.signals.error.emit(f"Batch analysis failed: {exc}")
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            if self.session_logger.is_active:
                self.session_logger.end_session()
            self._running = False

    def _write_summary_json(
        self,
        *,
        session_id: str,
        source_video: str,
        frames_processed: int,
        elapsed_s: float,
        stats,
    ) -> Path:
        csv_path = Path(stats.csv_path)
        summary_path = csv_path.with_name(f"{session_id}_summary.json")
        payload = {
            "session_id": session_id,
            "source_video": source_video,
            "frames_processed": frames_processed,
            "elapsed_s": round(elapsed_s, 3),
            "blink_count": stats.blink_count,
            "yawn_count": stats.yawn_count,
            "alert_count": stats.alert_count,
            "microsleep_episodes": stats.microsleep_episodes,
            "blinks_per_min": round(stats.blinks_per_min, 3),
            "yawns_per_hour": round(stats.yawns_per_hour, 3),
            "csv_path": stats.csv_path,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return summary_path
