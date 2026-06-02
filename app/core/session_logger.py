"""CSV session logging and aggregate statistics for thesis analysis."""

from __future__ import annotations

import csv
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app import config

CSV_HEADERS = [
    "timestamp_iso",
    "session_id",
    "event_type",
    "duration_s",
    "confidence",
    "left_eye",
    "right_eye",
    "yawn_state",
    "alert_level",
]


@dataclass
class SessionStatistics:
    session_id: str = ""
    elapsed_s: float = 0.0
    blink_count: int = 0
    yawn_count: int = 0
    alert_count: int = 0
    microsleep_episodes: int = 0
    blinks_per_min: float = 0.0
    yawns_per_hour: float = 0.0
    csv_path: str = ""
    trend_minutes: list[int] = field(default_factory=list)
    trend_blinks: list[int] = field(default_factory=list)
    trend_yawns: list[int] = field(default_factory=list)


class SessionLogger:
    """Append-only CSV logger with in-memory aggregates for the UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session_id = ""
        self._session_start: float = 0.0
        self._csv_path: Path | None = None
        self._file = None
        self._writer: csv.DictWriter | None = None
        self._blink_count = 0
        self._yawn_count = 0
        self._alert_count = 0
        self._microsleep_count = 0
        self._events_by_minute: dict[int, dict[str, int]] = defaultdict(
            lambda: {"blink": 0, "yawn": 0}
        )

    @property
    def is_active(self) -> bool:
        return self._csv_path is not None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def csv_path(self) -> Path | None:
        return self._csv_path

    def start_session(self) -> str:
        with self._lock:
            self._end_session_unlocked()
            config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._session_id = stamp
            self._session_start = time.time()
            self._csv_path = config.LOGS_DIR / f"{stamp}.csv"
            self._file = open(self._csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=CSV_HEADERS)
            self._writer.writeheader()
            self._file.flush()
            self._blink_count = 0
            self._yawn_count = 0
            self._alert_count = 0
            self._microsleep_count = 0
            self._events_by_minute.clear()
            self._log_event_unlocked("session_start", alert_level="info")
            return self._session_id

    def end_session(self) -> None:
        with self._lock:
            self._end_session_unlocked()

    def _end_session_unlocked(self) -> None:
        if self._writer is not None:
            self._log_event_unlocked("session_end", alert_level="info")
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None
        self._csv_path = None
        self._session_id = ""

    def log_event(
        self,
        event_type: str,
        *,
        duration_s: float = 0.0,
        confidence: float = 0.0,
        left_eye: str = "",
        right_eye: str = "",
        yawn_state: str = "",
        alert_level: str = "",
    ) -> None:
        with self._lock:
            self._log_event_unlocked(
                event_type,
                duration_s=duration_s,
                confidence=confidence,
                left_eye=left_eye,
                right_eye=right_eye,
                yawn_state=yawn_state,
                alert_level=alert_level,
            )

    def _log_event_unlocked(
        self,
        event_type: str,
        *,
        duration_s: float = 0.0,
        confidence: float = 0.0,
        left_eye: str = "",
        right_eye: str = "",
        yawn_state: str = "",
        alert_level: str = "",
    ) -> None:
        if self._writer is None:
            return

        row = {
            "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            "session_id": self._session_id,
            "event_type": event_type,
            "duration_s": round(duration_s, 3),
            "confidence": round(confidence, 3),
            "left_eye": left_eye,
            "right_eye": right_eye,
            "yawn_state": yawn_state,
            "alert_level": alert_level,
        }
        self._writer.writerow(row)
        self._file.flush()

        minute_bucket = int((time.time() - self._session_start) // 60)
        if event_type == "blink":
            self._blink_count += 1
            self._events_by_minute[minute_bucket]["blink"] += 1
        elif event_type in ("yawn_start", "yawn"):
            self._yawn_count += 1
            self._events_by_minute[minute_bucket]["yawn"] += 1
        elif event_type.startswith("alert_"):
            self._alert_count += 1
        elif event_type == "microsleep_start":
            self._microsleep_count += 1

    def get_statistics(self) -> SessionStatistics:
        with self._lock:
            if not self._session_id:
                return SessionStatistics()

            elapsed = max(time.time() - self._session_start, 0.001)
            minutes = elapsed / 60.0
            hours = elapsed / 3600.0

            window = config.TREND_WINDOW_MINUTES
            current_minute = int(elapsed // 60)
            start_minute = max(0, current_minute - window + 1)

            trend_minutes: list[int] = []
            trend_blinks: list[int] = []
            trend_yawns: list[int] = []
            for m in range(start_minute, current_minute + 1):
                trend_minutes.append(m)
                bucket = self._events_by_minute[m]
                trend_blinks.append(bucket["blink"])
                trend_yawns.append(bucket["yawn"])

            return SessionStatistics(
                session_id=self._session_id,
                elapsed_s=elapsed,
                blink_count=self._blink_count,
                yawn_count=self._yawn_count,
                alert_count=self._alert_count,
                microsleep_episodes=self._microsleep_count,
                blinks_per_min=self._blink_count / minutes if minutes > 0 else 0.0,
                yawns_per_hour=self._yawn_count / hours if hours > 0 else 0.0,
                csv_path=str(self._csv_path or ""),
                trend_minutes=trend_minutes,
                trend_blinks=trend_blinks,
                trend_yawns=trend_yawns,
            )
