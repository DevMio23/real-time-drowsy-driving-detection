"""Alert audio, cooldown, and in-session history (main-thread only)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime

from app import config

ALERT_LABELS = {
    "alert_yawn": "Prolonged yawn",
    "alert_microsleep": "Microsleep",
    "alert_blink_rate": "Excessive blinking",
    "warning_yawn": "Yawn duration rising",
    "warning_microsleep": "Eyes closed longer",
    "warning_blink_rate": "Blink rate rising",
}


@dataclass(frozen=True)
class AlertRecord:
    timestamp: datetime
    kind: str
    message: str
    level: str

    @property
    def time_str(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")

    @property
    def label(self) -> str:
        return ALERT_LABELS.get(self.kind, self.kind.replace("_", " ").title())


class AlertManager:
    """Plays alert audio with cooldown and keeps a bounded session history."""

    def __init__(self) -> None:
        self._last_sound_at = 0.0
        self._history: list[AlertRecord] = []

    def reset(self) -> None:
        self._last_sound_at = 0.0
        self._history.clear()

    @property
    def history(self) -> list[AlertRecord]:
        return list(self._history)

    def record_alert(self, kind: str, level: str) -> AlertRecord:
        message = ALERT_LABELS.get(kind, kind.replace("_", " ").title())
        record = AlertRecord(
            timestamp=datetime.now(),
            kind=kind,
            message=message,
            level=level,
        )
        self._history.insert(0, record)
        if len(self._history) > config.ALERT_HISTORY_MAX:
            del self._history[config.ALERT_HISTORY_MAX :]
        return record

    def play_alert_sound(self, *, enabled: bool = True) -> bool:
        if not enabled or not config.ALERT_AUDIO_ENABLED:
            return False

        now = time.time()
        if now - self._last_sound_at < config.ALERT_COOLDOWN_S:
            return False

        self._last_sound_at = now
        try:
            if sys.platform == "win32":
                import winsound

                winsound.Beep(440, 1000)
            else:
                print("\a", end="", flush=True)
        except Exception as exc:
            print(f"Alert sound failed: {exc}")
            return False
        return True
