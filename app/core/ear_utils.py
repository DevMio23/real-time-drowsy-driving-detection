"""Eye Aspect Ratio (EAR) helpers for reliable blink detection."""

from __future__ import annotations

import time

import numpy as np

from app import config


def compute_ear(
    face_landmarks,
    indices: list[int],
    image_width: int,
    image_height: int,
) -> float:
    """
    Compute Eye Aspect Ratio from six MediaPipe face-mesh landmarks.

    Open ~0.25–0.35, closed ~0.10–0.20.
    """
    points = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        points.append(np.array([lm.x * image_width, lm.y * image_height]))

    if len(points) != 6:
        return 1.0

    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 1.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def ear_to_eye_state(ear: float, closed_threshold: float) -> str:
    return "Closed" if ear < closed_threshold else "Open"


class EarBlinkTracker:
    """
    Adaptive EAR blink counter — ignores small head-motion dips.

    Uses a personal open-eye baseline and requires a sharp bilateral drop
    plus a quick close→open cycle before counting one blink.
    """

    def __init__(self) -> None:
        self.left_baseline = 0.30
        self.right_baseline = 0.30
        self._phase = "open"
        self._closure_start: float | None = None
        self._last_blink_time = 0.0

    def reset(self) -> None:
        self.left_baseline = 0.30
        self.right_baseline = 0.30
        self._phase = "open"
        self._closure_start = None
        self._last_blink_time = 0.0

    def _update_baselines(self, left_ear: float, right_ear: float) -> None:
        """Track personal open-eye EAR; baseline only rises when eyes are open."""
        alpha = config.EAR_BASELINE_ALPHA
        if left_ear >= self.left_baseline - 0.015:
            self.left_baseline = (1 - alpha) * self.left_baseline + alpha * left_ear
        if right_ear >= self.right_baseline - 0.015:
            self.right_baseline = (1 - alpha) * self.right_baseline + alpha * right_ear

    def _is_blink_closure(self, left_ear: float, right_ear: float) -> bool:
        if abs(left_ear - right_ear) > config.EAR_SYMMETRY_MAX_DIFF:
            return False

        def eye_closed(ear: float, baseline: float) -> bool:
            if ear > config.EAR_BLINK_ABS_MAX:
                return False
            drop_ok = (baseline - ear) >= config.EAR_DROP_ABSOLUTE
            ratio_ok = ear <= baseline * config.EAR_DROP_RATIO
            return drop_ok and ratio_ok

        return eye_closed(left_ear, self.left_baseline) and eye_closed(
            right_ear, self.right_baseline
        )

    def update(self, left_ear: float, right_ear: float) -> bool:
        """
        Update tracker; returns True if a blink was counted this frame.
        """
        self._update_baselines(left_ear, right_ear)
        now = time.perf_counter()
        closed = self._is_blink_closure(left_ear, right_ear)
        blink_counted = False

        if closed:
            if self._phase == "open":
                self._phase = "closed"
                self._closure_start = now
            elif self._closure_start is not None:
                duration = now - self._closure_start
                if duration > config.BLINK_MAX_DURATION_S:
                    self._phase = "held"
        else:
            if self._phase == "closed" and self._closure_start is not None:
                duration = now - self._closure_start
                since_last = now - self._last_blink_time
                if (
                    config.BLINK_MIN_DURATION_S
                    <= duration
                    <= config.BLINK_MAX_DURATION_S
                    and since_last >= config.BLINK_REFRACTORY_S
                ):
                    blink_counted = True
                    self._last_blink_time = now
            self._phase = "open"
            self._closure_start = None

        return blink_counted

    def closure_state(self, left_ear: float, right_ear: float) -> tuple[str, str]:
        """Display states — uses adaptive closure test."""
        if self._is_blink_closure(left_ear, right_ear):
            return "Closed", "Closed"
        left = ear_to_eye_state(left_ear, config.EAR_MICROSLEEP_THRESHOLD)
        right = ear_to_eye_state(right_ear, config.EAR_MICROSLEEP_THRESHOLD)
        return left, right

    def both_eyes_closed_for_microsleep(
        self, left_ear: float, right_ear: float
    ) -> bool:
        """Sustained closure check (fixed threshold, not adaptive blink)."""
        return (
            left_ear < config.EAR_MICROSLEEP_THRESHOLD
            and right_ear < config.EAR_MICROSLEEP_THRESHOLD
        )
