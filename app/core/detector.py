"""Detection engine (no Qt dependencies)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from app import config
from app.core.kalman_smoother import ScalarKalmanFilter


@dataclass
class DetectionMetrics:
    """Snapshot of drowsiness metrics for the UI."""

    alert_active: bool = False
    alert_html: str = ""
    recent_blinks: int = 0
    eyes_closed_duration: float = 0.0
    yawn_duration: float = 0.0
    total_yawns: int = 0
    left_eye_state: str = ""
    right_eye_state: str = ""
    yawn_state: str = ""
    process_fps: float = 0.0
    yolo_ran: bool = False
    left_eye_score: float = 0.0
    right_eye_score: float = 0.0
    yawn_score: float = 0.0
    yawn_suppressed: bool = False


@dataclass
class _CachedRois:
    mouth: np.ndarray | None = None
    right_eye: np.ndarray | None = None
    left_eye: np.ndarray | None = None
    annotation_points: list[tuple[int, int]] = field(default_factory=list)
    valid: bool = False


@dataclass
class _PendingState:
    target: str = "Open"
    count: int = 0


class DrowsinessDetectorEngine:
    """MediaPipe ROI extraction + dual YOLO inference + event logic."""

    def __init__(self) -> None:
        self.yawn_state = "No Yawn"
        self.left_eye_state = "Open"
        self.right_eye_state = "Open"

        self.blinks = 0
        self.yawns = 0
        self.yawn_duration = 0.0
        self.eyes_closed_duration = 0.0
        self.blink_timestamps: list[float] = []

        self.left_eye_still_closed = False
        self.right_eye_still_closed = False
        self.yawn_in_progress = False
        self.yawn_candidate_duration = 0.0

        self._frame_index = 0
        self._cached_rois = _CachedRois()
        self._last_process_fps = 0.0
        self._fps_window_start = time.perf_counter()
        self._fps_frame_count = 0

        self._left_kalman = ScalarKalmanFilter(initial=0.0)
        self._right_kalman = ScalarKalmanFilter(initial=0.0)
        self._yawn_kalman = ScalarKalmanFilter(initial=0.0)

        self._left_pending = _PendingState(target="Open")
        self._right_pending = _PendingState(target="Open")
        self._yawn_pending = _PendingState(target="No Yawn")

        self.show_debug_overlay = config.SHOW_DEBUG_OVERLAY

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.points_ids = config.LANDMARK_IDS

        self.detectyawn = YOLO(str(config.YAWN_MODEL_PATH))
        self.detecteye = YOLO(str(config.EYE_MODEL_PATH))

        self._default_frame_time = 1.0 / config.TARGET_PROCESS_FPS

    def close(self) -> None:
        self.face_mesh.close()

    def reset_state(self) -> None:
        self.yawn_state = "No Yawn"
        self.left_eye_state = "Open"
        self.right_eye_state = "Open"
        self.blinks = 0
        self.yawns = 0
        self.yawn_duration = 0.0
        self.eyes_closed_duration = 0.0
        self.blink_timestamps = []
        self.left_eye_still_closed = False
        self.right_eye_still_closed = False
        self.yawn_in_progress = False
        self.yawn_candidate_duration = 0.0
        self._frame_index = 0
        self._cached_rois = _CachedRois()
        self._fps_window_start = time.perf_counter()
        self._fps_frame_count = 0
        self._left_kalman.reset(0.0)
        self._right_kalman.reset(0.0)
        self._yawn_kalman.reset(0.0)
        self._left_pending = _PendingState(target="Open")
        self._right_pending = _PendingState(target="Open")
        self._yawn_pending = _PendingState(target="No Yawn")

    @staticmethod
    def _eye_closed_measurement(class_id: int, confidence: float) -> float | None:
        if class_id == 1 and confidence >= config.EYE_CONFIDENCE:
            return confidence
        if class_id == 0 and confidence >= config.EYE_CONFIDENCE:
            return 1.0 - confidence
        return None

    @staticmethod
    def _yawn_measurement(class_id: int, confidence: float) -> float | None:
        if class_id == 0 and confidence >= config.YAWN_CONFIDENCE:
            return confidence
        if class_id == 1 and confidence >= config.NO_YAWN_CONFIDENCE:
            return 1.0 - confidence
        return None

    def _measure_eye(self, eye_frame: np.ndarray) -> float | None:
        try:
            results_eye = self.detecteye.predict(
                eye_frame, verbose=False, imgsz=config.YOLO_IMGSZ
            )
            boxes = results_eye[0].boxes
            if len(boxes) == 0:
                return None
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_idx = int(np.argmax(confidences))
            return self._eye_closed_measurement(
                int(class_ids[max_idx]), float(confidences[max_idx])
            )
        except Exception:
            return None

    def _measure_yawn(self, yawn_frame: np.ndarray | None) -> float | None:
        if yawn_frame is None or yawn_frame.size == 0:
            return 0.0
        try:
            results_yawn = self.detectyawn.predict(
                yawn_frame, verbose=False, imgsz=config.YOLO_IMGSZ
            )
            boxes = results_yawn[0].boxes
            if len(boxes) == 0:
                return 0.0
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_idx = int(np.argmax(confidences))
            measured = self._yawn_measurement(
                int(class_ids[max_idx]), float(confidences[max_idx])
            )
            return measured if measured is not None else None
        except Exception as exc:
            print(f"Yawn prediction error: {exc}")
            return None

    @staticmethod
    def _score_to_binary_state(
        score: float,
        current: str,
        high: float,
        low: float,
        closed_label: str,
        open_label: str,
    ) -> str:
        if current == closed_label:
            return closed_label if score > low else open_label
        return closed_label if score > high else open_label

    def _apply_pending_state(
        self,
        pending: _PendingState,
        proposed: str,
        current: str,
    ) -> str:
        if proposed == pending.target:
            pending.count += 1
        else:
            pending.target = proposed
            pending.count = 1
        if pending.count >= config.STATE_CONFIRM_FRAMES:
            return pending.target
        return current

    def _update_eye_states_from_scores(self) -> None:
        left_score = self._left_kalman.x
        right_score = self._right_kalman.x

        left_proposed = self._score_to_binary_state(
            left_score,
            self.left_eye_state,
            config.EYE_CLOSED_SCORE_HIGH,
            config.EYE_CLOSED_SCORE_LOW,
            "Closed",
            "Open",
        )
        right_proposed = self._score_to_binary_state(
            right_score,
            self.right_eye_state,
            config.EYE_CLOSED_SCORE_HIGH,
            config.EYE_CLOSED_SCORE_LOW,
            "Closed",
            "Open",
        )

        self.left_eye_state = self._apply_pending_state(
            self._left_pending, left_proposed, self.left_eye_state
        )
        self.right_eye_state = self._apply_pending_state(
            self._right_pending, right_proposed, self.right_eye_state
        )

    def _update_yawn_state_from_score(self) -> None:
        yawn_score = self._yawn_kalman.x
        proposed = self._score_to_binary_state(
            yawn_score,
            self.yawn_state,
            config.YAWN_SCORE_HIGH,
            config.YAWN_SCORE_LOW,
            "Yawn",
            "No Yawn",
        )
        self.yawn_state = self._apply_pending_state(
            self._yawn_pending, proposed, self.yawn_state
        )

    def _extract_rois(
        self, frame: np.ndarray, face_landmarks
    ) -> _CachedRois:
        ih, iw, _ = frame.shape
        mouth_points = []
        for i in range(4):
            lm = face_landmarks.landmark[self.points_ids[i]]
            mouth_points.append((int(lm.x * iw), int(lm.y * ih)))

        right_eye_points = []
        for i in range(4, 6):
            lm = face_landmarks.landmark[self.points_ids[i]]
            right_eye_points.append((int(lm.x * iw), int(lm.y * ih)))

        left_eye_points = []
        for i in range(6, 8):
            lm = face_landmarks.landmark[self.points_ids[i]]
            left_eye_points.append((int(lm.x * iw), int(lm.y * ih)))

        all_mouth_x = [p[0] for p in mouth_points]
        all_mouth_y = [p[1] for p in mouth_points]
        x_mouth_min = max(0, min(all_mouth_x))
        x_mouth_max = min(iw, max(all_mouth_x))
        y_mouth_min = max(0, min(all_mouth_y))
        y_mouth_max = min(ih, max(all_mouth_y))

        all_reye_x = [p[0] for p in right_eye_points]
        all_reye_y = [p[1] for p in right_eye_points]
        x_reye_min = max(0, min(all_reye_x))
        x_reye_max = min(iw, max(all_reye_x))
        y_reye_min = max(0, min(all_reye_y))
        y_reye_max = min(ih, max(all_reye_y))

        all_leye_x = [p[0] for p in left_eye_points]
        all_leye_y = [p[1] for p in left_eye_points]
        x_leye_min = max(0, min(all_leye_x))
        x_leye_max = min(iw, max(all_leye_x))
        y_leye_min = max(0, min(all_leye_y))
        y_leye_max = min(ih, max(all_leye_y))

        rois_valid = (
            x_mouth_max > x_mouth_min
            and y_mouth_max > y_mouth_min
            and x_reye_max > x_reye_min
            and y_reye_max > y_reye_min
            and x_leye_max > x_leye_min
            and y_leye_max > y_leye_min
        )

        cached = _CachedRois(
            annotation_points=mouth_points + right_eye_points + left_eye_points,
            valid=rois_valid,
        )
        if rois_valid:
            cached.mouth = frame[y_mouth_min:y_mouth_max, x_mouth_min:x_mouth_max]
            cached.right_eye = frame[y_reye_min:y_reye_max, x_reye_min:x_reye_max]
            cached.left_eye = frame[y_leye_min:y_leye_max, x_leye_min:x_leye_max]
        return cached

    def _both_eyes_closed(self) -> bool:
        return self.left_eye_state == "Closed" and self.right_eye_state == "Closed"

    def _update_event_logic(self, frame_time: float) -> bool:
        """Update blink/microsleep/yawn counters. Returns True if yawn was suppressed."""
        both_closed = self._both_eyes_closed()
        yawn_suppressed = False

        if both_closed:
            if not (self.left_eye_still_closed and self.right_eye_still_closed):
                self.left_eye_still_closed = True
                self.right_eye_still_closed = True
                self.blinks += 1
                self.blink_timestamps.append(time.time())
            self.eyes_closed_duration += frame_time
        else:
            if self.left_eye_still_closed and self.right_eye_still_closed:
                self.left_eye_still_closed = False
                self.right_eye_still_closed = False
            self.eyes_closed_duration = 0.0

        if config.SUPPRESS_YAWN_WHEN_EYES_CLOSED and both_closed:
            self.yawn_candidate_duration = 0.0
            if self.yawn_in_progress:
                self.yawn_in_progress = False
            self.yawn_duration = 0.0
            return True

        if self.yawn_state == "Yawn":
            self.yawn_candidate_duration += frame_time
            if (
                not self.yawn_in_progress
                and self.yawn_candidate_duration >= config.YAWN_MIN_DURATION_S
            ):
                self.yawn_in_progress = True
                self.yawns += 1
            if self.yawn_in_progress:
                self.yawn_duration += frame_time
        else:
            self.yawn_candidate_duration = 0.0
            if self.yawn_in_progress:
                self.yawn_in_progress = False
            self.yawn_duration = 0.0

        return yawn_suppressed

    def _tick_fps_counter(self) -> float:
        self._fps_frame_count += 1
        elapsed = time.perf_counter() - self._fps_window_start
        if elapsed >= 1.0:
            self._last_process_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_window_start = time.perf_counter()
        return self._last_process_fps

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        delta_s: float | None = None,
    ) -> tuple[np.ndarray, DetectionMetrics]:
        self._frame_index += 1
        frame_time = (
            delta_s
            if delta_s is not None and delta_s > 0
            else self._default_frame_time
        )

        run_mediapipe = (self._frame_index % config.MEDIAPIPE_STRIDE) == 0
        run_yolo = (self._frame_index % config.YOLO_INFERENCE_STRIDE) == 0

        if run_mediapipe:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                self._cached_rois = self._extract_rois(
                    frame, results.multi_face_landmarks[0]
                )
            else:
                self._cached_rois = _CachedRois()

        rois = self._cached_rois
        if run_yolo and rois.valid:
            if rois.left_eye is not None:
                left_m = self._measure_eye(rois.left_eye)
                if left_m is not None:
                    self._left_kalman.update(left_m)
            if rois.right_eye is not None:
                right_m = self._measure_eye(rois.right_eye)
                if right_m is not None:
                    self._right_kalman.update(right_m)
            yawn_m = self._measure_yawn(rois.mouth)
            if yawn_m is not None:
                self._yawn_kalman.update(yawn_m)

            self._update_eye_states_from_scores()
            self._update_yawn_state_from_score()

        yawn_suppressed = False
        if rois.valid:
            yawn_suppressed = self._update_event_logic(frame_time)

        process_fps = self._tick_fps_counter()
        metrics = self._build_metrics(
            process_fps=process_fps,
            yolo_ran=run_yolo,
            yawn_suppressed=yawn_suppressed,
        )
        self._draw_annotations(frame, rois.annotation_points, metrics)
        return frame, metrics

    def _build_metrics(
        self,
        *,
        process_fps: float,
        yolo_ran: bool,
        yawn_suppressed: bool,
    ) -> DetectionMetrics:
        current_time = time.time()
        self.blink_timestamps = [
            t for t in self.blink_timestamps
            if current_time - t <= config.BLINK_WINDOW
        ]
        recent_blinks = len(self.blink_timestamps)

        alert_html = (
            "<p style='color: green; font-weight: bold;'>Status: Driver Alert</p>"
        )
        alert_active = False

        if (
            self.yawn_duration >= config.YAWN_THRESHOLD
            or self.eyes_closed_duration >= config.MICROSLEEP_THRESHOLD
            or recent_blinks >= config.BLINK_THRESHOLD
        ):
            alert_active = True
            if self.yawn_duration >= config.YAWN_THRESHOLD:
                alert_html = (
                    "<p style='color: red; font-weight: bold;'>"
                    "ALERT: Prolonged Yawn Detected</p>"
                )
            elif self.eyes_closed_duration >= config.MICROSLEEP_THRESHOLD:
                alert_html = (
                    "<p style='color: red; font-weight: bold;'>"
                    "ALERT: Microsleep Detected</p>"
                )
            elif recent_blinks >= config.BLINK_THRESHOLD:
                alert_html = (
                    "<p style='color: red; font-weight: bold;'>"
                    "ALERT: Excessive Blinking</p>"
                )

        return DetectionMetrics(
            alert_active=alert_active,
            alert_html=alert_html,
            recent_blinks=recent_blinks,
            eyes_closed_duration=self.eyes_closed_duration,
            yawn_duration=self.yawn_duration,
            total_yawns=self.yawns,
            left_eye_state=self.left_eye_state,
            right_eye_state=self.right_eye_state,
            yawn_state=self.yawn_state,
            process_fps=process_fps,
            yolo_ran=yolo_ran,
            left_eye_score=self._left_kalman.x,
            right_eye_score=self._right_kalman.x,
            yawn_score=self._yawn_kalman.x,
            yawn_suppressed=yawn_suppressed,
        )

    def _draw_annotations(
        self,
        frame: np.ndarray,
        points: list[tuple[int, int]],
        metrics: DetectionMetrics,
    ) -> None:
        for x, y in points:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        if metrics.alert_active:
            cv2.putText(
                frame,
                "ALERT",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        if not self.show_debug_overlay:
            return

        y = 60
        lines = [
            f"L:{metrics.left_eye_state} ({metrics.left_eye_score:.2f})",
            f"R:{metrics.right_eye_state} ({metrics.right_eye_score:.2f})",
            f"Y:{metrics.yawn_state} ({metrics.yawn_score:.2f})",
        ]
        if metrics.yawn_suppressed:
            lines.append("Yawn suppressed (eyes closed)")
        for line in lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
            )
            y += 22
