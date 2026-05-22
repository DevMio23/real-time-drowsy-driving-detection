"""Detection engine (no Qt dependencies)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from app import config


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


class DrowsinessDetectorEngine:
    """MediaPipe ROI extraction + dual YOLO inference + event logic."""

    def __init__(self) -> None:
        self.yawn_state = ""
        self.left_eye_state = ""
        self.right_eye_state = ""

        self.blinks = 0
        self.yawns = 0
        self.yawn_duration = 0.0
        self.eyes_closed_duration = 0.0
        self.blink_timestamps: list[float] = []

        self.left_eye_still_closed = False
        self.right_eye_still_closed = False
        self.yawn_in_progress = False
        self.alert_active = False

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1,
        )
        self.points_ids = config.LANDMARK_IDS

        self.detectyawn = YOLO(str(config.YAWN_MODEL_PATH))
        self.detecteye = YOLO(str(config.EYE_MODEL_PATH))

        self._frame_time = 1.0 / config.TARGET_FPS

    def close(self) -> None:
        self.face_mesh.close()

    def reset_state(self) -> None:
        self.yawn_state = ""
        self.left_eye_state = ""
        self.right_eye_state = ""
        self.blinks = 0
        self.yawns = 0
        self.yawn_duration = 0.0
        self.eyes_closed_duration = 0.0
        self.blink_timestamps = []
        self.left_eye_still_closed = False
        self.right_eye_still_closed = False
        self.yawn_in_progress = False
        self.alert_active = False

    def predict_eye(self, eye_frame: np.ndarray, eye_state: str) -> str:
        try:
            results_eye = self.detecteye.predict(eye_frame, verbose=False)
            boxes = results_eye[0].boxes
            if len(boxes) == 0:
                return eye_state

            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_idx = int(np.argmax(confidences))
            class_id = int(class_ids[max_idx])
            confidence = float(confidences[max_idx])

            if class_id == 1 and confidence > 0.7:
                return "Closed"
            if class_id == 0 and confidence > 0.7:
                return "Open"
            return eye_state
        except Exception:
            return eye_state

    def predict_yawn(self, yawn_frame: np.ndarray | None) -> None:
        if yawn_frame is None or yawn_frame.size == 0:
            self.yawn_state = "No Yawn"
            return

        try:
            results_yawn = self.detectyawn.predict(yawn_frame, verbose=False)
            boxes = results_yawn[0].boxes
            if len(boxes) == 0:
                self.yawn_state = "No Yawn"
                return

            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_idx = int(np.argmax(confidences))
            class_id = int(class_ids[max_idx])
            confidence = float(confidences[max_idx])

            if class_id == 0 and confidence > 0.7:
                self.yawn_state = "Yawn"
            elif class_id == 1 and confidence > 0.6:
                self.yawn_state = "No Yawn"
            else:
                self.yawn_state = "No Yawn"
        except Exception as exc:
            print(f"Yawn prediction error: {exc}")
            self.yawn_state = "No Yawn"

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, DetectionMetrics]:
        """Run detection on one BGR frame; returns annotated frame + metrics."""
        frame_time = self._frame_time
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        annotation_points: list[tuple[int, int]] = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
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

                if rois_valid:
                    mouth_roi = frame[y_mouth_min:y_mouth_max, x_mouth_min:x_mouth_max]
                    right_eye_roi = frame[y_reye_min:y_reye_max, x_reye_min:x_reye_max]
                    left_eye_roi = frame[y_leye_min:y_leye_max, x_leye_min:x_leye_max]

                    self.left_eye_state = self.predict_eye(left_eye_roi, self.left_eye_state)
                    self.right_eye_state = self.predict_eye(right_eye_roi, self.right_eye_state)
                    self.predict_yawn(mouth_roi)

                    both_eyes_closed = (
                        self.left_eye_state == "Closed"
                        and self.right_eye_state == "Closed"
                    )

                    if both_eyes_closed:
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

                    if self.yawn_state == "Yawn":
                        if not self.yawn_in_progress:
                            self.yawn_in_progress = True
                            self.yawns += 1
                        self.yawn_duration += frame_time
                    else:
                        if self.yawn_in_progress:
                            self.yawn_in_progress = False
                        self.yawn_duration = 0.0

                    annotation_points = mouth_points + right_eye_points + left_eye_points

        metrics = self._build_metrics()
        self._draw_annotations(frame, annotation_points, metrics.alert_active)
        return frame, metrics

    def _build_metrics(self) -> DetectionMetrics:
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
        )

    @staticmethod
    def _draw_annotations(
        frame: np.ndarray,
        points: list[tuple[int, int]],
        alert_active: bool,
    ) -> None:
        for x, y in points:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        if alert_active:
            cv2.putText(
                frame,
                "ALERT",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
