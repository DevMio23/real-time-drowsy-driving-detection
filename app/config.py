"""Application configuration (thresholds and paths)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EYE_MODEL_PATH = PROJECT_ROOT / "runs" / "detecteye" / "train" / "weights" / "best.pt"
YAWN_MODEL_PATH = PROJECT_ROOT / "runs" / "detectyawn" / "train" / "weights" / "best.pt"

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Phase 1 — CPU tuning
FRAME_QUEUE_SIZE = 1
CAPTURE_FPS = 15
TARGET_PROCESS_FPS = 12
PROCESS_QUEUE_TIMEOUT_S = 0.1
THREAD_JOIN_TIMEOUT_S = 2.0

YOLO_IMGSZ = 224
YOLO_INFERENCE_STRIDE = 1
MEDIAPIPE_STRIDE = 1

# Phase 2 — accuracy (sensitivity-first; Kalman optional)
USE_KALMAN_SMOOTHING = False
USE_STATE_CONFIRMATION = False
EYE_CONFIDENCE = 0.55
YAWN_CONFIDENCE = 0.55
NO_YAWN_CONFIDENCE = 0.50
YAWN_MIN_DURATION_S = 0.15
SUPPRESS_YAWN_COUNT_WHEN_EYES_CLOSED = True
YAWN_SUPPRESS_MIN_EYES_CLOSED_S = 0.25

YAWN_THRESHOLD = 2.0
MICROSLEEP_THRESHOLD = 1.2
BLINK_THRESHOLD = 10
BLINK_WINDOW = 30

SHOW_DEBUG_OVERLAY = False

# Phase 3 — session logging
LOGS_DIR = PROJECT_ROOT / "logs" / "sessions"
TREND_WINDOW_MINUTES = 10

# Blink detection — EAR from MediaPipe (more reliable than YOLO on tiny eye ROIs)
USE_EAR_FOR_BLINKS = True
MEDIAPIPE_REFINE_LANDMARKS = True
# Fixed threshold for microsleep (sustained closure)
EAR_MICROSLEEP_THRESHOLD = 0.23
# Adaptive blink — ignores small head-movement EAR dips
EAR_BLINK_ABS_MAX = 0.22
EAR_DROP_ABSOLUTE = 0.07
EAR_DROP_RATIO = 0.78
EAR_BASELINE_ALPHA = 0.08
EAR_SYMMETRY_MAX_DIFF = 0.12
BLINK_MIN_DURATION_S = 0.04
BLINK_MAX_DURATION_S = 0.50
BLINK_REFRACTORY_S = 0.22
# Six landmarks per eye (MediaPipe face mesh indices)
RIGHT_EYE_EAR_IDS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR_IDS = [362, 385, 387, 263, 373, 380]

LANDMARK_IDS = [187, 411, 152, 61, 68, 174, 399, 298]

PAGE_WELCOME = 0
PAGE_DETECTION = 1
