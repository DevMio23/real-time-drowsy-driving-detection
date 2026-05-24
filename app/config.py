"""Application configuration (thresholds and paths)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EYE_MODEL_PATH = PROJECT_ROOT / "runs" / "detecteye" / "train" / "weights" / "best.pt"
YAWN_MODEL_PATH = PROJECT_ROOT / "runs" / "detectyawn" / "train" / "weights" / "best.pt"

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Phase 1 — CPU tuning (Ryzen 5650U class laptops)
FRAME_QUEUE_SIZE = 1  # keep only the latest frame
CAPTURE_FPS = 15
TARGET_PROCESS_FPS = 12
PROCESS_QUEUE_TIMEOUT_S = 0.1
THREAD_JOIN_TIMEOUT_S = 2.0

YOLO_IMGSZ = 160
YOLO_INFERENCE_STRIDE = 2  # run eye+yawn models every Nth processed frames
MEDIAPIPE_STRIDE = 1  # landmark pass every Nth frame (1 = every processed frame)

# Phase 2 — detection accuracy
EYE_CONFIDENCE = 0.7
YAWN_CONFIDENCE = 0.7
NO_YAWN_CONFIDENCE = 0.6
EYE_CLOSED_SCORE_HIGH = 0.62
EYE_CLOSED_SCORE_LOW = 0.38
YAWN_SCORE_HIGH = 0.62
YAWN_SCORE_LOW = 0.38
STATE_CONFIRM_FRAMES = 3
YAWN_MIN_DURATION_S = 0.5
SUPPRESS_YAWN_WHEN_EYES_CLOSED = True

YAWN_THRESHOLD = 2.0
MICROSLEEP_THRESHOLD = 1.2
BLINK_THRESHOLD = 10
BLINK_WINDOW = 30

SHOW_DEBUG_OVERLAY = False

LANDMARK_IDS = [187, 411, 152, 61, 68, 174, 399, 298]

PAGE_WELCOME = 0
PAGE_DETECTION = 1
