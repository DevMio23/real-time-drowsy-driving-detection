"""Application configuration (thresholds and paths)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EYE_MODEL_PATH = PROJECT_ROOT / "runs" / "detecteye" / "train" / "weights" / "best.pt"
YAWN_MODEL_PATH = PROJECT_ROOT / "runs" / "detectyawn" / "train" / "weights" / "best.pt"

CAMERA_INDEX = 0
FRAME_QUEUE_SIZE = 2
CAPTURE_SLEEP_S = 0.01
PROCESS_QUEUE_TIMEOUT_S = 0.1
THREAD_JOIN_TIMEOUT_S = 2.0
TARGET_FPS = 30

YAWN_THRESHOLD = 2.0
MICROSLEEP_THRESHOLD = 1.2
BLINK_THRESHOLD = 10
BLINK_WINDOW = 30

LANDMARK_IDS = [187, 411, 152, 61, 68, 174, 399, 298]

PAGE_WELCOME = 0
PAGE_DETECTION = 1
