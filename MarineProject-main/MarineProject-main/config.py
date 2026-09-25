import os
from pathlib import Path


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_text(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# 학습 완료 후 아래 경로에 모델 파일을 배치하세요.
YOLO_MODEL_PATH = MODELS_DIR / "yolov5_ship.pt"
VIT_MODEL_PATH = MODELS_DIR / "vit_ship_classifier"

# Roboflow Serverless Hosted API. Model IDs can be overridden from
# Streamlit Secrets without changing source code.
ROBOFLOW_API_URL = _env_text(
    "ROBOFLOW_API_URL",
    "https://serverless.roboflow.com",
).rstrip("/")
ROBOFLOW_DETECTION_MODEL_ID = _env_text(
    "ROBOFLOW_DETECTION_MODEL_ID",
    "first-object-detection-big/7",
)
ROBOFLOW_CLASSIFICATION_MODEL_ID = _env_text(
    "ROBOFLOW_CLASSIFICATION_MODEL_ID",
    "first-classification-big/12",
)

# 탐지 후처리. Streamlit Cloud에서는 환경변수로 조절할 수 있다.
YOLO_CONF_THRESHOLD = _env_float("DETECTION_CONFIDENCE", 0.30, 0.05, 0.95)
NMS_IOU_THRESHOLD = _env_float("DETECTION_NMS_IOU", 0.45, 0.05, 0.95)
MAX_DETECTIONS_PER_FRAME = _env_int("MAX_DETECTIONS_PER_FRAME", 40, 1, 300)

# 가상 결과는 명시적으로 허용한 개발 환경에서만 사용한다.
ALLOW_DEMO_MODE = _env_bool("ALLOW_DEMO_MODE", False)

# 영상 처리량 제한. 원본 재생 시간은 유지하면서 출력/추론 프레임만 줄인다.
VIDEO_INFERENCE_FPS = _env_float("VIDEO_INFERENCE_FPS", 4.0, 0.2, 15.0)
VIDEO_OUTPUT_FPS = _env_float("VIDEO_OUTPUT_FPS", 15.0, 1.0, 30.0)
VIDEO_MAX_INFERENCE_CALLS = _env_int("VIDEO_MAX_INFERENCE_CALLS", 240, 10, 1000)
VIDEO_MAX_DIMENSION = _env_int("VIDEO_MAX_DIMENSION", 1280, 480, 1920)
INFERENCE_MAX_DIMENSION = _env_int("INFERENCE_MAX_DIMENSION", 1280, 480, 1920)
INFERENCE_JPEG_QUALITY = _env_int("INFERENCE_JPEG_QUALITY", 82, 60, 95)
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 150, 10, 500)
MAX_VIDEO_SECONDS = _env_int("MAX_VIDEO_SECONDS", 180, 5, 1800)
MAX_VIDEO_SOURCE_DIMENSION = _env_int("MAX_VIDEO_SOURCE_DIMENSION", 3840, 720, 7680)

# 함정 크기 클래스
SHIP_SIZE_CLASSES = ("대형", "중형", "소형")

# ViT 세부 톤급 클래스 (데모 / 학습 레이블 예시)
FINE_CLASSES = {
    "대형": ("1,000톤급", "1,500톤급", "3,000톤급", "5,000톤급"),
    "중형": ("300톤급", "500톤급", "800톤급"),
    "소형": ("50톤급", "100톤급", "200톤급"),
}
