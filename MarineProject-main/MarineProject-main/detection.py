"""YOLOv5 탐지 → Dynamic Crop → ViT 분류 파이프라인."""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    ALLOW_DEMO_MODE,
    FINE_CLASSES,
    INFERENCE_MAX_DIMENSION,
    MAX_DETECTIONS_PER_FRAME,
    NMS_IOU_THRESHOLD,
    ROBOFLOW_API_URL,
    ROBOFLOW_DETECTION_MODEL_ID,
    SHIP_SIZE_CLASSES,
    VIT_MODEL_PATH,
    YOLO_CONF_THRESHOLD,
    YOLO_MODEL_PATH,
)
from roboflow_inference import get_inference_client

SIZE_COLORS = {
    "대형": "#FF69B4",
    "중형": "#38BDF8",
    "소형": "#34D399",
}


# ============================================================
# Roboflow Object Detection
# ============================================================
# Roboflow에서 나오는 클래스 이름을 현재 앱에서 쓰는 이름으로 정규화.
SIZE_LABEL_ALIASES = {
    "big": "대형",
    "large": "대형",
    "대형": "대형",
    "medium": "중형",
    "middle": "중형",
    "mid": "중형",
    "중형": "중형",
    "small": "소형",
    "소형": "소형",
}

OBJECT_LABEL_ALIASES = {
    "ship": "ship",
    "boat": "ship",
    "vessel": "ship",
    "선박": "ship",
    "함정": "ship",
    "drone": "drone",
    "uav": "drone",
    "드론": "drone",
    "person": "person",
    "human": "person",
    "사람": "person",
}


@dataclass
class ShipPrediction:
    x: int
    y: int
    width: int
    height: int
    detection_confidence: float
    object_class: str
    size_class: str
    fine_class: str
    classification_confidence: float
    crop_base64: str | None = None
    track_id: int | None = None

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def label(self) -> str:
        if self.object_class != "ship":
            return self.object_class
        if self.fine_class in {"ship", "선박", "함정"}:
            return "함정"
        return self.fine_class.replace(",", "")


@dataclass
class PipelineResult:
    predictions: list[ShipPrediction] = field(default_factory=list)
    mode: str = "demo"
    pipeline_steps: list[str] = field(
        default_factory=lambda: [
            "input",
            "yolov5_detection",
            "dynamic_crop",
            "vit_classification",
            "visualization",
        ]
    )


_yolo_model = None
_vit_model = None
_vit_processor = None
_yolo_load_attempted = False
_vit_load_attempted = False
def _run_roboflow_detection_model(image: Image.Image) -> dict | list | None:
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        client = get_inference_client(
            ROBOFLOW_API_URL,
            api_key,
            confidence_threshold=YOLO_CONF_THRESHOLD,
        )
        return client.infer(image, model_id=ROBOFLOW_DETECTION_MODEL_ID)
    except Exception as exc:
        raise RuntimeError(f"Roboflow 객체 탐지 모델 호출 실패: {exc}") from exc


def _collect_roboflow_prediction_dicts(payload) -> list[dict]:
    """
    모델 응답 구조가 달라도 x/y/width/height/confidence를 가진
    detection dict를 재귀적으로 찾아낸다.
    """
    found: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            required = {"x", "y", "width", "height", "confidence"}
            if required.issubset(node.keys()):
                found.append(node)
                return

            for value in node.values():
                walk(value)

        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(payload)
    return found


def _prepare_inference_image(
    image: Image.Image,
) -> tuple[Image.Image, float, float]:
    """추론 전송량을 줄이고 결과 좌표를 원본 크기로 되돌릴 비율을 반환한다."""
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= INFERENCE_MAX_DIMENSION:
        return image, 1.0, 1.0

    resize_ratio = INFERENCE_MAX_DIMENSION / longest_side
    resized_width = max(1, int(round(width * resize_ratio)))
    resized_height = max(1, int(round(height * resize_ratio)))
    resized = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )
    return resized, width / resized_width, height / resized_height


def _box_iou(
    first: tuple[int, int, int, int, float, str | None],
    second: tuple[int, int, int, int, float, str | None],
) -> float:
    first_x2 = first[0] + first[2]
    first_y2 = first[1] + first[3]
    second_x2 = second[0] + second[2]
    second_y2 = second[1] + second[3]

    intersection_width = max(0, min(first_x2, second_x2) - max(first[0], second[0]))
    intersection_height = max(0, min(first_y2, second_y2) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0

    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / max(union, 1)


def _filter_detection_boxes(
    boxes: list[tuple[int, int, int, int, float, str | None]],
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int, float, str | None]]:
    """낮은 신뢰도·잘못된 좌표·중복 박스를 제거하고 프레임별 개수를 제한한다."""
    valid: list[tuple[int, int, int, int, float, str | None]] = []
    for x, y, width, height, confidence, label in boxes:
        if not math.isfinite(confidence) or confidence < YOLO_CONF_THRESHOLD:
            continue

        x = max(0, min(int(round(x)), image_width - 1))
        y = max(0, min(int(round(y)), image_height - 1))
        width = max(0, min(int(round(width)), image_width - x))
        height = max(0, min(int(round(height)), image_height - y))
        if width < 2 or height < 2:
            continue
        valid.append((x, y, width, height, float(confidence), label))

    selected: list[tuple[int, int, int, int, float, str | None]] = []
    for candidate in sorted(valid, key=lambda item: item[4], reverse=True):
        candidate_label = str(candidate[5] or "").strip().lower()
        duplicate = any(
            candidate_label == str(existing[5] or "").strip().lower()
            and _box_iou(candidate, existing) >= NMS_IOU_THRESHOLD
            for existing in selected
        )
        if duplicate:
            continue
        selected.append(candidate)
        if len(selected) >= MAX_DETECTIONS_PER_FRAME:
            break

    return selected


def _roboflow_detect(
    image: Image.Image,
) -> list[tuple[int, int, int, int, float, str | None]] | None:
    """
    Roboflow Serverless 모델로 객체 탐지.

    반환값:
    - None: API key가 설정되지 않아 Roboflow를 사용하지 않음
    - []: Roboflow 호출은 성공했지만 탐지 객체가 없음
    - list: (x1, y1, width, height, confidence, class_name)
    """
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        return None

    result = _run_roboflow_detection_model(image)
    if result is None:
        return None

    raw_predictions = _collect_roboflow_prediction_dicts(result)
    boxes: list[tuple[int, int, int, int, float, str | None]] = []

    for pred in raw_predictions:
        try:
            center_x = float(pred["x"])
            center_y = float(pred["y"])
            box_w = float(pred["width"])
            box_h = float(pred["height"])
            confidence = float(pred["confidence"])
        except (TypeError, ValueError, KeyError):
            continue

        if confidence < YOLO_CONF_THRESHOLD:
            continue

        # Roboflow object detection 좌표는 중심점 x/y + width/height 형식.
        x1 = int(round(center_x - box_w / 2))
        y1 = int(round(center_y - box_h / 2))
        width = max(1, int(round(box_w)))
        height = max(1, int(round(box_h)))

        class_name = (
            pred.get("class")
            or pred.get("class_name")
            or pred.get("label")
        )
        if class_name is not None:
            class_name = str(class_name)

        boxes.append((x1, y1, width, height, confidence, class_name))

    return boxes


def _normalize_model_label(
    yolo_label: str | None,
    task: str,
    image_width: int,
    image_height: int,
    box_width: int,
    box_height: int,
) -> tuple[str, str]:
    """모델 클래스명을 현재 앱의 object_class / size_class로 변환한다."""
    raw_label = str(yolo_label or "").strip()
    label_key = raw_label.lower()

    if label_key in SIZE_LABEL_ALIASES:
        return "ship", SIZE_LABEL_ALIASES[label_key]

    if label_key in OBJECT_LABEL_ALIASES:
        object_class = OBJECT_LABEL_ALIASES[label_key]
        size_class = _infer_size_class(
            image_width,
            image_height,
            box_width,
            box_height,
        )
        return object_class, size_class

    # 함정 탭의 단일 클래스 모델이라면 알 수 없는 클래스도 함정으로 취급.
    if task == "ship":
        return (
            "ship",
            _infer_size_class(
                image_width,
                image_height,
                box_width,
                box_height,
            ),
        )

    # 침투 객체 탭에서는 알 수 없는 클래스명을 그대로 유지한다.
    return (
        label_key or "unknown",
        _infer_size_class(
            image_width,
            image_height,
            box_width,
            box_height,
        ),
    )


def _load_yolo():
    global _yolo_model, _yolo_load_attempted
    if _yolo_load_attempted:
        return _yolo_model
    _yolo_load_attempted = True

    if not YOLO_MODEL_PATH.exists():
        return None

    try:
        from ultralytics import YOLO

        _yolo_model = YOLO(str(YOLO_MODEL_PATH))
        return _yolo_model
    except Exception:
        return None


def _load_vit():
    global _vit_model, _vit_processor, _vit_load_attempted
    if _vit_load_attempted:
        return _vit_model, _vit_processor
    _vit_load_attempted = True

    if not VIT_MODEL_PATH.exists():
        return None, None

    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch

        _vit_processor = AutoImageProcessor.from_pretrained(str(VIT_MODEL_PATH))
        _vit_model = AutoModelForImageClassification.from_pretrained(str(VIT_MODEL_PATH))
        _vit_model.eval()
        return _vit_model, _vit_processor
    except Exception:
        return None, None


def read_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def read_video_frame(data: bytes, suffix: str = ".mp4") -> tuple[Image.Image, int]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(data)
        temp_path = temp_file.name

    try:
        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError("영상 파일을 열 수 없습니다.")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        success, frame = capture.read()
        capture.release()

        if not success or frame is None:
            raise ValueError("영상에서 프레임을 읽을 수 없습니다.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb), frame_count
    finally:
        os.remove(temp_path)


def _infer_size_class(width: int, height: int, box_w: int, box_h: int) -> str:
    area_ratio = (box_w * box_h) / max(width * height, 1)
    if area_ratio >= 0.28:
        return "대형"
    if area_ratio >= 0.08:
        return "중형"
    return "소형"


def _demo_boxes(width: int, height: int) -> list[tuple[int, int, int, int, float]]:
    ship_w = int(width * 0.72)
    ship_h = int(height * 0.52)
    ship_x = int((width - ship_w) / 2)
    ship_y = int(height * 0.28)

    medium_w = int(width * 0.22)
    medium_h = int(height * 0.14)
    small_w = int(width * 0.10)
    small_h = int(height * 0.08)

    return [
        (ship_x, ship_y, ship_w, ship_h, 0.93),
        (int(width * 0.08), int(height * 0.62), medium_w, medium_h, 0.86),
        (int(width * 0.78), int(height * 0.68), small_w, small_h, 0.79),
    ]


def _yolo_detect(image: Image.Image) -> list[tuple[int, int, int, int, float, str | None]]:
    model = _load_yolo()
    if model is None:
        return []

    results = model.predict(
        source=np.array(image),
        conf=YOLO_CONF_THRESHOLD,
        verbose=False,
    )
    boxes: list[tuple[int, int, int, int, float, str | None]] = []
    for result in results:
        names = result.names or {}
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id)
            boxes.append((x1, y1, x2 - x1, y2 - y1, conf, cls_name))
    return boxes


def _demo_classify(crop: Image.Image, size_class: str, index: int) -> tuple[str, float]:
    options = FINE_CLASSES[size_class]
    fine_class = options[index % len(options)]
    return fine_class, round(0.82 + (index % 3) * 0.04, 2)


def _vit_classify(crop: Image.Image, size_class: str, index: int) -> tuple[str, float]:
    model, processor = _load_vit()
    if model is None or processor is None:
        return _demo_classify(crop, size_class, index)

    try:
        import torch

        inputs = processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1)[0]
            cls_id = int(probs.argmax())
            confidence = float(probs[cls_id])
            label = model.config.id2label.get(cls_id, str(cls_id))
            return label, round(confidence, 2)
    except Exception:
        return _demo_classify(crop, size_class, index)


def _encode_crop(crop: Image.Image, max_size: int = 96) -> str:
    import base64

    thumb = crop.copy()
    thumb.thumbnail((max_size, max_size))
    buffer = io.BytesIO()
    thumb.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def run_pipeline(image: Image.Image, task: str = "detection") -> PipelineResult:
    width, height = image.size
    inference_image, scale_x, scale_y = _prepare_inference_image(image)

    # 1순위: Roboflow Serverless 모델
    roboflow_boxes = _roboflow_detect(inference_image)

    if roboflow_boxes is not None:
        detection_boxes = [
            (
                int(round(x * scale_x)),
                int(round(y * scale_y)),
                int(round(box_width * scale_x)),
                int(round(box_height * scale_y)),
                confidence,
                label,
            )
            for x, y, box_width, box_height, confidence, label in roboflow_boxes
        ]
        mode = "roboflow"
        pipeline_steps = ["input", "roboflow_model_detection", "visualization"]
    else:
        # API key가 설정되지 않은 경우에만 기존 로컬 YOLO를 시도.
        local_model = _load_yolo()
        if local_model is not None:
            detection_boxes = _yolo_detect(inference_image)
            detection_boxes = [
                (
                    int(round(x * scale_x)),
                    int(round(y * scale_y)),
                    int(round(box_width * scale_x)),
                    int(round(box_height * scale_y)),
                    confidence,
                    label,
                )
                for x, y, box_width, box_height, confidence, label in detection_boxes
            ]
            mode = "local-yolo"
            pipeline_steps = ["input", "local_yolo_detection", "visualization"]
        else:
            if not ALLOW_DEMO_MODE:
                raise RuntimeError(
                    "실제 객체 탐지 모델을 사용할 수 없습니다. "
                    "Streamlit Secrets의 ROBOFLOW_API_KEY 또는 로컬 YOLO 모델을 확인해 주세요. "
                    "가상 결과가 필요한 개발 환경에서만 ALLOW_DEMO_MODE=true를 설정할 수 있습니다."
                )

            # 명시적으로 허용된 개발 환경에서만 데모 결과를 사용한다.
            raw_boxes = [
                (x, y, w, h, c)
                for x, y, w, h, c in _demo_boxes(width, height)
            ]
            demo_labels = (
                ["ship", "drone", "person"]
                if task == "detection"
                else [None] * len(raw_boxes)
            )
            detection_boxes = [
                (x, y, w, h, c, demo_labels[index % len(demo_labels)])
                for index, (x, y, w, h, c) in enumerate(raw_boxes)
            ]
            mode = "demo"
            pipeline_steps = ["input", "demo_detection", "visualization"]

    detection_boxes = _filter_detection_boxes(detection_boxes, width, height)

    predictions: list[ShipPrediction] = []
    for index, (x, y, w, h, det_conf, model_label) in enumerate(detection_boxes):
        x = max(0, min(int(x), width - 1))
        y = max(0, min(int(y), height - 1))
        w = max(1, min(int(w), width - x))
        h = max(1, min(int(h), height - y))

        object_class, size_class = _normalize_model_label(
            model_label,
            task,
            width,
            height,
            w,
            h,
        )

        # Detection 단계에서는 객체 탐지만 수행한다. 사용자가 선택한 crop의
        # 톤급 분류는 classification.py가 별도로 담당한다.
        fine_class = object_class
        cls_conf = 0.0

        predictions.append(
            ShipPrediction(
                x=x,
                y=y,
                width=w,
                height=h,
                detection_confidence=round(float(det_conf), 2),
                object_class=object_class,
                size_class=size_class,
                fine_class=fine_class,
                classification_confidence=round(float(cls_conf), 2),
                crop_base64=None,
            )
        )

    return PipelineResult(
        predictions=predictions,
        mode=mode,
        pipeline_steps=pipeline_steps,
    )

@lru_cache(maxsize=8)
def _get_label_font(font_size: int):
    font_candidates = (
        "C:/Windows/Fonts/malgunbd.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "NotoSansCJK-Bold.ttc",
        "DejaVuSans-Bold.ttf",
        "arialbd.ttf",
    )
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default(size=font_size)


def draw_predictions(image: Image.Image, predictions: list[ShipPrediction]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    shortest_side = min(image.size)
    font_size = max(18, min(40, int(shortest_side * 0.035)))
    line_width = max(3, min(8, int(shortest_side * 0.006)))
    label_font = _get_label_font(font_size)

    for pred in predictions:
        color = SIZE_COLORS.get(pred.size_class, "#FF69B4")
        box = (pred.x, pred.y, pred.x2, pred.y2)
        draw.rectangle(box, outline=color, width=line_width)

        label = pred.label
        if pred.track_id is not None:
            label = f"{label} · ID {pred.track_id:02d}"
        text_bbox = draw.textbbox((0, 0), label, font=label_font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        pad = max(6, int(font_size * 0.3))
        label_y = max(pred.y - text_h - pad * 2, 0)
        label_x = max(0, min(pred.x, image.width - text_w - pad * 2))
        label_box = (
            label_x,
            label_y,
            label_x + text_w + pad * 2,
            label_y + text_h + pad * 2,
        )
        draw.rectangle(label_box, fill=color)
        draw.text(
            (label_box[0] + pad, label_box[1] + pad - 1),
            label,
            fill="white",
            font=label_font,
            stroke_width=1,
            stroke_fill="#14202d",
        )

    return output


def summarize_predictions(predictions: list[ShipPrediction]) -> dict:
    size_counts = {name: 0 for name in SHIP_SIZE_CLASSES}
    object_counts = {"ship": 0, "drone": 0, "person": 0}
    for pred in predictions:
        if pred.object_class == "ship":
            size_counts[pred.size_class] = size_counts.get(pred.size_class, 0) + 1
        object_counts[pred.object_class] = object_counts.get(pred.object_class, 0) + 1

    det_confs = [p.detection_confidence for p in predictions]
    cls_confs = [p.classification_confidence for p in predictions]
    avg_det = round(sum(det_confs) / len(det_confs), 2) if det_confs else 0.0
    avg_cls = round(sum(cls_confs) / len(cls_confs), 2) if cls_confs else 0.0

    return {
        "total": len(predictions),
        "size_counts": size_counts,
        "object_counts": object_counts,
        "avg_detection_confidence": avg_det,
        "avg_classification_confidence": avg_cls,
    }


def pipeline_to_json(
    pipeline_result: PipelineResult,
    image_size: tuple[int, int],
    source_name: str,
    frame_index: int | None = None,
) -> dict:
    width, height = image_size
    items = []
    for pred in pipeline_result.predictions:
        items.append(
            {
                "track_id": pred.track_id,
                "bbox": {
                    "x": pred.x,
                    "y": pred.y,
                    "width": pred.width,
                    "height": pred.height,
                },
                "detection": {
                    "object_class": pred.object_class,
                    "size_class": pred.size_class,
                    "confidence": pred.detection_confidence,
                },
                "classification": {
                    "fine_class": pred.fine_class,
                    "confidence": pred.classification_confidence,
                },
            }
        )

    payload = {
        "pipeline": pipeline_result.pipeline_steps,
        "mode": pipeline_result.mode,
        "image": {"width": width, "height": height},
        "predictions": items,
        "source": source_name,
    }
    if frame_index is not None:
        payload["frame_index"] = frame_index
    return payload


def format_json_output(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def get_model_status() -> dict:
    return {
        "roboflow_configured": bool(os.getenv("ROBOFLOW_API_KEY", "").strip()),
        "roboflow_model_id": ROBOFLOW_DETECTION_MODEL_ID,
        "demo_mode_allowed": ALLOW_DEMO_MODE,
        "yolo_loaded": YOLO_MODEL_PATH.exists() and _load_yolo() is not None,
        "vit_loaded": VIT_MODEL_PATH.exists() and _load_vit()[0] is not None,
        "yolo_path": str(YOLO_MODEL_PATH),
        "vit_path": str(VIT_MODEL_PATH),
    }
