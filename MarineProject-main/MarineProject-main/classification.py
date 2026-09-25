from __future__ import annotations

import os
from PIL import Image

from config import ROBOFLOW_API_URL, ROBOFLOW_CLASSIFICATION_MODEL_ID
from roboflow_inference import get_inference_client


def _run_roboflow_model(image: Image.Image) -> dict | list:
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY 환경변수가 설정되지 않았습니다."
        )

    try:
        client = get_inference_client(ROBOFLOW_API_URL, api_key)
        return client.infer(image, model_id=ROBOFLOW_CLASSIFICATION_MODEL_ID)
    except Exception as exc:
        raise RuntimeError(
            f"Roboflow 분류 모델 호출 실패: {exc}"
        ) from exc


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_classification_result(payload):
    """
    Roboflow 모델 결과에서 가장 가능성이 높은
    classification label / confidence를 재귀적으로 찾는다.
    """

    candidates: list[tuple[str, float]] = []

    def walk(node):
        if isinstance(node, dict):
            # 흔한 classification 출력 형태:
            # {"top": "...", "confidence": 0.95}
            top = node.get("top")
            confidence = _to_float(node.get("confidence"))
            if top is not None and confidence is not None:
                candidates.append((str(top), confidence))

            # 다른 출력 형태:
            # {"predicted_class": "...", "confidence": 0.95}
            predicted_class = node.get("predicted_class")
            if predicted_class is None:
                predicted_class = node.get("predicted_label")

            confidence = _to_float(node.get("confidence"))
            if predicted_class is not None and confidence is not None:
                candidates.append((str(predicted_class), confidence))

            # predictions 내부의 {"class": "...", "confidence": ...}
            # Object Detection bbox dict와 혼동하지 않도록 bbox 필드가 없는 경우만 사용.
            class_name = node.get("class")
            confidence = _to_float(node.get("confidence"))
            has_bbox = all(
                key in node
                for key in ("x", "y", "width", "height")
            )
            if class_name is not None and confidence is not None and not has_bbox:
                candidates.append((str(class_name), confidence))

            for value in node.values():
                walk(value)

        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(payload)

    if not candidates:
        return None

    return max(candidates, key=lambda item: item[1])


def classify_ship(image: Image.Image) -> tuple[str, float]:
    """
    함정 crop 1장을 Roboflow Classification 모델에 전달하고
    (분류 클래스, confidence)를 반환한다.
    """
    result = _run_roboflow_model(image)
    parsed = _find_classification_result(result)

    if parsed is None:
        raise RuntimeError(
            "Classification 결과에서 class/confidence를 찾지 못했습니다. "
            f"Roboflow 반환값: {result}"
        )

    class_name, confidence = parsed

    return class_name, round(confidence, 4)


def get_classification_status() -> dict:
    return {
        "configured": bool(
            os.getenv("ROBOFLOW_API_KEY", "").strip()
        ),
        "model_id": ROBOFLOW_CLASSIFICATION_MODEL_ID,
    }
