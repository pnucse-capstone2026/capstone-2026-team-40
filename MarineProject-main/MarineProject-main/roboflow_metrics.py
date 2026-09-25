import os
from functools import lru_cache

import requests

from config import ROBOFLOW_DETECTION_MODEL_ID


# ============================================================
# Roboflow Object Detection 프로젝트 설정
# ============================================================

ROBOFLOW_API_URL = "https://api.roboflow.com"

ROBOFLOW_WORKSPACE = "rladbswl"

try:
    ROBOFLOW_PROJECT, _roboflow_version_text = (
        ROBOFLOW_DETECTION_MODEL_ID.rsplit("/", 1)
    )
    ROBOFLOW_VERSION = int(_roboflow_version_text)
except (ValueError, TypeError) as exc:
    raise RuntimeError(
        "ROBOFLOW_DETECTION_MODEL_ID는 'project/version' 형식이어야 합니다."
    ) from exc


# ============================================================
# 숫자 변환
# ============================================================

def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# 클래스 이름 정리
# ============================================================

def _normalize_class_name(name):
    return str(name).strip().lower()


# ============================================================
# Roboflow Version 정보 가져오기
# ============================================================

@lru_cache(maxsize=1)
def get_version_info():

    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY 환경변수가 설정되지 않았습니다."
        )

    url = (
        f"{ROBOFLOW_API_URL}/"
        f"{ROBOFLOW_WORKSPACE}/"
        f"{ROBOFLOW_PROJECT}/"
        f"{ROBOFLOW_VERSION}"
    )

    try:
        response = requests.get(
            url,
            params={"api_key": api_key},
            timeout=20,
        )

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Roboflow API 연결 실패: {exc}"
        ) from exc

    if response.status_code == 401:
        raise RuntimeError(
            "Roboflow API Key 인증에 실패했습니다."
        )

    if response.status_code == 403:
        raise RuntimeError(
            "Roboflow 프로젝트 접근 권한이 없습니다."
        )

    if response.status_code == 404:
        raise RuntimeError(
            f"Roboflow 프로젝트 또는 Version을 찾지 못했습니다. "
            f"Project={ROBOFLOW_PROJECT}, "
            f"Version={ROBOFLOW_VERSION}"
        )

    try:
        response.raise_for_status()

    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Roboflow API 오류: "
            f"{response.status_code} "
            f"{response.text}"
        ) from exc

    try:
        return response.json()

    except ValueError as exc:
        raise RuntimeError(
            "Roboflow API 응답을 JSON으로 읽지 못했습니다."
        ) from exc


# ============================================================
# class_map 가져오기
# ============================================================

def _get_class_map():

    data = get_version_info()

    version = data.get("version", {})

    train = version.get("train", {})

    results = train.get("results", {})

    class_map = results.get("class_map", {})

    if not isinstance(class_map, dict) or not class_map:
        raise RuntimeError(
            "Roboflow Version 정보에서 "
            "train.results.class_map을 찾지 못했습니다."
        )

    return class_map


# ============================================================
# split의 rows 가져오기
# ============================================================

def _get_split_rows(class_map, split_name):

    rows = class_map.get(split_name, [])

    if not isinstance(rows, list):
        return []

    return [
        row
        for row in rows
        if isinstance(row, dict)
    ]


# ============================================================
# 클래스별 행이 존재하는지 확인
# ============================================================

def _has_class_rows(rows):

    for row in rows:

        class_name = _normalize_class_name(
            row.get("class", "")
        )

        if class_name and class_name != "all":
            return True

    return False


# ============================================================
# 평가 Split 결정
#
# 1순위 test
# 단, test에 all만 있고 클래스별 값이 없다면
# valid에 클래스별 값이 있는지 확인
# ============================================================

def _select_split(class_map):

    test_rows = _get_split_rows(
        class_map,
        "test",
    )

    valid_rows = _get_split_rows(
        class_map,
        "valid",
    )

    # test에 클래스별 결과까지 존재
    if test_rows and _has_class_rows(test_rows):
        return "test", test_rows

    # test에는 all만 있고
    # valid에 클래스별 결과가 있는 경우
    if valid_rows and _has_class_rows(valid_rows):
        return "valid", valid_rows

    # 클래스별 여부와 관계없이 test가 존재
    if test_rows:
        return "test", test_rows

    # test가 없으면 valid
    if valid_rows:
        return "valid", valid_rows

    raise RuntimeError(
        "Roboflow에서 test 또는 valid "
        "성능평가 결과를 찾지 못했습니다."
    )


# ============================================================
# 한 행에서 metric 추출
# ============================================================

def _parse_metric_row(row):

    return {
        "precision": _to_float(
            row.get("precision")
        ),
        "recall": _to_float(
            row.get("recall")
        ),
        "map50": _to_float(
            row.get("map50")
        ),
        "map50_95": _to_float(
            row.get("map95")
        ),
        "images": int(
            _to_float(
                row.get("images"),
                0,
            )
        ),
        "targets": int(
            _to_float(
                row.get("targets"),
                0,
            )
        ),
    }


# ============================================================
# 클래스 평균 계산
#
# Roboflow all 행이 없을 때만 fallback으로 사용
# ============================================================

def _calculate_macro_average(class_metrics):

    if not class_metrics:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
            "images": 0,
            "targets": 0,
        }

    values = list(
        class_metrics.values()
    )

    count = len(values)

    return {
        "precision": sum(
            item["precision"]
            for item in values
        ) / count,
        "recall": sum(
            item["recall"]
            for item in values
        ) / count,
        "map50": sum(
            item["map50"]
            for item in values
        ) / count,
        "map50_95": sum(
            item["map50_95"]
            for item in values
        ) / count,
        "images": 0,
        "targets": sum(
            item["targets"]
            for item in values
        ),
    }


# ============================================================
# Object Detection 전체 성능 가져오기
# ============================================================

def get_detection_metrics():

    class_map = _get_class_map()

    split_name, rows = _select_split(
        class_map
    )

    overall_metrics = None

    class_metrics = {}

    raw_class_names = []

    for row in rows:

        class_name_raw = str(
            row.get("class", "")
        ).strip()

        if not class_name_raw:
            continue

        class_name = _normalize_class_name(
            class_name_raw
        )

        raw_class_names.append(
            class_name_raw
        )

        metric = _parse_metric_row(
            row
        )

        # --------------------------------------------
        # 전체 모델 성능
        # --------------------------------------------

        if class_name == "all":

            overall_metrics = metric

            continue

        # --------------------------------------------
        # 클래스별 성능
        # --------------------------------------------

        class_metrics[class_name] = metric

    # Roboflow에 all 행이 없는 경우만
    # 클래스별 macro average 사용
    overall_source = "roboflow_all"

    if overall_metrics is None:

        overall_metrics = _calculate_macro_average(
            class_metrics
        )

        overall_source = "macro_average"

    return {
        "workspace": ROBOFLOW_WORKSPACE,
        "project": ROBOFLOW_PROJECT,
        "version": ROBOFLOW_VERSION,
        "split": split_name,
        "overall": overall_metrics,
        "overall_source": overall_source,
        "classes": class_metrics,
        "available_classes": list(
            class_metrics.keys()
        ),
        "raw_class_names": raw_class_names,
    }


# ============================================================
# 전체 모델 성능
# ============================================================

def get_overall_metrics():

    metrics = get_detection_metrics()

    return metrics["overall"]


# ============================================================
# 특정 클래스 성능
# ============================================================

def get_class_metrics(class_name):

    metrics = get_detection_metrics()

    classes = metrics.get(
        "classes",
        {}
    )

    target = _normalize_class_name(
        class_name
    )

    if target in classes:
        return classes[target]

    available = ", ".join(
        classes.keys()
    )

    raise RuntimeError(
        f"Roboflow에서 '{class_name}' 클래스의 "
        f"성능평가 결과를 찾지 못했습니다. "
        f"현재 확인된 클래스: {available or '없음'}"
    )


# ============================================================
# Streamlit 표시 이름 ↔ Roboflow 클래스
# ============================================================

DISPLAY_CLASS_MAP = {
    "대형 함정": "big",
    "중형 함정": "middle",
    "소형 함정": "small",
    "드론": "drone",
    "사람": "human",
}


# ============================================================
# Streamlit용 특정 클래스 성능
# ============================================================

def get_display_class_metrics(display_name):

    roboflow_class = DISPLAY_CLASS_MAP.get(
        display_name
    )

    if roboflow_class is None:
        raise RuntimeError(
            f"알 수 없는 표시 클래스입니다: "
            f"{display_name}"
        )

    return get_class_metrics(
        roboflow_class
    )


# ============================================================
# Streamlit용 전체 또는 클래스 선택
# ============================================================

def get_display_metrics(display_name):

    if display_name == "전체":

        return {
            "type": "overall",
            "class_name": "all",
            "metrics": get_overall_metrics(),
        }

    roboflow_class = DISPLAY_CLASS_MAP.get(
        display_name
    )

    if roboflow_class is None:
        raise RuntimeError(
            f"알 수 없는 평가 대상입니다: "
            f"{display_name}"
        )

    return {
        "type": "class",
        "class_name": roboflow_class,
        "metrics": get_class_metrics(
            roboflow_class
        ),
    }


# ============================================================
# 직접 실행 테스트
# ============================================================

if __name__ == "__main__":

    metrics = get_detection_metrics()

    print()

    print(
        "========================================"
    )

    print(
        "Roboflow Object Detection Metrics"
    )

    print(
        "========================================"
    )

    print(
        f"Workspace : "
        f"{metrics['workspace']}"
    )

    print(
        f"Project   : "
        f"{metrics['project']}"
    )

    print(
        f"Version   : "
        f"{metrics['version']}"
    )

    print(
        f"Split     : "
        f"{metrics['split']}"
    )

    print(
        f"Classes   : "
        f"{metrics['available_classes']}"
    )

    print()

    print(
        "----------------------------------------"
    )

    print(
        "Overall Model Performance"
    )

    print(
        "----------------------------------------"
    )

    overall = metrics["overall"]

    print(
        f"Precision      : "
        f"{overall['precision']:.4f}"
    )

    print(
        f"Recall         : "
        f"{overall['recall']:.4f}"
    )

    print(
        f"mAP@0.5        : "
        f"{overall['map50']:.4f}"
    )

    print(
        f"mAP@0.5:0.95   : "
        f"{overall['map50_95']:.4f}"
    )

    print(
        f"Source         : "
        f"{metrics['overall_source']}"
    )

    print()

    print(
        "----------------------------------------"
    )

    print(
        "Class Performance"
    )

    print(
        "----------------------------------------"
    )

    if not metrics["classes"]:

        print(
            "클래스별 성능 결과가 없습니다."
        )

    else:

        for class_name, values in metrics["classes"].items():

            print()

            print(
                f"[{class_name}]"
            )

            print(
                f"Precision      : "
                f"{values['precision']:.4f}"
            )

            print(
                f"Recall         : "
                f"{values['recall']:.4f}"
            )

            print(
                f"AP@0.5         : "
                f"{values['map50']:.4f}"
            )

            print(
                f"AP@0.5:0.95    : "
                f"{values['map50_95']:.4f}"
            )

            print(
                f"Images         : "
                f"{values['images']}"
            )

            print(
                f"Targets        : "
                f"{values['targets']}"
            )

    print()

    print(
        "========================================"
    )

    print(
        "Raw class names returned by Roboflow"
    )

    print(
        "========================================"
    )

    for name in metrics["raw_class_names"]:

        print(
            f"- {name}"
        )
