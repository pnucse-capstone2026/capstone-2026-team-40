from __future__ import annotations

import csv
import io
from collections import Counter


def read_csv_rows(data: bytes) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig")
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


def calculate_classification_metrics(rows: list[dict[str, str]]) -> dict:
    required = {"true_label", "predicted_label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CSV에는 true_label, predicted_label 열이 필요합니다.")

    pairs = [
        (str(row["true_label"]).strip(), str(row["predicted_label"]).strip())
        for row in rows
        if str(row.get("true_label", "")).strip()
        and str(row.get("predicted_label", "")).strip()
    ]
    if not pairs:
        raise ValueError("평가할 분류 결과가 없습니다.")

    labels = sorted({label for pair in pairs for label in pair})
    confusion = {
        true_label: {predicted_label: 0 for predicted_label in labels}
        for true_label in labels
    }
    for true_label, predicted_label in pairs:
        confusion[true_label][predicted_label] += 1

    per_class = {}
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[other][label] for other in labels if other != label
        )
        false_negative = sum(
            confusion[label][other] for other in labels if other != label
        )
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }

    correct = sum(1 for true_label, predicted_label in pairs if true_label == predicted_label)
    return {
        "samples": len(pairs),
        "accuracy": correct / len(pairs),
        "macro_precision": sum(item["precision"] for item in per_class.values()) / len(labels),
        "macro_recall": sum(item["recall"] for item in per_class.values()) / len(labels),
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(labels),
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def calculate_tracking_metrics(rows: list[dict[str, str]]) -> dict:
    required = {"true_count", "predicted_count"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CSV에는 true_count, predicted_count 열이 필요합니다.")

    samples = []
    for row in rows:
        try:
            true_count = int(row["true_count"])
            predicted_count = int(row["predicted_count"])
            id_switches = int(row.get("id_switches", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("객체 수와 ID 변경 횟수는 정수여야 합니다.") from exc
        samples.append((true_count, predicted_count, id_switches))
    if not samples:
        raise ValueError("평가할 영상 추적 결과가 없습니다.")

    absolute_errors = [abs(true - predicted) for true, predicted, _ in samples]
    exact = sum(1 for error in absolute_errors if error == 0)
    error_distribution = Counter(absolute_errors)
    return {
        "videos": len(samples),
        "count_mae": sum(absolute_errors) / len(samples),
        "exact_count_rate": exact / len(samples),
        "total_id_switches": sum(item[2] for item in samples),
        "error_distribution": dict(sorted(error_distribution.items())),
    }
