from __future__ import annotations

import copy
import math
from typing import Any


Box = tuple[float, float, float, float]


def new_video_tracker(image_width: int, image_height: int) -> dict[str, Any]:
    return {
        "tracks": {},
        "next_track_id": 1,
        "step": 0,
        "image_width": image_width,
        "image_height": image_height,
        "image_diagonal": math.hypot(image_width, image_height),
    }


def bbox_center(box: Box) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def bbox_iou(first: Box, second: Box) -> float:
    intersection_width = max(
        0.0,
        min(first[0] + first[2], second[0] + second[2]) - max(first[0], second[0]),
    )
    intersection_height = max(
        0.0,
        min(first[1] + first[3], second[1] + second[3]) - max(first[1], second[1]),
    )
    intersection = intersection_width * intersection_height
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / max(union, 1.0)


def box_size_similarity(first: Box, second: Box) -> float:
    first_area = max(first[2] * first[3], 1.0)
    second_area = max(second[2] * second[3], 1.0)
    return min(first_area, second_area) / max(first_area, second_area)


def _track_motion_limit(
    tracker: dict[str, Any],
    track: dict[str, Any],
    pred_box: Box,
    gap: int,
) -> float:
    object_class = track["object_class"]
    class_diagonal_ratio = {
        "drone": 0.12,
        "person": 0.045,
        "ship": 0.035,
    }.get(object_class, 0.04)
    size_multiplier = {
        "drone": 8.0,
        "person": 4.0,
        "ship": 3.0,
    }.get(object_class, 3.5)
    last_box = track["last_bbox"]
    object_extent = max(
        pred_box[2],
        pred_box[3],
        last_box[2],
        last_box[3],
    )
    base_limit = max(
        tracker["image_diagonal"] * class_diagonal_ratio,
        object_extent * size_multiplier,
    )
    velocity_x, velocity_y = track["velocity"]
    predicted_motion = math.hypot(velocity_x, velocity_y) * gap
    adaptive_limit = max(
        base_limit * min(gap, 3),
        predicted_motion * 1.75 + object_extent * 2.0,
    )
    return min(adaptive_limit, tracker["image_diagonal"] * 0.28)


def update_video_tracker(tracker: dict[str, Any], predictions: list) -> None:
    """Connect detections across sampled frames and assign stable track IDs."""
    tracker["step"] += 1
    current_step = tracker["step"]
    tracks = tracker["tracks"]
    active_track_ids = [
        track_id
        for track_id, track in tracks.items()
        if current_step - track["last_seen"] <= 8
    ]

    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        pred_box = (prediction.x, prediction.y, prediction.width, prediction.height)
        pred_center = bbox_center(pred_box)
        for track_id in active_track_ids:
            track = tracks[track_id]
            if track["object_class"] != prediction.object_class:
                continue

            track_box = track["last_bbox"]
            gap = max(1, current_step - track["last_seen"])
            track_center = track["last_center"]
            velocity_x, velocity_y = track["velocity"]
            predicted_center = (
                track_center[0] + velocity_x * gap,
                track_center[1] + velocity_y * gap,
            )
            predicted_distance = math.hypot(
                pred_center[0] - predicted_center[0],
                pred_center[1] - predicted_center[1],
            )
            last_distance = math.hypot(
                pred_center[0] - track_center[0],
                pred_center[1] - track_center[1],
            )
            iou = bbox_iou(pred_box, track_box)
            size_similarity = box_size_similarity(pred_box, track_box)
            motion_limit = _track_motion_limit(tracker, track, pred_box, gap)
            association_distance = (
                min(predicted_distance, last_distance)
                if track["hits"] < 3
                else predicted_distance
            )
            if iou >= 0.03 or association_distance <= motion_limit:
                distance_score = max(0.0, 1 - association_distance / motion_limit)
                age_penalty = max(0, gap - 1) * 0.08
                score = (
                    iou * 2.0
                    + distance_score * 1.5
                    + size_similarity * 0.35
                    - age_penalty
                )
                candidates.append((score, prediction_index, track_id))

    matched_predictions = set()
    matched_tracks = set()
    for _, prediction_index, track_id in sorted(candidates, reverse=True):
        if prediction_index in matched_predictions or track_id in matched_tracks:
            continue
        prediction = predictions[prediction_index]
        track = tracks[track_id]
        prediction.track_id = track_id
        new_box = (
            prediction.x,
            prediction.y,
            prediction.width,
            prediction.height,
        )
        new_center = bbox_center(new_box)
        gap = max(1, current_step - track["last_seen"])
        observed_velocity = (
            (new_center[0] - track["last_center"][0]) / gap,
            (new_center[1] - track["last_center"][1]) / gap,
        )
        if track["hits"] == 1:
            track["velocity"] = observed_velocity
        else:
            velocity_weight = 0.68 if prediction.object_class == "drone" else 0.55
            track["velocity"] = (
                track["velocity"][0] * (1 - velocity_weight)
                + observed_velocity[0] * velocity_weight,
                track["velocity"][1] * (1 - velocity_weight)
                + observed_velocity[1] * velocity_weight,
            )
        track["last_bbox"] = new_box
        track["last_center"] = new_center
        track["last_seen"] = current_step
        track["hits"] += 1
        track["confidence_sum"] += prediction.detection_confidence
        track["confidence_samples"] += 1
        track["best_confidence"] = max(
            track["best_confidence"],
            prediction.detection_confidence,
        )
        size_votes = track["size_votes"]
        size_votes[prediction.size_class] = size_votes.get(prediction.size_class, 0) + 1
        matched_predictions.add(prediction_index)
        matched_tracks.add(track_id)

    for prediction_index, prediction in enumerate(predictions):
        if prediction_index in matched_predictions:
            continue
        track_id = tracker["next_track_id"]
        tracker["next_track_id"] += 1
        prediction.track_id = track_id
        new_box = (
            prediction.x,
            prediction.y,
            prediction.width,
            prediction.height,
        )
        new_center = bbox_center(new_box)
        tracks[track_id] = {
            "track_id": track_id,
            "object_class": prediction.object_class,
            "first_seen": current_step,
            "first_center": new_center,
            "last_bbox": new_box,
            "last_center": new_center,
            "velocity": (0.0, 0.0),
            "last_seen": current_step,
            "hits": 1,
            "confidence_sum": prediction.detection_confidence,
            "confidence_samples": 1,
            "best_confidence": prediction.detection_confidence,
            "size_votes": {prediction.size_class: 1},
        }


def project_predictions(
    predictions: list,
    tracker: dict[str, Any],
    frame_fraction: float,
) -> list:
    """Project tracked boxes between inference frames without changing counts."""
    fraction = max(0.0, min(float(frame_fraction), 1.0))
    if fraction <= 0:
        return predictions

    image_width = tracker["image_width"]
    image_height = tracker["image_height"]
    projected = []
    for prediction in predictions:
        rendered = copy.copy(prediction)
        track = tracker["tracks"].get(prediction.track_id)
        if track is not None:
            velocity_x, velocity_y = track["velocity"]
            rendered.x = max(
                0,
                min(int(round(prediction.x + velocity_x * fraction)), image_width - prediction.width),
            )
            rendered.y = max(
                0,
                min(int(round(prediction.y + velocity_y * fraction)), image_height - prediction.height),
            )
        projected.append(rendered)
    return projected


def finalize_video_tracker(tracker: dict[str, Any]) -> dict[str, Any]:
    all_tracks = list(tracker["tracks"].values())
    confirmed_tracks = [track for track in all_tracks if track["hits"] >= 2]
    tentative_tracks = [track for track in all_tracks if track["hits"] < 2]
    counted_tracks = confirmed_tracks or all_tracks
    size_counts = {"대형": 0, "중형": 0, "소형": 0}
    object_counts = {"ship": 0, "drone": 0, "person": 0}
    for track in counted_tracks:
        object_class = track["object_class"]
        object_counts[object_class] = object_counts.get(object_class, 0) + 1
        if object_class == "ship":
            size_class = max(track["size_votes"], key=track["size_votes"].get)
            size_counts[size_class] = size_counts.get(size_class, 0) + 1

    total = len(counted_tracks)
    confidence_sum = sum(track["confidence_sum"] for track in counted_tracks)
    confidence_samples = sum(track["confidence_samples"] for track in counted_tracks)
    mean_track_hits = (
        sum(track["hits"] for track in counted_tracks) / total
        if total
        else 0.0
    )
    track_records = [
        {
            "track_id": track["track_id"],
            "object_class": track["object_class"],
            "first_analyzed_frame": track["first_seen"],
            "last_analyzed_frame": track["last_seen"],
            "observations": track["hits"],
            "mean_confidence": round(
                track["confidence_sum"] / max(track["confidence_samples"], 1),
                4,
            ),
            "confirmed": track["hits"] >= 2,
        }
        for track in counted_tracks
    ]
    return {
        "total": total,
        "size_counts": size_counts,
        "object_counts": object_counts,
        "avg_detection_confidence": (
            round(confidence_sum / confidence_samples, 2)
            if confidence_samples
            else 0.0
        ),
        "avg_classification_confidence": 0.0,
        "count_label": "고유 객체 수",
        "breakdown_label": "객체별 고유 개수",
        "tracking_diagnostics": {
            "analyzed_steps": tracker["step"],
            "confirmed_tracks": len(confirmed_tracks),
            "tentative_tracks": len(tentative_tracks),
            "mean_observations_per_track": round(mean_track_hits, 2),
        },
        "tracks": track_records,
    }
