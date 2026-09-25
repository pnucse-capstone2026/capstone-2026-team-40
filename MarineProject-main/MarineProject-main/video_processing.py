from __future__ import annotations

import math
import os
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

from config import (
    MAX_VIDEO_SECONDS,
    MAX_VIDEO_SOURCE_DIMENSION,
    VIDEO_INFERENCE_FPS,
    VIDEO_MAX_DIMENSION,
    VIDEO_MAX_INFERENCE_CALLS,
    VIDEO_OUTPUT_FPS,
)
from detection import draw_predictions, run_pipeline
from tracking import (
    finalize_video_tracker,
    new_video_tracker,
    project_predictions,
    update_video_tracker,
)


def process_video_bytes(data: bytes, suffix: str, task: str, _progress_callback=None):
    input_path = output_path = None
    capture = writer = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as input_file:
            input_file.write(data)
            input_path = input_file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_file:
            output_path = output_file.name

        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise ValueError("영상 파일을 열 수 없습니다.")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = 24.0
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = frame_count / source_fps if frame_count > 0 else 0.0
        if duration_seconds > MAX_VIDEO_SECONDS:
            raise ValueError(
                f"영상 길이는 최대 {MAX_VIDEO_SECONDS}초까지 지원합니다. "
                f"현재 영상은 약 {duration_seconds:.1f}초입니다."
            )
        if max(source_width, source_height) > MAX_VIDEO_SOURCE_DIMENSION:
            raise ValueError(
                f"영상의 긴 변은 최대 {MAX_VIDEO_SOURCE_DIMENSION}px까지 지원합니다. "
                f"현재 해상도는 {source_width} x {source_height}입니다."
            )
        if _progress_callback is not None:
            _progress_callback(0.0, 0, frame_count)

        output_scale = min(
            1.0,
            VIDEO_MAX_DIMENSION / max(source_width, source_height, 1),
        )
        width = max(2, int(round(source_width * output_scale / 2)) * 2)
        height = max(2, int(round(source_height * output_scale / 2)) * 2)
        output_stride = max(1, int(round(source_fps / min(source_fps, VIDEO_OUTPUT_FPS))))
        output_fps = source_fps / output_stride
        inference_stride = max(1, int(round(source_fps / VIDEO_INFERENCE_FPS)))
        if frame_count > 0:
            inference_stride = max(
                inference_stride,
                math.ceil(frame_count / VIDEO_MAX_INFERENCE_CALLS),
            )

        writer = imageio_ffmpeg.write_frames(
            output_path,
            (width, height),
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=output_fps,
            quality=6,
            codec="libx264",
            macro_block_size=2,
            ffmpeg_log_level="error",
            output_params=["-preset", "veryfast", "-movflags", "+faststart"],
        )
        writer.send(None)

        fallback_image = fallback_result = None
        preview_image = preview_result = None
        preview_frame_index = 0
        latest_result = None
        latest_predictions = []
        video_tracker = new_video_tracker(width, height)
        source_frame_index = 0
        next_inference_frame = 0
        analyzed_frames = 0
        written_frames = 0
        last_inference_frame = 0
        progress_interval = max(1, frame_count // 60) if frame_count > 0 else 30

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if source_frame_index % output_stride == 0:
                if (frame.shape[1], frame.shape[0]) != (width, height):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if latest_result is None or source_frame_index >= next_inference_frame:
                    pipeline_result = run_pipeline(image, task=task)
                    latest_result = pipeline_result
                    latest_predictions = pipeline_result.predictions
                    update_video_tracker(video_tracker, latest_predictions)
                    last_inference_frame = source_frame_index
                    analyzed_frames += 1
                    next_inference_frame = source_frame_index + inference_stride
                    if fallback_result is None:
                        fallback_image = image
                        fallback_result = pipeline_result
                    if preview_result is None and latest_predictions:
                        preview_image = image
                        preview_result = pipeline_result
                        preview_frame_index = source_frame_index

                frame_fraction = (
                    (source_frame_index - last_inference_frame) / max(inference_stride, 1)
                )
                display_predictions = project_predictions(
                    latest_predictions,
                    video_tracker,
                    frame_fraction,
                )
                annotated = draw_predictions(image, display_predictions)
                annotated_array = np.ascontiguousarray(annotated, dtype=np.uint8)
                writer.send(annotated_array)
                written_frames += 1

            source_frame_index += 1
            if (
                _progress_callback is not None
                and source_frame_index % progress_interval == 0
            ):
                progress = (
                    min(source_frame_index / frame_count, 1.0)
                    if frame_count > 0
                    else 0.0
                )
                _progress_callback(progress, source_frame_index, frame_count)

        capture.release()
        capture = None
        writer.close()
        writer = None
        if fallback_image is None or fallback_result is None or written_frames == 0:
            raise ValueError("영상에 처리할 프레임이 없습니다.")
        if _progress_callback is not None:
            _progress_callback(1.0, frame_count, frame_count)

        if preview_image is None or preview_result is None:
            preview_image = fallback_image
            preview_result = fallback_result
            preview_frame_index = 0

        with open(output_path, "rb") as result_file:
            video_bytes = result_file.read()
        return (
            video_bytes,
            preview_image,
            preview_result,
            preview_result.predictions,
            frame_count,
            analyzed_frames,
            finalize_video_tracker(video_tracker),
            output_fps,
            preview_frame_index,
        )
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.close()
        for path in (input_path, output_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    pass
