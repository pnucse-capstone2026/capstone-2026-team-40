import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from detection import PipelineResult
import video_processing


class VideoProcessingTests(unittest.TestCase):
    def test_short_video_is_encoded_and_reports_progress(self):
        source_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as source_file:
                source_path = source_file.name
            writer = cv2.VideoWriter(
                source_path,
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (320, 180),
            )
            self.assertTrue(writer.isOpened())
            for index in range(12):
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                cv2.circle(frame, (30 + index * 10, 70), 8, (255, 255, 255), -1)
                writer.write(frame)
            writer.release()

            with open(source_path, "rb") as source_file:
                video_bytes = source_file.read()

            progress_updates = []
            with patch.object(
                video_processing,
                "run_pipeline",
                return_value=PipelineResult(predictions=[], mode="test"),
            ):
                result = video_processing.process_video_bytes(
                    video_bytes,
                    ".avi",
                    "detection",
                    _progress_callback=lambda progress, current, total: progress_updates.append(
                        (progress, current, total)
                    ),
                )

            self.assertGreater(len(result[0]), 100)
            self.assertEqual(result[4], 12)
            self.assertEqual(result[6]["total"], 0)
            self.assertEqual(progress_updates[-1][0], 1.0)
        finally:
            if source_path and os.path.exists(source_path):
                os.remove(source_path)


if __name__ == "__main__":
    unittest.main()
