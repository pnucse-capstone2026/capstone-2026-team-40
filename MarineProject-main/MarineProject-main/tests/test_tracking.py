import unittest
from types import SimpleNamespace

from tracking import (
    finalize_video_tracker,
    new_video_tracker,
    project_predictions,
    update_video_tracker,
)


def prediction(x, y=90, confidence=0.8):
    return SimpleNamespace(
        x=x,
        y=y,
        width=26,
        height=16,
        detection_confidence=confidence,
        object_class="drone",
        size_class="소형",
        track_id=None,
    )


class VideoTrackingTests(unittest.TestCase):
    def test_fast_drone_keeps_one_track(self):
        tracker = new_video_tracker(1280, 720)
        assigned_ids = []

        for x in [1120, 950, 700, 400, 160]:
            item = prediction(x)
            update_video_tracker(tracker, [item])
            assigned_ids.append(item.track_id)

        result = finalize_video_tracker(tracker)
        self.assertEqual(assigned_ids, [1, 1, 1, 1, 1])
        self.assertEqual(result["object_counts"]["drone"], 1)

    def test_two_crossing_drones_remain_distinct(self):
        tracker = new_video_tracker(1280, 720)
        sequence = [
            [(100, 100), (900, 150)],
            [(200, 100), (800, 150)],
            [(350, 100), (650, 150)],
            [(520, 100), (480, 150)],
            [(700, 100), (300, 150)],
        ]
        assigned_ids = []

        for frame in sequence:
            items = [prediction(x, y) for x, y in frame]
            update_video_tracker(tracker, items)
            assigned_ids.append([item.track_id for item in items])

        result = finalize_video_tracker(tracker)
        self.assertEqual(assigned_ids, [[1, 2]] * len(sequence))
        self.assertEqual(result["object_counts"]["drone"], 2)

    def test_short_detection_gap_does_not_create_duplicate(self):
        tracker = new_video_tracker(1280, 720)
        first = prediction(1100)
        second = prediction(930)
        update_video_tracker(tracker, [first])
        update_video_tracker(tracker, [second])
        update_video_tracker(tracker, [])
        update_video_tracker(tracker, [])
        returning = prediction(420)
        update_video_tracker(tracker, [returning])

        self.assertEqual(returning.track_id, first.track_id)
        self.assertEqual(finalize_video_tracker(tracker)["total"], 1)

    def test_projection_moves_rendered_box_without_mutating_detection(self):
        tracker = new_video_tracker(1280, 720)
        first = prediction(100)
        second = prediction(200)
        update_video_tracker(tracker, [first])
        update_video_tracker(tracker, [second])

        rendered = project_predictions([second], tracker, 0.5)

        self.assertEqual(second.x, 200)
        self.assertEqual(rendered[0].x, 250)
        self.assertEqual(rendered[0].track_id, second.track_id)

    def test_summary_uses_all_observation_confidences(self):
        tracker = new_video_tracker(1280, 720)
        update_video_tracker(tracker, [prediction(100, confidence=0.6)])
        update_video_tracker(tracker, [prediction(150, confidence=0.8)])

        result = finalize_video_tracker(tracker)

        self.assertEqual(result["avg_detection_confidence"], 0.7)
        self.assertEqual(result["tracking_diagnostics"]["confirmed_tracks"], 1)


if __name__ == "__main__":
    unittest.main()
