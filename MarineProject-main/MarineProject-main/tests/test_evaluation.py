import unittest

from evaluation import calculate_classification_metrics, calculate_tracking_metrics


class EvaluationMetricTests(unittest.TestCase):
    def test_classification_metrics(self):
        rows = [
            {"true_label": "50톤급", "predicted_label": "50톤급"},
            {"true_label": "50톤급", "predicted_label": "100톤급"},
            {"true_label": "100톤급", "predicted_label": "100톤급"},
            {"true_label": "100톤급", "predicted_label": "100톤급"},
        ]

        result = calculate_classification_metrics(rows)

        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], 0.7333333333)

    def test_tracking_metrics(self):
        rows = [
            {"true_count": "1", "predicted_count": "1", "id_switches": "0"},
            {"true_count": "2", "predicted_count": "3", "id_switches": "1"},
            {"true_count": "2", "predicted_count": "2", "id_switches": "0"},
        ]

        result = calculate_tracking_metrics(rows)

        self.assertEqual(result["videos"], 3)
        self.assertAlmostEqual(result["count_mae"], 1 / 3)
        self.assertAlmostEqual(result["exact_count_rate"], 2 / 3)
        self.assertEqual(result["total_id_switches"], 1)


if __name__ == "__main__":
    unittest.main()
