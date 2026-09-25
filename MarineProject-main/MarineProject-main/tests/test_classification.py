import os
import unittest
from unittest.mock import patch

from PIL import Image

import classification
from classification import _find_classification_result


class ClassificationResponseTests(unittest.TestCase):
    def test_serverless_inference_uses_new_classification_model(self):
        image = Image.new("RGB", (224, 224), "black")

        with (
            patch.dict(os.environ, {"ROBOFLOW_API_KEY": "test-key"}),
            patch.object(classification, "get_inference_client") as get_client,
        ):
            get_client.return_value.infer.return_value = {
                "predictions": [{"class": "50톤급", "confidence": 0.9}]
            }
            result = classification._run_roboflow_model(image)

        self.assertEqual(result["predictions"][0]["class"], "50톤급")
        get_client.assert_called_once_with(
            classification.ROBOFLOW_API_URL,
            "test-key",
        )
        get_client.return_value.infer.assert_called_once_with(
            image,
            model_id="first-classification-big/12",
        )

    def test_selects_highest_confidence_classification_result(self):
        payload = {
            "outputs": [
                {"top": "50톤급", "confidence": 0.71},
                {"predictions": [{"class": "100톤급", "confidence": 0.88}]},
            ]
        }

        self.assertEqual(
            _find_classification_result(payload),
            ("100톤급", 0.88),
        )

    def test_does_not_treat_detection_box_as_classification(self):
        payload = {
            "predictions": [
                {
                    "class": "ship",
                    "confidence": 0.99,
                    "x": 10,
                    "y": 20,
                    "width": 30,
                    "height": 40,
                }
            ]
        }

        self.assertIsNone(_find_classification_result(payload))


if __name__ == "__main__":
    unittest.main()
