import sys
import types
import unittest
from unittest.mock import patch

from roboflow_inference import get_inference_client


class RoboflowClientConfigurationTests(unittest.TestCase):
    def tearDown(self):
        get_inference_client.cache_clear()

    def test_client_uses_header_auth_and_detection_threshold(self):
        created_clients = []

        class FakeConfiguration:
            def __init__(self, **options):
                self.options = options

        class FakeClient:
            def __init__(self, **options):
                self.options = options
                self.configuration = None
                created_clients.append(self)

            def configure(self, configuration):
                self.configuration = configuration

        fake_sdk = types.ModuleType("inference_sdk")
        fake_sdk.InferenceConfiguration = FakeConfiguration
        fake_sdk.InferenceHTTPClient = FakeClient

        with patch.dict(sys.modules, {"inference_sdk": fake_sdk}):
            client = get_inference_client(
                "https://serverless.roboflow.com",
                "test-key",
                confidence_threshold=0.3,
            )

        self.assertIs(client, created_clients[0])
        self.assertEqual(
            client.options,
            {
                "api_url": "https://serverless.roboflow.com",
                "api_key": "test-key",
            },
        )
        self.assertEqual(
            client.configuration.options,
            {
                "api_key_transport": "header",
                "confidence_threshold": 0.3,
            },
        )


if __name__ == "__main__":
    unittest.main()
