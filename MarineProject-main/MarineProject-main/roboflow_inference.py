"""Shared Roboflow Serverless inference client configuration."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def get_inference_client(
    api_url: str,
    api_key: str,
    confidence_threshold: float | None = None,
):
    """Return a cached header-authenticated Roboflow inference client."""
    try:
        from inference_sdk import InferenceConfiguration, InferenceHTTPClient
    except ImportError as exc:
        raise RuntimeError(
            "inference-sdk가 설치되지 않았습니다. requirements.txt를 다시 설치해 주세요."
        ) from exc

    configuration_options = {"api_key_transport": "header"}
    if confidence_threshold is not None:
        configuration_options["confidence_threshold"] = confidence_threshold

    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    client.configure(InferenceConfiguration(**configuration_options))
    return client
