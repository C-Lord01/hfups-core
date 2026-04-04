"""Tests for hfups.nova.hf_client."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from hfups.nova.hf_client import invoke_hf_image, invoke_image_generation


def test_returns_bytes():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fakepng"

    with patch("requests.post", return_value=mock_response):
        result = invoke_image_generation("a sunset", token="test-token")

    assert result == b"fakepng"


def test_missing_token():
    env = {k: v for k, v in os.environ.items() if k != "HF_API_TOKEN"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="HF_API_TOKEN not set"):
            invoke_image_generation("a sunset")


def test_http_error():
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.text = "Service Unavailable"

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="503"):
            invoke_image_generation("a sunset", token="test-token")


# --- Tests for invoke_hf_image ---

def test_invoke_returns_bytes():
    import requests as _requests

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fakepng"
    mock_response.headers = {"content-type": "image/png"}
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        result = invoke_hf_image("a flooded street", token="test-token")

    assert isinstance(result, bytes)
    assert result == b"fakepng"


def test_missing_token_raises():
    env = {k: v for k, v in os.environ.items() if k != "HF_API_TOKEN"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="HF_API_TOKEN environment variable is not set"):
            invoke_hf_image("a sunset")


def test_http_error_raises():
    import requests as _requests

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    http_err = _requests.exceptions.HTTPError(response=mock_response)
    mock_response.raise_for_status = MagicMock(side_effect=http_err)

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="401"):
            invoke_hf_image("a sunset", token="test-token")


def test_timeout_raises():
    import requests as _requests

    with patch(
        "requests.post",
        side_effect=_requests.exceptions.Timeout("timed out"),
    ):
        with pytest.raises(RuntimeError, match="timed out"):
            invoke_hf_image("a sunset", token="test-token")
