"""Tests for hfups.nova.hf_client."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from hfups.nova.hf_client import invoke_image_generation


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
