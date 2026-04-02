"""Tests for hfups.nova.bedrock_client."""

import base64
import importlib
import json
import sys
from io import BytesIO
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


def _make_fake_response(image_bytes: bytes) -> dict:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    body_json = json.dumps({"images": [b64]}).encode("utf-8")
    return {"body": BytesIO(body_json)}


def _make_fake_botocore() -> ModuleType:
    """Build a minimal fake botocore module with a ClientError exception class."""
    fake_botocore = ModuleType("botocore")
    fake_exceptions = ModuleType("botocore.exceptions")

    class FakeBotoCoreError(Exception):
        pass

    class FakeClientError(FakeBotoCoreError):
        def __init__(self, error_response, operation_name):
            self.response = error_response
            self.operation_name = operation_name
            msg = error_response.get("Error", {}).get("Message", "Unknown")
            super().__init__(f"ClientError: {msg} ({operation_name})")

    fake_exceptions.BotoCoreError = FakeBotoCoreError
    fake_exceptions.ClientError = FakeClientError
    fake_botocore.exceptions = fake_exceptions
    return fake_botocore, fake_exceptions


def test_invoke_nova_canvas_returns_bytes() -> None:
    """Mock boto3 Session/client; confirm function returns bytes."""
    fake_image = b"\x89PNG\r\nfake_png_content"
    fake_response = _make_fake_response(fake_image)

    mock_client = MagicMock()
    mock_client.invoke_model.return_value = fake_response

    mock_session_instance = MagicMock()
    mock_session_instance.client.return_value = mock_client

    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value = mock_session_instance

    fake_botocore, fake_exceptions = _make_fake_botocore()

    with patch.dict(sys.modules, {
        "boto3": mock_boto3,
        "botocore": fake_botocore,
        "botocore.exceptions": fake_exceptions,
    }):
        import hfups.nova.bedrock_client as bc
        importlib.reload(bc)
        result = bc.invoke_nova_canvas("a test prompt")

    assert isinstance(result, bytes)
    assert result == fake_image


def test_invoke_nova_canvas_missing_boto3() -> None:
    """When boto3 is not importable, ImportError with helpful message is raised."""
    originals = {
        "boto3": sys.modules.get("boto3"),
        "botocore": sys.modules.get("botocore"),
        "botocore.exceptions": sys.modules.get("botocore.exceptions"),
    }

    sys.modules["boto3"] = None  # type: ignore[assignment]
    sys.modules["botocore"] = None  # type: ignore[assignment]
    sys.modules["botocore.exceptions"] = None  # type: ignore[assignment]

    try:
        import hfups.nova.bedrock_client as bc
        importlib.reload(bc)
        with pytest.raises(ImportError, match="boto3 is required"):
            bc.invoke_nova_canvas("test prompt")
    finally:
        for key, val in originals.items():
            if val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = val
        importlib.reload(bc)


def test_invoke_nova_canvas_boto3_error() -> None:
    """When client.invoke_model raises a ClientError, RuntimeError is raised."""
    fake_botocore, fake_exceptions = _make_fake_botocore()

    error_response = {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}}
    client_error = fake_exceptions.ClientError(error_response, "InvokeModel")

    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = client_error

    mock_session_instance = MagicMock()
    mock_session_instance.client.return_value = mock_client

    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value = mock_session_instance

    with patch.dict(sys.modules, {
        "boto3": mock_boto3,
        "botocore": fake_botocore,
        "botocore.exceptions": fake_exceptions,
    }):
        import hfups.nova.bedrock_client as bc
        importlib.reload(bc)
        with pytest.raises(RuntimeError, match="Bedrock Nova Canvas invocation failed"):
            bc.invoke_nova_canvas("test prompt")
