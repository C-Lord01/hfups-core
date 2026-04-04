"""Hugging Face Inference API client for text-to-image generation.

Replacement backend for the deprecated Nova Canvas client.
"""

from __future__ import annotations

import os

_HF_API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"


def invoke_hf_image(
    prompt: str,
    token: str | None = None,
    model: str = "black-forest-labs/FLUX.1-schnell",
    width: int = 1024,
    height: int = 1024,
) -> bytes:
    """
    Call HuggingFace Inference API to generate an image.

    Returns raw PNG bytes.
    Raises RuntimeError on any failure.

    token defaults to HF_API_TOKEN env var.
    """
    resolved_token = token or os.environ.get("HF_API_TOKEN", "")
    if not resolved_token:
        raise RuntimeError(
            "HF_API_TOKEN environment variable is not set. "
            "Set it with: $env:HF_API_TOKEN='your_token' (PowerShell) "
            "or export HF_API_TOKEN=your_token (bash)"
        )

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "requests is not installed. Install with: pip install requests"
        ) from exc

    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {resolved_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "width": width,
            "height": height,
        },
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("HuggingFace API request timed out after 120s") from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"HuggingFace API returned HTTP {response.status_code}: {response.text[:500]}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"HuggingFace API request failed: {exc}") from exc

    content_type = response.headers.get("content-type", "")
    if "image" not in content_type and "application/octet-stream" not in content_type:
        raise RuntimeError(
            f"Unexpected content-type from HF API: {content_type}. "
            f"Response: {response.text[:200]}"
        )

    return response.content


def invoke_image_generation(
    prompt: str,
    token: str | None = None,
    width: int = 1024,
    height: int = 1024,
) -> bytes:
    """Generate an image from a text prompt via the HF Inference API.

    Returns raw PNG bytes.

    Raises:
        RuntimeError: if HF_API_TOKEN is not set, requests is not installed,
                      or the API returns a non-200 response.
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required: pip install requests") from exc

    if token is None:
        token = os.environ.get("HF_API_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_API_TOKEN not set. Export it or pass token= explicitly."
            )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body = {
        "inputs": prompt,
        "parameters": {"width": width, "height": height},
    }

    response = requests.post(_HF_API_URL, headers=headers, json=body)

    if response.status_code != 200:
        raise RuntimeError(
            f"HF inference failed: {response.status_code} {response.text[:200]}"
        )

    return response.content
