"""Hugging Face Inference API client for text-to-image generation.

Replacement backend for the deprecated Nova Canvas client.
"""

from __future__ import annotations

import os

_HF_API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"


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
