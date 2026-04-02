"""Amazon Bedrock client for Nova Canvas image generation."""

from __future__ import annotations

import base64
import json


def invoke_nova_canvas(
    prompt: str,
    *,
    region: str = "us-east-1",
    profile: str | None = None,
    width: int = 512,
    height: int = 512,
) -> bytes:
    """
    Invoke Amazon Nova Canvas via Bedrock to generate an image from a text prompt.

    Returns raw PNG bytes.

    Raises:
        ImportError: if boto3 is not installed.
        RuntimeError: if the Bedrock invocation fails.
    """
    try:
        import boto3
        import botocore.exceptions
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for Nova Canvas invocation. "
            "Install it with: pip install boto3>=1.34"
        ) from exc

    try:
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        client = session.client("bedrock-runtime", region_name=region)

        body = json.dumps(
            {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {"text": prompt},
                "imageGenerationConfig": {
                    "width": width,
                    "height": height,
                    "numberOfImages": 1,
                },
            }
        )

        response = client.invoke_model(
            modelId="amazon.nova-canvas-v1:0",
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read())
        image_b64 = response_body["images"][0]
        return base64.b64decode(image_b64)

    except botocore.exceptions.BotoCoreError as exc:
        raise RuntimeError(
            f"Bedrock Nova Canvas invocation failed: {exc}"
        ) from exc
    except botocore.exceptions.ClientError as exc:
        raise RuntimeError(
            f"Bedrock Nova Canvas invocation failed: {exc}"
        ) from exc
