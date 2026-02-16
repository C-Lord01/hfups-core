"""HFUPS transmitter helpers."""

from hfups.framing import encode_frame


def send_demo_messages(link) -> None:
    """Send known-good demo payloads as framed wire messages."""
    iframe_payload = bytes.fromhex("0000400081")
    mf_payload = bytes.fromhex("C0A824")
    clip_payload = bytes.fromhex("C191")

    for payload in (iframe_payload, mf_payload, clip_payload):
        link.send(encode_frame(payload))
