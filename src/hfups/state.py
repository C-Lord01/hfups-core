"""Minimal HFUPS receiver state and payload dispatcher."""

from dataclasses import asdict, dataclass

from hfups.packets import ClipParams, IFrame, MFPacket, unpack_clip_params, unpack_iframe, unpack_mf


@dataclass
class HFUPSState:
    """Receiver-side state populated from decoded HFUPS payloads."""

    last_iframe: IFrame | None = None
    last_mf: MFPacket | None = None
    last_clip_params: ClipParams | None = None
    shot_id: int | None = None
    timestamp_s: int | None = None


def apply_payload(state: HFUPSState, payload: bytes) -> None:
    """Parse one payload and mutate state with the latest decoded packet."""
    if len(payload) == 5:
        iframe = unpack_iframe(payload)
        state.last_iframe = iframe
        state.shot_id = iframe.shot_id
        state.timestamp_s = iframe.timestamp_s
        return

    if payload.startswith(b"\xC0"):
        state.last_mf = unpack_mf(payload)
        return

    if payload.startswith(b"\xC1"):
        state.last_clip_params = unpack_clip_params(payload)
        return

    raise ValueError("Unknown payload type")


def state_to_dict(state: HFUPSState) -> dict:
    """Return a JSON-safe dictionary view of the receiver state."""
    return {
        "iframe": asdict(state.last_iframe) if state.last_iframe is not None else None,
        "mf": asdict(state.last_mf) if state.last_mf is not None else None,
        "clip_params": asdict(state.last_clip_params) if state.last_clip_params is not None else None,
        "shot_id": state.shot_id,
        "timestamp_s": state.timestamp_s,
    }
