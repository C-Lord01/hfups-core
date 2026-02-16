import json

import pytest

from hfups.packets import unpack_clip_params, unpack_iframe, unpack_mf
from hfups.state import HFUPSState, apply_payload, state_to_dict


def test_apply_payload_populates_state_in_order() -> None:
    state = HFUPSState()

    iframe_payload = bytes.fromhex("0000400081")
    mf_payload = bytes.fromhex("C0A824")
    clip_payload = bytes.fromhex("C191")

    apply_payload(state, iframe_payload)
    apply_payload(state, mf_payload)
    apply_payload(state, clip_payload)

    assert state.last_iframe == unpack_iframe(iframe_payload)
    assert state.last_mf == unpack_mf(mf_payload)
    assert state.last_clip_params == unpack_clip_params(clip_payload)
    assert state.shot_id == 0
    assert state.timestamp_s == 1024


def test_apply_payload_unknown_type_raises() -> None:
    state = HFUPSState()
    with pytest.raises(ValueError, match="Unknown payload type"):
        apply_payload(state, b"\xFF\x00")


def test_state_to_dict_is_json_serializable() -> None:
    state = HFUPSState()
    apply_payload(state, bytes.fromhex("0000400081"))
    apply_payload(state, bytes.fromhex("C0A824"))
    apply_payload(state, bytes.fromhex("C191"))

    state_dict = state_to_dict(state)

    assert set(state_dict.keys()) == {"iframe", "mf", "clip_params", "shot_id", "timestamp_s"}
    assert state_dict["shot_id"] == 0
    assert state_dict["timestamp_s"] == 1024

    json.dumps(state_dict)
