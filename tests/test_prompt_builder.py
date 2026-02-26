from hfups.cli_demo import run_demo
from hfups.prompt_builder import build_nova_prompt, build_scene_spec
from hfups.nova.prompt_builder import build_nova_prompt as build_nova_canvas_prompt
from hfups.vision.delta_packet import DeltaEntry, DeltaPacket
from hfups.vision.keyframe_packet import KeyframeObject, KeyframePacket
from hfups.vision.openimages_dict import OpenImagesClass, OpenImagesDict


def test_prompt_builder_from_run_demo_state() -> None:
    state_dict = run_demo()

    scene_spec = build_scene_spec(state_dict)
    assert {"summary", "camera", "clip", "quality", "timestamps"}.issubset(scene_spec.keys())

    prompt = build_nova_prompt(scene_spec)
    assert isinstance(prompt, str)
    assert prompt.strip() != ""

    movement = scene_spec["camera"]["movement"]
    fps = scene_spec["clip"]["fps"]
    assert movement in prompt
    assert ("fps" in prompt) or (str(fps) in prompt)


def test_prompt_builder_degraded_state_still_returns_prompt() -> None:
    degraded = {
        "iframe": None,
        "mf": None,
        "clip_params": None,
        "shot_id": None,
        "timestamp_s": None,
    }

    scene_spec = build_scene_spec(degraded)
    assert "warnings" in scene_spec
    assert len(scene_spec["warnings"]) > 0

    prompt = build_nova_prompt(scene_spec)
    assert isinstance(prompt, str)
    assert prompt.strip() != ""


def _nova_test_dict() -> OpenImagesDict:
    classes = [
        OpenImagesClass(id=0, label="/m/person", name="Person"),
        OpenImagesClass(id=1, label="/m/car", name="Car"),
    ]
    return OpenImagesDict(
        classes=classes,
        by_id={c.id: c for c in classes},
        by_label={c.label: c for c in classes},
    )


def test_nova_prompt_builder_contains_positions_and_motion() -> None:
    packet = KeyframePacket(
        objects=[
            KeyframeObject(class_id=0, track_id=3, cx=1, cy=6, size=1, confidence=14),
            KeyframeObject(class_id=1, track_id=5, cx=5, cy=4, size=2, confidence=12),
        ]
    )
    delta = DeltaPacket(entries=[DeltaEntry(track_id=3, dx=1, dy=-1)])

    prompt = build_nova_canvas_prompt(
        packet,
        _nova_test_dict(),
        caption="A rainy night scene",
        delta_packet=delta,
    )

    lowered = prompt.lower()
    assert "rainy night scene" in lowered
    assert "person" in lowered
    assert "car" in lowered
    assert "grid" in lowered
    assert "over the next second" in lowered
    assert "moves" in lowered
