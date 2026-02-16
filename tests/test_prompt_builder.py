from hfups.cli_demo import run_demo
from hfups.prompt_builder import build_nova_prompt, build_scene_spec


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
