from hfups.cli_demo import run_demo


def test_run_demo_returns_expected_state() -> None:
    result = run_demo()

    assert result["shot_id"] == 0
    assert result["timestamp_s"] == 1024
    assert result["iframe"] is not None
    assert result["mf"] is not None
    assert result["clip_params"] is not None
    assert result["rejected_frames"] == 0
