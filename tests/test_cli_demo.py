from hfups.cli_demo import run_demo, run_demo_sim


def test_run_demo_returns_expected_keys() -> None:
    result = run_demo()

    required_keys = {"encoded_bytes", "airtime_10kbps_seconds", "prompt", "object_count", "template"}
    assert required_keys.issubset(result.keys())
    assert result["object_count"] == 3
    assert result["template"] == "disaster_response"
    assert isinstance(result["encoded_bytes"], int)
    assert result["encoded_bytes"] > 0
    assert isinstance(result["airtime_10kbps_seconds"], float)
    assert isinstance(result["prompt"], str)
    assert len(result["prompt"]) > 0


def test_run_demo_sim_returns_metrics() -> None:
    result = run_demo_sim()

    required_keys = {"encoded_bytes", "airtime_10kbps_seconds", "prompt", "object_count", "template"}
    assert required_keys.issubset(result.keys())
    assert result["object_count"] >= 0
