from hfups.cli_demo import run_demo, run_demo_sim
from hfups.link_sim import LinkSimConfig


def test_run_demo_returns_expected_state() -> None:
    result = run_demo()

    assert result["shot_id"] == 0
    assert result["timestamp_s"] == 1024
    assert result["iframe"] is not None
    assert result["mf"] is not None
    assert result["clip_params"] is not None
    assert result["rejected_frames"] == 0


def test_run_demo_sim_with_flips_returns_metrics() -> None:
    result = run_demo_sim(LinkSimConfig(flip_rate=0.2, seed=12345, max_chunk=32))

    required_keys = {
        "iframe",
        "mf",
        "clip_params",
        "shot_id",
        "timestamp_s",
        "bytes_tx",
        "bytes_rx",
        "frames_sent",
        "frames_received",
        "rejected_frames",
    }
    assert required_keys.issubset(result.keys())
    assert result["rejected_frames"] >= 0
