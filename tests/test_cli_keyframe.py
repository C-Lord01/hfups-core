import builtins
import json
from pathlib import Path

import pytest

from hfups import cli_keyframe


def _write_test_dict(path: Path) -> None:
    payload = {
        "schema": "hfups.dict.openimages.boxable.v1",
        "source": {
            "dataset": "openimages",
            "version": "v7",
            "subset": "boxable",
            "input_file": "data/openimages/class-descriptions-boxable.csv",
        },
        "created_utc": "2026-02-26T00:00:00Z",
        "classes": [
            {"id": 0, "label": "/m/01g317", "name": "Person"},
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_test_mapping(path: Path) -> None:
    path.write_text(json.dumps({"person": "/m/01g317"}) + "\n", encoding="utf-8")


def test_cli_keyframe_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_keyframe.main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()
    assert "--debug-detections" in captured.out
    assert "--debug-mapping" in captured.out
    assert "--debug-top" in captured.out


def test_cli_keyframe_missing_required_args_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exc:
        cli_keyframe.main([])
    assert exc.value.code != 0


def test_cli_keyframe_missing_ultralytics_returns_friendly_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"not-an-image")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"stub")
    dict_path = tmp_path / "dict.json"
    mapping_path = tmp_path / "mapping.json"
    _write_test_dict(dict_path)
    _write_test_mapping(mapping_path)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "ultralytics":
            raise ImportError("simulated missing ultralytics")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    exit_code = cli_keyframe.main(
        [
            "--image",
            str(image_path),
            "--model",
            str(model_path),
            "--dict",
            str(dict_path),
            "--mapping",
            str(mapping_path),
            "--debug-detections",
            "--debug-mapping",
            "--debug-top",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Ultralytics not installed. Install with: python -m pip install ultralytics" in captured.err
