import importlib


def test_vara_probe_import_and_main_exists() -> None:
    mod = importlib.import_module("hfups.cli_vara_probe")
    assert hasattr(mod, "main")
