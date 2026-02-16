import importlib


def test_vara_smoke_import() -> None:
    mod = importlib.import_module("hfups.cli_vara_smoke")
    assert hasattr(mod, "main")
