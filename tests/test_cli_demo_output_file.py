import codecs
import json

from hfups import cli_demo


def test_cli_demo_writes_utf8_json_file(tmp_path) -> None:
    out_path = tmp_path / "demo.json"

    cli_demo.main(["--out", str(out_path)])

    raw = out_path.read_bytes()
    assert not raw.startswith(codecs.BOM_UTF16_LE)
    assert not raw.startswith(codecs.BOM_UTF16_BE)
    assert not raw.startswith(codecs.BOM_UTF8)

    with out_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert "scene_spec" in data
    assert "nova_prompt" in data
