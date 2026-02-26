from pathlib import Path


def test_repo_has_openimages_source_file() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_csv = repo_root / "data" / "openimages" / "class-descriptions-boxable.csv"
    assert source_csv.is_file()
