from hfups.transport.vara_tcp import VARATCPLink


def test_vara_tcp_import_and_instantiation() -> None:
    link = VARATCPLink()
    assert link is not None
