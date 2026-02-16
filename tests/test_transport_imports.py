import importlib


def test_import_tcp_link_module() -> None:
    importlib.import_module("hfups.transport.tcp_link")


def test_import_serial_link_module_without_pyserial_runtime_dependency() -> None:
    importlib.import_module("hfups.transport.serial_link")


def test_tcp_link_classes_can_be_instantiated() -> None:
    mod = importlib.import_module("hfups.transport.tcp_link")
    client = mod.TCPClientLink("127.0.0.1", 9000)
    server = mod.TCPServerLink("127.0.0.1", 9001)

    assert client is not None
    assert server is not None
