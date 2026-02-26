from hfups.integrations.transport_shim import TransportShim, read_bin, write_bin


def test_transport_shim_send_recv_roundtrip() -> None:
    shim = TransportShim()
    payload = b"\x01hello\x00world"

    frame = shim.send_payload(payload)
    out = shim.recv_payload()

    assert out == payload
    assert frame.endswith(b"\x00")


def test_transport_shim_bin_helpers(tmp_path) -> None:
    data = b"\x11\x22\x33\x00\x44"
    out_path = tmp_path / "stream.bin"

    write_bin(out_path, data)
    back = read_bin(out_path)

    assert back == data


def test_transport_shim_load_framed_stream() -> None:
    shim = TransportShim()
    f1 = shim.send_payload(b"\x01abc")
    f2 = shim.send_payload(b"\x02xyz")
    stream = f1 + f2

    rx = TransportShim()
    rx.load_framed_stream(stream)

    assert rx.recv_payload() == b"\x01abc"
    assert rx.recv_payload() == b"\x02xyz"
    assert rx.recv_payload() is None
