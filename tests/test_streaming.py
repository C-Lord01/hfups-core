from hfups.framing import encode_frame
from hfups.streaming import FrameStreamDecoder, iter_frames_from_stream


def test_iter_frames_reassembles_split_frame() -> None:
    frame = encode_frame(bytes.fromhex("000040"))
    chunks = [frame[:2], frame[2:4], frame[4:]]

    assert iter_frames_from_stream(chunks) == [frame]


def test_decoder_multiple_frames_in_one_chunk() -> None:
    frame1 = encode_frame(bytes.fromhex("000040"))
    frame2 = encode_frame(b"\x11\x22\x33")

    decoder = FrameStreamDecoder()
    assert decoder.feed(frame1 + frame2) == [frame1, frame2]


def test_frame_splitting_with_non_zero_noise_between_frames() -> None:
    frame1 = encode_frame(bytes.fromhex("000040"))
    frame2 = encode_frame(b"\x05\x06")
    noise = b"\xAA\xBB"

    decoder = FrameStreamDecoder()
    data = frame1 + noise + b"\x00" + frame2

    assert decoder.feed(data) == [frame1, noise + b"\x00", frame2]
