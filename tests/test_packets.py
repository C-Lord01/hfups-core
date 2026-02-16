import pytest

from hfups.packets import (
    ClipParams,
    IFrame,
    MFPacket,
    pack_clip_params,
    pack_iframe,
    pack_mf,
    unpack_clip_params,
    unpack_iframe,
    unpack_mf,
)


def test_iframe_golden_vector() -> None:
    # The requested q_* example fields are inferred from 0x0000400081 using the
    # declared 40-bit order: q_m,q_u,q_v,flags,timestamp_s,shot_id,psnr_idx,quant_idx,crc4.
    # int(0x0000400081) decodes to: q_m=0,q_u=0,q_v=0,flags=0,timestamp_s=1024,
    # shot_id=0,psnr_idx=2,quant_idx=0,crc4=1.
    msg = IFrame(
        q_m=0,
        q_u=0,
        q_v=0,
        flags=0,
        timestamp_s=1024,
        shot_id=0,
        psnr_idx=2,
        quant_idx=0,
        crc4=1,
    )
    assert pack_iframe(msg) == bytes.fromhex("0000400081")
    assert unpack_iframe(bytes.fromhex("0000400081")) == msg


def test_mf_golden_vector() -> None:
    msg = MFPacket(cam_move=42, framing=1, subject_path=4)
    assert pack_mf(msg) == bytes.fromhex("C0A824")
    assert unpack_mf(bytes.fromhex("C0A824")) == msg


def test_clip_params_roundtrip() -> None:
    msg = ClipParams(fps_idx=2, frame_count=17)
    assert pack_clip_params(msg) == bytes.fromhex("C191")
    assert unpack_clip_params(bytes.fromhex("C191")) == msg


def test_iframe_range_check_q_m_too_large() -> None:
    with pytest.raises(ValueError):
        pack_iframe(
            IFrame(
                q_m=8,
                q_u=0,
                q_v=0,
                flags=0,
                timestamp_s=0,
                shot_id=0,
                psnr_idx=0,
                quant_idx=0,
                crc4=0,
            )
        )


def test_clip_params_range_check_frame_count_too_large() -> None:
    with pytest.raises(ValueError):
        pack_clip_params(ClipParams(fps_idx=0, frame_count=64))


def test_unpack_iframe_wrong_length() -> None:
    with pytest.raises(ValueError):
        unpack_iframe(b"\x00\x00\x00\x00")


def test_unpack_mf_wrong_marker() -> None:
    with pytest.raises(ValueError):
        unpack_mf(bytes.fromhex("C1A824"))


def test_unpack_clip_params_wrong_marker() -> None:
    with pytest.raises(ValueError):
        unpack_clip_params(bytes.fromhex("C091"))
