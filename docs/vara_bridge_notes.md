# VARA Bridge Notes

## TCP Transport Mode

HFUPS semantic TX/RX CLIs support a live TCP byte-stream mode:

- RX: `python -m hfups.cli_rx_keyframe --transport tcp --tcp 127.0.0.1:8301 --playback`
- TX: `python -m hfups.cli_tx_keyframe --transport tcp --tcp 127.0.0.1:8301 --mock --loop 3`

In this mode, semantic payloads are still HFUPS-framed (COBS + CRC + delimiter) and sent over TCP.

## Suggested Default Endpoint

- `127.0.0.1:8301`

## Future VARA Bridge Placement

The future bridge app can connect HFUPS to VARA without changing semantic packet code:

`HFUPS TX -> TCP -> (VARA Bridge) -> VARA Modem -> RF -> VARA Modem -> (VARA Bridge) -> TCP -> HFUPS RX`

This prompt does **not** implement the VARA bridge itself.  
It only defines and validates the TCP interface that bridge will plug into.

