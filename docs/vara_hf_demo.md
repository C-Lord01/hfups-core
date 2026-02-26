# HFUPS VARA HF Demo (Text Bridge)

This demo path avoids VARA UI automation and uses a copy/paste-friendly text bridge.

## One-PC Loopback (No VARA Required)

Generate a framed semantic stream:

```bash
python -m hfups.cli_tx_keyframe --transport file --out-bin demo_stream.bin --mock --loop 3
```

Encode framed binary stream to VARA-friendly text:

```bash
python -m hfups.cli_vara_encode --in-bin demo_stream.bin --out-txt vara_out.txt
```

Decode text back to framed binary:

```bash
python -m hfups.cli_vara_decode --in-txt vara_out.txt --out-bin received_stream.bin
```

Play back the received stream:

```bash
python -m hfups.cli_rx_keyframe --transport file --in-bin received_stream.bin --playback
```

## Real VARA Workflow (Manual)

1. Generate and encode as above (`vara_out.txt`).
2. Open `vara_out.txt`.
3. Copy one or more chunk blocks (or the entire stream) into the VARA TX chat window and send.
4. On RX side, copy received text into `vara_in.txt`.
5. Decode and replay:

```bash
python -m hfups.cli_vara_decode --in-txt vara_in.txt --out-bin received_stream.bin
python -m hfups.cli_rx_keyframe --transport file --in-bin received_stream.bin --playback
```

## Notes

- Text format includes per-chunk CRC32, sequence numbers, and end-of-stream CRC32/byte count.
- Chunks can be decoded out of order.
- Decoder fails loudly on corruption, missing chunks, or malformed blocks.

