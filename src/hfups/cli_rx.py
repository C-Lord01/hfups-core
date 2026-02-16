"""HFUPS RX CLI for TCP, serial, or VARA links."""

import argparse
import json
import time

from hfups.rx import run_rx
from hfups.transport.serial_link import SerialLink
from hfups.transport.tcp_link import TCPClientLink, TCPServerLink
from hfups.transport.vara_tcp import VARATCPLink


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HFUPS receiver")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--tcp", nargs=2, metavar=("HOST", "PORT"), help="Receive over TCP")
    mode_group.add_argument("--serial", metavar="PORT", help="Receive over serial")
    mode_group.add_argument("--vara", nargs="?", const="127.0.0.1", metavar="HOST", help="Receive over VARA TCP")
    parser.add_argument("--listen", action=argparse.BooleanOptionalAction, default=True, help="Listen mode (VARA default on)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--vara-cmd-port", type=int, default=8300, help="VARA command TCP port")
    parser.add_argument("--vara-data-port", type=int, default=8301, help="VARA data TCP port")
    parser.add_argument("--mycall", help="Optional callsign for VARA MYCALL setup")
    parser.add_argument("--vara-debug-cmd", action="store_true", help="Print VARA command responses during setup")
    parser.add_argument("--max-seconds", type=float, default=10.0, help="Receiver runtime")
    return parser


def _print_response_lines(resp: str) -> None:
    for line in resp.splitlines(keepends=True):
        print(repr(line))


def _send_vara_cmd(link: VARATCPLink, cmd: str, debug: bool, response_window_s: float = 1.0) -> None:
    link.send_cmd(cmd)
    deadline = time.monotonic() + response_window_s
    while time.monotonic() < deadline:
        resp = link.recv_cmd()
        if not resp:
            continue
        if debug:
            _print_response_lines(resp)


def main() -> None:
    """Run RX loop and print resulting state JSON."""
    parser = _build_parser()
    args = parser.parse_args()

    vara_link: VARATCPLink | None = None

    if args.tcp:
        host, port_str = args.tcp
        if args.listen:
            link = TCPServerLink(host, int(port_str))
            link.open()
        else:
            link = TCPClientLink(host, int(port_str))
            link.open()
    elif args.serial:
        link = SerialLink(args.serial, baud=args.baud)
        link.open()
    else:
        vara_link = VARATCPLink(
            host=args.vara,
            command_port=args.vara_cmd_port,
            data_port=args.vara_data_port,
        )
        vara_link.open()
        link = vara_link

        if args.mycall:
            _send_vara_cmd(vara_link, f"MYCALL {args.mycall} {args.mycall}-T", args.vara_debug_cmd)

        _send_vara_cmd(vara_link, "LISTEN ON" if args.listen else "LISTEN OFF", args.vara_debug_cmd)

    try:
        result = run_rx(link, max_seconds=args.max_seconds)
    finally:
        if vara_link is not None and args.listen:
            try:
                _send_vara_cmd(vara_link, "LISTEN OFF", args.vara_debug_cmd, response_window_s=0.5)
            except Exception:
                pass
        link.close()

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
