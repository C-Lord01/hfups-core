"""HFUPS TX CLI for TCP, serial, or VARA links."""

import argparse
import time

from hfups.transport.serial_link import SerialLink
from hfups.transport.tcp_link import TCPClientLink
from hfups.transport.vara_tcp import VARATCPLink
from hfups.tx import send_demo_messages


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HFUPS transmitter")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--tcp", nargs=2, metavar=("HOST", "PORT"), help="Transmit over TCP")
    mode_group.add_argument("--serial", metavar="PORT", help="Transmit over serial")
    mode_group.add_argument("--vara", nargs="?", const="127.0.0.1", metavar="HOST", help="Transmit over VARA TCP")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--listen", action=argparse.BooleanOptionalAction, default=False, help="Listen mode (VARA default off)")
    parser.add_argument("--vara-cmd-port", type=int, default=8300, help="VARA command TCP port")
    parser.add_argument("--vara-data-port", type=int, default=8301, help="VARA data TCP port")
    parser.add_argument("--mycall", help="Optional callsign for VARA MYCALL setup")
    parser.add_argument("--vara-debug-cmd", action="store_true", help="Print VARA command responses during setup")
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
    """Send demo frames over the selected transport."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.tcp:
        host, port_str = args.tcp
        link = TCPClientLink(host, int(port_str))
        link.open()
    elif args.serial:
        link = SerialLink(args.serial, baud=args.baud)
        link.open()
    else:
        link = VARATCPLink(
            host=args.vara,
            command_port=args.vara_cmd_port,
            data_port=args.vara_data_port,
        )
        link.open()

        if args.mycall:
            _send_vara_cmd(link, f"MYCALL {args.mycall} {args.mycall}-T", args.vara_debug_cmd)

        _send_vara_cmd(link, "LISTEN ON" if args.listen else "LISTEN OFF", args.vara_debug_cmd)

    try:
        send_demo_messages(link)
    finally:
        link.close()


if __name__ == "__main__":
    main()
