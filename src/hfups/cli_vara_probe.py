"""VARA command-port probe utility."""

import argparse
import time

from hfups.transport.vara_tcp import VARATCPLink


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe VARA command/data ports")
    parser.add_argument("--vara", default="127.0.0.1", metavar="HOST", help="VARA host")
    parser.add_argument("--vara-cmd-port", type=int, default=8300, help="VARA command TCP port")
    parser.add_argument("--vara-data-port", type=int, default=8301, help="VARA data TCP port")
    parser.add_argument("--timeout", type=float, default=0.5, help="Socket timeout in seconds")
    parser.add_argument("--discover", action="store_true", help="Probe likely help/list commands")
    parser.add_argument("--cmd", action="append", default=[], help="Command to send (repeatable)")
    parser.add_argument("--common", action="store_true", help="Run common VARA HF command sequence")
    parser.add_argument("--call", help="Callsign used by --common")
    parser.add_argument("--connect-try", metavar="REMOTECALL", help="Try multiple possible connect commands")
    parser.add_argument("--mycall", metavar="CALLSIGN", help="Optional MYCALL used with --connect-try")
    return parser


def _print_response_lines(resp: str) -> None:
    for line in resp.splitlines(keepends=True):
        print(repr(line))


def _send_and_read(link: VARATCPLink, cmd: str, response_window_s: float) -> bool:
    print(f">>> {cmd}")
    link.send_cmd(cmd)

    saw_ok = False
    deadline = time.monotonic() + response_window_s
    while time.monotonic() < deadline:
        resp = link.recv_cmd()
        if not resp:
            continue
        _print_response_lines(resp)
        if "OK" in resp:
            saw_ok = True
    return saw_ok


def main() -> None:
    """Connect to VARA and probe command-port queries."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.connect_try:
        remote = args.connect_try
        commands = [
            f"CONNECT {remote}",
            f"CONNECT TO {remote}",
            f"CONNECTCALL {remote}",
            f"CALL {remote}",
            f"LINK {remote}",
            f"C {remote}",
            f"DIAL {remote}",
            f"TARGET {remote}",
            f"STATION {remote}",
            f"CONNECT {remote} 2300",
            f"CONNECT {remote} 500",
            f"CONNECT {remote} 2750",
        ]
        response_window_s = 2.0
    elif args.cmd:
        commands = args.cmd
        response_window_s = 1.5
    elif args.common:
        if not args.call:
            parser.error("--common requires --call CALLSIGN")
        call = args.call
        commands = [
            "VERSION",
            f"MYCALL {call} {call}-T",
            "LISTEN ON",
            "LISTEN OFF",
            "DISCONNECT",
        ]
        response_window_s = 1.5
    elif args.discover:
        commands = ["HELP", "COMMANDS", "LIST", "?", "CAT", "INFO", "VERSION"]
        response_window_s = 2.0
    else:
        commands = ["VERSION", "MYCALL", "STATE", "STATUS", "LISTEN"]
        response_window_s = 1.0

    link = VARATCPLink(
        host=args.vara,
        command_port=args.vara_cmd_port,
        data_port=args.vara_data_port,
        timeout_s=args.timeout,
    )

    link.open()
    print("Connected to VARA command/data ports")

    try:
        if args.connect_try and args.mycall:
            _send_and_read(link, f"MYCALL {args.mycall} {args.mycall}-T", response_window_s)

        for cmd in commands:
            accepted = _send_and_read(link, cmd, response_window_s)
            if args.connect_try and accepted:
                print(f"*** CONNECT COMMAND ACCEPTED: {cmd}")
                break
    finally:
        link.close()


if __name__ == "__main__":
    main()
