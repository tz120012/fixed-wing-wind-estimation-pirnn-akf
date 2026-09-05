#!/usr/bin/env python3
"""UDP responder for the campaign's NTP-like PC/Pi clock markers."""

from __future__ import annotations

import argparse
import json
import signal
import socket
import time
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1
MAX_DATAGRAM_BYTES = 16_384
_stop = False


def _request_stop(_signum: int, _frame: Any) -> None:
    global _stop
    _stop = True


def _integer(message: dict[str, Any], name: str) -> int:
    value = message.get(name)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be an integer")


def validate_probe(message: Any, session_id: str) -> tuple[int, int]:
    if not isinstance(message, dict):
        raise ValueError("probe must be a JSON object")
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be {PROTOCOL_VERSION}")
    if message.get("session_id") != session_id:
        raise ValueError("session_id mismatch")
    sample = _integer(message, "sample")
    t1_wall = _integer(message, "t1_pc_send_ns")
    return sample, t1_wall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0", help="UDP bind address")
    parser.add_argument("--port", type=int, default=45880, help="UDP listen port")
    parser.add_argument("--output", type=Path, required=True, help="JSONL marker log")
    parser.add_argument("--session", required=True, help="expected campaign session ID")
    parser.add_argument(
        "--duration", type=float, default=None, help="stop after this many seconds"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.port < 65_536:
        raise SystemExit("--port must be in 1..65535")
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be positive")

    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    with args.output.expanduser().open("a", encoding="utf-8", buffering=1) as log:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener:
            listener.bind((args.bind, args.port))
            listener.settimeout(0.25)
            while not _stop:
                if args.duration is not None and time.monotonic() - started >= args.duration:
                    break
                try:
                    payload, peer = listener.recvfrom(MAX_DATAGRAM_BYTES)
                except socket.timeout:
                    continue

                t2_wall = time.time_ns()
                t2_mono = time.monotonic_ns()
                try:
                    request = json.loads(payload.decode("utf-8"))
                    sample, t1_wall = validate_probe(request, args.session)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    log.write(
                        json.dumps(
                            {
                                "protocol_version": PROTOCOL_VERSION,
                                "type": "rejected_probe",
                                "session_id": args.session,
                                "peer": f"{peer[0]}:{peer[1]}",
                                "pi_receive_wall_time_ns": t2_wall,
                                "pi_receive_monotonic_ns": t2_mono,
                                "error": str(exc),
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    continue

                # Capture t3 immediately before serialization/send. Both clock domains
                # are retained in the audit log; wall-clock t1/t2/t3 are used for NTP.
                t3_wall = time.time_ns()
                t3_mono = time.monotonic_ns()
                reply = {
                    "protocol_version": PROTOCOL_VERSION,
                    "session_id": args.session,
                    "sample": sample,
                    "t1_pc_send_ns": t1_wall,
                    "t2_pi_recv_ns": t2_wall,
                    "t3_pi_send_ns": t3_wall,
                }
                listener.sendto(
                    json.dumps(reply, separators=(",", ":")).encode("utf-8"), peer
                )
                log.write(
                    json.dumps(
                        {
                            **reply,
                            "type": "served_probe",
                            "t2_pi_recv_monotonic_ns": t2_mono,
                            "t3_pi_send_monotonic_ns": t3_mono,
                            "peer": f"{peer[0]}:{peer[1]}",
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
