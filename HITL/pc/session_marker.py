#!/usr/bin/env python3
"""Probe a Pi clock listener and save NTP-style four-timestamp records."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from HITL.common.config import (  # noqa: E402
    DEFAULT_CAMPAIGN,
    SessionPaths,
    load_configs,
    parse_session_id,
)

PROTOCOL_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_reply(reply: Any, request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(reply, dict):
        raise ValueError("reply is not a JSON object")
    required = ("protocol_version", "session_id", "sample", "t1_pc_send_ns", "t2_pi_recv_ns", "t3_pi_send_ns")
    missing = [name for name in required if name not in reply]
    if missing:
        raise ValueError(f"reply is missing fields: {', '.join(missing)}")
    for name in ("protocol_version", "session_id", "sample", "t1_pc_send_ns"):
        if reply[name] != request[name]:
            raise ValueError(f"reply {name} does not match request")
    for name in ("t1_pc_send_ns", "t2_pi_recv_ns", "t3_pi_send_ns"):
        if not isinstance(reply[name], int):
            raise ValueError(f"reply {name} must be an integer")
    return reply


def probe(
    host: str,
    port: int,
    session_id: str,
    count: int,
    interval_s: float,
    timeout_s: float,
    output: Path,
) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = 0
    with output.open("a", encoding="utf-8") as stream, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((host, port))
        for sample in range(count):
            t1 = time.time_ns()
            request = {
                "protocol_version": PROTOCOL_VERSION,
                "session_id": session_id,
                "sample": sample,
                "t1_pc_send_ns": t1,
            }
            record: dict[str, Any]
            try:
                sock.send(json.dumps(request, separators=(",", ":")).encode("utf-8"))
                deadline = time.monotonic() + timeout_s
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise socket.timeout("probe timed out")
                    sock.settimeout(remaining)
                    payload = sock.recv(65535)
                    t4 = time.time_ns()
                    reply = _validate_reply(json.loads(payload.decode("utf-8")), request)
                    t2 = reply["t2_pi_recv_ns"]
                    t3 = reply["t3_pi_send_ns"]
                    rtt = (t4 - t1) - (t3 - t2)
                    offset = ((t2 - t1) + (t3 - t4)) / 2.0
                    uncertainty = rtt / 2.0
                    record = {
                        **reply,
                        "t4_pc_recv_ns": t4,
                        "offset_pi_minus_pc_ns": offset,
                        "rtt_ns": rtt,
                        "uncertainty_ns": uncertainty,
                        "recorded_utc": _utc_now(),
                        "peer": {"host": host, "port": port},
                        "ok": True,
                    }
                    successes += 1
                    break
            except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                record = {
                    **request,
                    "recorded_utc": _utc_now(),
                    "peer": {"host": host, "port": port},
                    "ok": False,
                    "error": str(exc),
                }
                failures += 1
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if sample + 1 < count and interval_s:
                time.sleep(interval_s)
    return successes, failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id", help="session identifier, for example id_01")
    parser.add_argument("--host", required=True, help="Pi marker-listener hostname or address")
    parser.add_argument("--port", type=int, help="listener UDP port; defaults to campaign config")
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--output", type=Path, help="JSONL output; defaults to the session PC directory")
    parser.add_argument(
        "--phase",
        choices=("start", "end"),
        default="start",
        help="probe set name used by offline clock-drift fitting",
    )
    parser.add_argument("--count", type=int, help="number of probes; defaults to campaign config")
    parser.add_argument("--interval", type=float, help="seconds between probes")
    parser.add_argument("--timeout", type=float, help="per-probe timeout in seconds")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    session_id = args.session_id.strip().lower()
    parse_session_id(session_id)
    campaign, _ = load_configs(args.campaign, require_local=False)
    alignment = campaign["alignment"]
    configured_count = alignment[
        "marker_samples_at_start" if args.phase == "start" else "marker_samples_at_end"
    ]
    count = args.count if args.count is not None else int(configured_count)
    interval = args.interval if args.interval is not None else float(alignment["marker_interval_ms"]) / 1000.0
    timeout = args.timeout if args.timeout is not None else float(alignment["marker_timeout_s"])
    port = args.port if args.port is not None else int(alignment["marker_udp_port"])
    if count <= 0 or interval < 0 or timeout <= 0 or not 1 <= port <= 65535:
        raise SystemExit("count/timeout/port must be positive and interval must be non-negative")
    output = (
        args.output.expanduser().resolve()
        if args.output
        else SessionPaths.build(campaign, session_id, create=True).pc
        / f"marker_{args.phase}.jsonl"
    )
    successes, failures = probe(args.host, port, session_id, count, interval, timeout, output)
    print(f"Saved {successes + failures} probes to {output} ({successes} successful, {failures} failed)")
    return 0 if successes == count else 2


if __name__ == "__main__":
    raise SystemExit(main())
