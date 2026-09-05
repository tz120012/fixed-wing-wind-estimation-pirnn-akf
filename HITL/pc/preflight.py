#!/usr/bin/env python3
"""Validate the PC before starting a HITL session."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from HITL.common.config import (  # noqa: E402
    DEFAULT_CAMPAIGN,
    DEFAULT_LOCAL,
    PROJECT_ROOT,
    load_configs,
)
from HITL.common.manifest import write_json_atomic  # noqa: E402


def _result(name: str, ok: bool, blocker: bool, message: str, **details: Any) -> dict[str, Any]:
    return {
        "name": name,
        "ok": ok,
        "blocker": bool(blocker and not ok),
        "message": message,
        "details": details,
    }


def _run(command: Sequence[str]) -> tuple[int | None, str]:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return completed.returncode, output


def _time_sync_check(local: dict[str, Any]) -> dict[str, Any]:
    clock = local.get("clock", {})
    required = bool(clock.get("require_ntp_synchronized", True))
    configured = str(clock.get("chrony_command", "chronyc tracking"))
    command = shlex.split(configured)
    rc, output = _run(command) if command else (None, "empty chrony command")
    chrony_ok = rc == 0 and (
        "Leap status     : Normal" in output or "Leap status: Normal" in output
    )
    if chrony_ok:
        return _result(
            "clock_sync",
            True,
            required,
            "chrony reports a synchronized clock",
            command=command,
            output=output,
        )

    td_command = ["timedatectl", "show", "--property=NTPSynchronized", "--value"]
    td_rc, td_output = _run(td_command)
    timedatectl_ok = td_rc == 0 and td_output.strip().lower() == "yes"
    return _result(
        "clock_sync",
        timedatectl_ok,
        required,
        (
            "timedatectl reports NTP synchronized"
            if timedatectl_ok
            else "clock synchronization could not be confirmed"
        ),
        chrony={"command": command, "returncode": rc, "output": output},
        timedatectl={"command": td_command, "returncode": td_rc, "output": td_output},
        required=required,
    )


def build_report(campaign_path: Path, local_path: Path, minimum_free_gb: float) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        campaign, local = load_configs(campaign_path, local_path)
        checks.append(_result("configuration", True, True, "campaign and local configs are valid"))
    except Exception as exc:
        checks.append(_result("configuration", False, True, str(exc)))
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "project_root": str(PROJECT_ROOT),
            "checks": checks,
            "ok": False,
            "blockers": ["configuration"],
        }

    pc = local.get("pc")
    if not isinstance(pc, dict):
        checks.append(_result("pc_config", False, True, "local config must contain a pc mapping"))
        pc = {}
    else:
        checks.append(_result("pc_config", True, True, "pc config is present"))

    px4_value = pc.get("px4_root")
    px4_root = Path(str(px4_value)).expanduser().resolve() if px4_value else None
    px4_ok = bool(px4_root and px4_root.is_dir())
    checks.append(
        _result("px4_root", px4_ok, True, "PX4 root exists" if px4_ok else "PX4 root is missing", path=str(px4_root) if px4_root else None)
    )

    hitl_run = px4_root / "Tools/hitl_run.sh" if px4_root else None
    script_ok = bool(hitl_run and hitl_run.is_file() and os.access(hitl_run, os.R_OK))
    checks.append(
        _result(
            "hitl_run",
            script_ok,
            True,
            "Tools/hitl_run.sh is readable" if script_ok else "Tools/hitl_run.sh is missing or unreadable",
            path=str(hitl_run) if hitl_run else None,
        )
    )

    bridge = (
        px4_root / "build/px4_sitl_default/build_jsbsim_bridge/jsbsim_bridge"
        if px4_root
        else None
    )
    bridge_ok = bool(bridge and bridge.is_file() and os.access(bridge, os.X_OK))
    checks.append(
        _result(
            "jsbsim_bridge",
            bridge_ok,
            True,
            "jsbsim_bridge is executable" if bridge_ok else "jsbsim_bridge is missing or not executable",
            path=str(bridge) if bridge else None,
        )
    )

    serial_value = pc.get("hitl_serial_device")
    serial_path = Path(str(serial_value)).expanduser().resolve() if serial_value else None
    serial_ok = bool(
        serial_path
        and serial_path.exists()
        and stat.S_ISCHR(serial_path.stat().st_mode)
        and os.access(serial_path, os.R_OK | os.W_OK)
    )
    checks.append(
        _result(
            "pc_serial",
            serial_ok,
            True,
            "PC serial device is a readable/writable character device"
            if serial_ok
            else "PC serial device is missing, inaccessible, or not a character device",
            path=str(serial_path) if serial_path else None,
        )
    )

    configured_truth = pc.get("truth_output_dir")
    truth_dir = (
        Path(str(configured_truth)).expanduser().resolve()
        if configured_truth
        else (px4_root / "Tools/jsbsim_bridge" if px4_root else None)
    )
    truth_ok = bool(truth_dir and truth_dir.is_dir() and os.access(truth_dir, os.R_OK | os.W_OK))
    checks.append(
        _result(
            "truth_output_dir",
            truth_ok,
            True,
            "truth output directory is readable and writable"
            if truth_ok
            else "truth output directory is missing or not writable",
            path=str(truth_dir) if truth_dir else None,
        )
    )

    disk_target = truth_dir if truth_dir and truth_dir.exists() else PROJECT_ROOT
    usage = shutil.disk_usage(disk_target)
    free_gb = usage.free / (1024.0**3)
    disk_ok = free_gb >= minimum_free_gb
    checks.append(
        _result(
            "disk_space",
            disk_ok,
            True,
            f"{free_gb:.2f} GiB free (minimum {minimum_free_gb:.2f} GiB)",
            path=str(disk_target),
            free_bytes=usage.free,
            minimum_free_bytes=int(minimum_free_gb * 1024**3),
        )
    )
    checks.append(_time_sync_check(local))

    blockers = [check["name"] for check in checks if check["blocker"]]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "campaign_config": str(campaign_path),
        "local_config": str(local_path),
        "checks": checks,
        "ok": not blockers,
        "blockers": blockers,
    }


def _print_text(report: dict[str, Any]) -> None:
    for check in report["checks"]:
        label = "PASS" if check["ok"] else ("BLOCK" if check["blocker"] else "WARN")
        print(f"[{label:5}] {check['name']}: {check['message']}")
    print("Preflight:", "READY" if report["ok"] else "BLOCKED")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--minimum-free-gb", type=float, default=2.0)
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    parser.add_argument("--report-json", type=Path, help="also write the report atomically")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.minimum_free_gb < 0:
        raise SystemExit("--minimum-free-gb must be non-negative")
    report = build_report(args.campaign.expanduser().resolve(), args.local.expanduser().resolve(), args.minimum_free_gb)
    if args.report_json:
        write_json_atomic(args.report_json.expanduser().resolve(), report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
