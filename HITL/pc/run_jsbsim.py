#!/usr/bin/env python3
"""Configure and launch one PC-side PX4/JSBSim HITL session."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from HITL.common.config import (  # noqa: E402
    DEFAULT_CAMPAIGN,
    DEFAULT_LOCAL,
    SessionPaths,
    load_configs,
    parse_session_id,
    session_wind,
)
from HITL.common.manifest import (  # noqa: E402
    artifact_record,
    software_manifest,
    write_json_atomic,
)

MPS_TO_FPS = 3.28084


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_local_path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _wind_config_text(wind: dict[str, float], protocol: dict[str, Any], delay_alt_m: float) -> tuple[str, dict[str, float]]:
    speed = float(wind["speed_mps"])
    direction = float(wind["direction_deg"])
    radians = math.radians(direction)
    north = speed * math.cos(radians)
    east = speed * math.sin(radians)
    down = 0.0
    gust_mps = max(0.0, float(wind.get("gust_magnitude_mps", 0.0)))
    warmup = max(0.0, float(protocol.get("warmup_exclusion_s", 0.0)))
    total = max(warmup, float(protocol.get("total_duration_s", warmup)))
    gust_startup = 1.0 if gust_mps else 0.0
    gust_end = 1.0 if gust_mps else 0.0
    gust_steady = max(0.0, total - warmup - gust_startup - gust_end) if gust_mps else 0.0
    gust_start = warmup if gust_mps else -1.0
    values = {
        "wind_north_mps": north,
        "wind_east_mps": east,
        "wind_down_mps": down,
        "gust_magnitude_mps": gust_mps,
    }
    lines = [
        f"WIND_NORTH_FPS={north * MPS_TO_FPS:.4f}",
        f"WIND_EAST_FPS={east * MPS_TO_FPS:.4f}",
        f"WIND_DOWN_FPS={down * MPS_TO_FPS:.4f}",
        f"TURB_GAIN={float(wind['turbulence_gain']):.2f}",
        "TURB_TYPE=3",
        f"WIND_DELAY_ALT_M={delay_alt_m:.1f}",
        f"GUST_MAGNITUDE_FPS={gust_mps * MPS_TO_FPS:.4f}",
        f"GUST_STARTUP_SEC={gust_startup:.4f}",
        f"GUST_STEADY_SEC={gust_steady:.4f}",
        f"GUST_END_SEC={gust_end:.4f}",
        f"GUST_NORTH_FPS={math.cos(radians) if gust_mps else 0.0:.4f}",
        f"GUST_EAST_FPS={math.sin(radians) if gust_mps else 0.0:.4f}",
        "GUST_DOWN_FPS=0.0000",
        f"GUST_START_TIME_SEC={gust_start:.4f}",
    ]
    return "\n".join(lines) + "\n", values


def _truth_snapshot(directory: Path) -> dict[str, tuple[int, int, int]]:
    result: dict[str, tuple[int, int, int]] = {}
    for path in directory.glob("wind_truth_*.csv"):
        if path.is_file():
            info = path.stat()
            result[path.name] = (info.st_ino, info.st_size, info.st_mtime_ns)
    return result


def _new_truth_files(directory: Path, before: dict[str, tuple[int, int, int]]) -> list[Path]:
    new_files: list[Path] = []
    for path in directory.glob("wind_truth_*.csv"):
        if not path.is_file():
            continue
        info = path.stat()
        signature = (info.st_ino, info.st_size, info.st_mtime_ns)
        if path.name not in before or before[path.name] != signature:
            new_files.append(path.resolve())
    return sorted(new_files)


def _backup_if_present(source: Path, destination: Path) -> Path | None:
    if not source.exists():
        return None
    if not source.is_file():
        raise RuntimeError(f"expected a regular file: {source}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite backup: {destination}")
    shutil.copy2(source, destination)
    return destination


def _restore_file(target: Path, backup: Path | None, originally_existed: bool) -> None:
    if backup is not None:
        shutil.copy2(backup, target)
    elif not originally_existed:
        try:
            target.unlink()
        except FileNotFoundError:
            pass


def _launch(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> tuple[int, int | None]:
    process: subprocess.Popen[bytes] | None = None
    received_signal: int | None = None
    previous: dict[int, Any] = {}

    def forward(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = signum
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, forward)
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            returncode = process.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return returncode, received_signal


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id", help="session identifier, for example id_01")
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--no-launch", action="store_true", help="write configuration and manifests only")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    session_id = args.session_id.strip().lower()
    parse_session_id(session_id)
    campaign_path = args.campaign.expanduser().resolve()
    local_path = args.local.expanduser().resolve()
    campaign, local = load_configs(campaign_path, local_path)
    pc = local.get("pc")
    if not isinstance(pc, dict):
        raise SystemExit("local config must contain a pc mapping")

    if not pc.get("px4_root"):
        raise SystemExit("pc.px4_root is required")
    px4_root = _resolve_local_path(pc["px4_root"], PROJECT_DIR)
    hitl_run = px4_root / "Tools/hitl_run.sh"
    bridge_root = px4_root / "Tools/jsbsim_bridge"
    bridge_binary = px4_root / "build/px4_sitl_default/build_jsbsim_bridge/jsbsim_bridge"
    serial_device = str(pc.get("hitl_serial_device", ""))
    if not px4_root.is_dir() or not hitl_run.is_file() or not bridge_binary.is_file() or not os.access(bridge_binary, os.X_OK):
        raise SystemExit("PX4 root, Tools/hitl_run.sh, or executable jsbsim_bridge is unavailable; run preflight.py")
    if not serial_device:
        raise SystemExit("pc.hitl_serial_device is required")

    configured_truth = pc.get("truth_output_dir")
    truth_dir = _resolve_local_path(configured_truth, px4_root) if configured_truth else bridge_root.resolve()
    if not truth_dir.is_dir():
        raise SystemExit(f"truth output directory does not exist: {truth_dir}")

    paths = SessionPaths.build(campaign, session_id, create=True)
    log_path = paths.pc / "jsbsim.log"
    session_config_path = paths.pc / "session_config.json"
    manifest_path = paths.pc / "pc_manifest.json"
    for destination in (log_path, session_config_path, manifest_path):
        if destination.exists():
            raise SystemExit(f"refusing to overwrite existing session artifact: {destination}")

    config_target = bridge_root / "wind_config.txt"
    phases_target = bridge_root / "wind_config_phases.txt"
    wind_backup = _backup_if_present(config_target, paths.pc / "original_wind_config.txt")
    phases_backup = _backup_if_present(phases_target, paths.pc / "original_wind_config_phases.txt")
    wind_existed = config_target.exists()
    phases_disabled: Path | None = None
    restore_external = not args.no_launch
    wind = session_wind(campaign, session_id)
    delay_alt_m = float(pc.get("wind_delay_alt_m", 50.0))
    config_text, components = _wind_config_text(wind, campaign["protocol"], delay_alt_m)

    try:
        if phases_target.exists():
            phases_disabled = phases_target.with_name(
                f"{phases_target.name}.hitl-disabled-{session_id}-{os.getpid()}"
            )
            if phases_disabled.exists():
                raise FileExistsError(phases_disabled)
            phases_target.rename(phases_disabled)
        _atomic_text(config_target, config_text)
        applied_copy = paths.pc / "wind_config_applied.txt"
        _atomic_text(applied_copy, config_text)

        session_config = {
            "created_utc": _utc_now(),
            "session_id": session_id,
            "campaign_config": str(campaign_path),
            "local_config": str(local_path),
            "wind": {**wind, **components, "direction_convention": "N/E vector bearing, 0 deg north, 90 deg east"},
            "protocol": campaign["protocol"],
            "launch": {
                "px4_root": str(px4_root),
                "hitl_run": str(hitl_run),
                "bridge_binary": str(bridge_binary),
                "serial_device": serial_device,
                "baudrate": int(pc.get("hitl_baudrate", 921600)),
                "model": str(pc.get("model", "rascal")),
                "world": str(pc.get("world", "LSZH")),
                "headless": bool(pc.get("headless", True)),
                "truth_output_dir": str(truth_dir),
                "qgc": "manual",
                "no_launch": args.no_launch,
            },
        }
        write_json_atomic(session_config_path, session_config)

        before = _truth_snapshot(truth_dir)
        returncode: int | None = None
        received_signal: int | None = None
        if not args.no_launch:
            command = [
                "bash",
                str(hitl_run),
                str(pc.get("model", "rascal")),
                str(pc.get("world", "LSZH")),
            ]
            env = os.environ.copy()
            if bool(pc.get("headless", True)):
                env["HEADLESS"] = "1"
            else:
                env.pop("HEADLESS", None)
            env.update(
                {
                    "HITL_SERIAL_DEVICE": serial_device,
                    "HITL_BAUDRATE": str(int(pc.get("hitl_baudrate", 921600))),
                }
            )
            print(f"Starting {' '.join(command)}; QGroundControl operation remains manual.")
            returncode, received_signal = _launch(command, px4_root, env, log_path)

        new_truth = _new_truth_files(truth_dir, before)
        truth_error = None
        if not args.no_launch and len(new_truth) != 1:
            truth_error = f"expected exactly one new wind_truth_*.csv, found {len(new_truth)}"

        artifacts = [artifact_record(applied_copy), artifact_record(session_config_path)]
        for backup in (wind_backup, phases_backup):
            if backup is not None:
                artifacts.append(artifact_record(backup))
        if log_path.exists():
            artifacts.append(artifact_record(log_path))
        if len(new_truth) == 1:
            truth_name = str(
                campaign.get("logging", {}).get(
                    "pc_truth_name", "jsbsim_truth.csv"
                )
            )
            canonical_truth = paths.pc / truth_name
            if canonical_truth.exists():
                raise FileExistsError(
                    f"refusing to overwrite canonical truth: {canonical_truth}"
                )
            shutil.copy2(new_truth[0], canonical_truth)
            artifacts.append(
                {
                    "source": artifact_record(new_truth[0]),
                    "canonical": artifact_record(canonical_truth),
                }
            )
        manifest = {
            **software_manifest(px4_root),
            "session_id": session_id,
            "artifacts": artifacts,
            "launch_returncode": returncode,
            "received_signal": received_signal,
            "new_truth_files": [str(path) for path in new_truth],
            "truth_error": truth_error,
            "external_config_restored": restore_external,
        }
        write_json_atomic(manifest_path, manifest)
        print(f"Wrote {session_config_path} and {manifest_path}")
        if truth_error:
            print(truth_error, file=sys.stderr)
            return 3
        if received_signal in (signal.SIGINT, signal.SIGTERM):
            return 0
        if returncode not in (None, 0):
            return returncode if 0 < returncode < 126 else 2
        return 0
    finally:
        if restore_external:
            _restore_file(config_target, wind_backup, wind_existed)
            if phases_disabled is not None and phases_disabled.exists():
                phases_disabled.rename(phases_target)


if __name__ == "__main__":
    raise SystemExit(main())
