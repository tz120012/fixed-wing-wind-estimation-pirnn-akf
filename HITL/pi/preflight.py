#!/usr/bin/env python3
"""Validate a Raspberry Pi before starting a repeated HITL session."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HITL.common.config import load_configs, parse_session_id, session_wind  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--campaign", type=Path, default=PROJECT_ROOT / "HITL/config/campaign.yaml"
    )
    parser.add_argument(
        "--local", type=Path, default=PROJECT_ROOT / "HITL/config/local.yaml"
    )
    parser.add_argument("--benchmark-runs", type=int, default=30)
    return parser.parse_args()


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _feature_count(scaler: Any) -> int:
    count = getattr(scaler, "n_features_in_", None)
    if count is None:
        mean = getattr(scaler, "mean_", None)
        if mean is None:
            raise ValueError("scaler_X has neither n_features_in_ nor mean_")
        count = len(mean)
    return int(count)


def _ntp_status(clock: dict[str, Any]) -> dict[str, Any]:
    command = str(clock.get("chrony_command", "chronyc tracking"))
    try:
        completed = subprocess.run(
            shlex.split(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = (completed.stdout + completed.stderr).strip()
        lowered = text.lower()
        synchronized = completed.returncode == 0 and (
            "leap status     : normal" in lowered
            or "leap status: normal" in lowered
            or "ntpsynchronized=yes" in lowered
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "synchronized": synchronized,
            "output": text,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "command": command,
            "returncode": None,
            "synchronized": False,
            "output": str(exc),
        }


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "ok": False,
        "session_id": args.session_id,
        "checks": {},
        "blockers": [],
    }
    blockers: list[str] = report["blockers"]

    def check(name: str, operation: Callable[[], Any]) -> Any:
        try:
            detail = operation()
            report["checks"][name] = {"ok": True, "detail": detail}
            return detail
        except Exception as exc:
            message = f"{name}: {exc}"
            blockers.append(message)
            report["checks"][name] = {"ok": False, "error": str(exc)}
            return None

    configs = check(
        "configuration",
        lambda: load_configs(args.campaign, args.local, require_local=True),
    )
    if configs is None:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    campaign, local = configs
    pi = local.get("pi", {})
    clock = local.get("clock", {})
    pi_root = _resolve(PROJECT_ROOT, pi.get("project_root", PROJECT_ROOT))

    def validate_session() -> dict[str, Any]:
        condition, index = parse_session_id(args.session_id)
        required = int(campaign["protocol"]["required_sessions_per_condition"])
        if not 1 <= index <= required:
            raise ValueError(f"session index must be in 01..{required:02d}")
        return {"condition": condition, "index": index, "wind": session_wind(campaign, args.session_id)}

    check("campaign_session", validate_session)

    serial_path = _resolve(pi_root, pi.get("telem_serial_device", ""))

    def validate_serial() -> dict[str, Any]:
        if not pi.get("telem_serial_device"):
            raise ValueError("pi.telem_serial_device is missing")
        if not serial_path.exists():
            raise FileNotFoundError(serial_path)
        readable = os.access(serial_path, os.R_OK)
        writable = os.access(serial_path, os.W_OK)
        if not (readable and writable):
            raise PermissionError(
                f"{serial_path} needs read/write access; verify dialout membership"
            )
        return {"path": str(serial_path), "readable": readable, "writable": writable}

    check("serial", validate_serial)

    model_path = _resolve(pi_root, pi.get("model_path", ""))
    norm_path = _resolve(pi_root, pi.get("norm_params_path", ""))
    deployment_path = _resolve(pi_root, pi.get("deployment_config", ""))

    deployment_config = check(
        "deployment_config_load",
        lambda: yaml.safe_load(deployment_path.read_text(encoding="utf-8")) or {},
    )

    def validate_deployment() -> dict[str, Any]:
        cfg = deployment_config
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        deploy_cfg = cfg.get("deployment", {})
        actual = {
            "input_size": int(model_cfg.get("input_size", -1)),
            "sequence_length": int(data_cfg.get("sequence_length", -1)),
            "backend": str(deploy_cfg.get("backend", "")).lower(),
            "inference_rate": float(deploy_cfg.get("inference_rate", -1)),
        }
        expected = {
            "input_size": 41,
            "sequence_length": 100,
            "backend": "onnx",
            "inference_rate": 50.0,
        }
        if actual != expected:
            raise ValueError(f"expected {expected}, got {actual}")
        return {"path": str(deployment_path), **actual}

    if deployment_config is not None:
        check("deployment_contract", validate_deployment)

    def validate_norm() -> dict[str, Any]:
        if not norm_path.is_file():
            raise FileNotFoundError(norm_path)
        with norm_path.open("rb") as stream:
            metadata = pickle.load(stream)
        count = _feature_count(metadata["scaler_X"])
        if count != 41:
            raise ValueError(f"scaler_X feature count is {count}, expected 41")
        return {"path": str(norm_path), "scaler_features": count}

    check("normalization", validate_norm)

    def validate_onnx() -> dict[str, Any]:
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        if session.get_providers()[0] != "CPUExecutionProvider":
            raise RuntimeError(f"CPU provider not active: {session.get_providers()}")
        model_input = session.get_inputs()[0]
        shape = model_input.shape
        if len(shape) != 3 or shape[1:] != [100, 41]:
            raise ValueError(f"input shape is {shape}, expected [batch, 100, 41]")

        sample = np.random.default_rng(0).standard_normal((1, 100, 41)).astype(np.float32)
        input_feed = {model_input.name: sample}
        session.run(None, input_feed)
        first = session.run(None, input_feed)
        second = session.run(None, input_feed)
        if len(first) != len(second) or any(
            not np.array_equal(left, right) for left, right in zip(first, second)
        ):
            raise RuntimeError("identical smoke inputs produced non-deterministic outputs")
        timings = []
        for _ in range(args.benchmark_runs):
            started_ns = time.perf_counter_ns()
            outputs = session.run(None, input_feed)
            timings.append((time.perf_counter_ns() - started_ns) / 1e6)
        if any(not np.all(np.isfinite(value)) for value in outputs):
            raise RuntimeError("smoke inference produced non-finite output")
        return {
            "path": str(model_path),
            "input_name": model_input.name,
            "input_shape": shape,
            "providers": session.get_providers(),
            "benchmark_runs": args.benchmark_runs,
            "latency_ms_mean": float(np.mean(timings)),
            "latency_ms_p95": float(np.percentile(timings, 95)),
            "latency_ms_max": float(np.max(timings)),
            "deterministic": True,
        }

    if args.benchmark_runs <= 0:
        blockers.append("onnx: --benchmark-runs must be positive")
    else:
        check("onnx", validate_onnx)

    ntp = _ntp_status(clock)
    ntp_required = bool(clock.get("require_ntp_synchronized", True))
    ntp_ok = ntp["synchronized"] or not ntp_required
    report["checks"]["ntp"] = {
        "ok": ntp_ok,
        "required": ntp_required,
        "detail": ntp,
    }
    if not ntp_ok:
        blockers.append("ntp: clock is not reported synchronized")

    report["ok"] = not blockers
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
