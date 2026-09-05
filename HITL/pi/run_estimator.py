#!/usr/bin/env python3
"""Run the legacy Pi estimator with canonical repeated-HITL logging."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
LEGACY_PATH = SRC_DIR / "6c_online_deployment_pigru-akf_rsbpi.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HITL.common.config import (  # noqa: E402
    SessionPaths,
    load_configs,
    parse_session_id,
    session_wind,
)
from HITL.common.schema import PI_REQUIRED_COLUMNS  # noqa: E402
from HITL.common.manifest import sha256_file, software_manifest, write_json_atomic  # noqa: E402


CLOCK_LISTENER_MARGIN_S = 70.0
# Paper 50 Hz budget is only a latency threshold, not a loop cap.
PAPER_DEADLINE_MS = 20.0


DIAGNOSTIC_COLUMNS = (
    "q_scale_n",
    "q_scale_e",
    "q_scale_d",
    "r_scale_gps",
    "r_scale_tas",
    "r_scale_att",
    "angle_alpha_rad",
    "angle_beta_rad",
    "tas_scale",
    "confidence",
    "groundspeed_mps",
    "airspeed_mps",
    "sensor_hold",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--campaign", type=Path, default=PROJECT_ROOT / "HITL/config/campaign.yaml"
    )
    parser.add_argument(
        "--local", type=Path, default=PROJECT_ROOT / "HITL/config/local.yaml"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="override campaign duration"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="session Pi directory (default: campaign sessions/<id>/pi)",
    )
    return parser.parse_args()


def _format_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _progress_bar(elapsed: float, duration: float, width: int = 28) -> str:
    if duration <= 0:
        return "#" * width
    ratio = min(1.0, max(0.0, elapsed / duration))
    filled = int(round(width * ratio))
    return "#" * filled + "-" * (width - filled)


def _print_run_banner(session_id: str, duration: float) -> None:
    listener_s = int(round(duration + CLOCK_LISTENER_MARGIN_S))
    print("=" * 72, flush=True)
    print(f" HITL estimator  session={session_id}  duration={duration:.0f}s", flush=True)
    print("=" * 72, flush=True)
    print(
        f"循环按树莓派最大能力推理，不限 50 Hz；deadline 仍按 {PAPER_DEADLINE_MS:.0f} ms。\n"
        f"进度条到 100% 后本进程会自动退出。随后立刻：\n"
        f"  1. PC: session_marker.py {session_id} --phase end\n"
        f"  2. PC: 停止 JSBSim (Ctrl+C)\n"
        f"  3. 下载本次 ULog，再进行下一次会话\n"
        f"时钟监听建议 --duration {listener_s}，须覆盖开始探针+{duration:.0f}s估计+结束探针。",
        flush=True,
    )
    print("=" * 72, flush=True)


def _print_run_complete(session_id: str, duration: float) -> None:
    print("\n" + "=" * 72, flush=True)
    print(f" 估计器已跑满 {duration:.0f}s，可以做后续收尾。", flush=True)
    print("=" * 72, flush=True)
    print(
        f"下一步（保持当前飞行不要改风场）：\n"
        f"  1. PC: .venv/bin/python HITL/pc/session_marker.py {session_id} "
        f"--host <PI_IP> --phase end\n"
        f"  2. QGC 结束任务后，PC 终端对 run_jsbsim.py 按 Ctrl+C\n"
        f"  3. 保存本次 ULog，从树莓派取回 pi_estimator.csv / 时钟探针 / manifest\n"
        f"  4. 再进入会话顺序中的下一次",
        flush=True,
    )
    print("=" * 72, flush=True)


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_legacy_module() -> Any:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    spec = importlib.util.spec_from_file_location("hitl_legacy_estimator", LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy estimator: {LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _latest_fc_boot_time_us(msg_dict: dict[str, Any]) -> int:
    timestamps: list[int] = []
    for name in ("GLOBAL_POSITION_INT", "ATTITUDE", "VFR_HUD", "HIGHRES_IMU"):
        message = msg_dict.get(name)
        if message is None:
            continue
        time_usec = int(getattr(message, "time_usec", 0) or 0)
        time_boot_ms = int(getattr(message, "time_boot_ms", 0) or 0)
        if time_usec > 0:
            timestamps.append(time_usec)
        elif time_boot_ms > 0:
            timestamps.append(time_boot_ms * 1_000)
    return max(timestamps, default=0)


def _patched_config(
    base: dict[str, Any],
    *,
    serial_device: Path,
    baudrate: int,
    model_path: Path,
    norm_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.setdefault("model", {})["input_size"] = 41
    config.setdefault("data", {})["sequence_length"] = 100
    config.setdefault("data", {})["sampling_rate"] = 50
    config.setdefault("training", {})["model_save_path"] = str(norm_path.parent)
    deployment = config.setdefault("deployment", {})
    deployment.update(
        {
            "backend": "onnx",
            "mavlink_connection": str(serial_device),
            "baudrate": int(baudrate),
            "model_path_onnx": str(model_path),
            "inference_rate": 50,
            "use_gpu": False,
        }
    )
    logging_config = config.setdefault("logging", {})
    logging_config["log_dir"] = str(output_dir)
    logging_config["save_dir"] = str(output_dir)
    return config


def _build_adapter(
    legacy: Any,
    *,
    output_dir: Path,
    csv_name: str,
    session_id: str,
    condition: str,
) -> type:
    class CampaignEstimator(legacy.OnlineWindEstimator):
        def __init__(self, config_path: str) -> None:
            self._campaign_output_dir = output_dir
            self._campaign_csv_name = csv_name
            self._campaign_session_id = session_id
            self._campaign_condition = condition
            try:
                super().__init__(config_path=config_path)
            except Exception:
                csv_file = getattr(self, "csv_file", None)
                if csv_file is not None and not csv_file.closed:
                    csv_file.close()
                raise

        def _init_hitl_progress(self) -> None:
            if getattr(self, "_hitl_progress_ready", False):
                return
            self._hitl_progress_ready = True
            self._hitl_last_log = 0.0
            self._hitl_ok = 0
            self._hitl_hold = 0
            self._hitl_miss = 0
            self._hitl_last_held = 0
            self._hitl_last_missing: list[str] = []
            self._hitl_announced_first_frame = False
            self._hitl_announced_first_row = False
            self.logger.info(
                "HITL progress: 每5秒写一行。缓存齐四条后按树莓派最大能力推理，"
                "不人为限到 50 Hz；deadline 仍按论文 20 ms 统计。"
                "缺新包则保持（sensor_hold=1）。"
                "mav_ok=四条都刷新 mav_hold=用缓存 mav_miss=尚未齐套。"
            )

        def _maybe_log_progress(self, result: dict[str, Any] | None = None) -> None:
            self._init_hitl_progress()
            now = time.monotonic()
            if now - self._hitl_last_log < 5.0:
                return
            self._hitl_last_log = now
            duration = float(getattr(self, "_campaign_duration", 0.0) or 0.0)
            start = float(self.performance.get("start_time") or time.time())
            runtime = max(0.0, time.time() - start)
            remaining = max(0.0, duration - runtime) if duration else 0.0
            pct = min(100.0, 100.0 * runtime / duration) if duration else 0.0
            buffer_n = len(getattr(self, "data_buffer", []) or [])
            seq = int(getattr(self, "sequence_length", 100))
            rows = int(self.performance.get("inference_count", 0))
            wind_txt = ""
            wind = None
            if result is not None:
                wind = np.asarray(result["wind_estimate"], dtype=float).reshape(-1)
            elif getattr(self, "last_wind_estimate", None) is not None:
                wind = np.asarray(self.last_wind_estimate, dtype=float).reshape(-1)
            if wind is not None:
                wind_txt = f"  wind=[{wind[0]:+.2f},{wind[1]:+.2f},{wind[2]:+.2f}]"
            missing_txt = ""
            if self._hitl_last_missing:
                missing_txt = f"  missing={self._hitl_last_missing}"
            counts = getattr(self, "_msg_counts", {}) or {}
            seen = ", ".join(
                f"{name}:{counts[name]}"
                for name in (
                    "HEARTBEAT",
                    "VFR_HUD",
                    "ATTITUDE",
                    "GLOBAL_POSITION_INT",
                    "HIGHRES_IMU",
                    "SERVO_OUTPUT_RAW",
                )
                if counts.get(name, 0)
            ) or "none"
            self.logger.info(
                "HITL progress %6.1f/%.0fs %5.1f%% ETA %s  "
                "buffer=%d/%d  csv_rows=%d  mav_ok=%d mav_hold=%d mav_miss=%d%s%s  recv=[%s]",
                runtime,
                duration,
                pct,
                _format_hms(remaining),
                buffer_n,
                seq,
                rows,
                self._hitl_ok,
                self._hitl_hold,
                self._hitl_miss,
                wind_txt,
                missing_txt,
                seen,
            )
            self._hitl_ok = 0
            self._hitl_hold = 0
            self._hitl_miss = 0

        def connect_mavlink(self) -> bool:
            from pymavlink import mavutil

            wait_s = 20.0
            self.logger.info("Connecting MAVLink: %s", self.mavlink_connection)
            try:
                if str(self.mavlink_connection).startswith("/dev/"):
                    self.connection = mavutil.mavlink_connection(
                        str(self.mavlink_connection), baud=int(self.baudrate)
                    )
                else:
                    self.connection = mavutil.mavlink_connection(
                        str(self.mavlink_connection)
                    )

                self.logger.info(
                    "HITL 等待飞控 HEARTBEAT，最多 %.0f 秒。"
                    "进度条和 csv 都要等心跳成功后才会出现。",
                    wait_s,
                )
                port = getattr(self.connection, "port", None)
                if port is not None:
                    time.sleep(2.0)
                    waiting = int(getattr(port, "in_waiting", 0) or 0)
                    sample = port.read(min(waiting, 64)) if waiting else b""
                    preview = sample[:16].hex(" ")
                    mavlike = any(byte in sample for byte in (0xFD, 0xFE))
                    self.logger.info(
                        "HITL 串口探测 2s: bytes=%d mavlink_magic=%s hex=%s",
                        len(sample),
                        mavlike,
                        preview or "(empty)",
                    )
                    if not sample:
                        self.logger.error(
                            "HITL 串口 2 秒内 0 字节：不是软件解析问题，"
                            "是 TELEM2 没有把数据送到 %s",
                            self.mavlink_connection,
                        )
                started = time.monotonic()
                heartbeat = None
                while time.monotonic() - started < wait_s:
                    heartbeat = self.connection.wait_heartbeat(timeout=2.0)
                    if heartbeat is not None:
                        break
                    self.logger.info(
                        "HITL 仍未收到 HEARTBEAT（已等 %.0fs / %.0fs）。"
                        "确认 JSBSim 在跑、飞控已上电、TELEM2 交叉接线、"
                        "MAV_1_CONFIG=TELEM2，且没有其它程序占用 %s",
                        time.monotonic() - started,
                        wait_s,
                        self.mavlink_connection,
                    )
                if heartbeat is None:
                    self.logger.error(
                        "HITL MAVLink 超时：%s 在 %.0f 秒内没有 HEARTBEAT",
                        self.mavlink_connection,
                        wait_s,
                    )
                    return False

                self.logger.info(
                    "HITL MAVLink 已连接 sys=%s comp=%s",
                    self.connection.target_system,
                    self.connection.target_component,
                )
                self.connection.mav.request_data_stream_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL,
                    50,
                    1,
                )
                required_msg_ids = {
                    "GLOBAL_POSITION_INT": 33,
                    "ATTITUDE": 30,
                    "VFR_HUD": 74,
                    "HIGHRES_IMU": 105,
                    "SERVO_OUTPUT_RAW": 36,
                    "RC_CHANNELS": 65,
                    "ATTITUDE_TARGET": 83,
                    "POSITION_TARGET_LOCAL_NED": 85,
                }
                for msg_id in required_msg_ids.values():
                    self.connection.mav.command_long_send(
                        self.connection.target_system,
                        self.connection.target_component,
                        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                        0,
                        msg_id,
                        20000,
                        0,
                        0,
                        0,
                        0,
                        0,
                    )
                self.logger.info(
                    "Data streams requested (50 Hz): %s",
                    list(required_msg_ids),
                )
                return True
            except Exception as exc:
                self.logger.error("MAVLink connection failed: %s", exc)
                return False

        def collect_mavlink_data(self) -> Any:
            """Drain TELEM2 briefly, then estimate from the latest cached set.

            Campaign HITL measures Pi inference throughput at inference_rate.
            Required messages need only be cached once; a cycle without a fresh
            packet keeps the last measurement instead of blocking up to 1 s.
            """
            self._init_hitl_progress()
            required = (
                "GLOBAL_POSITION_INT",
                "ATTITUDE",
                "VFR_HUD",
                "HIGHRES_IMU",
            )
            fresh: set[str] = set()

            def _ingest(message: Any) -> None:
                name = message.get_type()
                if name == "BAD_DATA":
                    return
                self._latest_msgs[name] = message
                self._msg_counts[name] = self._msg_counts.get(name, 0) + 1
                if name in required:
                    fresh.add(name)

            while True:
                message = self.connection.recv_match(blocking=False)
                if message is None:
                    break
                _ingest(message)

            latest = getattr(self, "_latest_msgs", {})
            if any(name not in latest for name in required):
                deadline = time.monotonic() + 0.004
                while time.monotonic() < deadline:
                    message = self.connection.recv_match(blocking=False)
                    if message is None:
                        time.sleep(0.0004)
                        continue
                    _ingest(message)
                    if fresh.issuperset(required):
                        break

            latest = getattr(self, "_latest_msgs", {})
            missing = [name for name in required if name not in latest]
            self._hitl_last_missing = missing
            if missing:
                self._hitl_miss += 1
                self._hitl_last_held = 0
                self._maybe_log_progress()
                return None

            held = int(not fresh.issuperset(required))
            self._hitl_last_held = held
            if held:
                self._hitl_hold += 1
            else:
                self._hitl_ok += 1
            if not self._hitl_announced_first_frame:
                self._hitl_announced_first_frame = True
                self.logger.info(
                    "HITL 必需消息已齐套，之后按最大能力估计（缺新包则保持）: %s",
                    ", ".join(sorted(k for k in latest if k != "BAD_DATA")),
                )
            self._maybe_log_progress()
            return dict(latest)

        def run(self, phase: str = "steady", duration: float | None = None) -> None:
            old_rate = float(self.inference_rate)
            # Parent sleeps to 1/inference_rate. A huge rate removes that cap.
            self.inference_rate = 1_000_000.0
            try:
                self.logger.info(
                    "HITL 不限循环频率，按最大能力推理；论文 deadline 仍为 %.1f ms",
                    PAPER_DEADLINE_MS,
                )
                super().run(phase=phase, duration=duration)
            finally:
                self.inference_rate = old_rate

        def _init_csv_logger(self) -> None:
            self._campaign_output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = self._campaign_output_dir / self._campaign_csv_name
            self.csv_file = csv_path.open("w", encoding="utf-8", newline="")
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=[*PI_REQUIRED_COLUMNS, *DIAGNOSTIC_COLUMNS],
                extrasaction="raise",
            )
            self.csv_writer.writeheader()
            self.csv_file.flush()
            self.logger.info("Canonical Pi CSV data file: %s", csv_path)

        def _log_csv(
            self,
            result: dict[str, Any],
            msg_dict: dict[str, Any],
            t_recv_ns: int | None = None,
            phase: str = "steady",
        ) -> None:
            del phase
            if self.csv_writer is None:
                return
            output_mono_ns = time.monotonic_ns()
            output_wall_ns = time.time_ns()
            receive_ns = output_mono_ns if t_recv_ns is None else int(t_recv_ns)
            processing_ms = max(0.0, (output_mono_ns - receive_ns) / 1e6)
            inference_ms = float(result["inference_time"]) * 1_000.0
            deadline_ms = PAPER_DEADLINE_MS
            final_wind = np.asarray(result["wind_estimate"], dtype=float).reshape(-1)
            nn_wind = np.asarray(result["wind_nn"], dtype=float).reshape(-1)
            q_scale = np.asarray(result["q_scale"], dtype=float).reshape(-1)
            r_scale = np.asarray(result["r_scale"], dtype=float).reshape(-1)
            angles = np.asarray(result["angles"], dtype=float).reshape(-1)

            gps = msg_dict.get("GLOBAL_POSITION_INT")
            hud = msg_dict.get("VFR_HUD")
            groundspeed = float("nan")
            airspeed = float("nan")
            if gps is not None:
                groundspeed = float(
                    np.linalg.norm(
                        [gps.vx / 100.0, gps.vy / 100.0, gps.vz / 100.0]
                    )
                )
            if hud is not None:
                airspeed = float(hud.airspeed)

            row = {
                "session_id": self._campaign_session_id,
                "condition": self._campaign_condition,
                "pi_wall_time_ns": output_wall_ns,
                "pi_monotonic_ns": output_mono_ns,
                "fc_boot_time_us": _latest_fc_boot_time_us(msg_dict),
                "estimated_wind_n_mps": final_wind[0],
                "estimated_wind_e_mps": final_wind[1],
                "estimated_wind_d_mps": final_wind[2],
                "nn_wind_n_mps": nn_wind[0],
                "nn_wind_e_mps": nn_wind[1],
                "nn_wind_d_mps": nn_wind[2],
                "inference_latency_ms": inference_ms,
                "companion_processing_latency_ms": processing_ms,
                "deadline_missed": int(processing_ms > deadline_ms),
                "q_scale_n": q_scale[0],
                "q_scale_e": q_scale[1],
                "q_scale_d": q_scale[2],
                "r_scale_gps": r_scale[0],
                "r_scale_tas": r_scale[1],
                "r_scale_att": r_scale[2],
                "angle_alpha_rad": angles[0],
                "angle_beta_rad": angles[1],
                "tas_scale": angles[2],
                "confidence": float(result.get("confidence", float("nan"))),
                "groundspeed_mps": groundspeed,
                "airspeed_mps": airspeed,
                "sensor_hold": int(getattr(self, "_hitl_last_held", 0)),
            }
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            if not getattr(self, "_hitl_announced_first_row", False):
                self._hitl_announced_first_row = True
                self.logger.info(
                    "HITL 已写入首行 CSV，正式记录开始 session=%s",
                    self._campaign_session_id,
                )

        def print_status(self, result: dict[str, Any]) -> None:
            self._maybe_log_progress(result)
            duration = float(getattr(self, "_campaign_duration", 0.0) or 0.0)
            runtime = time.time() - self.performance["start_time"]
            wind = np.asarray(result["wind_estimate"], dtype=float).reshape(-1)
            wind_mag = float(np.linalg.norm(wind))
            count = int(self.performance.get("inference_count", 0))
            if duration > 0:
                remaining = max(0.0, duration - runtime)
                pct = min(100.0, 100.0 * runtime / duration)
                timer = (
                    f"{runtime:6.1f}/{duration:.0f}s "
                    f"[{_progress_bar(runtime, duration)}] {pct:5.1f}% "
                    f"ETA {_format_hms(remaining)}"
                )
            else:
                timer = f"Runtime {runtime:.1f}s"
            sys.stdout.write(
                f"\r{timer}  n={count:<6d}  "
                f"W=[{wind[0]:+5.2f},{wind[1]:+5.2f},{wind[2]:+5.2f}] "
                f"|{wind_mag:5.2f} m/s   "
            )
            sys.stdout.flush()

    CampaignEstimator.__name__ = "CampaignEstimator"
    return CampaignEstimator


def _close_estimator(estimator: Any) -> None:
    csv_file = getattr(estimator, "csv_file", None)
    if csv_file is not None and not csv_file.closed:
        csv_file.flush()
        csv_file.close()
    backend = getattr(estimator, "backend", None)
    if backend is not None:
        backend.close()
        estimator.backend = None
    connection = getattr(estimator, "connection", None)
    if connection is not None:
        try:
            connection.close()
        finally:
            estimator.connection = None


def main() -> int:
    args = parse_args()
    campaign, local = load_configs(args.campaign, args.local, require_local=True)
    condition, index = parse_session_id(args.session_id)
    required = int(campaign["protocol"]["required_sessions_per_condition"])
    if not 1 <= index <= required:
        raise SystemExit(f"session index must be in 01..{required:02d}")
    session_wind(campaign, args.session_id)
    if args.duration is not None and args.duration <= 0:
        raise SystemExit("--duration must be positive")

    pi = local.get("pi", {})
    pi_root = _resolve(PROJECT_ROOT, pi.get("project_root", PROJECT_ROOT))
    deployment_path = _resolve(pi_root, pi["deployment_config"])
    model_path = _resolve(pi_root, pi["model_path"])
    norm_path = _resolve(pi_root, pi["norm_params_path"])
    serial_device = _resolve(pi_root, pi["telem_serial_device"])
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else SessionPaths.build(campaign, args.session_id, create=True).pi
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "pi_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")

    with deployment_path.open("r", encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream) or {}
    runtime_config = _patched_config(
        base_config,
        serial_device=serial_device,
        baudrate=int(pi.get("telem_baudrate", 921600)),
        model_path=model_path,
        norm_path=norm_path,
        output_dir=output_dir,
    )

    temporary_path: Path | None = None
    estimator = None
    old_session = os.environ.get("HITL_SESSION_ID")
    old_condition = os.environ.get("HITL_CONDITION")
    manifest = {
        **software_manifest(),
        "schema_version": 1,
        "session_id": args.session_id,
        "condition": condition,
        "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "normalization_path": str(norm_path),
        "normalization_sha256": sha256_file(norm_path),
        "deployment_config_path": str(deployment_path),
        "config_sha256": sha256_file(deployment_path),
        "serial_device": str(serial_device),
        "baudrate": int(pi.get("telem_baudrate", 921600)),
    }
    write_json_atomic(manifest_path, manifest)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".yaml",
            prefix=".runtime_estimator_",
            dir=Path(__file__).resolve().parent,
            delete=False,
        ) as temporary:
            yaml.safe_dump(runtime_config, temporary, sort_keys=False)
            temporary_path = Path(temporary.name)

        os.environ["HITL_SESSION_ID"] = args.session_id
        os.environ["HITL_CONDITION"] = condition
        legacy = _load_legacy_module()
        adapter = _build_adapter(
            legacy,
            output_dir=output_dir,
            csv_name=str(campaign["logging"]["pi_csv_name"]),
            session_id=args.session_id,
            condition=condition,
        )
        estimator = adapter(config_path=str(temporary_path))
        duration = (
            float(args.duration)
            if args.duration is not None
            else float(campaign["protocol"]["total_duration_s"])
        )
        estimator._campaign_duration = duration
        _print_run_banner(args.session_id, duration)
        run_started = time.monotonic()
        estimator.run(phase="steady", duration=duration)
        if time.monotonic() - run_started + 1.0 >= duration:
            _print_run_complete(args.session_id, duration)
        else:
            print(
                "\n估计器提前退出，不要当作一次有效会话。"
                "请归档本次目录后再重跑。",
                flush=True,
            )
        manifest["status"] = "completed"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(manifest_path, manifest)
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = str(exc)
        write_json_atomic(manifest_path, manifest)
        raise
    finally:
        if estimator is not None:
            _close_estimator(estimator)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if old_session is None:
            os.environ.pop("HITL_SESSION_ID", None)
        else:
            os.environ["HITL_SESSION_ID"] = old_session
        if old_condition is None:
            os.environ.pop("HITL_CONDITION", None)
        else:
            os.environ["HITL_CONDITION"] = old_condition


if __name__ == "__main__":
    raise SystemExit(main())
