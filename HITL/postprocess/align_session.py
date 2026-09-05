"""Align Pi estimates to PC truth using only explicit network clock probes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HITL.common.config import DEFAULT_CAMPAIGN, DEFAULT_LOCAL, SessionPaths, load_configs
from HITL.common.manifest import write_json_atomic
from HITL.common.schema import (
    PI_REQUIRED_COLUMNS,
    TRUTH_REQUIRED_COLUMNS,
    require_columns,
    require_constant_text,
    require_finite,
)
from HITL.postprocess._util import find_one, first_key, read_jsonl


PC_SEND_KEYS = ("t1_pc_send_ns", "pc_send_wall_time_ns", "pc_send_ns")
PC_RECV_KEYS = ("t4_pc_recv_ns", "pc_receive_wall_time_ns", "pc_recv_ns")
PI_RECV_KEYS = ("t2_pi_recv_ns", "pi_receive_wall_time_ns")
PI_SEND_KEYS = ("t3_pi_send_ns", "pi_response_wall_time_ns")
SESSION_KEYS = ("session_id", "session")


def _probe_rows(files: Iterable[Path], session_id: str) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for path in files:
        for number, record in enumerate(read_jsonl(path), 1):
            source = f"{path}:{number}"
            if record.get("ok") is False:
                continue
            marker_session = next((str(record[k]).lower() for k in SESSION_KEYS if k in record), None)
            if marker_session is not None and marker_session != session_id.lower():
                raise ValueError(f"{source}: session {marker_session!r} != {session_id!r}")
            t1 = first_key(record, PC_SEND_KEYS, source)
            t4 = first_key(record, PC_RECV_KEYS, source)
            t2 = first_key(record, PI_RECV_KEYS, source)
            t3 = first_key(record, PI_SEND_KEYS, source)
            if t4 < t1:
                raise ValueError(f"{source}: PC receive precedes send")
            if t3 < t2:
                raise ValueError(f"{source}: Pi send precedes receive")
            rtt = (t4 - t1) - (t3 - t2)
            if rtt < 0:
                raise ValueError(f"{source}: negative network RTT")
            midpoint = (t1 + t4) / 2.0
            offset = ((t2 - t1) + (t3 - t4)) / 2.0
            rows.append(
                {
                    "pc_mid_ns": midpoint,
                    "pi_minus_pc_ns": offset,
                    "rtt_ns": rtt,
                }
            )
    return pd.DataFrame(rows)


def fit_clock_model(
    probes: pd.DataFrame,
    *,
    minimum_span_s: float = 30.0,
    minimum_fit_probes: int = 4,
) -> dict[str, Any]:
    """Fit pi-PC = intercept + slope*(PC-reference), after low-RTT selection."""
    if len(probes) < 2:
        raise ValueError("at least two valid clock probes are required")
    for column in ("pc_mid_ns", "pi_minus_pc_ns", "rtt_ns"):
        values = pd.to_numeric(probes[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"clock probes contain non-finite {column}")
    cutoff = float(np.quantile(probes["rtt_ns"], 0.5))
    selected = probes.loc[probes["rtt_ns"] <= cutoff].copy()
    if len(selected) < 2:
        selected = probes.nsmallest(2, "rtt_ns").copy()
    reference = float(np.median(selected["pc_mid_ns"]))
    x = selected["pc_mid_ns"].to_numpy(float) - reference
    y = selected["pi_minus_pc_ns"].to_numpy(float)
    span_s = float(np.ptp(x) / 1e9)
    linear = len(selected) >= minimum_fit_probes and span_s >= minimum_span_s
    if linear:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0.0, float(np.median(y))
    residual = y - (intercept + slope * x)
    robust_sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    uncertainty_ns = max(
        robust_sigma,
        float(np.quantile(selected["rtt_ns"], 0.95)) / 2.0,
    )
    return {
        "kind": "linear" if linear else "constant",
        "reference_pc_wall_ns": reference,
        "intercept_pi_minus_pc_ns": float(intercept),
        "slope_ns_per_ns": float(slope),
        "drift_ppm": float(slope * 1e6),
        "uncertainty_ms": float(uncertainty_ns / 1e6),
        "probe_count": int(len(probes)),
        "selected_probe_count": int(len(selected)),
        "selected_rtt_p95_ms": float(np.quantile(selected["rtt_ns"], 0.95) / 1e6),
        "selected_span_s": span_s,
    }


def pi_to_pc_wall_ns(pi_wall_ns: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    reference = float(model["reference_pc_wall_ns"])
    intercept = float(model["intercept_pi_minus_pc_ns"])
    slope = float(model["slope_ns_per_ns"])
    return reference + (np.asarray(pi_wall_ns, float) - reference - intercept) / (1.0 + slope)


def _merge_nearest(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    tolerance: float,
    suffix: str = "",
) -> pd.DataFrame:
    left = left.copy()
    right = right.copy()
    left[left_on] = pd.to_numeric(left[left_on], errors="coerce").astype(float)
    right[right_on] = pd.to_numeric(right[right_on], errors="coerce").astype(float)
    merged = pd.merge_asof(
        left.sort_values(left_on, kind="mergesort"),
        right.sort_values(right_on, kind="mergesort"),
        left_on=left_on,
        right_on=right_on,
        direction="nearest",
        tolerance=float(tolerance),
        suffixes=("", suffix),
    )
    # FC boot times can repeat when the estimator holds a sample; restore Pi order.
    if "pi_monotonic_ns" in merged.columns:
        merged = merged.sort_values("pi_monotonic_ns", kind="mergesort")
    return merged


def _export_ulog(ulog: Path, destination: Path) -> Path | None:
    executable = shutil.which("ulog2csv")
    if executable is None:
        return None
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [executable, str(ulog), "-o", str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    candidates: list[Path] = []
    for candidate in destination.glob("*.csv"):
        try:
            columns = pd.read_csv(candidate, nrows=0).columns
        except (OSError, pd.errors.ParserError):
            continue
        if any(name in columns for name in ("fc_boot_time_us", "timestamp", "timestamp_us")):
            candidates.append(candidate)
    # ULog normally contains many topics. A timestamp-only alignment check does
    # not privilege a flight signal, so use the largest timestamped topic.
    return sorted(candidates, key=lambda path: (-path.stat().st_size, path.name))[0] if candidates else None


def _merge_fc(
    aligned: pd.DataFrame,
    fc_csv: Path,
    tolerance_us: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fc = pd.read_csv(fc_csv)
    timestamp_column = next(
        (name for name in ("fc_boot_time_us", "timestamp", "timestamp_us") if name in fc.columns),
        None,
    )
    if timestamp_column is None:
        raise ValueError(f"{fc_csv}: no FC boot timestamp column")
    values = pd.to_numeric(fc[timestamp_column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError(f"{fc_csv}: FC timestamps must be finite and strictly increasing")
    fc = fc.rename(columns={timestamp_column: "_fc_topic_time_us"})
    fc = fc.rename(
        columns={
            column: f"fc_{column}"
            for column in fc.columns
            if column != "_fc_topic_time_us" and not column.startswith("fc_")
        }
    )
    result = _merge_nearest(
        aligned,
        fc,
        left_on="fc_boot_time_us",
        right_on="_fc_topic_time_us",
        tolerance=tolerance_us,
        suffix="_topic",
    )
    residual = np.abs(result["fc_boot_time_us"] - result["_fc_topic_time_us"]) / 1000.0
    result["fc_alignment_residual_ms"] = residual
    matched = residual.notna()
    report = {
        "source": str(fc_csv.resolve()),
        "coverage": float(matched.mean()),
        "matched_rows": int(matched.sum()),
        "residual_p95_ms": float(residual[matched].quantile(0.95)) if matched.any() else None,
        "residual_max_ms": float(residual[matched].max()) if matched.any() else None,
    }
    return result, report


def align_session(
    session_id: str,
    *,
    campaign_path: str | Path = DEFAULT_CAMPAIGN,
    local_path: str | Path = DEFAULT_LOCAL,
    fc_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    campaign, _ = load_configs(campaign_path, local_path, require_local=False)
    paths = SessionPaths.build(campaign, session_id)
    logging = campaign["logging"]
    pi_path = paths.pi / logging["pi_csv_name"]
    truth_path = paths.pc / logging["pc_truth_name"]
    markers = [
        find_one(paths.pc, (f"marker_{phase}.jsonl", f"*{phase}*probe*.jsonl"), required=True)
        for phase in ("start", "end")
    ]
    pi = pd.read_csv(pi_path)
    truth = pd.read_csv(truth_path)
    require_columns(pi, PI_REQUIRED_COLUMNS, source=str(pi_path))
    require_columns(truth, TRUTH_REQUIRED_COLUMNS, source=str(truth_path))
    require_finite(pi, [c for c in PI_REQUIRED_COLUMNS if c not in ("session_id", "condition")], source=str(pi_path))
    require_finite(truth, TRUTH_REQUIRED_COLUMNS, source=str(truth_path))
    require_constant_text(pi, "session_id", session_id, source=str(pi_path))
    require_constant_text(pi, "condition", session_id.split("_")[0], source=str(pi_path))

    probe_frame = _probe_rows([path for path in markers if path is not None], session_id)
    clock = fit_clock_model(probe_frame)
    pi = pi.copy()
    pi["_pc_wall_ns"] = pi_to_pc_wall_ns(pi["pi_wall_time_ns"].to_numpy(float), clock)
    truth = truth.rename(
        columns={
            "wall_time_usec": "truth_time_us",
            "total_wind_north_ms": "truth_wind_n_mps",
            "total_wind_east_ms": "truth_wind_e_mps",
            "total_wind_down_ms": "truth_wind_d_mps",
        }
    )
    truth["_truth_pc_wall_ns"] = truth["truth_time_us"].astype(float) * 1000.0
    tolerance_ms = float(campaign["alignment"]["max_truth_match_residual_ms"])
    aligned = _merge_nearest(
        pi,
        truth,
        left_on="_pc_wall_ns",
        right_on="_truth_pc_wall_ns",
        tolerance=tolerance_ms * 1e6,
    )
    aligned["truth_alignment_residual_ms"] = (
        np.abs(aligned["_pc_wall_ns"] - aligned["_truth_pc_wall_ns"]) / 1e6
    )
    truth_matched = aligned["truth_time_us"].notna()
    report: dict[str, Any] = {
        "schema_version": 1,
        "session_id": session_id,
        "method": "low-RTT marker probes; no estimator-response correlation",
        "clock_model": clock,
        "truth": {
            "coverage": float(truth_matched.mean()),
            "matched_rows": int(truth_matched.sum()),
            "residual_p95_ms": (
                float(aligned.loc[truth_matched, "truth_alignment_residual_ms"].quantile(0.95))
                if truth_matched.any()
                else None
            ),
            "residual_max_ms": (
                float(aligned.loc[truth_matched, "truth_alignment_residual_ms"].max())
                if truth_matched.any()
                else None
            ),
        },
        "fc": None,
    }

    selected_fc = Path(fc_csv).resolve() if fc_csv else find_one(paths.fc, ("*.csv",), required=False)
    if selected_fc is None:
        ulog = paths.fc / logging["fc_ulog_name"]
        if ulog.exists():
            with tempfile.TemporaryDirectory(prefix="hitl_ulog_") as temporary:
                exported = _export_ulog(ulog, Path(temporary))
                if exported:
                    aligned, report["fc"] = _merge_fc(
                        aligned,
                        exported,
                        float(campaign["alignment"]["max_fc_match_residual_ms"]) * 1000.0,
                    )
                else:
                    report["fc"] = {"source": str(ulog), "status": "ulog2csv unavailable or ambiguous export"}
    else:
        aligned, report["fc"] = _merge_fc(
            aligned,
            selected_fc,
            float(campaign["alignment"]["max_fc_match_residual_ms"]) * 1000.0,
        )

    aligned = aligned.drop(columns=["_pc_wall_ns", "_truth_pc_wall_ns"], errors="ignore")
    output = paths.aligned / logging["aligned_csv_name"]
    output.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output, index=False)
    write_json_atomic(paths.aligned / "alignment_report.json", report)
    return aligned, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id")
    parser.add_argument("--campaign", default=str(DEFAULT_CAMPAIGN))
    parser.add_argument("--local", default=str(DEFAULT_LOCAL))
    parser.add_argument("--fc-csv")
    args = parser.parse_args()
    _, report = align_session(
        args.session_id,
        campaign_path=args.campaign,
        local_path=args.local,
        fc_csv=args.fc_csv,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
