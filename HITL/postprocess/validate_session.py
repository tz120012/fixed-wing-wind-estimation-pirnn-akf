"""Strictly validate an aligned HITL session without silently dropping rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HITL.common.config import DEFAULT_CAMPAIGN, DEFAULT_LOCAL, SessionPaths, load_configs
from HITL.common.manifest import write_json_atomic
from HITL.common.schema import (
    ALIGNED_REQUIRED_COLUMNS,
    require_columns,
    require_constant_text,
)
from HITL.postprocess._util import read_json


def _hash_values(value: Any, output: dict[str, set[str]]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            is_digest = "hash" in normalized or "sha256" in normalized
            if is_digest and "model" in normalized:
                output.setdefault("model_hash", set()).add(str(child))
            elif is_digest and "config" in normalized:
                output.setdefault("config_hash", set()).add(str(child))
            else:
                _hash_values(child, output)
    elif isinstance(value, list):
        for child in value:
            _hash_values(child, output)


def validate_session(
    session_id: str,
    *,
    campaign_path: str | Path = DEFAULT_CAMPAIGN,
    local_path: str | Path = DEFAULT_LOCAL,
) -> dict[str, Any]:
    campaign, _ = load_configs(campaign_path, local_path, require_local=False)
    paths = SessionPaths.build(campaign, session_id)
    aligned_path = paths.aligned / campaign["logging"]["aligned_csv_name"]
    report_path = paths.aligned / "alignment_report.json"
    output_path = paths.summary / "validation.json"
    errors: list[str] = []
    metrics: dict[str, Any] = {}

    try:
        frame = pd.read_csv(aligned_path)
        require_columns(frame, ALIGNED_REQUIRED_COLUMNS, source=str(aligned_path))
        if frame.empty:
            raise ValueError("aligned CSV is empty")
        require_constant_text(frame, "session_id", session_id, source=str(aligned_path))
        require_constant_text(frame, "condition", session_id.split("_")[0], source=str(aligned_path))
    except Exception as exc:
        errors.append(str(exc))
        result = {"schema_version": 1, "session_id": session_id, "valid": False, "errors": errors, "metrics": metrics}
        write_json_atomic(output_path, result)
        return result

    numeric_columns = [name for name in ALIGNED_REQUIRED_COLUMNS if name not in ("session_id", "condition")]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(numeric.to_numpy(float))
    if invalid.any():
        row, column = np.argwhere(invalid)[0]
        errors.append(f"non-finite value at row {int(row)}, column {numeric_columns[int(column)]}; rows are never dropped")

    for column in ("pi_wall_time_ns", "pi_monotonic_ns"):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
            errors.append(f"{column} must be finite and strictly increasing")
    fc_boot = pd.to_numeric(frame["fc_boot_time_us"], errors="coerce").to_numpy(float)
    # Last-known MAVLink holds reuse the same FC boot time; allow ties.
    if not np.isfinite(fc_boot).all() or np.any(np.diff(fc_boot) < 0):
        errors.append("fc_boot_time_us must be finite and non-decreasing")

    monotonic = pd.to_numeric(frame["pi_monotonic_ns"], errors="coerce").to_numpy(float)
    if len(monotonic) >= 2 and np.isfinite(monotonic).all():
        elapsed_s = (monotonic - monotonic[0]) / 1e9
        duration_s = float(elapsed_s[-1])
        warmup_s = float(campaign["protocol"]["warmup_exclusion_s"])
        post_duration_s = max(0.0, duration_s - warmup_s)
        intervals = np.diff(monotonic) / 1e9
        loop_hz = float(1.0 / np.mean(intervals))
        metrics.update(duration_s=duration_s, post_warmup_duration_s=post_duration_s, loop_hz=loop_hz)
        minimum_valid_s = float(
            campaign["protocol"]["minimum_valid_post_warmup_s"]
        )
        minimum_logged_s = warmup_s + minimum_valid_s
        metrics["minimum_required_logged_duration_s"] = minimum_logged_s
        # Estimator output begins only after the 100-step sequence buffer fills,
        # so total process runtime is not a valid lower bound for CSV duration.
        if duration_s < minimum_logged_s:
            errors.append(
                "logged duration is below warmup_exclusion_s + "
                "minimum_valid_post_warmup_s"
            )
        if post_duration_s < minimum_valid_s:
            errors.append("post-warmup duration is below minimum_valid_post_warmup_s")
        if loop_hz < float(campaign["quality"]["minimum_achieved_loop_hz"]):
            errors.append("achieved loop rate is below minimum_achieved_loop_hz")

    missed = pd.to_numeric(frame["deadline_missed"], errors="coerce")
    miss_ratio = float(missed.mean()) if len(missed) else float("nan")
    metrics["deadline_miss_ratio"] = miss_ratio
    if not np.isfinite(miss_ratio) or miss_ratio > float(campaign["quality"]["maximum_deadline_miss_ratio"]):
        errors.append("deadline miss ratio exceeds maximum_deadline_miss_ratio")

    truth_residual = pd.to_numeric(frame["truth_alignment_residual_ms"], errors="coerce")
    truth_matched = truth_residual.notna() & np.isfinite(truth_residual)
    truth_coverage = float(truth_matched.mean())
    truth_p95 = float(truth_residual[truth_matched].quantile(0.95)) if truth_matched.any() else float("inf")
    metrics.update(truth_match_coverage=truth_coverage, truth_residual_p95_ms=truth_p95)
    if truth_coverage < float(campaign["alignment"]["minimum_truth_match_ratio"]):
        errors.append("truth match coverage is below minimum_truth_match_ratio")
    if truth_p95 > float(campaign["alignment"]["max_truth_match_residual_ms"]):
        errors.append("truth alignment residual exceeds max_truth_match_residual_ms")

    try:
        alignment = read_json(report_path)
        if str(alignment.get("session_id", "")).lower() != session_id.lower():
            errors.append("alignment report session_id mismatch")
        uncertainty = float(alignment["clock_model"]["uncertainty_ms"])
        metrics["clock_uncertainty_ms"] = uncertainty
        if not np.isfinite(uncertainty) or uncertainty > float(campaign["alignment"]["max_clock_uncertainty_ms"]):
            errors.append("clock uncertainty exceeds max_clock_uncertainty_ms")
        fc_report = alignment.get("fc")
        if isinstance(fc_report, Mapping) and "coverage" in fc_report:
            metrics["fc_match_coverage"] = float(fc_report["coverage"])
            metrics["fc_residual_p95_ms"] = fc_report.get("residual_p95_ms")
            if float(fc_report["coverage"]) < float(campaign["alignment"]["minimum_fc_match_ratio"]):
                errors.append("FC match coverage is below minimum_fc_match_ratio")
            if fc_report.get("residual_p95_ms") is None or float(fc_report["residual_p95_ms"]) > float(
                campaign["alignment"]["max_fc_match_residual_ms"]
            ):
                errors.append("FC alignment residual exceeds max_fc_match_residual_ms")
    except Exception as exc:
        errors.append(f"invalid alignment report: {exc}")

    horizontal = np.hypot(
        pd.to_numeric(frame["truth_wind_n_mps"], errors="coerce"),
        pd.to_numeric(frame["truth_wind_e_mps"], errors="coerce"),
    )
    finite_wind = horizontal[np.isfinite(horizontal)]
    if len(finite_wind):
        q10, median, q90 = np.quantile(finite_wind, [0.1, 0.5, 0.9])
        low, high = map(float, campaign["conditions"][session_id.split("_")[0]]["horizontal_wind_range_mps"])
        tolerance = float(campaign.get("quality", {}).get("wind_range_tolerance_mps", 0.01))
        metrics["horizontal_truth_wind_mps"] = {"q10": float(q10), "median": float(median), "q90": float(q90)}
        metrics["wind_range_criterion"] = (
            "median must be inside configured range and central 80% interval must overlap it"
        )
        metrics["wind_range_tolerance_mps"] = tolerance
        if not (
            low - tolerance <= median <= high + tolerance
            and q90 >= low - tolerance
            and q10 <= high + tolerance
        ):
            errors.append(
                f"actual horizontal truth wind mismatches condition range [{low}, {high}] m/s"
            )
    else:
        errors.append("no finite horizontal truth wind")

    hashes: dict[str, set[str]] = {}
    for manifest in sorted(paths.root.glob("**/*manifest*.json")) + [paths.root / "inventory.json"]:
        if not manifest.is_file():
            continue
        try:
            value = read_json(manifest)
            manifest_session = value.get("session_id")
            if manifest_session is not None and str(manifest_session).lower() != session_id.lower():
                errors.append(f"{manifest}: session_id mismatch")
            _hash_values(value, hashes)
        except Exception as exc:
            errors.append(f"{manifest}: invalid manifest: {exc}")
    for column in frame.columns:
        normalized = column.lower()
        if ("hash" in normalized or "sha256" in normalized) and "model" in normalized:
            hashes.setdefault("model_hash", set()).update(frame[column].dropna().astype(str).unique())
        elif ("hash" in normalized or "sha256" in normalized) and "config" in normalized:
            hashes.setdefault("config_hash", set()).update(frame[column].dropna().astype(str).unique())
    for key, values in hashes.items():
        if len(values) > 1:
            errors.append(f"inconsistent {key} values: {sorted(values)}")
        metrics[key] = sorted(values)

    result = {
        "schema_version": 1,
        "session_id": session_id,
        "condition": session_id.split("_")[0],
        "valid": not errors,
        "errors": errors,
        "metrics": metrics,
    }
    write_json_atomic(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id")
    parser.add_argument("--campaign", default=str(DEFAULT_CAMPAIGN))
    parser.add_argument("--local", default=str(DEFAULT_LOCAL))
    args = parser.parse_args()
    result = validate_session(args.session_id, campaign_path=args.campaign, local_path=args.local)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
