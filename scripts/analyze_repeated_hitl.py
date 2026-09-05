#!/usr/bin/env python3
"""Aggregate repeated ID/OOD HITL sessions using session-level replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REQUIRED = {
    "session_id",
    "condition",
    "fc_boot_time_us",
    "companion_monotonic_ns",
    "truth_time_us",
    "truth_wind_n_mps",
    "truth_wind_e_mps",
    "truth_wind_d_mps",
    "estimated_wind_n_mps",
    "estimated_wind_e_mps",
    "estimated_wind_d_mps",
    "inference_latency_ms",
    "companion_processing_latency_ms",
    "deadline_missed",
}


def direction_mae(truth: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.linalg.norm(truth[:, :2], axis=1) >= 0.5
    true_angle = np.degrees(np.arctan2(truth[:, 1], truth[:, 0]))
    pred_angle = np.degrees(np.arctan2(prediction[:, 1], prediction[:, 0]))
    error = np.abs((pred_angle - true_angle + 180.0) % 360.0 - 180.0)
    return float(np.mean(error[valid])) if valid.any() else float("nan")


def parse_deadline_missed(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(float) != 0.0
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"0", "1", "false", "true", "no", "yes"}
    unknown = set(normalized.unique()) - allowed
    if unknown:
        raise ValueError(f"Unrecognized deadline_missed values: {sorted(unknown)}")
    return normalized.isin({"1", "true", "yes"}).to_numpy(bool)


def summarize_session(
    frame: pd.DataFrame,
    warmup_s: float,
    minimum_valid_s: float,
    alignment_tolerance_ms: float,
) -> dict:
    frame = frame.sort_values("companion_monotonic_ns").copy()
    time_all = frame["companion_monotonic_ns"].to_numpy(np.int64) * 1e-9
    keep = time_all >= time_all[0] + warmup_s
    frame = frame.loc[keep].copy()
    if len(frame) < 3:
        raise ValueError("Fewer than three valid frames remain after warm-up exclusion")
    truth = frame[
        ["truth_wind_n_mps", "truth_wind_e_mps", "truth_wind_d_mps"]
    ].to_numpy(float)
    prediction = frame[
        ["estimated_wind_n_mps", "estimated_wind_e_mps", "estimated_wind_d_mps"]
    ].to_numpy(float)
    error = prediction - truth
    time_s = frame["companion_monotonic_ns"].to_numpy(float) * 1e-9
    duration = float(time_s[-1] - time_s[0])
    if duration < minimum_valid_s:
        raise ValueError(
            f"Post-warm-up duration {duration:.2f}s is below {minimum_valid_s:.2f}s"
        )
    intervals = np.diff(time_s)
    if np.any(intervals <= 0):
        raise ValueError("companion_monotonic_ns must be strictly increasing")
    fc_time = frame["fc_boot_time_us"].to_numpy(np.int64)
    truth_time = frame["truth_time_us"].to_numpy(np.int64)
    if np.any(np.diff(fc_time) < 0) or np.any(np.diff(truth_time) < 0):
        raise ValueError("FC and truth timestamps must be monotonic")
    alignment_us = truth_time - fc_time
    alignment_residual_ms = np.abs(
        alignment_us - np.median(alignment_us)
    ) / 1000.0
    if float(np.max(alignment_residual_ms)) > alignment_tolerance_ms:
        raise ValueError(
            "Truth/FC timestamp alignment residual exceeds "
            f"{alignment_tolerance_ms:.1f}ms"
        )
    missed = parse_deadline_missed(frame["deadline_missed"])
    jumps = np.linalg.norm(np.diff(prediction, axis=0), axis=1)
    jitter = np.linalg.norm(np.diff(prediction, n=2, axis=0), axis=1)
    return {
        "session_id": str(frame["session_id"].iloc[0]),
        "condition": str(frame["condition"].iloc[0]).lower(),
        "n_valid_frames": len(frame),
        "duration_s": duration,
        "rmse_3d": float(np.sqrt(np.mean(error**2))),
        "rmse_n": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rmse_e": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "rmse_d": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "direction_mae_deg": direction_mae(truth, prediction),
        "achieved_loop_hz": float(1.0 / np.median(intervals)),
        "missed_deadline_ratio": float(missed.mean()),
        "inference_latency_mean_ms": float(frame["inference_latency_ms"].mean()),
        "inference_latency_p95_ms": float(frame["inference_latency_ms"].quantile(0.95)),
        "companion_processing_mean_ms": float(
            frame["companion_processing_latency_ms"].mean()
        ),
        "companion_processing_p95_ms": float(
            frame["companion_processing_latency_ms"].quantile(0.95)
        ),
        "jitter_second_difference": float(np.mean(jitter)),
        "max_instantaneous_jump": float(np.max(jumps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-dir", default="HITL/revision_sessions")
    parser.add_argument("--protocol", default="config/revision_hitl.yaml")
    parser.add_argument("--output-dir", default="HITL/revision_results")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sessions_dir = (root / args.sessions_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (root / args.protocol).open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    warmup_s = float(protocol["sessions"]["warmup_exclusion_s"])
    minimum_valid_s = float(protocol["sessions"]["minimum_valid_post_warmup_s"])
    alignment_tolerance_ms = float(
        protocol["sessions"]["truth_alignment_tolerance_ms"]
    )

    files = sorted(sessions_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No aligned session CSV files in {sessions_dir}")
    rows = []
    for path in files:
        frame = pd.read_csv(path)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(REQUIRED))
        if frame["session_id"].nunique() != 1 or frame["condition"].nunique() != 1:
            raise ValueError(f"{path.name}: expected exactly one session and condition")
        try:
            rows.append(summarize_session(
                frame,
                warmup_s,
                minimum_valid_s,
                alignment_tolerance_ms,
            ))
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc

    session_df = pd.DataFrame(rows).sort_values(["condition", "session_id"])
    if session_df["session_id"].duplicated().any():
        duplicates = session_df.loc[
            session_df["session_id"].duplicated(keep=False), "session_id"
        ].tolist()
        raise ValueError(f"Session IDs must be globally unique: {duplicates}")
    expected = {
        item["name"]: int(item["independent_sessions"])
        for item in protocol["sessions"]["conditions"]
    }
    observed = session_df.groupby("condition")["session_id"].nunique().to_dict()
    missing_sessions = {
        condition: count - int(observed.get(condition, 0))
        for condition, count in expected.items()
        if int(observed.get(condition, 0)) < count
    }
    if missing_sessions:
        raise RuntimeError(f"Insufficient independent sessions: {missing_sessions}")

    metrics = [
        column
        for column in session_df.columns
        if column not in {"session_id", "condition"}
    ]
    aggregate = session_df.groupby("condition")[metrics].agg(["mean", "std"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate = aggregate.reset_index()
    session_df.to_csv(output_dir / "hitl_session_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "hitl_condition_summary.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "replicate_unit": "independent_session",
                "observed_sessions": observed,
                "protocol": str((root / args.protocol).resolve()),
                "transport_latency_reported": False,
                "reason": "Transport latency requires synchronized FC/companion clocks.",
                "warmup_exclusion_s": warmup_s,
                "minimum_valid_post_warmup_s": minimum_valid_s,
                "truth_alignment_tolerance_ms": alignment_tolerance_ms,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
