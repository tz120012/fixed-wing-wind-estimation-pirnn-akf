"""Compute repeated-session HITL metrics and condition-level mean ± unbiased SD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HITL.common.config import DEFAULT_CAMPAIGN, DEFAULT_LOCAL, SessionPaths, load_configs
from HITL.common.manifest import write_json_atomic
from HITL.common.schema import ALIGNED_REQUIRED_COLUMNS, require_columns, require_finite
from HITL.postprocess._util import circular_direction_error_deg, read_json


def _session_metrics(frame: pd.DataFrame, session_id: str, campaign: dict[str, Any]) -> dict[str, Any]:
    require_columns(frame, ALIGNED_REQUIRED_COLUMNS, source=session_id)
    required_numeric = [name for name in ALIGNED_REQUIRED_COLUMNS if name not in ("session_id", "condition")]
    require_finite(frame, required_numeric, source=session_id)
    monotonic = frame["pi_monotonic_ns"].to_numpy(float)
    if np.any(np.diff(monotonic) <= 0):
        raise ValueError(f"{session_id}: non-increasing Pi monotonic time")
    warmup_ns = float(campaign["protocol"]["warmup_exclusion_s"]) * 1e9
    keep = monotonic - monotonic[0] >= warmup_ns
    data = frame.loc[keep].copy()
    if len(data) < 3:
        raise ValueError(f"{session_id}: fewer than three post-warmup rows")

    estimate = data[
        ["estimated_wind_n_mps", "estimated_wind_e_mps", "estimated_wind_d_mps"]
    ].to_numpy(float)
    truth = data[["truth_wind_n_mps", "truth_wind_e_mps", "truth_wind_d_mps"]].to_numpy(float)
    error = estimate - truth
    rmse_axis = np.sqrt(np.mean(error**2, axis=0))
    # Paper definition: root mean square across the three component errors,
    # equivalently sqrt(mean([RMSE_N^2, RMSE_E^2, RMSE_D^2])).
    rmse_3d = float(np.sqrt(np.mean(error**2)))
    horizontal_truth = np.hypot(truth[:, 0], truth[:, 1])
    direction_mask = horizontal_truth >= float(campaign["protocol"]["direction_min_horizontal_wind_mps"])
    if not direction_mask.any():
        raise ValueError(f"{session_id}: no rows meet direction wind threshold")
    direction_error = circular_direction_error_deg(
        estimate[direction_mask, 0],
        estimate[direction_mask, 1],
        truth[direction_mask, 0],
        truth[direction_mask, 1],
    )
    second_difference = np.diff(estimate[:, :2], n=2, axis=0)
    jumps = np.diff(estimate[:, :2], axis=0)
    intervals_s = np.diff(data["pi_monotonic_ns"].to_numpy(float)) / 1e9
    inference = data["inference_latency_ms"].to_numpy(float)
    companion = data["companion_processing_latency_ms"].to_numpy(float)
    return {
        "session_id": session_id,
        "condition": session_id.split("_")[0],
        "post_warmup_rows": int(len(data)),
        "rmse_3d_mps": rmse_3d,
        "rmse_n_mps": float(rmse_axis[0]),
        "rmse_e_mps": float(rmse_axis[1]),
        "rmse_d_mps": float(rmse_axis[2]),
        "direction_mae_deg": float(np.mean(direction_error)),
        "second_diff_jitter_mps": float(np.mean(np.linalg.norm(second_difference, axis=1))),
        "max_jump_mps": float(np.max(np.linalg.norm(jumps, axis=1))),
        "inference_latency_mean_ms": float(np.mean(inference)),
        "inference_latency_p95_ms": float(np.quantile(inference, 0.95)),
        "companion_latency_mean_ms": float(np.mean(companion)),
        "companion_latency_p95_ms": float(np.quantile(companion, 0.95)),
        "loop_hz": float(1.0 / np.median(intervals_s)),
        "deadline_misses": int(np.sum(data["deadline_missed"].to_numpy(float) != 0)),
        "deadline_miss_ratio": float(np.mean(data["deadline_missed"].to_numpy(float) != 0)),
    }


def analyze_campaign(
    *,
    campaign_path: str | Path = DEFAULT_CAMPAIGN,
    local_path: str | Path = DEFAULT_LOCAL,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    campaign, _ = load_configs(campaign_path, local_path, require_local=False)
    required = int(campaign["protocol"]["required_sessions_per_condition"])
    if required != 5:
        raise ValueError("campaign analysis requires exactly 5 sessions per condition")
    expected = [f"{condition}_{index:02d}" for condition in ("id", "ood") for index in range(1, 6)]
    session_rows: list[dict[str, Any]] = []
    for session_id in expected:
        paths = SessionPaths.build(campaign, session_id)
        validation_path = paths.summary / "validation.json"
        if not validation_path.is_file():
            raise ValueError(f"{session_id}: missing validation.json")
        validation = read_json(validation_path)
        if validation.get("session_id") != session_id or validation.get("valid") is not True:
            raise ValueError(f"{session_id}: session is not valid")
        aligned_path = paths.aligned / campaign["logging"]["aligned_csv_name"]
        session_rows.append(_session_metrics(pd.read_csv(aligned_path), session_id, campaign))

    sessions = pd.DataFrame(session_rows)
    counts = sessions.groupby("condition")["session_id"].nunique().to_dict()
    if counts != {"id": 5, "ood": 5} or len(sessions) != 10:
        raise ValueError(f"expected exactly 5 ID and 5 OOD sessions, got {counts}")

    numeric_columns = [
        column
        for column in sessions.columns
        if column not in ("session_id", "condition", "post_warmup_rows")
    ]
    summaries: list[dict[str, Any]] = []
    for condition in ("id", "ood"):
        group = sessions.loc[sessions["condition"] == condition]
        row: dict[str, Any] = {"condition": condition, "session_count": int(len(group))}
        for column in numeric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_sd"] = float(group[column].std(ddof=1))
        summaries.append(row)
    conditions = pd.DataFrame(summaries)

    root = Path(campaign["logging"]["sessions_root"])
    if not root.is_absolute():
        root = Path(campaign_path).resolve().parents[2] / root
    destination = Path(output_dir).resolve() if output_dir else root / "campaign_summary"
    destination.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(destination / "session_metrics.csv", index=False)
    conditions.to_csv(destination / "condition_summary.csv", index=False)
    payload = {
        "schema_version": 1,
        "aggregation_unit": "session",
        "session_count": 10,
        "sessions": sessions.to_dict(orient="records"),
        "conditions": conditions.to_dict(orient="records"),
    }
    write_json_atomic(destination / "campaign_summary.json", payload)
    return sessions, conditions, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=str(DEFAULT_CAMPAIGN))
    parser.add_argument("--local", default=str(DEFAULT_LOCAL))
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    _, _, result = analyze_campaign(
        campaign_path=args.campaign,
        local_path=args.local,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
