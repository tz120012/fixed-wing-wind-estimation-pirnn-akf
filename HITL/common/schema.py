"""Canonical CSV schemas and numeric validation for revision HITL logs."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


PI_REQUIRED_COLUMNS = (
    "session_id",
    "condition",
    "pi_wall_time_ns",
    "pi_monotonic_ns",
    "fc_boot_time_us",
    "estimated_wind_n_mps",
    "estimated_wind_e_mps",
    "estimated_wind_d_mps",
    "nn_wind_n_mps",
    "nn_wind_e_mps",
    "nn_wind_d_mps",
    "inference_latency_ms",
    "companion_processing_latency_ms",
    "deadline_missed",
)

TRUTH_REQUIRED_COLUMNS = (
    "wall_time_usec",
    "total_wind_north_ms",
    "total_wind_east_ms",
    "total_wind_down_ms",
)

ALIGNED_REQUIRED_COLUMNS = (
    *PI_REQUIRED_COLUMNS,
    "truth_time_us",
    "truth_wind_n_mps",
    "truth_wind_e_mps",
    "truth_wind_d_mps",
    "truth_alignment_residual_ms",
)

PI_NUMERIC_COLUMNS = tuple(
    name
    for name in PI_REQUIRED_COLUMNS
    if name not in {"session_id", "condition"}
)


def require_columns(
    frame: pd.DataFrame, required: Iterable[str], *, source: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")


def require_finite(
    frame: pd.DataFrame, columns: Iterable[str], *, source: str
) -> None:
    names = list(columns)
    values = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    invalid = ~np.isfinite(values)
    if invalid.any():
        row, column = np.argwhere(invalid)[0]
        raise ValueError(
            f"{source}: non-finite value at row {int(row)}, column {names[int(column)]}"
        )


def require_constant_text(
    frame: pd.DataFrame, column: str, expected: str, *, source: str
) -> None:
    values = frame[column].astype(str).str.strip().str.lower().unique()
    if len(values) != 1 or values[0] != expected.lower():
        raise ValueError(
            f"{source}: {column} must be constant {expected!r}, got {values.tolist()}"
        )


def require_strictly_increasing(
    frame: pd.DataFrame, column: str, *, source: str
) -> None:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if len(values) < 2 or np.any(np.diff(values) <= 0):
        raise ValueError(f"{source}: {column} must be strictly increasing")
