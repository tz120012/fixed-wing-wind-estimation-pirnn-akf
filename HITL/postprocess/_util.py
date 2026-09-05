"""Private helpers shared by the command-line post-processors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def find_one(directory: Path, names: Iterable[str], *, required: bool) -> Path | None:
    """Find exactly one file from candidate names/globs, rejecting ambiguity."""
    matches: set[Path] = set()
    for name in names:
        matches.update(path for path in directory.glob(name) if path.is_file())
    ordered = sorted(matches)
    if len(ordered) > 1:
        raise ValueError(f"{directory}: ambiguous files: {[p.name for p in ordered]}")
    if not ordered:
        if required:
            raise FileNotFoundError(f"{directory}: no file matching {list(names)}")
        return None
    return ordered[0]


def first_key(record: Mapping[str, Any], keys: Iterable[str], source: str) -> float:
    for key in keys:
        if key in record:
            value = float(record[key])
            if not np.isfinite(value):
                break
            return value
    raise ValueError(f"{source}: missing finite value for one of {list(keys)}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            records.append(value)
    if not records:
        raise ValueError(f"{path}: no marker records")
    return records


def numeric_csv(path: Path, required: Iterable[str], source: str) -> pd.DataFrame:
    from HITL.common.schema import require_columns, require_finite

    frame = pd.read_csv(path)
    require_columns(frame, required, source=source)
    require_finite(frame, required, source=source)
    return frame


def circular_direction_error_deg(
    estimate_n: np.ndarray,
    estimate_e: np.ndarray,
    truth_n: np.ndarray,
    truth_e: np.ndarray,
) -> np.ndarray:
    estimate = np.degrees(np.arctan2(estimate_e, estimate_n))
    truth = np.degrees(np.arctan2(truth_e, truth_n))
    return np.abs((estimate - truth + 180.0) % 360.0 - 180.0)
