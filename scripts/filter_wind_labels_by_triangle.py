#!/usr/bin/env python3
"""Filter lag-aligned wind labels by velocity-triangle consistency.

The script keeps the repaired CSV dataset separate from the input:

* residual <= keep_threshold: keep with weight 1.0
* keep_threshold < residual <= drop_threshold: keep with reduced quality weight
* residual > drop_threshold: drop from the filtered CSV

Residual definition:

    abs(||V_ground - W_label|| - TAS)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


FPS_TO_MPS = 0.3048
WIND_COLS = [
    "/fdm/jsbsim/atmosphere/wind-north-fps",
    "/fdm/jsbsim/atmosphere/wind-east-fps",
    "/fdm/jsbsim/atmosphere/wind-down-fps",
]
VEL_COLS = [
    "/fdm/jsbsim/velocities/v-north-fps",
    "/fdm/jsbsim/velocities/v-east-fps",
    "/fdm/jsbsim/velocities/v-down-fps",
]
TAS_COL = "/fdm/jsbsim/velocities/vtrue-fps"


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def compute_abs_residual(df: pd.DataFrame) -> np.ndarray:
    tas = df[TAS_COL].to_numpy(float) * FPS_TO_MPS
    vg = df[VEL_COLS].to_numpy(float) * FPS_TO_MPS
    wind = df[WIND_COLS].to_numpy(float) * FPS_TO_MPS
    residual = np.linalg.norm(vg - wind, axis=1) - tas
    return np.abs(residual)


def quality_weights(abs_residual: np.ndarray, keep_threshold: float, drop_threshold: float, suspicious_weight: float) -> np.ndarray:
    weights = np.ones(len(abs_residual), dtype=np.float32)
    suspicious = (abs_residual > keep_threshold) & (abs_residual <= drop_threshold)
    weights[suspicious] = suspicious_weight
    return weights


def summarize_residual(abs_residual: np.ndarray) -> dict[str, float | int]:
    finite = abs_residual[np.isfinite(abs_residual)]
    if len(finite) == 0:
        return {
            "n": 0,
            "mean_abs_residual_mps": float("nan"),
            "rmse_residual_mps": float("nan"),
            "median_abs_residual_mps": float("nan"),
            "p90_abs_residual_mps": float("nan"),
            "p95_abs_residual_mps": float("nan"),
            "max_abs_residual_mps": float("nan"),
        }
    return {
        "n": int(len(finite)),
        "mean_abs_residual_mps": float(np.mean(finite)),
        "rmse_residual_mps": rmse(finite),
        "median_abs_residual_mps": float(np.median(finite)),
        "p90_abs_residual_mps": float(np.percentile(finite, 90)),
        "p95_abs_residual_mps": float(np.percentile(finite, 95)),
        "max_abs_residual_mps": float(np.max(finite)),
    }


def process_csv(path: Path, out_path: Path, args: argparse.Namespace) -> dict[str, float | int | str]:
    df = pd.read_csv(path, low_memory=False)
    required = [TAS_COL, *VEL_COLS, *WIND_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return {
            "file": str(path),
            "output_file": str(out_path),
            "status": "copied_missing_required",
            "missing": ";".join(missing),
            "n_original": len(df),
            "n_kept": len(df),
            "n_dropped": 0,
        }

    abs_residual = compute_abs_residual(df)
    keep_mask = np.isfinite(abs_residual) & (abs_residual <= args.drop_threshold_mps)
    quality = quality_weights(abs_residual, args.keep_threshold_mps, args.drop_threshold_mps, args.suspicious_weight)

    out = df.loc[keep_mask].copy()
    out["triangle_abs_residual_mps"] = abs_residual[keep_mask]
    out["label_quality_weight"] = quality[keep_mask]
    out["label_quality_flag"] = np.where(
        out["triangle_abs_residual_mps"].to_numpy(float) <= args.keep_threshold_mps,
        "clean",
        "suspicious",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    before = summarize_residual(abs_residual)
    after = summarize_residual(abs_residual[keep_mask])
    n_original = len(df)
    n_kept = int(np.sum(keep_mask))
    n_suspicious = int(np.sum((abs_residual > args.keep_threshold_mps) & (abs_residual <= args.drop_threshold_mps)))
    n_dropped = n_original - n_kept
    return {
        "file": str(path),
        "output_file": str(out_path),
        "status": "filtered",
        "n_original": n_original,
        "n_kept": n_kept,
        "n_suspicious_kept": n_suspicious,
        "n_dropped": n_dropped,
        "kept_ratio": n_kept / max(n_original, 1),
        "dropped_ratio": n_dropped / max(n_original, 1),
        "before_rmse_residual_mps": before["rmse_residual_mps"],
        "after_rmse_residual_mps": after["rmse_residual_mps"],
        "before_p90_abs_residual_mps": before["p90_abs_residual_mps"],
        "after_p90_abs_residual_mps": after["p90_abs_residual_mps"],
        "before_p95_abs_residual_mps": before["p95_abs_residual_mps"],
        "after_p95_abs_residual_mps": after["p95_abs_residual_mps"],
        "before_max_abs_residual_mps": before["max_abs_residual_mps"],
        "after_max_abs_residual_mps": after["max_abs_residual_mps"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="src/dataset_generation/data/processed_lag_aligned")
    parser.add_argument("--output-root", default="src/dataset_generation/data/processed_lag_aligned_filtered")
    parser.add_argument("--report", default="src/dataset_generation/data/processed_lag_aligned_filtered/filter_report.csv")
    parser.add_argument("--splits", default="train,val,test_id,test_ood")
    parser.add_argument("--keep-threshold-mps", type=float, default=0.8)
    parser.add_argument("--drop-threshold-mps", type=float, default=1.5)
    parser.add_argument("--suspicious-weight", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    rows: list[dict[str, float | int | str]] = []

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        for path in sorted((input_root / split).glob("*.csv")):
            out_path = output_root / split / path.name
            rows.append(process_csv(path, out_path, args))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    filtered = [r for r in rows if r.get("status") == "filtered"]
    if filtered:
        n_original = sum(int(r["n_original"]) for r in filtered)
        n_kept = sum(int(r["n_kept"]) for r in filtered)
        n_suspicious = sum(int(r["n_suspicious_kept"]) for r in filtered)
        n_dropped = sum(int(r["n_dropped"]) for r in filtered)
        before = np.array([float(r["before_rmse_residual_mps"]) for r in filtered])
        after = np.array([float(r["after_rmse_residual_mps"]) for r in filtered])
        print(f"Processed {len(filtered)} CSV files")
        print(f"Rows: kept {n_kept:,}/{n_original:,} ({n_kept / max(n_original, 1):.2%}), dropped {n_dropped:,} ({n_dropped / max(n_original, 1):.2%})")
        print(f"Suspicious kept with weight {args.suspicious_weight:g}: {n_suspicious:,} ({n_suspicious / max(n_original, 1):.2%})")
        print(f"Per-file residual RMSE mean: {before.mean():.3f} -> {after.mean():.3f} m/s")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
