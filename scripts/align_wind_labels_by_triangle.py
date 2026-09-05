#!/usr/bin/env python3
"""Align planned wind labels by minimizing velocity-triangle residuals.

This is a data-repair experiment for the existing SITL dataset. It does not
overwrite the original CSV files. For each flight segment CSV, it estimates a
single integer row lag for the wind label columns:

    TAS ~= ||V_ground - W_label(t - lag)||

Then it writes a copied CSV with shifted wind labels and gust diagnostic fields
to a new output root, plus a per-file report.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


FPS_TO_MPS = 0.3048
MPS_TO_FPS = 1.0 / FPS_TO_MPS

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
TIME_COL = "/fdm/jsbsim/simulation/sim-time-sec"

MPS_DIAG_COLS = [
    "base_wind_north_mps",
    "base_wind_east_mps",
    "base_wind_down_mps",
    "gust_delta_north_mps",
    "gust_delta_east_mps",
    "gust_delta_down_mps",
]
NON_NUMERIC_DIAG_COLS = ["wind_regime", "gust_phase"]
NUMERIC_DIAG_COLS = ["gust_factor", *MPS_DIAG_COLS]


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def closure_residual(vg_mps: np.ndarray, tas_mps: np.ndarray, wind_mps: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vg_mps - wind_mps, axis=1) - tas_mps


def shifted_indices(target_idx: np.ndarray, lag: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    src_idx = target_idx - lag
    valid = (src_idx >= 0) & (src_idx < n)
    return target_idx[valid], src_idx[valid]


def choose_alignment_mask(df: pd.DataFrame, wind_mps: np.ndarray, min_mask_rows: int) -> tuple[np.ndarray, str]:
    n = len(df)
    if "gust_phase" in df.columns:
        phase = df["gust_phase"].astype(str).str.lower().to_numpy()
        hold = np.where(phase == "hold")[0]
        if len(hold) >= min_mask_rows:
            return hold, "gust_hold"

        dynamic = np.where(np.isin(phase, ["rise", "hold", "fall"]))[0]
        if len(dynamic) >= min_mask_rows:
            return dynamic, "gust_dynamic"

    mag = np.linalg.norm(wind_mps, axis=1)
    high = np.where(mag >= 3.0)[0]
    if len(high) >= min_mask_rows:
        return high, "high_wind"

    valid = np.arange(n)
    return valid, "all"


def estimate_best_lag(
    df: pd.DataFrame,
    max_lag_rows: int,
    coarse_step_rows: int,
    fine_radius_rows: int,
    min_overlap: int,
    min_mask_rows: int,
) -> dict[str, float | int | str]:
    tas = df[TAS_COL].to_numpy(float) * FPS_TO_MPS
    vg = df[VEL_COLS].to_numpy(float) * FPS_TO_MPS
    wind = df[WIND_COLS].to_numpy(float) * FPS_TO_MPS
    n = len(df)

    mask_idx, mask_name = choose_alignment_mask(df, wind, min_mask_rows)

    def score(lag: int) -> tuple[float, float, int]:
        dst, src = shifted_indices(mask_idx, lag, n)
        if len(dst) < min_overlap:
            return float("inf"), float("nan"), len(dst)
        residual = closure_residual(vg[dst], tas[dst], wind[src])
        return rmse(residual), float(np.mean(residual)), len(dst)

    rmse0, mean0, n0 = score(0)
    best_rmse, best_lag, best_mean, best_n = rmse0, 0, mean0, n0

    coarse_step = max(1, int(coarse_step_rows))
    coarse_lags = list(range(-max_lag_rows, max_lag_rows + 1, coarse_step))
    if 0 not in coarse_lags:
        coarse_lags.append(0)
    for lag in coarse_lags:
        cur_rmse, cur_mean, cur_n = score(lag)
        if cur_rmse < best_rmse:
            best_rmse, best_lag, best_mean, best_n = cur_rmse, lag, cur_mean, cur_n

    fine_radius = max(coarse_step, int(fine_radius_rows))
    fine_start = max(-max_lag_rows, best_lag - fine_radius)
    fine_end = min(max_lag_rows, best_lag + fine_radius)
    for lag in range(fine_start, fine_end + 1):
        cur_rmse, cur_mean, cur_n = score(lag)
        if cur_rmse < best_rmse:
            best_rmse, best_lag, best_mean, best_n = cur_rmse, lag, cur_mean, cur_n

    # Estimate sample rate from the actual CSV timestamps.
    if TIME_COL in df.columns and len(df) >= 2:
        t = df[TIME_COL].to_numpy(float)
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        sample_rate_hz = float(1.0 / np.median(dt)) if len(dt) else float("nan")
    else:
        sample_rate_hz = float("nan")
    lag_seconds = float(best_lag / sample_rate_hz) if np.isfinite(sample_rate_hz) and sample_rate_hz > 0 else float("nan")

    return {
        "mask": mask_name,
        "mask_rows": int(len(mask_idx)),
        "sample_rate_hz": sample_rate_hz,
        "best_lag_rows": int(best_lag),
        "best_lag_seconds": lag_seconds,
        "baseline_rmse_mps": float(rmse0),
        "aligned_rmse_mps": float(best_rmse),
        "baseline_mean_mps": float(mean0),
        "aligned_mean_mps": float(best_mean),
        "overlap_rows": int(best_n),
    }


def shift_array(values: np.ndarray, lag: int) -> np.ndarray:
    n = len(values)
    src = np.clip(np.arange(n) - lag, 0, n - 1)
    return values[src]


def apply_lag_to_dataframe(df: pd.DataFrame, lag: int) -> pd.DataFrame:
    out = df.copy()
    for col in WIND_COLS:
        if col in out.columns:
            out[col] = shift_array(df[col].to_numpy(), lag)

    # Keep diagnostic columns consistent with shifted planned-wind labels.
    for col in NUMERIC_DIAG_COLS:
        if col in out.columns:
            out[col] = shift_array(df[col].to_numpy(), lag)
    for col in NON_NUMERIC_DIAG_COLS:
        if col in out.columns:
            out[col] = shift_array(df[col].astype(str).to_numpy(), lag)

    if all(c in out.columns for c in WIND_COLS):
        wind_mps = out[WIND_COLS].to_numpy(float) * FPS_TO_MPS
        if all(c in out.columns for c in MPS_DIAG_COLS[:3]):
            base = out[MPS_DIAG_COLS[:3]].to_numpy(float)
            delta = wind_mps - base
            for i, col in enumerate(MPS_DIAG_COLS[3:]):
                if col in out.columns:
                    out[col] = delta[:, i]
    return out


def evaluate_full_file(df: pd.DataFrame) -> tuple[float, float]:
    tas = df[TAS_COL].to_numpy(float) * FPS_TO_MPS
    vg = df[VEL_COLS].to_numpy(float) * FPS_TO_MPS
    wind = df[WIND_COLS].to_numpy(float) * FPS_TO_MPS
    residual = closure_residual(vg, tas, wind)
    return rmse(residual), float(np.mean(residual))


def process_csv(path: Path, out_path: Path, args: argparse.Namespace) -> dict[str, float | int | str]:
    df = pd.read_csv(path, low_memory=False)
    required = [TAS_COL, *VEL_COLS, *WIND_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out_path)
        return {
            "file": str(path),
            "status": "copied_missing_required",
            "missing": ";".join(missing),
        }

    full_rmse0, full_mean0 = evaluate_full_file(df)
    lag_info = estimate_best_lag(
        df,
        max_lag_rows=args.max_lag_rows,
        coarse_step_rows=args.coarse_step_rows,
        fine_radius_rows=args.fine_radius_rows,
        min_overlap=args.min_overlap,
        min_mask_rows=args.min_mask_rows,
    )

    aligned = apply_lag_to_dataframe(df, int(lag_info["best_lag_rows"]))
    full_rmse1, full_mean1 = evaluate_full_file(aligned)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(out_path, index=False)

    return {
        "file": str(path),
        "output_file": str(out_path),
        "status": "aligned",
        **lag_info,
        "full_baseline_rmse_mps": full_rmse0,
        "full_aligned_rmse_mps": full_rmse1,
        "full_baseline_mean_mps": full_mean0,
        "full_aligned_mean_mps": full_mean1,
        "full_rmse_delta_mps": full_rmse1 - full_rmse0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="src/dataset_generation/data/processed")
    parser.add_argument("--output-root", default="src/dataset_generation/data/processed_lag_aligned")
    parser.add_argument("--report", default="src/dataset_generation/data/processed_lag_aligned/wind_label_lag_report.csv")
    parser.add_argument("--splits", default="train,val,test_id,test_ood")
    parser.add_argument("--max-lag-rows", type=int, default=1500)
    parser.add_argument("--coarse-step-rows", type=int, default=25)
    parser.add_argument("--fine-radius-rows", type=int, default=40)
    parser.add_argument("--min-overlap", type=int, default=80)
    parser.add_argument("--min-mask-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    rows: list[dict[str, float | int | str]] = []
    for split in splits:
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

    aligned_rows = [r for r in rows if r.get("status") == "aligned"]
    if aligned_rows:
        before = np.array([float(r["full_baseline_rmse_mps"]) for r in aligned_rows])
        after = np.array([float(r["full_aligned_rmse_mps"]) for r in aligned_rows])
        best = np.array([float(r["aligned_rmse_mps"]) for r in aligned_rows])
        lags = np.array([float(r["best_lag_seconds"]) for r in aligned_rows])
        print(f"Processed {len(aligned_rows)} CSV files")
        print(f"Full-file closure RMSE mean: {before.mean():.3f} -> {after.mean():.3f} m/s")
        print(f"Alignment-mask RMSE mean after lag: {best.mean():.3f} m/s")
        print(
            "Best lag seconds median/p10/p90: "
            f"{np.nanmedian(lags):+.2f} / {np.nanpercentile(lags, 10):+.2f} / {np.nanpercentile(lags, 90):+.2f}"
        )
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
