#!/usr/bin/env python3
"""Project inconsistent wind labels onto the velocity-triangle constraint.

This creates an experimental dataset from the lag-aligned CSVs without deleting
rows. Rows with large closure residuals are repaired by preserving the original
wind direction and changing only the wind magnitude when possible:

    ||V_ground - a * unit(W_label)|| = TAS

If no exact non-negative magnitude exists along the original wind direction, the
closest magnitude along that direction is used and the row is flagged.
"""

from __future__ import annotations

import argparse
import csv
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
MPS_DIAG_COLS = [
    "base_wind_north_mps",
    "base_wind_east_mps",
    "base_wind_down_mps",
    "gust_delta_north_mps",
    "gust_delta_east_mps",
    "gust_delta_down_mps",
]


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else float("nan")


def closure_residual(vg_mps: np.ndarray, tas_mps: np.ndarray, wind_mps: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vg_mps - wind_mps, axis=1) - tas_mps


def quality_weight(abs_residual: np.ndarray, clean_threshold: float, suspicious_threshold: float, suspicious_weight: float) -> np.ndarray:
    weight = np.ones(len(abs_residual), dtype=np.float32)
    weight[(abs_residual > clean_threshold) & (abs_residual <= suspicious_threshold)] = suspicious_weight
    weight[abs_residual > suspicious_threshold] = 0.0
    return weight


def quality_flag(abs_residual: np.ndarray, clean_threshold: float, suspicious_threshold: float) -> np.ndarray:
    return np.where(
        abs_residual <= clean_threshold,
        "clean",
        np.where(abs_residual <= suspicious_threshold, "suspicious", "bad"),
    )


def project_wind_magnitude(
    vg_mps: np.ndarray,
    tas_mps: np.ndarray,
    wind_mps: np.ndarray,
    project_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    projected = wind_mps.copy()
    mode = np.full(len(wind_mps), "unchanged", dtype=object)
    idx = np.where(project_mask)[0]
    if len(idx) == 0:
        return projected, mode

    g = vg_mps[idx]
    t = tas_mps[idx]
    w = wind_mps[idx]
    mag0 = np.linalg.norm(w, axis=1)

    nonzero = mag0 > 1e-6
    if not np.all(nonzero):
        zero_idx = idx[~nonzero]
        # With no original direction, use the ground-speed direction as a stable fallback.
        g_zero = vg_mps[zero_idx]
        g_mag = np.linalg.norm(g_zero, axis=1)
        u_zero = g_zero / np.maximum(g_mag[:, None], 1e-6)
        a_zero = np.maximum(g_mag - tas_mps[zero_idx], 0.0)
        projected[zero_idx] = a_zero[:, None] * u_zero
        mode[zero_idx] = "projected_zero_direction"

    valid_idx = idx[nonzero]
    if len(valid_idx) == 0:
        return projected, mode

    g = vg_mps[valid_idx]
    t = tas_mps[valid_idx]
    w = wind_mps[valid_idx]
    mag0 = np.linalg.norm(w, axis=1)
    u = w / mag0[:, None]

    dot = np.sum(g * u, axis=1)
    c = np.sum(g * g, axis=1) - t * t
    disc = dot * dot - c

    exact = disc >= 0.0
    if np.any(exact):
        roots_sqrt = np.sqrt(np.maximum(disc[exact], 0.0))
        root1 = dot[exact] - roots_sqrt
        root2 = dot[exact] + roots_sqrt
        candidates = np.stack([root1, root2], axis=1)
        candidates = np.where(candidates >= 0.0, candidates, np.nan)
        fallback = np.maximum(dot[exact], 0.0)
        all_nan = np.all(~np.isfinite(candidates), axis=1)
        candidates[all_nan, 0] = fallback[all_nan]
        mag_exact0 = mag0[exact]
        pick = np.nanargmin(np.abs(candidates - mag_exact0[:, None]), axis=1)
        a = candidates[np.arange(len(candidates)), pick]
        exact_idx = valid_idx[exact]
        projected[exact_idx] = a[:, None] * u[exact]
        mode[exact_idx] = "projected_exact"

    if np.any(~exact):
        # Closest point on the ray a*u to the ground velocity vector.
        a = np.maximum(dot[~exact], 0.0)
        closest_idx = valid_idx[~exact]
        projected[closest_idx] = a[:, None] * u[~exact]
        mode[closest_idx] = "projected_closest"

    return projected, mode


def summarize(abs_residual: np.ndarray) -> dict[str, float | int]:
    finite = abs_residual[np.isfinite(abs_residual)]
    if len(finite) == 0:
        return {
            "n": 0,
            "rmse_abs_residual_mps": float("nan"),
            "mean_abs_residual_mps": float("nan"),
            "median_abs_residual_mps": float("nan"),
            "p90_abs_residual_mps": float("nan"),
            "p95_abs_residual_mps": float("nan"),
            "max_abs_residual_mps": float("nan"),
        }
    return {
        "n": int(len(finite)),
        "rmse_abs_residual_mps": rmse(finite),
        "mean_abs_residual_mps": float(np.mean(finite)),
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
        }

    tas = df[TAS_COL].to_numpy(float) * FPS_TO_MPS
    vg = df[VEL_COLS].to_numpy(float) * FPS_TO_MPS
    wind = df[WIND_COLS].to_numpy(float) * FPS_TO_MPS
    residual_before = closure_residual(vg, tas, wind)
    abs_before = np.abs(residual_before)

    project_mask = np.isfinite(abs_before) & (abs_before > args.project_threshold_mps)
    projected_wind, mode = project_wind_magnitude(vg, tas, wind, project_mask)
    residual_after = closure_residual(vg, tas, projected_wind)
    abs_after = np.abs(residual_after)

    out = df.copy()
    out[WIND_COLS] = projected_wind * MPS_TO_FPS
    out["triangle_abs_residual_before_mps"] = abs_before
    out["triangle_abs_residual_mps"] = abs_after
    out["wind_projection_mode"] = mode
    out["wind_projected"] = mode != "unchanged"
    out["label_quality_weight"] = quality_weight(abs_after, args.clean_threshold_mps, args.suspicious_threshold_mps, args.suspicious_weight)
    out["label_quality_flag"] = quality_flag(abs_after, args.clean_threshold_mps, args.suspicious_threshold_mps)

    if all(c in out.columns for c in MPS_DIAG_COLS[:3]):
        base = out[MPS_DIAG_COLS[:3]].to_numpy(float)
        delta = projected_wind - base
        for i, col in enumerate(MPS_DIAG_COLS[3:]):
            if col in out.columns:
                out[col] = delta[:, i]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    before = summarize(abs_before)
    after = summarize(abs_after)
    n_projected = int(np.sum(project_mask))
    n_exact = int(np.sum(mode == "projected_exact"))
    n_closest = int(np.sum(mode == "projected_closest"))
    n_zero_dir = int(np.sum(mode == "projected_zero_direction"))
    n_bad_after = int(np.sum(abs_after > args.suspicious_threshold_mps))
    return {
        "file": str(path),
        "output_file": str(out_path),
        "status": "projected",
        "n_original": len(df),
        "n_projected": n_projected,
        "projected_ratio": n_projected / max(len(df), 1),
        "n_projected_exact": n_exact,
        "n_projected_closest": n_closest,
        "n_projected_zero_direction": n_zero_dir,
        "n_bad_after": n_bad_after,
        "bad_after_ratio": n_bad_after / max(len(df), 1),
        "before_rmse_abs_residual_mps": before["rmse_abs_residual_mps"],
        "after_rmse_abs_residual_mps": after["rmse_abs_residual_mps"],
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
    parser.add_argument("--output-root", default="src/dataset_generation/data/processed_lag_aligned_projected")
    parser.add_argument("--report", default="src/dataset_generation/data/processed_lag_aligned_projected/projection_report.csv")
    parser.add_argument("--splits", default="train,val,test_id,test_ood")
    parser.add_argument("--project-threshold-mps", type=float, default=1.5)
    parser.add_argument("--clean-threshold-mps", type=float, default=0.8)
    parser.add_argument("--suspicious-threshold-mps", type=float, default=1.5)
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

    projected = [r for r in rows if r.get("status") == "projected"]
    if projected:
        n_original = sum(int(r["n_original"]) for r in projected)
        n_projected = sum(int(r["n_projected"]) for r in projected)
        n_exact = sum(int(r["n_projected_exact"]) for r in projected)
        n_closest = sum(int(r["n_projected_closest"]) for r in projected)
        n_bad_after = sum(int(r["n_bad_after"]) for r in projected)
        before = np.array([float(r["before_rmse_abs_residual_mps"]) for r in projected])
        after = np.array([float(r["after_rmse_abs_residual_mps"]) for r in projected])
        print(f"Processed {len(projected)} CSV files")
        print(f"Projected rows: {n_projected:,}/{n_original:,} ({n_projected / max(n_original, 1):.2%})")
        print(f"Projection modes: exact={n_exact:,}, closest={n_closest:,}")
        print(f"Rows still above suspicious threshold: {n_bad_after:,} ({n_bad_after / max(n_original, 1):.2%})")
        print(f"Per-file residual RMSE mean: {before.mean():.3f} -> {after.mean():.3f} m/s")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
