"""Experiment 6: does PI-GRU dynamic Q/R help the AKF *state* itself?

The deployed system output is a blend ``fused = w*akf + (1-w)*nn`` which
dilutes the effect of the Kalman covariances. To isolate the contribution of
the network-predicted dynamic Q/R, this experiment evaluates BOTH:

  * ``fused``     : the deployed system output, and
  * ``akf_state`` : the pure Kalman state (diagnostics['wind_akf']),

for four backend configurations (fixed Q/R, dynamic Q only, dynamic R only,
dynamic Q/R) across clean windows, sensor-anomaly windows, and windows
stratified by true-wind dynamics (high vs low). Results are aggregated and
plotted; nothing is hard-coded toward a desired conclusion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from akf_experiment_utils import (
    PROJECT_ROOT,
    ExperimentContext,
    closure_rmse,
    horizontal_rmse,
    jitter_mean,
    load_context,
    max_step_jump,
    parse_starts,
    parse_strings,
    savefig_all,
    set_plot_style,
)
from src.experiments import paper_evidence_chain_eval as evidence

QR_CONFIGS = {
    "fixed Q/R": (False, False),
    "dyn Q": (True, False),
    "dyn R": (False, True),
    "dyn Q/R": (True, True),
}


def apply_qr_flags(pigru_out: dict, use_q: bool, use_r: bool) -> dict:
    out = {k: np.array(v, copy=True) for k, v in pigru_out.items()}
    if not use_q:
        out["q_scale"] = np.ones_like(out["q_scale"])
    if not use_r:
        out["r_scale"] = np.ones_like(out["r_scale"])
    return out


def metric_row(wind: np.ndarray, wind_true: np.ndarray, last_phys: np.ndarray, sl: slice) -> dict:
    return {
        "h_rmse": horizontal_rmse(wind_true[sl], wind[sl]),
        "jitter": jitter_mean(wind[sl]),
        "max_jump": max_step_jump(wind[sl]),
        "closure_rmse": closure_rmse(wind[sl], last_phys[sl]),
    }


def run_window(
    ctx: ExperimentContext,
    start_idx: int,
    window_size: int,
    anomaly_type: str,
    strength: float,
    batch_size: int,
    seed: int,
) -> list[dict]:
    end_idx = start_idx + window_size
    X_window = ctx.X[start_idx:end_idx].copy()
    wind_true = ctx.wind_true[start_idx:end_idx]

    if anomaly_type == "clean":
        X_used = X_window
        # skip a short warmup so the AKF can initialise
        sl = slice(int(0.1 * window_size), window_size)
    else:
        X_used, a0, a1 = evidence.inject_anomaly(X_window, ctx.scaler_X, anomaly_type, strength, seed=seed)
        sl = slice(a0, a1)

    last_phys = evidence.denorm_last_step(X_used, ctx.scaler_X)
    pigru_out = evidence.predict_pigru(ctx.model, X_used, batch_size, ctx.device)
    pigru_wind = evidence.denorm_wind(pigru_out["wind"], ctx.scaler_y)

    # dynamics regime of this window (true-wind horizontal speed variability)
    true_speed = np.linalg.norm(wind_true[:, :2], axis=1)
    dyn_std = float(np.std(true_speed))

    rows = []
    base = {
        "start_idx": start_idx,
        "anomaly_type": anomaly_type,
        "strength": strength if anomaly_type != "clean" else 0.0,
        "dyn_std": dyn_std,
    }
    # PI-GRU raw is the front-end reference (no backend)
    r = metric_row(pigru_wind, wind_true, last_phys, sl)
    r.update(base, method="PI-GRU (Raw)", output="front-end", qr_config="-")
    rows.append(r)

    for cfg, (uq, ur) in QR_CONFIGS.items():
        variant = apply_qr_flags(pigru_out, uq, ur)
        fused, diag = evidence.run_fast_pirnn_akf(
            ctx.config, variant, X_used, ctx.scaler_X, ctx.scaler_y, continuous=True
        )
        akf_state = diag["wind_akf"]
        for out_name, wind in [("fused", fused), ("akf_state", akf_state)]:
            rr = metric_row(wind, wind_true, last_phys, sl)
            rr.update(base, method=f"AKF {cfg}", output=out_name, qr_config=cfg)
            rows.append(rr)
    return rows


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = ["h_rmse", "jitter", "max_jump", "closure_rmse"]
    agg = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
    return agg


def plot_pure_vs_fused(detail: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    cfgs = ["fixed Q/R", "dyn Q", "dyn R", "dyn Q/R"]
    colors = ["#BDBDBD", "#4DBBD5", "#00A087", "#1F77B4"]
    metrics = [
        ("h_rmse", "Horizontal RMSE (m/s)", "(a) Tracking Error"),
        ("jitter", "Jitter Mean", "(b) Jitter"),
        ("max_jump", "Max Step Jump (m/s)", "(c) Instantaneous Jump"),
        ("closure_rmse", "Closure RMSE (m/s)", "(d) Airspeed Closure"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(11.0, 7.4))
    width = 0.35
    x = np.arange(len(cfgs))
    for ax, (col, ylabel, title) in zip(axs.ravel(), metrics):
        for j, out_name in enumerate(["akf_state", "fused"]):
            vals = []
            for cfg in cfgs:
                sub = detail[(detail["qr_config"] == cfg) & (detail["output"] == out_name)]
                vals.append(sub[col].mean())
            offset = (j - 0.5) * width
            label = "Pure AKF state" if out_name == "akf_state" else "Deployed (fused)"
            ax.bar(x + offset, vals, width, label=label,
                   color=("#D55E00" if out_name == "akf_state" else "#1F77B4"),
                   alpha=0.9, edgecolor="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs, rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y")
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
    axs[0, 0].legend(frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout()
    savefig_all(fig, out_dir / "experiment6_pure_vs_fused")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-AKF dynamic Q/R isolation experiment.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_dir", default="data/figure5/akf_experiments")
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--n_windows", type=int, default=12)
    parser.add_argument("--anomaly_types", default="clean,gps_spike,tas_spike,gaussian_burst")
    parser.add_argument("--anomaly_strength", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_context(args.model)
    starts = parse_starts(None, len(ctx.X), args.window_size, args.n_windows)
    anomaly_types = parse_strings(args.anomaly_types)

    all_rows = []
    for start in starts:
        for idx, anomaly_type in enumerate(anomaly_types):
            all_rows.extend(
                run_window(ctx, start, args.window_size, anomaly_type,
                           args.anomaly_strength, args.batch_size, seed=42 + idx)
            )
    detail = pd.DataFrame(all_rows)

    # stratify windows into high/low dynamics by median dyn_std
    median_std = detail["dyn_std"].median()
    detail["regime"] = np.where(detail["dyn_std"] >= median_std, "high-dynamics", "low-dynamics")

    detail.to_csv(out_dir / "experiment6_dynamic_qr_proof_detail.csv", index=False)
    summary = aggregate(detail, ["method", "output"])
    summary.to_csv(out_dir / "experiment6_dynamic_qr_proof_summary.csv", index=False)
    regime_summary = aggregate(
        detail[detail["output"] == "akf_state"], ["regime", "qr_config"]
    )
    regime_summary.to_csv(out_dir / "experiment6_dynamic_qr_proof_regime.csv", index=False)

    plot_pure_vs_fused(detail, out_dir)

    print("=== summary (method x output) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== pure AKF state by dynamics regime ===")
    print(regime_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
