"""Experiment 7: volatility-driven adaptive process noise (cheap, no retrain).

Diagnostics showed the network ``q_scale`` head is inert: its range keeps the
process noise Q (nominal 0.05) far below the measurement noise R (nominal 1.0),
so scaling it never changes the Kalman gain. This experiment tests a
lightweight, no-retraining alternative: drive the *effective* process noise
from an observable proxy of true-wind volatility -- the rate of change of the
PI-GRU wind estimate. When the front-end detects fast wind change, Q opens up
so the filter tracks; in steady wind, Q stays tight so jitter stays low.

The proxy is still produced by the PI-GRU output trajectory, so this remains an
"AKF process noise adapted from the network", just sourced from the wind-rate
signal instead of the (mis-calibrated) q_scale head.

We compare PI-GRU raw, AKF (static Q + dynamic R), and AKF (volatility Q +
dynamic R) on CLEAN windows stratified into high/low true-wind dynamics, where
tracking process change -- not anomaly rejection -- is the relevant task.
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
    savefig_all,
    set_plot_style,
)
from paper_plot_style import GRAY, GREEN, ORANGE, apply_style, format_axes, save_figure
from src.experiments import paper_evidence_chain_eval as evidence


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc = alpha * x[i] + (1.0 - alpha) * acc
        out[i] = acc
    return out


def volatility_gain(wind_nn: np.ndarray, thr: float, k: float, alpha: float) -> np.ndarray:
    """Per-step process-noise multiplier from PI-GRU wind change rate."""
    delta = np.linalg.norm(np.diff(wind_nn[:, :2], axis=0), axis=1)
    delta = np.concatenate([[0.0], delta])
    vol = ema(delta, alpha)
    vol_ratio = np.clip(vol / max(thr, 1e-6), 0.0, 1.0)
    return 1.0 + k * vol_ratio


def lag_tracking_error(wind_true: np.ndarray, wind_pred: np.ndarray, max_lag: int = 30) -> tuple[float, int]:
    """Best-aligned RMSE over small lags -> residual tracking error and lag (steps)."""
    best_rmse = horizontal_rmse(wind_true, wind_pred)
    best_lag = 0
    for lag in range(1, max_lag + 1):
        e = horizontal_rmse(wind_true[lag:], wind_pred[:-lag])
        if e < best_rmse:
            best_rmse, best_lag = e, lag
    return best_rmse, best_lag


def metric_row(wind, wind_true, last_phys, sl) -> dict:
    aligned_rmse, lag = lag_tracking_error(wind_true[sl], wind[sl])
    return {
        "h_rmse": horizontal_rmse(wind_true[sl], wind[sl]),
        "aligned_rmse": aligned_rmse,
        "lag_steps": lag,
        "jitter": jitter_mean(wind[sl]),
        "max_jump": max_step_jump(wind[sl]),
        "closure_rmse": closure_rmse(wind[sl], last_phys[sl]),
    }


def run_window(ctx: ExperimentContext, start_idx: int, window_size: int,
               thr: float, k: float, alpha: float, batch_size: int) -> tuple[list[dict], dict]:
    end = start_idx + window_size
    X_window = ctx.X[start_idx:end].copy()
    wind_true = ctx.wind_true[start_idx:end]
    sl = slice(int(0.1 * window_size), window_size)

    last_phys = evidence.denorm_last_step(X_window, ctx.scaler_X)
    pigru_out = evidence.predict_pigru(ctx.model, X_window, batch_size, ctx.device)
    pigru_wind = evidence.denorm_wind(pigru_out["wind"], ctx.scaler_y)

    dyn_std = float(np.std(np.linalg.norm(wind_true[:, :2], axis=1)))

    # baseline: static Q + dynamic R (the current effective best)
    base_out = {kk: np.array(vv, copy=True) for kk, vv in pigru_out.items()}
    base_out["q_scale"] = np.ones_like(base_out["q_scale"])
    fused_b, diag_b = evidence.run_fast_pirnn_akf(
        ctx.config, base_out, X_window, ctx.scaler_X, ctx.scaler_y, continuous=True)

    # volatility-driven Q + dynamic R
    gain = volatility_gain(pigru_wind, thr, k, alpha)
    vol_out = {kk: np.array(vv, copy=True) for kk, vv in pigru_out.items()}
    vol_out["q_scale"] = gain[:, None] * np.ones_like(vol_out["q_scale"])
    fused_v, diag_v = evidence.run_fast_pirnn_akf(
        ctx.config, vol_out, X_window, ctx.scaler_X, ctx.scaler_y, continuous=True)

    rows = []
    base = {"start_idx": start_idx, "dyn_std": dyn_std}
    for method, wind in [
        ("PI-GRU (Raw)", pigru_wind),
        ("AKF static-Q + dyn-R", diag_b["wind_akf"]),
        ("AKF volatility-Q + dyn-R", diag_v["wind_akf"]),
    ]:
        rr = metric_row(wind, wind_true, last_phys, sl)
        rr.update(base, method=method)
        rows.append(rr)

    series = {
        "time_s": np.arange(window_size) * 0.02,
        "wind_true_n": wind_true[:, 0],
        "wind_true_e": wind_true[:, 1],
        "pigru_n": pigru_wind[:, 0],
        "akf_static_n": diag_b["wind_akf"][:, 0],
        "akf_vol_n": diag_v["wind_akf"][:, 0],
        "q_gain": gain,
        "dyn_std": dyn_std,
        "start_idx": start_idx,
    }
    return rows, series


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = ["h_rmse", "aligned_rmse", "lag_steps", "jitter", "max_jump", "closure_rmse"]
    agg = df.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
    return agg


def plot_regime_bars(detail: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    methods = ["PI-GRU (Raw)", "AKF static-Q + dyn-R", "AKF volatility-Q + dyn-R"]
    labels = ["PI-GRU", "AKF static-Q", "AKF vol-Q"]
    colors = ["#7F7F7F", "#00A087", "#D55E00"]
    metrics = [
        ("h_rmse", "Horizontal RMSE (m/s)", "(a) Tracking Error"),
        ("jitter", "Jitter Mean", "(b) Jitter"),
    ]
    regimes = ["high-dynamics", "low-dynamics"]
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.4))
    width = 0.25
    x = np.arange(len(regimes))
    for ax, (col, ylabel, title) in zip(axs, metrics):
        for j, (m, lbl, c) in enumerate(zip(methods, labels, colors)):
            vals = [detail[(detail["regime"] == rg) & (detail["method"] == m)][col].mean()
                    for rg in regimes]
            ax.bar(x + (j - 1) * width, vals, width, label=lbl, color=c,
                   edgecolor="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y")
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
    axs[0].legend(frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout()
    savefig_all(fig, out_dir / "experiment7_volatility_q_regime")


def plot_tracking(series_list: list[dict], out_dir: Path) -> None:
    apply_style()
    # pick the highest-dynamics window for the illustrative trace
    s = max(series_list, key=lambda d: d["dyn_std"])
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(7.0, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    t = s["time_s"]
    axs[0].plot(t, s["wind_true_n"], color="black", linewidth=1.8, label="True Wind (N)")
    axs[0].plot(t, s["pigru_n"], color=GRAY, linewidth=1.0, alpha=0.7, label="PI-GRU")
    axs[0].plot(
        t,
        s["akf_static_n"],
        color=GREEN,
        linewidth=1.4,
        linestyle="--",
        label="PI-GRU + AKF (Static $Q$)",
    )
    axs[0].plot(
        t,
        s["akf_vol_n"],
        color=ORANGE,
        linewidth=1.1,
        marker="o",
        markersize=2.2,
        markevery=max(1, len(t) // 25),
        label="PI-GRU + AKF (Volatility $Q$)",
    )
    axs[0].set_ylabel("North Wind (m/s)")
    axs[0].legend(frameon=True, edgecolor="black", fancybox=False, ncol=2)
    axs[1].plot(t, s["q_gain"], color=ORANGE, linewidth=1.3)
    axs[1].set_ylabel("$Q$ Gain")
    difference = s["akf_vol_n"] - s["akf_static_n"]
    axs[2].plot(t, difference, color="#3C5488", linewidth=1.2)
    axs[2].axhline(0.0, color="black", linewidth=0.7)
    axs[2].set_ylabel("$\\Delta w_N$\n(m/s)")
    axs[2].set_xlabel("Time (s)")
    axs[2].text(
        0.99,
        0.88,
        f"max $|\\Delta|$={np.max(np.abs(difference)):.3f} m/s",
        transform=axs[2].transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )
    for ax in axs:
        format_axes(ax)
    fig.tight_layout()
    save_figure(fig, out_dir / "experiment7_volatility_q_tracking")
    save_figure(fig, out_dir / "figureA1_dynamic_q")


def main() -> None:
    parser = argparse.ArgumentParser(description="Volatility-driven adaptive Q experiment.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_dir", default="data/figure5/akf_experiments")
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--n_windows", type=int, default=20)
    parser.add_argument("--vol_thr", type=float, default=0.05,
                        help="NN wind-rate (m/s/step) that maps to full Q opening")
    parser.add_argument("--vol_k", type=float, default=18.0,
                        help="max extra Q multiplier at full volatility")
    parser.add_argument("--vol_alpha", type=float, default=0.1,
                        help="EMA smoothing for the volatility proxy")
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_context(args.model)
    starts = parse_starts(None, len(ctx.X), args.window_size, args.n_windows)

    all_rows, series_list = [], []
    for start in starts:
        rows, series = run_window(ctx, start, args.window_size,
                                  args.vol_thr, args.vol_k, args.vol_alpha, args.batch_size)
        all_rows.extend(rows)
        series_list.append(series)

    detail = pd.DataFrame(all_rows)
    median_std = detail["dyn_std"].median()
    detail["regime"] = np.where(detail["dyn_std"] >= median_std, "high-dynamics", "low-dynamics")
    detail.to_csv(out_dir / "experiment7_volatility_q_detail.csv", index=False)

    overall = aggregate(detail, ["method"])
    regime = aggregate(detail, ["regime", "method"])
    overall.to_csv(out_dir / "experiment7_volatility_q_summary.csv", index=False)
    regime.to_csv(out_dir / "experiment7_volatility_q_regime.csv", index=False)

    plot_regime_bars(detail, out_dir)
    plot_tracking(series_list, out_dir)

    print("=== overall (clean windows) ===")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== by dynamics regime ===")
    print(regime.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
