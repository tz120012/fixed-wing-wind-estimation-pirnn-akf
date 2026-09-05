"""Experiment 9 - Adaptivity dominance: track real transient AND reject sensor anomaly.

Motivation (reviewer Route A): a fixed low-pass (EMA) cannot simultaneously (i) track a
genuine fast wind change without lag and (ii) reject a sensor-induced spike. A small alpha
rejects the spike but lags the real transient; a large alpha tracks the transient but passes
the spike. An adaptive backend (AKF) that scales measurement trust R from the network can do
both. We construct windows that contain BOTH a clean real-wind transient (first half) and an
injected GPS spike (second half, non-overlapping), and measure each requirement separately.

Outputs: per-window detail CSV, aggregated summary CSV, and a Pareto scatter
(transient tracking RMSE vs anomaly excursion).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from akf_experiment_utils import load_context, savefig_all, set_plot_style
from paper_plot_style import BLUE, GRAY, GREEN, ORANGE, RED, apply_style, format_axes, save_figure
from src.experiments import paper_evidence_chain_eval as evidence

W = 1000
TRANS_HALF = 50          # transient slice half-width (steps)
TRANS_SEARCH = (120, 430)  # search the real transient in the (clean) first half
ANOM_START, ANOM_END = 720, 770  # injected spike location (clean of real transient)
ANOM_PAD = 15
EMA_ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
GPS_SPIKE_STRENGTH = 3.0
N_WINDOWS = 12


def ema_filter(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def ema_vec(w: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(w)
    for ax in range(w.shape[1]):
        out[:, ax] = ema_filter(w[:, ax], alpha)
    return out


def inject_gps_spike(X_window, scaler_X, a0, a1, strength):
    n, seq_len, n_feat = X_window.shape
    X_phys = scaler_X.inverse_transform(X_window.reshape(-1, n_feat)).reshape(n, seq_len, n_feat)
    X_phys[a0:a1, -1, 0:3] += strength
    X_corrupt = scaler_X.transform(X_phys.reshape(-1, n_feat)).reshape(n, seq_len, n_feat)
    return X_corrupt.astype(np.float32)


def find_transient_windows(wind_true: np.ndarray, n_windows: int) -> list[tuple[int, int]]:
    """Pick window starts whose first half contains a strong real horizontal-wind transient."""
    max_start = len(wind_true) - W
    cands = np.linspace(0, max_start, 60, dtype=int)
    scored = []
    for s in cands:
        w = wind_true[s:s + W, :2]
        lo, hi = TRANS_SEARCH
        # local change magnitude over +/-TRANS_HALF
        peak_t, peak_mag = lo, 0.0
        for t in range(lo, hi):
            mag = np.linalg.norm(w[t + TRANS_HALF] - w[t - TRANS_HALF])
            if mag > peak_mag:
                peak_mag, peak_t = mag, t
        scored.append((peak_mag, int(s), int(peak_t)))
    scored.sort(reverse=True)
    return [(s, pt) for _, s, pt in scored[:n_windows]]


def metrics_for(method, wind_clean, wind_inj, wind_true, trans_sl, anom_sl):
    # transient tracking: error vs true on the clean real-change segment (no injection involved)
    te = wind_clean[trans_sl] - wind_true[trans_sl]
    trans_rmse = float(np.sqrt(np.mean(te[:, :2] ** 2)))
    # anomaly leakage: how much the injected spike perturbs the output vs the clean run
    # (isolates pure spike rejection, removing baseline OOD error)
    leak = np.linalg.norm(wind_inj[anom_sl, :2] - wind_clean[anom_sl, :2], axis=1)
    anom_leakage = float(np.max(leak))
    anom_jitter = float(np.mean(np.linalg.norm(np.diff(wind_inj[anom_sl, :2], n=2, axis=0), axis=1)))
    return {"method": method, "transient_rmse": trans_rmse,
            "anomaly_leakage": anom_leakage, "anomaly_jitter": anom_jitter}


def main():
    apply_style()
    out_dir = PROJECT_ROOT / "data/figure5/akf_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_context("train_data1/train_lambda0.0_0.1_0.3_0.5_0.8_1.0_20260518_175113/train_lambda0.1_20260518_185047")

    windows = find_transient_windows(ctx.wind_true, N_WINDOWS)
    print(f"selected {len(windows)} transient windows")

    rows = []
    for s, peak_t in windows:
        X_window = ctx.X[s:s + W].copy()
        wind_true = ctx.wind_true[s:s + W]
        X_anom = inject_gps_spike(X_window, ctx.scaler_X, ANOM_START, ANOM_END, GPS_SPIKE_STRENGTH)

        # clean (no injection) and injected runs
        clean_out = evidence.predict_pigru(ctx.model, X_window.astype(np.float32), 1000, ctx.device)
        clean_pi = evidence.denorm_wind(clean_out["wind"], ctx.scaler_y)
        clean_akf, _ = evidence.run_fast_pirnn_akf(
            ctx.config, clean_out, X_window.astype(np.float32), ctx.scaler_X, ctx.scaler_y, continuous=True)
        inj_out = evidence.predict_pigru(ctx.model, X_anom, 1000, ctx.device)
        inj_pi = evidence.denorm_wind(inj_out["wind"], ctx.scaler_y)
        inj_akf, _ = evidence.run_fast_pirnn_akf(
            ctx.config, inj_out, X_anom, ctx.scaler_X, ctx.scaler_y, continuous=True)

        trans_sl = slice(peak_t - TRANS_HALF, peak_t + TRANS_HALF)
        anom_sl = slice(ANOM_START - ANOM_PAD, ANOM_END + ANOM_PAD)

        clean_outputs = {"PI-GRU (Raw)": clean_pi, "PIRNN-AKF": clean_akf}
        inj_outputs = {"PI-GRU (Raw)": inj_pi, "PIRNN-AKF": inj_akf}
        for a in EMA_ALPHAS:
            clean_outputs[f"EMA a={a:g}"] = ema_vec(clean_pi, a)
            inj_outputs[f"EMA a={a:g}"] = ema_vec(inj_pi, a)
        for method in clean_outputs:
            r = metrics_for(method, clean_outputs[method], inj_outputs[method], wind_true, trans_sl, anom_sl)
            r.update(start_idx=s, peak_t=peak_t)
            rows.append(r)

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "experiment9_transient_vs_anomaly_detail.csv", index=False)

    summ = detail.groupby("method").agg(
        transient_rmse_mean=("transient_rmse", "mean"),
        transient_rmse_std=("transient_rmse", "std"),
        anomaly_leakage_mean=("anomaly_leakage", "mean"),
        anomaly_leakage_std=("anomaly_leakage", "std"),
        anomaly_jitter_mean=("anomaly_jitter", "mean"),
    ).reset_index()

    # worst-of-two normalized cost: normalize each axis by PI-GRU raw, take max of the two
    base = summ.set_index("method").loc["PI-GRU (Raw)"]
    summ["trans_norm"] = summ["transient_rmse_mean"] / base["transient_rmse_mean"]
    summ["leak_norm"] = summ["anomaly_leakage_mean"] / base["anomaly_leakage_mean"]
    summ["worst_of_two"] = summ[["trans_norm", "leak_norm"]].max(axis=1)
    summ = summ.sort_values("worst_of_two")
    summ.to_csv(out_dir / "experiment9_transient_vs_anomaly_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== summary (sorted by worst-of-two normalized cost; lower=better) ===")
    print(summ[["method", "transient_rmse_mean", "anomaly_leakage_mean",
                "anomaly_jitter_mean", "worst_of_two"]].to_string(index=False))

    plot_methods = ["PI-GRU (Raw)", "EMA a=0.1", "EMA a=0.5", "EMA a=0.9", "PIRNN-AKF"]
    plot_labels = ["PI-GRU", "EMA\n$\\alpha=0.1$", "EMA\n$\\alpha=0.5$", "EMA\n$\\alpha=0.9$", "PIRNN-\nAKF"]
    plot_df = summ.set_index("method").loc[plot_methods]
    x = np.arange(len(plot_methods))
    colors = [BLUE, "#BDBDBD", "#BDBDBD", "#BDBDBD", RED]

    fig_main, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    panels = [
        (axs[0], "transient_rmse_mean", "Transient tracking RMSE (m/s)", "(a) Tracking real wind transients"),
        (axs[1], "anomaly_jitter_mean", "Anomaly-window jitter (m/s)", "(b) Smoothing sensor-anomaly response"),
    ]
    for ax, col, ylabel, title in panels:
        values = plot_df[col].to_numpy(dtype=float)
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.8, width=0.64, zorder=2)
        for idx, (bar, val) in enumerate(zip(bars, values)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + values.max() * 0.025,
                f"{val:.3f}" if col == "transient_rmse_mean" else f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=RED if plot_methods[idx] == "PIRNN-AKF" else "black",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(plot_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, values.max() * 1.24)
        format_axes(ax, grid_axis="y")

    fig_main.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.86, wspace=0.32)
    fig6_dir = PROJECT_ROOT / "data/figure6"
    fig6_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig_main, fig6_dir / "figure6_tracking_smoothness_comparison", copy_to_paper="figure6")
    print(f"\nsaved -> {out_dir} and {fig6_dir}")


if __name__ == "__main__":
    main()
