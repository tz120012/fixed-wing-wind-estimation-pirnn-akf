import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from akf_experiment_utils import (
    PROJECT_ROOT,
    ExperimentContext,
    add_relative_metrics,
    aggregate,
    evaluate_output,
    load_context,
    parse_starts,
    parse_strings,
    savefig_all,
    set_plot_style,
)
from src.experiments import paper_evidence_chain_eval as evidence


VARIANTS = {
    "PI-GRU (Raw)": None,
    "AKF fixed Q/R": (False, False),
    "AKF dynamic Q only": (True, False),
    "AKF dynamic R only": (False, True),
    "AKF dynamic Q/R": (True, True),
}


def make_variant_output(pigru_out: dict, use_q: bool, use_r: bool) -> dict:
    out = {k: np.array(v, copy=True) for k, v in pigru_out.items()}
    if not use_q:
        out["q_scale"] = np.ones_like(out["q_scale"])
    if not use_r:
        out["r_scale"] = np.ones_like(out["r_scale"])
    return out


def run_qr_case(
    ctx: ExperimentContext,
    start_idx: int,
    window_size: int,
    anomaly_type: str,
    strength: float,
    batch_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    end_idx = start_idx + window_size
    X_window = ctx.X[start_idx:end_idx].copy()
    wind_true = ctx.wind_true[start_idx:end_idx]
    X_anom, anom_start, anom_end = evidence.inject_anomaly(
        X_window, ctx.scaler_X, anomaly_type, strength, seed=seed
    )
    last_phys = evidence.denorm_last_step(X_anom, ctx.scaler_X)
    pigru_out = evidence.predict_pigru(ctx.model, X_anom, batch_size, ctx.device)
    pigru_wind = evidence.denorm_wind(pigru_out["wind"], ctx.scaler_y)

    anomaly_slice = slice(anom_start, anom_end)
    boundary_slice = slice(max(0, anom_start - 20), min(window_size, anom_end + 20))
    rows = []
    diag_rows = []

    raw_row = evaluate_output("PI-GRU (Raw)", pigru_wind, wind_true, last_phys, anomaly_slice, boundary_slice)
    raw_row.update({
        "start_idx": start_idx,
        "window_size": window_size,
        "anomaly_type": anomaly_type,
        "strength": strength,
        "anomaly_start_step": anom_start,
        "anomaly_end_step": anom_end,
    })
    rows.append(raw_row)

    for method, flags in VARIANTS.items():
        if flags is None:
            continue
        use_q, use_r = flags
        variant_out = make_variant_output(pigru_out, use_q=use_q, use_r=use_r)
        wind, diagnostics = evidence.run_fast_pirnn_akf(
            ctx.config, variant_out, X_anom, ctx.scaler_X, ctx.scaler_y, continuous=True
        )
        row = evaluate_output(method, wind, wind_true, last_phys, anomaly_slice, boundary_slice)
        row.update({
            "start_idx": start_idx,
            "window_size": window_size,
            "anomaly_type": anomaly_type,
            "strength": strength,
            "anomaly_start_step": anom_start,
            "anomaly_end_step": anom_end,
        })
        rows.append(row)

        diag = pd.DataFrame({
            "method": method,
            "step": np.arange(window_size),
            "time_s": np.arange(window_size) * 0.02,
            "start_idx": start_idx,
            "anomaly_type": anomaly_type,
            "strength": strength,
            "q_scale_mean": np.mean(variant_out["q_scale"], axis=1),
            "r_scale_mean": np.mean(variant_out["r_scale"], axis=1),
            "Q_diag_mean": np.mean(diagnostics["Q_diag"], axis=1),
            "R_diag_mean": np.mean(diagnostics["R_diag"], axis=1),
            "akf_weight": diagnostics["akf_weight"],
            "nn_weight": diagnostics["nn_weight"],
            "innovation_norm": diagnostics["innovation_norm"],
        })
        diag_rows.append(diag)

    return pd.DataFrame(rows), pd.concat(diag_rows, ignore_index=True)


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    methods = [
        "PI-GRU (Raw)",
        "AKF fixed Q/R",
        "AKF dynamic Q only",
        "AKF dynamic R only",
        "AKF dynamic Q/R",
    ]
    sub = summary.set_index("method").loc[methods].reset_index()
    labels = ["PI-GRU", "Fixed Q/R", "Dyn Q", "Dyn R", "Dyn Q/R"]
    colors = ["#7F7F7F", "#BDBDBD", "#4DBBD5", "#00A087", "#1F77B4"]
    metrics = [
        ("h_rmse_mean", "Horizontal RMSE (m/s)", "(a) Tracking Error"),
        ("jitter_mean", "Jitter Mean", "(b) Jitter"),
        ("max_jump_mean", "Max Step Jump (m/s)", "(c) Instantaneous Jump"),
        ("closure_rmse_mean", "Closure RMSE (m/s)", "(d) Airspeed Closure"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.0))
    x = np.arange(len(sub))
    for ax, (col, ylabel, title) in zip(axs.ravel(), metrics):
        ax.bar(x, sub[col], color=colors, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y")
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
    fig.tight_layout()
    savefig_all(fig, out_dir / "experiment5_dynamic_qr_ablation")


def plot_diagnostics(diag: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    # Use one representative dynamic-vs-fixed GPS case.
    gps = diag[(diag["anomaly_type"] == "gps_spike") & (diag["start_idx"] == diag["start_idx"].min())]
    if gps.empty:
        gps = diag[diag["start_idx"] == diag["start_idx"].min()]
    methods = ["AKF fixed Q/R", "AKF dynamic Q/R"]
    fig, axs = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)
    for method in methods:
        sub = gps[gps["method"] == method]
        if sub.empty:
            continue
        label = method.replace("AKF ", "")
        axs[0].plot(sub["time_s"], sub["Q_diag_mean"], linewidth=1.7, label=label)
        axs[1].plot(sub["time_s"], sub["R_diag_mean"], linewidth=1.7, label=label)
        axs[2].plot(sub["time_s"], sub["innovation_norm"], linewidth=1.7, label=label)
    axs[0].set_ylabel("Mean diag(Q)")
    axs[0].set_title("(a) Process Covariance")
    axs[1].set_ylabel("Mean diag(R)")
    axs[1].set_title("(b) Measurement Covariance")
    axs[2].set_ylabel("Innovation Norm")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_title("(c) Innovation")
    for ax in axs:
        ax.axvspan(22.5, 27.5, color="red", alpha=0.1)
        ax.grid(True)
        ax.legend(frameon=True, edgecolor="black", fancybox=False)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
    fig.tight_layout()
    savefig_all(fig, out_dir / "experiment5_dynamic_qr_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic Q/R ablation for PI-GRU-driven AKF.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_dir", default="data/figure5/akf_experiments")
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--starts", default=None)
    parser.add_argument("--n_windows", type=int, default=8)
    parser.add_argument("--anomaly_types", default="gps_spike,tas_spike,sensor_dropout,gaussian_burst")
    parser.add_argument("--anomaly_strength", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_context(args.model)
    starts = parse_starts(args.starts, len(ctx.X), args.window_size, args.n_windows)
    anomaly_types = parse_strings(args.anomaly_types)

    detail_rows = []
    diag_rows = []
    for start in starts:
        for idx, anomaly_type in enumerate(anomaly_types):
            detail, diag = run_qr_case(
                ctx,
                start,
                args.window_size,
                anomaly_type,
                args.anomaly_strength,
                args.batch_size,
                seed=42 + idx,
            )
            detail_rows.append(detail)
            diag_rows.append(diag)

    detail = add_relative_metrics(pd.concat(detail_rows, ignore_index=True))
    diagnostics = pd.concat(diag_rows, ignore_index=True)
    summary = aggregate(detail, ["method"])
    detail.to_csv(out_dir / "experiment5_dynamic_qr_ablation_detail.csv", index=False)
    summary.to_csv(out_dir / "experiment5_dynamic_qr_ablation_summary.csv", index=False)
    diagnostics.to_csv(out_dir / "experiment5_dynamic_qr_ablation_diagnostics.csv", index=False)
    plot_summary(summary, out_dir)
    plot_diagnostics(diagnostics, out_dir)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
