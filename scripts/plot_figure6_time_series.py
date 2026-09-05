import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper_plot_style import BLUE, CYAN, GRAY, GREEN, RED, apply_style, format_axes, save_figure
from src.experiments import paper_evidence_chain_eval as evidence

def compute_ema(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to PI-GRU model directory")
    parser.add_argument("--data_dir", type=str, default="data/dataset_new_processed")
    parser.add_argument("--out_dir", type=str, default="data/figure5")
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--start_idx", type=int, default=None, help="Start index of the Test-OOD window")
    parser.add_argument("--anomaly_type", type=str, default="gps_spike")
    parser.add_argument("--anomaly_strength", type=float, default=3.0)
    parser.add_argument("--output_prefix", type=str, default="figure5_time_series")
    parser.add_argument("--dump_csv", action="store_true",
                        help="Also write the plotted per-step series to <output_prefix>_source_data.csv")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Test-OOD data...")
    scaler_X, scaler_y = evidence.load_norm_params(data_dir)
    X_full = np.load(data_dir / "X_test_ood.npy")
    y_full = np.load(data_dir / "y_test_ood.npy")
    
    # Pick a continuous window. By default, use the middle of Test-OOD;
    # for paper diagnostics, pass --start_idx to use a vetted window.
    start_idx = len(X_full) // 2 if args.start_idx is None else args.start_idx
    end_idx = start_idx + args.window_size
    if start_idx < 0 or end_idx > len(X_full):
        raise ValueError(f"Invalid window [{start_idx}, {end_idx}) for dataset length {len(X_full)}")
    
    X_window = X_full[start_idx:end_idx].copy()
    y_window = y_full[start_idx:end_idx].copy()
    wind_true = evidence.denorm_y(y_window, scaler_y)[:, :3]

    # Inject anomaly in the middle of the window
    print(f"Injecting {args.anomaly_type} anomaly...")
    X_anom, anom_start, anom_end = evidence.inject_anomaly(
        X_window, scaler_X, args.anomaly_type, args.anomaly_strength, seed=42
    )
    n_feat = X_window.shape[-1]
    gps_clean = scaler_X.inverse_transform(X_window.reshape(-1, n_feat)).reshape(X_window.shape)[:, -1, 0:2]
    gps_injected = scaler_X.inverse_transform(X_anom.reshape(-1, n_feat)).reshape(X_anom.shape)[:, -1, 0:2]
    gps_delta = gps_injected - gps_clean

    print("Loading PI-GRU model...")
    ckpt_path = PROJECT_ROOT / args.model / "best_composite_model.pth"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / args.model / "best_model.pth"
    model = evidence.load_pigru_model(ckpt_path, scaler_X, scaler_y, device)
    
    print("Running PI-GRU...")
    pigru_out = evidence.predict_pigru(model, X_anom, batch_size=args.window_size, device=device)
    pigru_wind = evidence.denorm_wind(pigru_out["wind"], scaler_y)

    print("Running EMA...")
    ema_wind = np.zeros_like(pigru_wind)
    for i in range(3):
        ema_wind[:, i] = compute_ema(pigru_wind[:, i], alpha=0.1)

    print("Running PIRNN-AKF...")
    with open(PROJECT_ROOT / "config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    pirnn_akf_wind, _ = evidence.run_fast_pirnn_akf(
        config, pigru_out, X_anom, scaler_X, scaler_y, continuous=True
    )

    if args.dump_csv:
        import pandas as pd
        t = np.arange(args.window_size) * 0.02
        source = pd.DataFrame({
            "time_s": t,
            "in_anomaly_window": (np.arange(args.window_size) >= anom_start)
                                 & (np.arange(args.window_size) <= anom_end),
            "gps_velocity_error_n_mps": gps_delta[:, 0],
            "gps_velocity_error_e_mps": gps_delta[:, 1],
        })
        for axis, idx in (("n", 0), ("e", 1), ("d", 2)):
            source[f"wind_true_{axis}_mps"] = wind_true[:, idx]
            source[f"pigru_{axis}_mps"] = pigru_wind[:, idx]
            source[f"ema_alpha0.1_{axis}_mps"] = ema_wind[:, idx]
            source[f"pirnn_akf_{axis}_mps"] = pirnn_akf_wind[:, idx]
        csv_path = out_dir / f"{args.output_prefix}_source_data.csv"
        source.to_csv(csv_path, index=False)
        print(f"Wrote per-step source data -> {csv_path} "
              f"({len(source)} rows, anomaly window steps {anom_start}-{anom_end})")

    print("Plotting Figure 5...")
    apply_style()

    # Plot injected GPS perturbation and all three axis-wise estimation errors.
    fig, axs = plt.subplots(4, 1, figsize=(7.0, 7.2), sharex=True)
    
    time_axis = np.arange(args.window_size) * 0.02  # assuming 50Hz
    
    color_true = 'black'
    color_raw = GRAY
    color_ema = GREEN
    color_akf = BLUE
    color_gps_n = RED
    color_gps_e = CYAN
    anomaly_span_color = GRAY
    anomaly_span_alpha = 0.18

    # Injected GPS velocity perturbation
    axs[0].plot(time_axis, gps_delta[:, 0], color=color_gps_n, linewidth=1.8, label=r'$\Delta V_{g,N}$')
    axs[0].plot(time_axis, gps_delta[:, 1], color=color_gps_e, linewidth=1.8, linestyle='--', label=r'$\Delta V_{g,E}$')
    axs[0].axvspan(time_axis[anom_start], time_axis[anom_end], color=anomaly_span_color, alpha=anomaly_span_alpha, label='Injected GPS Spike')
    axs[0].set_ylabel('GPS Velocity Error (m/s)')
    axs[0].grid(True)
    axs[0].text(0.01, 0.92, "(a)", transform=axs[0].transAxes, fontsize=9, fontweight="bold", va="top", ha="left")
    axs[0].legend(
        loc='upper right',
        ncol=3,
        frameon=True,
        edgecolor='black',
        fancybox=False,
        fontsize=7,
        handlelength=1.1,
        columnspacing=0.8,
    )
    
    axis_names = ("North", "East", "Down")
    methods = (
        ("PI-GRU", pigru_wind, color_raw, (0, (2, 1)), 1.0),
        ("EMA ($\\alpha=0.1$)", ema_wind, color_ema, "--", 1.4),
        ("PIRNN-AKF", pirnn_akf_wind, color_akf, "-", 1.8),
    )
    anomaly_slice = slice(anom_start, anom_end + 1)
    for axis_index, axis_name in enumerate(axis_names):
        ax = axs[axis_index + 1]
        for label, estimate, color, linestyle, linewidth in methods:
            error = estimate[:, axis_index] - wind_true[:, axis_index]
            ax.plot(
                time_axis,
                error,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
            )
        ax.axhline(0.0, color=color_true, linewidth=0.8, alpha=0.7)
        ax.axvspan(
            time_axis[anom_start],
            time_axis[anom_end],
            color=anomaly_span_color,
            alpha=anomaly_span_alpha,
        )
        raw_error = pigru_wind[anomaly_slice, axis_index] - wind_true[anomaly_slice, axis_index]
        akf_error = pirnn_akf_wind[anomaly_slice, axis_index] - wind_true[anomaly_slice, axis_index]
        raw_rmse = np.sqrt(np.mean(raw_error**2))
        akf_rmse = np.sqrt(np.mean(akf_error**2))
        raw_jitter = np.std(np.diff(raw_error))
        akf_jitter = np.std(np.diff(akf_error))

        # Reserve a clear band above the data for the quantitative annotation.
        # Without this headroom the Down-error trace sits directly behind the
        # upper-right text box.
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + 0.35 * (y_max - y_min))
        ax.text(
            0.99,
            0.96,
            f"anomaly RMSE: {raw_rmse:.2f}$\\rightarrow${akf_rmse:.2f} m/s\n"
            f"jitter: {raw_jitter:.2f}$\\rightarrow${akf_jitter:.2f} m/s",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.7,
            bbox={"facecolor": "white", "edgecolor": "0.5", "alpha": 0.85, "pad": 2},
        )
        ax.set_ylabel(f"{axis_name}\nError (m/s)")
        ax.grid(True)
        ax.text(
            0.01,
            0.92,
            f"({chr(ord('b') + axis_index)})",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
        )
    axs[1].legend(
        loc="lower right",
        ncol=3,
        frameon=True,
        edgecolor="black",
        fancybox=False,
        fontsize=7,
        handlelength=1.1,
        columnspacing=0.8,
    )
    axs[-1].set_xlabel("Time (s)")

    # Remove top and right spines
    for ax in axs:
        format_axes(ax)

    plt.tight_layout()
    
    out_base = out_dir / args.output_prefix
    save_figure(fig, out_base, copy_to_paper="figure5")

    # Keep the LaTeX submission figure synchronized with the Markdown figure.
    mdpi_base = PROJECT_ROOT / "Paper_2/MDPI_template_APA/figures/figure5"
    mdpi_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if ext in {"png", "pdf"}:
            kwargs["dpi"] = 600
        fig.savefig(mdpi_base.with_suffix(f".{ext}"), **kwargs)
    
    print(f"Saved Figure 5 to {out_base}, Paper_2/figures, and MDPI figures")

if __name__ == "__main__":
    main()
