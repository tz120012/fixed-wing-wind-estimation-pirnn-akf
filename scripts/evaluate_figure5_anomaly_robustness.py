import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import paper_evidence_chain_eval as evidence
from plot_figure6_time_series import compute_ema


def jitter_mean(wind: np.ndarray) -> float:
    if len(wind) < 3:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(wind[:, :2], n=2, axis=0), axis=1)))


def max_step_jump(wind: np.ndarray) -> float:
    if len(wind) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(np.diff(wind[:, :2], axis=0), axis=1)))


def horizontal_rmse(wind_true: np.ndarray, wind_pred: np.ndarray) -> float:
    err = wind_pred[:, :2] - wind_true[:, :2]
    return float(np.sqrt(np.mean(err ** 2)))


def closure_rmse(wind_pred: np.ndarray, last_phys: np.ndarray) -> float:
    vg = last_phys[:, 0:3]
    tas = last_phys[:, evidence.FEATURE_IDX["airspeed"]]
    residual = np.linalg.norm(vg - wind_pred, axis=1) - tas
    return float(np.sqrt(np.mean(residual ** 2)))


def build_rows(method_outputs, wind_true, last_phys, anomaly_slice, boundary_slice):
    rows = []
    baseline_jitter = jitter_mean(method_outputs["PI-GRU (Raw)"][anomaly_slice])
    baseline_jump = max_step_jump(method_outputs["PI-GRU (Raw)"][boundary_slice])
    for method, wind in method_outputs.items():
        win = wind[anomaly_slice]
        boundary = wind[boundary_slice]
        rows.append({
            "method": method,
            "anomaly_window_h_rmse_mps": horizontal_rmse(wind_true[anomaly_slice], win),
            "anomaly_window_jitter_mean": jitter_mean(win),
            "jitter_reduction_vs_pigru_pct": (1.0 - jitter_mean(win) / baseline_jitter) * 100.0 if baseline_jitter > 0 else 0.0,
            "max_step_jump_mps": max_step_jump(boundary),
            "max_step_jump_reduction_vs_pigru_pct": (1.0 - max_step_jump(boundary) / baseline_jump) * 100.0 if baseline_jump > 0 else 0.0,
            "airspeed_closure_rmse_mps": closure_rmse(win, last_phys[anomaly_slice]),
        })
    return pd.DataFrame(rows)


def plot_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 1.1,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })

    methods = df["method"].tolist()
    colors = ["#7F7F7F", "#00A087", "#1F77B4"]
    metrics = [
        ("anomaly_window_h_rmse_mps", "Horizontal RMSE (m/s)", "(a) Error in Spike Window"),
        ("anomaly_window_jitter_mean", "Jitter Mean", "(b) Jitter in Spike Window"),
        ("max_step_jump_mps", "Max Step Jump (m/s)", "(c) Maximum Instantaneous Jump"),
        ("airspeed_closure_rmse_mps", "Closure RMSE (m/s)", "(d) Airspeed Closure Consistency"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.8))
    for ax, (col, ylabel, title) in zip(axs.ravel(), metrics):
        values = df[col].to_numpy(dtype=float)
        ax.bar(np.arange(len(methods)), values, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y")
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

    plt.tight_layout()
    png_path = out_dir / "figure5_anomaly_robustness_metrics.png"
    svg_path = out_dir / "figure5_anomaly_robustness_metrics.svg"
    pdf_path = out_dir / "figure5_anomaly_robustness_metrics.pdf"
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=600, bbox_inches="tight")
    print(f"Saved anomaly robustness figure to {png_path}, {svg_path}, and {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Figure 5 GPS-spike anomaly robustness metrics.")
    parser.add_argument("--model", type=str, required=True, help="Path to PI-GRU model directory")
    parser.add_argument("--data_dir", type=str, default="data/dataset_new_processed")
    parser.add_argument("--out_dir", type=str, default="data/figure5")
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--start_idx", type=int, default=29000)
    parser.add_argument("--anomaly_type", type=str, default="gps_spike")
    parser.add_argument("--anomaly_strength", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / args.data_dir
    model_dir = PROJECT_ROOT / args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_X, scaler_y = evidence.load_norm_params(data_dir)
    X_full = np.load(data_dir / "X_test_ood.npy")
    y_full = np.load(data_dir / "y_test_ood.npy")
    end_idx = args.start_idx + args.window_size
    if args.start_idx < 0 or end_idx > len(X_full):
        raise ValueError(f"Invalid window [{args.start_idx}, {end_idx}) for dataset length {len(X_full)}")

    X_window = X_full[args.start_idx:end_idx].copy()
    y_window = y_full[args.start_idx:end_idx].copy()
    wind_true = evidence.denorm_y(y_window, scaler_y)[:, :3]
    X_anom, anom_start, anom_end = evidence.inject_anomaly(
        X_window, scaler_X, args.anomaly_type, args.anomaly_strength, seed=42
    )
    last_phys_anom = evidence.denorm_last_step(X_anom, scaler_X)

    ckpt_path = model_dir / "best_composite_model.pth"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "best_model.pth"
    model = evidence.load_pigru_model(ckpt_path, scaler_X, scaler_y, device)

    pigru_out = evidence.predict_pigru(model, X_anom, args.batch_size, device)
    pigru_wind = evidence.denorm_wind(pigru_out["wind"], scaler_y)
    ema_wind = np.zeros_like(pigru_wind)
    for i in range(3):
        ema_wind[:, i] = compute_ema(pigru_wind[:, i], alpha=0.1)

    with open(PROJECT_ROOT / "config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    akf_wind, diagnostics = evidence.run_fast_pirnn_akf(
        config, pigru_out, X_anom, scaler_X, scaler_y, continuous=True
    )

    method_outputs = {
        "PI-GRU (Raw)": pigru_wind,
        "PI-GRU + EMA": ema_wind,
        "PIRNN-AKF": akf_wind,
    }
    anomaly_slice = slice(anom_start, anom_end)
    boundary_slice = slice(max(0, anom_start - 20), min(args.window_size, anom_end + 20))
    df = build_rows(method_outputs, wind_true, last_phys_anom, anomaly_slice, boundary_slice)
    df.insert(0, "anomaly_type", args.anomaly_type)
    df.insert(1, "anomaly_strength", args.anomaly_strength)
    df.insert(2, "start_idx", args.start_idx)
    df.insert(3, "window_size", args.window_size)
    df.insert(4, "anomaly_start_step", anom_start)
    df.insert(5, "anomaly_end_step", anom_end)

    csv_path = out_dir / "figure5_anomaly_robustness_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved anomaly robustness metrics to {csv_path}")

    diag_df = pd.DataFrame({
        "step": np.arange(args.window_size),
        "time_s": np.arange(args.window_size) * 0.02,
        "r_scale_mean": np.mean(pigru_out["r_scale"], axis=1),
        "akf_weight": diagnostics["akf_weight"],
        "nn_weight": diagnostics["nn_weight"],
        "innovation_norm": diagnostics["innovation_norm"],
    })
    diag_path = out_dir / "figure5_anomaly_robustness_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False)
    print(f"Saved AKF diagnostics to {diag_path}")

    plot_metrics(df, out_dir)


if __name__ == "__main__":
    main()
