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


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_starts(text: str | None, dataset_len: int, window_size: int, n_windows: int) -> list[int]:
    max_start = dataset_len - window_size
    if max_start < 0:
        raise ValueError(f"window_size={window_size} exceeds dataset length={dataset_len}")
    if text:
        starts = [int(x.strip()) for x in text.split(",") if x.strip()]
    else:
        starts = np.linspace(0, max_start, n_windows, dtype=int).tolist()
    starts = sorted(set(starts))
    invalid = [s for s in starts if s < 0 or s > max_start]
    if invalid:
        raise ValueError(f"Invalid starts {invalid}; valid range is [0, {max_start}]")
    return starts


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


def evaluate_output(
    method: str,
    wind: np.ndarray,
    wind_true: np.ndarray,
    last_phys: np.ndarray,
    anomaly_slice: slice,
    boundary_slice: slice,
) -> dict[str, float | str]:
    win = wind[anomaly_slice]
    return {
        "method": method,
        "anomaly_window_h_rmse_mps": horizontal_rmse(wind_true[anomaly_slice], win),
        "anomaly_window_jitter_mean": jitter_mean(win),
        "max_step_jump_mps": max_step_jump(wind[boundary_slice]),
        "airspeed_closure_rmse_mps": closure_rmse(win, last_phys[anomaly_slice]),
    }


def add_relative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["jitter_reduction_vs_pigru_pct"] = np.nan
    out["max_jump_reduction_vs_pigru_pct"] = np.nan
    group_cols = ["start_idx", "anomaly_type"]
    for _, idx in out.groupby(group_cols).groups.items():
        sub = out.loc[idx]
        base = sub[sub["method"] == "PI-GRU (Raw)"]
        if base.empty:
            continue
        base_jitter = float(base["anomaly_window_jitter_mean"].iloc[0])
        base_jump = float(base["max_step_jump_mps"].iloc[0])
        if base_jitter > 0:
            out.loc[idx, "jitter_reduction_vs_pigru_pct"] = (
                1.0 - out.loc[idx, "anomaly_window_jitter_mean"] / base_jitter
            ) * 100.0
        if base_jump > 0:
            out.loc[idx, "max_jump_reduction_vs_pigru_pct"] = (
                1.0 - out.loc[idx, "max_step_jump_mps"] / base_jump
            ) * 100.0
    return out


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "anomaly_window_h_rmse_mps",
        "anomaly_window_jitter_mean",
        "max_step_jump_mps",
        "airspeed_closure_rmse_mps",
        "jitter_reduction_vs_pigru_pct",
        "max_jump_reduction_vs_pigru_pct",
    ]
    agg = df.groupby("method")[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]

    # A conservative balanced score: lower is better. It normalizes each cost by
    # PI-GRU raw means, so an all-around method must not win by smoothing alone.
    base = agg[agg["method"] == "PI-GRU (Raw)"].iloc[0]
    costs = {
        "anomaly_window_h_rmse_mps_mean": float(base["anomaly_window_h_rmse_mps_mean"]),
        "anomaly_window_jitter_mean_mean": float(base["anomaly_window_jitter_mean_mean"]),
        "max_step_jump_mps_mean": float(base["max_step_jump_mps_mean"]),
        "airspeed_closure_rmse_mps_mean": float(base["airspeed_closure_rmse_mps_mean"]),
    }
    score = np.zeros(len(agg), dtype=float)
    for col, denom in costs.items():
        if denom > 0:
            score += agg[col].to_numpy(dtype=float) / denom
    agg["balanced_score_lower_is_better"] = score / len(costs)
    return agg.sort_values("balanced_score_lower_is_better")


def plot_tradeoff(summary: pd.DataFrame, out_dir: Path) -> None:
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

    methods = summary["method"].tolist()
    color_map = {
        "PI-GRU (Raw)": "#7F7F7F",
        "PIRNN-AKF": "#1F77B4",
    }
    colors = [color_map.get(m, "#00A087" if "EMA" in m else "#4DBBD5") for m in methods]

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axs[0].scatter(
        summary["anomaly_window_h_rmse_mps_mean"],
        summary["anomaly_window_jitter_mean_mean"],
        s=80,
        c=colors,
        edgecolors="black",
        linewidths=0.8,
    )
    for _, row in summary.iterrows():
        axs[0].annotate(row["method"], (row["anomaly_window_h_rmse_mps_mean"], row["anomaly_window_jitter_mean_mean"]),
                        xytext=(4, 3), textcoords="offset points", fontsize=8)
    axs[0].set_xlabel("Mean Horizontal RMSE (m/s)")
    axs[0].set_ylabel("Mean Jitter")
    axs[0].set_title("Tracking-Smoothing Trade-off")
    axs[0].grid(True)

    x = np.arange(len(methods))
    axs[1].bar(x, summary["balanced_score_lower_is_better"], color=colors, edgecolor="black", linewidth=0.8)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(methods, rotation=25, ha="right")
    axs[1].set_ylabel("Balanced Score (lower is better)")
    axs[1].set_title("Cross-Anomaly Balanced Cost")
    axs[1].grid(True, axis="y")

    for ax in axs:
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

    plt.tight_layout()
    for suffix, kwargs in {
        ".png": {"dpi": 600},
        ".svg": {},
        ".pdf": {"dpi": 600},
    }.items():
        plt.savefig(out_dir / f"figure5_akf_vs_ema_anomaly_sweep{suffix}", bbox_inches="tight", **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-window anomaly sweep for AKF vs fixed EMA baselines.")
    parser.add_argument("--model", type=str, required=True, help="Path to PI-GRU model directory")
    parser.add_argument("--data_dir", type=str, default="data/dataset_new_processed")
    parser.add_argument("--out_dir", type=str, default="data/figure5")
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--starts", type=str, default=None, help="Comma-separated Test-OOD start indices")
    parser.add_argument("--n_windows", type=int, default=8, help="Uniform windows if --starts is omitted")
    parser.add_argument("--anomaly_types", type=str, default="gps_spike,tas_spike,sensor_dropout,gaussian_burst")
    parser.add_argument("--anomaly_strength", type=float, default=3.0)
    parser.add_argument("--ema_alphas", type=str, default="0.1,0.3,0.5,0.7,0.9")
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
    wind_true_all = evidence.denorm_y(y_full, scaler_y)[:, :3]
    starts = parse_starts(args.starts, len(X_full), args.window_size, args.n_windows)
    anomaly_types = parse_csv_strings(args.anomaly_types)
    ema_alphas = parse_csv_floats(args.ema_alphas)

    ckpt_path = model_dir / "best_composite_model.pth"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "best_model.pth"
    model = evidence.load_pigru_model(ckpt_path, scaler_X, scaler_y, device)
    with open(PROJECT_ROOT / "config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    rows = []
    print(f"Evaluating {len(starts)} windows x {len(anomaly_types)} anomaly types.")
    for start_idx in starts:
        end_idx = start_idx + args.window_size
        X_window = X_full[start_idx:end_idx].copy()
        wind_true = wind_true_all[start_idx:end_idx]
        for anomaly_idx, anomaly_type in enumerate(anomaly_types):
            X_anom, anom_start, anom_end = evidence.inject_anomaly(
                X_window, scaler_X, anomaly_type, args.anomaly_strength, seed=42 + anomaly_idx
            )
            last_phys = evidence.denorm_last_step(X_anom, scaler_X)
            pigru_out = evidence.predict_pigru(model, X_anom, args.batch_size, device)
            pigru_wind = evidence.denorm_wind(pigru_out["wind"], scaler_y)
            akf_wind, _ = evidence.run_fast_pirnn_akf(
                config, pigru_out, X_anom, scaler_X, scaler_y, continuous=True
            )
            outputs = {
                "PI-GRU (Raw)": pigru_wind,
                "PIRNN-AKF": akf_wind,
            }
            for alpha in ema_alphas:
                ema_wind = np.zeros_like(pigru_wind)
                for axis in range(3):
                    ema_wind[:, axis] = compute_ema(pigru_wind[:, axis], alpha=alpha)
                outputs[f"EMA alpha={alpha:g}"] = ema_wind

            anomaly_slice = slice(anom_start, anom_end)
            boundary_slice = slice(max(0, anom_start - 20), min(args.window_size, anom_end + 20))
            for method, wind in outputs.items():
                row = evaluate_output(method, wind, wind_true, last_phys, anomaly_slice, boundary_slice)
                row.update({
                    "start_idx": start_idx,
                    "window_size": args.window_size,
                    "anomaly_type": anomaly_type,
                    "anomaly_strength": args.anomaly_strength,
                    "anomaly_start_step": anom_start,
                    "anomaly_end_step": anom_end,
                })
                rows.append(row)

    detail = add_relative_metrics(pd.DataFrame(rows))
    summary = aggregate_results(detail)
    detail_path = out_dir / "figure5_akf_vs_ema_anomaly_sweep_detail.csv"
    summary_path = out_dir / "figure5_akf_vs_ema_anomaly_sweep_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_tradeoff(summary, out_dir)

    print(f"Saved detail CSV to {detail_path}")
    print(f"Saved summary CSV to {summary_path}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
