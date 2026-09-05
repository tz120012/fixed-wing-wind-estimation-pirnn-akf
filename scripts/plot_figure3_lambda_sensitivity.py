import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper_plot_style import BLUE, GREEN, GRAY, RED, apply_style, format_axes, save_figure
from src.experiments import paper_evidence_chain_eval as evidence


def parse_lambda_from_dirname(dirname: str) -> float:
    """Extract lambda value from directory name like 'train_lambda0.05_...'."""
    parts = dirname.split("_")
    for part in parts:
        if part.startswith("lambda"):
            try:
                return float(part[6:])
            except ValueError:
                pass
    raise ValueError(f"Could not parse lambda from {dirname}")


def evaluate_model_for_lambda(
    model_dir: Path,
    X: np.ndarray,
    wind_true: np.ndarray,
    scaler_X,
    scaler_y,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a single PI-GRU model and return metrics."""
    # Load model (need to point to the actual .pth file)
    ckpt_path = model_dir / "best_composite_model.pth"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "best_model.pth"
    model = evidence.load_pigru_model(ckpt_path, scaler_X, scaler_y, device)
    
    # Predict
    pigru_out = evidence.predict_pigru(model, X, batch_size, device)
    wind_pred = evidence.denorm_wind(pigru_out["wind"], scaler_y)
    
    # Calculate RMSE and MAE
    err = wind_pred - wind_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    
    # Direction MAE
    true_dir = np.degrees(np.arctan2(wind_true[:, 1], wind_true[:, 0]))
    pred_dir = np.degrees(np.arctan2(wind_pred[:, 1], wind_pred[:, 0]))
    dir_diff = (pred_dir - true_dir + 180.0) % 360.0 - 180.0
    dir_mae = float(np.mean(np.abs(dir_diff)))
    
    # Calculate Physics Residual (Airspeed closure error)
    # Reconstruct airspeed using predicted wind
    last_phys = evidence.denorm_last_step(X, scaler_X)
    vg_ned = last_phys[:, 0:3]
    roll, pitch, yaw = last_phys[:, 9], last_phys[:, 10], last_phys[:, 11]
    tas_true = last_phys[:, 19]
    
    # Calculate R_b2n for all samples
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    R_b2n = np.zeros((len(X), 3, 3))
    R_b2n[:, 0, 0] = cp * cy
    R_b2n[:, 0, 1] = sr * sp * cy - cr * sy
    R_b2n[:, 0, 2] = cr * sp * cy + sr * sy
    R_b2n[:, 1, 0] = cp * sy
    R_b2n[:, 1, 1] = sr * sp * sy + cr * cy
    R_b2n[:, 1, 2] = cr * sp * sy - sr * cy
    R_b2n[:, 2, 0] = -sp
    R_b2n[:, 2, 1] = sr * cp
    R_b2n[:, 2, 2] = cr * cp
    
    # Calculate air velocity in NED frame
    v_air_ned = vg_ned - wind_pred
    
    # Rotate to body frame
    v_air_body = np.einsum('nij,nj->ni', np.transpose(R_b2n, (0, 2, 1)), v_air_ned)
    
    # Reconstructed airspeed
    tas_reconstructed = np.linalg.norm(v_air_body, axis=1)
    
    # Physics residual (RMSE of airspeed)
    physics_residual = float(np.sqrt(np.mean((tas_reconstructed - tas_true) ** 2)))
    
    return {
        "rmse": rmse,
        "dir_mae": dir_mae,
        "physics_residual": physics_residual
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 3 (Lambda Sensitivity)")
    parser.add_argument("--models-dir", type=str, 
                        default="train_data1/train_lambda0.01_0.03_0.05_0.07_0.09_0.11_0.13_0.15_0.17_0.19_20260518_235629",
                        help="Directory containing the lambda scan models")
    parser.add_argument("--data-dir", type=str, default="data/dataset_new_processed",
                        help="Processed dataset directory")
    parser.add_argument("--output-dir", type=str, default="data/figure3",
                        help="Output directory for figure and csv")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    models_dir = PROJECT_ROOT / args.models_dir
    data_dir = PROJECT_ROOT / args.data_dir
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find all lambda model directories
    model_dirs = []
    for d in models_dir.iterdir():
        if d.is_dir() and d.name.startswith("train_lambda"):
            try:
                lam = parse_lambda_from_dirname(d.name)
                model_dirs.append((lam, d))
            except ValueError:
                continue
                
    model_dirs.sort(key=lambda x: x[0])  # Sort by lambda value
    
    if not model_dirs:
        raise ValueError(f"No valid lambda model directories found in {models_dir}")
        
    print(f"Found {len(model_dirs)} models with lambda values: {[m[0] for m in model_dirs]}")

    # Load Test-ID data
    print("Loading Test-ID data...")
    scaler_X, scaler_y = evidence.load_norm_params(data_dir)
    X = np.load(data_dir / "X_test_id.npy")
    y = np.load(data_dir / "y_test_id.npy")
    wind_true = evidence.denorm_y(y, scaler_y)[:, :3]

    # Evaluate each model
    results = []
    for lam, mdir in model_dirs:
        print(f"\nEvaluating lambda = {lam} ({mdir.name})")
        metrics = evaluate_model_for_lambda(
            mdir, X, wind_true, scaler_X, scaler_y, args.batch_size, device
        )
        metrics["lambda_physics"] = lam
        results.append(metrics)
        print(f"  RMSE: {metrics['rmse']:.4f} m/s")
        print(f"  Dir MAE: {metrics['dir_mae']:.4f}°")
        print(f"  Physics Res: {metrics['physics_residual']:.4f} m/s")

    # Save to CSV
    df = pd.DataFrame(results)
    df = df[["lambda_physics", "rmse", "dir_mae", "physics_residual"]]
    csv_path = out_dir / "figure3_lambda_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")

    # Plot Figure 3
    print("Plotting Figure 3...")
    
    apply_style()

    fig, axs = plt.subplots(3, 1, figsize=(7.0, 5.2), sharex=True)

    color_rmse = BLUE
    color_dir = RED
    color_phys = GREEN

    panels = [
        (axs[0], "rmse", color_rmse, "Wind RMSE (m/s)", "(a) Wind-Speed Error"),
        (axs[1], "physics_residual", color_phys, "Closure Residual (m/s)", "(b) Physics Closure"),
        (axs[2], "dir_mae", color_dir, "Direction MAE (deg)", "(c) Direction Error"),
    ]
    for ax, col, color, ylabel, title in panels:
        ax.axvspan(0.09, 0.13, color=GRAY, alpha=0.15)
        ax.axvline(0.10, color="black", linestyle="--", linewidth=0.9)
        ax.plot(df["lambda_physics"], df[col], marker="o", color=color, linewidth=1.4, markersize=4)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        format_axes(ax, grid_axis="both")

    axs[-1].set_xlabel(r"Physics Loss Weight ($\lambda_{\mathrm{physics}}$)")
    axs[0].text(0.092, axs[0].get_ylim()[1] * 0.97, "Balanced region", color=GRAY, fontsize=8, va="top")
    axs[0].text(0.102, axs[0].get_ylim()[0] + (axs[0].get_ylim()[1] - axs[0].get_ylim()[0]) * 0.08,
                r"$\lambda=0.10$", color="black", fontsize=8)
    fig.tight_layout(h_pad=0.8)

    # Save plots
    save_figure(fig, out_dir / "figure3_lambda_sensitivity", copy_to_paper="figure3")
    plt.close()

    print(f"Saved figure to {out_dir / 'figure3_lambda_sensitivity'} and paper/figures/figure3")


if __name__ == "__main__":
    main()
