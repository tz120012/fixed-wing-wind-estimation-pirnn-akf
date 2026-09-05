import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper_plot_style import BLUE, GRAY, ORANGE, RED, apply_style, format_axes, save_figure
from src.experiments import paper_evidence_chain_eval as evidence

def evaluate_model(model_dir: Path, X: np.ndarray, scaler_X, scaler_y, batch_size: int, device: torch.device):
    ckpt_path = model_dir / "best_composite_model.pth"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "best_model.pth"
    model = evidence.load_pigru_model(ckpt_path, scaler_X, scaler_y, device)
    out = evidence.predict_pigru(model, X, batch_size, device)
    wind_pred = evidence.denorm_wind(out["wind"], scaler_y)
    return wind_pred

def calculate_metrics(wind_true, wind_pred, mask):
    err = wind_pred[mask] - wind_true[mask]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    
    true_dir = np.degrees(np.arctan2(wind_true[mask, 1], wind_true[mask, 0]))
    pred_dir = np.degrees(np.arctan2(wind_pred[mask, 1], wind_pred[mask, 0]))
    dir_diff = (pred_dir - true_dir + 180.0) % 360.0 - 180.0
    dir_mae = float(np.mean(np.abs(dir_diff)))
    dir_p95 = float(np.percentile(np.abs(dir_diff), 95))
    
    pred_mag = np.linalg.norm(wind_pred[mask, :2], axis=1)
    collapse_ratio = float(np.mean(pred_mag < 0.2)) * 100
    
    return rmse, dir_mae, dir_p95, collapse_ratio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_c", type=str, required=True, help="Path to Model C (No Anti-Collapse)")
    parser.add_argument("--model_d", type=str, required=True, help="Path to Model D (With Anti-Collapse)")
    parser.add_argument("--data_dir", type=str, default="data/dataset_new_processed")
    parser.add_argument("--out_dir", type=str, default="data/figure4")
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading Test-ID data...")
    scaler_X, scaler_y = evidence.load_norm_params(data_dir)
    X = np.load(data_dir / "X_test_id.npy")
    y = np.load(data_dir / "y_test_id.npy")
    wind_true = evidence.denorm_y(y, scaler_y)[:, :3]

    print("Evaluating Model C (Baseline)...")
    pred_c = evaluate_model(PROJECT_ROOT / args.model_c, X, scaler_X, scaler_y, args.batch_size, device)
    
    print("Evaluating Model D (Hinge + SNR-aware)...")
    pred_d = evaluate_model(PROJECT_ROOT / args.model_d, X, scaler_X, scaler_y, args.batch_size, device)

    # Filter for weak wind
    true_mag = np.linalg.norm(wind_true[:, :2], axis=1)
    weak_mask = true_mag < 1.5
    
    print(f"\nWeak wind samples (<1.5 m/s): {np.sum(weak_mask)} / {len(true_mag)}")
    
    rmse_c, dir_mae_c, dir_p95_c, col_c = calculate_metrics(wind_true, pred_c, weak_mask)
    rmse_d, dir_mae_d, dir_p95_d, col_d = calculate_metrics(wind_true, pred_d, weak_mask)
    
    print("\n--- Quantitative Results on Weak Wind Subset ---")
    print(f"Model C (Baseline): RMSE={rmse_c:.2f} m/s, Dir P95={dir_p95_c:.1f}°, Collapse Ratio={col_c:.1f}%")
    print(f"Model D (Proposed): RMSE={rmse_d:.2f} m/s, Dir P95={dir_p95_d:.1f}°, Collapse Ratio={col_d:.1f}%")

    # Plotting
    apply_style()

    fig, axs = plt.subplots(1, 3, figsize=(7.8, 3.05), gridspec_kw={"width_ratios": [1.0, 1.15, 1.1]})
    
    # NOTE: keep this consistent with the rest of the paper's color convention
    # (blue = PI-GRU family, red = PIRNN-AKF). The ablated "PI-GRU (no weak-wind
    # reg.)" baseline here is neither the plain PI-GRU-only front end nor the
    # AKF-fused model, so it uses orange (the "alternative/ablated variant"
    # color used elsewhere for EMA/Vanilla-GRU comparators) instead of red,
    # which is reserved for PIRNN-AKF in every other figure.
    color_c = ORANGE
    color_d = BLUE
    
    pred_mag_c = np.linalg.norm(pred_c[weak_mask, :2], axis=1)
    pred_mag_d = np.linalg.norm(pred_d[weak_mask, :2], axis=1)
    true_mag_weak = true_mag[weak_mask]

    # Subplot (a): predicted magnitude distribution in weak-wind samples.
    parts = axs[0].violinplot(
        [pred_mag_c, pred_mag_d],
        positions=[0, 1],
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], [color_c, color_d]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.35)
        body.set_linewidth(0.8)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.0)
    box = axs[0].boxplot(
        [pred_mag_c, pred_mag_d],
        positions=[0, 1],
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], [color_c, color_d]):
        patch.set_facecolor(color)
        patch.set_alpha(0.20)
    axs[0].axhline(float(np.median(true_mag_weak)), color="black", linestyle="--", linewidth=1.0, label="True median")
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(["PI-GRU", "PI-GRU\n+ weak-wind reg."])
    axs[0].set_ylabel("Predicted Wind Magnitude\n(m/s)")
    axs[0].set_ylim(0, 2.0)
    axs[0].set_title("(a) Weak-Wind Magnitude")
    axs[0].legend(frameon=True, edgecolor="black", fancybox=False, loc="upper right", fontsize=7)

    # Subplot (b): Direction Error CDF
    true_dir = np.degrees(np.arctan2(wind_true[weak_mask, 1], wind_true[weak_mask, 0]))
    pred_dir_c = np.degrees(np.arctan2(pred_c[weak_mask, 1], pred_c[weak_mask, 0]))
    pred_dir_d = np.degrees(np.arctan2(pred_d[weak_mask, 1], pred_d[weak_mask, 0]))
    
    err_c = np.abs((pred_dir_c - true_dir + 180.0) % 360.0 - 180.0)
    err_d = np.abs((pred_dir_d - true_dir + 180.0) % 360.0 - 180.0)
    tail45_c = float(np.mean(err_c > 45.0) * 100)
    tail45_d = float(np.mean(err_d > 45.0) * 100)
    tail90_c = float(np.mean(err_c > 90.0) * 100)
    tail90_d = float(np.mean(err_d > 90.0) * 100)

    summary = pd.DataFrame([
        {
            "method": "PI-GRU",
            "weak_rmse_mps": rmse_c,
            "direction_mae_deg": dir_mae_c,
            "direction_p95_deg": dir_p95_c,
            "collapse_ratio_pct": col_c,
            "direction_error_gt45_pct": tail45_c,
            "direction_error_gt90_pct": tail90_c,
        },
        {
            "method": "PI-GRU w/ weak-wind reg.",
            "weak_rmse_mps": rmse_d,
            "direction_mae_deg": dir_mae_d,
            "direction_p95_deg": dir_p95_d,
            "collapse_ratio_pct": col_d,
            "direction_error_gt45_pct": tail45_d,
            "direction_error_gt90_pct": tail90_d,
        },
    ])
    summary.to_csv(out_dir / "figure4_weak_wind_summary.csv", index=False)
    
    # Subplot (b): direction-tail indicators normalized by the baseline.
    tail_metrics = [
        ("P95", dir_p95_c, dir_p95_d, "°"),
        (">45°", tail45_c, tail45_d, "%"),
        (">90°", tail90_c, tail90_d, "%"),
    ]
    x_tail = np.arange(len(tail_metrics))
    tail_width = 0.36
    tail_c = np.full(len(tail_metrics), 100.0)
    tail_d = np.array([100.0 * m[2] / m[1] if m[1] > 0 else np.nan for m in tail_metrics])
    axs[1].bar(x_tail - tail_width / 2, tail_c, tail_width, label="PI-GRU", color=color_c, alpha=0.9, edgecolor="black", linewidth=0.8)
    axs[1].bar(x_tail + tail_width / 2, tail_d, tail_width, label="PI-GRU + weak-wind reg.", color=color_d, alpha=0.9, edgecolor="black", linewidth=0.8)
    for i, (_, vc, vd, unit) in enumerate(tail_metrics):
        axs[1].text(i - tail_width / 2, tail_c[i] + 5.0, f"{vc:.1f}{unit}" if unit == "°" else f"{vc:.2f}{unit}",
                    ha="center", va="bottom", fontsize=7, color=color_c)
        axs[1].text(i + tail_width / 2, tail_d[i] + 5.0, f"{vd:.1f}{unit}" if unit == "°" else f"{vd:.2f}{unit}",
                    ha="center", va="bottom", fontsize=7, color=color_d)
    axs[1].set_xticks(x_tail)
    axs[1].set_xticklabels([m[0] for m in tail_metrics])
    axs[1].set_ylabel("Relative to PI-GRU (%)")
    axs[1].set_title("(b) Direction-Tail Indicators")
    axs[1].set_ylim(0, 170)

    # Subplot (c): compact summary of failure-mode indicators.
    metrics = [
        ("Collapse\n<0.2", col_c, col_d),
        ("Dir. err.\n>45°", tail45_c, tail45_d),
        ("Dir. err.\n>90°", tail90_c, tail90_d),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    vals_c = [m[1] for m in metrics]
    vals_d = [m[2] for m in metrics]
    axs[2].bar(x - width / 2, vals_c, width, label="PI-GRU", color=color_c, alpha=0.9, edgecolor="black", linewidth=0.8)
    axs[2].bar(x + width / 2, vals_d, width, label="PI-GRU + weak-wind reg.", color=color_d, alpha=0.9, edgecolor="black", linewidth=0.8)
    for i, (vc, vd) in enumerate(zip(vals_c, vals_d)):
        offset = max(vals_c + vals_d) * 0.045
        axs[2].text(i - width / 2, vc + offset, f"{vc:.1f}", ha="center", va="bottom", fontsize=7, color=color_c)
        axs[2].text(i + width / 2, vd + offset, f"{vd:.1f}", ha="center", va="bottom", fontsize=7, color=color_d)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels([m[0] for m in metrics])
    axs[2].set_ylabel("Sample Ratio (%)")
    axs[2].set_title("(c) Failure Indicators")
    axs[2].set_ylim(0.0, 3.0)
    axs[2].legend(
        frameon=True,
        edgecolor="black",
        fancybox=False,
        loc="upper right",
        ncol=1,
        fontsize=7,
        handlelength=1.0,
    )

    for ax in axs:
        format_axes(ax, grid_axis="y")

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.78, wspace=0.36)
    
    save_figure(fig, out_dir / "figure4_weak_wind", copy_to_paper="figure4")
    print(f"\nSaved Figure 4 to {out_dir / 'figure4_weak_wind'} and paper/figures/figure4")

if __name__ == "__main__":
    main()
