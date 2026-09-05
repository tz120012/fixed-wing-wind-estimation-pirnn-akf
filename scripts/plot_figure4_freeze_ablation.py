import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_history(model_dir: Path):
    import torch
    ckpt_path = model_dir / "best_composite_model.pth"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "best_model.pth"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("history", None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True, help="Path to Model A (No freeze) directory")
    parser.add_argument("--model_b", type=str, required=True, help="Path to Model B (Freeze) directory")
    parser.add_argument("--out_dir", type=str, default="data/figure4")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_a = load_history(PROJECT_ROOT / args.model_a)
    hist_b = load_history(PROJECT_ROOT / args.model_b)

    if not hist_a or not hist_b:
        print("Could not load history for one or both models.")
        return

    # Extract metrics
    epochs_a = np.arange(len(hist_a['train_loss']))
    epochs_b = np.arange(len(hist_b['train_loss']))

    # 1. Physics Loss
    phys_a = hist_a['train_physics_loss']
    phys_b = hist_b['train_physics_loss']

    # 2. Wind RMSE
    rmse_a = hist_a['val_rmse']
    rmse_b = hist_b['val_rmse']

    # 3. Angle Magnitude (alpha/beta)
    angle_mag_a = np.array(hist_a.get('val_angle_mag', []))
    angle_mag_b = np.array(hist_b.get('val_angle_mag', []))
    
    # Plotting
    # Set academic plotting style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.4,
        'grid.linestyle': '--'
    })

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    
    color_a = '#E64B35' # Red
    color_b = '#1F77B4' # Blue
    
    # (a) Physics Loss
    axs[0].plot(epochs_a, phys_a, label='Model A (No Freeze)', color=color_a, alpha=0.9, linewidth=1.5)
    axs[0].plot(epochs_b, phys_b, label='Model B (Freeze-Anneal)', color=color_b, alpha=0.9, linewidth=1.5)
    axs[0].set_yscale('log')
    axs[0].set_title('(a) Physics Loss (Log Scale)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(fontsize=9, frameon=True, edgecolor='black', fancybox=False)
    axs[0].grid(True)

    # (b) Validation Wind RMSE
    axs[1].plot(epochs_a, rmse_a, label='Model A (No Freeze)', color=color_a, alpha=0.9, linewidth=1.5)
    axs[1].plot(epochs_b, rmse_b, label='Model B (Freeze-Anneal)', color=color_b, alpha=0.9, linewidth=1.5)
    axs[1].set_title('(b) Validation Wind RMSE')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('RMSE (m/s)')
    axs[1].legend(fontsize=9, frameon=True, edgecolor='black', fancybox=False)
    axs[1].grid(True)

    # (c) Angle Correction Magnitude
    if len(angle_mag_a) > 0 and len(angle_mag_b) > 0:
        # Convert to degrees
        mag_a_deg = np.degrees(np.mean(angle_mag_a[:, :2], axis=1))
        mag_b_deg = np.degrees(np.mean(angle_mag_b[:, :2], axis=1))
        axs[2].plot(epochs_a, mag_a_deg, label='Model A (No Freeze)', color=color_a, alpha=0.9, linewidth=1.5)
        axs[2].plot(epochs_b, mag_b_deg, label='Model B (Freeze-Anneal)', color=color_b, alpha=0.9, linewidth=1.5)
        axs[2].axvspan(0, 50, color='gray', alpha=0.15, label='Freeze Period (Model B)')
        axs[2].set_title(r'(c) Mean $\Delta\alpha, \Delta\beta$ Magnitude')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Degrees (°)')
        axs[2].legend(fontsize=9, frameon=True, edgecolor='black', fancybox=False)
        axs[2].grid(True)

    for ax in axs:
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

    plt.tight_layout()
    
    png_path = out_dir / "figure4_freeze_ablation.png"
    svg_path = out_dir / "figure4_freeze_ablation.svg"
    pdf_path = out_dir / "figure4_freeze_ablation.pdf"
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=600, bbox_inches='tight')
    print(f"Saved Figure 4 to {png_path}, {svg_path}, and {pdf_path}")

if __name__ == "__main__":
    main()
