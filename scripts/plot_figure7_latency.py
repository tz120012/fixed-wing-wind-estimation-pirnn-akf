import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import BLUE, CYAN, GRAY, GREEN, RED, apply_style, format_axes, save_figure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTROL_PERIOD_MS = 20.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hitl_dir", type=str, default="HITL/logs_in_rasbpi")
    parser.add_argument("--out_dir", type=str, default="data/figure7")
    args = parser.parse_args()

    hitl_dir = PROJECT_ROOT / args.hitl_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "hitl_data_20260416_204239.csv": "PyTorch",
        "hitl_data_20260416_210242.csv": "ONNX Runtime",
        "hitl_data_20260416_212248.csv": "PyTorch (AKF)",
        "hitl_data_20260416_214252.csv": "ONNX Runtime (AKF)"
    }

    dfs = []
    for filename, label in mapping.items():
        filepath = hitl_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df["Backend"] = label
            df = df.iloc[100:]
            dfs.append(df)
            
    if not dfs:
        print("No HITL data found!")
        return
        
    all_data = pd.concat(dfs, ignore_index=True)
    all_data = all_data.dropna(subset=["inference_ms", "latency_e2e_ms"])
    order = ["PyTorch", "ONNX Runtime", "PyTorch (AKF)", "ONNX Runtime (AKF)"]
    available = [name for name in order if name in set(all_data["Backend"])]
    summary = all_data.groupby("Backend").agg(
        inference_mean_ms=("inference_ms", "mean"),
        inference_p95_ms=("inference_ms", lambda x: float(np.percentile(x, 95))),
        e2e_mean_ms=("latency_e2e_ms", "mean"),
        e2e_p95_ms=("latency_e2e_ms", lambda x: float(np.percentile(x, 95))),
    ).reindex(available)
    summary.to_csv(out_dir / "figure7_latency_summary.csv")

    apply_style()

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Subplot (a): real-time latency budget instead of a raw engineering boxplot.
    budget_labels = ["PI-GRU\nmean inference", "PIRNN-AKF\nmean inference", "HITL E2E\n$p_{95}$"]
    onnx = summary.loc["ONNX Runtime"]
    onnx_akf = summary.loc["ONNX Runtime (AKF)"]
    budget_values = [
        float(onnx["inference_mean_ms"]),
        float(onnx_akf["inference_mean_ms"]),
        float(onnx_akf["e2e_p95_ms"]),
    ]
    y = np.arange(len(budget_values))
    colors_budget = [GRAY, BLUE, RED]
    axs[0].barh(y, budget_values, color=colors_budget, edgecolor="black", linewidth=0.8, height=0.55)
    axs[0].axvline(CONTROL_PERIOD_MS, color="black", linestyle="--", linewidth=1.0, label="20 ms control period")
    for yi, val in zip(y, budget_values):
        axs[0].text(val + 0.35, yi, f"{val:.2f} ms", va="center", ha="left", fontsize=8)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(budget_labels)
    axs[0].set_xlabel("Latency (ms)")
    axs[0].set_title("(a) Real-Time Latency Budget")
    axs[0].set_xlim(0, CONTROL_PERIOD_MS * 1.12)
    axs[0].invert_yaxis()
    axs[0].legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False, fontsize=7)

    # Subplot (b): End-to-End Latency CDF
    line_styles = ['-', '--', '-.', ':']
    colors = [GRAY, BLUE, CYAN, RED]
    
    for i, backend in enumerate(available):
        subset = all_data[all_data["Backend"] == backend]["latency_e2e_ms"].dropna()
        if len(subset) > 0:
            x = np.sort(subset)
            y = np.arange(1, len(x) + 1) / len(x)
            axs[1].plot(x, y, linewidth=1.4, linestyle=line_styles[i % 4], color=colors[i % 4], label=backend)

    p95 = float(onnx_akf["e2e_p95_ms"])
    axs[1].axvline(p95, color=RED, linestyle="--", linewidth=1.0)
    axs[1].axvline(CONTROL_PERIOD_MS, color="black", linestyle="--", linewidth=1.0)
    axs[1].text(p95 + 0.05, 0.58, f"$p_{{95}}$={p95:.2f} ms", color=RED, fontsize=8, rotation=90, va="bottom")
    axs[1].text(CONTROL_PERIOD_MS + 0.2, 0.10, "20 ms", color="black", fontsize=8, rotation=90, va="bottom")

    axs[1].set_xlabel("End-to-End Latency (ms)")
    axs[1].set_ylabel("Cumulative Probability")
    axs[1].set_title("(b) HITL End-to-End Latency CDF")
    
    axs[1].legend(
        loc='lower left',
        bbox_to_anchor=(0.08, 0.02),
        frameon=True,
        edgecolor='black',
        fancybox=False,
        fontsize=7,
        ncol=1,
        handlelength=1.4,
        columnspacing=0.8,
        framealpha=0.88,
    )
    
    p95_max = all_data["latency_e2e_ms"].quantile(0.95)
    axs[1].set_xlim(max(0, all_data["latency_e2e_ms"].quantile(0.01) * 0.85), max(p95_max * 1.18, CONTROL_PERIOD_MS * 1.08))
    axs[1].set_ylim(0, 1.0)

    for ax in axs:
        format_axes(ax, grid_axis="x" if ax is axs[0] else "both")

    plt.tight_layout()
    
    save_figure(fig, out_dir / "figure7_latency", copy_to_paper="figure7")
    
    print(f"Saved Figure 7 to {out_dir / 'figure7_latency'} and paper/figures/figure7")

if __name__ == "__main__":
    main()
