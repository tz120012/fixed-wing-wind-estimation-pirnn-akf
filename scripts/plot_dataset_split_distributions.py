#!/usr/bin/env python3
"""Generate split-distribution reports for prepared wind datasets."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "lag_aligned": ROOT / "data" / "dataset_lag_aligned",
    "lag_aligned_filtered": ROOT / "data" / "dataset_lag_aligned_filtered",
    "lag_aligned_projected": ROOT / "data" / "dataset_lag_aligned_projected",
}
MAIN_SPLITS = ("train", "val", "test_id")
ALL_SPLITS = ("train", "val", "test_id", "test_ood")
COMPONENTS = ("wind_north", "wind_east", "wind_down", "wind_magnitude")
COLORS = {
    "train": "#4472C4",
    "val": "#70AD47",
    "test_id": "#ED7D31",
    "test_ood": "#C00000",
}
LINESTYLES = {
    "train": "-",
    "val": "--",
    "test_id": "-.",
    "test_ood": ":",
}
LABELS = {
    "train": "Train",
    "val": "Val",
    "test_id": "Test-ID",
    "test_ood": "Test-OOD",
}


def load_scaler_params(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with (dataset_dir / "norm_params.pkl").open("rb") as fh:
        metadata = pickle.load(fh)
    scaler_y = metadata["scaler_y"]
    mean = np.asarray(scaler_y.mean_[:3], dtype=np.float32)
    scale = np.asarray(scaler_y.scale_[:3], dtype=np.float32)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean, scale


def load_wind_labels(dataset_dir: Path, split: str, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    y_path = dataset_dir / f"y_{split}.npy"
    y = np.load(y_path, mmap_mode="r")
    return np.asarray(y[:, :3], dtype=np.float32) * scale + mean


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def density_line(values: np.ndarray, bins: np.ndarray) -> tuple[list[float], list[float]]:
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers.round(4).tolist(), hist.round(8).tolist()


def plot_dataset(
    dataset_name: str,
    split_values: dict[str, dict[str, np.ndarray]],
    splits: tuple[str, ...],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    for idx, component in enumerate(COMPONENTS):
        ax = axes[idx]
        present = [split for split in splits if split in split_values]
        vals = [split_values[split][component] for split in present]
        lo = min(float(np.percentile(v, 0.5)) for v in vals)
        hi = max(float(np.percentile(v, 99.5)) for v in vals)
        if np.isclose(lo, hi):
            lo -= 1.0
            hi += 1.0
        bins = np.linspace(lo, hi, 90)

        for split in present:
            values = split_values[split][component]
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=COLORS[split],
                linestyle=LINESTYLES[split],
                label=f"{LABELS[split]} (n={len(values):,})",
            )

        ax.set_title(component.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Inverse-scaled label value")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)

    fig.suptitle(f"{dataset_name}: Wind Label Distribution by Split", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    outdir = ROOT / "data" / "dataset_distribution_report"
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    canvas_payload: dict[str, object] = {"datasets": []}

    for dataset_name, dataset_dir in DATASETS.items():
        mean, scale = load_scaler_params(dataset_dir)
        split_values: dict[str, dict[str, np.ndarray]] = {}

        for split in ALL_SPLITS:
            if not (dataset_dir / f"y_{split}.npy").exists():
                continue
            wind = load_wind_labels(dataset_dir, split, mean, scale)
            split_values[split] = {
                "wind_north": wind[:, 0],
                "wind_east": wind[:, 1],
                "wind_down": wind[:, 2],
                "wind_magnitude": np.linalg.norm(wind, axis=1),
            }

        train_mag = split_values["train"]["wind_magnitude"]
        train_p05 = float(np.percentile(train_mag, 5))
        train_p95 = float(np.percentile(train_mag, 95))

        dataset_entry = {
            "name": dataset_name,
            "source": str(dataset_dir.relative_to(ROOT)),
            "mainPlot": str((outdir / f"{dataset_name}_train_val_test_id_distribution.png").relative_to(ROOT)),
            "allPlot": str((outdir / f"{dataset_name}_all_available_splits_distribution.png").relative_to(ROOT)),
            "splits": [],
            "magnitudeHistogram": {"categories": [], "series": []},
        }

        for split in ALL_SPLITS:
            if split not in split_values:
                continue
            mag = split_values[split]["wind_magnitude"]
            overlap = float(np.mean((mag >= train_p05) & (mag <= train_p95)) * 100.0)
            split_summary = {
                "split": split,
                "n": int(len(mag)),
                "magMean": float(np.mean(mag)),
                "magP05": float(np.percentile(mag, 5)),
                "magP50": float(np.percentile(mag, 50)),
                "magP95": float(np.percentile(mag, 95)),
                "trainP05P95OverlapPct": overlap,
            }
            dataset_entry["splits"].append(split_summary)

            for component in COMPONENTS:
                stats = describe(split_values[split][component])
                rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "component": component,
                        "n": int(len(split_values[split][component])),
                        "train_mag_p05_p95_overlap_pct": overlap,
                        **stats,
                    }
                )

        # Keep canvas charts compact by using one common magnitude axis for train/val/test_id.
        main_mags = [split_values[split]["wind_magnitude"] for split in MAIN_SPLITS if split in split_values]
        lo = min(float(np.percentile(v, 1)) for v in main_mags)
        hi = max(float(np.percentile(v, 99)) for v in main_mags)
        bins = np.linspace(lo, hi, 22)
        categories = None
        series = []
        for split in MAIN_SPLITS:
            centers, hist = density_line(split_values[split]["wind_magnitude"], bins)
            categories = centers
            series.append({"name": LABELS[split], "data": hist})
        dataset_entry["magnitudeHistogram"] = {
            "categories": [f"{c:.1f}" for c in categories],
            "series": series,
        }

        plot_dataset(
            dataset_name,
            split_values,
            MAIN_SPLITS,
            outdir / f"{dataset_name}_train_val_test_id_distribution.png",
        )
        plot_dataset(
            dataset_name,
            split_values,
            tuple(split for split in ALL_SPLITS if split in split_values),
            outdir / f"{dataset_name}_all_available_splits_distribution.png",
        )
        canvas_payload["datasets"].append(dataset_entry)

    csv_path = outdir / "split_distribution_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = outdir / "split_distribution_summary.json"
    with json_path.open("w") as fh:
        json.dump(canvas_payload, fh, indent=2)

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
