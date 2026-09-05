#!/usr/bin/env python3
"""Compute frozen seed-, block-bootstrap- and window-level main statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SEEDS = (26, 42, 2024, 2025, 2026)
METHOD_KEYS = {
    "Vanilla GRU": "vanilla_gru",
    "Vanilla LSTM": "vanilla_lstm",
    "PI-GRU": "pigru",
    "PIRNN-AKF": "pirnn_akf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--kalmannet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--splits", default="test_id,test_ood")
    parser.add_argument("--block-length", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--random-seed", type=int, default=260826)
    return parser.parse_args()


def infer_segments(x: np.ndarray) -> list[slice]:
    if len(x) == 0:
        return []
    if len(x) == 1:
        return [slice(0, 1)]
    consecutive = np.all(x[1:, 0, :] == x[:-1, 1, :], axis=1)
    starts = np.r_[0, np.flatnonzero(~consecutive) + 1]
    stops = np.r_[starts[1:], len(x)]
    return [slice(int(a), int(b)) for a, b in zip(starts, stops)]


def valid_block_starts(segments: list[slice], block_length: int) -> np.ndarray:
    parts = [
        np.arange(item.start, item.stop - block_length + 1, dtype=np.int64)
        for item in segments
        if item.stop - item.start >= block_length
    ]
    if not parts:
        raise RuntimeError(
            f"No contiguous segment is at least {block_length} samples"
        )
    return np.concatenate(parts)


def nonoverlap_windows(
    segments: list[slice],
    window_length: int,
) -> list[slice]:
    windows = []
    for item in segments:
        for start in range(item.start, item.stop - window_length + 1, window_length):
            windows.append(slice(start, start + window_length))
    if not windows:
        raise RuntimeError("No complete paired windows available")
    return windows


def load_errors(
    results_dir: Path,
    kalmannet_dir: Path,
    seeds: tuple[int, ...],
    split: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    truths = []
    predictions: dict[str, list[np.ndarray]] = {
        **{name: [] for name in METHOD_KEYS},
        "KalmanNet": [],
    }
    for seed in seeds:
        archive = np.load(
            results_dir / "predictions" / f"seed{seed}_{split}.npz"
        )
        truths.append(np.asarray(archive["wind_true"], dtype=np.float64))
        for name, key in METHOD_KEYS.items():
            predictions[name].append(
                np.asarray(archive[key], dtype=np.float64)
            )
        predictions["KalmanNet"].append(
            np.asarray(
                np.load(kalmannet_dir / f"seed{seed}_{split}_kalmannet.npy"),
                dtype=np.float64,
            )
        )
    truth = truths[0]
    if any(not np.array_equal(truth, item) for item in truths[1:]):
        raise RuntimeError(f"Truth arrays differ across seeds for {split}")
    errors = {}
    for name, values in predictions.items():
        stack = np.stack(values)
        if stack.shape[1:] != truth.shape:
            raise RuntimeError(
                f"{name}/{split} prediction shape {stack.shape} "
                f"does not match truth {truth.shape}"
            )
        errors[name] = stack - truth[None, :, :]
    return truth, errors


def main() -> None:
    args = parse_args()
    seeds = tuple(int(item) for item in args.seeds.split(","))
    if len(seeds) < 5:
        raise ValueError("Frozen revision protocol requires at least five seeds")
    splits = tuple(item.strip() for item in args.splits.split(","))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    seed_rows = []
    ci_rows = []
    wilcoxon_rows = []
    segment_modes = {}
    for split in splits:
        x = np.load(
            args.data_dir / f"X_{split}.npy",
            mmap_mode="r",
        )
        segments = infer_segments(x)
        if not any(
            item.stop - item.start >= args.block_length for item in segments
        ):
            # The archived revision arrays retain chronological save order but
            # were sparsely sampled within source files, so exact one-step
            # overlap cannot recover long 50-Hz episodes. Preserve local
            # dependence in saved-sample order and label this fallback
            # explicitly; do not describe its block length as elapsed time.
            segments = [slice(0, len(x))]
            segment_modes[split] = "ordered_sparse_sample_blocks"
            print(
                f"{split}: exact contiguous episodes are shorter than "
                f"{args.block_length}; using ordered sparse-sample blocks"
            )
        else:
            segment_modes[split] = "exact_window_overlap"
        truth, errors = load_errors(
            args.results_dir, args.kalmannet_dir, seeds, split
        )
        if len(x) != len(truth):
            raise RuntimeError(f"Feature/truth length mismatch for {split}")

        per_sample_mse = {}
        for method, error in errors.items():
            seed_rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
            for seed, value in zip(seeds, seed_rmse):
                seed_rows.append({
                    "split": split,
                    "method": method,
                    "seed": seed,
                    "rmse_3d": float(value),
                })
            per_sample_mse[method] = np.mean(error**2, axis=(0, 2))

        starts = valid_block_starts(segments, args.block_length)
        n = len(truth)
        n_blocks = int(np.ceil(n / args.block_length))
        boot = {method: np.empty(args.bootstrap_replicates)
                for method in errors}
        offsets = np.arange(args.block_length, dtype=np.int64)
        for replicate in range(args.bootstrap_replicates):
            chosen = rng.choice(starts, size=n_blocks, replace=True)
            index = (chosen[:, None] + offsets[None, :]).reshape(-1)[:n]
            for method, values in per_sample_mse.items():
                boot[method][replicate] = np.sqrt(np.mean(values[index]))

        for method, distribution in boot.items():
            ci_rows.append({
                "split": split,
                "method": method,
                "estimate": float(np.sqrt(np.mean(per_sample_mse[method]))),
                "ci95_low": float(np.quantile(distribution, 0.025)),
                "ci95_high": float(np.quantile(distribution, 0.975)),
                "block_length": args.block_length,
                "bootstrap_replicates": args.bootstrap_replicates,
            })
        reference = boot["PIRNN-AKF"]
        for method, distribution in boot.items():
            if method == "PIRNN-AKF":
                continue
            difference = reference - distribution
            ci_rows.append({
                "split": split,
                "method": f"PIRNN-AKF minus {method}",
                "estimate": float(
                    np.sqrt(np.mean(per_sample_mse["PIRNN-AKF"]))
                    - np.sqrt(np.mean(per_sample_mse[method]))
                ),
                "ci95_low": float(np.quantile(difference, 0.025)),
                "ci95_high": float(np.quantile(difference, 0.975)),
                "block_length": args.block_length,
                "bootstrap_replicates": args.bootstrap_replicates,
            })

        windows = nonoverlap_windows(segments, args.block_length)
        window_rmse = {
            method: np.asarray([
                np.sqrt(np.mean(error[:, item, :] ** 2))
                for item in windows
            ])
            for method, error in errors.items()
        }
        reference_windows = window_rmse["PIRNN-AKF"]
        for method, values in window_rmse.items():
            if method == "PIRNN-AKF":
                continue
            delta = reference_windows - values
            test = wilcoxon(
                reference_windows,
                values,
                alternative="two-sided",
                zero_method="pratt",
                method="auto",
            )
            wilcoxon_rows.append({
                "split": split,
                "comparison": f"PIRNN-AKF vs {method}",
                "n_windows": len(windows),
                "median_paired_rmse_difference": float(np.median(delta)),
                "wilcoxon_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
            })

    seed_df = pd.DataFrame(seed_rows)
    seed_summary = (
        seed_df.groupby(["split", "method"], as_index=False)
        .agg(
            rmse_mean=("rmse_3d", "mean"),
            rmse_sd=("rmse_3d", "std"),
            n_seeds=("seed", "count"),
        )
    )
    seed_df.to_csv(args.output_dir / "seed_level_rmse.csv", index=False)
    seed_summary.to_csv(
        args.output_dir / "seed_level_rmse_summary.csv", index=False
    )
    pd.DataFrame(ci_rows).to_csv(
        args.output_dir / "moving_block_bootstrap_ci.csv", index=False
    )
    pd.DataFrame(wilcoxon_rows).to_csv(
        args.output_dir / "paired_window_wilcoxon.csv", index=False
    )
    manifest = {
        "seeds": seeds,
        "splits": splits,
        "block_length": args.block_length,
        "bootstrap_replicates": args.bootstrap_replicates,
        "random_seed": args.random_seed,
        "segment_modes": segment_modes,
        "block_length_unit": "saved samples",
        "block_length_is_elapsed_time": False,
        "test_data_used_for_selection": False,
    }
    (args.output_dir / "statistics_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(seed_summary.to_string(index=False))
    print(f"Wrote revision statistics to {args.output_dir}")


if __name__ == "__main__":
    main()
