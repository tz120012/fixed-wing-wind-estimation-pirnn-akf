#!/usr/bin/env python3
"""Zero-shot evaluation on processed CSVs from a second JSBSim airframe.

The Rascal scaler, checkpoint, and AKF constants are loaded unchanged. Each CSV
is evaluated as an independent temporal session so windows never cross flight
boundaries.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
REMOVED = {38, 39, 40, 41}
KEEP_41 = [i for i in range(45) if i not in REMOVED]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--norm-params",
        type=Path,
        default=ROOT / "data/dataset_revision_41d/norm_params.pkl",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "data/revision_cross_airframe"
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--sampling-rate", type=int, default=50)
    parser.add_argument("--downsample-factor", type=int, default=1)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild condition- and seed-level summaries from the existing raw CSV.",
    )
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    """Accept either a checkpoint file or a per-seed PI-GRU output root."""
    if path.is_file():
        return path
    candidates = list(path.rglob("best_model.pth")) if path.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"No best_model.pth found below {path}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def checkpoint_seed_label(checkpoint: Path) -> str:
    return next(
        (part.removeprefix("seed_") for part in checkpoint.parts if part.startswith("seed_")),
        checkpoint.parent.name,
    )


def write_seed_level_summaries(raw: pd.DataFrame, out_dir: Path) -> None:
    """Pool sessions within each seed, then report five-seed mean and SD."""
    if "condition" not in raw:
        raw["condition"] = raw["source_csv"].map(
            lambda value: Path(value).parent.name
        )
    pooled_rows = []
    rmse_metrics = [
        "rmse",
        "magnitude_rmse",
        "airspeed_closure_rmse",
    ]
    mean_metrics = [
        "mae",
        "direction_mae",
        "airspeed_closure_mae",
        "jitter_mean",
        "nonfinite_failure_rate",
    ]
    for (condition, method, seed), group in raw.groupby(
        ["condition", "method", "seed"], sort=True
    ):
        weights = group["n"].to_numpy(dtype=np.float64)
        total = float(weights.sum())
        row = {
            "condition": condition,
            "method": method,
            "seed": int(seed),
            "n": int(total),
        }
        for metric in rmse_metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            row[metric] = float(np.sqrt(np.sum(weights * values**2) / total))
        for metric in mean_metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            row[metric] = float(np.sum(weights * values) / total)
        row["max_instantaneous_jump"] = float(
            group["max_instantaneous_jump"].max()
        )
        pooled_rows.append(row)
    by_seed = pd.DataFrame(pooled_rows)
    by_seed.to_csv(out_dir / "cross_airframe_metrics_by_seed.csv", index=False)
    metrics = rmse_metrics + mean_metrics + ["max_instantaneous_jump"]
    summary = (
        by_seed.groupby(["condition", "method"], as_index=False)[metrics]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if part)
        for column in summary.columns.to_flat_index()
    ]
    summary.to_csv(
        out_dir / "cross_airframe_metrics_summary.csv", index=False
    )


def normalize_features(X: np.ndarray, scaler) -> np.ndarray:
    flat = X.reshape(-1, X.shape[-1])
    return scaler.transform(flat).reshape(X.shape).astype(np.float32)


def summarize(
    evidence,
    session: str,
    method: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    last_phys: np.ndarray,
) -> dict:
    finite = np.isfinite(prediction).all(axis=1)
    if not finite.any():
        raise RuntimeError(f"{session}/{method} produced no finite predictions")
    truth_f = truth[finite]
    pred_f = prediction[finite]
    row = evidence.vector_metrics(session, method, truth_f, pred_f)
    row.update(evidence.physical_consistency_metrics(
        session, method, pred_f, last_phys[finite], tol=1.0
    ))
    row.update(evidence.temporal_metrics(session, method, pred_f))
    jumps = np.linalg.norm(np.diff(pred_f, axis=0), axis=1)
    row["max_instantaneous_jump"] = (
        float(np.max(jumps)) if jumps.size else float("nan")
    )
    row["nonfinite_failure_rate"] = float(1.0 - finite.mean())
    return row


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "cross_airframe_metrics_raw.csv"
    if args.summarize_only:
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        raw = pd.read_csv(raw_path)
        write_seed_level_summaries(raw, args.out_dir)
        print(f"Rebuilt cross-airframe summaries in {args.out_dir}")
        return
    csv_paths = sorted(args.csv_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No processed CSV files under {args.csv_dir}")

    preprocessing = load_module(ROOT / "src/1_preprocessing_data.py", "cross_pre")
    evidence = load_module(
        ROOT / "src/experiments/paper_evidence_chain_eval.py", "cross_evidence"
    )
    with args.norm_params.open("rb") as stream:
        norm = pickle.load(stream)
    scaler_X, scaler_y = norm["scaler_X"], norm["scaler_y"]
    if int(scaler_X.n_features_in_) != 41:
        raise ValueError("Cross-airframe protocol requires the frozen 41-D scaler")
    with args.config.open() as stream:
        config = yaml.safe_load(stream)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    prediction_records = []

    resolved_checkpoints = [resolve_checkpoint(path) for path in args.checkpoints]
    for checkpoint in resolved_checkpoints:
        model = evidence.load_pigru_model(checkpoint, scaler_X, scaler_y, device)
        seed_label = checkpoint_seed_label(checkpoint)
        for csv_path in csv_paths:
            X45, y, _, _ = preprocessing.build_features_labels_from_csv(
                str(csv_path),
                seq_len=100,
                weight_config={"enabled": False},
                sampling_rate=args.sampling_rate,
                downsample_factor=args.downsample_factor,
            )
            if len(X45) == 0:
                continue
            X = normalize_features(X45[..., KEEP_41], scaler_X)
            truth = y[:, :3].astype(np.float64)
            last_phys = X45[:, -1, KEEP_41].astype(np.float64)
            out = evidence.predict_pigru(model, X, args.batch_size, device)
            pi = evidence.denorm_wind(out["wind"], scaler_y)
            fused, _ = evidence.run_fast_pirnn_akf(
                config, out, X, scaler_X, scaler_y, continuous=True
            )
            session = csv_path.stem
            condition = csv_path.parent.name
            for method, pred in (("PI-GRU", pi), ("PIRNN-AKF", fused)):
                row = summarize(
                    evidence, session, method, truth, pred, last_phys
                )
                row["seed"] = seed_label
                row["condition"] = condition
                row["checkpoint"] = str(checkpoint.resolve())
                row["source_csv"] = str(csv_path.resolve())
                rows.append(row)
            prediction_records.append({
                "seed": seed_label,
                "condition": condition,
                "session": session,
                "n": int(len(truth)),
                "truth": truth,
                "pi_gru": pi,
                "pirnn_akf": fused,
            })

    if not rows:
        raise RuntimeError("All cross-airframe CSVs failed preprocessing")
    raw = pd.DataFrame(rows)
    raw.to_csv(args.out_dir / "cross_airframe_metrics_raw.csv", index=False)
    write_seed_level_summaries(raw, args.out_dir)
    np.savez_compressed(
        args.out_dir / "cross_airframe_predictions.npz",
        records=np.asarray(prediction_records, dtype=object),
    )
    manifest = {
        "airframe": "Malolo",
        "transfer": "zero-shot",
        "normalization": str(args.norm_params.resolve()),
        "checkpoints": [str(path.resolve()) for path in resolved_checkpoints],
        "csv_files": [str(path.resolve()) for path in csv_paths],
        "retuning": False,
        "rows": len(rows),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(raw.to_string(index=False))
    print(f"Wrote cross-airframe evidence to {args.out_dir}")


if __name__ == "__main__":
    main()
