#!/usr/bin/env python3
"""Evaluate PI-GRU checkpoints on turn-aware split masks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path_value).resolve()


def load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_norm_params(scaler_X, scaler_y) -> Dict[str, np.ndarray]:
    return {
        "X_mean": scaler_X.mean_,
        "X_scale": scaler_X.scale_,
        "y_mean": scaler_y.mean_,
        "y_scale": scaler_y.scale_,
    }


def load_pigru_model(checkpoint_path: Path, scaler_X, scaler_y, device: torch.device):
    module = load_module(PROJECT_ROOT / "src" / "2_pigru_module.py", "pigru_model_turnaware_eval")
    PIGRU = module.PIGRU
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {}).get("model", {})
    yaw_invariant = bool(cfg.get("yaw_invariant", False))
    model = PIGRU(
        input_size=cfg.get("input_size", 20),
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        dropout=0.0,
        enable_wind_head=True,
        enable_noise_heads=cfg.get("enable_noise_heads", True),
        enable_angles_head=cfg.get("enable_angles_head", True),
        enable_confidence_head=cfg.get("enable_confidence_head", True),
        yaw_invariant=yaw_invariant,
        norm_params=model_norm_params(scaler_X, scaler_y) if yaw_invariant else None,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_wind(model, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb_np = np.array(X[start:start + batch_size], copy=True)
            xb = torch.from_numpy(xb_np).float().to(device)
            out = model(xb, return_dict=True)
            outs.append(out["wind_estimate"].detach().cpu().numpy())
    return np.vstack(outs).astype(np.float64)


def denorm_wind(wind_norm: np.ndarray, scaler_y) -> np.ndarray:
    return wind_norm * scaler_y.scale_[:3] + scaler_y.mean_[:3]


def denorm_y(y_norm: np.ndarray, scaler_y) -> np.ndarray:
    return scaler_y.inverse_transform(y_norm)


def denorm_last_step(X_norm: np.ndarray, scaler_X) -> np.ndarray:
    last = np.asarray(X_norm[:, -1, :], dtype=np.float64)
    return last * scaler_X.scale_ + scaler_X.mean_


def direction_error_deg(wind_true: np.ndarray, wind_pred: np.ndarray) -> np.ndarray:
    true_dir = np.degrees(np.arctan2(wind_true[:, 1], wind_true[:, 0]))
    pred_dir = np.degrees(np.arctan2(wind_pred[:, 1], wind_pred[:, 0]))
    diff = (pred_dir - true_dir + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def temporal_metrics(wind_pred: np.ndarray) -> Dict[str, float]:
    if len(wind_pred) < 3:
        return {"tv_mean": np.nan, "jitter_mean": np.nan}
    d1 = np.diff(wind_pred, axis=0)
    d2 = np.diff(d1, axis=0)
    return {
        "tv_mean": float(np.mean(np.linalg.norm(d1, axis=1))),
        "jitter_mean": float(np.mean(np.linalg.norm(d2, axis=1))),
    }


def airspeed_closure_metrics(X_phys: np.ndarray, wind_pred: np.ndarray) -> Dict[str, float]:
    vg = X_phys[:, 0:3]
    tas = X_phys[:, 19]
    pred_tas = np.linalg.norm(vg - wind_pred, axis=1)
    residual = pred_tas - tas
    return {
        "airspeed_closure_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "airspeed_closure_mae": float(np.mean(np.abs(residual))),
        "airspeed_closure_p95": float(np.percentile(np.abs(residual), 95)),
    }


def vector_metrics(
    split: str,
    bucket: str,
    method: str,
    wind_true: np.ndarray,
    wind_pred: np.ndarray,
    X_phys: np.ndarray,
) -> Dict[str, float]:
    err = wind_pred - wind_true
    mag_true = np.linalg.norm(wind_true, axis=1)
    mag_pred = np.linalg.norm(wind_pred, axis=1)
    dir_err = direction_error_deg(wind_true, wind_pred)
    row = {
        "split": split,
        "bucket": bucket,
        "method": method,
        "n": int(len(wind_true)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "vector_error_mean": float(np.mean(np.linalg.norm(err, axis=1))),
        "north_rmse": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "east_rmse": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "down_rmse": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "magnitude_rmse": float(np.sqrt(np.mean((mag_pred - mag_true) ** 2))),
        "magnitude_mae": float(np.mean(np.abs(mag_pred - mag_true))),
        "direction_mae": float(np.mean(dir_err)),
        "direction_p95": float(np.percentile(dir_err, 95)),
    }
    row.update(airspeed_closure_metrics(X_phys, wind_pred))
    row.update(temporal_metrics(wind_pred))
    return row


def bucket_masks(turn_class: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "full": np.ones_like(turn_class, dtype=bool),
        "steady": turn_class == 0,
        "mild_turn": turn_class == 1,
        "turn": turn_class == 2,
        "aggressive_turn": turn_class == 3,
        "maneuver_all": turn_class >= 1,
        "turn_plus_aggressive": turn_class >= 2,
    }


def parse_splits(raw: str) -> Iterable[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/dataset_lag_aligned_filtered_turnaware")
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--turnaware-model", required=True)
    parser.add_argument("--splits", default="test_id,test_ood")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output-dir", default="data/paper_turnaware_experiments/evaluation")
    args = parser.parse_args()

    data_dir = resolve_path(args.data_dir)
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "norm_params.pkl", "rb") as f:
        meta = pickle.load(f)
    scaler_X = meta["scaler_X"]
    scaler_y = meta["scaler_y"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        "filtered_main": load_pigru_model(resolve_path(args.baseline_model), scaler_X, scaler_y, device),
        "turnaware": load_pigru_model(resolve_path(args.turnaware_model), scaler_X, scaler_y, device),
    }

    rows = []
    manifest = {
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "baseline_model": str(resolve_path(args.baseline_model)),
        "turnaware_model": str(resolve_path(args.turnaware_model)),
        "splits": list(parse_splits(args.splits)),
        "batch_size": args.batch_size,
        "device": str(device),
    }

    for split in parse_splits(args.splits):
        X = np.load(data_dir / f"X_{split}.npy", mmap_mode="r")
        y = np.load(data_dir / f"y_{split}.npy", mmap_mode="r")
        turn_class = np.load(data_dir / "turn_masks" / f"turn_class_{split}.npy")
        y_true = denorm_y(np.asarray(y), scaler_y)[:, :3]
        X_phys = denorm_last_step(X, scaler_X)

        preds = {}
        for name, model in models.items():
            print(f"[{split}] predicting {name} ({len(X)} samples)")
            pred_norm = predict_wind(model, X, args.batch_size, device)
            preds[name] = denorm_wind(pred_norm, scaler_y)

        for bucket, mask in bucket_masks(turn_class).items():
            if not np.any(mask):
                continue
            for name, pred in preds.items():
                rows.append(vector_metrics(split, bucket, name, y_true[mask], pred[mask], X_phys[mask]))

        np.savez_compressed(
            out_dir / f"predictions_{split}.npz",
            wind_true=y_true,
            turn_class=turn_class,
            filtered_main=preds["filtered_main"],
            turnaware=preds["turnaware"],
        )

    df = pd.DataFrame(rows)
    metrics_csv = out_dir / "turnaware_split_metrics.csv"
    df.to_csv(metrics_csv, index=False)

    pivot = df.pivot_table(
        index=["split", "bucket"],
        columns="method",
        values=["rmse", "magnitude_rmse", "direction_mae", "airspeed_closure_rmse", "jitter_mean"],
    )
    pivot_csv = out_dir / "turnaware_split_metrics_pivot.csv"
    pivot.to_csv(pivot_csv)

    manifest["metrics_csv"] = str(metrics_csv)
    manifest["pivot_csv"] = str(pivot_csv)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("=" * 72)
    print(f"saved: {metrics_csv}")
    print(f"saved: {pivot_csv}")
    print(f"saved: {out_dir / 'manifest.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
