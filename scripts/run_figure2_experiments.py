#!/usr/bin/env python3
"""Run all Figure 2 experiments and generate the RMSE bar plot.

Figure 2 compares EKF, Vanilla GRU, parameter-matched Vanilla LSTM, PI-GRU, and PIRNN-AKF on
Test-ID/Test-OOD. This script is an orchestration layer: it
creates seed-specific configs, trains missing neural checkpoints, evaluates
all methods with one metric implementation, and writes the raw data needed for
error bars.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PX4_EKF_DIR = SRC_DIR / "px4_ekf2"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "dataset_new_processed"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"

METHOD_ORDER = [
    "EKF",
    "Vanilla GRU",
    "Vanilla LSTM",
    "PI-GRU",
    "PIRNN-AKF",
    "KalmanNet",
]
METHOD_LABELS = {
    "EKF": "EKF",
    "Vanilla GRU": "Vanilla GRU",
    "Vanilla LSTM": "Vanilla LSTM",
    "PI-GRU": "PI-GRU",
    "PIRNN-AKF": "PI-GRU + AKF",
    "KalmanNet": "KalmanNet",
}
SPLIT_ORDER = ["test_id", "test_ood"]
SPLIT_LABELS = {"test_id": "Test-ID", "test_ood": "Test-OOD"}


def load_module(module_path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_csv_ints(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("seeds must contain at least one integer")
    return values


def parse_splits(raw: str) -> List[str]:
    splits = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in splits if item not in SPLIT_ORDER]
    if invalid:
        raise ValueError(f"Unsupported split(s): {invalid}. Allowed: {SPLIT_ORDER}")
    return splits


def safe_lambda_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / str(path).lstrip("../")).resolve()


def latest_checkpoint(base_dir: Path, prefix: str = "", checkpoint_name: str = "best_model.pth") -> Path | None:
    if not base_dir.exists():
        return None
    candidates = []
    for path in base_dir.rglob(checkpoint_name):
        if prefix and not any(parent.name.startswith(prefix) for parent in path.parents):
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def copy_norm_params(data_dir: Path, model_root: Path) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    src = data_dir / "norm_params.pkl"
    if not src.exists():
        raise FileNotFoundError(f"Missing normalization file: {src}")
    shutil.copy2(src, model_root / "norm_params.pkl")


def write_seed_config(
    base_config: Path,
    output_path: Path,
    data_dir: Path,
    model_root: Path,
    seed: int,
    lambda_physics: float,
    run_tag: str,
    rnn_type: str = "gru",
    hidden_size: int | None = None,
) -> None:
    with open(base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("experiment", {})
    cfg.setdefault("data", {})
    cfg.setdefault("training", {})
    cfg.setdefault("logging", {})
    cfg.setdefault("model", {})

    with open(data_dir / "norm_params.pkl", "rb") as f:
        metadata = pickle.load(f)

    cfg["experiment"]["mode"] = "single"
    cfg["experiment"]["lambda_physics_override"] = float(lambda_physics)
    cfg["experiment"]["processed_dir"] = str(data_dir)
    cfg["experiment"]["output_base_dir"] = str(model_root)
    cfg["experiment"]["run_tag"] = run_tag

    cfg["data"]["processed_dir"] = str(data_dir)
    cfg["model"]["input_size"] = int(metadata["input_size"])
    cfg["model"]["rnn_type"] = str(rnn_type)
    if hidden_size is not None:
        cfg["model"]["hidden_size"] = int(hidden_size)
    cfg["training"]["seed"] = int(seed)
    cfg["training"]["num_epochs"] = 150
    cfg["training"]["early_stopping_patience"] = 30
    cfg["training"]["lambda_physics"] = float(lambda_physics)
    cfg["training"]["model_save_path"] = str(model_root)
    cfg["logging"]["save_dir"] = str(model_root / "logs")
    cfg["logging"]["tensorboard_dir"] = str(model_root / "tensorboard")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def run_command(
    cmd: List[str],
    log_path: Path,
    cwd: Path = PROJECT_ROOT,
    env_overrides: Dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {' '.join(cmd)}")
    print(f"    log: {log_path}")
    with open(log_path, "w", encoding="utf-8") as log_file:
        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}. See {log_path}")


def train_for_seed(
    seed: int,
    args: argparse.Namespace,
    config_dir: Path,
    model_dir: Path,
    log_dir: Path,
) -> Tuple[Path, Path, Path]:
    data_dir = resolve_path(args.data_dir)
    base_config = resolve_path(args.config)
    lambda_tag = safe_lambda_tag(args.lambda_physics)
    child_env = {"CUDA_VISIBLE_DEVICES": ""} if args.cpu else None

    vanilla_root = model_dir / f"seed_{seed}" / "vanilla"
    lstm_root = model_dir / f"seed_{seed}" / "lstm"
    pigru_root = model_dir / f"seed_{seed}" / "pigru"
    copy_norm_params(data_dir, vanilla_root)
    copy_norm_params(data_dir, lstm_root)
    copy_norm_params(data_dir, pigru_root)

    vanilla_cfg = config_dir / f"fig2_vanilla_seed{seed}_lambda{lambda_tag}.yaml"
    lstm_cfg = config_dir / f"fig2_lstm_seed{seed}_lambda{lambda_tag}.yaml"
    pigru_cfg = config_dir / f"fig2_pigru_seed{seed}_lambda{lambda_tag}.yaml"
    write_seed_config(
        base_config,
        vanilla_cfg,
        data_dir,
        vanilla_root,
        seed,
        args.lambda_physics,
        f"fig2_vanilla_seed{seed}_lambda{lambda_tag}",
    )
    write_seed_config(
        base_config,
        lstm_cfg,
        data_dir,
        lstm_root,
        seed,
        args.lambda_physics,
        f"fig2_lstm_seed{seed}_lambda{lambda_tag}",
        rnn_type="lstm",
        hidden_size=110,
    )
    write_seed_config(
        base_config,
        pigru_cfg,
        data_dir,
        pigru_root,
        seed,
        args.lambda_physics,
        f"fig2_pigru_seed{seed}_lambda{lambda_tag}",
    )

    vanilla_ckpt = latest_checkpoint(vanilla_root, prefix="vanilla_gru_")
    if vanilla_ckpt is None or args.force_train:
        run_command(
            [sys.executable, "src/3b_train_vanilla_gru.py", "--config_path", str(vanilla_cfg)],
            log_dir / f"train_vanilla_seed{seed}.log",
            env_overrides=child_env,
        )
        vanilla_ckpt = latest_checkpoint(vanilla_root, prefix="vanilla_gru_")
    else:
        print(f"  Vanilla GRU seed={seed}: reuse {vanilla_ckpt}")

    lstm_ckpt = latest_checkpoint(lstm_root, prefix="vanilla_gru_")
    if lstm_ckpt is None or args.force_train:
        run_command(
            [sys.executable, "src/3b_train_vanilla_gru.py", "--config_path", str(lstm_cfg)],
            log_dir / f"train_lstm_seed{seed}.log",
            env_overrides=child_env,
        )
        lstm_ckpt = latest_checkpoint(lstm_root, prefix="vanilla_gru_")
    else:
        print(f"  Vanilla LSTM seed={seed}: reuse {lstm_ckpt}")

    pigru_ckpt = latest_checkpoint(pigru_root, prefix="train_")
    if pigru_ckpt is None or args.force_train:
        run_command(
            [
                sys.executable,
                "src/3_train_pigru.py",
                "--config_path",
                str(pigru_cfg),
                "--mode",
                "single",
                "--lambda_physics",
                f"{args.lambda_physics:g}",
                "--processed_dir_override",
                str(data_dir),
                "--model_save_path_override",
                str(pigru_root),
                "--run_tag",
                f"fig2_pigru_seed{seed}_lambda{lambda_tag}",
            ],
            log_dir / f"train_pigru_seed{seed}.log",
            env_overrides=child_env,
        )
        pigru_ckpt = latest_checkpoint(pigru_root, prefix="train_")
    else:
        print(f"  PI-GRU seed={seed}: reuse {pigru_ckpt}")

    if vanilla_ckpt is None:
        raise FileNotFoundError(f"Vanilla GRU checkpoint not found under {vanilla_root}")
    if lstm_ckpt is None:
        raise FileNotFoundError(f"Vanilla LSTM checkpoint not found under {lstm_root}")
    if pigru_ckpt is None:
        raise FileNotFoundError(f"PI-GRU checkpoint not found under {pigru_root}")
    return vanilla_ckpt, lstm_ckpt, pigru_ckpt


def load_norm_params(data_dir: Path):
    with open(data_dir / "norm_params.pkl", "rb") as f:
        meta = pickle.load(f)
    return meta["scaler_X"], meta["scaler_y"]


def denorm_y(y_norm: np.ndarray, scaler_y) -> np.ndarray:
    return scaler_y.inverse_transform(y_norm)


def infer_contiguous_slices(x: np.ndarray) -> List[slice]:
    """Recover independent sliding-window sessions from overlap continuity."""
    if len(x) == 0:
        return []
    if len(x) == 1:
        return [slice(0, 1)]
    consecutive = np.all(x[1:, 0, :] == x[:-1, 1, :], axis=1)
    starts = np.r_[0, np.flatnonzero(~consecutive) + 1]
    stops = np.r_[starts[1:], len(x)]
    return [
        slice(int(start), int(stop))
        for start, stop in zip(starts, stops)
    ]


def rmse_metrics(
    split: str,
    method: str,
    seed: str,
    wind_true: np.ndarray,
    wind_pred: np.ndarray,
    segments: List[slice],
    last_phys: np.ndarray,
) -> Dict[str, float | str | int]:
    err = wind_pred - wind_true
    true_mag = np.linalg.norm(wind_true, axis=1)
    pred_mag = np.linalg.norm(wind_pred, axis=1)
    horizontal = np.linalg.norm(wind_true[:, :2], axis=1) >= 0.5
    true_dir = np.degrees(np.arctan2(wind_true[:, 1], wind_true[:, 0]))
    pred_dir = np.degrees(np.arctan2(wind_pred[:, 1], wind_pred[:, 0]))
    dir_error = np.abs((pred_dir - true_dir + 180.0) % 360.0 - 180.0)
    first_parts = [
        np.diff(wind_pred[item, :2], axis=0)
        for item in segments
        if item.stop - item.start >= 2
    ]
    second_parts = [
        np.diff(wind_pred[item, :2], n=2, axis=0)
        for item in segments
        if item.stop - item.start >= 3
    ]
    first_delta = (
        np.concatenate(first_parts)
        if first_parts
        else np.empty((0, 2))
    )
    second_delta = (
        np.concatenate(second_parts)
        if second_parts
        else np.empty((0, 2))
    )
    closure_residual = (
        np.linalg.norm(last_phys[:, 0:3] - wind_pred, axis=1)
        - last_phys[:, 19]
    )
    return {
        "split": split,
        "method": method,
        "seed": seed,
        "n_samples": int(len(wind_true)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "rmse_n": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "rmse_e": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "rmse_d": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "magnitude_rmse": float(np.sqrt(np.mean((pred_mag - true_mag) ** 2))),
        "direction_mae_deg": (
            float(np.mean(dir_error[horizontal]))
            if horizontal.any()
            else float("nan")
        ),
        "jitter": (
            float(np.mean(np.linalg.norm(second_delta, axis=1)))
            if len(second_delta)
            else float("nan")
        ),
        "max_jump": (
            float(np.max(np.linalg.norm(first_delta, axis=1)))
            if len(first_delta)
            else float("nan")
        ),
        "airspeed_closure_rmse": float(
            np.sqrt(np.mean(closure_residual**2))
        ),
    }


def predict_px4_ekf2(X: np.ndarray, scaler_X, sampling_rate: float) -> np.ndarray:
    sys.path.insert(0, str(PX4_EKF_DIR))
    module = load_module(PX4_EKF_DIR / "eval_px4_ekf2.py", "fig2_px4_ekf2_eval")
    estimator = module.PX4EKF2WindEstimator(dt=1.0 / sampling_rate)
    pred = np.zeros((len(X), 3), dtype=np.float64)
    for i in range(len(X)):
        estimator.reset()
        Xd = scaler_X.inverse_transform(X[i])
        last = np.zeros(3, dtype=np.float64)
        for t in range(Xd.shape[0]):
            last = estimator.step(Xd[t, 0:3], float(Xd[t, 19]))
        pred[i] = last
        if (i + 1) % 5000 == 0:
            print(f"    EKF progress: {i + 1:,}/{len(X):,}")
    return pred


def evaluate_all(
    args: argparse.Namespace,
    out_dir: Path,
    checkpoints: Dict[int, Tuple[Path, Path, Path]],
) -> pd.DataFrame:
    evidence = load_module(SRC_DIR / "experiments" / "paper_evidence_chain_eval.py", "fig2_evidence_eval")

    data_dir = resolve_path(args.data_dir)
    config_path = resolve_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.setdefault("data", {})["sampling_rate"] = float(config.get("data", {}).get("sampling_rate", 50))

    scaler_X, scaler_y = load_norm_params(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    splits = parse_splits(args.splits)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float | str | int]] = []
    split_cache = {}
    segments_by_split = {}
    for split in splits:
        X = np.load(data_dir / f"X_{split}.npy")
        y_norm = np.load(data_dir / f"y_{split}.npy")
        if args.max_samples > 0:
            X = X[: args.max_samples]
            y_norm = y_norm[: args.max_samples]
        split_cache[split] = (
            X,
            y_norm,
            denorm_y(y_norm, scaler_y)[:, :3],
            evidence.denorm_last_step(X, scaler_X),
        )
        segments_by_split[split] = infer_contiguous_slices(X)

    if not args.skip_ekf:
        ekf_cache = out_dir / "ekf_predictions.npz"
        ekf_data = {}
        if ekf_cache.exists() and not args.force_eval:
            cached = np.load(ekf_cache)
            ekf_data = {k: cached[k] for k in cached.files}
        for split in splits:
            X, _, wind_true, last_phys = split_cache[split]
            if split not in ekf_data:
                print(f"\n[EKF] evaluating {split}")
                ekf_data[split] = predict_px4_ekf2(X, scaler_X, config["data"]["sampling_rate"])
            rows.append(rmse_metrics(
                split, "EKF", "deterministic", wind_true, ekf_data[split],
                segments_by_split[split], last_phys,
            ))
        if args.save_predictions:
            np.savez_compressed(ekf_cache, **ekf_data)

    if not args.skip_kalmannet:
        kalmannet_dir = resolve_path(args.kalmannet_dir)
        for seed in parse_csv_ints(args.seeds):
            for split in splits:
                path = kalmannet_dir / f"seed{seed}_{split}_kalmannet.npy"
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing five-seed KalmanNet prediction: {path}"
                    )
                prediction = np.load(path)
                _, _, wind_true, last_phys = split_cache[split]
                if args.max_samples > 0:
                    prediction = prediction[: args.max_samples]
                if len(prediction) != len(wind_true):
                    raise ValueError(
                        f"KalmanNet {seed}/{split} length {len(prediction)} "
                        f"does not match truth length {len(wind_true)}"
                    )
                rows.append(rmse_metrics(
                    split, "KalmanNet", str(seed), wind_true, prediction,
                    segments_by_split[split], last_phys,
                ))

    for seed, (vanilla_ckpt, lstm_ckpt, pigru_ckpt) in checkpoints.items():
        print(f"\n[Seed {seed}] loading checkpoints")
        vanilla_model = evidence.load_vanilla_model(vanilla_ckpt, device)
        lstm_model = evidence.load_vanilla_model(lstm_ckpt, device)
        pigru_model = evidence.load_pigru_model(pigru_ckpt, scaler_X, scaler_y, device)

        for split in splits:
            print(f"[Seed {seed}] evaluating {split}")
            X, _, wind_true, last_phys = split_cache[split]

            vanilla_norm = evidence.predict_vanilla(vanilla_model, X, args.batch_size, device)
            vanilla_pred = evidence.denorm_wind(vanilla_norm, scaler_y)
            rows.append(rmse_metrics(
                split, "Vanilla GRU", str(seed), wind_true, vanilla_pred,
                segments_by_split[split], last_phys,
            ))

            lstm_norm = evidence.predict_vanilla(lstm_model, X, args.batch_size, device)
            lstm_pred = evidence.denorm_wind(lstm_norm, scaler_y)
            rows.append(rmse_metrics(
                split, "Vanilla LSTM", str(seed), wind_true, lstm_pred,
                segments_by_split[split], last_phys,
            ))

            pigru_out = evidence.predict_pigru(pigru_model, X, args.batch_size, device)
            pigru_pred = evidence.denorm_wind(pigru_out["wind"], scaler_y)
            rows.append(rmse_metrics(
                split, "PI-GRU", str(seed), wind_true, pigru_pred,
                segments_by_split[split], last_phys,
            ))

            pirnn_akf_pred, _ = evidence.run_fast_pirnn_akf(
                config, pigru_out, X, scaler_X, scaler_y, continuous=True
            )
            rows.append(rmse_metrics(
                split, "PIRNN-AKF", str(seed), wind_true, pirnn_akf_pred,
                segments_by_split[split], last_phys,
            ))

            if args.save_predictions:
                np.savez_compressed(
                    pred_dir / f"seed{seed}_{split}.npz",
                    wind_true=wind_true,
                    vanilla_gru=vanilla_pred,
                    vanilla_lstm=lstm_pred,
                    pigru=pigru_pred,
                    pirnn_akf=pirnn_akf_pred,
                )

    return pd.DataFrame(rows)


def summarize_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw_df.groupby(["method", "split"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            rmse_count=("rmse", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            direction_mae_mean=("direction_mae_deg", "mean"),
            direction_mae_std=("direction_mae_deg", "std"),
            magnitude_rmse_mean=("magnitude_rmse", "mean"),
            magnitude_rmse_std=("magnitude_rmse", "std"),
            rmse_n_mean=("rmse_n", "mean"),
            rmse_n_std=("rmse_n", "std"),
            rmse_e_mean=("rmse_e", "mean"),
            rmse_e_std=("rmse_e", "std"),
            rmse_d_mean=("rmse_d", "mean"),
            rmse_d_std=("rmse_d", "std"),
            jitter_mean=("jitter", "mean"),
            jitter_std=("jitter", "std"),
            max_jump_mean=("max_jump", "mean"),
            max_jump_std=("max_jump", "std"),
            airspeed_closure_rmse_mean=("airspeed_closure_rmse", "mean"),
            airspeed_closure_rmse_std=("airspeed_closure_rmse", "std"),
        )
        .sort_values(
            by=["method", "split"],
            key=lambda s: s.map({name: i for i, name in enumerate(METHOD_ORDER + SPLIT_ORDER)}).fillna(999),
        )
    )
    summary["rmse_std"] = summary["rmse_std"].fillna(0.0)
    summary["mae_std"] = summary["mae_std"].fillna(0.0)
    summary["direction_mae_std"] = summary["direction_mae_std"].fillna(0.0)
    summary["magnitude_rmse_std"] = summary["magnitude_rmse_std"].fillna(0.0)
    for column in (
        "rmse_n_std", "rmse_e_std", "rmse_d_std",
        "jitter_std", "max_jump_std",
        "airspeed_closure_rmse_std",
    ):
        summary[column] = summary[column].fillna(0.0)
    return summary


def plot_figure2(summary: pd.DataFrame, output_base: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(METHOD_ORDER))
    width = 0.40
    # Academic colors (e.g. from Nature/Science palettes)
    colors = {"test_id": "#1F77B4", "test_ood": "#E64B35"}

    for idx, split in enumerate(SPLIT_ORDER):
        sub = summary[summary["split"] == split].set_index("method")
        means = [float(sub.loc[m, "rmse_mean"]) if m in sub.index else np.nan for m in METHOD_ORDER]
        stds = [float(sub.loc[m, "rmse_std"]) if m in sub.index else 0.0 for m in METHOD_ORDER]
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=SPLIT_LABELS[split],
            color=colors[split],
            alpha=0.9,
            edgecolor="black",
            linewidth=1.0,
        )

    ax.set_ylabel("Wind Velocity RMSE (m/s)")
    ax.set_xlabel("") # Remove redundant x-label
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=0, ha="center")
    ax.grid(axis="y")
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    
    ax.legend(fontsize=9, frameon=True, edgecolor='black', fancybox=False)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=600)
    fig.savefig(output_base.with_suffix(".svg"))
    fig.savefig(output_base.with_suffix(".pdf"), dpi=600)
    plt.close(fig)


def load_existing_checkpoints(
    args: argparse.Namespace, model_dir: Path
) -> Dict[int, Tuple[Path, Path, Path]]:
    checkpoints: Dict[int, Tuple[Path, Path, Path]] = {}
    for seed in parse_csv_ints(args.seeds):
        vanilla_root = model_dir / f"seed_{seed}" / "vanilla"
        lstm_root = model_dir / f"seed_{seed}" / "lstm"
        pigru_root = model_dir / f"seed_{seed}" / "pigru"
        vanilla_ckpt = latest_checkpoint(vanilla_root, prefix="vanilla_gru_")
        lstm_ckpt = latest_checkpoint(lstm_root, prefix="vanilla_gru_")
        pigru_ckpt = latest_checkpoint(pigru_root, prefix="train_")
        if vanilla_ckpt is None or lstm_ckpt is None or pigru_ckpt is None:
            missing = []
            if vanilla_ckpt is None:
                missing.append("Vanilla GRU")
            if lstm_ckpt is None:
                missing.append("Vanilla LSTM")
            if pigru_ckpt is None:
                missing.append("PI-GRU")
            raise FileNotFoundError(f"Seed {seed} missing checkpoint(s): {', '.join(missing)}")
        checkpoints[seed] = (vanilla_ckpt, lstm_ckpt, pigru_ckpt)
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Figure 2 Test-ID/Test-OOD RMSE experiments.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Base YAML config.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Processed dataset directory.")
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to data/figure2/<timestamp>.")
    parser.add_argument("--seeds", default="26,42,2026", help="Comma-separated random seeds for neural models.")
    parser.add_argument("--lambda-physics", type=float, default=0.10, help="PI-GRU physics loss weight for Figure 2.")
    parser.add_argument("--splits", default="test_id,test_ood", help="Comma-separated splits to evaluate.")
    parser.add_argument("--stage", choices=["all", "train", "eval", "plot"], default="all")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional per-split cap for smoke tests.")
    parser.add_argument("--force-train", action="store_true", help="Retrain even if checkpoints already exist.")
    parser.add_argument("--force-eval", action="store_true", help="Recompute cached deterministic predictions.")
    parser.add_argument("--skip-ekf", action="store_true", help="Skip EKF baseline.")
    parser.add_argument(
        "--kalmannet-dir",
        default="data/revision_kalmannet_41d",
        help="Directory containing five-seed KalmanNet predictions.",
    )
    parser.add_argument(
        "--skip-kalmannet", action="store_true", help="Skip KalmanNet baseline."
    )
    parser.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.set_defaults(save_predictions=True)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = resolve_path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "figure2" / f"fig2_{timestamp}"
    config_dir = out_dir / "configs"
    model_dir = out_dir / "models"
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": timestamp,
        "config": str(resolve_path(args.config)),
        "data_dir": str(resolve_path(args.data_dir)),
        "output_dir": str(out_dir),
        "seeds": parse_csv_ints(args.seeds),
        "lambda_physics": args.lambda_physics,
        "splits": parse_splits(args.splits),
        "stage": args.stage,
        "max_samples": args.max_samples,
        "methods": METHOD_ORDER,
        "kalmannet_dir": str(resolve_path(args.kalmannet_dir)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    checkpoints: Dict[int, Tuple[Path, Path, Path]] = {}
    if args.stage in {"all", "train"}:
        for seed in parse_csv_ints(args.seeds):
            print(f"\n=== Training seed {seed} ===")
            checkpoints[seed] = train_for_seed(seed, args, config_dir, model_dir, log_dir)
        if args.stage == "train":
            print("\nTraining stage complete.")
            print(f"Output directory: {out_dir}")
            print(f"Model directory:  {model_dir}")
            return
    elif args.stage == "eval":
        checkpoints = load_existing_checkpoints(args, model_dir)

    raw_csv = out_dir / "figure2_metrics_raw.csv"
    summary_csv = out_dir / "figure2_metrics_summary.csv"

    if args.stage in {"all", "eval"}:
        if not checkpoints:
            checkpoints = load_existing_checkpoints(args, model_dir)
        raw_df = evaluate_all(args, out_dir, checkpoints)
        raw_df["lambda_physics"] = float(args.lambda_physics)
        raw_df.to_csv(raw_csv, index=False)
        summary_df = summarize_metrics(raw_df)
        summary_df.to_csv(summary_csv, index=False)
    else:
        if not raw_csv.exists():
            raise FileNotFoundError(f"Missing raw metrics for plotting: {raw_csv}")
        raw_df = pd.read_csv(raw_csv)
        summary_df = summarize_metrics(raw_df)
        summary_df.to_csv(summary_csv, index=False)

    if args.stage in {"all", "eval", "plot"}:
        plot_figure2(summary_df, out_dir / "figure2_rmse_bar")

    print("\nFigure 2 pipeline complete.")
    print(f"Output directory: {out_dir}")
    print(f"Raw metrics:      {raw_csv}")
    print(f"Summary metrics:  {summary_csv}")
    print(f"Figure:           {out_dir / 'figure2_rmse_bar.png'}")


if __name__ == "__main__":
    main()
