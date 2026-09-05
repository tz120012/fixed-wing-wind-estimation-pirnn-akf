from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper_plot_style import apply_style, save_figure
from src.experiments import paper_evidence_chain_eval as evidence
from plot_figure6_time_series import compute_ema


@dataclass
class ExperimentContext:
    X: np.ndarray
    wind_true: np.ndarray
    scaler_X: object
    scaler_y: object
    model: object
    config: dict
    device: torch.device


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_strings(text: str) -> list[str]:
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


def load_context(model_dir: str, data_dir: str = "data/dataset_new_processed") -> ExperimentContext:
    data_path = PROJECT_ROOT / data_dir
    model_path = PROJECT_ROOT / model_dir
    scaler_X, scaler_y = evidence.load_norm_params(data_path)
    X = np.load(data_path / "X_test_ood.npy")
    y = np.load(data_path / "y_test_ood.npy")
    wind_true = evidence.denorm_y(y, scaler_y)[:, :3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = model_path / "best_composite_model.pth"
    if not ckpt.exists():
        ckpt = model_path / "best_model.pth"
    model = evidence.load_pigru_model(ckpt, scaler_X, scaler_y, device)
    with open(PROJECT_ROOT / "config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return ExperimentContext(X, wind_true, scaler_X, scaler_y, model, config, device)


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
        "h_rmse": horizontal_rmse(wind_true[anomaly_slice], win),
        "jitter": jitter_mean(win),
        "max_jump": max_step_jump(wind[boundary_slice]),
        "closure_rmse": closure_rmse(win, last_phys[anomaly_slice]),
    }


def run_case(
    ctx: ExperimentContext,
    start_idx: int,
    window_size: int,
    anomaly_type: str,
    strength: float,
    ema_alphas: Iterable[float],
    batch_size: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    end_idx = start_idx + window_size
    X_window = ctx.X[start_idx:end_idx].copy()
    wind_true = ctx.wind_true[start_idx:end_idx]
    X_anom, anom_start, anom_end = evidence.inject_anomaly(
        X_window, ctx.scaler_X, anomaly_type, strength, seed=seed
    )
    last_phys = evidence.denorm_last_step(X_anom, ctx.scaler_X)
    pigru_out = evidence.predict_pigru(ctx.model, X_anom, batch_size, ctx.device)
    pigru_wind = evidence.denorm_wind(pigru_out["wind"], ctx.scaler_y)
    akf_wind, diagnostics = evidence.run_fast_pirnn_akf(
        ctx.config, pigru_out, X_anom, ctx.scaler_X, ctx.scaler_y, continuous=True
    )

    outputs = {"PI-GRU (Raw)": pigru_wind, "PIRNN-AKF": akf_wind}
    for alpha in ema_alphas:
        ema_wind = np.zeros_like(pigru_wind)
        for axis in range(3):
            ema_wind[:, axis] = compute_ema(pigru_wind[:, axis], alpha=alpha)
        outputs[f"EMA alpha={alpha:g}"] = ema_wind

    anomaly_slice = slice(anom_start, anom_end)
    boundary_slice = slice(max(0, anom_start - 20), min(window_size, anom_end + 20))
    rows = []
    for method, wind in outputs.items():
        row = evaluate_output(method, wind, wind_true, last_phys, anomaly_slice, boundary_slice)
        row.update({
            "start_idx": start_idx,
            "window_size": window_size,
            "anomaly_type": anomaly_type,
            "strength": strength,
            "anomaly_start_step": anom_start,
            "anomaly_end_step": anom_end,
        })
        rows.append(row)

    diag = pd.DataFrame({
        "step": np.arange(window_size),
        "time_s": np.arange(window_size) * 0.02,
        "start_idx": start_idx,
        "anomaly_type": anomaly_type,
        "strength": strength,
        "r_scale_mean": np.mean(pigru_out["r_scale"], axis=1),
        "akf_weight": diagnostics["akf_weight"],
        "nn_weight": diagnostics["nn_weight"],
        "innovation_norm": diagnostics["innovation_norm"],
    })
    return pd.DataFrame(rows), diag


def add_relative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["jitter_reduction_vs_pigru_pct"] = np.nan
    out["jump_reduction_vs_pigru_pct"] = np.nan
    for _, idx in out.groupby(["start_idx", "anomaly_type", "strength"]).groups.items():
        sub = out.loc[idx]
        base = sub[sub["method"] == "PI-GRU (Raw)"]
        if base.empty:
            continue
        base_jitter = float(base["jitter"].iloc[0])
        base_jump = float(base["max_jump"].iloc[0])
        if base_jitter > 0:
            out.loc[idx, "jitter_reduction_vs_pigru_pct"] = (1.0 - out.loc[idx, "jitter"] / base_jitter) * 100.0
        if base_jump > 0:
            out.loc[idx, "jump_reduction_vs_pigru_pct"] = (1.0 - out.loc[idx, "max_jump"] / base_jump) * 100.0
    return out


def aggregate(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["method"]
    metric_cols = [
        "h_rmse",
        "jitter",
        "max_jump",
        "closure_rmse",
        "jitter_reduction_vs_pigru_pct",
        "jump_reduction_vs_pigru_pct",
    ]
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
    return add_balanced_score(agg, group_cols)


def add_balanced_score(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = summary.copy()
    out["balanced_score"] = np.nan
    scope_cols = [c for c in group_cols if c != "method"]
    if not scope_cols:
        groups = [((), out.index)]
    else:
        groups = out.groupby(scope_cols).groups.items()
    for _, idx in groups:
        sub = out.loc[idx]
        base = sub[sub["method"] == "PI-GRU (Raw)"]
        if base.empty:
            continue
        base = base.iloc[0]
        score = np.zeros(len(sub), dtype=float)
        for col in ["h_rmse_mean", "jitter_mean", "max_jump_mean", "closure_rmse_mean"]:
            denom = float(base[col])
            if denom > 0:
                score += sub[col].to_numpy(dtype=float) / denom
        out.loc[sub.index, "balanced_score"] = score / 4.0
    return out.sort_values(scope_cols + ["balanced_score"] if scope_cols else ["balanced_score"])


def set_plot_style() -> None:
    apply_style()


def savefig_all(fig, out_base: Path) -> None:
    save_figure(fig, out_base)
