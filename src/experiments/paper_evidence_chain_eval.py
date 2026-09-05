"""Build paper evidence tables for PIRNN-AKF.

This script is intentionally self-contained: it evaluates the current
reproducible checkpoints on Test-ID/Test-OOD and exports the diagnostics that
support the paper narrative beyond average RMSE.
"""

import argparse
import importlib.util
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_IDX = {
    "vel_n": 0,
    "vel_e": 1,
    "vel_d": 2,
    "roll": 9,
    "pitch": 10,
    "yaw": 11,
    "airspeed": 19,
}


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path_value.lstrip("../")).resolve()


def load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def latest_checkpoint(base_dir: Path, prefix: str, checkpoint_name: str = "best_model.pth") -> Path:
    dirs = sorted(
        [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)],
        reverse=True,
    )
    for run_dir in dirs:
        candidate = run_dir / checkpoint_name
        if candidate.exists():
            return candidate
        nested = sorted([p for p in run_dir.iterdir() if p.is_dir()], reverse=True)
        for sub in nested:
            candidate = sub / checkpoint_name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No {checkpoint_name} under {base_dir} with prefix={prefix}")


def load_norm_params(data_dir: Path):
    norm_path = data_dir / "norm_params.pkl"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing norm params: {norm_path}")
    with open(norm_path, "rb") as f:
        meta = pickle.load(f)
    return meta["scaler_X"], meta["scaler_y"]


def model_norm_params(scaler_X, scaler_y) -> Dict[str, np.ndarray]:
    return {
        "X_mean": scaler_X.mean_,
        "X_scale": scaler_X.scale_,
        "y_mean": scaler_y.mean_,
        "y_scale": scaler_y.scale_,
    }


def load_pigru_model(checkpoint_path: Path, scaler_X, scaler_y, device: torch.device):
    module = load_module(PROJECT_ROOT / "src" / "2_pigru_module.py", "pigru_model_for_evidence")
    PIGRU = module.PIGRU
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {}).get("model", {})
    yaw_invariant = bool(cfg.get("yaw_invariant", False))
    model = PIGRU(
        input_size=cfg.get("input_size", 20),
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        dropout=0.0,
        rnn_type=cfg.get("rnn_type", "gru"),
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


def load_vanilla_model(checkpoint_path: Path, device: torch.device):
    module = load_module(PROJECT_ROOT / "src" / "2b_vanilla_gru.py", "vanilla_model_for_evidence")
    VanillaGRU = module.VanillaGRU
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {}).get("model", {})
    model = VanillaGRU(
        input_size=cfg.get("input_size", 20),
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        dropout=0.0,
        rnn_type=cfg.get("rnn_type", checkpoint.get("rnn_type", "gru")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_vanilla(model, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(X), batch_size), desc="Vanilla GRU"):
            xb = torch.from_numpy(X[start:start + batch_size]).float().to(device)
            outs.append(model(xb, return_dict=False).cpu().numpy())
    return np.vstack(outs).astype(np.float64)


def predict_pigru(model, X: np.ndarray, batch_size: int, device: torch.device) -> Dict[str, np.ndarray]:
    accum: Dict[str, List[np.ndarray]] = {
        "wind": [],
        "q_scale": [],
        "r_scale": [],
        "angles": [],
        "confidence": [],
    }
    with torch.no_grad():
        for start in tqdm(range(0, len(X), batch_size), desc="PI-GRU"):
            xb = torch.from_numpy(X[start:start + batch_size]).float().to(device)
            out = model(xb, return_dict=True)
            accum["wind"].append(out["wind_estimate"].cpu().numpy())
            accum["q_scale"].append(out["q_scale"].cpu().numpy())
            accum["r_scale"].append(out["r_scale"].cpu().numpy())
            accum["angles"].append(out["angles"].cpu().numpy())
            accum["confidence"].append(out["confidence"].cpu().numpy())
    return {k: np.vstack(v).astype(np.float64) for k, v in accum.items()}


def denorm_wind(wind_norm: np.ndarray, scaler_y) -> np.ndarray:
    return wind_norm * scaler_y.scale_[:3] + scaler_y.mean_[:3]


def denorm_y(y_norm: np.ndarray, scaler_y) -> np.ndarray:
    return scaler_y.inverse_transform(y_norm)


def denorm_last_step(X_norm: np.ndarray, scaler_X) -> np.ndarray:
    last = X_norm[:, -1, :]
    return last * scaler_X.scale_ + scaler_X.mean_


def direction_error_deg(wind_true: np.ndarray, wind_pred: np.ndarray) -> np.ndarray:
    true_dir = np.degrees(np.arctan2(wind_true[:, 1], wind_true[:, 0]))
    pred_dir = np.degrees(np.arctan2(wind_pred[:, 1], wind_pred[:, 0]))
    diff = (pred_dir - true_dir + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def vector_metrics(split: str, method: str, wind_true: np.ndarray, wind_pred: np.ndarray) -> Dict[str, float]:
    err = wind_pred - wind_true
    mag_true = np.linalg.norm(wind_true, axis=1)
    mag_pred = np.linalg.norm(wind_pred, axis=1)
    dir_err = direction_error_deg(wind_true, wind_pred)
    return {
        "split": split,
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


def physical_consistency_metrics(
    split: str,
    method: str,
    wind_pred: np.ndarray,
    last_phys: np.ndarray,
    tol: float,
) -> Dict[str, float]:
    vg = last_phys[:, 0:3]
    tas = last_phys[:, FEATURE_IDX["airspeed"]]
    tas_theory = np.linalg.norm(vg - wind_pred, axis=1)
    residual = tas_theory - tas
    abs_residual = np.abs(residual)
    return {
        "split": split,
        "method": method,
        "airspeed_closure_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "airspeed_closure_mae": float(np.mean(abs_residual)),
        "airspeed_closure_p95": float(np.percentile(abs_residual, 95)),
        "violation_rate_gt_tol": float(np.mean(abs_residual > tol)),
        "tol_mps": float(tol),
    }


def weakwind_rows(split: str, method: str, wind_true: np.ndarray, wind_pred: np.ndarray) -> List[Dict[str, float]]:
    wh_true = np.linalg.norm(wind_true[:, :2], axis=1)
    wh_pred = np.linalg.norm(wind_pred[:, :2], axis=1)
    dir_err = direction_error_deg(wind_true, wind_pred)
    bins = [
        (0.0, 0.5, "<0.5"),
        (0.5, 1.0, "0.5-1.0"),
        (1.0, 1.5, "1.0-1.5"),
        (1.5, 2.0, "1.5-2.0"),
        (2.0, 2.5, "2.0-2.5"),
        (2.5, 3.0, "2.5-3.0"),
        (3.0, np.inf, ">3.0"),
    ]
    rows = []
    for lo, hi, label in bins:
        mask = (wh_true >= lo) & (wh_true < hi)
        if not np.any(mask):
            continue
        rows.append({
            "split": split,
            "method": method,
            "wind_bin": label,
            "n": int(np.sum(mask)),
            "frac": float(np.mean(mask)),
            "true_h_mag_mean": float(np.mean(wh_true[mask])),
            "pred_h_mag_median": float(np.median(wh_pred[mask])),
            "dir_mae": float(np.mean(dir_err[mask])),
            "evil_ratio_dir_gt_60": float(np.mean(dir_err[mask] > 60.0)),
            "collapse_rate_pred_h_lt_0p3": float(np.mean(wh_pred[mask] < 0.3)),
        })
    weak_mask = wh_true < 1.5
    if np.any(weak_mask):
        rows.append({
            "split": split,
            "method": method,
            "wind_bin": "weak<1.5",
            "n": int(np.sum(weak_mask)),
            "frac": float(np.mean(weak_mask)),
            "true_h_mag_mean": float(np.mean(wh_true[weak_mask])),
            "pred_h_mag_median": float(np.median(wh_pred[weak_mask])),
            "dir_mae": float(np.mean(dir_err[weak_mask])),
            "evil_ratio_dir_gt_60": float(np.mean(dir_err[weak_mask] > 60.0)),
            "collapse_rate_pred_h_lt_0p3": float(np.mean(wh_pred[weak_mask] < 0.3)),
        })
    return rows


def temporal_metrics(split: str, method: str, wind_pred: np.ndarray) -> Dict[str, float]:
    if len(wind_pred) < 3:
        return {
            "split": split,
            "method": method,
            "tv_mean": np.nan,
            "tv_p95": np.nan,
            "jitter_mean": np.nan,
            "jitter_p95": np.nan,
        }
    d1 = np.diff(wind_pred, axis=0)
    d2 = np.diff(d1, axis=0)
    tv = np.linalg.norm(d1, axis=1)
    jitter = np.linalg.norm(d2, axis=1)
    return {
        "split": split,
        "method": method,
        "tv_mean": float(np.mean(tv)),
        "tv_p95": float(np.percentile(tv, 95)),
        "jitter_mean": float(np.mean(jitter)),
        "jitter_p95": float(np.percentile(jitter, 95)),
    }


def run_fast_pirnn_akf(
    config: Dict,
    pigru_out: Dict[str, np.ndarray],
    X: np.ndarray,
    scaler_X,
    scaler_y,
    continuous: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    fusion = load_module(PROJECT_ROOT / "src" / "5_pigru_akf_fusion.py", "akf_for_evidence")
    AdaptiveKalmanFilter = fusion.AdaptiveKalmanFilter
    akf_cfg = (config.get("akf", {}) or {})
    constants = {
        "prediction_delta_gain": 0.25,
        "mahalanobis_gate": 9.0,
        "maneuver_gyro_weight": 0.45,
        "maneuver_acc_weight": 0.30,
        "maneuver_control_weight": 0.20,
        "maneuver_throttle_weight": 0.05,
        "disagreement_scale": 3.0,
        "kinematic_trust_base": 0.90,
        "kinematic_trust_slope": 0.70,
        "kinematic_trust_min": 0.20,
        "kinematic_trust_max": 0.90,
        "q_maneuver_gain": 0.45,
        "r_maneuver_gain": 0.20,
        "r_disagreement_gain": 1.50,
        "r_outlier_multiplier": 2.0,
        "fusion_base": 0.58,
        "fusion_confidence_gain": 0.10,
        "fusion_disagreement_gain": 0.28,
        "fusion_covariance_gain": 0.10,
        "fusion_min": 0.18,
        "fusion_max": 0.72,
        "fusion_outlier_cap": 0.28,
    }
    constants.update(akf_cfg.get("constants", {}) or {})
    c = {key: float(value) for key, value in constants.items()}
    dt = 1.0 / config["data"]["sampling_rate"]
    if akf_cfg.get("process_aware", False):
        # process-aware 体制：抬高标称 Q、适度抬高运动学伪量测 R，使动态 q_scale 真正改变增益
        akf = AdaptiveKalmanFilter(
            dt=dt,
            Q_nominal=np.diag(akf_cfg.get("q_nominal", [0.05, 0.05, 0.02])),
            R_kin_nominal=np.diag(akf_cfg.get("r_kin_nominal", [1.0, 1.0, 0.36])),
            R_nn_nominal=np.diag(akf_cfg.get("r_nn_nominal", [0.06, 0.06, 0.04])),
            prediction_delta_gain=c["prediction_delta_gain"],
            outlier_threshold=c["mahalanobis_gate"],
        )
    else:
        akf = AdaptiveKalmanFilter(
            dt=dt,
            prediction_delta_gain=c["prediction_delta_gain"],
            outlier_threshold=c["mahalanobis_gate"],
        )

    wind_mean = scaler_y.mean_[:3]
    wind_std = scaler_y.scale_[:3]
    response_profile = str((config.get("akf", {}) or {}).get("response_profile", "smooth")).lower()
    if response_profile not in {"smooth", "tracking"}:
        response_profile = "smooth"
    wind_nn = denorm_wind(pigru_out["wind"], scaler_y)
    q_scale = pigru_out["q_scale"]
    r_scale = pigru_out["r_scale"]
    angles = pigru_out["angles"]
    confidence = pigru_out["confidence"].reshape(-1)
    last_phys = denorm_last_step(X, scaler_X)

    n = len(X)
    fused = np.zeros((n, 3), dtype=np.float64)
    wind_akf = np.zeros((n, 3), dtype=np.float64)
    wind_kin = np.zeros((n, 3), dtype=np.float64)
    innovation_norm = np.zeros(n, dtype=np.float64)
    nis = np.zeros(n, dtype=np.float64)
    nn_weight = np.zeros(n, dtype=np.float64)
    akf_weight = np.zeros(n, dtype=np.float64)
    P_diag = np.zeros((n, 3), dtype=np.float64)
    Q_diag = np.zeros((n, 3), dtype=np.float64)
    R_diag = np.zeros((n, 3), dtype=np.float64)

    initialized = False
    prev_wind_nn: Optional[np.ndarray] = None

    def compute_maneuver(row: np.ndarray) -> float:
        gyro = row[12:15]
        acc = row[6:9]
        ctrl = row[15:18]
        throttle = float(row[18])
        score = (
            c["maneuver_gyro_weight"] * np.linalg.norm(gyro) / 1.2
            + c["maneuver_acc_weight"] * np.linalg.norm(acc) / 8.0
            + c["maneuver_control_weight"] * np.linalg.norm(ctrl) / 1.2
            + c["maneuver_throttle_weight"] * abs(throttle - 0.5) * 2.0
        )
        return float(np.clip(score, 0.0, 3.0))

    def stabilize_kinematic(kin: np.ndarray, nn: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        gap_vec = kin - nn
        gap = float(np.linalg.norm(gap_vec))
        gap_ratio = np.clip(gap / c["disagreement_scale"], 0.0, 1.0)
        wind_limit = float(config.get("physics", {}).get("wind_magnitude_max", 15.0))
        kin_outlier = gap > 4.0 or float(np.linalg.norm(kin)) > wind_limit * 1.2
        kin_trust = np.clip(
            c["kinematic_trust_base"] - c["kinematic_trust_slope"] * gap_ratio,
            c["kinematic_trust_min"],
            c["kinematic_trust_max"],
        )
        if kin_outlier:
            kin_trust = min(kin_trust, 0.35)
        return kin_trust * kin + (1.0 - kin_trust) * nn, gap, bool(kin_outlier)

    for i in tqdm(range(n), desc="Fast PIRNN-AKF"):
        if not continuous or i == 0:
            akf.reset()
            initialized = False
            prev_wind_nn = None

        row = last_phys[i]
        vg_ned = row[0:3]
        roll, pitch, yaw = row[9:12]
        airspeed = float(max(row[19], 0.1))
        maneuver = compute_maneuver(row)
        wind_kin_raw, _, _ = akf.construct_kinematic_wind_measurement(
            vg_ned=vg_ned,
            tas=airspeed,
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            angles=angles[i],
        )
        wind_kin_stable, gap, kin_outlier = stabilize_kinematic(wind_kin_raw, wind_nn[i])
        gap_ratio = np.clip(gap / c["disagreement_scale"], 0.0, 1.0)

        q_eff = np.clip(q_scale[i] * (1.0 + c["q_maneuver_gain"] * maneuver), 0.1, 20.0)
        r_gain = 1.0 + c["r_disagreement_gain"] * gap_ratio
        if kin_outlier:
            r_gain *= c["r_outlier_multiplier"]
        r_eff = np.clip(
            r_scale[i] * (1.0 + c["r_maneuver_gain"] * maneuver) * r_gain,
            0.1,
            20.0,
        )

        if not initialized:
            akf.reset()
            akf.x = 0.80 * wind_nn[i].astype(np.float64) + 0.20 * wind_kin_stable.astype(np.float64)
            akf.P = np.diag([2.0, 2.0, 0.6]).astype(np.float64)
            initialized = True
            neural_delta = np.zeros(3, dtype=np.float64)
        else:
            neural_delta = wind_nn[i] - prev_wind_nn if prev_wind_nn is not None else np.zeros(3)

        akf.update_noise_covariance(q_eff, r_eff)
        akf.predict(neural_wind_delta=neural_delta)
        z_meas = np.array([vg_ned[0], vg_ned[1], vg_ned[2], airspeed], dtype=np.float64)
        akf.update(
            z_meas,
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            confidence=float(confidence[i]),
            angles=angles[i],
            nn_measurement=wind_nn[i],
            maneuver_score=maneuver,
            wind_kin_override=wind_kin_stable,
        )

        diagnostics = akf.get_diagnostics()
        cur_akf = akf.get_wind_estimate()
        p_ratio = np.clip(np.mean(diagnostics["P_diag"]) / 2.0, 0.0, 1.0)
        if response_profile == "tracking":
            maneuver_ratio = np.clip(maneuver / 2.0, 0.0, 1.0)
            cur_akf_weight = np.clip(
                0.45 + 0.06 * confidence[i] - 0.32 * gap_ratio - 0.12 * p_ratio - 0.10 * maneuver_ratio,
                0.10,
                0.62,
            )
        else:
            cur_akf_weight = np.clip(
                c["fusion_base"]
                + c["fusion_confidence_gain"] * confidence[i]
                - c["fusion_disagreement_gain"] * gap_ratio
                - c["fusion_covariance_gain"] * p_ratio,
                c["fusion_min"],
                c["fusion_max"],
            )
        if kin_outlier:
            cur_akf_weight = min(cur_akf_weight, c["fusion_outlier_cap"])
        cur_nn_weight = 1.0 - cur_akf_weight
        fused[i] = cur_akf_weight * cur_akf + cur_nn_weight * wind_nn[i]
        wind_akf[i] = cur_akf
        wind_kin[i] = wind_kin_raw
        innovation_norm[i] = diagnostics["innovation_norm"]
        nis[i] = diagnostics["nis"]
        nn_weight[i] = cur_nn_weight
        akf_weight[i] = cur_akf_weight
        P_diag[i] = diagnostics["P_diag"]
        Q_diag[i] = diagnostics["Q_diag"]
        R_diag[i] = diagnostics["R_diag"]
        prev_wind_nn = wind_nn[i].copy()

    diagnostics = {
        "wind_nn": wind_nn,
        "wind_akf": wind_akf,
        "wind_kin": wind_kin,
        "q_scale": q_scale,
        "r_scale": r_scale,
        "confidence": confidence,
        "nn_weight": nn_weight,
        "akf_weight": akf_weight,
        "innovation_norm": innovation_norm,
        "nis": nis,
        "P_diag": P_diag,
        "Q_diag": Q_diag,
        "R_diag": R_diag,
        "wind_fused_norm": (fused - wind_mean) / wind_std,
    }
    return fused, diagnostics


def inject_anomaly(
    X: np.ndarray,
    scaler_X,
    anomaly_type: str,
    strength: float,
    seed: int,
) -> Tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    n, seq_len, n_feat = X.shape
    X_phys = scaler_X.inverse_transform(X.reshape(-1, n_feat)).reshape(n, seq_len, n_feat)
    start = max(1, int(n * 0.45))
    end = max(start + 1, int(n * 0.55))
    if anomaly_type == "gps_spike":
        X_phys[start:end, -1, 0:3] += strength
    elif anomaly_type == "tas_spike":
        X_phys[start:end, -1, 19] += strength
    elif anomaly_type == "attitude_spike":
        X_phys[start:end, -1, 9:12] += np.deg2rad(strength)
    elif anomaly_type == "sensor_dropout":
        X_phys[start:end, -1, 0:3] = 0.0
        X_phys[start:end, -1, 19] = 0.0
    elif anomaly_type == "gaussian_burst":
        X_phys[start:end, -1, 0:3] += rng.normal(0.0, strength, size=(end - start, 3))
        X_phys[start:end, -1, 19] += rng.normal(0.0, strength, size=(end - start,))
    else:
        raise ValueError(f"Unknown anomaly_type={anomaly_type}")
    X_corrupt = scaler_X.transform(X_phys.reshape(-1, n_feat)).reshape(n, seq_len, n_feat)
    return X_corrupt.astype(np.float32), start, end


def anomaly_rows(
    split: str,
    X: np.ndarray,
    y_true: np.ndarray,
    pigru_model,
    scaler_y,
    config: Dict,
    scaler_X,
    batch_size: int,
    device: torch.device,
    max_samples: int,
    strength: float,
    seed: int,
) -> List[Dict[str, float]]:
    if max_samples <= 0:
        return []
    X_base = X[:max_samples]
    y_base = y_true[:max_samples]
    clean_pigru = predict_pigru(pigru_model, X_base, batch_size, device)
    clean_pi = denorm_wind(clean_pigru["wind"], scaler_y)
    clean_sys, _ = run_fast_pirnn_akf(config, clean_pigru, X_base, scaler_X, scaler_y, continuous=True)
    clean_pi_p95 = float(np.percentile(np.linalg.norm(clean_pi - y_base, axis=1), 95))
    clean_sys_p95 = float(np.percentile(np.linalg.norm(clean_sys - y_base, axis=1), 95))

    rows = []
    types = ["gps_spike", "tas_spike", "attitude_spike", "sensor_dropout", "gaussian_burst"]
    for idx, anomaly_type in enumerate(types):
        X_anom, start, end = inject_anomaly(X_base, scaler_X, anomaly_type, strength, seed + idx)
        pigru_anom = predict_pigru(pigru_model, X_anom, batch_size, device)
        pi_anom = denorm_wind(pigru_anom["wind"], scaler_y)
        sys_anom, _ = run_fast_pirnn_akf(config, pigru_anom, X_anom, scaler_X, scaler_y, continuous=True)
        for method, pred, threshold in [
            ("PI-GRU", pi_anom, clean_pi_p95),
            ("PIRNN-AKF", sys_anom, clean_sys_p95),
        ]:
            err = np.linalg.norm(pred - y_base, axis=1)
            rows.append({
                "split": split,
                "anomaly_type": anomaly_type,
                "method": method,
                "n": int(len(y_base)),
                "rmse": float(np.sqrt(np.mean((pred - y_base) ** 2))),
                "peak_error": float(np.max(err)),
                "window_error_mean": float(np.mean(err[start:end])),
                "threshold_p95_clean": threshold,
                "window_start": int(start),
                "window_end": int(end),
            })
    return rows


def save_plots(out_dir: Path, weak_df: pd.DataFrame, stability_df: pd.DataFrame, diag_by_split: Dict[str, Dict[str, np.ndarray]]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    weak_focus = weak_df[weak_df["wind_bin"].isin(["<0.5", "0.5-1.0", "1.0-1.5", "weak<1.5"])]
    if not weak_focus.empty:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for method, sub in weak_focus[weak_focus["split"] == "test_id"].groupby("method"):
            ax.plot(sub["wind_bin"], sub["evil_ratio_dir_gt_60"] * 100.0, marker="o", label=method)
        ax.set_ylabel("Evil ratio dir_err > 60 deg (%)")
        ax.set_xlabel("Horizontal wind bin (m/s)")
        ax.set_title("Weak-wind failure ratio on Test-ID")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "weakwind_evil_ratio.png", dpi=220)
        fig.savefig(fig_dir / "weakwind_evil_ratio.svg")
        plt.close(fig)

    if "test_id" in diag_by_split:
        diag = diag_by_split["test_id"]
        fig, ax = plt.subplots(figsize=(7.6, 4.0))
        d2_pi = np.linalg.norm(np.diff(np.diff(diag["wind_pi"], axis=0), axis=0), axis=1)
        d2_sys = np.linalg.norm(np.diff(np.diff(diag["wind_pirnn_akf"], axis=0), axis=0), axis=1)
        upper = np.percentile(np.concatenate([d2_pi, d2_sys]), 99)
        ax.hist(d2_pi, bins=80, range=(0, upper), alpha=0.55, density=True, label="PI-GRU")
        ax.hist(d2_sys, bins=80, range=(0, upper), alpha=0.55, density=True, label="PIRNN-AKF")
        ax.set_xlabel("Jitter norm")
        ax.set_ylabel("Density")
        ax.set_title("Jitter distribution on Test-ID")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "jitter_distribution_test_id.png", dpi=220)
        fig.savefig(fig_dir / "jitter_distribution_test_id.svg")
        plt.close(fig)


def parse_splits(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PIRNN-AKF paper evidence chain.")
    parser.add_argument("--config", default="config/config_strat_expB.yaml")
    parser.add_argument("--data-dir", default="data/dataset_stratified")
    parser.add_argument("--pigru-model", default="train_data_stratified/train_lambda0.1_20260511_013913/train_lambda0.02_20260511_013918/best_model.pth")
    parser.add_argument("--vanilla-model", default="train_within_run_yaw_invariant/vanilla_gru_20260511_141939/best_model.pth")
    parser.add_argument("--splits", default="test_id,test_ood")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap per split for quick runs.")
    parser.add_argument("--anomaly-samples", type=int, default=30000)
    parser.add_argument("--anomaly-strength", type=float, default=2.0)
    parser.add_argument("--physical-tol", type=float, default=1.0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    data_dir = resolve_path(args.data_dir)
    pigru_model_path = resolve_path(args.pigru_model)
    vanilla_model_path = resolve_path(args.vanilla_model)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = resolve_path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "paper_evidence" / f"evidence_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_X, scaler_y = load_norm_params(data_dir)
    pigru_model = load_pigru_model(pigru_model_path, scaler_X, scaler_y, device)
    vanilla_model = load_vanilla_model(vanilla_model_path, device)

    manifest = {
        "timestamp": timestamp,
        "device": str(device),
        "config": str(config_path),
        "data_dir": str(data_dir),
        "pigru_model": str(pigru_model_path),
        "vanilla_model": str(vanilla_model_path),
        "splits": parse_splits(args.splits),
        "max_samples": args.max_samples,
        "anomaly_samples": args.anomaly_samples,
    }
    with open(out_dir / "artifact_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    method_rows: List[Dict[str, float]] = []
    physics_rows: List[Dict[str, float]] = []
    weak_rows: List[Dict[str, float]] = []
    stability_rows: List[Dict[str, float]] = []
    anomaly_all_rows: List[Dict[str, float]] = []
    diag_by_split: Dict[str, Dict[str, np.ndarray]] = {}

    for split in parse_splits(args.splits):
        X_path = data_dir / f"X_{split}.npy"
        y_path = data_dir / f"y_{split}.npy"
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Missing split arrays: {X_path} / {y_path}")
        X = np.load(X_path)
        y_norm = np.load(y_path)
        if args.max_samples > 0:
            X = X[:args.max_samples]
            y_norm = y_norm[:args.max_samples]

        y_denorm = denorm_y(y_norm, scaler_y)
        wind_true = y_denorm[:, :3]
        last_phys = denorm_last_step(X, scaler_X)

        vanilla_norm = predict_vanilla(vanilla_model, X, args.batch_size, device)
        vanilla = denorm_wind(vanilla_norm, scaler_y)
        pigru_out = predict_pigru(pigru_model, X, args.batch_size, device)
        pi = denorm_wind(pigru_out["wind"], scaler_y)
        pirnn_akf, akf_diag = run_fast_pirnn_akf(config, pigru_out, X, scaler_X, scaler_y, continuous=True)

        predictions = {
            "Vanilla GRU": vanilla,
            "PI-GRU": pi,
            "PIRNN-AKF": pirnn_akf,
        }
        for method, pred in predictions.items():
            method_rows.append(vector_metrics(split, method, wind_true, pred))
            physics_rows.append(physical_consistency_metrics(split, method, pred, last_phys, args.physical_tol))
            weak_rows.extend(weakwind_rows(split, method, wind_true, pred))
            if method in {"PI-GRU", "PIRNN-AKF"}:
                stability_rows.append(temporal_metrics(split, method, pred))

        diag_by_split[split] = {
            "wind_true": wind_true,
            "wind_pi": pi,
            "wind_pirnn_akf": pirnn_akf,
            **akf_diag,
        }
        np.savez_compressed(out_dir / f"diagnostics_{split}.npz", **diag_by_split[split])

        anomaly_n = min(args.anomaly_samples, len(X))
        anomaly_all_rows.extend(
            anomaly_rows(
                split=split,
                X=X,
                y_true=wind_true,
                pigru_model=pigru_model,
                scaler_y=scaler_y,
                config=config,
                scaler_X=scaler_X,
                batch_size=args.batch_size,
                device=device,
                max_samples=anomaly_n,
                strength=args.anomaly_strength,
                seed=args.seed,
            )
        )

    method_df = pd.DataFrame(method_rows)
    physics_df = pd.DataFrame(physics_rows)
    weak_df = pd.DataFrame(weak_rows)
    stability_df = pd.DataFrame(stability_rows)
    anomaly_df = pd.DataFrame(anomaly_all_rows)

    method_df.to_csv(out_dir / "method_metrics.csv", index=False)
    physics_df.to_csv(out_dir / "physics_consistency_metrics.csv", index=False)
    weak_df.to_csv(out_dir / "weakwind_metrics.csv", index=False)
    stability_df.to_csv(out_dir / "stability_metrics.csv", index=False)
    anomaly_df.to_csv(out_dir / "anomaly_robustness.csv", index=False)

    save_plots(out_dir, weak_df, stability_df, diag_by_split)

    print("=" * 80)
    print("Paper evidence chain complete")
    print(f"Output: {out_dir}")
    print("Files:")
    for name in [
        "artifact_manifest.json",
        "method_metrics.csv",
        "physics_consistency_metrics.csv",
        "weakwind_metrics.csv",
        "stability_metrics.csv",
        "anomaly_robustness.csv",
    ]:
        print(f"  - {out_dir / name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
