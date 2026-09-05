#!/usr/bin/env python3
"""Evaluate the latest Vanilla GRU baseline on stratified test_id/test_ood.

This script mirrors the direction diagnostics used by aggregate_dir_metrics.py,
but only for the VanillaGRU model saved by src/3b_train_vanilla_gru.py.
"""
import json
import pickle
from importlib import util
from pathlib import Path

import numpy as np
import torch


PROJ = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ / "data" / "dataset_stratified"
MODEL_ROOT = PROJ / "train_within_run_yaw_invariant"
OUT = PROJ / "data" / "evaluation" / "vanilla_gru_metrics.json"


def import_vanilla():
    path = PROJ / "src" / "2b_vanilla_gru.py"
    spec = util.spec_from_file_location("vanilla_gru", path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VanillaGRU


def latest_model_dir():
    dirs = sorted([p for p in MODEL_ROOT.glob("vanilla_gru_*") if p.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"No vanilla_gru_* directory under {MODEL_ROOT}")
    return dirs[-1]


def load_norm():
    with open(DATA_DIR / "norm_params.pkl", "rb") as f:
        return pickle.load(f)


def load_model():
    VanillaGRU = import_vanilla()
    model_dir = latest_model_dir()
    ckpt_path = model_dir / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = VanillaGRU(input_size=20, hidden_size=128, num_layers=2, dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_dir, ckpt


def predict(model, X, batch_size=8192):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            yp = model(xb).detach().cpu().numpy()
            preds.append(yp)
    return np.concatenate(preds, axis=0)


def angle_error_deg(pred_h, true_h):
    dp = np.degrees(np.arctan2(pred_h[:, 1], pred_h[:, 0]))
    dt = np.degrees(np.arctan2(true_h[:, 1], true_h[:, 0]))
    return np.abs((dp - dt + 180.0) % 360.0 - 180.0)


def metrics(y_real, yp_real):
    err = yp_real - y_real
    rmse_3d = float(np.sqrt(np.mean(err[:, :3] ** 2)))
    rmse_per = np.sqrt(np.mean(err[:, :3] ** 2, axis=0)).tolist()
    mag_rmse = float(np.sqrt(np.mean(
        (np.linalg.norm(yp_real[:, :3], axis=1) - np.linalg.norm(y_real[:, :3], axis=1)) ** 2
    )))

    true_h = y_real[:, :2]
    pred_h = yp_real[:, :2]
    true_hmag = np.linalg.norm(true_h, axis=1)
    dir_err = angle_error_deg(pred_h, true_h)

    m_eng = true_hmag >= 1.5
    m_clean = dir_err <= 30.0
    m_evil = dir_err > 30.0

    bins = [(0, 0.5, "<0.5"), (0.5, 1.0, "0.5-1.0"), (1.0, 1.5, "1.0-1.5"),
            (1.5, 2.0, "1.5-2.0"), (2.0, 2.5, "2.0-2.5"),
            (2.5, 3.0, "2.5-3.0"), (3.0, np.inf, ">3.0")]
    by_bin = []
    for lo, hi, label in bins:
        m = (true_hmag >= lo) & (true_hmag < hi)
        if not np.any(m):
            continue
        err_h = yp_real[m, :2] - y_real[m, :2]
        rmse_h = float(np.sqrt(np.mean(err_h ** 2)))
        mean_w = float(np.mean(true_hmag[m]))
        by_bin.append({
            "label": label,
            "n": int(np.sum(m)),
            "frac": float(np.mean(m)),
            "mean_w": mean_w,
            "rmse_h": rmse_h,
            "dir_mae": float(np.mean(dir_err[m])),
            "geom_lower": float(np.degrees(np.arctan2(rmse_h, max(mean_w, 1e-9)))),
        })

    return {
        "rmse_3d": rmse_3d,
        "rmse_per": rmse_per,
        "mag_rmse": mag_rmse,
        "raw_dir": float(np.mean(dir_err)),
        "eng_dir": float(np.mean(dir_err[m_eng])),
        "eng_frac": float(np.mean(m_eng)),
        "conf_dir": float("nan"),
        "conf_thr": float("nan"),
        "evil_pct": float(np.mean(m_evil)),
        "evil_dir": float(np.mean(dir_err[m_evil])),
        "clean_dir": float(np.mean(dir_err[m_clean])),
        "evil_w_med": float(np.median(true_hmag[m_evil])),
        "by_bin": by_bin,
        "n_total": int(len(y_real)),
    }


def main():
    print("=" * 90)
    print("  Vanilla GRU baseline evaluation on stratified test sets")
    print("=" * 90)
    norm = load_norm()
    sy = norm["scaler_y"]
    model, model_dir, ckpt = load_model()
    print(f"  model_dir: {model_dir}")
    print(f"  ckpt epoch: {ckpt.get('epoch')}")

    per_split = {}
    for split in ["test_id", "test_ood"]:
        X = np.load(DATA_DIR / f"X_{split}.npy")
        y = np.load(DATA_DIR / f"y_{split}.npy")
        yp = predict(model, X)
        y_real = y[:, :3] * sy.scale_[:3] + sy.mean_[:3]
        yp_real = yp[:, :3] * sy.scale_[:3] + sy.mean_[:3]
        per_split[split] = metrics(y_real, yp_real)
        r = per_split[split]
        print(
            f"  {split}: RMSE_3D={r['rmse_3d']:.4f}, "
            f"mag_RMSE={r['mag_rmse']:.4f}, raw_dir={r['raw_dir']:.2f}°, "
            f"eng_dir={r['eng_dir']:.2f}°, clean_dir={r['clean_dir']:.2f}°, "
            f"evil={100*r['evil_pct']:.2f}%"
        )

    info = {
        "meta": {
            "name": "Vanilla GRU",
            "note": "no physics heads, no AKF, same stratified data",
            "model_dir": str(model_dir),
        },
        "per_split": per_split,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    print(f"  saved: {OUT}")


if __name__ == "__main__":
    main()
