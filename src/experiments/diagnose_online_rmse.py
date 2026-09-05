"""
diagnose_online_rmse.py — 归因在线/HITL 段 RMSE 崩溃的根因（无需硬件/SITL）
================================================================================

思路
----
verify_online_45d_chain.py 已证明：在 50Hz、真实特征下，在线特征装配可复现离线
RMSE (~0.2/0.58)。但真机在线 (online_6c) 实测 RMSE ~3.9，估计幅值仅真值 ~1/3。

本脚本用**同一批离线 test 数据**，逐一注入"真机在线才有"的失真，量化各自对 RMSE
的贡献，从而确认主因：

  A. baseline           : 50Hz + 真实特征（对照，应≈0.2/0.58）
  B. rate_29hz          : 把 50Hz 流重采样到 ~29Hz（真机实测环路速率），
                          提取器仍按 sampling_rate=50 计算加速度 dt（复刻真机 bug）,
                          再按 29Hz 间隔构造 100 长窗口 → 时序被拉伸 + 加速度被放大
  C. accel_dt_only      : 仅把加速度 6-8 按 dt=1/29 的错误缩放（隔离 dt bug）
  D. zero_targets       : 把 target_v* (32-34) 置 0、并令 v*_err=vel（隔离"若期望地速缺失"）

用法
----
  .venv/bin/python src/experiments/diagnose_online_rmse.py --dataset test_id --max_files 8
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

from online_feature_extractor import FEATURE_IDX, StreamingFeatureExtractor  # noqa: E402

SEQ_LEN = 100
DOWNSAMPLE = 5
TRAIN_RATE = 50.0  # 训练/离线等效采样率 (Hz)

# 复用 verify 脚本的清洗与列映射
_vspec = importlib.util.spec_from_file_location(
    "verify_chain", _SRC / "experiments" / "verify_online_45d_chain.py")
_vmod = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(_vmod)
COL2SIG = _vmod.COL2SIG


def _load_preproc():
    spec = importlib.util.spec_from_file_location("preproc", _SRC / "1_preprocessing_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_online_feat_from_df(cleaned_df, sampling_rate=TRAIN_RATE):
    cols = {c: cleaned_df[c].to_numpy(dtype=np.float64)
            for c in COL2SIG if c in cleaned_df.columns}
    ext = StreamingFeatureExtractor(sampling_rate=sampling_rate)
    T = len(cleaned_df)
    feat = np.zeros((T, 45), dtype=np.float32)
    for t in range(T):
        sig = {COL2SIG[c]: cols[c][t] for c in cols}
        feat[t] = ext.push(sig)
    return feat


def load_model():
    cfg_path = _ROOT / "config" / "config_sitl.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    from inference_backends import create_backend
    backend = create_backend(cfg)
    backend.load()
    msp = cfg['training']['model_save_path']
    norm_path = (Path(msp) if os.path.isabs(msp) else _ROOT / msp) / "norm_params.pkl"
    with open(norm_path, 'rb') as f:
        norm = pickle.load(f)
    sx = norm['scaler_X']
    sy = norm['scaler_y']
    return backend, sx, np.asarray(sy.mean_, np.float32)[:3], np.asarray(sy.scale_, np.float32)[:3]


def predict(backend, sx, wm, ws, feat, n_win, batch=2048):
    import torch
    fn = sx.transform(feat.astype(np.float32)).astype(np.float32)
    preds = np.zeros((n_win, 3), np.float32)
    i = 0
    while i < n_win:
        e = min(i + batch, n_win)
        wins = np.stack([fn[j:j + SEQ_LEN] for j in range(i, e)])
        with torch.no_grad():
            out = backend.model(torch.from_numpy(wins).to(backend.device), return_dict=True)
            w = out['wind_estimate'].detach().cpu().numpy()
        preds[i:e] = w[:, :3] * ws + wm
        i = e
    return preds


def windows_from_feat(feat):
    n = feat.shape[0] - SEQ_LEN
    return max(n, 0)


def resample_rows(df, src_rate, dst_rate):
    """按 dst/src 比例抽取行，模拟环路只跑到 dst_rate 时看到的稀疏流。"""
    n = len(df)
    step = src_rate / dst_rate           # >1
    idx = np.floor(np.arange(0, n, step)).astype(int)
    idx = idx[idx < n]
    return df.iloc[idx].reset_index(drop=True), idx


def rmse(pred, truth):
    per = np.sqrt(np.mean((pred - truth) ** 2, axis=0))
    return float(np.mean(per)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["test_id", "test_ood"], default="test_id")
    ap.add_argument("--max_files", type=int, default=8)
    ap.add_argument("--live_rate", type=float, default=29.3)
    args = ap.parse_args()

    preproc = _load_preproc()
    build = preproc.build_features_labels_from_csv
    backend, sx, wm, ws = load_model()

    files = sorted(glob.glob(str(_ROOT / "data" / "data_csv" / args.dataset / "*.csv")))[:args.max_files]
    print(f"[data] {args.dataset}: {len(files)} files | live_rate={args.live_rate}Hz\n")

    acc = {k: {"p": [], "t": []} for k in ["A_base", "B_rate", "C_acceldt", "D_zerotgt"]}

    for fp in files:
        X_ref, y, _, _ = build(fp, seq_len=SEQ_LEN, downsample_factor=DOWNSAMPLE)
        if X_ref.shape[0] == 0:
            continue
        cleaned = _vmod.clean_df_like_preproc(fp, preproc, downsample=DOWNSAMPLE)
        feat_ref = _vmod.reconstruct_feat_from_windows(X_ref)
        if len(cleaned) != feat_ref.shape[0]:
            continue
        truth = y[:, 0:3]
        n_win = X_ref.shape[0]

        # A. baseline: 50Hz 在线特征
        featA = build_online_feat_from_df(cleaned, sampling_rate=TRAIN_RATE)
        acc["A_base"]["p"].append(predict(backend, sx, wm, ws, featA, n_win))
        acc["A_base"]["t"].append(truth)

        # B. rate 29Hz: 重采样流 + 提取器仍按50Hz算加速度 → 构窗预测
        #    真值取每个29Hz窗口末端对应的原50Hz行的 truth
        cl29, idx29 = resample_rows(cleaned, TRAIN_RATE, args.live_rate)
        featB = build_online_feat_from_df(cl29, sampling_rate=TRAIN_RATE)  # dt=1/50 (bug保留)
        nB = windows_from_feat(featB)
        if nB > 0:
            predB = predict(backend, sx, wm, ws, featB, nB)
            # 每个窗口末端在原始行中的下标 → 对齐 truth（truth 索引 = 原始行 - SEQ_LEN）
            end_orig = idx29[SEQ_LEN:SEQ_LEN + nB]
            tji = end_orig - SEQ_LEN
            valid = (tji >= 0) & (tji < len(truth))
            acc["B_rate"]["p"].append(predB[valid])
            acc["B_rate"]["t"].append(truth[tji[valid]])

        # C. accel_dt_only: 仅把加速度按 (50/live_rate) 放大（复刻dt错误），其余不变
        featC = featA.copy()
        featC[:, 6:9] *= (TRAIN_RATE / args.live_rate)
        acc["C_acceldt"]["p"].append(predict(backend, sx, wm, ws, featC, n_win))
        acc["C_acceldt"]["t"].append(truth)

        # D. zero_targets: 期望地速置0 → v*_err = vel（隔离期望地速缺失）
        featD = featA.copy()
        for ti, ei, vi in [(32, 35, 0), (33, 36, 1), (34, 37, 2)]:
            featD[:, ti] = 0.0
            featD[:, ei] = featD[:, vi]  # err = vel - 0 = vel
        acc["D_zerotgt"]["p"].append(predict(backend, sx, wm, ws, featD, n_win))
        acc["D_zerotgt"]["t"].append(truth)

        print(f"  [ok] {os.path.basename(fp):24s} win50={n_win} win29={nB}")

    print("\n" + "=" * 70)
    print(f" 在线 RMSE 归因 — {args.dataset}")
    print("=" * 70)
    labels = {
        "A_base": "A 基线(50Hz真实特征)",
        "B_rate": f"B 采样率{args.live_rate}Hz(时序拉伸+加速度dt错)",
        "C_acceldt": "C 仅加速度dt缩放错",
        "D_zerotgt": "D 期望地速缺失(置0)",
    }
    for k in ["A_base", "B_rate", "C_acceldt", "D_zerotgt"]:
        if not acc[k]["p"]:
            continue
        p = np.concatenate(acc[k]["p"])
        t = np.concatenate(acc[k]["t"])
        m, per = rmse(p, t)
        magp = np.linalg.norm(p[:, :2], axis=1).mean()
        magt = np.linalg.norm(t[:, :2], axis=1).mean()
        print(f"  [{labels[k]:32s}] RMSE={m:.3f}  (N={per[0]:.2f} E={per[1]:.2f} D={per[2]:.2f})  "
              f"|est|={magp:.2f} |gt|={magt:.2f}  n={len(t)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
