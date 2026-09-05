"""
verify_online_45d_chain.py — 无硬件验证：45 维在线特征链路能否复现离线精度
=============================================================================

动机
----
HITL 精度崩溃的根因是当年在线部署跑的是 20 维旧模型 + 20 维特征提取，而论文/重播
用的是 45 维模型。本脚本在 **不需要任何硬件、也不需要启动 PX4 SITL** 的前提下，验证
新的 45 维在线特征装配 (`online_feature_extractor.StreamingFeatureExtractor`) 能否
用与训练一致的原子信号复现离线精度。

方法
----
对每个 test 集 CSV：
  1. 参照真值特征：调用训练侧 `1_preprocessing_data.build_features_labels_from_csv`
     得到 X_ref (n,100,45) 和 y (n,7, 其中 [:,0:3] 为 m/s 风真值)，并从 X_ref 重建
     逐帧 feat_ref。
  2. 在线特征：用与预处理完全一致的清洗流程得到 cleaned_df，逐行喂给流式提取器，得到
     feat_online（因果）。
  3. 断言 len 对齐；逐特征比较 feat_online vs feat_ref（定位任何非因果加速度以外的偏差）。
  4. 用同一 45 维 PI-GRU（config_sitl.yaml 指定的权重）对三种特征做批量前向 + 反归一化，
     计算风速 RMSE：
       - ref            : 参照特征（应≈离线基准 0.219/0.580）
       - online         : 在线因果特征（真实在线可达精度）
       - online+ref-acc : 在线特征但把加速度 6-8 换成参照值（隔离"因果加速度"代价）

用法
----
  .venv/bin/python src/experiments/verify_online_45d_chain.py --dataset test_id --max_files 12
  .venv/bin/python src/experiments/verify_online_45d_chain.py --dataset test_ood --max_files 12
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

from online_feature_extractor import (  # noqa: E402
    FEATURE_IDX,
    SIGNAL_KEYS,
    StreamingFeatureExtractor,
)

SEQ_LEN = 100
DOWNSAMPLE = 5
FEATURE_NAMES = [name for name, _ in sorted(FEATURE_IDX.items(), key=lambda kv: kv[1])]

# CSV 列 -> 原子信号名（rename 后的列名）
COL2SIG = {
    "velocity_north": "vel_n", "velocity_east": "vel_e", "velocity_down": "vel_d",
    "roll": "roll", "pitch": "pitch", "yaw": "yaw",
    "roll_rate": "roll_rate", "pitch_rate": "pitch_rate", "yaw_rate": "yaw_rate",
    "aileron_cmd": "aileron_cmd", "elevator_cmd": "elevator_cmd",
    "rudder_cmd": "rudder_cmd", "throttle_cmd": "throttle_cmd",
    "airspeed": "airspeed",
    "target_roll": "target_roll", "target_pitch": "target_pitch", "target_yaw": "target_yaw",
    "target_p": "target_p", "target_q": "target_q", "target_r": "target_r",
    "target_vn": "target_vn", "target_ve": "target_ve", "target_vd": "target_vd",
    "aileron_actual": "aileron_actual", "elevator_actual": "elevator_actual",
    "rudder_actual": "rudder_actual", "throttle_actual": "throttle_actual",
    "imu_ax": "imu_ax", "imu_ay": "imu_ay", "imu_az": "imu_az",
}


def _load_preproc_module():
    path = _SRC / "1_preprocessing_data.py"
    spec = importlib.util.spec_from_file_location("preproc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clean_df_like_preproc(csv_path, preproc, downsample=DOWNSAMPLE):
    """与 build_features_labels_from_csv 完全一致的清洗流程，返回 cleaned df。

    直接复用预处理模块内部逻辑（rename map + fps 转换 + 去退化段 + dropna），
    以保证行对齐与参照特征一致。
    """
    df = pd.read_csv(csv_path)
    if downsample and downsample > 1:
        df = df.iloc[::int(downsample)].reset_index(drop=True)
    col_map = {}
    rename_map = {
        '/fdm/jsbsim/atmosphere/wind-north-fps': 'wind_north',
        '/fdm/jsbsim/atmosphere/wind-east-fps': 'wind_east',
        '/fdm/jsbsim/atmosphere/wind-down-fps': 'wind_down',
        '/fdm/jsbsim/velocities/v-north-fps': 'velocity_north',
        '/fdm/jsbsim/velocities/v-east-fps': 'velocity_east',
        '/fdm/jsbsim/velocities/v-down-fps': 'velocity_down',
        '/fdm/jsbsim/velocities/vtrue-fps': 'airspeed',
        '/fdm/jsbsim/attitude/pitch-rad': 'pitch',
        '/fdm/jsbsim/attitude/roll-rad': 'roll',
        '/fdm/jsbsim/attitude/psi-rad': 'yaw',
        '/fdm/jsbsim/velocities/p-rad_sec': 'roll_rate',
        '/fdm/jsbsim/velocities/q-rad_sec': 'pitch_rate',
        '/fdm/jsbsim/velocities/r-rad_sec': 'yaw_rate',
        '/fdm/jsbsim/fcs/aileron-cmd-norm': 'aileron_cmd',
        '/fdm/jsbsim/fcs/elevator-cmd-norm': 'elevator_cmd',
        '/fdm/jsbsim/fcs/throttle-cmd-norm': 'throttle_cmd',
        '/fdm/jsbsim/fcs/rudder-cmd-norm': 'rudder_cmd',
        'target_roll_rad': 'target_roll',
        'target_pitch_rad': 'target_pitch',
        'target_yaw_rad': 'target_yaw',
        'target_p_rad_s': 'target_p',
        'target_q_rad_s': 'target_q',
        'target_r_rad_s': 'target_r',
    }
    for old, new in rename_map.items():
        if old in df.columns:
            col_map[old] = new
    df = df.rename(columns=col_map)

    fps_cols = ['wind_north', 'wind_east', 'wind_down',
                'velocity_north', 'velocity_east', 'velocity_down', 'airspeed']
    for c in fps_cols:
        if c in df.columns:
            df[c] = df[c] * 0.3048

    df = df.replace([np.inf, -np.inf], np.nan)
    valid_mask = ~df.isna().any(axis=1)
    df = df[valid_mask].reset_index(drop=True)
    return df


def reconstruct_feat_from_windows(X_ref):
    """从窗口张量 X_ref (n,seq,45) 重建逐帧特征 feat (T,45)，T=n+seq。"""
    n = X_ref.shape[0]
    T = n + SEQ_LEN
    feat = np.zeros((T, X_ref.shape[2]), dtype=np.float32)
    feat[:SEQ_LEN] = X_ref[0]
    for i in range(n):
        feat[SEQ_LEN - 1 + i] = X_ref[i, -1]
    feat[T - 1] = feat[T - 2]
    return feat


def build_online_feat(cleaned_df):
    """用流式提取器把 cleaned_df 逐行装配成 (T,45) 因果特征。"""
    T = len(cleaned_df)
    cols = {c: cleaned_df[c].to_numpy(dtype=np.float64)
            for c in COL2SIG if c in cleaned_df.columns}
    ext = StreamingFeatureExtractor(sampling_rate=50.0)
    feat = np.zeros((T, 45), dtype=np.float32)
    for t in range(T):
        sig = {}
        for col, sname in COL2SIG.items():
            if col in cols:
                sig[sname] = cols[col][t]
        feat[t] = ext.push(sig)
    return feat


def load_model_and_norm():
    cfg_path = _ROOT / "config" / "config_sitl.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    from inference_backends import create_backend
    backend = create_backend(cfg)
    info = backend.load()

    msp = cfg['training']['model_save_path']
    norm_path = Path(msp) / "norm_params.pkl" if os.path.isabs(msp) \
        else _ROOT / msp / "norm_params.pkl"
    with open(norm_path, 'rb') as f:
        norm = pickle.load(f)
    scaler_X = norm['scaler_X']
    scaler_y = norm['scaler_y']
    wind_mean = np.asarray(scaler_y.mean_, dtype=np.float32)[:3]
    wind_std = np.asarray(scaler_y.scale_, dtype=np.float32)[:3]
    print(f"[model] {info.get('model_path')}")
    print(f"[model] input_dim={scaler_X.n_features_in_} epoch={info.get('checkpoint_epoch')}")
    return backend, scaler_X, wind_mean, wind_std


def predict_windows(backend, scaler_X, wind_mean, wind_std, feat, n_windows, batch=4096):
    """把逐帧特征 feat 切成 n_windows 个窗口，批量前向 + 反归一化，返回 (n,3) 风速估计。"""
    import torch
    feat_norm = scaler_X.transform(feat.astype(np.float32)).astype(np.float32)
    model = backend.model
    device = backend.device
    preds = np.zeros((n_windows, 3), dtype=np.float32)
    # 用滑窗视图批量构造
    idx = 0
    while idx < n_windows:
        end = min(idx + batch, n_windows)
        wins = np.stack([feat_norm[i:i + SEQ_LEN] for i in range(idx, end)])
        with torch.no_grad():
            x = torch.from_numpy(wins).to(device)
            out = model(x, return_dict=True)
            w = out['wind_estimate'].detach().cpu().numpy()
        preds[idx:end] = w[:, :3] * wind_std + wind_mean
        idx = end
    return preds


def rmse_report(preds, truth, tag):
    err = preds - truth
    per = np.sqrt(np.mean(err ** 2, axis=0))
    mean_rmse = float(np.mean(per))
    print(f"  [{tag:16s}] RMSE  N={per[0]:.3f}  E={per[1]:.3f}  D={per[2]:.3f}  "
          f"mean={mean_rmse:.3f} m/s  (n={len(truth)})")
    return mean_rmse, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["test_id", "test_ood"], default="test_id")
    ap.add_argument("--max_files", type=int, default=12)
    args = ap.parse_args()

    preproc = _load_preproc_module()
    build = preproc.build_features_labels_from_csv

    backend, scaler_X, wind_mean, wind_std = load_model_and_norm()

    csv_dir = _ROOT / "data" / "data_csv" / args.dataset
    files = sorted(glob.glob(str(csv_dir / "*.csv")))[:args.max_files]
    print(f"[data] {args.dataset}: using {len(files)} / "
          f"{len(sorted(glob.glob(str(csv_dir / '*.csv'))))} files\n")

    all_truth, all_ref, all_online, all_online_refacc = [], [], [], []
    feat_absdiff_sum = np.zeros(45)
    feat_absdiff_max = np.zeros(45)
    n_rows_diff = 0

    for fp in files:
        X_ref, y, _, _ = build(fp, seq_len=SEQ_LEN, downsample_factor=DOWNSAMPLE)
        if X_ref.shape[0] == 0:
            print(f"  [skip] {os.path.basename(fp)} (empty after cleaning)")
            continue
        feat_ref = reconstruct_feat_from_windows(X_ref)
        cleaned = clean_df_like_preproc(fp, preproc, downsample=DOWNSAMPLE)
        if len(cleaned) != feat_ref.shape[0]:
            print(f"  [skip] {os.path.basename(fp)} row mismatch "
                  f"clean={len(cleaned)} ref={feat_ref.shape[0]}")
            continue
        feat_online = build_online_feat(cleaned)

        # 逐特征偏差（跳过前 SEQ_LEN 行以避开因果加速度预热）
        d = np.abs(feat_online[SEQ_LEN:] - feat_ref[SEQ_LEN:])
        feat_absdiff_sum += d.sum(axis=0)
        feat_absdiff_max = np.maximum(feat_absdiff_max, d.max(axis=0))
        n_rows_diff += d.shape[0]

        feat_online_refacc = feat_online.copy()
        feat_online_refacc[:, 6:9] = feat_ref[:, 6:9]

        n_win = X_ref.shape[0]
        truth = y[:, 0:3]
        all_truth.append(truth)
        all_ref.append(predict_windows(backend, scaler_X, wind_mean, wind_std, feat_ref, n_win))
        all_online.append(predict_windows(backend, scaler_X, wind_mean, wind_std, feat_online, n_win))
        all_online_refacc.append(
            predict_windows(backend, scaler_X, wind_mean, wind_std, feat_online_refacc, n_win))
        print(f"  [ok] {os.path.basename(fp):22s} windows={n_win}")

    truth = np.concatenate(all_truth)
    ref = np.concatenate(all_ref)
    online = np.concatenate(all_online)
    online_refacc = np.concatenate(all_online_refacc)

    print("\n" + "=" * 74)
    print(f" 45 维在线链路精度验证 — {args.dataset}  (合并 {len(truth)} 个窗口)")
    print("=" * 74)
    rmse_report(ref, truth, "reference")
    rmse_report(online_refacc, truth, "online+ref-acc")
    rmse_report(online, truth, "online(causal)")
    baseline = 0.219 if args.dataset == "test_id" else 0.580
    print(f"  [baseline paper ] mean RMSE ~= {baseline:.3f} m/s")

    print("\n 逐特征在线-参照最大绝对偏差 (Top-8，排除加速度预热):")
    mean_abs = feat_absdiff_sum / max(n_rows_diff, 1)
    order = np.argsort(feat_absdiff_max)[::-1][:8]
    for j in order:
        print(f"   {FEATURE_NAMES[j]:14s} (idx {j:2d})  max={feat_absdiff_max[j]:.4e}  "
              f"mean={mean_abs[j]:.4e}")
    print("=" * 74)


if __name__ == "__main__":
    main()
