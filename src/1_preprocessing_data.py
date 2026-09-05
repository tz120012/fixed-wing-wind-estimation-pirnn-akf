#!/usr/bin/env python3
"""
数据预处理脚本（PX4-SITL + JSBSim 新数据集）

关键功能：
  1. 读取 postprocess_to_jsbsim_csv.py 转换后的 CSV（标准列名格式）
  2. 速度三角形一致性检查（|vtrue - ||V_gnd - wind||| < 阈值），
     过滤物理不一致文件
  3. airspeed 退化检测（空速 ≈ 地速的污染段自动跳过）
  4. 根据 gust_phase / gust_factor / wind_regime 计算 sample weight（w_*.npy），
     供 3_train_pigru.py 的动态段加权训练使用
  5. within_run_temporal split 策略，保持每 run 内的时序完整性

使用方式：
  python src/1_preprocessing_data.py
  python src/1_preprocessing_data.py --data_root data/data_1 --outdir data/dataset_new_processed
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ─────────────────────────────────────────────
# 特征下标常量（阶段 1+2，全 45 维布局）
#
# 设计原则：
#   - 0-19 保持与原 20 维布局完全兼容，旧 checkpoint/AKF 逻辑无需改动
#   - 20-31 阶段 1 新增（目标姿态 + 姿态误差 + 目标角速度 + 角速度误差）
#   - 32-44 阶段 2 新增（目标速度 + 速度误差 + 实际舵面 + IMU 加速度）
#
# 训练/推理代码请通过 FEATURE_IDX 常量切片，避免硬编码 9:12 / 0:3 等导致改维度错位。
# ─────────────────────────────────────────────
FEATURE_IDX = {
    "vel_n": 0, "vel_e": 1, "vel_d": 2,
    "vx_body": 3, "vy_body": 4, "vz_body": 5,
    "ax": 6, "ay": 7, "az": 8,
    "roll": 9, "pitch": 10, "yaw": 11,
    "p_rate": 12, "q_rate": 13, "r_rate": 14,
    "aileron_cmd": 15, "elevator_cmd": 16, "rudder_cmd": 17,
    "throttle_cmd": 18, "airspeed": 19,
    "target_roll": 20, "target_pitch": 21, "target_yaw": 22,
    "roll_err": 23, "pitch_err": 24, "yaw_err": 25,
    "target_p": 26, "target_q": 27, "target_r": 28,
    "p_err": 29, "q_err": 30, "r_err": 31,
    "target_vn": 32, "target_ve": 33, "target_vd": 34,
    "vn_err": 35, "ve_err": 36, "vd_err": 37,
    "aileron_act": 38, "elevator_act": 39, "rudder_act": 40,
    "throttle_act": 41,
    "imu_ax": 42, "imu_ay": 43, "imu_az": 44,
}

FEATURE_DIM_LEGACY = 20      # 旧版（仅基础态）
FEATURE_DIM_STAGE1 = 32      # 阶段 1：+ 目标姿态/姿态误差/目标角速度/角速度误差
FEATURE_DIM_STAGE2 = 45      # 阶段 2：+ 目标速度/速度误差/实际舵面/IMU 加速度

# 当前活动特征维度（阶段 1 =32；阶段 2 时改为 45）
FEATURE_DIM = FEATURE_DIM_STAGE2


def _wrap_angle(a):
    """把角度归一到 (-π, π]。numpy 兼容。"""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def sort_csv_paths(csv_paths):
    """按 run/segment 编号排序 CSV。"""
    def sort_key(path):
        name = os.path.basename(path)
        m = re.search(r'datasets-(\d+)-(\d+)', name)
        if m:
            return (0, int(m.group(1)), int(m.group(2)), name)
        return (1, 0, 0, name)
    return sorted(csv_paths, key=sort_key)


def check_velocity_triangle(csv_path, threshold_fps=3.0):
    """
    检查单个 CSV 文件的速度三角形一致性。

    物理约束: TAS = ||V_gnd - V_wind||
    如果 |vtrue - ||V_gnd - wind||| 的均值超过 threshold_fps，
    说明风标签与输入特征不一致，应跳过此文件。

    Returns:
        (pass: bool, mean_error_fps: float, info: str)
    """
    try:
        cols = [
            '/fdm/jsbsim/velocities/vtrue-fps',
            '/fdm/jsbsim/velocities/v-north-fps',
            '/fdm/jsbsim/velocities/v-east-fps',
            '/fdm/jsbsim/velocities/v-down-fps',
            '/fdm/jsbsim/atmosphere/wind-north-fps',
            '/fdm/jsbsim/atmosphere/wind-east-fps',
            '/fdm/jsbsim/atmosphere/wind-down-fps',
        ]
        df = pd.read_csv(csv_path, usecols=cols)
        vtrue = df[cols[0]].values
        vn, ve, vd = df[cols[1]].values, df[cols[2]].values, df[cols[3]].values
        wn, we, wd = df[cols[4]].values, df[cols[5]].values, df[cols[6]].values

        valid = np.isfinite(vtrue) & np.isfinite(vn) & np.isfinite(wn) & (vtrue > 1.0)
        if valid.sum() < 50:
            return False, float('inf'), "too few valid rows"

        # 跳过前 10% 数据（起飞/过渡阶段可能不稳定）
        start = int(valid.sum() * 0.1)
        idx = np.where(valid)[0][start:]

        air_mag = np.sqrt(
            (vn[idx] - wn[idx])**2 +
            (ve[idx] - we[idx])**2 +
            (vd[idx] - wd[idx])**2
        )
        diff = np.abs(vtrue[idx] - air_mag)
        mean_err = float(np.mean(diff))
        median_err = float(np.median(diff))

        passed = median_err < threshold_fps
        info = f"median_err={median_err:.2f} fps ({median_err*0.3048:.2f} m/s)"
        return passed, median_err, info
    except Exception as e:
        return False, float('inf'), f"error: {e}"


def read_csv_wind_stats(csv_path):
    """轻量读取 CSV 的风速统计。"""
    cols = [
        '/fdm/jsbsim/atmosphere/wind-north-fps',
        '/fdm/jsbsim/atmosphere/wind-east-fps',
        '/fdm/jsbsim/atmosphere/wind-down-fps',
    ]
    try:
        df = pd.read_csv(csv_path, usecols=cols)
        wn = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy() * 0.3048
        we = pd.to_numeric(df.iloc[:, 1], errors='coerce').to_numpy() * 0.3048
        wd = pd.to_numeric(df.iloc[:, 2], errors='coerce').to_numpy() * 0.3048
        valid = np.isfinite(wn) & np.isfinite(we) & np.isfinite(wd)
        if valid.sum() == 0:
            return None
        w = np.stack([wn[valid], we[valid], wd[valid]], axis=1)
        mag = np.linalg.norm(w, axis=1)
        return {
            'mag_mean': float(mag.mean()),
            'mag_p90': float(np.percentile(mag, 90)),
            'mag_max': float(mag.max()),
            'mean_n': float(w[:, 0].mean()),
            'mean_e': float(w[:, 1].mean()),
        }
    except Exception:
        return None


_DEFAULT_SAMPLE_WEIGHT_CONFIG = {
    'enabled': True,
    'steady': 1.0,
    'dynamic_baseline': 1.5,
    'hold': 2.0,
    'transition': 5.0,          # 阵风过渡期（rise/fall）从 3.0→5.0，加强模型对瞬态跟踪的训练信号
    'factor_boost': 0.5,
    'gust_factor_threshold': 0.05,
    'turn_discount': 0.4,        # 转弯段样本权重折扣：速度三角形受向心力/yaw变化污染，标签噪声高
}


def _compute_per_row_sample_weights(df: pd.DataFrame, weight_config: dict) -> np.ndarray:
    """根据 wind_regime / gust_phase / gust_factor 等动态风诊断字段，
    给 CSV 中每一行计算 sample weight。

    规则（默认配置）：
      - steady（pre/post，wind_regime=='steady' 或字段缺失）   -> 1.0
      - dynamic_baseline（动态 regime 但相位不在 rise/fall/hold） -> 1.5
      - hold（gust_phase == 'hold'，阵风维持段）              -> 2.0
      - transition（gust_phase ∈ {'rise','fall'}，阵风过渡段）  -> 3.0
      - 进一步乘以 (1 + factor_boost * |gust_factor|)，让阵风峰值样本权重更高
    """
    cfg = {**_DEFAULT_SAMPLE_WEIGHT_CONFIG, **(weight_config or {})}
    n_rows = len(df)
    if n_rows == 0:
        return np.zeros(0, dtype=np.float32)

    if not bool(cfg.get('enabled', True)):
        return np.ones(n_rows, dtype=np.float32)

    weights = np.full(n_rows, float(cfg.get('steady', 1.0)), dtype=np.float32)

    has_regime = 'wind_regime' in df.columns
    has_phase = 'gust_phase' in df.columns
    has_factor = 'gust_factor' in df.columns

    if not (has_regime or has_phase or has_factor):
        return weights

    if has_regime:
        regime = df['wind_regime'].astype(str).fillna('unknown').str.lower().values
        is_dynamic = ~np.isin(regime, ('steady', 'unknown', 'nan', ''))
        weights = np.where(is_dynamic,
                           np.maximum(weights, float(cfg.get('dynamic_baseline', 1.5))),
                           weights)

    if has_phase:
        phase = df['gust_phase'].astype(str).fillna('unknown').str.lower().values
        weights = np.where(np.isin(phase, ('hold',)),
                           np.maximum(weights, float(cfg.get('hold', 2.0))),
                           weights)
        weights = np.where(np.isin(phase, ('rise', 'fall')),
                           np.maximum(weights, float(cfg.get('transition', 3.0))),
                           weights)

    if has_factor:
        gf = pd.to_numeric(df['gust_factor'], errors='coerce').fillna(0.0).abs().to_numpy()
        gf = np.clip(gf, 0.0, 1.0).astype(np.float32)
        threshold = float(cfg.get('gust_factor_threshold', 0.05))
        boost = float(cfg.get('factor_boost', 0.5))
        # 仅在 gust_factor 显著时再叠加放大系数，避免稳态样本被错误加权
        active = gf >= threshold
        weights = np.where(active, weights * (1.0 + boost * gf), weights)

    if 'label_quality_weight' in df.columns:
        quality_w = pd.to_numeric(df['label_quality_weight'], errors='coerce').fillna(1.0).to_numpy(dtype=np.float32)
        quality_w = np.clip(quality_w, 0.0, 1.0)
        weights = weights * quality_w

    # 转弯段折扣：向心加速度和快速 yaw 变化污染速度三角形，标签噪声高，
    # 降低其对梯度的贡献，避免模型在噪声标签上 overfit 而产生均值回归。
    if 'turn_class' in df.columns:
        turn_discount = float(cfg.get('turn_discount', 1.0))
        if turn_discount < 1.0:
            turn_col = pd.to_numeric(df['turn_class'], errors='coerce').fillna(-1).to_numpy(dtype=np.int8)
            is_turning = turn_col == 1
            weights = np.where(is_turning, weights * turn_discount, weights)

    return weights.astype(np.float32)


def build_features_labels_from_csv(csv_path, seq_len, weight_config=None, sampling_rate=50,
                                   downsample_factor=1):
    """从 CSV 构建特征、标签和 sample weight 序列。

    Args:
        downsample_factor: 原始 CSV 行下采样因子。新数据集采集 ~240 Hz、训练/部署
            50 Hz，应设 5（240→50）以保证训练-部署频率一致；旧数据集 50 Hz 设 1。

    Returns:
        (X, y, w, turn_cls):
            X.shape=(n, seq_len, FEATURE_DIM), y.shape=(n,7), w.shape=(n,) float32,
            turn_cls.shape=(n,) int8（0=非转弯, 1=转弯, -1=未知）
            FEATURE_DIM 由模块级常量决定：阶段 1=32，阶段 2=45。
    """
    df = pd.read_csv(csv_path)
    if downsample_factor and downsample_factor > 1:
        df = df.iloc[::int(downsample_factor)].reset_index(drop=True)
    col_map = {}
    for old, new in {
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
        # 阶段 1：PX4 目标量（已是 rad/rad·s 单位，无需缩放）
        'target_roll_rad': 'target_roll',
        'target_pitch_rad': 'target_pitch',
        'target_yaw_rad': 'target_yaw',
        'target_p_rad_s': 'target_p',
        'target_q_rad_s': 'target_q',
        'target_r_rad_s': 'target_r',
    }.items():
        if old in df.columns:
            col_map[old] = new
    df = df.rename(columns=col_map)

    fps_cols = ['wind_north', 'wind_east', 'wind_down',
                'velocity_north', 'velocity_east', 'velocity_down', 'airspeed']
    for c in fps_cols:
        if c in df.columns:
            df[c] = df[c] * 0.3048

    # 质量门槛: airspeed 退化为 groundspeed 的段（PX4 airspeed_selector NaN 后被前向填充
    # 兜底为地速）必须剔除，否则模型学不到风对空速的影响。判据: 段内 |airspeed - |v_horizontal||
    # 中位数 < 0.05 m/s 且 ≥70% 样本几乎完全相等 → 退化段。
    if {'airspeed', 'velocity_north', 'velocity_east'}.issubset(df.columns):
        gs = np.hypot(df['velocity_north'].values, df['velocity_east'].values)
        a_arr = df['airspeed'].values
        diff = np.abs(a_arr - gs)
        finite = np.isfinite(diff)
        if finite.sum() >= 100:
            zero_pct = float(np.mean(diff[finite] < 0.001))
            med_diff = float(np.median(diff[finite]))
            if zero_pct > 0.7 and med_diff < 0.05:
                print(f"  [跳过] {csv_path} airspeed 退化为地速 "
                      f"(zero_pct={zero_pct:.1%}, med_diff={med_diff:.4f})")
                empty_X = np.zeros((0, seq_len, FEATURE_DIM), dtype=np.float32)
                empty_y = np.zeros((0, 7), dtype=np.float32)
                empty_w = np.zeros((0,), dtype=np.float32)
                empty_turn = np.zeros((0,), dtype=np.int8)
                return empty_X, empty_y, empty_w, empty_turn

    # 转弯标签：优先使用 turn_class；若缺失则由 turn_state 推断
    if 'turn_class' in df.columns:
        raw_turn = pd.to_numeric(df['turn_class'], errors='coerce').fillna(-1).to_numpy(dtype=np.int8)
    elif 'turn_state' in df.columns:
        state = df['turn_state'].astype(str).str.lower().fillna('unknown')
        raw_turn = np.where(state == 'turning', 1, np.where(state == 'non_turning', 0, -1)).astype(np.int8)
    else:
        raw_turn = np.full(len(df), -1, dtype=np.int8)

    # NOTE: 必须先在原始 df 上计算 sample weight，后续 dropna 可能丢行。
    # 由于 dropna 仅会针对数值列触发（gust_phase/wind_regime 是字符串），
    # 行索引经过 dropna 后仍与原 df 对应，重置 index 后即可对齐。
    raw_weights = _compute_per_row_sample_weights(df, weight_config)

    df = df.replace([np.inf, -np.inf], np.nan)
    valid_mask = ~df.isna().any(axis=1)
    raw_weights = raw_weights[valid_mask.values]
    raw_turn = raw_turn[valid_mask.values]
    df = df[valid_mask].reset_index(drop=True)

    vel_n = df['velocity_north'].values
    vel_e = df['velocity_east'].values
    vel_d = df['velocity_down'].values
    roll = df['roll'].values
    pitch = df['pitch'].values
    yaw = df['yaw'].values

    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)

    vx = cp * cy * vel_n + cp * sy * vel_e - sp * vel_d
    vy = (sr * sp * cy - cr * sy) * vel_n + (sr * sp * sy + cr * cy) * vel_e + sr * cp * vel_d
    vz = (cr * sp * cy + sr * sy) * vel_n + (cr * sp * sy - sr * cy) * vel_e + cr * cp * vel_d

    dt = 1.0 / sampling_rate          # 与采集端/部署端采样率保持一致
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    az = np.gradient(vz, dt)
    win = min(51, max(5, (len(df) // 10) | 1))
    if win >= 5 and len(df) > win:
        ax = savgol_filter(ax, win, 3)
        ay = savgol_filter(ay, win, 3)
        az = savgol_filter(az, win, 3)

    feat = np.zeros((len(df), FEATURE_DIM), dtype=np.float32)
    # ---------- 0-19：原 20 维基础态（保持不变）----------
    feat[:, FEATURE_IDX["vel_n"]] = np.nan_to_num(vel_n, nan=0.0)
    feat[:, FEATURE_IDX["vel_e"]] = np.nan_to_num(vel_e, nan=0.0)
    feat[:, FEATURE_IDX["vel_d"]] = np.nan_to_num(vel_d, nan=0.0)
    feat[:, FEATURE_IDX["vx_body"]] = np.nan_to_num(vx, nan=0.0)
    feat[:, FEATURE_IDX["vy_body"]] = np.nan_to_num(vy, nan=0.0)
    feat[:, FEATURE_IDX["vz_body"]] = np.nan_to_num(vz, nan=0.0)
    feat[:, FEATURE_IDX["ax"]] = np.nan_to_num(ax, nan=0.0)
    feat[:, FEATURE_IDX["ay"]] = np.nan_to_num(ay, nan=0.0)
    feat[:, FEATURE_IDX["az"]] = np.nan_to_num(az, nan=0.0)
    feat[:, FEATURE_IDX["roll"]] = np.nan_to_num(roll, nan=0.0)
    feat[:, FEATURE_IDX["pitch"]] = np.nan_to_num(pitch, nan=0.0)
    feat[:, FEATURE_IDX["yaw"]] = np.nan_to_num(yaw, nan=0.0)
    p_rate_arr = df['roll_rate'].values if 'roll_rate' in df.columns else np.zeros(len(df))
    q_rate_arr = df['pitch_rate'].values if 'pitch_rate' in df.columns else np.zeros(len(df))
    r_rate_arr = df['yaw_rate'].values if 'yaw_rate' in df.columns else np.zeros(len(df))
    feat[:, FEATURE_IDX["p_rate"]] = np.nan_to_num(p_rate_arr, nan=0.0)
    feat[:, FEATURE_IDX["q_rate"]] = np.nan_to_num(q_rate_arr, nan=0.0)
    feat[:, FEATURE_IDX["r_rate"]] = np.nan_to_num(r_rate_arr, nan=0.0)
    feat[:, FEATURE_IDX["aileron_cmd"]] = np.nan_to_num(df['aileron_cmd'].values if 'aileron_cmd' in df.columns else 0, nan=0.0)
    feat[:, FEATURE_IDX["elevator_cmd"]] = np.nan_to_num(df['elevator_cmd'].values if 'elevator_cmd' in df.columns else 0, nan=0.0)
    feat[:, FEATURE_IDX["rudder_cmd"]] = np.nan_to_num(df['rudder_cmd'].values if 'rudder_cmd' in df.columns else 0, nan=0.0)
    feat[:, FEATURE_IDX["throttle_cmd"]] = np.nan_to_num(df['throttle_cmd'].values if 'throttle_cmd' in df.columns else 0.5, nan=0.5)
    feat[:, FEATURE_IDX["airspeed"]] = np.nan_to_num(df['airspeed'].values if 'airspeed' in df.columns else 15.0, nan=15.0)

    # ---------- 阶段 1：20-31 PX4 目标量与误差 ----------
    if FEATURE_DIM >= FEATURE_DIM_STAGE1:
        target_roll_arr = df['target_roll'].values if 'target_roll' in df.columns else np.zeros(len(df))
        target_pitch_arr = df['target_pitch'].values if 'target_pitch' in df.columns else np.zeros(len(df))
        target_yaw_arr = df['target_yaw'].values if 'target_yaw' in df.columns else np.zeros(len(df))
        target_p_arr = df['target_p'].values if 'target_p' in df.columns else np.zeros(len(df))
        target_q_arr = df['target_q'].values if 'target_q' in df.columns else np.zeros(len(df))
        target_r_arr = df['target_r'].values if 'target_r' in df.columns else np.zeros(len(df))

        feat[:, FEATURE_IDX["target_roll"]] = np.nan_to_num(target_roll_arr, nan=0.0)
        feat[:, FEATURE_IDX["target_pitch"]] = np.nan_to_num(target_pitch_arr, nan=0.0)
        feat[:, FEATURE_IDX["target_yaw"]] = np.nan_to_num(target_yaw_arr, nan=0.0)

        # 姿态误差（actual - target），yaw 必须 wrap 到 (-π, π]
        roll_err = np.nan_to_num(roll - target_roll_arr, nan=0.0)
        pitch_err = np.nan_to_num(pitch - target_pitch_arr, nan=0.0)
        yaw_err = _wrap_angle(np.nan_to_num(yaw - target_yaw_arr, nan=0.0))
        feat[:, FEATURE_IDX["roll_err"]] = roll_err.astype(np.float32)
        feat[:, FEATURE_IDX["pitch_err"]] = pitch_err.astype(np.float32)
        feat[:, FEATURE_IDX["yaw_err"]] = yaw_err.astype(np.float32)

        feat[:, FEATURE_IDX["target_p"]] = np.nan_to_num(target_p_arr, nan=0.0)
        feat[:, FEATURE_IDX["target_q"]] = np.nan_to_num(target_q_arr, nan=0.0)
        feat[:, FEATURE_IDX["target_r"]] = np.nan_to_num(target_r_arr, nan=0.0)

        feat[:, FEATURE_IDX["p_err"]] = np.nan_to_num(p_rate_arr - target_p_arr, nan=0.0)
        feat[:, FEATURE_IDX["q_err"]] = np.nan_to_num(q_rate_arr - target_q_arr, nan=0.0)
        feat[:, FEATURE_IDX["r_err"]] = np.nan_to_num(r_rate_arr - target_r_arr, nan=0.0)

    # ---------- 阶段 2：32-44 目标速度/速度误差/实际舵面/IMU 加速度 ----------
    # 阶段 1 时 FEATURE_DIM=32，下面的填充会被跳过；阶段 2 时按需填充
    if FEATURE_DIM >= FEATURE_DIM_STAGE2:
        target_vn_arr = df['target_vn'].values if 'target_vn' in df.columns else np.zeros(len(df))
        target_ve_arr = df['target_ve'].values if 'target_ve' in df.columns else np.zeros(len(df))
        target_vd_arr = df['target_vd'].values if 'target_vd' in df.columns else np.zeros(len(df))
        feat[:, FEATURE_IDX["target_vn"]] = np.nan_to_num(target_vn_arr, nan=0.0)
        feat[:, FEATURE_IDX["target_ve"]] = np.nan_to_num(target_ve_arr, nan=0.0)
        feat[:, FEATURE_IDX["target_vd"]] = np.nan_to_num(target_vd_arr, nan=0.0)
        feat[:, FEATURE_IDX["vn_err"]] = np.nan_to_num(vel_n - target_vn_arr, nan=0.0)
        feat[:, FEATURE_IDX["ve_err"]] = np.nan_to_num(vel_e - target_ve_arr, nan=0.0)
        feat[:, FEATURE_IDX["vd_err"]] = np.nan_to_num(vel_d - target_vd_arr, nan=0.0)

        feat[:, FEATURE_IDX["aileron_act"]] = np.nan_to_num(df['aileron_actual'].values if 'aileron_actual' in df.columns else 0, nan=0.0)
        feat[:, FEATURE_IDX["elevator_act"]] = np.nan_to_num(df['elevator_actual'].values if 'elevator_actual' in df.columns else 0, nan=0.0)
        feat[:, FEATURE_IDX["rudder_act"]] = np.nan_to_num(df['rudder_actual'].values if 'rudder_actual' in df.columns else 0, nan=0.0)
        feat[:, FEATURE_IDX["throttle_act"]] = np.nan_to_num(df['throttle_actual'].values if 'throttle_actual' in df.columns else 0.5, nan=0.5)

        feat[:, FEATURE_IDX["imu_ax"]] = np.nan_to_num(df['imu_ax'].values if 'imu_ax' in df.columns else 0, nan=0.0)
        feat[:, FEATURE_IDX["imu_ay"]] = np.nan_to_num(df['imu_ay'].values if 'imu_ay' in df.columns else 0, nan=0.0)
        feat[:, FEATURE_IDX["imu_az"]] = np.nan_to_num(df['imu_az'].values if 'imu_az' in df.columns else 0, nan=0.0)

    lbl = np.zeros((len(df), 7), dtype=np.float32)
    lbl[:, 0] = np.nan_to_num(df['wind_north'].values if 'wind_north' in df.columns else 0, nan=0.0)
    lbl[:, 1] = np.nan_to_num(df['wind_east'].values if 'wind_east' in df.columns else 0, nan=0.0)
    lbl[:, 2] = np.nan_to_num(df['wind_down'].values if 'wind_down' in df.columns else 0, nan=0.0)
    lbl[:, 3] = feat[:, FEATURE_IDX["vel_n"]]
    lbl[:, 4] = feat[:, FEATURE_IDX["vel_e"]]
    lbl[:, 5] = feat[:, FEATURE_IDX["vel_d"]]
    lbl[:, 6] = feat[:, FEATURE_IDX["airspeed"]]

    n_seq = len(df) - seq_len
    if n_seq <= 0:
        return (
            np.zeros((0, seq_len, FEATURE_DIM), dtype=np.float32),
            np.zeros((0, 7), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
        )

    X = np.stack([feat[i:i + seq_len] for i in range(n_seq)], dtype=np.float32)
    y = lbl[seq_len:].astype(np.float32)
    # weight 与 y 对齐：第 i 个窗口预测 t=seq_len+i 时刻 → 取该时刻的 raw_weight
    w = raw_weights[seq_len:].astype(np.float32)
    turn_cls = raw_turn[seq_len:].astype(np.int8)
    return X, y, w, turn_cls


# ─────────────────────────────────────────────
# 块级统计与分层
# ─────────────────────────────────────────────

def collect_run_blocks(csv_files):
    """将 CSV 文件按 run 聚合，返回 block 列表。"""
    blocks = {}
    for fp in csv_files:
        name = os.path.basename(fp)
        m = re.search(r'datasets-(\d+)-(\d+)', name)
        run_id = int(m.group(1)) if m else 0
        if run_id not in blocks:
            blocks[run_id] = {'run': run_id, 'files': []}
        blocks[run_id]['files'].append(fp)

    result = []
    for run_id in sorted(blocks.keys()):
        b = blocks[run_id]
        b['files'] = sort_csv_paths(b['files'])
        stats_list = [read_csv_wind_stats(f) for f in b['files']]
        stats_list = [s for s in stats_list if s is not None]
        if not stats_list:
            continue
        n_samples = [s['mag_mean'] * 100 for s in stats_list]
        total_w = sum(n_samples) or 1.0
        wn_total = sum(s['mean_n'] * n for s, n in zip(stats_list, n_samples)) / total_w
        we_total = sum(s['mean_e'] * n for s, n in zip(stats_list, n_samples)) / total_w
        result.append({
            'run': run_id,
            'n_files': len(b['files']),
            'files': b['files'],
            'wind_score': sum(s['mag_p90'] * n for s, n in zip(stats_list, n_samples)) / total_w,
            'wind_mean': sum(s['mag_mean'] * n for s, n in zip(stats_list, n_samples)) / total_w,
            'wind_max': max(s['mag_max'] for s in stats_list),
            'mean_n': wn_total,
            'mean_e': we_total,
            'dir_deg': float((np.degrees(np.arctan2(we_total, wn_total)) + 360.0) % 360.0),
        })
    return result


def split_arrays_within_file_temporal(X, y, w, turn_cls, train_frac=0.70, val_frac=0.15, gap=None):
    """每个 CSV 文件的窗口数组按"行索引时间"切成 train / val / test_id 三段，
    中间留 gap（默认 = seq_len）个窗口防止滑窗在边界处重叠泄漏。

    设计动机：原 assign_within_run_temporal 按 CSV 整文件分配，导致每个 run 的
    "前几个文件=稳态、后几个文件=阵风"被原样拆给 train/val/test，结果 val 几乎
    全是阵风段（dynamic_ratio≈99.6%），train 仅 35%。这里改为每个文件内部按时间
    切，保证 train/val/test 三个 split 在每个文件、每段风况下都看到混合样本。

    返回 dict[name -> (Xs, ys, ws, ts)]，长度不足时对应键为 (None, None, None, None)。
    """
    n = X.shape[0]
    if n < 100:
        return {'train': (X, y, w, turn_cls), 'val': (None, None, None, None), 'test_id': (None, None, None, None)}
    if gap is None:
        gap = X.shape[1]  # = seq_len
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    if n_train + gap >= n or n_train + gap + n_val + gap >= n:
        return {'train': (X, y, w, turn_cls), 'val': (None, None, None, None), 'test_id': (None, None, None, None)}

    train_end = n_train
    val_start = train_end + gap
    val_end = val_start + n_val
    test_start = val_end + gap
    out = {
        'train': (
            X[:train_end],
            y[:train_end],
            w[:train_end] if w is not None else None,
            turn_cls[:train_end] if turn_cls is not None else None,
        ),
        'val': (
            X[val_start:val_end],
            y[val_start:val_end],
            w[val_start:val_end] if w is not None else None,
            turn_cls[val_start:val_end] if turn_cls is not None else None,
        ),
        'test_id': (
            X[test_start:],
            y[test_start:],
            w[test_start:] if w is not None else None,
            turn_cls[test_start:] if turn_cls is not None else None,
        ),
    }
    return out


def assign_leave_run_out(id_files_clean, val_runs=None, test_runs=None,
                          train_dir_runs=None, val_dir_runs=None, test_dir_runs=None):
    """Leave-run-out 划分：把整个 run（同次飞行的所有 csv 段）作为最小单元，
    val 与 train 完全 run 不重合，强制评估"未见过的飞行段"上的真实泛化能力。

    设计动机：原 within_file_temporal 把每个 csv 文件内部按时间 70/15/15 切，
    train/val 仅相隔 50 帧 gap，物理状态高度自相关 → val 误差被"训练时记住的
    弱风 csv"拖累而不是真正的泛化失败。Leave-run-out 让 val/test 全是 train
    完全没见过的飞行段，给出真实泛化下界，是"诚实评估"的金标准。

    优先级：
      1. 若显式给定 train_dir_runs / val_dir_runs / test_dir_runs，按目录划分
         （信任 dataset_generation 阶段的原始 run 级别隔离，最推荐）
      2. 否则按 val_runs / test_runs 集合显式指定
      3. 都未提供时报错，要求用户给定划分

    Args:
        id_files_clean: 通过速度三角形过滤的 ID csv 文件列表
        val_runs: set[int]，强制划入 val 的 run 编号集合
        test_runs: set[int]，强制划入 test_id 的 run 编号集合
        train_dir_runs / val_dir_runs / test_dir_runs:
            按 data_root 下 train/ val/ test_id/ 子目录中实际存在的 run 集合

    Returns:
        dict[name -> list of file paths]，name ∈ {'train', 'val', 'test_id'}
    """
    result = {'train': [], 'val': [], 'test_id': []}

    if (train_dir_runs is not None and val_dir_runs is not None and test_dir_runs is not None):
        # 按原始目录 run 集合划分：信任 dataset_generation 时的物理隔离
        for fp in id_files_clean:
            name = os.path.basename(fp)
            m = re.search(r'datasets-(\d+)-(\d+)', name)
            if not m:
                continue
            rid = int(m.group(1))
            if rid in train_dir_runs:
                result['train'].append(fp)
            elif rid in val_dir_runs:
                result['val'].append(fp)
            elif rid in test_dir_runs:
                result['test_id'].append(fp)
            # 不在任一集合中的 run 静默丢弃（理论不应该发生）
    elif val_runs is not None and test_runs is not None:
        val_runs = set(val_runs)
        test_runs = set(test_runs)
        for fp in id_files_clean:
            name = os.path.basename(fp)
            m = re.search(r'datasets-(\d+)-(\d+)', name)
            if not m:
                continue
            rid = int(m.group(1))
            if rid in val_runs:
                result['val'].append(fp)
            elif rid in test_runs:
                result['test_id'].append(fp)
            else:
                result['train'].append(fp)
    else:
        raise ValueError("leave_run_out 必须指定 (train_dir_runs, val_dir_runs, test_dir_runs) "
                         "或 (val_runs, test_runs)")

    # sort each split for deterministic ordering
    for k in ('train', 'val', 'test_id'):
        result[k] = sort_csv_paths(result[k])
    return result


def assign_stratified_run_out(id_files_clean,
                               train_frac=0.80,
                               val_frac=0.10,
                               n_strata=4,
                               seed=26,
                               verbose=True):
    """按"水平风强度分层"对 run 做 leave-run-out 划分。

    设计动机（PIRNN-AKF 算法约束）：
      1. 数据是连续时序数据集——AKF 状态、协方差按时间递推，PI-GRU 滑窗内部 50 帧也必须连续
      2. 因此 csv 段是不可切割的原子单位（within_file_temporal 违反此约束）
      3. 同一 run 的 5 段 csv 是同次飞行的连续段，相关性极强，应同进同出（run 是更稳的原子单位）
      4. 但纯按 run-id 顺序分配（旧 120/20/14）会导致风强度分布在 train/val/test 之间不均衡，
         val 误差既受"未见过的 csv"影响、又受"未见过的风强度区间"影响，两个变量耦合无法解耦
      5. Stratified-by-wind 让每个 split 在弱/中/强风段上同分布，val 误差只反映"csv 泛化"

    算法：
      - Step 1: 对每个 run，把它的所有 csv 拼接后算 |w_h| 平均（run-level wind score）
      - Step 2: 按 run wind score 分位数把所有 run 分成 n_strata 个分层（默认 4 档）
      - Step 3: 每个分层内部按种子打散，按 train_frac/val_frac/test_frac 切分配
                （test_frac = 1 - train_frac - val_frac）
      - Step 4: 合并各分层的 train/val/test_id run 列表 → 按 run 收集所有 csv 文件

    Args:
        id_files_clean: 通过过滤的 ID csv 文件列表（不含 OOD）
        train_frac / val_frac: 每分层内 train/val 占比；test_id = 1 - train_frac - val_frac
        n_strata: 分层数（默认 4 档：极弱/弱/中/强）
        seed: 分层内打散的随机种子
        verbose: 是否打印分层统计

    Returns:
        dict[name -> list of file paths]，name ∈ {'train', 'val', 'test_id'}
        以及一份分层报告 dict（用于元数据保存）
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(f"非法划分比例: train={train_frac}, val={val_frac}（test = 1 - train - val 必须 > 0）")

    # ── Step 1: 按 run 聚合 csv，计算 run-level wind score ──
    blocks = collect_run_blocks(id_files_clean)
    n_runs = len(blocks)
    if n_runs < n_strata * 3:
        raise ValueError(f"只有 {n_runs} 个 run，无法做 {n_strata}-strata × (train+val+test) 分层")

    # blocks 里每个 b 已经有 wind_mean (m/s) 等统计；用 wind_mean 做分层
    run_wind_scores = np.array([b['wind_mean'] for b in blocks])
    run_ids = np.array([b['run'] for b in blocks])

    # ── Step 2: 按 quantile 分层 ──
    # 用 quantile 而不是等距分箱，保证每层 run 数大致相等，避免极端分层只有 1-2 个 run
    quantiles = np.linspace(0, 1, n_strata + 1)
    bin_edges = np.quantile(run_wind_scores, quantiles)
    # 为兼容边界：把最后一个 edge 稍微抬高，让最大值落到最后一档
    bin_edges[-1] = run_wind_scores.max() + 1e-6
    bin_edges[0] = run_wind_scores.min() - 1e-6
    stratum_idx = np.clip(np.digitize(run_wind_scores, bin_edges, right=False) - 1, 0, n_strata - 1)

    rng = np.random.default_rng(seed)

    # ── Step 3: 每个分层内 80/10/10 划分 ──
    train_runs, val_runs, test_runs = [], [], []
    stratum_report = []

    for s in range(n_strata):
        in_stratum = np.where(stratum_idx == s)[0]
        if len(in_stratum) == 0:
            continue
        # 分层内打散
        order = rng.permutation(in_stratum)
        n_s = len(order)
        n_train_s = max(1, int(round(n_s * train_frac)))
        n_val_s = max(1, int(round(n_s * val_frac)))
        # 边界保护：保证 test 至少有 1 个 run
        if n_train_s + n_val_s >= n_s:
            n_train_s = max(1, n_s - 2)
            n_val_s = 1

        s_train = order[:n_train_s]
        s_val = order[n_train_s:n_train_s + n_val_s]
        s_test = order[n_train_s + n_val_s:]

        train_runs.extend(int(run_ids[i]) for i in s_train)
        val_runs.extend(int(run_ids[i]) for i in s_val)
        test_runs.extend(int(run_ids[i]) for i in s_test)

        stratum_report.append({
            'stratum': s,
            'wind_range': (float(bin_edges[s]), float(bin_edges[s + 1])),
            'n_runs': int(n_s),
            'n_train': int(len(s_train)),
            'n_val': int(len(s_val)),
            'n_test': int(len(s_test)),
            'train_runs': sorted(int(run_ids[i]) for i in s_train),
            'val_runs': sorted(int(run_ids[i]) for i in s_val),
            'test_runs': sorted(int(run_ids[i]) for i in s_test),
        })

    train_runs_set = set(train_runs)
    val_runs_set = set(val_runs)
    test_runs_set = set(test_runs)
    assert not (train_runs_set & val_runs_set), "train/val run 集合重叠"
    assert not (train_runs_set & test_runs_set), "train/test run 集合重叠"
    assert not (val_runs_set & test_runs_set), "val/test run 集合重叠"

    # ── Step 4: 把 csv 文件按 run 分配到三个 split ──
    result = {'train': [], 'val': [], 'test_id': []}
    for fp in id_files_clean:
        m = re.search(r'datasets-(\d+)-(\d+)', os.path.basename(fp))
        if not m:
            continue
        rid = int(m.group(1))
        if rid in train_runs_set:
            result['train'].append(fp)
        elif rid in val_runs_set:
            result['val'].append(fp)
        elif rid in test_runs_set:
            result['test_id'].append(fp)
    for k in ('train', 'val', 'test_id'):
        result[k] = sort_csv_paths(result[k])

    if verbose:
        print(f"\n  ── Stratified-by-wind 分层报告 (n_strata={n_strata}) ──")
        print(f"  {'层':>3}  {'风强度区间(m/s)':>18}  {'n_runs':>7}  {'train':>6}  {'val':>4}  {'test':>5}")
        print(f"  {'-'*3}  {'-'*18}  {'-'*7}  {'-'*6}  {'-'*4}  {'-'*5}")
        for r in stratum_report:
            wlo, whi = r['wind_range']
            print(f"  {r['stratum']:>3}  [{wlo:>5.2f}, {whi:>5.2f}]    {r['n_runs']:>7}  "
                  f"{r['n_train']:>6}  {r['n_val']:>4}  {r['n_test']:>5}")
        n_train_total = sum(r['n_train'] for r in stratum_report)
        n_val_total = sum(r['n_val'] for r in stratum_report)
        n_test_total = sum(r['n_test'] for r in stratum_report)
        print(f"  {'-'*3}  {'-'*18}  {'-'*7}  {'-'*6}  {'-'*4}  {'-'*5}")
        print(f"  {'合':>3}  {'计':>18}    {n_runs:>5}  {n_train_total:>6}  "
              f"{n_val_total:>4}  {n_test_total:>5}")
        print(f"\n  实际 run 编号分配：")
        print(f"    train  ({len(train_runs_set)} runs): {sorted(train_runs_set)}")
        print(f"    val    ({len(val_runs_set)} runs): {sorted(val_runs_set)}")
        print(f"    test_id({len(test_runs_set)} runs): {sorted(test_runs_set)}")

    meta = {
        'n_strata': n_strata,
        'train_frac': train_frac,
        'val_frac': val_frac,
        'seed': seed,
        'bin_edges_m_s': [float(x) for x in bin_edges],
        'stratum_report': stratum_report,
        'train_runs': sorted(train_runs_set),
        'val_runs': sorted(val_runs_set),
        'test_runs': sorted(test_runs_set),
    }
    return result, meta


def assign_within_run_temporal(blocks, train_frac=0.70, val_frac=0.15):
    """
    Within-run interleaved split：每个 run 的 5 个文件按固定模式分配，
    让 train/val/test 各自覆盖飞行全程（而非前/中/后段），消除时序偏移。

    对于 5 个文件的 run（文件按时间排序 0-4）：
      train:   文件 0, 1, 3   (70%)
      val:     文件 2         (15%)
      test_id: 文件 4         (15%)

    通用规则（任意 n 个文件）：
      - 每隔 round(1/val_frac) 个取一个给 val
      - 最后一个给 test_id
      - 其余给 train
    这样 val 和 train 都分布在飞行全程，消除"前段 train、后段 val"的时序偏移。
    """
    result = {'train': [], 'val': [], 'test_id': []}
    val_interval = max(2, round(1.0 / val_frac))  # 默认 round(1/0.15) ≈ 7，但对5个文件取round(1/0.2)=5

    for b in blocks:
        files = b['files']
        n = len(files)

        if n < 3:
            result['train'].append({**b, 'files': files})
            continue

        # 最后一个文件 → test_id（保持时序：test 永远是最新数据）
        test_files = [files[-1]]
        remaining = files[:-1]  # 前 n-1 个文件分配给 train/val

        # 每隔 val_interval 个取一个给 val，其余给 train
        val_files = []
        train_files = []
        for i, f in enumerate(remaining):
            if (i + 1) % val_interval == 0:
                val_files.append(f)
            else:
                train_files.append(f)

        # 若 val 为空（文件数太少），从 remaining 中间取一个
        if not val_files and len(remaining) >= 2:
            mid = len(remaining) // 2
            val_files = [remaining[mid]]
            train_files = [f for i, f in enumerate(remaining) if i != mid]

        if train_files:
            result['train'].append({**b, 'files': train_files})
        if val_files:
            result['val'].append({**b, 'files': val_files})
        if test_files:
            result['test_id'].append({**b, 'files': test_files})

    return result


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="data_1 preprocessing with velocity-triangle quality filter")
    parser.add_argument('--data_root', default='data/data_1')
    parser.add_argument('--outdir', default='data/data_1_processed_temporal')
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--train_frac', type=float, default=0.70)
    parser.add_argument('--val_frac', type=float, default=0.15)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--tri_threshold_fps', type=float, default=3.0,
                        help="速度三角形过滤阈值 (fps)。median|vtrue-||v-w||| 超过此值的文件会被丢弃。"
                             "默认 3.0 fps ≈ 0.91 m/s")
    parser.add_argument('--skip_filter', action='store_true',
                        help="跳过速度三角形过滤（保留所有文件）")
    # ===== 动态段加权（gust_phase / gust_factor / wind_regime） =====
    parser.add_argument('--sample_weight_enabled', type=int, default=1,
                        help="是否根据 gust_phase 等字段计算 sample weight（1=启用，0=全部置 1）")
    parser.add_argument('--weight_steady', type=float, default=1.0,
                        help="稳态样本权重 (regime=steady 或字段缺失)")
    parser.add_argument('--weight_dynamic_baseline', type=float, default=1.5,
                        help="动态 regime 但非阵风核心阶段的样本权重")
    parser.add_argument('--weight_hold', type=float, default=2.0,
                        help="gust_phase=hold（阵风维持段）的样本权重")
    parser.add_argument('--weight_transition', type=float, default=5.0,
                        help="gust_phase∈{rise,fall}（阵风过渡段）的样本权重（由 3.0 提高至 5.0）")
    parser.add_argument('--weight_factor_boost', type=float, default=0.5,
                        help="基于 |gust_factor| 的额外放大系数 (final = base * (1 + boost * |gf|))")
    parser.add_argument('--turn_discount', type=float, default=0.4,
                        help="转弯段（turn_class==1）样本权重折扣系数（0~1，默认 0.4；1.0=不折扣）")
    parser.add_argument('--split_strategy', type=str, default='within_file_temporal',
                        choices=['within_run_temporal', 'within_file_temporal', 'leave_run_out',
                                 'stratified_run_out'],
                        help="数据集划分策略：\n"
                             "  within_run_temporal  = 按 CSV 整文件分配（旧）\n"
                             "  within_file_temporal = 每个 CSV 内按时间 70/15/15 切（默认；train/val 同 csv，物理状态强相关）\n"
                             "  leave_run_out        = 按 run 整段划分，信任原始目录（120/20/14）\n"
                             "  stratified_run_out   = 按 run 整段 + 按风强度分层均匀打散（推荐：解决 train/val/test 风强度分布不一致）")
    parser.add_argument('--stratified_train_frac', type=float, default=0.80,
                        help="stratified_run_out 模式下每分层内 train 占比（默认 0.80）")
    parser.add_argument('--stratified_val_frac', type=float, default=0.10,
                        help="stratified_run_out 模式下每分层内 val 占比（默认 0.10；test_id = 1 - train - val）")
    parser.add_argument('--stratified_n_strata', type=int, default=4,
                        help="stratified_run_out 模式下风强度分层数（默认 4 档：极弱/弱/中/强）")
    parser.add_argument('--clip_sigma', type=float, default=8.0,
                        help="归一化空间下对 X 做 ±N·σ winsorize 裁剪，斩掉数十倍 σ 的 spike；0=关闭")
    parser.add_argument('--seed', type=int, default=26,
                        help="随机种子（用于 sample_ratio 抽样等）")
    parser.add_argument('--sampling_rate', type=int, default=50,
                        help="数据采样率 Hz（默认 50，需与采集端和 config.yaml 一致；用于加速度特征 dt 计算）")
    parser.add_argument('--downsample_factor', type=int, default=1,
                        help="原始 CSV 行下采样因子。新数据集 ~240 Hz、训练/部署 50 Hz 应设 5；旧 50 Hz 数据设 1")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    # ─── Step 1: 收集 CSV ───
    print(f"\n{'='*70}\nStep 1: 收集 CSV 文件\n{'='*70}")
    all_csv_files = []
    # 同时记录每个 split 子目录下的 run id 集合（仅 leave_run_out 用到）
    dir_runs = {'train': set(), 'val': set(), 'test_id': set(), 'test_ood': set()}
    for split in ['train', 'val', 'test_id']:
        d = os.path.join(args.data_root, split)
        if not os.path.exists(d):
            continue
        files = sort_csv_paths(glob.glob(os.path.join(d, '*.csv')))
        for fp in files:
            m = re.search(r'datasets-(\d+)-(\d+)', os.path.basename(fp))
            if m:
                dir_runs[split].add(int(m.group(1)))
        print(f"  {split}: {len(files)} files, {len(dir_runs[split])} runs")
        all_csv_files.extend(files)

    ood_dir = os.path.join(args.data_root, 'test_ood')
    ood_files = sort_csv_paths(glob.glob(os.path.join(ood_dir, '*.csv'))) if os.path.exists(ood_dir) else []
    for fp in ood_files:
        m = re.search(r'datasets-(\d+)-(\d+)', os.path.basename(fp))
        if m:
            dir_runs['test_ood'].add(int(m.group(1)))
    print(f"  test_ood: {len(ood_files)} files, {len(dir_runs['test_ood'])} runs")
    print(f"  总计: {len(all_csv_files)} ID + {len(ood_files)} OOD")

    # ─── Step 2: 速度三角形一致性过滤 ───
    print(f"\n{'='*70}\nStep 2: 速度三角形一致性过滤 (阈值={args.tri_threshold_fps:.1f} fps ≈ {args.tri_threshold_fps*0.3048:.2f} m/s)\n{'='*70}")

    if args.skip_filter:
        print("  ⚠️  已跳过速度三角形过滤")
        id_files_clean = all_csv_files
        ood_files_clean = ood_files
    else:
        id_files_clean = []
        id_files_reject = []
        for fp in tqdm(all_csv_files, desc="  检查 ID 文件"):
            passed, err, info = check_velocity_triangle(fp, args.tri_threshold_fps)
            if passed:
                id_files_clean.append(fp)
            else:
                id_files_reject.append((fp, err, info))

        ood_files_clean = []
        ood_files_reject = []
        for fp in tqdm(ood_files, desc="  检查 OOD 文件"):
            passed, err, info = check_velocity_triangle(fp, args.tri_threshold_fps)
            if passed:
                ood_files_clean.append(fp)
            else:
                ood_files_reject.append((fp, err, info))

        n_id_total = len(all_csv_files)
        n_id_pass = len(id_files_clean)
        n_ood_total = len(ood_files)
        n_ood_pass = len(ood_files_clean)

        print(f"\n  ID  文件: {n_id_pass}/{n_id_total} 通过 ({n_id_pass/max(n_id_total,1)*100:.1f}%)")
        print(f"  OOD 文件: {n_ood_pass}/{max(n_ood_total,1)} 通过 ({n_ood_pass/max(n_ood_total,1)*100:.1f}%)")

        if id_files_reject:
            errs = [e for _, e, _ in id_files_reject if np.isfinite(e)]
            print(f"  被拒绝 ID 文件的误差: mean={np.mean(errs):.2f} fps" if errs else "")
            print(f"  示例拒绝文件:")
            for fp, err, info in id_files_reject[:5]:
                print(f"    {os.path.basename(fp)}: {info}")

    # ─── Step 3: 数据集划分 ───
    print(f"\n{'='*70}\nStep 3: 数据集划分 (strategy={args.split_strategy})\n{'='*70}")
    sample_weight_config = {
        'enabled': bool(args.sample_weight_enabled),
        'steady': args.weight_steady,
        'dynamic_baseline': args.weight_dynamic_baseline,
        'hold': args.weight_hold,
        'transition': args.weight_transition,
        'factor_boost': args.weight_factor_boost,
        'gust_factor_threshold': 0.05,
        'turn_discount': args.turn_discount,
    }
    print(f"  sample_weight 配置: {sample_weight_config}")

    def build_sequences(files):
        file_infos = []
        total_n = 0
        x_shape = None
        y_shape = None
        for fp in tqdm(files, desc="  统计文件"):
            try:
                X, y, w, turn_cls = build_features_labels_from_csv(fp, args.seq_len, sample_weight_config, args.sampling_rate, args.downsample_factor)
            except Exception:
                continue
            if len(X) == 0:
                continue
            n_out = len(X)
            if args.sample_ratio < 1.0:
                n = len(X)
                n_out = min(n, max(100, int(n * args.sample_ratio)))
            file_infos.append((fp, n_out))
            total_n += n_out
            if x_shape is None:
                x_shape = X.shape[1:]
                y_shape = y.shape[1:]
            del X, y, w, turn_cls

        if not file_infos:
            return None, None, None, None

        X_all = np.empty((total_n, *x_shape), dtype=np.float32)
        y_all = np.empty((total_n, *y_shape), dtype=np.float32)
        w_all = np.empty((total_n,), dtype=np.float32)
        turn_all = np.empty((total_n,), dtype=np.int8)
        offset = 0
        for fp, n_expected in tqdm(file_infos, desc="  处理文件"):
            try:
                X, y, w, turn_cls = build_features_labels_from_csv(fp, args.seq_len, sample_weight_config, args.sampling_rate, args.downsample_factor)
            except Exception:
                continue
            if len(X) == 0:
                continue
            if args.sample_ratio < 1.0:
                n = len(X)
                n_keep = min(n, max(100, int(n * args.sample_ratio)))
                idx = np.sort(np.random.default_rng(args.seed).choice(n, n_keep, replace=False))
                X, y, w, turn_cls = X[idx], y[idx], w[idx], turn_cls[idx]
            n_cur = min(len(X), n_expected)
            X_all[offset:offset + n_cur] = X[:n_cur]
            y_all[offset:offset + n_cur] = y[:n_cur]
            w_all[offset:offset + n_cur] = w[:n_cur]
            turn_all[offset:offset + n_cur] = turn_cls[:n_cur]
            offset += n_cur
            del X, y, w, turn_cls

        return X_all[:offset], y_all[:offset], w_all[:offset], turn_all[:offset]

    split_data = {}
    # streaming_done=True 表示 stratified_run_out 分支已在 Step 3 内部完成
    # build/fit/normalize/save/release 全流程，后续 Step 5/5b 应跳过
    streaming_done = False
    # 这些变量供 Step 8 元数据保存使用（流式分支会在 Step 3 内部填充）
    scaler_X = None
    scaler_y = None
    mean_X = scale_X = mean_y = scale_y = None
    clip_stats_streaming = {}
    wind_dist_stats_streaming = {}
    weight_stats_streaming = {}
    stratified_meta = None

    if args.split_strategy == 'within_run_temporal':
        # ── 旧策略：按 CSV 整文件分配 ──
        blocks = collect_run_blocks(id_files_clean)
        print(f"  ID runs (过滤后): {len(blocks)}")
        assigned = assign_within_run_temporal(blocks, args.train_frac, args.val_frac)
        for split in ['train', 'val', 'test_id']:
            n_runs = len(assigned[split])
            n_files = sum(len(b['files']) for b in assigned[split])
            print(f"  {split:8s}: {n_runs} runs, {n_files} files  "
                  f"(每个 run 各贡献 {n_files // max(n_runs, 1)} 个文件到本 split)")

        print(f"\n{'='*70}\nStep 4: 构建序列（按 split 分别构建）\n{'='*70}")
        for split in ['train', 'val', 'test_id']:
            files = []
            for b in assigned[split]:
                files.extend(b['files'])
            files = sort_csv_paths(files)
            print(f"\n  构建 {split}: {len(files)} files")
            X, y, w, turn_cls = build_sequences(files)
            if X is not None:
                print(f"    {X.shape}, {y.shape}, w={w.shape}")
                split_data[split] = {'X': X, 'y': y, 'w': w, 'turn_class': turn_cls}
            del X, y, w, turn_cls
    elif args.split_strategy == 'leave_run_out':
        # ── 新策略：按整个 run（一次飞行）划分；val/test 完全 run 不重合 ──
        # 直接信任 dataset_generation 阶段的物理隔离（train/ val/ test_id/ 子目录）
        if not (dir_runs['train'] and dir_runs['val'] and dir_runs['test_id']):
            raise RuntimeError("leave_run_out 需要 data_root 下同时存在 train/ val/ test_id/ 三个子目录")

        # 检查 run 集合是否互斥
        overlap_tv = dir_runs['train'] & dir_runs['val']
        overlap_tt = dir_runs['train'] & dir_runs['test_id']
        overlap_vt = dir_runs['val'] & dir_runs['test_id']
        if overlap_tv or overlap_tt or overlap_vt:
            print(f"  ⚠ 警告：原始目录 run 集合存在重叠！"
                  f" train∩val={sorted(overlap_tv)}, train∩test_id={sorted(overlap_tt)}, "
                  f"val∩test_id={sorted(overlap_vt)}")

        assigned_files = assign_leave_run_out(
            id_files_clean,
            train_dir_runs=dir_runs['train'],
            val_dir_runs=dir_runs['val'],
            test_dir_runs=dir_runs['test_id'],
        )
        for split in ['train', 'val', 'test_id']:
            split_run_ids = sorted({
                int(re.search(r'datasets-(\d+)-(\d+)', os.path.basename(fp)).group(1))
                for fp in assigned_files[split]
                if re.search(r'datasets-(\d+)-(\d+)', os.path.basename(fp))
            })
            print(f"  {split:8s}: {len(split_run_ids)} runs, {len(assigned_files[split])} files "
                  f"(runs={split_run_ids[:5]}...{split_run_ids[-3:] if len(split_run_ids) > 8 else ''})")

        print(f"\n{'='*70}\nStep 4: 构建序列（按 split 分别构建，整段保留）\n{'='*70}")
        for split in ['train', 'val', 'test_id']:
            files = assigned_files[split]
            print(f"\n  构建 {split}: {len(files)} files")
            X, y, w, turn_cls = build_sequences(files)
            if X is not None:
                print(f"    {X.shape}, {y.shape}, w={w.shape}")
                split_data[split] = {'X': X, 'y': y, 'w': w, 'turn_class': turn_cls}
            del X, y, w, turn_cls
    elif args.split_strategy == 'stratified_run_out':
        # ── 推荐策略：按 run 整段（保连续性）+ 按风强度分层均匀打散 ──
        # 流式处理：每个 split build → fit/transform → save → 释放，避免同时占多份内存
        # 这是为内存受限环境（如 WSL2 31 GB）专门做的优化
        import gc

        assigned_files, stratified_meta = assign_stratified_run_out(
            id_files_clean,
            train_frac=args.stratified_train_frac,
            val_frac=args.stratified_val_frac,
            n_strata=args.stratified_n_strata,
            seed=args.seed,
            verbose=True,
        )

        print(f"\n{'='*70}\nStep 4 (流式): build → fit/transform → save 每个 split\n{'='*70}")
        os.makedirs(args.outdir, exist_ok=True)

        clip_sigma = float(args.clip_sigma)
        if clip_sigma > 0:
            print(f"  X 裁剪: ±{clip_sigma:g}σ winsorize")

        n_feat = FEATURE_DIM
        seq_len = args.seq_len

        # 处理顺序固定：train 先（fit scaler），其它共用 scaler
        # OOD 整段保留，不做时间切但参与归一化
        process_order = [
            ('train',    assigned_files['train'], True),
            ('val',      assigned_files['val'],   False),
            ('test_id',  assigned_files['test_id'], False),
            ('test_ood', ood_files_clean,         False),
        ]

        for name, files, is_train in process_order:
            if not files:
                print(f"\n  ⚠ {name} 无文件可处理，跳过")
                continue

            print(f"\n  ── 处理 {name}: {len(files)} files ──")
            X, y, w, turn_cls = build_sequences(files)
            if X is None or len(X) == 0:
                print(f"    ⚠ {name} 构建失败")
                continue
            print(f"    raw shape: X={X.shape}, y={y.shape}")

            # 在归一化前算原始单位 m/s 下的风强度统计
            wh_mag = np.sqrt(y[:, 0]**2 + y[:, 1]**2)
            wind_dist_stats_streaming[name] = {
                'n_samples': int(len(wh_mag)),
                'wh_mean': float(wh_mag.mean()),
                'wh_std': float(wh_mag.std()),
                'wh_p10': float(np.percentile(wh_mag, 10)),
                'wh_median': float(np.median(wh_mag)),
                'wh_p90': float(np.percentile(wh_mag, 90)),
                'weak_ratio': float(np.mean(wh_mag < 1.5)),
                'medium_ratio': float(np.mean((wh_mag >= 1.5) & (wh_mag < 2.5))),
                'strong_ratio': float(np.mean((wh_mag >= 2.5) & (wh_mag < 3.5))),
                'ood_strong_ratio': float(np.mean(wh_mag >= 3.5)),
            }
            del wh_mag

            # train 先 fit scaler；其它共用
            # 内存优化：sklearn 内部走 float64，train X 整体 fit 会瞬时 +33GB→OOM。
            # 这里只在 train 子样本上 fit（最多 200k 行），精度足够（>3σ 估计稳定）。
            if is_train:
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_flat_for_fit = X.reshape(-1, n_feat)
                fit_cap = min(200000, X_flat_for_fit.shape[0])
                fit_idx = np.random.default_rng(args.seed).choice(
                    X_flat_for_fit.shape[0], fit_cap, replace=False
                )
                scaler_X.fit(X_flat_for_fit[fit_idx])
                del fit_idx, X_flat_for_fit
                y_fit_cap = min(200000, y.shape[0])
                y_fit_idx = np.random.default_rng(args.seed + 1).choice(
                    y.shape[0], y_fit_cap, replace=False
                )
                scaler_y.fit(y[y_fit_idx])
                del y_fit_idx
                mean_X = scaler_X.mean_.astype(np.float32)
                scale_X = np.where(scaler_X.scale_ < 1e-8, 1.0, scaler_X.scale_).astype(np.float32)
                mean_y = scaler_y.mean_.astype(np.float32)
                scale_y = np.where(scaler_y.scale_ < 1e-8, 1.0, scaler_y.scale_).astype(np.float32)
                print(f"    fit scaler on train (subset N={fit_cap:,}): μ_y wind N/E/D={mean_y[:3].round(3)}, "
                      f"σ_y wind N/E/D={scale_y[:3].round(3)}")

            # in-place f32 normalize：避免 (X-μ)/σ 创建临时副本（峰值 +2× X 内存）。
            # X 自身就地修改，后面 del X 清理。
            X_flat = X.reshape(-1, n_feat)
            np.subtract(X_flat, mean_X, out=X_flat)
            np.divide(X_flat, scale_X, out=X_flat)
            X_n = X_flat.reshape(-1, seq_len, n_feat)

            if clip_sigma > 0:
                pre_min = float(X_n.min())
                pre_max = float(X_n.max())
                n_total = int(X_n.size)
                n_clipped = int(((X_n < -clip_sigma) | (X_n > clip_sigma)).sum())
                np.clip(X_n, -clip_sigma, clip_sigma, out=X_n)
                clip_stats_streaming[name] = {
                    'pre_min': pre_min, 'pre_max': pre_max,
                    'n_clipped': n_clipped, 'n_total': n_total,
                    'clipped_ratio': n_clipped / max(n_total, 1),
                }
                print(f"    pre_range=[{pre_min:+.2f}, {pre_max:+.2f}] "
                      f"裁剪 {n_clipped}/{n_total} ({100*n_clipped/max(n_total,1):.4f}%)")

            np.subtract(y, mean_y, out=y)
            np.divide(y, scale_y, out=y)
            y_n = y

            # 立刻全部落盘
            np.save(os.path.join(args.outdir, f'X_{name}.npy'), X_n)
            np.save(os.path.join(args.outdir, f'y_{name}.npy'), y_n)
            np.save(os.path.join(args.outdir, f'w_{name}.npy'), w.astype(np.float32))
            np.save(os.path.join(args.outdir, f'turn_class_{name}.npy'), turn_cls.astype(np.int8))

            mean_w = float(np.mean(w))
            n_dyn = int(np.sum(w > sample_weight_config['steady'] + 1e-6))
            weight_stats_streaming[name] = {
                'mean': mean_w, 'min': float(np.min(w)), 'max': float(np.max(w)),
                'dynamic_count': n_dyn, 'dynamic_ratio': n_dyn / max(len(w), 1),
            }
            print(f"    saved X/y/w_{name}.npy  N={len(y_n):,}  "
                  f"w mean={mean_w:.3f} 动态样本={100*n_dyn/max(len(w),1):.1f}%")

            # 仅保留 y_n 给后续 Step 9 绘图用，X 立即释放
            split_data[name] = {
                'X': None,
                'y': y_n,
                'w': w.astype(np.float32),
                'turn_class': turn_cls.astype(np.int8),
            }

            del X, X_flat, X_n, y, y_n, w, turn_cls
            gc.collect()

        streaming_done = True
        # 提前打印 Step 4b 的诊断表（流式分支内部已经累计 stats）
        print(f"\n{'='*70}\nStep 4b: 真实风强度分布统计 (归一化前，m/s)\n{'='*70}")
        print(f"  {'split':<10s} {'N':>10s}  {'|w_h| mean':>11s} "
              f"{'弱风<1.5':>10s} {'中风1.5-2.5':>13s} {'强风>2.5':>10s} {'OOD>3.5':>9s}")
        print(f"  {'-'*10} {'-'*10}  {'-'*11} {'-'*10} {'-'*13} {'-'*10} {'-'*9}")
        for name in ['train', 'val', 'test_id', 'test_ood']:
            s = wind_dist_stats_streaming.get(name)
            if not s:
                continue
            print(f"  {name:<10s} {s['n_samples']:>10,}  {s['wh_mean']:>11.3f} "
                  f"{s['weak_ratio']*100:>9.1f}% {s['medium_ratio']*100:>12.1f}% "
                  f"{s['strong_ratio']*100:>9.1f}% {s['ood_strong_ratio']*100:>8.1f}%")
    else:
        # ── 新策略：每个 CSV 文件内部按时间 70/15/15 切，各 split 同分布 ──
        print(f"  共 {len(id_files_clean)} 个 ID 文件，逐文件按时间切 (gap=seq_len={args.seq_len} 防止滑窗泄漏)")
        gap = args.seq_len
        accum = {'train': {'X': [], 'y': [], 'w': [], 'turn_class': []},
                 'val':   {'X': [], 'y': [], 'w': [], 'turn_class': []},
                 'test_id': {'X': [], 'y': [], 'w': [], 'turn_class': []}}
        n_files_used = 0
        n_files_too_short = 0
        for fp in tqdm(id_files_clean, desc="  处理文件"):
            try:
                X, y, w, turn_cls = build_features_labels_from_csv(fp, args.seq_len, sample_weight_config, args.sampling_rate, args.downsample_factor)
            except Exception:
                continue
            if X is None or len(X) == 0:
                continue
            splits = split_arrays_within_file_temporal(X, y, w, turn_cls, args.train_frac, args.val_frac, gap=gap)
            if splits['val'][0] is None or splits['test_id'][0] is None:
                n_files_too_short += 1
            for name, (Xs, ys, ws, ts) in splits.items():
                if Xs is None or len(Xs) == 0:
                    continue
                accum[name]['X'].append(Xs)
                accum[name]['y'].append(ys)
                accum[name]['w'].append(ws)
                accum[name]['turn_class'].append(ts)
            n_files_used += 1
            del X, y, w, turn_cls

        # vstack 一次只做一个 split，做完立刻 clear 它的段列表，避免段列表 + vstack 输出同时占内存
        for name in ['train', 'val', 'test_id']:
            if not accum[name]['X']:
                print(f"  ⚠ {name} 为空（所有文件都太短或缺失）")
                continue
            X_cat = np.vstack(accum[name]['X'])
            accum[name]['X'].clear()  # 立即释放段引用，节省一份 ~ X 大小的内存
            y_cat = np.vstack(accum[name]['y'])
            accum[name]['y'].clear()
            w_cat = np.concatenate(accum[name]['w'])
            accum[name]['w'].clear()
            t_cat = np.concatenate(accum[name]['turn_class']).astype(np.int8)
            accum[name]['turn_class'].clear()
            split_data[name] = {'X': X_cat, 'y': y_cat, 'w': w_cat, 'turn_class': t_cat}
            print(f"  {name:8s}: {X_cat.shape}, w mean={w_cat.mean():.3f}")
        print(f"\n  生效文件: {n_files_used} / {len(id_files_clean)}  "
              f"（其中 {n_files_too_short} 个文件太短，已整段并入 train）")

    if not streaming_done and ood_files_clean:
        print(f"\n  构建 test_ood: {len(ood_files_clean)} files (整段保留，不做时间切)")
        X_ood, y_ood, w_ood, t_ood = build_sequences(ood_files_clean)
        if X_ood is not None:
            print(f"    {X_ood.shape}, {y_ood.shape}, w={w_ood.shape}")
            split_data['test_ood'] = {'X': X_ood, 'y': y_ood, 'w': w_ood, 'turn_class': t_ood}
        del X_ood, y_ood, w_ood, t_ood

    # ─── Step 4b: 真实风强度分布统计（归一化前，单位 m/s） ───
    # 关键诊断：弱风段 |w_h| < 1.5 m/s 占比是否 > 13.5%（论文 §4.5 阈值）
    # 流式分支已经在 Step 3 内打印过同名表，这里跳过
    if not streaming_done:
        print(f"\n{'='*70}\nStep 4b: 真实风强度分布统计 (归一化前，m/s)\n{'='*70}")
        print(f"  {'split':<10s} {'N':>10s}  {'|w_h| mean':>11s} {'弱风<1.5':>10s} {'中风1.5-2.5':>13s} {'强风>2.5':>10s} {'OOD>3.5':>9s}")
        print(f"  {'-'*10} {'-'*10}  {'-'*11} {'-'*10} {'-'*13} {'-'*10} {'-'*9}")
        wind_dist_stats = {}
        for name in ['train', 'val', 'test_id', 'test_ood']:
            if name not in split_data:
                continue
            y = split_data[name]['y']
            # y[:, :3] = wind_north / east / down (m/s)
            wh_mag = np.sqrt(y[:, 0]**2 + y[:, 1]**2)  # 水平风模值
            n = len(wh_mag)
            weak = float(np.mean(wh_mag < 1.5))
            med = float(np.mean((wh_mag >= 1.5) & (wh_mag < 2.5)))
            strong = float(np.mean((wh_mag >= 2.5) & (wh_mag < 3.5)))
            ood_strong = float(np.mean(wh_mag >= 3.5))
            wind_dist_stats[name] = {
                'n_samples': int(n),
                'wh_mean': float(wh_mag.mean()),
                'wh_std': float(wh_mag.std()),
                'wh_p10': float(np.percentile(wh_mag, 10)),
                'wh_median': float(np.median(wh_mag)),
                'wh_p90': float(np.percentile(wh_mag, 90)),
                'weak_ratio': weak,
                'medium_ratio': med,
                'strong_ratio': strong,
                'ood_strong_ratio': ood_strong,
            }
            print(f"  {name:<10s} {n:>10,}  {wh_mag.mean():>11.3f} "
                  f"{weak*100:>9.1f}% {med*100:>12.1f}% {strong*100:>9.1f}% {ood_strong*100:>8.1f}%")
    else:
        wind_dist_stats = wind_dist_stats_streaming

    # ─── Step 5: Fit scaler on TRAIN only（避免用 test 数据影响归一化，同时节省内存） ───
    if streaming_done:
        # 流式分支已完成 fit + transform + save，此处跳过
        print(f"\n{'='*70}\nStep 5/5b: 已由流式分支完成（streaming_done=True），跳过\n{'='*70}")
        clip_sigma = float(args.clip_sigma)
        clip_stats = clip_stats_streaming
        # n_feat / seq_len 给 Step 8 元数据 dump 用
        n_feat = FEATURE_DIM
        seq_len = args.seq_len
    else:
        print(f"\n{'='*70}\nStep 5: Fit scaler on TRAIN data only\n{'='*70}")
        X_tr = split_data['train']['X']
        y_tr = split_data['train']['y']
        n_samples, seq_len, n_feat = X_tr.shape
        print(f"  train: X={X_tr.shape}")

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        scaler_X.fit(X_tr.reshape(-1, n_feat))
        scaler_y.fit(y_tr)
        print(f"  scaler_y wind N/E/D: mean={scaler_y.mean_[:3].round(3)}, scale={scaler_y.scale_[:3].round(3)}")

        # 手工 float32 normalize：sklearn 的 .transform 会先输出 float64（瞬时占用翻倍），
        # 在 30GB 内存机器上极易 OOM。这里直接用 (x - μ) / σ 在 f32 空间完成。
        mean_X = scaler_X.mean_.astype(np.float32)
        scale_X = np.where(scaler_X.scale_ < 1e-8, 1.0, scaler_X.scale_).astype(np.float32)
        mean_y = scaler_y.mean_.astype(np.float32)
        scale_y = np.where(scaler_y.scale_ < 1e-8, 1.0, scaler_y.scale_).astype(np.float32)

        clip_sigma = float(args.clip_sigma)
        if clip_sigma > 0:
            print(f"  X 裁剪: ±{clip_sigma:g}σ winsorize（消除 acc/vel ±25–30σ 离群点）")
        clip_stats = {name: {'pre_min': None, 'pre_max': None, 'n_clipped': 0, 'n_total': 0}
                      for name in split_data.keys()}

        # ─── Step 5b: 逐 split 归一化 + 裁剪 + 立刻落盘释放 X，避免同时占用所有 split 的 X ───
        os.makedirs(args.outdir, exist_ok=True)
        for name in list(split_data.keys()):
            X_r = split_data[name]['X']
            y_r = split_data[name]['y']
            # f32 in-place normalize（保证 X_r 的连续性，避免 reshape 触发 copy）
            X_flat = np.ascontiguousarray(X_r).reshape(-1, n_feat)
            X_n = ((X_flat - mean_X) / scale_X).astype(np.float32, copy=False)
            X_n = X_n.reshape(-1, seq_len, n_feat)
            if clip_sigma > 0:
                pre_min = float(X_n.min())
                pre_max = float(X_n.max())
                n_total = int(X_n.size)
                n_clipped = int(((X_n < -clip_sigma) | (X_n > clip_sigma)).sum())
                np.clip(X_n, -clip_sigma, clip_sigma, out=X_n)
                clip_stats[name] = {
                    'pre_min': pre_min, 'pre_max': pre_max,
                    'n_clipped': n_clipped, 'n_total': n_total,
                    'clipped_ratio': n_clipped / max(n_total, 1),
                }
                print(f"  {name:8s}: pre_range=[{pre_min:+.2f}, {pre_max:+.2f}] "
                      f"裁剪 {n_clipped}/{n_total} ({100*n_clipped/max(n_total,1):.4f}%)")
            y_n = ((y_r - mean_y) / scale_y).astype(np.float32, copy=False)

            # 立刻把归一化后的 X 写盘并从内存释放，y/w 留在 split_data 给后续 stats / 绘图使用
            np.save(os.path.join(args.outdir, f'X_{name}.npy'), X_n)
            split_data[name]['y'] = y_n
            split_data[name]['X'] = None  # 释放
            del X_r, X_flat, X_n

    # ─── Step 6: 分布统计 ───
    print(f"\n{'='*70}\nStep 6: 归一化后分布统计\n{'='*70}")
    for name in ['train', 'val', 'test_id', 'test_ood']:
        if name not in split_data:
            continue
        y = split_data[name]['y']
        mag = np.linalg.norm(y[:, :3], axis=1)
        w = y[:, :3]
        print(f"  {name:10s}: N={len(y):>8,}  |w| mean={mag.mean():.3f} std={mag.std():.3f} "
              f"[{mag.min():.3f}, {mag.max():.3f}]  "
              f"N={w[:,0].mean():+.3f} E={w[:,1].mean():+.3f} D={w[:,2].mean():+.3f}")

    # ─── Step 7: 保存 y / w（X 已在 Step 5b 中落盘释放） ───
    if streaming_done:
        # 流式分支已经保存了 y/w，跳过
        print(f"\n{'='*70}\nStep 7: 已由流式分支完成，跳过\n{'='*70}")
        weight_stats = weight_stats_streaming
        turn_stats = {}
        for name in ['train', 'val', 'test_id', 'test_ood']:
            if name not in split_data:
                continue
            turn_arr = split_data[name].get('turn_class')
            if turn_arr is None:
                continue
            turn_stats[name] = {
                'turning_ratio': float(np.mean(turn_arr == 1)),
                'non_turning_ratio': float(np.mean(turn_arr == 0)),
                'unknown_ratio': float(np.mean(turn_arr < 0)),
            }
    else:
        print(f"\n{'='*70}\nStep 7: 保存 y / w（X 已落盘）\n{'='*70}")
        weight_stats = {}
        turn_stats = {}
        for name in ['train', 'val', 'test_id', 'test_ood']:
            if name not in split_data:
                continue
            y = split_data[name]['y']
            w = split_data[name].get('w')
            turn_arr = split_data[name].get('turn_class')
            np.save(os.path.join(args.outdir, f'y_{name}.npy'), y)
            if w is not None:
                np.save(os.path.join(args.outdir, f'w_{name}.npy'), w.astype(np.float32))
                mean_w = float(np.mean(w))
                max_w = float(np.max(w))
                min_w = float(np.min(w))
                n_total = int(w.shape[0])
                n_dyn = int(np.sum(w > sample_weight_config['steady'] + 1e-6))
                ratio_dyn = n_dyn / max(n_total, 1)
                weight_stats[name] = {
                    'mean': mean_w, 'min': min_w, 'max': max_w,
                    'dynamic_count': n_dyn, 'dynamic_ratio': ratio_dyn,
                }
                print(f"  {name}: y={y.shape}, w mean={mean_w:.3f} max={max_w:.3f} 动态样本={ratio_dyn*100:.1f}%")
            else:
                print(f"  {name}: y={y.shape}")
            if turn_arr is not None:
                np.save(os.path.join(args.outdir, f'turn_class_{name}.npy'), turn_arr.astype(np.int8))
                turn_stats[name] = {
                    'turning_ratio': float(np.mean(turn_arr == 1)),
                    'non_turning_ratio': float(np.mean(turn_arr == 0)),
                    'unknown_ratio': float(np.mean(turn_arr < 0)),
                }
                print(
                    f"       turn: turning={turn_stats[name]['turning_ratio']*100:.1f}% "
                    f"non_turning={turn_stats[name]['non_turning_ratio']*100:.1f}% "
                    f"unknown={turn_stats[name]['unknown_ratio']*100:.1f}%"
                )

    # ─── Step 8: 保存元数据 ───
    model_dir = os.path.join(project_root, 'train_data1')
    os.makedirs(model_dir, exist_ok=True)
    feature_names_full = [
        'vel_n', 'vel_e', 'vel_d', 'vel_x_body', 'vel_y_body', 'vel_z_body',
        'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw',
        'gyro_x', 'gyro_y', 'gyro_z',
        'aileron', 'elevator', 'rudder', 'throttle', 'airspeed',
        # 阶段 1 新增
        'target_roll', 'target_pitch', 'target_yaw',
        'roll_err', 'pitch_err', 'yaw_err',
        'target_p', 'target_q', 'target_r',
        'p_err', 'q_err', 'r_err',
        # 阶段 2 新增
        'target_vn', 'target_ve', 'target_vd',
        'vn_err', 've_err', 'vd_err',
        'aileron_act', 'elevator_act', 'rudder_act', 'throttle_act',
        'imu_ax', 'imu_ay', 'imu_az',
    ]
    metadata = {
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': feature_names_full[:FEATURE_DIM],
        'feature_idx': FEATURE_IDX,
        'label_names': ['wind_north', 'wind_east', 'wind_down', 'vel_n', 'vel_e', 'vel_d', 'airspeed'],
        'sequence_length': args.seq_len,
        'input_size': FEATURE_DIM,
        'output_size': 7,
        'split_strategy': args.split_strategy,
        'scaler_fit_source': 'train_only',
        'data_root': args.data_root,
        'velocity_triangle_threshold_fps': args.tri_threshold_fps,
        'clip_sigma': clip_sigma,
        'clip_stats': clip_stats,
        'splits': {name: {'n_samples': len(split_data[name]['y'])} for name in split_data},
        'sample_weight_config': sample_weight_config,
        'sample_weight_stats': weight_stats,
        'turn_label_stats': turn_stats,
        'aux_label_files': [f"turn_class_{name}.npy" for name in split_data.keys()],
        'wind_distribution_stats': wind_dist_stats,
        'dir_runs_observed': {k: sorted(v) for k, v in dir_runs.items()},
        'stratified_meta': stratified_meta,
    }
    with open(os.path.join(model_dir, 'norm_params.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    # 也在 outdir 内保存一份方便查看
    with open(os.path.join(args.outdir, 'norm_params.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n  norm_params.pkl -> {model_dir} + {args.outdir}")
    print(f"  输出目录: {args.outdir}")

    # ─── Step 9: 绘图 ───
    print(f"\n{'='*70}\nStep 9: 绘制分布图\n{'='*70}")
    _plot_all_splits_kde(split_data, args.outdir)
    if 'test_id' in split_data:
        _plot_comparison(split_data['train']['y'], split_data['test_id']['y'],
                         'Train', 'Test-ID',
                         os.path.join(args.outdir, 'distribution_train_vs_test_id.png'))
    if 'test_ood' in split_data:
        _plot_comparison(split_data['train']['y'], split_data['test_ood']['y'],
                         'Train', 'Test-OOD',
                         os.path.join(args.outdir, 'distribution_train_vs_test_ood.png'))

    print(f"\n{'='*70}\n✅ data_1 预处理完成！\n{'='*70}")


# ─────────────────────────────────────────────
# 绘图函数
# ─────────────────────────────────────────────

def _plot_comparison(y_ref, y_cmp, ref_label, cmp_label, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ref_c, cmp_c = '#4472C4', '#ED7D31'

    labels = ['Wind North', 'Wind East', 'Wind Down']
    for idx in range(3):
        ax = axes[idx // 2, idx % 2]
        rd, cd = y_ref[:, idx], y_cmp[:, idx]
        p5, p95 = float(np.percentile(rd, 5)), float(np.percentile(rd, 95))
        x_lo = min(rd.min(), cd.min()) - 0.5
        x_hi = max(rd.max(), cd.max()) + 0.5
        xr = np.linspace(x_lo, x_hi, 300)
        n1, n2 = min(10000, len(rd)), min(10000, len(cd))
        rk = gaussian_kde(np.random.default_rng(26).choice(rd, n1, replace=False))
        ck = gaussian_kde(np.random.default_rng(27).choice(cd, n2, replace=False))
        ax.fill_between(xr, rk(xr), alpha=0.25, color=ref_c)
        ax.fill_between(xr, ck(xr), alpha=0.25, color=cmp_c)
        ax.plot(xr, rk(xr), color=ref_c, lw=2, label=ref_label)
        ax.plot(xr, ck(xr), color=cmp_c, lw=2, ls='--', label=cmp_label)
        ov = float(np.mean((p5 <= cd) & (cd <= p95)) * 100)
        ax.set_title(f'{labels[idx]}\nref p5-p95=[{p5:.2f},{p95:.2f}] ov={ov:.1f}%', fontsize=10, fontweight='bold')
        ax.set_xlabel('Normalized'); ax.set_ylabel('Density')
        ax.legend(fontsize=9); ax.grid(alpha=0.3, ls='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1, 1]
    mr = np.linalg.norm(y_ref[:, :3], axis=1)
    mc = np.linalg.norm(y_cmp[:, :3], axis=1)
    p5, p95 = float(np.percentile(mr, 5)), float(np.percentile(mr, 95))
    xr = np.linspace(min(mr.min(), mc.min()) - 0.2, max(mr.max(), mc.max()) + 0.2, 300)
    rk = gaussian_kde(np.random.default_rng(26).choice(mr, min(10000, len(mr)), replace=False))
    ck = gaussian_kde(np.random.default_rng(27).choice(mc, min(10000, len(mc)), replace=False))
    ax.fill_between(xr, rk(xr), alpha=0.25, color=ref_c)
    ax.fill_between(xr, ck(xr), alpha=0.25, color=cmp_c)
    ax.plot(xr, rk(xr), color=ref_c, lw=2, label=ref_label)
    ax.plot(xr, ck(xr), color=cmp_c, lw=2, ls='--', label=cmp_label)
    ov = float(np.mean((p5 <= mc) & (mc <= p95)) * 100)
    ax.set_title(f'Wind Magnitude\nref p5-p95=[{p5:.2f},{p95:.2f}] ov={ov:.1f}%', fontsize=10, fontweight='bold')
    ax.set_xlabel('|w| (normalized)'); ax.set_ylabel('Density')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.suptitle(f'{ref_label} vs {cmp_label} Distribution', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  saved {save_path}")


def _plot_all_splits_kde(split_data, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {'train': '#4472C4', 'val': '#70AD47', 'test_id': '#ED7D31', 'test_ood': '#C00000'}
    lss = {'train': '-', 'val': '--', 'test_id': '-.', 'test_ood': ':'}
    labels_map = {'train': 'Train', 'val': 'Val', 'test_id': 'Test-ID', 'test_ood': 'Test-OOD'}

    mags = []
    for name in ['train', 'val', 'test_id', 'test_ood']:
        if name not in split_data:
            continue
        mags.append((name, np.linalg.norm(split_data[name]['y'][:, :3], axis=1)))

    if not mags:
        return

    xr = np.linspace(min(v[1].min() for v in mags) - 0.2,
                     max(v[1].max() for v in mags) + 0.2, 300)
    for name, mag in mags:
        n_s = min(10000, len(mag))
        kde = gaussian_kde(np.random.default_rng(26).choice(mag, n_s, replace=False))
        ax.plot(xr, kde(xr), color=colors[name], lw=2.5, ls=lss[name], label=labels_map[name])
        ax.fill_between(xr, kde(xr), alpha=0.08, color=colors[name])

    ax.set_title('Wind Magnitude Distribution — All Splits (data_1)', fontsize=12, fontweight='bold')
    ax.set_xlabel('|w| (normalized)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10); ax.grid(alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_path = os.path.join(outdir, 'distribution_all_splits.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  saved {save_path}")


if __name__ == '__main__':
    main()
