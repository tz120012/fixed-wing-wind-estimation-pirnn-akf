"""
online_feature_extractor.py
===========================
在线部署与离线验证共用的原子特征装配模块。修回协议先装配历史
45 维布局，再删除索引 38--41 的执行器代理，形成最终 41 维模型输入。

设计目标
--------
把 "原子信号 dict -> 45 维特征向量" 这一步从 MAVLink 解析中剥离，做成一个纯粹、
可单测的流式提取器。这样：

  * 在线部署 (6c_online_deployment) 负责把 MAVLink 消息解析成 `signals` dict，
    再交给本模块装配；
  * 离线验证 (experiments/verify_online_45d_chain) 负责把预处理后的 CSV 行解析成
    同样的 `signals` dict，再交给本模块装配。

两条路径调用完全相同的 `StreamingFeatureExtractor.push()`，从而保证 "训练特征定义"
与 "在线特征定义" 严格一致——这是 HITL 精度能否复现离线精度的前提。

特征布局与训练侧 `1_preprocessing_data.py` 的 `FEATURE_IDX` 完全对齐（阶段 2，45 维）。
唯一无法逐点复刻的是索引 6-8（机体速度加速度）：训练侧用 `np.gradient + savgol`
（非因果、整段），在线只能因果估计，本模块用 "末端局部多项式回归导数" 做因果逼近。
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import numpy as np

# ── 45 维特征索引（必须与 1_preprocessing_data.py FEATURE_IDX 逐一对齐）──────────
FEATURE_IDX: Dict[str, int] = {
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

FEATURE_DIM = 45
REVISION_REMOVED_INDICES = (38, 39, 40, 41)
REVISION_KEEP_INDICES = np.asarray(
    [i for i in range(FEATURE_DIM) if i not in REVISION_REMOVED_INDICES],
    dtype=np.int64,
)


def select_model_features(features: np.ndarray, input_size: int) -> np.ndarray:
    """Select the frozen model feature profile from a full 45-D vector/array."""
    values = np.asarray(features)
    if values.shape[-1] != FEATURE_DIM:
        raise ValueError(f"Expected trailing feature dimension 45, got {values.shape}")
    if int(input_size) == FEATURE_DIM:
        return values
    if int(input_size) == len(REVISION_KEEP_INDICES):
        return values[..., REVISION_KEEP_INDICES]
    raise ValueError(f"Unsupported model input_size={input_size}; expected 41 or 45")

# 原子信号键——在线 MAVLink 与离线 CSV 都必须提供这些量（缺省用安全默认值填充）。
SIGNAL_KEYS = (
    "vel_n", "vel_e", "vel_d",                       # NED 地速 (m/s)
    "roll", "pitch", "yaw",                          # 姿态角 (rad)
    "roll_rate", "pitch_rate", "yaw_rate",           # 机体角速率 (rad/s)
    "aileron_cmd", "elevator_cmd", "rudder_cmd", "throttle_cmd",  # 指令舵面 (norm)
    "airspeed",                                      # 真空速 (m/s)
    "target_roll", "target_pitch", "target_yaw",     # 期望姿态 (rad)
    "target_p", "target_q", "target_r",              # 期望角速率 (rad/s)
    "target_vn", "target_ve", "target_vd",           # 期望地速 (m/s)
    "aileron_actual", "elevator_actual", "rudder_actual", "throttle_actual",  # 实际舵面
    "imu_ax", "imu_ay", "imu_az",                    # 原始 IMU 机体加速度 (m/s^2)
)

_SIGNAL_DEFAULTS: Dict[str, float] = {k: 0.0 for k in SIGNAL_KEYS}
_SIGNAL_DEFAULTS["throttle_cmd"] = 0.5
_SIGNAL_DEFAULTS["throttle_actual"] = 0.5
_SIGNAL_DEFAULTS["airspeed"] = 15.0


def _wrap_angle(a: float) -> float:
    """把角度 wrap 到 (-pi, pi]，与预处理 _wrap_angle 一致。"""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def body_velocity_from_ned(vel_n, vel_e, vel_d, roll, pitch, yaw):
    """NED 地速经 3-2-1 姿态旋转到机体系。

    公式与 1_preprocessing_data.py (vx/vy/vz) 逐项一致，保证机体速度特征零偏差。
    """
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    vx = cp * cy * vel_n + cp * sy * vel_e - sp * vel_d
    vy = (sr * sp * cy - cr * sy) * vel_n + (sr * sp * sy + cr * cy) * vel_e + sr * cp * vel_d
    vz = (cr * sp * cy + sr * sy) * vel_n + (cr * sp * sy - sr * cy) * vel_e + cr * cp * vel_d
    return vx, vy, vz


class StreamingFeatureExtractor:
    """流式 45 维特征提取器（因果、可复位）。

    用法::

        ext = StreamingFeatureExtractor(sampling_rate=50.0)
        for signals in stream:                # signals 为 SIGNAL_KEYS 的 dict
            feat_row = ext.push(signals)      # -> np.ndarray shape (45,)

    加速度（索引 6-8）由机体速度的因果局部多项式回归导数估计；其余 42 维为逐帧确定量。
    """

    def __init__(self, sampling_rate: float = 50.0, accel_window: int = 25,
                 accel_polyorder: int = 2):
        self.dt = 1.0 / float(sampling_rate)          # 名义采样间隔（缺省/离线用）
        self.accel_window = int(accel_window)
        self.accel_polyorder = int(accel_polyorder)
        self._vbody_hist: deque = deque(maxlen=self.accel_window)
        # 每帧的累计时间戳（秒）。在线传入实测 dt 时用真实时间轴做导数，
        # 从而消除"名义 50Hz 但真机只到 ~29Hz"造成的加速度尺度误差。
        self._t_hist: deque = deque(maxlen=self.accel_window)
        self._t_accum: float = 0.0

    def reset(self) -> None:
        self._vbody_hist.clear()
        self._t_hist.clear()
        self._t_accum = 0.0

    def _causal_body_accel(self) -> np.ndarray:
        """机体速度末端因果导数（局部多项式回归），因果逼近训练侧 gradient+savgol。

        时间轴取自 `_t_hist`（在线为实测累计时间；离线/缺省为名义 dt 等间隔），
        保证 dv/dt 用的是真实间隔而非名义 50Hz。
        """
        n = len(self._vbody_hist)
        if n < 3:
            return np.zeros(3, dtype=np.float32)
        arr = np.asarray(self._vbody_hist, dtype=np.float64)  # (n, 3)
        t = np.asarray(self._t_hist, dtype=np.float64)
        t = t - t[0]
        if t[-1] <= 0.0:  # 退化保护：时间轴无效则回退到名义等间隔
            t = np.arange(n, dtype=np.float64) * self.dt
        order = self.accel_polyorder if n >= (self.accel_polyorder + 3) else 1
        acc = np.zeros(3, dtype=np.float64)
        for k in range(3):
            coeffs = np.polyfit(t, arr[:, k], order)
            deriv = np.polyder(coeffs)
            acc[k] = np.polyval(deriv, t[-1])
        return acc.astype(np.float32)

    def push(self, signals: Dict[str, float], dt: Optional[float] = None) -> np.ndarray:
        s = {**_SIGNAL_DEFAULTS, **{k: float(v) for k, v in signals.items()
                                    if v is not None and np.isfinite(v)}}
        f = np.zeros(FEATURE_DIM, dtype=np.float32)

        vel_n, vel_e, vel_d = s["vel_n"], s["vel_e"], s["vel_d"]
        roll, pitch, yaw = s["roll"], s["pitch"], s["yaw"]

        # 0-2 NED 地速
        f[FEATURE_IDX["vel_n"]] = vel_n
        f[FEATURE_IDX["vel_e"]] = vel_e
        f[FEATURE_IDX["vel_d"]] = vel_d

        # 3-5 机体速度
        vx, vy, vz = body_velocity_from_ned(vel_n, vel_e, vel_d, roll, pitch, yaw)
        f[FEATURE_IDX["vx_body"]] = vx
        f[FEATURE_IDX["vy_body"]] = vy
        f[FEATURE_IDX["vz_body"]] = vz

        # 6-8 机体速度加速度（因果估计）
        self._vbody_hist.append((vx, vy, vz))
        step = float(dt) if (dt is not None and np.isfinite(dt) and dt > 0.0) else self.dt
        self._t_accum += step
        self._t_hist.append(self._t_accum)
        acc = self._causal_body_accel()
        f[FEATURE_IDX["ax"]] = acc[0]
        f[FEATURE_IDX["ay"]] = acc[1]
        f[FEATURE_IDX["az"]] = acc[2]

        # 9-11 姿态角
        f[FEATURE_IDX["roll"]] = roll
        f[FEATURE_IDX["pitch"]] = pitch
        f[FEATURE_IDX["yaw"]] = yaw

        # 12-14 机体角速率
        p_rate, q_rate, r_rate = s["roll_rate"], s["pitch_rate"], s["yaw_rate"]
        f[FEATURE_IDX["p_rate"]] = p_rate
        f[FEATURE_IDX["q_rate"]] = q_rate
        f[FEATURE_IDX["r_rate"]] = r_rate

        # 15-18 指令舵面 + 油门
        f[FEATURE_IDX["aileron_cmd"]] = s["aileron_cmd"]
        f[FEATURE_IDX["elevator_cmd"]] = s["elevator_cmd"]
        f[FEATURE_IDX["rudder_cmd"]] = s["rudder_cmd"]
        f[FEATURE_IDX["throttle_cmd"]] = s["throttle_cmd"]

        # 19 空速
        f[FEATURE_IDX["airspeed"]] = s["airspeed"]

        # 20-22 期望姿态
        t_roll, t_pitch, t_yaw = s["target_roll"], s["target_pitch"], s["target_yaw"]
        f[FEATURE_IDX["target_roll"]] = t_roll
        f[FEATURE_IDX["target_pitch"]] = t_pitch
        f[FEATURE_IDX["target_yaw"]] = t_yaw

        # 23-25 姿态误差（actual - target），yaw wrap 到 (-pi, pi]
        f[FEATURE_IDX["roll_err"]] = roll - t_roll
        f[FEATURE_IDX["pitch_err"]] = pitch - t_pitch
        f[FEATURE_IDX["yaw_err"]] = _wrap_angle(yaw - t_yaw)

        # 26-28 期望角速率
        t_p, t_q, t_r = s["target_p"], s["target_q"], s["target_r"]
        f[FEATURE_IDX["target_p"]] = t_p
        f[FEATURE_IDX["target_q"]] = t_q
        f[FEATURE_IDX["target_r"]] = t_r

        # 29-31 角速率误差（actual - target）
        f[FEATURE_IDX["p_err"]] = p_rate - t_p
        f[FEATURE_IDX["q_err"]] = q_rate - t_q
        f[FEATURE_IDX["r_err"]] = r_rate - t_r

        # 32-34 期望地速
        t_vn, t_ve, t_vd = s["target_vn"], s["target_ve"], s["target_vd"]
        f[FEATURE_IDX["target_vn"]] = t_vn
        f[FEATURE_IDX["target_ve"]] = t_ve
        f[FEATURE_IDX["target_vd"]] = t_vd

        # 35-37 地速误差（actual - target）
        f[FEATURE_IDX["vn_err"]] = vel_n - t_vn
        f[FEATURE_IDX["ve_err"]] = vel_e - t_ve
        f[FEATURE_IDX["vd_err"]] = vel_d - t_vd

        # 38-41 实际舵面
        f[FEATURE_IDX["aileron_act"]] = s["aileron_actual"]
        f[FEATURE_IDX["elevator_act"]] = s["elevator_actual"]
        f[FEATURE_IDX["rudder_act"]] = s["rudder_actual"]
        f[FEATURE_IDX["throttle_act"]] = s["throttle_actual"]

        # 42-44 原始 IMU 机体加速度
        f[FEATURE_IDX["imu_ax"]] = s["imu_ax"]
        f[FEATURE_IDX["imu_ay"]] = s["imu_ay"]
        f[FEATURE_IDX["imu_az"]] = s["imu_az"]

        return f
