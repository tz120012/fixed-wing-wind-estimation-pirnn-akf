"""
PIRNN-AKF 融合模块 - 论文核心方法
Physics-Informed Recurrent Neural Network augmented Adaptive Kalman Filter

核心改进：
  1. AKF 连续运行，只在段边界 reset，不再每个样本 reset
  2. 利用 20 维输入中的 GPS 地速 / 姿态 / 空速构造风伪量测
  3. PI-GRU 提供 q_scale / r_scale / confidence / [Δα, Δβ, s_tas]
  4. AKF 以物理量测为主、NN 风估计为辅，输出连续融合结果
"""

import numpy as np
import pickle
import yaml
import os
from typing import Dict, Optional, Tuple
from collections import deque

try:
    from tqdm import tqdm
except ImportError:  # Raspberry Pi ONNX runtime does not need tqdm
    def tqdm(iterable, **_kwargs):
        return iterable

# ─────────────────────────────────────────────
# 特征下标常量（须与 src/1_preprocessing_data.py FEATURE_IDX 一致）
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

def _load_pigru_class():
    """Load PIGRU only when a PyTorch checkpoint path is used."""
    import importlib.util
    model_file = os.path.join(os.path.dirname(__file__), '..', 'src', '2_pigru_module.py')
    spec = importlib.util.spec_from_file_location("model_definition", model_file)
    if spec and spec.loader:
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        return model_module.PIGRU
    raise ImportError("Cannot load 2_pigru_module.py")


class AdaptiveKalmanFilter:
    """
    自适应卡尔曼滤波器（3维风状态）

    状态向量: x = [wind_N, wind_E, wind_D]^T
    过程模型: 随机游走 + 可选神经增量补偿

    量测来源：
      1. 运动学风伪量测（主量测）
         w_meas = v_gps_ned - R_b2n @ v_air_body
      2. PI-GRU 风估计（辅量测）

    设计目标：
      - 尽量贴近 EKF2 的“连续状态估计”范式
      - 保留 PI-GRU 对 Q/R 的自适应调节能力
    """

    def __init__(self,
                 dt: float = 0.02,
                 Q_nominal: Optional[np.ndarray] = None,
                 R_kin_nominal: Optional[np.ndarray] = None,
                 R_nn_nominal: Optional[np.ndarray] = None,
                 prediction_delta_gain: float = 0.25,
                 outlier_threshold: float = 9.0):
        self.dt = dt
        self.n_states = 3
        self.n_meas = 3
        self.prediction_delta_gain = float(prediction_delta_gain)
        self.outlier_threshold = float(outlier_threshold)

        # Q_nominal: 过程噪声，dt=0.05s 时每步方差增量 = Q*dt ≈ 0.0025 m²/s²
        # 允许风速以约 0.07 m/s/step 的速率随机游走，足以跟踪中等湍流
        self.Q_nominal = Q_nominal if Q_nominal is not None else np.diag([0.05, 0.05, 0.02])
        # R_kin_nominal: 运动学风伪量测噪声；EKF 北向/东向 RMSE ≈ 0.76/0.80 m/s → 方差 ≈ 0.58/0.64
        self.R_kin_nominal = R_kin_nominal if R_kin_nominal is not None else np.diag([1.00, 1.00, 0.36])
        # R_nn_nominal: PI-GRU 测量噪声，标定为 ~1.2× 实测方差
        # PI-GRU 各轴 RMSE ≈ 0.22 m/s (N/E) / 0.18 m/s (D) → 方差 ≈ 0.048/0.032
        # 原值 [2.25, 2.25, 1.00] 比实际大 40-50 倍，导致 Kalman 增益趋近于 0，AKF 无效
        self.R_nn_nominal = R_nn_nominal if R_nn_nominal is not None else np.diag([0.06, 0.06, 0.04])

        self.F = np.eye(self.n_states)
        self.H = np.eye(self.n_states)

        self.reset()

    def reset(self):
        self.x = np.zeros(self.n_states, dtype=np.float64)
        self.P = np.diag([2.0, 2.0, 0.6]).astype(np.float64)
        self.Q = self.Q_nominal.copy().astype(np.float64)
        self.R = self.R_kin_nominal.copy().astype(np.float64)

        self.q_scale_current = np.ones(3, dtype=np.float64)
        self.r_scale_current = np.ones(3, dtype=np.float64)

        self.innovation_history = deque(maxlen=20)
        self.last_innovation = np.zeros(self.n_meas, dtype=np.float64)
        self.last_nis = np.nan
        self.last_gain = np.zeros((self.n_states, self.n_meas), dtype=np.float64)
        self.last_measurement_label = 'none'
        self.last_wind_kin = np.zeros(3, dtype=np.float64)
        self.last_measurement_gap = 0.0

    @staticmethod
    def euler_to_R_b2n(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """欧拉角 -> 机体系到 NED 的旋转矩阵。"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        return np.array([
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp,     sr * cp,                  cr * cp]
        ], dtype=np.float64)

    @staticmethod
    def build_air_velocity_body(tas: float, angles: Optional[np.ndarray] = None) -> np.ndarray:
        """
        用 PI-GRU 输出的小角修正构造机体系空速向量。
        约定：body-z 为向下正，与训练阶段保持一致。
        """
        if angles is None:
            d_alpha, d_beta, s_tas = 0.0, 0.0, 1.0
        else:
            d_alpha, d_beta, s_tas = [float(v) for v in angles]

        tas_eff = max(float(tas), 0.1) * max(float(s_tas), 0.1)
        ca, sa = np.cos(d_alpha), np.sin(d_alpha)
        cb, sb = np.cos(d_beta), np.sin(d_beta)

        return np.array([
            tas_eff * ca * cb,
            tas_eff * sb,
            tas_eff * sa * cb
        ], dtype=np.float64)

    def construct_kinematic_wind_measurement(self,
                                             vg_ned: np.ndarray,
                                             tas: float,
                                             roll: float,
                                             pitch: float,
                                             yaw: float,
                                             angles: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """由地速、姿态、空速构造风伪量测。"""
        vg_ned = np.asarray(vg_ned, dtype=np.float64)
        v_air_body = self.build_air_velocity_body(tas, angles=angles)
        R_b2n = self.euler_to_R_b2n(roll, pitch, yaw)
        v_air_ned = R_b2n @ v_air_body
        wind_kin = vg_ned - v_air_ned
        return wind_kin, v_air_body, v_air_ned

    def update_process_covariance(self, q_scale: np.ndarray):
        q_scale = np.asarray(q_scale, dtype=np.float64)
        q_scale = np.clip(q_scale[:3], 0.1, 20.0)
        self.q_scale_current = q_scale
        self.Q = np.diag(np.diag(self.Q_nominal) * q_scale)

    def update_noise_covariance(self, q_scale: np.ndarray, r_scale: np.ndarray):
        """保留兼容接口：更新当前 q/r scale。"""
        self.update_process_covariance(q_scale)
        r_scale = np.asarray(r_scale, dtype=np.float64)
        self.r_scale_current = np.clip(r_scale[:3], 0.1, 20.0)
        self.R = np.diag(np.diag(self.R_kin_nominal) * self.r_scale_current)

    def predict(self, neural_wind_delta: Optional[np.ndarray] = None):
        """
        预测步骤：随机游走 + 轻量神经增量补偿。
        neural_wind_delta 来自相邻样本的 PI-GRU 风估计变化量。
        """
        delta = np.zeros(self.n_states, dtype=np.float64)
        if neural_wind_delta is not None:
            delta = np.asarray(neural_wind_delta, dtype=np.float64)
            delta = self.prediction_delta_gain * np.clip(delta, -1.0, 1.0)

        self.x = self.F @ self.x + delta
        self.P = self.F @ self.P @ self.F.T + self.Q * self.dt
        self.P = (self.P + self.P.T) / 2.0

    def _measurement_update(self,
                            z: np.ndarray,
                            R_override: np.ndarray,
                            label: str,
                            outlier_threshold: Optional[float] = None) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        R = np.asarray(R_override, dtype=np.float64)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        mahal = float(innovation.T @ S_inv @ innovation)
        threshold = self.outlier_threshold if outlier_threshold is None else float(outlier_threshold)

        if mahal > threshold:
            gate_factor = np.sqrt(threshold / max(mahal, 1e-9))
            gate_factor = np.clip(gate_factor, 0.1, 1.0)
        else:
            gate_factor = 1.0

        K = gate_factor * (self.P @ self.H.T @ S_inv)
        self.x = self.x + K @ innovation

        I = np.eye(self.n_states)
        IKH = I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = (self.P + self.P.T) / 2.0

        P_floor = np.diag(np.maximum(np.diag(self.Q_nominal) * 2.0, np.array([1e-4, 1e-4, 1e-4])))
        self.P = np.maximum(self.P, P_floor)

        self.R = R.copy()
        self.last_innovation = innovation.copy()
        self.last_gain = K.copy()
        self.last_nis = mahal
        self.last_measurement_label = label
        self.innovation_history.append(innovation.copy())
        return innovation

    def _build_kinematic_R(self,
                           confidence: float,
                           maneuver_score: float,
                           airspeed: float,
                           gap_mag: float = 0.0) -> np.ndarray:
        conf_scale = np.clip(1.15 - 0.35 * confidence, 0.75, 1.25)
        maneuver_scale = 1.0 + 0.35 * np.clip(maneuver_score, 0.0, 3.0)
        low_speed_scale = 1.0 + np.clip((8.0 - airspeed) / 8.0, 0.0, 1.5)
        gap_scale = 1.0 + np.clip(gap_mag / 2.0, 0.0, 4.0)
        scale = self.r_scale_current * conf_scale * maneuver_scale * low_speed_scale * gap_scale
        return np.diag(np.diag(self.R_kin_nominal) * scale)


    def _build_nn_R(self, confidence: float, gap_vec: np.ndarray) -> np.ndarray:
        gap_mag = float(np.linalg.norm(gap_vec))
        conf_scale = np.clip(1.35 - 0.70 * confidence, 0.70, 1.40)
        gap_scale = 1.0 + np.clip(gap_mag / 2.0, 0.0, 3.0)
        scale = self.r_scale_current * conf_scale * gap_scale
        return np.diag(np.diag(self.R_nn_nominal) * scale)

    def update(self,
               z: np.ndarray,
               roll: Optional[float] = None,
               pitch: Optional[float] = None,
               yaw: Optional[float] = None,
               confidence: float = 0.5,
               angles: Optional[np.ndarray] = None,
               nn_measurement: Optional[np.ndarray] = None,
               maneuver_score: float = 0.0,
               wind_kin_override: Optional[np.ndarray] = None) -> np.ndarray:
        """
        兼容接口：
          - z.shape == (3,): 直接把 z 当作风量测
          - z.shape >= (4,): 解释为 [vg_n, vg_e, vg_d, tas]，构造物理风伪量测
        """
        z = np.asarray(z, dtype=np.float64)

        if z.shape[0] == 3 and roll is None:
            R_nn = self._build_nn_R(confidence, np.zeros(3, dtype=np.float64))
            return self._measurement_update(z, R_override=R_nn, label='nn_only')

        if z.shape[0] < 4 or roll is None or pitch is None or yaw is None:
            raise ValueError('AKF.update 需要 [vg_n, vg_e, vg_d, tas] + 姿态角，或直接传入 3 维风量测')

        vg_ned = z[:3]
        tas = float(z[3])
        wind_kin_raw, _, _ = self.construct_kinematic_wind_measurement(
            vg_ned=vg_ned,
            tas=tas,
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            angles=angles
        )
        wind_kin = np.asarray(
            wind_kin_override if wind_kin_override is not None else wind_kin_raw,
            dtype=np.float64
        )
        self.last_wind_kin = wind_kin.copy()

        if nn_measurement is not None:
            gap_vec = np.asarray(nn_measurement, dtype=np.float64) - wind_kin
            self.last_measurement_gap = float(np.linalg.norm(gap_vec))
        else:
            gap_vec = np.zeros(3, dtype=np.float64)
            self.last_measurement_gap = 0.0

        R_kin = self._build_kinematic_R(
            confidence,
            maneuver_score,
            tas,
            gap_mag=self.last_measurement_gap
        )
        innovation = self._measurement_update(wind_kin, R_override=R_kin, label='kinematic')

        if nn_measurement is not None:
            R_nn = self._build_nn_R(confidence, gap_vec)
            innovation = self._measurement_update(
                np.asarray(nn_measurement, dtype=np.float64),
                R_override=R_nn,
                label='nn_aux'
            )

        return innovation


    def get_wind_estimate(self) -> np.ndarray:
        return self.x.copy()

    def get_diagnostics(self) -> Dict[str, np.ndarray]:
        return {
            'state': self.x.copy(),
            'P_diag': np.diag(self.P).copy(),
            'Q_diag': np.diag(self.Q).copy(),
            'R_diag': np.diag(self.R).copy(),
            'innovation': self.last_innovation.copy(),
            'innovation_norm': float(np.linalg.norm(self.last_innovation)),
            'nis': float(self.last_nis),
            'gain': self.last_gain.copy(),
            'measurement_label': self.last_measurement_label,
            'wind_kin': self.last_wind_kin.copy(),
            'measurement_gap': float(self.last_measurement_gap)
        }


class PIRNN_AKF:
    """
    PIRNN-AKF 融合估计器

    工作流程：
      1. PI-GRU 输出 wind_nn / q_scale / r_scale / confidence / [Δα, Δβ, s_tas]
      2. 用 20 维输入末帧的地速 / 姿态 / 空速构造风伪量测 wind_kin
      3. AKF 段内连续运行：predict -> kinematic update -> nn auxiliary update
      4. 输出以 AKF 为主、NN 为辅的连续融合结果
    """

    FEATURE_IDX = {
        'vel_n': 0, 'vel_e': 1, 'vel_d': 2,
        'vel_x_body': 3, 'vel_y_body': 4, 'vel_z_body': 5,
        'acc_x': 6, 'acc_y': 7, 'acc_z': 8,
        'roll': 9, 'pitch': 10, 'yaw': 11,
        'gyro_x': 12, 'gyro_y': 13, 'gyro_z': 14,
        'aileron': 15, 'elevator': 16, 'rudder': 17,
        'throttle': 18,
        'airspeed': 19,
    }

    def __init__(self, config_path: Optional[str] = None, model_path: Optional[str] = None,
                 enable_q_scale: bool = True, enable_r_scale: bool = True):
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            config_path = os.path.join(project_root, 'config', 'config.yaml')

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.enable_q_scale = enable_q_scale
        self.enable_r_scale = enable_r_scale
        akf_cfg = self.config.get('akf', {}) or {}
        constants = {
            'prediction_delta_gain': 0.25,
            'mahalanobis_gate': 9.0,
            'maneuver_gyro_weight': 0.45,
            'maneuver_acc_weight': 0.30,
            'maneuver_control_weight': 0.20,
            'maneuver_throttle_weight': 0.05,
            'disagreement_scale': 3.0,
            'kinematic_trust_base': 0.90,
            'kinematic_trust_slope': 0.70,
            'kinematic_trust_min': 0.20,
            'kinematic_trust_max': 0.90,
            'q_maneuver_gain': 0.45,
            'r_maneuver_gain': 0.20,
            'r_disagreement_gain': 1.50,
            'r_outlier_multiplier': 2.0,
            'fusion_base': 0.58,
            'fusion_confidence_gain': 0.10,
            'fusion_disagreement_gain': 0.28,
            'fusion_covariance_gain': 0.10,
            'fusion_min': 0.18,
            'fusion_max': 0.72,
            'fusion_outlier_cap': 0.28,
        }
        constants.update(akf_cfg.get('constants', {}) or {})
        self.akf_constants = {key: float(value) for key, value in constants.items()}
        self.response_profile = str(akf_cfg.get('response_profile', 'smooth')).lower()
        if self.response_profile not in {'smooth', 'tracking'}:
            self.response_profile = 'smooth'

        self.load_normalization_params()
        self.load_pigru_model(model_path)

        self.akf = AdaptiveKalmanFilter(
            dt=1.0 / self.config['data']['sampling_rate'],
            Q_nominal=np.diag(akf_cfg.get('q_nominal', [0.05, 0.05, 0.02])),
            R_kin_nominal=np.diag(akf_cfg.get('r_kin_nominal', [1.00, 1.00, 0.36])),
            R_nn_nominal=np.diag(akf_cfg.get('r_nn_nominal', [0.06, 0.06, 0.04])),
            prediction_delta_gain=self.akf_constants['prediction_delta_gain'],
            outlier_threshold=self.akf_constants['mahalanobis_gate'],
        )

        self.prev_log_q = None
        self.prev_log_r = None
        self.prev_wind_nn = None
        self.akf_initialized = False

        print(f"\n【PIRNN-AKF 初始化完成】")
        print(f"  设备: {self.device}")
        print(f"  采样率: {self.config['data']['sampling_rate']} Hz")
        print(f"  AKF响应模式: {self.response_profile}")

    def load_pigru_model(self, model_path: Optional[str] = None):
        """加载 PI-GRU 模型。"""
        if model_path is None:
            model_save_path = self.config['training']['model_save_path']
            if not os.path.isabs(model_save_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                model_save_path = os.path.join(project_root, model_save_path.lstrip('../'))

            train_dirs = [
                d for d in os.listdir(model_save_path)
                if d.startswith('train_') and os.path.isdir(os.path.join(model_save_path, d))
            ]
            if not train_dirs:
                raise FileNotFoundError("未找到PI-GRU训练目录")
            train_dirs.sort(reverse=True)
            model_path = os.path.join(model_save_path, train_dirs[0], 'best_model.pth')

        import torch
        PIGRU = _load_pigru_class()
        print(f"加载PI-GRU模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 优先从 checkpoint 读取模型配置，确保与训练时完全一致
        ckpt_model_cfg = checkpoint.get('config', {}).get('model', {})
        yaw_inv = ckpt_model_cfg.get('yaw_invariant', False)
        # yaw_invariant=True 时需传 norm_params（已在 load_normalization_params 中加载）
        norm_params_arg = None
        if yaw_inv and hasattr(self, 'scaler_X'):
            norm_params_arg = {
                'X_mean': self.x_mean,
                'X_scale': self.x_std,
                'y_mean': self.y_mean,
                'y_scale': self.y_std,
            }
        self.pigru = PIGRU(
            input_size=ckpt_model_cfg.get('input_size', self.config['model']['input_size']),
            hidden_size=ckpt_model_cfg.get('hidden_size', self.config['model']['hidden_size']),
            num_layers=ckpt_model_cfg.get('num_layers', self.config['model']['num_layers']),
            dropout=0.0,
            enable_wind_head=True,
            enable_noise_heads=ckpt_model_cfg.get('enable_noise_heads', False),
            enable_angles_head=ckpt_model_cfg.get('enable_angles_head', False),
            enable_confidence_head=ckpt_model_cfg.get('enable_confidence_head', False),
            yaw_invariant=yaw_inv,
            norm_params=norm_params_arg,
        ).to(self.device)

        self.pigru.load_state_dict(checkpoint['model_state_dict'])
        self.pigru.eval()
        print("  ✓ PI-GRU加载成功")

    def load_normalization_params(self):
        """加载归一化参数。优先使用预处理目录，回退到训练目录。"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)

        # 候选路径列表（按优先级）
        candidate_paths = []
        # 1. 预处理数据目录（最可靠）
        data_proc_path = self.config.get('data', {}).get('processed_dir', '')
        if data_proc_path:
            if not os.path.isabs(data_proc_path):
                data_proc_path = os.path.join(project_root, data_proc_path.lstrip('../'))
            candidate_paths.append(os.path.join(data_proc_path, 'norm_params.pkl'))
        # 2. 遍历项目下已知处理目录名
        for guess in ['data/data_1_processed_temporal', 'data/processed', 'data']:
            candidate_paths.append(os.path.join(project_root, guess, 'norm_params.pkl'))
        # 3. 训练输出目录（config 默认路径）
        model_save_path = self.config['training']['model_save_path']
        if not os.path.isabs(model_save_path):
            model_save_path = os.path.join(project_root, model_save_path.lstrip('../'))
        candidate_paths.append(os.path.join(model_save_path, 'norm_params.pkl'))

        norm_path = None
        for p in candidate_paths:
            if os.path.isfile(p):
                norm_path = p
                break
        if norm_path is None:
            raise FileNotFoundError(
                f"norm_params.pkl 未找到，已尝试路径：{candidate_paths}")

        print(f"  加载归一化参数: {norm_path}")
        with open(norm_path, 'rb') as f:
            metadata = pickle.load(f)

        self.scaler_X = metadata['scaler_X']
        self.scaler_y = metadata['scaler_y']

        self.x_mean = self.scaler_X.mean_
        self.x_std = self.scaler_X.scale_
        self.y_mean = self.scaler_y.mean_
        self.y_std = self.scaler_y.scale_

        self.wind_mean = self.y_mean[0:3]
        self.wind_std = self.y_std[0:3]

    def reset(self):
        """按段重置融合估计器。"""
        self.akf.reset()
        self.prev_log_q = None
        self.prev_log_r = None
        self.prev_wind_nn = None
        self.akf_initialized = False

    def _denormalize_last_step(self, X_sequence: np.ndarray) -> np.ndarray:
        last_step = np.asarray(X_sequence[-1], dtype=np.float64)
        return last_step * self.x_std + self.x_mean

    def _compute_maneuver_score(self, last_step_phys: np.ndarray) -> float:
        gyro = last_step_phys[FEATURE_IDX["p_rate"]:FEATURE_IDX["r_rate"] + 1]
        acc = last_step_phys[FEATURE_IDX["ax"]:FEATURE_IDX["az"] + 1]
        ctrl = last_step_phys[FEATURE_IDX["aileron_cmd"]:FEATURE_IDX["rudder_cmd"] + 1]
        throttle = float(last_step_phys[FEATURE_IDX["throttle_cmd"]])

        gyro_score = np.linalg.norm(gyro) / 1.2
        acc_score = np.linalg.norm(acc) / 8.0
        ctrl_score = np.linalg.norm(ctrl) / 1.2
        throttle_score = abs(throttle - 0.5) * 2.0

        c = self.akf_constants
        maneuver = (
            c['maneuver_gyro_weight'] * gyro_score
            + c['maneuver_acc_weight'] * acc_score
            + c['maneuver_control_weight'] * ctrl_score
            + c['maneuver_throttle_weight'] * throttle_score
        )
        return float(np.clip(maneuver, 0.0, 3.0))

    def _normalize_wind(self, wind_phys: np.ndarray) -> np.ndarray:
        return (np.asarray(wind_phys, dtype=np.float64) - self.wind_mean) / self.wind_std

    def _stabilize_kinematic_measurement(self,
                                         wind_kin: np.ndarray,
                                         wind_nn: np.ndarray) -> Tuple[np.ndarray, float, float, bool]:
        """当wind_kin与NN分歧过大时，软约束其回到更保守的范围。"""
        wind_kin = np.asarray(wind_kin, dtype=np.float64)
        wind_nn = np.asarray(wind_nn, dtype=np.float64)
        gap_vec = wind_kin - wind_nn
        measurement_gap = float(np.linalg.norm(gap_vec))
        c = self.akf_constants
        gap_ratio = np.clip(measurement_gap / c['disagreement_scale'], 0.0, 1.0)

        wind_limit = float(self.config.get('physics', {}).get('wind_magnitude_max', 15.0))
        kin_norm = float(np.linalg.norm(wind_kin))
        kin_outlier = measurement_gap > 4.0 or kin_norm > wind_limit * 1.2

        kin_trust = np.clip(
            c['kinematic_trust_base'] - c['kinematic_trust_slope'] * gap_ratio,
            c['kinematic_trust_min'],
            c['kinematic_trust_max'],
        )
        if kin_outlier:
            kin_trust = min(kin_trust, 0.35)

        wind_kin_stable = kin_trust * wind_kin + (1.0 - kin_trust) * wind_nn
        return wind_kin_stable, measurement_gap, float(kin_trust), bool(kin_outlier)

    def estimate_sequence(self, X_sequence: np.ndarray, reset_filter: bool = False) -> Dict:

        """
        对单个序列进行融合估计。

        关键变化：
          - 默认不 reset AKF，连续使用上一时刻状态
          - 只有段起点（由外部传入 reset_filter=True）才 reset
        """
        if reset_filter:
            self.reset()

        import torch
        X_tensor = torch.FloatTensor(X_sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.pigru.predict_online(
                X_tensor,
                prev_log_q=self.prev_log_q,
                prev_log_r=self.prev_log_r,
                ema_alpha=0.1,
                clamp=True
            )
            self.prev_log_q = out['log_q_scale']
            self.prev_log_r = out['log_r_scale']

        wind_nn_norm = out['wind_estimate'].cpu().numpy()[0]
        q_scale = out['q_scale'].cpu().numpy()[0]
        r_scale = out['r_scale'].cpu().numpy()[0]
        angles = out['angles'].cpu().numpy()[0]

        if 'confidence' in out and out['confidence'] is not None:
            confidence = float(out['confidence'].cpu().numpy()[0, 0])
        else:
            confidence = 1.0 / (1.0 + np.std(q_scale))

        wind_nn = wind_nn_norm * self.wind_std + self.wind_mean
        last_step_phys = self._denormalize_last_step(X_sequence)

        vg_ned = last_step_phys[FEATURE_IDX["vel_n"]:FEATURE_IDX["vel_d"] + 1]
        roll = float(last_step_phys[FEATURE_IDX["roll"]])
        pitch = float(last_step_phys[FEATURE_IDX["pitch"]])
        yaw = float(last_step_phys[FEATURE_IDX["yaw"]])
        airspeed = float(max(last_step_phys[FEATURE_IDX["airspeed"]], 0.1))
        maneuver_score = self._compute_maneuver_score(last_step_phys)

        wind_kin, _, _ = self.akf.construct_kinematic_wind_measurement(
            vg_ned=vg_ned,
            tas=airspeed,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            angles=angles
        )
        wind_kin_stable, measurement_gap, kin_trust, kin_outlier = self._stabilize_kinematic_measurement(
            wind_kin,
            wind_nn
        )

        c = self.akf_constants
        gap_ratio = np.clip(measurement_gap / c['disagreement_scale'], 0.0, 1.0)
        q_scale_raw = q_scale if self.enable_q_scale else np.ones(3, dtype=np.float64)
        r_scale_raw = r_scale if self.enable_r_scale else np.ones(3, dtype=np.float64)
        q_scale_eff = np.clip(
            q_scale_raw * (1.0 + c['q_maneuver_gain'] * maneuver_score),
            0.1,
            20.0,
        )
        r_scale_gain = 1.0 + c['r_disagreement_gain'] * gap_ratio
        if kin_outlier:
            r_scale_gain *= c['r_outlier_multiplier']
        r_scale_eff = np.clip(
            r_scale_raw
            * (1.0 + c['r_maneuver_gain'] * maneuver_score)
            * r_scale_gain,
            0.1,
            20.0,
        )

        if not self.akf_initialized:
            self.akf.reset()
            self.akf.x = (0.80 * wind_nn.astype(np.float64) + 0.20 * wind_kin_stable.astype(np.float64))
            self.akf.P = np.diag([2.0, 2.0, 0.6]).astype(np.float64)
            self.akf_initialized = True
            neural_delta = np.zeros(3, dtype=np.float64)
        else:
            neural_delta = wind_nn - self.prev_wind_nn if self.prev_wind_nn is not None else np.zeros(3, dtype=np.float64)

        self.akf.update_noise_covariance(q_scale_eff, r_scale_eff)
        self.akf.predict(neural_wind_delta=neural_delta)

        z_meas = np.array([vg_ned[0], vg_ned[1], vg_ned[2], airspeed], dtype=np.float64)
        innovation = self.akf.update(
            z_meas,
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            confidence=confidence,
            angles=angles,
            nn_measurement=wind_nn,
            maneuver_score=maneuver_score,
            wind_kin_override=wind_kin_stable
        )

        diagnostics = self.akf.get_diagnostics()
        wind_akf = self.akf.get_wind_estimate()

        p_ratio = np.clip(np.mean(diagnostics['P_diag']) / 2.0, 0.0, 1.0)
        if self.response_profile == 'tracking':
            maneuver_ratio = np.clip(maneuver_score / 2.0, 0.0, 1.0)
            akf_weight = np.clip(
                0.45 + 0.06 * confidence - 0.32 * gap_ratio - 0.12 * p_ratio - 0.10 * maneuver_ratio,
                0.10,
                0.62,
            )
        else:
            akf_weight = np.clip(
                c['fusion_base']
                + c['fusion_confidence_gain'] * confidence
                - c['fusion_disagreement_gain'] * gap_ratio
                - c['fusion_covariance_gain'] * p_ratio,
                c['fusion_min'],
                c['fusion_max'],
            )
        if kin_outlier:
            akf_weight = min(akf_weight, c['fusion_outlier_cap'])
        nn_weight = 1.0 - akf_weight

        wind_fused = akf_weight * wind_akf + nn_weight * wind_nn

        wind_fused_norm = self._normalize_wind(wind_fused)

        self.prev_wind_nn = wind_nn.copy()

        return {
            'wind_estimate': wind_fused_norm,
            'wind_nn': wind_nn_norm,
            'wind_akf': self._normalize_wind(wind_akf),
            'wind_kin': self._normalize_wind(wind_kin),
            'q_scale': q_scale_eff,
            'r_scale': r_scale_eff,
            'angles': angles,
            'confidence': confidence,
            'nn_weight': float(nn_weight),
            'akf_weight': float(akf_weight),
            'innovation': innovation,
            'innovation_norm': diagnostics['innovation_norm'],
            'nis': diagnostics['nis'],
            'P_diag': diagnostics['P_diag'],
            'Q_diag': diagnostics['Q_diag'],
            'R_diag': diagnostics['R_diag'],
            'maneuver_score': float(maneuver_score),
            'measurement_gap': measurement_gap,
        }

    def estimate_batch(self, X_test: np.ndarray, continuous: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        批量估计。

        Args:
            X_test: [N, seq_len, feat] 输入样本
            continuous: 为 True 时仅首样本 reset、随后连续滤波；
                        为 False 时每个样本独立 reset，适用于乱序/滑窗测试集。
        """
        N = X_test.shape[0]
        wind_estimates = np.zeros((N, 3))
        wind_nn = np.zeros((N, 3))
        wind_akf = np.zeros((N, 3))
        wind_kin = np.zeros((N, 3))
        q_scales = np.zeros((N, 3))
        r_scales = np.zeros((N, 3))
        confidences = np.zeros(N)
        nn_weights = np.zeros(N)
        akf_weights = np.zeros(N)
        innovations = np.zeros((N, self.akf.n_meas))
        innovation_norms = np.zeros(N)
        nis_values = np.zeros(N)
        p_diags = np.zeros((N, self.akf.n_states))
        q_diags = np.zeros((N, self.akf.n_states))
        r_diags = np.zeros((N, self.akf.n_meas))
        maneuver_scores = np.zeros(N)
        measurement_gaps = np.zeros(N)

        self.reset()

        for i in tqdm(range(N), desc='PIRNN-AKF估计'):
            result = self.estimate_sequence(
                X_test[i],
                reset_filter=(not continuous or i == 0)
            )
            wind_estimates[i] = result['wind_estimate']
            wind_nn[i] = result['wind_nn']
            wind_akf[i] = result['wind_akf']
            wind_kin[i] = result['wind_kin']
            q_scales[i] = result['q_scale']
            r_scales[i] = result['r_scale']
            confidences[i] = result['confidence']
            nn_weights[i] = result['nn_weight']
            akf_weights[i] = result['akf_weight']
            innovations[i] = result['innovation']
            innovation_norms[i] = result['innovation_norm']
            nis_values[i] = result['nis']
            p_diags[i] = result['P_diag']
            q_diags[i] = result['Q_diag']
            r_diags[i] = result['R_diag']
            maneuver_scores[i] = result['maneuver_score']
            measurement_gaps[i] = result['measurement_gap']


        return wind_estimates, {
            'wind_nn': wind_nn,
            'wind_akf': wind_akf,
            'wind_kin': wind_kin,
            'q_scale': q_scales,
            'r_scale': r_scales,
            'confidence': confidences,
            'nn_weight': nn_weights,
            'akf_weight': akf_weights,
            'innovation': innovations,
            'innovation_norm': innovation_norms,
            'nis': nis_values,
            'P_diag': p_diags,
            'Q_diag': q_diags,
            'R_diag': r_diags,
            'maneuver_score': maneuver_scores,
            'measurement_gap': measurement_gaps,
        }

    def get_model_info(self) -> Dict:
        return {
            'model_type': 'PIRNN-AKF',
            'has_physics_loss': True,
            'has_adaptive_params': True,
            'fusion_method': 'Continuous Kinematic AKF + PI-GRU Adaptive Covariance',
            'neural_backbone': 'PI-GRU'
        }


if __name__ == "__main__":
    print("=" * 70)
    print(" PIRNN-AKF 融合估计器测试")
    print("=" * 70)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, 'config', 'config.yaml')

        pirnn_akf = PIRNN_AKF(config_path=config_path)

        model_info = pirnn_akf.get_model_info()
        print(f"\n【模型信息】")
        for k, v in model_info.items():
            print(f"  {k}: {v}")

        print("\n【功能测试】")
        test_input = np.random.randn(50, 20).astype(np.float32)
        result = pirnn_akf.estimate_sequence(test_input, reset_filter=True)

        print(f"  ✓ 风速估计: {result['wind_estimate']}")
        print(f"  ✓ q_scale: {result['q_scale']}")
        print(f"  ✓ r_scale: {result['r_scale']}")
        print(f"  ✓ 置信度: {result['confidence']:.3f}")
        print(f"  ✓ 机动强度: {result['maneuver_score']:.3f}")

        print("\n" + "=" * 70)
        print("✅ PIRNN-AKF 测试完成")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
