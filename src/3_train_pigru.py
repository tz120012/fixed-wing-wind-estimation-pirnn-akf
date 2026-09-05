"""
训练模块 v3.0 (EKF融合增强版)
功能：训练PI-GRU模型，包含改进的物理损失和多通道自适应噪声
优化：
  - 支持字典输出模式
  - 姿态旋转的物理一致性
    - q_scale/r_scale 多通道独立调节
  - 小角修正正则化
  - 增强的监控指标
"""

import os
import copy
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
from matplotlib import font_manager
import matplotlib
import pickle
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

# ─────────────────────────────────────────────
# 特征下标常量（须与 src/1_preprocessing_data.py / 2_pigru_module.py FEATURE_IDX 一致）
# 当前数据集（240→50 Hz 下采样后）= 完整 45 维（阶段 2）。
# 阶段 1（input_size=32，不含 32–44）保留供回滚，本脚本兼容两种维度。
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

# 标签下标（与 1_preprocessing_data.py 中 lbl 对应）
LABEL_IDX = {
    "wind_n": 0, "wind_e": 1, "wind_d": 2,
    "vel_n": 3, "vel_e": 4, "vel_d": 5,
    "airspeed": 6,
}

# q_scale 波动监督目标列在 y 中的起始下标（追加在原 7 维标签之后）
Q_VOL_TARGET_START = 7


def append_volatility_targets(y_train, y_val, cfg):
    """为 q_scale 体态监督追加"真实风过程波动"目标列。

    现有 q_scale 监督的是风估计残差（≈ NN 输出不确定度，与过程噪声无关）。
    本函数在归一化风标签上计算逐步增量 |w_true[t]-w_true[t-1]|（per-axis），
    并按训练集 p10/p90 线性映射到 [lo, hi]（q_scale 量程内），作为新的监督目标，
    使 q_scale 真正承载"风变化快慢"信息。返回 (y_train_aug, y_val_aug, info)。

    注意：样本在 npy 中按文件分段时间连续，文件边界处的 diff 为少量离群，
    通过 p99 裁剪鲁棒化处理。
    """
    train_cfg = cfg.get('training', {})
    if str(train_cfg.get('q_supervision', 'residual')).lower() != 'volatility':
        return y_train, y_val, None

    lo = float(train_cfg.get('q_vol_target_lo', 0.5))
    hi = float(train_cfg.get('q_vol_target_hi', 3.5))
    win = int(train_cfg.get('q_vol_window', 50))

    def raw_vol(y):
        # 窗口化湍流强度：真实风的因果滚动标准差（per-axis）。
        # 单步增量本质是不可预测的高频噪声（autocorr≈0.2），网络会塌成常数；
        # 窗口化湍流强度高度持久（autocorr≈0.998），是可学习的过程波动信号。
        w = np.asarray(y[:, 0:3], dtype=np.float64)
        n = len(w)
        c = np.cumsum(np.insert(w, 0, 0.0, axis=0), axis=0)
        c2 = np.cumsum(np.insert(w * w, 0, 0.0, axis=0), axis=0)
        idx = np.arange(n)
        lo_idx = np.maximum(0, idx - win + 1)
        k = (idx + 1 - lo_idx).reshape(-1, 1)
        mean = (c[idx + 1] - c[lo_idx]) / k
        var = (c2[idx + 1] - c2[lo_idx]) / k - mean ** 2
        return np.sqrt(np.maximum(var, 0.0))

    vol_train = raw_vol(y_train)
    vol_val = raw_vol(y_val)

    # 鲁棒裁剪 + 按训练集 p10/p90 做 per-axis 线性映射到 [lo, hi]
    p99 = np.percentile(vol_train, 99, axis=0)
    vol_train = np.minimum(vol_train, p99)
    vol_val = np.minimum(vol_val, p99)
    p10 = np.percentile(vol_train, 10, axis=0)
    p90 = np.percentile(vol_train, 90, axis=0)
    span = np.maximum(p90 - p10, 1e-6)

    def to_target(vol):
        t = lo + (hi - lo) * (vol - p10) / span
        return np.clip(t, lo, hi).astype(y_train.dtype)

    tgt_train = to_target(vol_train)
    tgt_val = to_target(vol_val)
    y_train_aug = np.concatenate([y_train, tgt_train], axis=1)
    y_val_aug = np.concatenate([y_val, tgt_val], axis=1)
    info = {
        'p10': p10.tolist(), 'p90': p90.tolist(), 'lo': lo, 'hi': hi,
        'target_mean': tgt_train.mean(axis=0).tolist(),
        'target_std': tgt_train.std(axis=0).tolist(),
    }
    return y_train_aug, y_val_aug, info

# 导入模型定义
import importlib.util
model_file = os.path.join(os.path.dirname(__file__), '2_pigru_module.py')
spec = importlib.util.spec_from_file_location("model_definition", model_file)
if spec and spec.loader:
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    PIGRU = model_module.PIGRU
else:
    raise ImportError("Cannot load 2_pigru_module.py")


class _Tee:
    """同时写多个文件流；用于把 stdout/stderr 镜像到 logs/*.log。"""
    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        for s in self._streams:
            try:
                if s.isatty():
                    return True
            except Exception:
                pass
        return False


def setup_file_logging(project_root: str, args) -> str:
    """把 stdout/stderr 同步镜像到 logs/<lambda>_<mode>_<ts>.log。返回日志路径。"""
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    parts = ['train']
    if getattr(args, 'lambda_physics', None) is not None:
        parts.append(f"lambda{args.lambda_physics:g}")
    elif getattr(args, 'lambda_list', None):
        parts.append(f"lambda{args.lambda_list.replace(',', '_')}")
    if getattr(args, 'mode', None):
        parts.append(args.mode)
    if getattr(args, 'resume', None):
        parts.append('resume')
    parts.append(ts)
    log_path = os.path.join(log_dir, '_'.join(parts) + '.log')
    log_file = open(log_path, 'w', buffering=1)  # 行缓冲，便于 tail -f
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path


def normalize_physics_mode(mode: str) -> str:
    """标准化物理损失模式名称。"""
    normalized = mode.strip().lower()
    aliases = {
        '6-dof': '6dof',
        'att': 'attitude',
        'kinematic': 'attitude',
        'v2': 'attitude',
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {'simple', 'attitude', '6dof'}:
        raise ValueError(f"不支持的 physics_mode: {mode}")
    return normalized


def infer_physics_mode(physics_config: dict) -> str:
    """根据配置推断当前物理损失模式。"""
    if physics_config.get('use_6dof_physics', False):
        return '6dof'
    if physics_config.get('use_attitude_physics', True):
        return 'attitude'
    return 'simple'


def sanitize_run_tag(run_tag) -> str:
    """清理 run_tag，避免路径或日志名包含特殊字符。"""
    if not run_tag:
        return ''
    cleaned = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '-' for ch in run_tag.strip())
    return cleaned.strip('-_')


def resolve_project_path(project_root: str, path_value):
    """将配置中的相对路径解析为项目根目录下的绝对路径。"""
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(project_root, path_value.lstrip('../')))


TEMPORAL_PROFILE_KEYWORDS = ('temporal', 'chronological')
DIRECTION_FOCUS_PROFILE_KEYWORDS = (
    'data2_balanced_random_sr025',
    'data2_balanced_sr025',
    'direction_focus',
    'dirfocus',
)


def contains_temporal_hint(value) -> bool:
    """判断路径或标签中是否包含 temporal split 相关关键词。"""
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return any(keyword in text for keyword in TEMPORAL_PROFILE_KEYWORDS)


def contains_direction_focus_hint(value) -> bool:
    """判断路径或标签中是否包含 data2 / 风向优先训练相关关键词。"""
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return any(keyword in text for keyword in DIRECTION_FOCUS_PROFILE_KEYWORDS)


def apply_training_profile_overrides(config: dict, training_profile: str, run_tag: str = '',
                                     lambda_physics_override=None):
    """检测当前训练场景，仅用于日志记录，不修改任何参数。
    所有训练参数均以 config.yaml 为准。
    """
    training_cfg = config.setdefault('training', {})
    data_cfg = config.setdefault('data', {})

    temporal_hints = []
    direction_hints = []
    processed_dir = data_cfg.get('processed_dir')
    model_save_path = training_cfg.get('model_save_path')
    if contains_temporal_hint(processed_dir):
        temporal_hints.append(f"processed_dir={processed_dir}")
    if contains_temporal_hint(model_save_path):
        temporal_hints.append(f"model_save_path={model_save_path}")
    if contains_temporal_hint(run_tag):
        temporal_hints.append(f"run_tag={run_tag}")
    if contains_direction_focus_hint(processed_dir):
        direction_hints.append(f"processed_dir={processed_dir}")
    if contains_direction_focus_hint(model_save_path):
        direction_hints.append(f"model_save_path={model_save_path}")
    if contains_direction_focus_hint(run_tag):
        direction_hints.append(f"run_tag={run_tag}")

    if training_profile == 'default':
        reason = '显式指定 default'
        active_profile = 'default'
    elif training_profile == 'temporal_stable':
        reason = '命令行显式启用 temporal_stable'
        active_profile = 'temporal_stable'
    elif training_profile == 'direction_focus':
        reason = '命令行显式启用 direction_focus'
        active_profile = 'direction_focus'
    else:
        if direction_hints:
            reason = '自动检测到 direction_focus 场景: ' + ', '.join(direction_hints)
            active_profile = 'direction_focus'
        elif temporal_hints:
            reason = '自动检测到 temporal split 场景: ' + ', '.join(temporal_hints)
            active_profile = 'temporal_stable'
        else:
            reason = '未检测到特殊训练场景'
            active_profile = 'default'

    training_cfg['training_profile'] = active_profile
    training_cfg['training_profile_reason'] = reason
    training_cfg['training_profile_changes'] = []
    return active_profile, reason, []


class Trainer:
    def __init__(self, config_path=None, lambda_physics_override=None, physics_mode_override=None, run_tag=None,
                 processed_dir_override=None, model_save_path_override=None, training_profile='auto',
                 physics_warmup_epochs_override=None, angles_warmup_epochs_override=None):
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.run_tag = sanitize_run_tag(run_tag)
        if processed_dir_override is not None:
            self.config.setdefault('data', {})['processed_dir'] = processed_dir_override
        if model_save_path_override is not None:
            self.config.setdefault('training', {})['model_save_path'] = model_save_path_override

        self.training_profile, self.training_profile_reason, self.training_profile_changes = apply_training_profile_overrides(
            self.config,
            training_profile=training_profile,
            run_tag=self.run_tag,
            lambda_physics_override=lambda_physics_override,
        )
        if physics_warmup_epochs_override is not None:
            self.config.setdefault('training', {})['physics_warmup_epochs'] = int(physics_warmup_epochs_override)
        if angles_warmup_epochs_override is not None:
            self.config.setdefault('training', {})['angles_warmup_epochs'] = int(angles_warmup_epochs_override)

        physics_config = self.config.setdefault('physics', {})
        if physics_mode_override is not None:
            physics_mode = normalize_physics_mode(physics_mode_override)
            physics_config['use_6dof_physics'] = physics_mode == '6dof'
            physics_config['use_attitude_physics'] = physics_mode in {'attitude', '6dof'}

        self.physics_mode = infer_physics_mode(physics_config)
        self.config.setdefault('training', {})['physics_mode'] = self.physics_mode
        if self.run_tag:
            self.config['training']['run_tag'] = self.run_tag

        self.seed = int(self.config.get('training', {}).get('seed', 26))
        self.set_global_seed(self.seed)
        
        # 设置matplotlib参数
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        print(f"随机种子: {self.seed}")
        if self.device.type == 'cuda':
            print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 初始化模型
        self.model = PIGRU(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            rnn_type=self.config['model'].get('rnn_type', 'gru'),
            enable_wind_head=True,  # 训练时启用风速回归头
            enable_noise_heads=bool(self.config['model'].get('enable_noise_heads', True)),
            enable_angles_head=bool(self.config['model'].get('enable_angles_head', True)),
            enable_confidence_head=bool(self.config['model'].get('enable_confidence_head', True)),
            angle_limit_deg=self.config.get('physics', {}).get('angle_limit_deg', 5.0),
            s_tas_range=tuple(self.config.get('physics', {}).get('s_tas_range', [0.9, 1.1])),
            qr_scale_range=tuple(
                self.config.get('physics', {}).get(
                    'qr_scale_range',
                    self.config.get('physics', {}).get('alpha_beta_range', [0.3, 10.0])
                )
            )
        ).to(self.device)
        
        # 打印模型信息
        model_info = self.model.get_model_info()
        print(f"\n【模型配置】")
        print(f"  输入维度: {model_info['input_size']}")
        print(f"  隐藏层维度: {model_info['hidden_size']}")
        print(f"  GRU层数: {model_info['num_layers']}")
        print(f"  总参数量: {model_info['total_params']:,}")
        print(f"  可训练参数量: {model_info['trainable_params']:,}")
        print(f"  风速回归头: {'启用' if model_info['enable_wind_head'] else '禁用'}")
        print(f"  噪声自适应头: {'启用' if model_info['enable_noise_heads'] else '禁用'}")
        print(f"  角度修正头: {'启用' if model_info['enable_angles_head'] else '禁用'}")
        print(f"  置信度头: {'启用' if model_info['enable_confidence_head'] else '禁用'}")
        
        # 加载归一化参数
        self.load_normalization_params()
        self.load_attitude_norm_params()  # 新增：加载姿态归一化参数

        # ========== Yaw-invariant 后注入（B 方案）==========
        # 模型构造已在归一化参数加载之前完成。如果 config.model.yaw_invariant=True，
        # 这里把归一化参数注入模型，并启用 yaw-invariant 内部坐标变换。
        # 配合 A（rotation augmentation）形成"显式 + 隐式"双重 yaw-invariant 学习。
        if bool(self.config.get('model', {}).get('yaw_invariant', False)):
            self.model.yaw_invariant = True
            X_mean_t = torch.tensor(self.scaler_X.mean_, dtype=torch.float32)
            X_scale_t = torch.tensor(self.scaler_X.scale_, dtype=torch.float32)
            y_mean_t = torch.tensor(self.scaler_y.mean_, dtype=torch.float32)
            y_scale_t = torch.tensor(self.scaler_y.scale_, dtype=torch.float32)
            # register_buffer：确保随 .to(device) 自动迁移、随 state_dict 保存
            self.model.register_buffer('_X_mean', X_mean_t.to(self.device))
            self.model.register_buffer('_X_scale', X_scale_t.to(self.device))
            self.model.register_buffer('_y_mean', y_mean_t.to(self.device))
            self.model.register_buffer('_y_scale', y_scale_t.to(self.device))
            self.model.register_buffer('_track_along_mean', torch.tensor(15.05, device=self.device))
            self.model.register_buffer('_track_along_std',  torch.tensor(4.84, device=self.device))
            self.model.register_buffer('_track_cross_mean', torch.tensor(0.0, device=self.device))
            self.model.register_buffer('_track_cross_std',  torch.tensor(4.55, device=self.device))
            self.model.register_buffer('_yaw_diff_std',     torch.tensor(0.137, device=self.device))
            print(f"\n【Yaw-invariant 表示】已启用")
            print(f"  - 模型内部把 NED 输入旋转到 track 系（沿/垂直机体航向）")
            print(f"  - 输出风速自动反变换回 NED 归一化空间")
            print(f"  - track 系归一化常数(实测): vel_along={15.05:.2f}±{4.84:.2f}, vel_cross={0.0:.1f}±{4.55:.2f} m/s, yaw_diff_std={0.137:.3f} rad")

        # NaN 恢复机制：保存干净的权重快照
        self._weight_snapshot = None
        self._optimizer_state_snapshot = None
        self._nan_recovery_count = 0
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.0001)
        )
        
        # 学习率调度器
        scheduler_config = self.config.get('scheduler', {})
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 10),
            min_lr=scheduler_config.get('min_lr', 1e-6)
        )
        
        # 混合精度训练 (AMP) - 提升速度和减少显存
        # 新版 PyTorch (2.0+) API：torch.amp.GradScaler('cuda', ...)
        # init_scale 默认 65536（2^16）会让前 5–20 个 batch 几乎必爆 fp16，
        # 表现为大量 [NaN GRAD] 日志。我们的损失数值本身不大（~1–30），
        # 用 2^12 = 4096 起步可大幅缩短"探测期"，同时 growth_interval 调小
        # 让 scale 更快上调到稳定值。
        self.scaler = torch.amp.GradScaler(
            'cuda',
            init_scale=2.0**12,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=500,
            enabled=(self.device.type == 'cuda'),
        )
        self.use_amp = self.device.type == 'cuda'
        if self.use_amp:
            print(f"\n【混合精度训练】")
            print(f"  AMP 已启用 - 预计提升速度 2-3倍")
            
            # 启用 TF32 (仅RTX 30系列及Ampere+架构支持)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"  TF32 已启用 - Ampere/Blackwell 架构专属加速")
        
        # 损失权重与稳定训练参数
        self.lambda_physics = self.config['training'].get('lambda_physics', 1.0)
        if lambda_physics_override is not None:
            self.lambda_physics = float(lambda_physics_override)
        # 关键：将覆盖后的 lambda 写回 config，确保 checkpoint 中记录的值正确
        self.config['training']['lambda_physics'] = float(self.lambda_physics)
        self.lambda_wind = float(self.config['training'].get('lambda_wind', 1.0))
        self.lambda_dir = float(self.config['training'].get('lambda_dir', 0.3))
        self.lambda_mag = float(self.config['training'].get('lambda_mag', 0.2))
        # phys_dir 损失现为 cos 距离 ∈ [0, 2]，默认权重相应放大（旧默认 0.5 是按
        # acos 度数设计的，对新损失数值过小，会让物理方向监督失效）
        self.lambda_phys_dir = float(self.config['training'].get('lambda_phys_dir', 5.0))
        self.lambda_phys_down = float(self.config['training'].get('lambda_phys_down', 0.3))
        self.lambda_mag_relative = float(self.config['training'].get('lambda_mag_relative', 0.0))
        self.lambda_mag_under = float(self.config['training'].get('lambda_mag_under', 0.0))
        self.high_wind_threshold = float(self.config['training'].get('high_wind_threshold', 3.0))
        transition_cfg = self.config['training'].get('transition_boost', {}) or {}
        self.transition_boost_enabled = bool(transition_cfg.get('enabled', False))
        self.lambda_transition_mag = float(transition_cfg.get('lambda_transition_mag', 0.0))
        self.direction_loss_min_horizontal_wind = float(
            self.config['training'].get('direction_loss_min_horizontal_wind', 0.5)
        )

        # ========== 弱风段失效专项（evil-sample 分析后引入） ==========
        # ① 防崩塌：当 ||w_pred_h|| 低于阈值时给 hinge 平方惩罚，
        #    阻止模型在弱风段把预测压到 ~0 m/s 形成"假装无风"的偷懒解。
        self.lambda_anti_collapse = float(self.config['training'].get('lambda_anti_collapse', 0.0))
        self.anti_collapse_threshold = float(
            self.config['training'].get('anti_collapse_threshold', 0.3)
        )
        # ② SNR-aware 损失权重：弱风段（truth_h_mag < weak_threshold）
        #    降低 direction loss 权重（避免低 SNR 干扰梯度），同时
        #    提升 magnitude loss 权重（强迫模型保留幅值信号）
        snr_cfg = self.config['training'].get('snr_aware', {}) or {}
        self.snr_aware_enabled = bool(snr_cfg.get('enabled', False))
        self.snr_weak_threshold = float(snr_cfg.get('weak_threshold', 1.5))
        self.snr_dir_min_weight = float(snr_cfg.get('dir_min_weight', 0.2))
        self.snr_mag_boost = float(snr_cfg.get('mag_boost', 1.5))
        # ③ Wind-bin re-sampling：按真值水平风强度做反频率重采样权重，
        #    弱风样本数量稀少时显著提升其训练权重。
        bin_cfg = self.config['training'].get('bin_rebalance', {}) or {}
        self.bin_rebalance_enabled = bool(bin_cfg.get('enabled', False))
        self.bin_rebalance_n_bins = int(bin_cfg.get('n_bins', 12))
        self.bin_rebalance_alpha = float(bin_cfg.get('alpha', 0.5))
        self.bin_rebalance_max = float(bin_cfg.get('max_weight', 5.0))

        self.selection_alpha = float(self.config['training'].get('selection_alpha', 0.5))
        self.selection_beta = float(self.config['training'].get('selection_beta', 0.5))
        self.selection_gamma = float(self.config['training'].get('selection_gamma_high_under', 0.0))
        self.selection_delta = float(self.config['training'].get('selection_delta_transition', 0.0))
        self.selection_metric_name = self.config['training'].get('selection_metric', 'composite_score')
        if self.selection_metric_name == 'loss':
            self.selection_metric_name = 'total'
        self.lambda_reg = float(self.config['training'].get('lambda_reg', 0.01))
        self.physics_warmup_epochs = int(self.config['training'].get('physics_warmup_epochs', 20))
        self.reg_huber_delta = float(self.config['training'].get('reg_huber_delta', 0.5))
        self.reg_boundary_penalty = float(self.config['training'].get('reg_boundary_penalty', 10.0))
        self.reg_target_q_scale = torch.tensor(
            self.config['training'].get('reg_target_q_scale', [1.0, 1.0, 1.5]),
            dtype=torch.float32,
            device=self.device,
        )
        self.reg_target_r_scale = torch.tensor(
            self.config['training'].get('reg_target_r_scale', [1.2, 1.0, 0.8]),
            dtype=torch.float32,
            device=self.device,
        )
        self.enable_uncertainty_loss = bool(self.config['training'].get('enable_uncertainty_loss', True))
        self.lambda_uncertainty = float(self.config['training'].get('lambda_uncertainty', 0.2))
        self.q_supervision = str(self.config['training'].get('q_supervision', 'residual')).lower()
        self.uncertainty_wind_weight = float(self.config['training'].get('uncertainty_wind_weight', 1.0))
        self.uncertainty_gps_weight = float(self.config['training'].get('uncertainty_gps_weight', 0.6))
        self.uncertainty_tas_weight = float(self.config['training'].get('uncertainty_tas_weight', 0.4))
        self.uncertainty_att_weight = float(self.config['training'].get('uncertainty_att_weight', 0.2))
        self.config['training']['lambda_wind'] = float(self.lambda_wind)
        self.config['training']['lambda_dir'] = float(self.lambda_dir)
        self.config['training']['lambda_mag'] = float(self.lambda_mag)
        self.config['training']['lambda_mag_relative'] = float(self.lambda_mag_relative)
        self.config['training']['lambda_mag_under'] = float(self.lambda_mag_under)
        self.config['training']['high_wind_threshold'] = float(self.high_wind_threshold)
        self.config['training']['transition_boost'] = {
            'enabled': self.transition_boost_enabled,
            'lambda_transition_mag': float(self.lambda_transition_mag),
        }
        self.config['training']['direction_loss_min_horizontal_wind'] = float(self.direction_loss_min_horizontal_wind)
        self.config['training']['lambda_anti_collapse'] = float(self.lambda_anti_collapse)
        self.config['training']['anti_collapse_threshold'] = float(self.anti_collapse_threshold)
        self.config['training']['snr_aware'] = {
            'enabled': self.snr_aware_enabled,
            'weak_threshold': float(self.snr_weak_threshold),
            'dir_min_weight': float(self.snr_dir_min_weight),
            'mag_boost': float(self.snr_mag_boost),
        }
        self.config['training']['bin_rebalance'] = {
            'enabled': self.bin_rebalance_enabled,
            'n_bins': int(self.bin_rebalance_n_bins),
            'alpha': float(self.bin_rebalance_alpha),
            'max_weight': float(self.bin_rebalance_max),
        }
        self.config['training']['selection_alpha'] = float(self.selection_alpha)
        self.config['training']['selection_beta'] = float(self.selection_beta)
        self.config['training']['selection_gamma_high_under'] = float(self.selection_gamma)
        self.config['training']['selection_delta_transition'] = float(self.selection_delta)
        self.config['training']['selection_metric'] = self.selection_metric_name
        self.config['training']['lambda_reg'] = float(self.lambda_reg)
        self.config['training']['physics_warmup_epochs'] = int(self.physics_warmup_epochs)
        self.config['training']['reg_huber_delta'] = float(self.reg_huber_delta)
        self.config['training']['reg_boundary_penalty'] = float(self.reg_boundary_penalty)
        self.config['training']['reg_target_q_scale'] = self.reg_target_q_scale.detach().cpu().tolist()
        self.config['training']['reg_target_r_scale'] = self.reg_target_r_scale.detach().cpu().tolist()
        self.config['training']['enable_uncertainty_loss'] = bool(self.enable_uncertainty_loss)
        self.config['training']['lambda_uncertainty'] = float(self.lambda_uncertainty)
        self.config['training']['uncertainty_wind_weight'] = float(self.uncertainty_wind_weight)
        self.config['training']['uncertainty_gps_weight'] = float(self.uncertainty_gps_weight)
        self.config['training']['uncertainty_tas_weight'] = float(self.uncertainty_tas_weight)
        self.config['training']['uncertainty_att_weight'] = float(self.uncertainty_att_weight)
        self.aux_head_warmup_epochs = int(self.config['training'].get('aux_head_warmup_epochs', 0))
        self.angles_warmup_epochs = int(self.config['training'].get('angles_warmup_epochs', 50))

        # 三轴风速数据损失的分量权重：wind_down 标准差仅 ~0.2 m/s（vs N/E ~1.3 m/s），
        # 约占总 MSE 的 2-3%，但若等权会让 Down 方向主导归一化空间的梯度。
        # 将 Down 系数降至 0.1，使 N/E 各占 ~47.6%、Down 占 ~4.8%，与物理重要性一致。
        _comp_w = self.config['training'].get('wind_component_weights', [1.0, 1.0, 0.1])
        self._wind_component_weights = torch.tensor(
            _comp_w, dtype=torch.float32, device=self.device
        )
        self.config['training']['wind_component_weights'] = [float(v) for v in _comp_w]

        # ========== 动态段加权（gust_phase / gust_factor sample weight） ==========
        # 配合 1_preprocessing_data.py 输出的 w_*.npy，让阵风过渡段在 data_loss
        # / magnitude_loss 中得到更高权重，引导模型学习阵风跟踪。
        dyn_w_cfg = self.config['training'].get('dynamic_sample_weight', {}) or {}
        self.dynamic_sample_weight_enabled = bool(dyn_w_cfg.get('enabled', True))
        # apply_to: 控制哪些损失参与加权。默认 data + magnitude。
        self.dynamic_sample_weight_apply = set(
            (dyn_w_cfg.get('apply_to') or ['data', 'magnitude'])
        )
        # weight_clamp: 防止极端阵风样本主导梯度
        clamp_range = dyn_w_cfg.get('clamp', [0.5, 5.0])
        self.dynamic_sample_weight_clamp = (
            float(clamp_range[0]) if clamp_range else 0.0,
            float(clamp_range[1]) if clamp_range and len(clamp_range) > 1 else 5.0,
        )
        self.config['training']['dynamic_sample_weight'] = {
            'enabled': self.dynamic_sample_weight_enabled,
            'apply_to': sorted(self.dynamic_sample_weight_apply),
            'clamp': list(self.dynamic_sample_weight_clamp),
        }
        if self.dynamic_sample_weight_enabled:
            print(f"\n【动态段加权】")
            print(f"  启用: True (apply_to={sorted(self.dynamic_sample_weight_apply)}, "
                  f"clamp={self.dynamic_sample_weight_clamp})")
            print(f"  注意: 仅当存在 w_train.npy/w_val.npy 时实际生效，否则等价于均匀权重")
        else:
            print(f"\n【动态段加权】禁用 (training.dynamic_sample_weight.enabled=false)")

        # ========== 训练时 yaw 旋转增强配置 ==========
        # 解决 77 个 run 风向多样性不足的根本问题：通过随机绕 NED-D 轴旋转，
        # 强制模型学习 yaw-invariant 的风场预测能力
        aug_cfg = self.config.get('data', {}).get('augmentation', {}) or {}
        self.aug_enabled = bool(aug_cfg.get('enabled', False))
        rotation_range_deg = float(aug_cfg.get('rotation_range', 0.0))
        self.aug_rotation_range_rad = rotation_range_deg * np.pi / 180.0
        self.aug_noise_std = float(aug_cfg.get('noise_std', 0.0))
        if self.aug_enabled:
            print(f"\n【训练数据增强】")
            print(f"  yaw 旋转: ±{rotation_range_deg:.0f}° (强制 yaw-invariant 学习)")
            if self.aug_noise_std > 0:
                print(f"  高斯噪声: σ={self.aug_noise_std} (在归一化空间)")

        self.ema_decay = float(self.config['training'].get('ema_decay', 0.0))
        self.use_ema = self.ema_decay > 0.0
        self.use_ema_validation = bool(self.config['training'].get('use_ema_validation', False)) and self.use_ema
        self._aux_heads_trainable = None
        self.ema_state = None
        if self.use_ema:
            self.ema_state = {
                key: value.detach().clone()
                for key, value in self.model.state_dict().items()
            }
        self.lambda_tag = f"lambda{self.lambda_physics:g}"
        
        # 物理约束配置
        physics_config = self.config.get('physics', {})
        self.wind_magnitude_max = physics_config.get('wind_magnitude_max', 15.0)
        self.use_attitude_physics = physics_config.get('use_attitude_physics', True)  # 新增
        self.use_6dof_physics = physics_config.get('use_6dof_physics', True)  # 新增：启用完整 6-DOF 物理损失
        print(f"\n【物理损失模式】 {self.physics_mode}")
        print(f"【训练预设】 {self.training_profile}")
        print(f"  启用原因: {self.training_profile_reason}")
        if self.training_profile_changes:
            for change in self.training_profile_changes:
                print(f"  - {change}")
        print(f"【稳定策略】 aux_head_warmup_epochs={self.aux_head_warmup_epochs}, "
              f"EMA={'on' if self.use_ema_validation else 'off'}"
              + (f" (decay={self.ema_decay:.4f})" if self.use_ema else ""))
        print(
            f"【q/r 不确定性损失】 {'on' if self.enable_uncertainty_loss else 'off'} | "
            f"λ={self.lambda_uncertainty:.3f}, wind/gps/tas/att="
            f"{self.uncertainty_wind_weight:.2f}/{self.uncertainty_gps_weight:.2f}/"
            f"{self.uncertainty_tas_weight:.2f}/{self.uncertainty_att_weight:.2f}"
        )
        
        # ===== 6-DOF / Rascal110-JSBSim 物理参数 =====
        # JSBSim set 文件指定 fuel-fraction=0.8，因此默认质量按 13 lb 空重 + 1.5 lb * 0.8 估算
        self.gravity = 9.81  # 重力加速度 [m/s^2]
        self.empty_weight_lbs = float(physics_config.get('empty_weight_lbs', 13.0))
        self.fuel_capacity_lbs = float(physics_config.get('fuel_capacity_lbs', 1.5))
        self.fuel_fraction = float(physics_config.get('fuel_fraction', 0.8))
        rascal_mass_default = (self.empty_weight_lbs + self.fuel_capacity_lbs * self.fuel_fraction) * 0.45359237

        self.uav_mass = float(physics_config.get('uav_mass', rascal_mass_default))
        self.wing_area = float(physics_config.get('wing_area', 10.57 * 0.09290304))
        self.air_density = float(physics_config.get('air_density', 1.225))

        # Rascal110 控制面行程（来自 Rascal110-JSBSim.xml）
        self.elevator_rad_range = tuple(physics_config.get('elevator_rad_range', [-0.35, 0.30]))
        self.elevator_norm_domain = tuple(physics_config.get('elevator_norm_domain', [-0.30, 0.30]))
        self.aileron_rad_range = tuple(physics_config.get('aileron_rad_range', [-0.35, 0.35]))
        self.rudder_rad_range = tuple(physics_config.get('rudder_rad_range', [-0.35, 0.35]))

        # Rascal110 气动查表（来自 Rascal110-JSBSim.xml）
        self.rascal_use_lookup_tables = bool(physics_config.get('rascal_use_lookup_tables', True))
        self.drag_alpha_table_rad = torch.tensor(
            physics_config.get('drag_alpha_table_rad', [-1.57, -0.26, 0.0, 0.26, 1.57]),
            dtype=torch.float32,
            device=self.device,
        )
        self.drag_alpha_table_values = torch.tensor(
            physics_config.get('drag_alpha_table_values', [1.5, 0.056, 0.028, 0.056, 1.5]),
            dtype=torch.float32,
            device=self.device,
        )
        self.drag_beta_table_rad = torch.tensor(
            physics_config.get('drag_beta_table_rad', [-1.57, -0.26, 0.0, 0.26, 1.57]),
            dtype=torch.float32,
            device=self.device,
        )
        self.drag_beta_table_values = torch.tensor(
            physics_config.get('drag_beta_table_values', [1.23, 0.05, 0.0, 0.05, 1.23]),
            dtype=torch.float32,
            device=self.device,
        )
        self.drag_induced_factor = float(physics_config.get('drag_induced_factor', 0.04))
        self.drag_elevator_norm_coeff = float(physics_config.get('drag_elevator_norm_coeff', 0.03))

        self.lift_alpha_table_rad = torch.tensor(
            physics_config.get('lift_alpha_table_rad', [-0.20, 0.0, 0.23, 0.60]),
            dtype=torch.float32,
            device=self.device,
        )
        self.lift_alpha_table_values = torch.tensor(
            physics_config.get('lift_alpha_table_values', [-0.75, 0.25, 1.40, 0.71]),
            dtype=torch.float32,
            device=self.device,
        )
        self.C_L_delta_e = float(physics_config.get('C_L_delta_e', 0.20))
        self.C_C0 = float(physics_config.get('C_C0', 0.0))
        self.C_C_beta = float(physics_config.get('C_C_beta', -1.0))
        self.C_C_delta_a = float(physics_config.get('C_C_delta_a', 0.0))

        # Rascal110 电机 + 18x8 螺旋桨参数
        self.engine_power_watts = float(physics_config.get('engine_power_watts', 1050.0))
        self.propeller_diameter_m = float(physics_config.get('propeller_diameter_m', 18.0 * 0.0254))
        self.propeller_power_lookup_j_max = float(physics_config.get('propeller_power_lookup_j_max', 1.0))
        self.propeller_ct_advance_ratio = torch.tensor(
            physics_config.get('propeller_ct_advance_ratio', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.4]),
            dtype=torch.float32,
            device=self.device,
        )
        self.propeller_ct_values = torch.tensor(
            physics_config.get('propeller_ct_values', [0.0776, 0.0744, 0.0712, 0.0655, 0.0588, 0.0518, 0.0419, 0.0318, 0.0172, -0.0058, -0.0549]),
            dtype=torch.float32,
            device=self.device,
        )
        self.propeller_cp_advance_ratio = torch.tensor(
            physics_config.get('propeller_cp_advance_ratio', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.4]),
            dtype=torch.float32,
            device=self.device,
        )
        self.propeller_cp_values = torch.tensor(
            physics_config.get('propeller_cp_values', [0.0902, 0.0893, 0.0880, 0.0860, 0.0810, 0.0742, 0.0681, 0.0572, 0.0467, 0.0167, -0.0803]),
            dtype=torch.float32,
            device=self.device,
        )

        # 旧多项式参数仅保留为回退分支，默认不再作为 Rascal 主模型使用
        self.C_D0 = float(physics_config.get('C_D0', 0.0305))
        self.C_D_alpha = float(physics_config.get('C_D_alpha', 0.10))
        self.C_D_alpha_delta_e = float(physics_config.get('C_D_alpha_delta_e', 0.12))
        self.C_D_alpha2 = float(physics_config.get('C_D_alpha2', 1.40))
        self.C_L0 = float(physics_config.get('C_L0', 0.25))
        self.C_L_alpha = float(physics_config.get('C_L_alpha', 5.0))

        print(f"\n【6-DOF / Rascal110 参数配置】")
        print(f"  UAV质量: {self.uav_mass:.3f} kg, 翼面积: {self.wing_area:.3f} m², 空气密度: {self.air_density:.3f} kg/m³")
        print(f"  lookup_tables={'on' if self.rascal_use_lookup_tables else 'off'}, 电机功率={self.engine_power_watts:.0f} W, 螺旋桨直径={self.propeller_diameter_m:.4f} m")
        print(f"  升降舵范围={self.elevator_rad_range}, 副翼范围={self.aileron_rad_range}, 方向舵范围={self.rudder_rad_range}")
        print(f"  CYβ={self.C_C_beta:.4f}, CYδa={self.C_C_delta_a:.4f}, CLδe={self.C_L_delta_e:.4f}")
        
        # 早停与选模
        self.early_stopping_patience = self.config['training']['early_stopping_patience']
        metric_labels = {
            'total': '验证总损失',
            'rmse': '验证RMSE',
            'wind_mag_error': '验证风速大小MAE',
            'wind_mag_rmse': '验证风速大小RMSE',
            'high_wind_mag_rmse': '高风速段模值RMSE',
            'high_wind_under_bias': '高风速段低估惩罚',
            'transition_mag_rmse': '动态段模值RMSE',
            'wind_direction_error': '验证风向误差',
            'horizontal_direction_error': '验证水平风向误差',
            'composite_score': '验证组合分数',
            'mag_tracking_score': '幅值跟踪组合分数',
        }
        self.monitor_metric_name = self.selection_metric_name
        if self.monitor_metric_name not in metric_labels:
            raise ValueError(f"不支持的 selection_metric: {self.monitor_metric_name}")
        self.monitor_metric_label = metric_labels[self.monitor_metric_name]
        self.best_monitor_value = float('inf')
        self.best_val_loss = float('inf')
        self.best_val_rmse = float('inf')
        self.best_val_wind_mag_error = float('inf')
        self.best_val_wind_mag_rmse = float('inf')
        self.best_val_direction_error = float('inf')
        self.best_val_horizontal_direction_error = float('inf')
        self.best_val_composite_score = float('inf')
        self.best_val_mag_tracking_score = float('inf')
        self.best_epochs = {
            'loss': None,
            'rmse': None,
            'wind_mag_error': None,
            'wind_mag_rmse': None,
            'high_wind_mag_rmse': None,
            'high_wind_under_bias': None,
            'transition_mag_rmse': None,
            'wind_direction_error': None,
            'horizontal_direction_error': None,
            'composite_score': None,
            'mag_tracking_score': None,
        }
        self.patience_counter = 0
        
        # 创建带时间戳的输出目录
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # TensorBoard - 使用固定目录，所有训练记录在同一个地方
        tensorboard_dir = self.config.get('logging', {}).get('tensorboard_dir', '../tensorboard_logs/')
        # 获取项目根目录（src/的父目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        # 使用固定的 tensorboard_logs 目录，通过 run_name 区分不同训练
        tensorboard_dir = os.path.join(project_root, 'tensorboard_logs')
        
        # 使用描述性的 run_name，日期在前，超参数在后
        lr = self.config['training']['learning_rate']
        bs = self.config['training']['batch_size']
        run_suffix = f"_{self.run_tag}" if self.run_tag else ""
        run_name = f"{self.timestamp}_{self.lambda_tag}_{self.physics_mode}_bs{bs}{run_suffix}"
        tensorboard_log_dir = os.path.join(tensorboard_dir, run_name)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_log_dir)
        
        # 模型保存目录（带时间戳）
        base_save_path = self.config['training']['model_save_path']
        if not os.path.isabs(base_save_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            base_save_path = os.path.join(project_root, base_save_path.lstrip('../'))
        
        self.model_save_dir = os.path.join(base_save_path, f"train_{self.lambda_tag}_{self.timestamp}")
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        print(f"\n【TensorBoard】")
        print(f"  日志目录: {tensorboard_log_dir}")
        print(f"  启动命令: tensorboard --logdir={tensorboard_dir}")
        print(f"\n【模型保存目录】")
        print(f"  {self.model_save_dir}")
        print(f"\n【选模策略】")
        print(f"  主模型 `best_model.pth` 按 {self.monitor_metric_label} 保存")
        print("  额外保留 `best_rmse_model.pth`、`best_mag_model.pth`、`best_dir_model.pth`、`best_composite_model.pth`、`best_mag_tracking_model.pth`")
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_data_loss': [],
            'train_physics_loss': [],
            'train_wind_loss': [],
            'train_dir_loss': [],
            'train_mag_loss': [],
            'train_mag_relative_loss': [],
            'train_mag_under_loss': [],
            'train_transition_mag_loss': [],
            'train_reg_loss': [],
            'train_uncertainty_loss': [],
            'train_grad_norm': [],
            'val_loss': [],
            'val_data_loss': [],
            'val_physics_loss': [],
            'val_wind_loss': [],
            'val_dir_loss': [],
            'val_mag_loss': [],
            'val_mag_relative_loss': [],
            'val_mag_under_loss': [],
            'val_transition_mag_loss': [],
            'val_reg_loss': [],
            'val_uncertainty_loss': [],
            'val_weighted_data_loss': [],
            'learning_rate': [],
            'val_mae': [],
            'val_rmse': [],
            'val_wind_mag_error': [],
            'val_wind_mag_rmse': [],
            'val_high_wind_mag_rmse': [],
            'val_high_wind_mag_bias': [],
            'val_high_wind_under_bias': [],
            'val_transition_mag_rmse': [],
            'val_wind_direction_error': [],
            'val_horizontal_direction_error': [],
            'val_composite_score': [],
            'val_mag_tracking_score': [],
            'val_q_scale_mean': [],
            'val_r_scale_mean': [],
            'val_angle_mag': []
        }

    @staticmethod
    def set_global_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 训练性能优先：deterministic=True + benchmark=False 会让 cuDNN 无法
        # 选择最快的 GRU kernel，单 epoch 慢 30–50%。这里改为优先速度；如需
        # 严格 bit-by-bit 复现，可手动设置 PIGRU_DETERMINISTIC=1 环境变量。
        deterministic = os.environ.get('PIGRU_DETERMINISTIC', '0') == '1'
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    def set_aux_heads_trainable(self, trainable: bool):
        """冻结/解冻辅助自适应分支，避免 temporal split 早期过拟合。

        修复：仅切换 requires_grad 不会清空 Adam 已经积累的 momentum；
        warmup 结束解冻时第一次 step 会用陈旧 momentum 大幅扰动参数，
        进而引发物理损失/q-r 头爆炸 → NaN。这里同步清空对应参数的
        optimizer state，确保解冻后从干净状态重新开始。
        """
        if self._aux_heads_trainable is trainable:
            return
        aux_prefixes = ('head_alphaQ', 'head_beta', 'head_angles', 'head_confidence')
        affected_params = []
        for name, param in self.model.named_parameters():
            if name.startswith(aux_prefixes):
                param.requires_grad = trainable
                affected_params.append(param)
        # 清空对应的 Adam state（exp_avg / exp_avg_sq），避免陈旧 momentum 扰动
        cleared = 0
        for param in affected_params:
            if param in self.optimizer.state:
                self.optimizer.state.pop(param, None)
                cleared += 1
        self._aux_heads_trainable = trainable
        state_text = '解冻' if trainable else '冻结'
        extra = f"，清空 {cleared} 个参数的 Adam state" if cleared > 0 else ""
        print(f"🧩 辅助分支已{state_text}: q/r/angles/confidence{extra}")

    @torch.no_grad()
    def update_ema(self):
        """更新 EMA 阴影权重。"""
        if not self.use_ema or self.ema_state is None:
            return
        for key, value in self.model.state_dict().items():
            ema_value = self.ema_state[key]
            if torch.is_floating_point(value):
                ema_value.mul_(self.ema_decay).add_(value.detach(), alpha=1.0 - self.ema_decay)
            else:
                ema_value.copy_(value)

    def apply_ema_weights(self):
        """临时用 EMA 权重覆盖模型，用于验证和保存。"""
        if not self.use_ema or self.ema_state is None:
            return None
        backup_state = {
            key: value.detach().clone()
            for key, value in self.model.state_dict().items()
        }
        self.model.load_state_dict(self.ema_state, strict=True)
        return backup_state

    def restore_model_weights(self, backup_state):
        """恢复 apply_ema_weights 前的原始模型权重。"""
        if backup_state is not None:
            self.model.load_state_dict(backup_state, strict=True)
    
    def load_normalization_params(self):
        """加载数据归一化参数"""
        # 处理相对路径
        model_save_path = self.config['training']['model_save_path']
        if not os.path.isabs(model_save_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_save_path = os.path.join(project_root, model_save_path.lstrip('../'))
        
        norm_params_path = os.path.join(
            model_save_path,
            'norm_params.pkl'
        )
        
        if not os.path.exists(norm_params_path):
            raise FileNotFoundError(
                f"归一化参数文件不存在: {norm_params_path}\n"
                f"请先运行 1_data_preprocessing.py"
            )
        
        with open(norm_params_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # 验证特征维度
        expected_input_size = self.config['model']['input_size']
        if 'input_size' in metadata and metadata['input_size'] != expected_input_size:
            print(f"⚠️  警告: 归一化参数中的input_size({metadata['input_size']}) "
                  f"与配置文件({expected_input_size})不一致")
        
        # 保存完整的scaler对象（用于物理损失）
        self.scaler_X = metadata['scaler_X']
        self.scaler_y = metadata['scaler_y']
        
        # 提取归一化参数
        self.y_mean = torch.tensor(self.scaler_y.mean_, dtype=torch.float32).to(self.device)
        self.y_std = torch.tensor(self.scaler_y.scale_, dtype=torch.float32).to(self.device)
        
        # 提取各部分的索引
        self.wind_mean = self.y_mean[LABEL_IDX["wind_n"]:LABEL_IDX["wind_d"] + 1]
        self.wind_std = self.y_std[LABEL_IDX["wind_n"]:LABEL_IDX["wind_d"] + 1]
        self.vel_mean = self.y_mean[LABEL_IDX["vel_n"]:LABEL_IDX["vel_d"] + 1]
        self.vel_std = self.y_std[LABEL_IDX["vel_n"]:LABEL_IDX["vel_d"] + 1]
        self.airspeed_mean = self.y_mean[LABEL_IDX["airspeed"]]
        self.airspeed_std = self.y_std[LABEL_IDX["airspeed"]]
        
        print("\n【归一化参数加载成功】")
        print(f"  风速均值: [{self.wind_mean[0]:.2f}, {self.wind_mean[1]:.2f}, {self.wind_mean[2]:.2f}] m/s")
        print(f"  风速标准差: [{self.wind_std[0]:.2f}, {self.wind_std[1]:.2f}, {self.wind_std[2]:.2f}] m/s")
        print(f"  地速均值: [{self.vel_mean[0]:.2f}, {self.vel_mean[1]:.2f}, {self.vel_mean[2]:.2f}] m/s")
        print(f"  空速均值: {self.airspeed_mean:.2f} m/s, 标准差: {self.airspeed_std:.2f} m/s")
    
    def load_attitude_norm_params(self):
        """
        加载姿态/加速度/舵面/机体速度的归一化参数（用于物理损失中的反归一化）。

        特征布局参考模块顶层 FEATURE_IDX（与 1_preprocessing_data.py 同步）。
        当前数据集 input_size=45：
          0-2   vel_n,e,d  3-5 vel_body  6-8 acc  9-11 attitude  12-14 gyro
          15-18 ctrl_cmd  19 airspeed   20-44 阶段 2 扩展项（target/err/act/imu）
        本函数只取前 20 维子集做物理损失反归一化；FEATURE_IDX 顺序变更时
        以下 indices 会自动跟随，避免硬编码漂移。
        """
        att_indices = [FEATURE_IDX["roll"], FEATURE_IDX["pitch"], FEATURE_IDX["yaw"]]
        self.att_mean = torch.tensor(
            self.scaler_X.mean_[att_indices], 
            dtype=torch.float32
        ).to(self.device)
        self.att_std = torch.tensor(
            self.scaler_X.scale_[att_indices], 
            dtype=torch.float32
        ).to(self.device)
        
        acc_indices = [FEATURE_IDX["ax"], FEATURE_IDX["ay"], FEATURE_IDX["az"]]
        self.acc_mean = torch.tensor(
            self.scaler_X.mean_[acc_indices], 
            dtype=torch.float32
        ).to(self.device)
        self.acc_std = torch.tensor(
            self.scaler_X.scale_[acc_indices], 
            dtype=torch.float32
        ).to(self.device)
        
        ctrl_indices = [
            FEATURE_IDX["aileron_cmd"], FEATURE_IDX["elevator_cmd"],
            FEATURE_IDX["rudder_cmd"], FEATURE_IDX["throttle_cmd"],
        ]
        self.ctrl_mean = torch.tensor(
            self.scaler_X.mean_[ctrl_indices], 
            dtype=torch.float32
        ).to(self.device)
        self.ctrl_std = torch.tensor(
            self.scaler_X.scale_[ctrl_indices],
            dtype=torch.float32
        ).to(self.device)

        body_indices = [FEATURE_IDX["vx_body"], FEATURE_IDX["vy_body"], FEATURE_IDX["vz_body"]]
        self.body_mean = torch.tensor(
            self.scaler_X.mean_[body_indices],
            dtype=torch.float32
        ).to(self.device)
        self.body_std = torch.tensor(
            self.scaler_X.scale_[body_indices],
            dtype=torch.float32
        ).to(self.device)

        # 完整 X/y scaler 张量（供 yaw 旋转增强使用）
        # 旋转增强需要：unnormalize → rotate → normalize 流程，因此需要全字段访问
        self.X_mean_full = torch.tensor(self.scaler_X.mean_, dtype=torch.float32).to(self.device)
        self.X_std_full = torch.tensor(self.scaler_X.scale_, dtype=torch.float32).to(self.device)
        self.y_mean_full = torch.tensor(self.scaler_y.mean_, dtype=torch.float32).to(self.device)
        self.y_std_full = torch.tensor(self.scaler_y.scale_, dtype=torch.float32).to(self.device)

        print(f"\n【姿态归一化参数】")
        print(f"  Roll:  μ={self.att_mean[0]:.3f} rad, σ={self.att_std[0]:.3f}")
        print(f"  Pitch: μ={self.att_mean[1]:.3f} rad, σ={self.att_std[1]:.3f}")
        print(f"  Yaw:   μ={self.att_mean[2]:.3f} rad, σ={self.att_std[2]:.3f}")
        print(f"\n【6-DOF 物理损失所需参数】")
        print(f"  加速度均值: [{self.acc_mean[0]:.2f}, {self.acc_mean[1]:.2f}, {self.acc_mean[2]:.2f}] m/s²")
        print(f"  舵面均值: ail={self.ctrl_mean[0]:.3f}, ele={self.ctrl_mean[1]:.3f}, rud={self.ctrl_mean[2]:.3f}")
        print(f"  机体速度均值: [{self.body_mean[0]:.2f}, {self.body_mean[1]:.2f}, {self.body_mean[2]:.2f}] m/s")

    def _apply_rotation_augmentation(self, X_batch, y_batch):
        """绕 NED-Down 轴随机旋转 yaw_offset，让模型学习 yaw-invariant 的风场预测。

        旋转语义：把"世界绕机体所在点的 D 轴旋转 yaw_offset"。
        等价于把所有 NED 水平矢量旋转 yaw_offset，并把姿态 yaw 加上 yaw_offset。
        Down 分量、机体系全部量、空速、舵面均不受 yaw 旋转影响。

        旋转字段（X 中）：
            [..., 0:2]  vel_n, vel_e   ← NED 水平地速
            [..., 11]   yaw            ← 姿态偏航角（叠加 yaw_offset 并 wrap 到 [-π,π]）
        旋转字段（y 中）：
            [..., 0:2]  wind_north, wind_east
            [..., 3:5]  vel_n, vel_e

        实现方式：unnormalize → rotate → normalize（因为 vel_n/vel_e 的均值/方差不同，
        无法在归一化空间直接旋转保持原有分布语义）。
        """
        if not self.aug_enabled or self.aug_rotation_range_rad <= 0:
            return X_batch, y_batch

        B = X_batch.shape[0]
        device = X_batch.device
        dtype = X_batch.dtype

        # 每个 sample 独立的 yaw_offset，同一序列共用一个 offset（旋转整个时间窗口）
        yaw_offset = torch.empty(B, device=device, dtype=dtype).uniform_(
            -self.aug_rotation_range_rad, self.aug_rotation_range_rad
        )
        cos_o = torch.cos(yaw_offset)
        sin_o = torch.sin(yaw_offset)
        cos_o_T = cos_o.unsqueeze(1)  # [B, 1] 广播到时间维
        sin_o_T = sin_o.unsqueeze(1)

        # ===== 1. 旋转 X 中 NED 水平地速 [..., FEATURE_IDX["vel_n":"vel_e"]] =====
        IDX_VN = FEATURE_IDX["vel_n"]
        IDX_VE = FEATURE_IDX["vel_e"]
        IDX_YAW = FEATURE_IDX["yaw"]
        x_mu = self.X_mean_full
        x_sd = self.X_std_full
        vel_n_phys = X_batch[..., IDX_VN] * x_sd[IDX_VN] + x_mu[IDX_VN]  # [B, T]
        vel_e_phys = X_batch[..., IDX_VE] * x_sd[IDX_VE] + x_mu[IDX_VE]
        vel_n_rot = cos_o_T * vel_n_phys - sin_o_T * vel_e_phys
        vel_e_rot = sin_o_T * vel_n_phys + cos_o_T * vel_e_phys

        # ===== 2. 旋转 X 中 yaw [..., FEATURE_IDX["yaw"]] (wrap 到 [-π, π]) =====
        yaw_phys = X_batch[..., IDX_YAW] * x_sd[IDX_YAW] + x_mu[IDX_YAW]  # [B, T]
        yaw_phys_new = yaw_phys + yaw_offset.unsqueeze(1)
        yaw_phys_new = torch.atan2(torch.sin(yaw_phys_new), torch.cos(yaw_phys_new))

        X_new = X_batch.clone()
        X_new[..., IDX_VN] = (vel_n_rot - x_mu[IDX_VN]) / x_sd[IDX_VN]
        X_new[..., IDX_VE] = (vel_e_rot - x_mu[IDX_VE]) / x_sd[IDX_VE]
        X_new[..., IDX_YAW] = (yaw_phys_new - x_mu[IDX_YAW]) / x_sd[IDX_YAW]

        # ===== 3. 旋转 y 中 wind_n, wind_e [0:2] 和 vel_n, vel_e [3:5] =====
        y_mu = self.y_mean_full
        y_sd = self.y_std_full

        wind_n_phys = y_batch[..., 0] * y_sd[0] + y_mu[0]
        wind_e_phys = y_batch[..., 1] * y_sd[1] + y_mu[1]
        wind_n_rot = cos_o * wind_n_phys - sin_o * wind_e_phys
        wind_e_rot = sin_o * wind_n_phys + cos_o * wind_e_phys

        yvel_n_phys = y_batch[..., 3] * y_sd[3] + y_mu[3]
        yvel_e_phys = y_batch[..., 4] * y_sd[4] + y_mu[4]
        yvel_n_rot = cos_o * yvel_n_phys - sin_o * yvel_e_phys
        yvel_e_rot = sin_o * yvel_n_phys + cos_o * yvel_e_phys

        y_new = y_batch.clone()
        y_new[..., 0] = (wind_n_rot - y_mu[0]) / y_sd[0]
        y_new[..., 1] = (wind_e_rot - y_mu[1]) / y_sd[1]
        y_new[..., 3] = (yvel_n_rot - y_mu[3]) / y_sd[3]
        y_new[..., 4] = (yvel_e_rot - y_mu[4]) / y_sd[4]

        # ===== 4. 可选：归一化空间高斯噪声 =====
        if self.aug_noise_std > 0:
            X_new = X_new + torch.randn_like(X_new) * self.aug_noise_std

        return X_new, y_new

    def _snapshot_weights(self):
        """保存当前权重的干净快照，用于 NaN 恢复。

        修复：旧实现的 dict comprehension 只对顶层 tensor 做 clone，
        而 optimizer.state_dict() 的 'state' 子字典里嵌套着 exp_avg / exp_avg_sq
        等 tensor，未被深拷贝。结果 NaN 恢复后 Adam 动量仍是污染的，下一步会
        立即再次爆 NaN —— 训练陷入"恢复-崩溃"死循环。这里改为 deepcopy。
        """
        self._weight_snapshot = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }
        # deepcopy 处理嵌套结构（state[param_id]['exp_avg'] 等）
        self._optimizer_state_snapshot = copy.deepcopy(self.optimizer.state_dict())

    def _restore_weights(self):
        """从快照恢复权重和优化器状态，并应用 nan_to_num 清扫残留 NaN。"""
        if self._weight_snapshot is None:
            return
        nan_param_count = 0
        # 直接通过 named_parameters 写回 .data，避免依赖 state_dict() 的临时引用
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._weight_snapshot:
                    saved = self._weight_snapshot[name]
                    if torch.isnan(saved).any() or torch.isinf(saved).any():
                        nan_param_count += 1
                    clean = saved.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                    param.data.copy_(clean)
        # 恢复优化器状态（清除被 NaN 污染的 momentum）
        if self._optimizer_state_snapshot is not None:
            self.optimizer.load_state_dict(copy.deepcopy(self._optimizer_state_snapshot))
        self._nan_recovery_count += 1
        lr = self.optimizer.param_groups[0]['lr']
        print(f"\n  [NaN RECOVER] 恢复权重+优化器 (#{self._nan_recovery_count}, {nan_param_count} 个参数含NaN), lr={lr:.6f}")

    def euler_to_R_b2n(self, roll, pitch, yaw):
        """
        欧拉角 -> 旋转矩阵 R_b2n (Body to NED)
        ZYX 顺序: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        
        Args:
            roll, pitch, yaw: [B] tensors (弧度)
        
        Returns:
            R: [B, 3, 3]
        """
        B = roll.shape[0]
        
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        
        # 构建旋转矩阵（批量）
        R = torch.zeros(B, 3, 3, device=roll.device, dtype=roll.dtype)
        
        R[:, 0, 0] = cy * cp
        R[:, 0, 1] = cy * sp * sr - sy * cr
        R[:, 0, 2] = cy * sp * cr + sy * sr
        
        R[:, 1, 0] = sy * cp
        R[:, 1, 1] = sy * sp * sr + cy * cr
        R[:, 1, 2] = sy * sp * cr - cy * sr
        
        R[:, 2, 0] = -sp
        R[:, 2, 1] = cp * sr
        R[:, 2, 2] = cp * cr
        
        return R

    def _interp1d(self, x, xp, fp):
        """批量一维线性插值，超出范围时取边界值。"""
        x_flat = x.reshape(-1)
        idx = torch.bucketize(x_flat, xp)
        idx = torch.clamp(idx, 1, xp.numel() - 1)

        x0 = xp[idx - 1]
        x1 = xp[idx]
        y0 = fp[idx - 1]
        y1 = fp[idx]
        slope = (y1 - y0) / (x1 - x0 + 1e-8)
        y = y0 + slope * (x_flat - x0)
        y = torch.where(x_flat <= xp[0], fp[0], y)
        y = torch.where(x_flat >= xp[-1], fp[-1], y)
        return y.reshape_as(x)

    @staticmethod
    def _cmd_to_surface_rad(cmd, rad_range):
        """将 [-1, 1] 归一化舵量映射到实际舵偏角（rad）。"""
        cmd = torch.clamp(cmd, -1.0, 1.0)
        lo, hi = rad_range
        return lo + 0.5 * (cmd + 1.0) * (hi - lo)

    @staticmethod
    def _surface_rad_to_norm(surface_rad, rad_domain):
        """将实际舵偏角映射回 JSBSim 的 normalized surface。"""
        lo, hi = rad_domain
        return -1.0 + 2.0 * (surface_rad - lo) / (hi - lo + 1e-8)

    def _compute_rascal_aero_coefficients(self, alpha, beta, delta_e_rad, delta_e_norm, delta_a_cmd):
        """根据 Rascal110-JSBSim 查表或回退多项式计算气动系数。"""
        if self.rascal_use_lookup_tables:
            cl_base = self._interp1d(alpha, self.lift_alpha_table_rad, self.lift_alpha_table_values)
            cd_alpha = self._interp1d(alpha, self.drag_alpha_table_rad, self.drag_alpha_table_values)
            cd_beta = self._interp1d(beta, self.drag_beta_table_rad, self.drag_beta_table_values)

            c_l = cl_base + self.C_L_delta_e * delta_e_rad
            c_d = cd_alpha + self.drag_induced_factor * c_l**2 + cd_beta + self.drag_elevator_norm_coeff * delta_e_norm
            c_y = self.C_C0 + self.C_C_beta * beta + self.C_C_delta_a * delta_a_cmd
            return c_l, c_d, c_y

        c_d = (
            self.C_D0
            + self.C_D_alpha * alpha
            + self.C_D_alpha_delta_e * alpha * delta_e_rad
            + self.C_D_alpha2 * alpha**2
        )
        c_l = self.C_L0 + self.C_L_alpha * alpha + self.C_L_delta_e * delta_e_rad
        c_y = self.C_C0 + self.C_C_beta * beta + self.C_C_delta_a * delta_a_cmd
        return c_l, c_d, c_y

    def _compute_rascal_propeller_thrust(self, throttle, axial_airspeed):
        """使用 Rascal 的 1050W 电机 + 18x8 螺旋桨表近似求解推力。"""
        throttle = torch.clamp(throttle, 0.0, 1.0)
        axial_airspeed = torch.clamp(axial_airspeed, min=0.0)

        rho = self.air_density
        diameter = self.propeller_diameter_m
        power_available = throttle * self.engine_power_watts
        active_mask = power_available > 1e-6

        cp0 = float(self.propeller_cp_values[0].item())
        denom0 = rho * cp0 * (diameter ** 5) + 1e-6
        # 修复：torch.pow(x, 1/3) 在 x=0 处梯度为无穷，throttle=0 时会污染 NaN。
        # 用 clamp(min=eps) 把输入稳到一个非零下界。
        pow_eps = 1e-6
        n_rev_per_sec = torch.where(
            active_mask,
            torch.pow(torch.clamp(power_available / denom0, min=pow_eps), 1.0 / 3.0),
            torch.zeros_like(power_available),
        )

        for _ in range(5):
            n_safe = torch.clamp(n_rev_per_sec, min=1e-3)
            advance_ratio = axial_airspeed / (n_safe * diameter + 1e-6)
            advance_ratio = torch.clamp(
                advance_ratio,
                min=float(self.propeller_cp_advance_ratio[0].item()),
                max=self.propeller_power_lookup_j_max,
            )
            c_p = self._interp1d(advance_ratio, self.propeller_cp_advance_ratio, self.propeller_cp_values)
            # c_p 在大 advance_ratio 下可能 → 0 甚至负，clamp 防止除零/取负三次根
            denom = rho * torch.clamp(c_p, min=1e-3) * (diameter ** 5) + 1e-6
            solved_n = torch.pow(torch.clamp(power_available / denom, min=pow_eps), 1.0 / 3.0)
            n_rev_per_sec = torch.where(active_mask, solved_n, torch.zeros_like(solved_n))

        n_safe = torch.clamp(n_rev_per_sec, min=1e-3)
        advance_ratio = axial_airspeed / (n_safe * diameter + 1e-6)
        advance_ratio = torch.clamp(
            advance_ratio,
            min=float(self.propeller_ct_advance_ratio[0].item()),
            max=float(self.propeller_ct_advance_ratio[-2].item()),
        )
        c_t = self._interp1d(advance_ratio, self.propeller_ct_advance_ratio, self.propeller_ct_values)
        thrust = c_t * rho * (n_safe ** 2) * (diameter ** 4)
        thrust = torch.where(active_mask, thrust, torch.zeros_like(thrust))
        thrust = torch.clamp(thrust, min=0.0)
        return thrust, advance_ratio, n_safe
    
    def denormalize_wind(self, wind_estimate, y_batch):
        """将预测风速与真值反归一化到物理空间。"""
        wind_truth_norm = y_batch[:, :3]
        wind_truth = wind_truth_norm * self.wind_std + self.wind_mean
        wind_estimate_real = wind_estimate * self.wind_std + self.wind_mean
        return wind_estimate_real, wind_truth

    def calculate_data_loss(self, wind_estimate, y_batch, sample_weight=None):
        """计算三轴风速监督损失（归一化空间 MSE，支持分量加权）。

        分量权重 self._wind_component_weights（默认 [1.0, 1.0, 0.1]）：
          - wind_down 标准差约为 N/E 的 1/6，等权时 Down 在归一化空间会压制 N/E 梯度；
          - 降权至 0.1 使损失聚焦于水平风（对飞行安全更重要）。
        如果提供 sample_weight (shape [B])，则对逐样本损失额外按动态权重加权。
        """
        wind_truth = y_batch[:, :3]
        comp_w = self._wind_component_weights.to(wind_estimate.device)  # [3]
        comp_w_sum = comp_w.sum().clamp(min=1e-6)
        per_dim = (wind_estimate - wind_truth).pow(2)                    # [B, 3]
        per_sample = (per_dim * comp_w).sum(dim=1) / comp_w_sum         # [B]
        if sample_weight is None or 'data' not in self.dynamic_sample_weight_apply:
            return per_sample.mean()
        w = sample_weight.view(-1).to(per_sample.dtype)
        w_sum = w.sum().clamp(min=1e-6)
        return (per_sample * w).sum() / w_sum

    def calculate_direction_loss(self, wind_estimate, y_batch):
        """
        计算水平风向角度差损失（可微分，范围 [0, 180]°）。
        使用余弦相似度计算角度差，在任意角度都有稳定梯度。

        修复要点：
          1. 移除 pred_mag >= 1e-3 的过滤条件，避免模型初始化阶段方向梯度为零。
          2. 在 autocast(False) 中以 fp32 计算 acos，并将 cos_sim 严格 clamp 到
             (-1+eps, 1-eps)，防止 fp16 边界处 d/dx acos = -1/√(1-x²) 发散
             导致 NaN/Inf 梯度。
        """
        wind_estimate_real, wind_truth = self.denormalize_wind(wind_estimate, y_batch)
        pred_horizontal = wind_estimate_real[:, :2]
        truth_horizontal = wind_truth[:, :2]

        pred_mag = torch.norm(pred_horizontal, dim=1)
        truth_mag = torch.norm(truth_horizontal, dim=1)

        min_wind = max(self.direction_loss_min_horizontal_wind, 0.2)
        valid_mask = truth_mag >= min_wind

        if not torch.any(valid_mask):
            return torch.zeros((), device=wind_estimate.device, dtype=wind_estimate.dtype)

        pred_h = pred_horizontal[valid_mask]
        truth_h = truth_horizontal[valid_mask]
        pred_m = pred_mag[valid_mask]
        truth_m = truth_mag[valid_mask]

        # 强制 fp32 + 严格 clamp，避免 acos 边界梯度爆炸（NaN 安全网频繁触发的元凶）
        with torch.amp.autocast('cuda', enabled=False):
            pred_h_f = pred_h.float()
            truth_h_f = truth_h.float()
            pred_m_f = pred_m.float().clamp(min=1e-6)
            truth_m_f = truth_m.float().clamp(min=1e-6)

            pred_unit = pred_h_f / pred_m_f.unsqueeze(-1)
            truth_unit = truth_h_f / truth_m_f.unsqueeze(-1)

            cos_sim = torch.sum(pred_unit * truth_unit, dim=1)
            cos_sim = cos_sim.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
            diff_deg = torch.acos(cos_sim) * (180.0 / torch.pi)
            if self.snr_aware_enabled:
                # SNR-aware 权重：truth_m 从 min_wind ramp 到 weak_threshold，
                # 弱风样本得到 snr_dir_min_weight，强风样本得到 1.0
                ramp_span = max(self.snr_weak_threshold - min_wind, 1e-3)
                ramp = torch.clamp((truth_m_f - min_wind) / ramp_span, min=0.0, max=1.0)
                snr_w = self.snr_dir_min_weight + (1.0 - self.snr_dir_min_weight) * ramp
                w_sum = snr_w.sum().clamp(min=1e-6)
                loss = (diff_deg * snr_w).sum() / w_sum
            else:
                loss = torch.mean(diff_deg)

        return loss.to(wind_estimate.dtype)

    def calculate_magnitude_loss(self, wind_estimate, y_batch, sample_weight=None):
        """计算风速模值的 Smooth L1 损失，抑制缩幅解。

        新增 SNR-aware 权重：弱风段 (truth_h_mag < weak_threshold) 上调 magnitude
        loss 权重，强迫模型保留幅值信号、不要崩塌到 0。
        """
        wind_estimate_real, wind_truth = self.denormalize_wind(wind_estimate, y_batch)
        pred_mag = torch.norm(wind_estimate_real, dim=1)
        truth_mag = torch.norm(wind_truth, dim=1)
        per_sample = F.smooth_l1_loss(pred_mag, truth_mag, beta=0.5, reduction='none')  # [B]

        weights = None
        if self.snr_aware_enabled:
            truth_h_mag = torch.norm(wind_truth[:, :2], dim=1)
            ramp = torch.clamp(truth_h_mag / max(self.snr_weak_threshold, 1e-3), min=0.0, max=1.0)
            snr_w = 1.0 + (self.snr_mag_boost - 1.0) * (1.0 - ramp)
            weights = snr_w.to(per_sample.dtype)

        if sample_weight is not None and 'magnitude' in self.dynamic_sample_weight_apply:
            dyn_w = sample_weight.view(-1).to(per_sample.dtype)
            weights = dyn_w if weights is None else (weights * dyn_w)

        if weights is None:
            return per_sample.mean()
        w_sum = weights.sum().clamp(min=1e-6)
        return (per_sample * weights).sum() / w_sum

    def calculate_magnitude_tracking_losses(self, wind_estimate, y_batch, sample_weight=None):
        """额外的幅值跟踪损失：相对误差、高风速低估惩罚、动态段模值误差。

        这些项只在对应 lambda > 0 时进入总损失，默认关闭以保持旧实验兼容。
        """
        wind_estimate_real, wind_truth = self.denormalize_wind(wind_estimate, y_batch)
        pred_mag = torch.norm(wind_estimate_real, dim=1)
        truth_mag = torch.norm(wind_truth, dim=1)
        eps = 0.2

        if self.lambda_mag_relative > 0:
            rel_err = (pred_mag - truth_mag) / (truth_mag + eps)
            rel_loss = F.smooth_l1_loss(rel_err, torch.zeros_like(rel_err), beta=0.2, reduction='mean')
        else:
            rel_loss = torch.zeros((), device=wind_estimate.device, dtype=wind_estimate.dtype)

        if self.lambda_mag_under > 0:
            high_mask = truth_mag >= self.high_wind_threshold
            if torch.any(high_mask):
                under = torch.clamp(truth_mag[high_mask] - pred_mag[high_mask], min=0.0)
                # 用相对低估惩罚避免高风平台被保守均值压低。
                under_loss = torch.mean((under / (truth_mag[high_mask] + eps)) ** 2)
            else:
                under_loss = torch.zeros((), device=wind_estimate.device, dtype=wind_estimate.dtype)
        else:
            under_loss = torch.zeros((), device=wind_estimate.device, dtype=wind_estimate.dtype)

        if self.transition_boost_enabled and self.lambda_transition_mag > 0 and sample_weight is not None:
            dyn_w = sample_weight.view(-1).to(pred_mag.dtype)
            dyn_mask = dyn_w > (1.0 + 1e-6)
            if torch.any(dyn_mask):
                per_sample = F.smooth_l1_loss(pred_mag, truth_mag, beta=0.5, reduction='none')
                weights = torch.clamp(dyn_w - 1.0, min=0.0)
                weights = torch.where(dyn_mask, weights, torch.zeros_like(weights))
                transition_loss = (per_sample * weights).sum() / weights.sum().clamp(min=1e-6)
            else:
                transition_loss = torch.zeros((), device=wind_estimate.device, dtype=wind_estimate.dtype)
        else:
            transition_loss = torch.zeros((), device=wind_estimate.device, dtype=wind_estimate.dtype)

        return {
            'relative': rel_loss,
            'under': under_loss.to(wind_estimate.dtype),
            'transition': transition_loss.to(wind_estimate.dtype),
        }

    def calculate_anti_collapse_loss(self, wind_estimate):
        """
        防崩塌正则项：当模型预测的水平风幅值低于阈值时，施加二次 hinge 惩罚。

        设计动机（来自 evil-sample 分析）：
          模型在弱风段 (|w_h| 真值 0.5-1.5 m/s) 系统性地把预测压到接近 0，
          表现为：
            - pred_h_mag 中位数 ~0.03 m/s （与 truth ~1.2 m/s 完全不符）
            - 幅值误差 ≈ -truth（系统性低估）
            - arctan2(noise, noise) → 方向随机，dir_err 扭曲到 ~180°
          这是模型在低 SNR 段的"投降式"偷懒解，不是数据噪声。

        Loss = E[max(0, threshold - ||pred_h||)^2]
        threshold 设小一点 (~0.3 m/s)，仅对真正崩塌的预测起作用，
        不影响正常输出小风的样本。
        """
        if self.lambda_anti_collapse <= 0:
            return torch.zeros((), device=wind_estimate.device, dtype=wind_estimate.dtype)
        wind_estimate_real = wind_estimate * self.wind_std + self.wind_mean
        pred_h_mag = torch.norm(wind_estimate_real[:, :2], dim=1)
        deficit = torch.clamp(self.anti_collapse_threshold - pred_h_mag, min=0.0)
        return torch.mean(deficit ** 2)

    def calculate_horizontal_direction_loss(self, wind_estimate, X_batch, y_batch):
        """
        水平风向"按真值幅值加权的余弦距离"损失，与 calculate_direction_loss 互补。

        设计动机：
          - calculate_direction_loss 输出度数（acos 形式），所有 valid 样本权重相同；
          - 本函数输出 (1 - cos_sim) 形式，**按真值水平风速大小加权**，让大风样本
            主导梯度（大风样本物理意义更明确，且模型估计的可靠性更高）；
          - 两者形式不同（一个是角度均值，一个是加权余弦距离），数值层面正交，
            避免"同一信号叠 6 倍 + 触发 NaN 安全网丢 batch"的旧问题。
          - 1 - cos_sim 永远可微，无 acos 端点发散，不需要 fp32 强转。

        说明：
          原实现使用 JSBSim 风向真值 + acos，与 calculate_direction_loss 完全冗余。
          注释里也指出物理推导（Vg = Va + wind）在此数据集失效（vel_*_body 实际是
          地速旋转），所以仍以监督真值为准，但改成加权 cos 距离形式。
        """
        # === 1. 反归一化 ===
        wind_est_real, wind_truth = self.denormalize_wind(wind_estimate, y_batch)
        pred_h = wind_est_real[:, :2]
        truth_h = wind_truth[:, :2]

        # === 2. 单位向量（保留所有样本，靠 weight 自然抑制弱风噪声） ===
        pred_mag = torch.norm(pred_h, dim=1).clamp(min=1e-6)
        truth_mag = torch.norm(truth_h, dim=1).clamp(min=1e-6)

        pred_unit = pred_h / pred_mag.unsqueeze(-1)
        truth_unit = truth_h / truth_mag.unsqueeze(-1)

        cos_sim = torch.sum(pred_unit * truth_unit, dim=1)
        # cos_distance ∈ [0, 2]，pred==truth 时为 0，反向时为 2；处处可微
        cos_distance = 1.0 - cos_sim

        # === 3. 按真值幅值加权 ===
        # 弱风样本权重接近 0，强风样本权重 ≈ 1，避免弱风噪声主导梯度
        min_wind = max(self.direction_loss_min_horizontal_wind, 0.2)
        weight = torch.clamp(torch.norm(truth_h, dim=1) / (min_wind * 2.0), min=0.0, max=1.0)
        weight_sum = weight.sum().clamp(min=1.0)

        weighted_loss = (cos_distance * weight).sum() / weight_sum
        return weighted_loss

    def calculate_vertical_direction_loss(self, wind_estimate, X_batch, y_batch, angles):
        """
        垂向风损失：简化版，不依赖模型预测的攻角。

        物理原理（NED 系，D 轴向下为正）：
          - 机体 X 轴沿机头方向，Va_body ≈ [TAS, 0, 0]（小迎角假设）
          - 旋转到 NED：Va_D = -sin(pitch) · TAS
            （pitch 抬头>0 → 飞机爬升 → Va 朝上 → Va_D < 0）
          - 速度三角形：Vg = Va + wind  →  wind_D = Vg_D - Va_D = Vg_D + TAS·sin(pitch)

        修复：旧实现写成 `Va_down = tas * sin(pitch)`，符号反了，
        导致 wind_theory_down 始终偏离真值 ~2*TAS*sin(pitch)，
        给垂向风学习引入系统性错误监督信号。
        """
        vg_norm = y_batch[:, LABEL_IDX["vel_n"]:LABEL_IDX["vel_d"] + 1]
        vg = vg_norm * self.vel_std + self.vel_mean  # [B, 3]

        last_step = X_batch[:, -1, :]
        att_norm = last_step[:, FEATURE_IDX["roll"]:FEATURE_IDX["yaw"] + 1]
        att = att_norm * self.att_std + self.att_mean  # [B, 3] 弧度
        pitch = att[:, 1]

        tas_norm = y_batch[:, LABEL_IDX["airspeed"]]
        tas = tas_norm * self.airspeed_std + self.airspeed_mean

        # 修正：Va_D = -TAS * sin(pitch)（pitch>0 时 Va 朝上，D 轴负向）
        Va_down = -tas * torch.sin(pitch)

        # 理论垂向风
        wind_theory_down = vg[:, 2] - Va_down

        wind_est_real = wind_estimate * self.wind_std + self.wind_mean
        pred_down = wind_est_real[:, 2]

        # 只在垂向风超过阈值时监督
        min_down = 0.1
        valid_mask = torch.abs(wind_theory_down) >= min_down

        if not torch.any(valid_mask):
            return torch.zeros((), device=wind_estimate.device, dtype=wind_estimate.dtype)

        return F.mse_loss(pred_down[valid_mask], wind_theory_down[valid_mask])

    def calculate_heteroscedastic_nll(self, residual, scale):
        """基于预测 scale 的高斯 NLL，鼓励 q/r 按样本调节而不是塌成常数。

        数值保护：
          - scale clamp 到 [0.1, 10.0]（防止 log(scale) 极端偏移）
          - residual/scale 比值 clamp 到 [-10, 10]（防止极端离群点爆炸梯度）
          - 最终 loss clamp 到 [0, 20]（保证非负）

        AMP 修复：原实现在 `residual = residual.float()` 后又通过广播被推回 fp16
        计算 ratio²，会导致大 residual 下精度崩溃。这里整段强制 fp32 计算。
        """
        with torch.amp.autocast('cuda', enabled=False):
            # .float() 保留 autograd 图，可正常回传梯度到 wind_estimate / scale
            residual_f = residual.float()
            scale_f = torch.clamp(scale.float(), min=0.1, max=10.0)
            while scale_f.dim() < residual_f.dim():
                scale_f = scale_f.unsqueeze(-1)
            ratio = torch.clamp(residual_f / scale_f, min=-10.0, max=10.0)
            nll = 0.5 * torch.mean(ratio ** 2 + 2.0 * torch.log(scale_f))
            nll = torch.clamp(nll, min=0.0, max=20.0)
        # 回到原 dtype（保留梯度）；如果原本就是 fp32，cast 是 no-op
        return nll.to(scale.dtype)

    def build_uncertainty_residuals(self, wind_estimate, X_batch, y_batch, angles):
        """构造 q/r 异方差损失所需的监督残差和物理残差。"""
        wind_truth_norm = y_batch[:, LABEL_IDX["wind_n"]:LABEL_IDX["wind_d"] + 1]
        wind_residual = wind_estimate - wind_truth_norm

        vg = y_batch[:, LABEL_IDX["vel_n"]:LABEL_IDX["vel_d"] + 1] * self.vel_std + self.vel_mean
        tas = y_batch[:, LABEL_IDX["airspeed"]] * self.airspeed_std + self.airspeed_mean
        wind_real = wind_estimate * self.wind_std + self.wind_mean

        if self.use_attitude_physics or self.use_6dof_physics:
            last_step = X_batch[:, -1, :]
            att_norm = last_step[:, FEATURE_IDX["roll"]:FEATURE_IDX["yaw"] + 1]
            att = att_norm * self.att_std + self.att_mean
            roll, pitch, yaw = att[:, 0], att[:, 1], att[:, 2]

            d_alpha = angles[:, 0]
            d_beta = angles[:, 1]
            s_tas = angles[:, 2]

            ca = torch.cos(d_alpha)
            sa = torch.sin(d_alpha)
            cb = torch.cos(d_beta)
            sb = torch.sin(d_beta)

            # 用 TAS * cos(pitch) 构造机体空速向量
            # 原因同上：特征中的 vel_x_body 是地面航迹速度，不是空速
            va_x = tas * s_tas * torch.cos(pitch)
            v_air_body = torch.stack([
                va_x * ca * cb,
                tas * s_tas * sb,
                -va_x * sa * cb,
            ], dim=1)
            r_b2n = self.euler_to_R_b2n(roll, pitch, yaw)
            v_air_ned = torch.bmm(r_b2n, v_air_body.unsqueeze(-1)).squeeze(-1)
            gps_residual = (vg - (v_air_ned + wind_real)) / (self.vel_std + 1e-6)
            tas_residual = (torch.norm(v_air_body, dim=1) - tas) / (self.airspeed_std + 1e-6)
            att_residual = torch.stack([
                d_alpha / (self.model.angle_limit + 1e-6),
                d_beta / (self.model.angle_limit + 1e-6),
            ], dim=1)
        else:
            vel_air_theory = vg - wind_real
            tas_residual = (torch.norm(vel_air_theory, dim=1) - tas) / (self.airspeed_std + 1e-6)
            gps_residual = (vg - wind_real) / (self.vel_std + 1e-6)
            att_residual = torch.zeros(wind_estimate.shape[0], 2, device=wind_estimate.device, dtype=wind_estimate.dtype)

        return {
            'wind': wind_residual,
            'gps': gps_residual,
            'tas': tas_residual.unsqueeze(1),
            'att': att_residual,
        }

    def calculate_uncertainty_loss(self, wind_estimate, X_batch, y_batch, q_scale, r_scale, angles):
        """让 q/r 头直接优化监督残差与物理残差，避免只被固定目标正则拉成常数。"""
        if not self.enable_uncertainty_loss:
            return torch.zeros((), device=wind_estimate.device, dtype=wind_estimate.dtype)

        residuals = self.build_uncertainty_residuals(wind_estimate, X_batch, y_batch, angles)
        q_scale = torch.clamp(q_scale, min=1e-3)
        r_scale = torch.clamp(r_scale, min=1e-3)

        # q_scale 监督目标：
        #  - volatility 模式：直接回归预先映射好的"风过程波动目标"列（让 q_scale 表征风变化快慢）
        #  - residual   模式：旧行为，按风估计残差做异方差 NLL
        if self.q_supervision == 'volatility' and y_batch.shape[1] >= Q_VOL_TARGET_START + 3:
            q_target = y_batch[:, Q_VOL_TARGET_START:Q_VOL_TARGET_START + 3]
            wind_nll = torch.mean((torch.log(q_scale) - torch.log(torch.clamp(q_target, min=1e-3))) ** 2)
        else:
            wind_nll = self.calculate_heteroscedastic_nll(residuals['wind'], q_scale)
        gps_nll = self.calculate_heteroscedastic_nll(residuals['gps'], r_scale[:, 0:1])
        tas_nll = self.calculate_heteroscedastic_nll(residuals['tas'], r_scale[:, 1:2])
        att_nll = self.calculate_heteroscedastic_nll(residuals['att'], r_scale[:, 2:3])

        return self.lambda_uncertainty * (
            self.uncertainty_wind_weight * wind_nll
            + self.uncertainty_gps_weight * gps_nll
            + self.uncertainty_tas_weight * tas_nll
            + self.uncertainty_att_weight * att_nll
        )

    def calculate_composite_score(
        self,
        rmse,
        wind_mag_rmse,
        wind_direction_error,
        high_wind_under_bias=0.0,
        transition_mag_rmse=0.0,
    ):
        """
        组合 RMSE、模值误差和风向误差，作为辅助选模分数。

        修复：旧实现风向误差最大贡献仅 0.1（误差 90°），而 RMSE 通常 1–3、
        wind_mag_rmse 0.5–2，导致"最佳模型"实际上完全按 RMSE 选，风向被忽略。
        现在改为按 selection_beta 加权（30° 误差 ≈ selection_beta），让风向
        贡献和 RMSE 同量级；clamp 到 [0, 90] 避免极端值压扁分数。
        """
        capped_dir_error = min(max(wind_direction_error, 0.0), 90.0)
        dir_contrib = self.selection_beta * (capped_dir_error / 30.0)
        return float(
            rmse
            + self.selection_alpha * wind_mag_rmse
            + dir_contrib
            + self.selection_gamma * max(high_wind_under_bias, 0.0)
            + self.selection_delta * max(transition_mag_rmse, 0.0)
        )
    
    def calculate_physics_loss_simple(self, wind_estimate, y_batch, epoch=None):
        """
        简单物理损失（基于速度三角形，不含姿态旋转）
        保留用于消融对比实验
        
        物理原理：V_air = V_ground - V_wind (均在NED系)
        """
        # 提取归一化数据
        vel_ground_norm = y_batch[:, LABEL_IDX["vel_n"]:LABEL_IDX["vel_d"] + 1]
        airspeed_measured_norm = y_batch[:, LABEL_IDX["airspeed"]]
        
        # 反归一化到物理空间
        vel_ground = vel_ground_norm * self.vel_std + self.vel_mean
        airspeed_measured = airspeed_measured_norm * self.airspeed_std + self.airspeed_mean
        wind_estimate_real = wind_estimate * self.wind_std + self.wind_mean
        
        # 计算理论空速矢量
        vel_air_theory = vel_ground - wind_estimate_real
        airspeed_theory = torch.norm(vel_air_theory, dim=1)
        
        # 空速一致性损失（相对误差）
        airspeed_error = torch.abs(airspeed_theory - airspeed_measured) / (airspeed_measured + 1e-3)
        airspeed_loss = torch.mean(airspeed_error)
        
        # 风速约束损失（限制风速大小）
        wind_magnitude = torch.norm(wind_estimate_real, dim=1)
        wind_constraint_loss = torch.mean(
            F.relu(wind_magnitude - self.wind_magnitude_max)
        )
        
        physics_loss = airspeed_loss + 0.1 * wind_constraint_loss
        
        return physics_loss
    
    def calculate_physics_loss_v2(self, wind_estimate, X_batch, y_batch, angles, epoch=None):
        """
        改进版物理损失（含姿态旋转和小角修正）

        物理原理：
          Vg = R_b2n @ Va_body + wind  (速度三角形)
          wind = Vg - R_b2n @ Va_body
        其中 Va_body 取特征中的真实机体速度向量，然后通过 Δα/Δβ/s_tas 进行修正。

        修复：之前错误地用 Va_body_x = TAS * cos(pitch) 构造假空速，
        导致物理残差达到 ~4 m/s 而非预期的 ~1.6 m/s，使物理损失主导并干扰训练。
        """
        B = X_batch.shape[0]

        # === 1. 反归一化 ===
        # 地速 (NED)
        vg_norm = y_batch[:, LABEL_IDX["vel_n"]:LABEL_IDX["vel_d"] + 1]
        vg = vg_norm * self.vel_std + self.vel_mean  # [B, 3]

        # 空速（标量）
        tas_norm = y_batch[:, LABEL_IDX["airspeed"]]
        tas = tas_norm * self.airspeed_std + self.airspeed_mean  # [B]

        # 风速
        wind_real = wind_estimate * self.wind_std + self.wind_mean  # [B, 3]

        # === 2. 从特征提取姿态角 ===
        last_step = X_batch[:, -1, :]  # [B, FEATURE_DIM]
        att_norm = last_step[:, FEATURE_IDX["roll"]:FEATURE_IDX["yaw"] + 1]  # [B, 3]
        att = att_norm * self.att_std + self.att_mean  # 弧度
        roll, pitch, yaw = att[:, 0], att[:, 1], att[:, 2]

        # === 3. 用 TAS * cos(pitch) 构造机体空速向量 ===
        # 重要说明：特征中的 vel_x/y/z_body 实际是 Vg 旋转到机体的值（地面航迹速度），
        # 而非空速向量。因此不能直接使用，而要用 TAS 和姿态角构造空速。
        # 对于小迎角固定翼：Va_body_x = TAS * cos(pitch), Va_body_y/z ≈ 0
        Va_body_x = tas * torch.cos(pitch)  # [B]
        # Va_body_y 和 Va_body_z 的影响很小（相对于 Va_body_x ≈ 15 m/s）
        # 简化为 0，由 Δβ 和 Δα 分别处理侧滑和迎角效应
        Va_body_y = torch.zeros_like(Va_body_x)
        Va_body_z = torch.zeros_like(Va_body_x)

        # === 4. 应用迎角/侧滑/尺度修正 ===
        d_alpha = angles[:, 0]  # [B]
        d_beta  = angles[:, 1]
        s_tas   = angles[:, 2]

        # 修正后的空速标量
        tas_corrected = tas * s_tas  # [B]

        # 用 Δα 修正迎角（旋转 Va_body_z 的符号）
        # 用 Δβ 修正侧滑（直接加到 Va_body_y）
        # Va_body_x 保持不变（小迎角假设）
        Va_body_corrected = torch.stack([
            Va_body_x,                                          # x: 沿机头
            Va_body_y + tas_corrected * torch.sin(d_beta),      # y: 侧滑修正
            Va_body_z - tas_corrected * torch.sin(d_alpha)      # z: 迎角修正
        ], dim=1)  # [B, 3]

        # === 5. 旋转到 NED ===
        R_b2n = self.euler_to_R_b2n(roll, pitch, yaw)  # [B, 3, 3]
        Va_ned = torch.bmm(R_b2n, Va_body_corrected.unsqueeze(-1)).squeeze(-1)  # [B, 3]

        # === 5. 三角残差损失 ===
        # 理论关系: v_g = Va_ned + wind
        # 残差: r = v_g - (Va_ned + wind_est)
        r_vec = vg - (Va_ned + wind_real)  # [B, 3]
        r_vec = torch.clamp(r_vec, min=-50.0, max=50.0)
        tri_loss = torch.clamp(torch.mean(torch.sum(r_vec**2, dim=1)), max=200.0)

        # === 6. 空速模长一致性 ===
        tas_theory = torch.norm(Va_body_corrected, dim=1)  # [B] = tas_corrected
        tas_err = torch.abs(tas_theory - tas) / (tas + 1e-3)  # 相对误差
        tas_loss = torch.mean(tas_err)

        # === 7. 小角/尺度正则 ===
        ang_reg = 0.1 * (torch.mean(torch.abs(d_alpha)) + torch.mean(torch.abs(d_beta)))
        scale_reg = 0.1 * torch.mean((s_tas - 1.0).abs())

        # === 8. 风速约束 ===
        wind_magnitude = torch.norm(wind_real, dim=1)
        wind_constraint_loss = torch.mean(
            F.relu(wind_magnitude - self.wind_magnitude_max)
        )

        # === 9. 总物理损失 ===
        physics_loss = (tri_loss +
                       tas_loss +
                       ang_reg +
                       scale_reg +
                       0.1 * wind_constraint_loss)

        # === 调试输出 ===
        if epoch is not None and epoch % 10 == 0:
            if not hasattr(self, '_last_debug_epoch_v2') or self._last_debug_epoch_v2 != epoch:
                print(f"\n【物理损失v2诊断 - Epoch {epoch}】")
                print(f"  三角残差: {tri_loss.item():.4f} | "
                      f"空速MSE: {tas_loss.item():.4f} | "
                      f"角度正则: {ang_reg.item():.4f} | "
                      f"尺度正则: {scale_reg.item():.4f} | "
                      f"风速约束: {wind_constraint_loss.item():.4f}")
                print(f"  TAS理论: {tas_theory.mean():.2f}±{tas_theory.std():.2f} m/s | "
                      f"TAS测量: {tas.mean():.2f}±{tas.std():.2f} m/s")
                print(f"  Δα: {d_alpha.mean()*180/3.14159:.2f}° | "
                      f"Δβ: {d_beta.mean()*180/3.14159:.2f}° | "
                      f"s_tas: {s_tas.mean():.3f}")
                print(f"  风速大小: {wind_magnitude.mean():.2f}±{wind_magnitude.std():.2f} m/s")
                self._last_debug_epoch_v2 = epoch

        return physics_loss
    
    def calculate_physics_loss_6dof(self, wind_estimate, X_batch, y_batch, angles, epoch=None):
        """
        基于 Rascal110-JSBSim 参数的 6-DOF 平动力物理损失。

        当前实现仍然只约束平动力残差（F=ma），但气动系数、控制面范围和推力近似
        均尽量对齐 Rascal110-JSBSim.xml，而不再使用 X8 风格的简化二次多项式。
        """
        # =============================================
        # Step 1: 反归一化并提取物理量
        # =============================================
        last_step = X_batch[:, -1, :].float()

        vg_norm = y_batch[:, LABEL_IDX["vel_n"]:LABEL_IDX["vel_d"] + 1].float()
        vg = vg_norm * self.vel_std + self.vel_mean

        wind_estimate = wind_estimate.float()
        wind_est = wind_estimate * self.wind_std + self.wind_mean

        tas_norm = y_batch[:, LABEL_IDX["airspeed"]].float()
        tas_meas = tas_norm * self.airspeed_std + self.airspeed_mean

        att_norm = last_step[:, FEATURE_IDX["roll"]:FEATURE_IDX["yaw"] + 1]
        att = att_norm * self.att_std + self.att_mean
        roll, pitch, yaw = att[:, 0], att[:, 1], att[:, 2]

        acc_norm = last_step[:, FEATURE_IDX["ax"]:FEATURE_IDX["az"] + 1]
        acc_body = acc_norm * self.acc_std + self.acc_mean

        # 控制量（aileron/elevator/rudder/throttle 各 cmd），下标 15-18 连续 4 个
        ctrl_norm = last_step[:, FEATURE_IDX["aileron_cmd"]:FEATURE_IDX["throttle_cmd"] + 1]
        ctrl = ctrl_norm * self.ctrl_std + self.ctrl_mean
        delta_a_cmd = torch.clamp(ctrl[:, 0], -1.0, 1.0)
        delta_e_cmd = torch.clamp(ctrl[:, 1], -1.0, 1.0)
        delta_r_cmd = torch.clamp(ctrl[:, 2], -1.0, 1.0)
        throttle = torch.clamp(ctrl[:, 3], 0.0, 1.0)

        delta_a_rad = self._cmd_to_surface_rad(delta_a_cmd, self.aileron_rad_range)
        delta_e_rad = self._cmd_to_surface_rad(delta_e_cmd, self.elevator_rad_range)
        delta_r_rad = self._cmd_to_surface_rad(delta_r_cmd, self.rudder_rad_range)
        delta_e_norm = self._surface_rad_to_norm(delta_e_rad, self.elevator_norm_domain)

        # =============================================
        # Step 2: 风速三角形 + 姿态坐标变换
        # =============================================
        # 注意：V_air_ned = Vg - wind_est 使用的是模型预测的风（而非真值），
        # 这是有意设计——如果风估计正确，V_air_body 就是真实气动速度。
        # Δα/Δβ 则补偿传感器偏差和气动模型简化误差。
        V_air_ned = vg - wind_est
        R_b2n = self.euler_to_R_b2n(roll, pitch, yaw)
        R_n2b = R_b2n.transpose(1, 2)
        V_air_body = torch.bmm(R_n2b, V_air_ned.unsqueeze(-1)).squeeze(-1)

        u = V_air_body[:, 0]
        v = V_air_body[:, 1]
        w = V_air_body[:, 2]

        # =============================================
        # Step 3: 气流角 + 网络修正
        # =============================================
        V_T = torch.sqrt(u**2 + v**2 + w**2 + 1e-6)
        alpha = torch.atan2(w, u + 1e-6)
        beta = torch.asin(torch.clamp(v / V_T, -0.99, 0.99))

        angles = angles.float()
        d_alpha = angles[:, 0]
        d_beta = angles[:, 1]
        s_tas = angles[:, 2]

        alpha_corrected = alpha + d_alpha
        beta_corrected = beta + d_beta
        # 用模型估计的真空速 V_T（来自 V_air_body 模长）乘以学习的尺度系数
        V_T_corrected = V_T * s_tas

        # =============================================
        # Step 4: 动压 + Rascal 查表气动
        # =============================================
        q_bar = 0.5 * self.air_density * V_T_corrected**2
        C_L, C_D, C_C = self._compute_rascal_aero_coefficients(
            alpha_corrected,
            beta_corrected,
            delta_e_rad,
            delta_e_norm,
            delta_a_cmd,
        )

        S = self.wing_area
        F_Ax = -C_D * q_bar * S
        F_Ay = C_C * q_bar * S
        F_Az = -C_L * q_bar * S
        F_aero_body = torch.stack([F_Ax, F_Ay, F_Az], dim=1)

        # =============================================
        # Step 5: Rascal 1050W 电机 + 18x8 螺旋桨推力
        # =============================================
        axial_airspeed = torch.clamp(
            V_T_corrected * torch.cos(alpha_corrected) * torch.cos(beta_corrected),
            min=0.0,
        )
        F_thrust, prop_advance_ratio, prop_rev_per_sec = self._compute_rascal_propeller_thrust(
            throttle,
            axial_airspeed,
        )
        F_thrust_body = torch.stack([
            F_thrust,
            torch.zeros_like(F_thrust),
            torch.zeros_like(F_thrust),
        ], dim=1)

        # =============================================
        # Step 6: 重力项 + 力平衡残差
        # =============================================
        g = self.gravity
        m = self.uav_mass
        G_body = torch.stack([
            -m * g * torch.sin(pitch),
            m * g * torch.cos(pitch) * torch.sin(roll),
            m * g * torch.cos(pitch) * torch.cos(roll),
        ], dim=1)

        F_total_model = F_thrust_body + F_aero_body + G_body
        F_measured = m * acc_body
        r_aero = F_measured - F_total_model
        force_residual_loss = torch.mean(torch.sum(r_aero**2, dim=1))

        # =============================================
        # Step 7: 运动学一致性与正则
        # =============================================
        tas_theory = V_T_corrected
        tas_err = torch.abs(tas_theory - tas_meas) / (tas_meas + 1e-3)
        tas_loss = torch.mean(tas_err)

        wind_magnitude = torch.norm(wind_est, dim=1)
        wind_constraint = torch.mean(F.relu(wind_magnitude - self.wind_magnitude_max))

        angle_reg = 0.1 * (torch.mean(torch.abs(d_alpha)) + torch.mean(torch.abs(d_beta)))
        scale_reg = 0.1 * torch.mean((s_tas - 1.0).abs())

        force_scale = (m * self.gravity)**2
        normalized_force_loss = force_residual_loss / (force_scale + 1e-6)
        physics_loss = (
            normalized_force_loss
            + tas_loss
            + angle_reg
            + scale_reg
            + 0.1 * wind_constraint
        )

        if epoch is not None and epoch % 10 == 0:
            if not hasattr(self, '_last_debug_epoch_6dof') or self._last_debug_epoch_6dof != epoch:
                print(f"\n【6-DOF / Rascal 物理损失诊断 - Epoch {epoch}】")
                print(f"  力残差 (F=ma): {normalized_force_loss.item():.4f}")
                print(f"  空速一致性: {tas_loss.item():.4f} | 角度正则: {angle_reg.item():.4f} | 尺度正则: {scale_reg.item():.4f}")
                print(f"  总物理损失: {physics_loss.item():.4f}")
                print(f"  α={alpha.mean()*180/3.14159:.2f}° | β={beta.mean()*180/3.14159:.2f}° | δe={delta_e_rad.mean()*180/3.14159:.2f}° | δr={delta_r_rad.mean()*180/3.14159:.2f}°")
                print(f"  C_L={C_L.mean():.4f}, C_D={C_D.mean():.4f}, C_Y={C_C.mean():.4f}")
                print(f"  F_aero: [{F_Ax.mean():.1f}, {F_Ay.mean():.1f}, {F_Az.mean():.1f}] N | F_thrust={F_thrust.mean():.1f} N")
                print(f"  Prop: J={prop_advance_ratio.mean():.3f}, n={prop_rev_per_sec.mean():.1f} rps, throttle={throttle.mean():.3f}")
                print(f"  F_measured: [{(m*acc_body[:,0]).mean():.1f}, {(m*acc_body[:,1]).mean():.1f}, {(m*acc_body[:,2]).mean():.1f}] N")
                self._last_debug_epoch_6dof = epoch

        return physics_loss

    def calculate_physics_loss_dyn(self, wind_estimate, X_batch, y_batch, angles, epoch=None):
        """
        阶段 3：动力学残差损失 (L_dyn)

        与 calculate_physics_loss_6dof 共用同一套 6DOF 力建模（气动 + 推力 + 重力），
        差异在于：
          1. 控制输入用 **实际舵面**（feat 38-41，已经是 rad / norm，无需 cmd→rad 映射）
          2. 残差对象用 **IMU 实测机体加速度**（feat 42-44）而非派生 acc_body

        残差: r = m * imu_acc - F_total_model    其中 F_total_model = F_aero + F_thrust + G

        默认通过 training.lambda_phys_dyn=0.0 关闭；启用时建议 0.05~0.1。
        """
        # =============================================
        # Step 1: 反归一化基础量
        # =============================================
        last_step = X_batch[:, -1, :].float()

        vg_norm = y_batch[:, LABEL_IDX["vel_n"]:LABEL_IDX["vel_d"] + 1].float()
        vg = vg_norm * self.vel_std + self.vel_mean

        wind_estimate = wind_estimate.float()
        wind_est = wind_estimate * self.wind_std + self.wind_mean

        tas_norm = y_batch[:, LABEL_IDX["airspeed"]].float()
        tas_meas = tas_norm * self.airspeed_std + self.airspeed_mean

        att_norm = last_step[:, FEATURE_IDX["roll"]:FEATURE_IDX["yaw"] + 1]
        att = att_norm * self.att_std + self.att_mean
        roll, pitch, yaw = att[:, 0], att[:, 1], att[:, 2]

        # IMU 实测机体加速度（重力补偿后的加速度，单位 m/s²）。
        # 注意：归一化均值/方差仍按 _DEFAULT 的 acc_std / acc_mean，使用 last_step 直接读出物理量。
        # 由于 IMU 未额外注册 mean/std，这里直接读原始数值（fixture 中已是 m/s²）。
        # 如果用户的 IMU 加速度被归一化（acc_std）则要走相同变换；为安全起见，提供 dyn_imu_scale 配置降级。
        imu_acc_body_phys = last_step[:, FEATURE_IDX["imu_ax"]:FEATURE_IDX["imu_az"] + 1]

        # 实际舵面：38-40 是 rad，41 是 norm；同样未单独归一化，直接用原值
        delta_a_rad = last_step[:, FEATURE_IDX["aileron_act"]]
        delta_e_rad = last_step[:, FEATURE_IDX["elevator_act"]]
        delta_r_rad = last_step[:, FEATURE_IDX["rudder_act"]]
        throttle = torch.clamp(last_step[:, FEATURE_IDX["throttle_act"]], 0.0, 1.0)

        # 把 elevator_rad 转回 norm，给 _compute_rascal_aero_coefficients 用
        delta_e_norm = self._surface_rad_to_norm(delta_e_rad, self.elevator_norm_domain)
        # delta_a_cmd 占位（aileron 系数主要走 delta_a_rad；这里取归一化范围内 sign）
        if hasattr(self, 'aileron_rad_range'):
            lo, hi = self.aileron_rad_range
            delta_a_cmd = torch.clamp(2.0 * (delta_a_rad - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0)
        else:
            delta_a_cmd = torch.clamp(delta_a_rad, -1.0, 1.0)

        # =============================================
        # Step 2: 风速三角形 + 姿态坐标变换（与 6dof 相同）
        # =============================================
        V_air_ned = vg - wind_est
        R_b2n = self.euler_to_R_b2n(roll, pitch, yaw)
        R_n2b = R_b2n.transpose(1, 2)
        V_air_body = torch.bmm(R_n2b, V_air_ned.unsqueeze(-1)).squeeze(-1)
        u, v, w_b = V_air_body[:, 0], V_air_body[:, 1], V_air_body[:, 2]

        V_T = torch.sqrt(u ** 2 + v ** 2 + w_b ** 2 + 1e-6)
        alpha = torch.atan2(w_b, u + 1e-6)
        beta = torch.asin(torch.clamp(v / V_T, -0.99, 0.99))

        angles = angles.float()
        d_alpha = angles[:, 0]
        d_beta = angles[:, 1]
        s_tas = angles[:, 2]
        alpha_corrected = alpha + d_alpha
        beta_corrected = beta + d_beta
        V_T_corrected = V_T * s_tas

        # =============================================
        # Step 3: 气动力 + 推力 + 重力（与 6dof 相同 builder）
        # =============================================
        q_bar = 0.5 * self.air_density * V_T_corrected ** 2
        C_L, C_D, C_C = self._compute_rascal_aero_coefficients(
            alpha_corrected, beta_corrected, delta_e_rad, delta_e_norm, delta_a_cmd,
        )
        S = self.wing_area
        F_aero_body = torch.stack([
            -C_D * q_bar * S,
            C_C * q_bar * S,
            -C_L * q_bar * S,
        ], dim=1)

        axial_airspeed = torch.clamp(
            V_T_corrected * torch.cos(alpha_corrected) * torch.cos(beta_corrected),
            min=0.0,
        )
        F_thrust, _, _ = self._compute_rascal_propeller_thrust(throttle, axial_airspeed)
        F_thrust_body = torch.stack([
            F_thrust, torch.zeros_like(F_thrust), torch.zeros_like(F_thrust),
        ], dim=1)

        m = self.uav_mass
        g = self.gravity
        G_body = torch.stack([
            -m * g * torch.sin(pitch),
            m * g * torch.cos(pitch) * torch.sin(roll),
            m * g * torch.cos(pitch) * torch.cos(roll),
        ], dim=1)
        F_total_model = F_thrust_body + F_aero_body + G_body

        # =============================================
        # Step 4: 残差损失
        # 与 6dof 不同：用 IMU 实测加速度替代 acc_body
        # =============================================
        F_measured = m * imu_acc_body_phys
        r_dyn = F_measured - F_total_model
        force_residual_loss = torch.mean(torch.sum(r_dyn ** 2, dim=1))
        force_scale = (m * g) ** 2
        normalized_residual = force_residual_loss / (force_scale + 1e-6)

        # =============================================
        # Step 5: 风约束（保持与 6dof 一致，避免重复正则）
        # =============================================
        wind_magnitude = torch.norm(wind_est, dim=1)
        wind_constraint = torch.mean(F.relu(wind_magnitude - self.wind_magnitude_max))

        physics_loss = normalized_residual + 0.1 * wind_constraint

        if epoch is not None and epoch % 10 == 0:
            if not hasattr(self, '_last_debug_epoch_dyn') or self._last_debug_epoch_dyn != epoch:
                print(f"\n【L_dyn 残差损失诊断 - Epoch {epoch}】")
                print(f"  力残差(IMU): normalized={normalized_residual.item():.4f}, "
                      f"raw={force_residual_loss.item():.2f} N²")
                print(f"  实际舵面: δa={delta_a_rad.mean()*180/3.14159:.2f}°, "
                      f"δe={delta_e_rad.mean()*180/3.14159:.2f}°, "
                      f"δr={delta_r_rad.mean()*180/3.14159:.2f}°, "
                      f"throttle={throttle.mean():.3f}")
                print(f"  IMU 加速度: [{imu_acc_body_phys[:,0].mean():.2f}, "
                      f"{imu_acc_body_phys[:,1].mean():.2f}, "
                      f"{imu_acc_body_phys[:,2].mean():.2f}] m/s²")
                self._last_debug_epoch_dyn = epoch

        return physics_loss

    def calculate_metrics(self, wind_estimate, y_batch):
        """计算物理空间评估指标，包含方向/模值/组合选模分数。"""
        wind_estimate_real, wind_truth = self.denormalize_wind(wind_estimate, y_batch)

        mae = torch.mean(torch.abs(wind_estimate_real - wind_truth)).item()
        rmse = torch.sqrt(torch.mean((wind_estimate_real - wind_truth) ** 2)).item()

        wind_mag_truth = torch.norm(wind_truth, dim=1)
        wind_mag_estimate = torch.norm(wind_estimate_real, dim=1)
        wind_mag_error = torch.mean(torch.abs(wind_mag_estimate - wind_mag_truth)).item()
        wind_mag_rmse = torch.sqrt(torch.mean((wind_mag_estimate - wind_mag_truth) ** 2)).item()
        high_mask = wind_mag_truth >= self.high_wind_threshold
        if torch.any(high_mask):
            high_diff = wind_mag_estimate[high_mask] - wind_mag_truth[high_mask]
            high_wind_mag_rmse = torch.sqrt(torch.mean(high_diff ** 2)).item()
            high_wind_mag_bias = torch.mean(high_diff).item()
            high_wind_under_bias = torch.mean(torch.clamp(-high_diff, min=0.0)).item()
        else:
            high_wind_mag_rmse = 0.0
            high_wind_mag_bias = 0.0
            high_wind_under_bias = 0.0

        truth_horizontal = wind_truth[:, :2]
        pred_horizontal = wind_estimate_real[:, :2]
        truth_horizontal_mag = torch.norm(truth_horizontal, dim=1)
        pred_horizontal_mag = torch.norm(pred_horizontal, dim=1)
        # 只按真值过滤（风向必须有足够的水平分量才有意义），不按预测值过滤
        valid_mask = truth_horizontal_mag >= self.direction_loss_min_horizontal_wind
        if torch.any(valid_mask):
            pred_h = pred_horizontal[valid_mask].float()
            truth_h = truth_horizontal[valid_mask].float()
            pred_unit = pred_h / pred_h.norm(dim=1, keepdim=True).clamp(min=1e-6)
            truth_unit = truth_h / truth_h.norm(dim=1, keepdim=True).clamp(min=1e-6)
            # 评估指标也使用严格 clamp，避免 fp16/fp32 噪声触发 acos 端点 NaN
            cos_sim = (pred_unit * truth_unit).sum(dim=1).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
            horizontal_direction_error = torch.rad2deg(torch.acos(cos_sim)).mean().item()
        else:
            horizontal_direction_error = 0.0

        composite_score = self.calculate_composite_score(
            rmse=rmse,
            wind_mag_rmse=wind_mag_rmse,
            wind_direction_error=horizontal_direction_error,
            high_wind_under_bias=high_wind_under_bias,
            transition_mag_rmse=0.0,
        )

        # 注意：wind_direction_error 与 horizontal_direction_error 是同义字段，
        # 都指水平风向余弦角度差。保留两个键是为了向后兼容
        # (selection_metric 历史上既可设 'wind_direction_error' 也可设
        # 'horizontal_direction_error'，外部脚本如 3b_train_vanilla_gru 也依赖这两个键)。
        return {
            'mae': mae,
            'rmse': rmse,
            'wind_mag_error': wind_mag_error,
            'wind_mag_rmse': wind_mag_rmse,
            'high_wind_mag_rmse': high_wind_mag_rmse,
            'high_wind_mag_bias': high_wind_mag_bias,
            'high_wind_under_bias': high_wind_under_bias,
            'wind_direction_error': horizontal_direction_error,
            'horizontal_direction_error': horizontal_direction_error,
            'composite_score': composite_score,
        }

    def calculate_scale_regularization(self, q_scale, r_scale):
        """统一计算 q_scale / r_scale 的软约束与边界约束。"""
        target_q_scale = self.reg_target_q_scale.to(device=q_scale.device, dtype=q_scale.dtype)
        target_r_scale = self.reg_target_r_scale.to(device=r_scale.device, dtype=r_scale.dtype)
        delta = self.reg_huber_delta

        error_q_scale = q_scale - target_q_scale
        error_r_scale = r_scale - target_r_scale

        huber_q_scale = torch.where(
            torch.abs(error_q_scale) < delta,
            0.5 * error_q_scale**2,
            delta * (torch.abs(error_q_scale) - 0.5 * delta)
        )
        huber_r_scale = torch.where(
            torch.abs(error_r_scale) < delta,
            0.5 * error_r_scale**2,
            delta * (torch.abs(error_r_scale) - 0.5 * delta)
        )

        lower_bound = self.model.alpha_lo
        upper_bound = self.model.alpha_hi
        boundary_q_scale = F.relu(lower_bound - q_scale) + F.relu(q_scale - upper_bound)
        boundary_r_scale = F.relu(lower_bound - r_scale) + F.relu(r_scale - upper_bound)

        reg_loss = self.lambda_reg * (
            torch.mean(huber_q_scale) + torch.mean(huber_r_scale) +
            self.reg_boundary_penalty * torch.mean(boundary_q_scale + boundary_r_scale)
        )
        return reg_loss

    def train_epoch(self, train_loader, lambda_physics, lambda_wind, epoch):
        """训练一个epoch。"""
        self.model.train()

        epoch_total_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        epoch_wind_loss = 0.0
        epoch_dir_loss = 0.0
        epoch_mag_loss = 0.0
        epoch_mag_rel_loss = 0.0
        epoch_mag_under_loss = 0.0
        epoch_transition_mag_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_uncertainty_loss = 0.0
        epoch_grad_norm = 0.0

        # AMP 梯度监控：记录"连续失败次数"和"本 epoch 总失败次数"，
        # 用于聚合统计与"连续 N 次失败才恢复权重"的判定。
        consecutive_nan_grads = 0
        epoch_nan_grad_count = 0
        epoch_nan_loss_count = 0
        epoch_nan_output_count = 0

        # 每个 epoch 开始时保存权重快照（用于 NaN 恢复）
        self._snapshot_weights()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # 兼容 2/3 元组：新数据 (X, y, w)，老数据 (X, y)
            if len(batch) == 3:
                X_batch, y_batch, w_batch = batch
                w_batch = w_batch.to(self.device)
            else:
                X_batch, y_batch = batch
                w_batch = None
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # 训练时 yaw 旋转增强（只在 train，不在 validate）
            # 强制模型学习 yaw-invariant 风场预测，弥补 77 个 run 风向多样性不足
            X_batch, y_batch = self._apply_rotation_augmentation(X_batch, y_batch)

            # 使用混合精度训练 (AMP)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                out = self.model(X_batch, return_dict=True)
                wind_estimate = out['wind_estimate']
                q_scale = out['q_scale']
                r_scale = out['r_scale']
                angles = out['angles']

                # 方案B：在预热期内强制屏蔽角度修正，迫使物理梯度更新风场
                if epoch < self.angles_warmup_epochs:
                    angles = angles.detach().clone()
                    angles[:, 0:2] = 0.0  # d_alpha = 0, d_beta = 0
                    angles[:, 2] = 1.0    # s_tas = 1.0

                # 如果模型输出有 NaN，跳过此 batch（静默）。
                # 这种情况罕见——只有当权重已被污染才会发生，由下游的"连续失败 N 次才恢复"逻辑处理。
                if not torch.isfinite(wind_estimate).all():
                    epoch_nan_output_count += 1
                    consecutive_nan_grads += 1  # 计入连续失败
                    continue

                data_loss = self.calculate_data_loss(wind_estimate, y_batch, sample_weight=w_batch)
                dir_loss = self.calculate_direction_loss(wind_estimate, y_batch)
                mag_loss = self.calculate_magnitude_loss(wind_estimate, y_batch, sample_weight=w_batch)
                mag_tracking_losses = self.calculate_magnitude_tracking_losses(
                    wind_estimate, y_batch, sample_weight=w_batch
                )
                phys_dir_loss = self.calculate_horizontal_direction_loss(wind_estimate, X_batch, y_batch)
                phys_down_loss = self.calculate_vertical_direction_loss(wind_estimate, X_batch, y_batch, angles)
                anti_collapse_loss = self.calculate_anti_collapse_loss(wind_estimate)
                supervised_loss = (
                    lambda_wind * data_loss
                    + self.lambda_dir * dir_loss
                    + self.lambda_mag * mag_loss
                    + self.lambda_mag_relative * mag_tracking_losses['relative']
                    + self.lambda_mag_under * mag_tracking_losses['under']
                    + self.lambda_transition_mag * mag_tracking_losses['transition']
                    + self.lambda_phys_dir * phys_dir_loss
                    + self.lambda_phys_down * phys_down_loss
                    + self.lambda_anti_collapse * anti_collapse_loss
                )

                if self.use_6dof_physics:
                    physics_loss = self.calculate_physics_loss_6dof(
                        wind_estimate, X_batch, y_batch, angles, epoch
                    )
                elif self.use_attitude_physics:
                    physics_loss = self.calculate_physics_loss_v2(
                        wind_estimate, X_batch, y_batch, angles, epoch
                    )
                else:
                    physics_loss = self.calculate_physics_loss_simple(
                        wind_estimate, y_batch, epoch
                    )

                # 阶段 3：L_dyn 动力学残差损失（默认关闭；启用条件：lambda_phys_dyn>0
                # 且模型输入维度足以支持 imu_az 索引）
                lambda_phys_dyn = float(self.config.get('training', {}).get('lambda_phys_dyn', 0.0))
                use_dyn_residual = bool(self.config.get('physics', {}).get('use_dyn_residual', False))
                if use_dyn_residual and lambda_phys_dyn > 0.0 and X_batch.shape[-1] >= FEATURE_IDX["imu_az"] + 1:
                    physics_loss_dyn = self.calculate_physics_loss_dyn(
                        wind_estimate, X_batch, y_batch, angles, epoch
                    )
                else:
                    physics_loss_dyn = torch.zeros((), device=wind_estimate.device, dtype=wind_estimate.dtype)

                reg_loss = self.calculate_scale_regularization(q_scale, r_scale)
                uncertainty_loss = self.calculate_uncertainty_loss(
                    wind_estimate, X_batch, y_batch, q_scale, r_scale, angles
                )
                total_loss = (
                    supervised_loss
                    + lambda_physics * physics_loss
                    + lambda_phys_dyn * physics_loss_dyn
                    + reg_loss
                    + uncertainty_loss
                )

                # 如果总损失有 NaN，跳过此 batch（静默统计）
                if not torch.isfinite(total_loss):
                    epoch_nan_loss_count += 1
                    consecutive_nan_grads += 1  # 计入连续失败
                    continue

            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()

            self.scaler.unscale_(self.optimizer)

            # 矢量化 NaN/Inf 检查：clip_grad_norm_ 内部 reduce 出标量，
            # 同时给出梯度范数（用于裁剪 + 异常诊断），无需逐参数 sync。
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config['training'].get('gradient_clip_norm', 0.5)
            )
            epoch_grad_norm += grad_norm.item() if torch.isfinite(grad_norm) else 0.0

            # 关键设计：GradScaler 在前 5–20 个 batch 会故意用大 scale 探测梯度上限，
            # 触发 fp16 溢出 → grad_norm=inf 是预期行为，不需要恢复权重。
            # scaler.step() 内部检测到 inf/nan 会自动跳过参数更新（这是 GradScaler 的设计）。
            # 我们只需要做两件事：
            #   1) 累计连续失败次数，超过阈值才认为权重真的坏了
            #   2) 静默统计，避免每 batch 打印破坏 tqdm
            grad_is_finite = torch.isfinite(grad_norm).item()
            if grad_is_finite:
                consecutive_nan_grads = 0
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.use_ema:
                    self.update_ema()
            else:
                consecutive_nan_grads += 1
                epoch_nan_grad_count += 1
                # scaler.step 会自动跳过（不会用坏梯度更新参数），但仍需调用以保持状态
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # 仅在连续失败 ≥ 阈值时恢复权重（说明权重真的坏了，不只是 GradScaler 在探测）
                nan_recover_threshold = int(self.config['training'].get('nan_recover_threshold', 20))
                if consecutive_nan_grads >= nan_recover_threshold:
                    print(f"\n  [NaN RECOVER] 连续 {consecutive_nan_grads} 次梯度异常，"
                          f"恢复权重快照 (累计 {self._nan_recovery_count + 1} 次)")
                    self.optimizer.zero_grad(set_to_none=True)
                    self._restore_weights()
                    consecutive_nan_grads = 0
                continue

            # 累加各项训练损失，用于 epoch 末统计
            total_loss_val = total_loss.item() if torch.isfinite(total_loss) else 0.0
            data_loss_val = data_loss.item() if torch.isfinite(data_loss) else 0.0
            physics_loss_val = physics_loss.item() if torch.isfinite(physics_loss) else 0.0
            dir_loss_val = dir_loss.item() if torch.isfinite(dir_loss) else 0.0
            mag_loss_val = mag_loss.item() if torch.isfinite(mag_loss) else 0.0
            mag_rel_loss_val = (
                mag_tracking_losses['relative'].item()
                if torch.isfinite(mag_tracking_losses['relative']) else 0.0
            )
            mag_under_loss_val = (
                mag_tracking_losses['under'].item()
                if torch.isfinite(mag_tracking_losses['under']) else 0.0
            )
            transition_mag_loss_val = (
                mag_tracking_losses['transition'].item()
                if torch.isfinite(mag_tracking_losses['transition']) else 0.0
            )
            reg_loss_val = reg_loss.item() if torch.isfinite(reg_loss) else 0.0
            uncertainty_loss_val = uncertainty_loss.item() if torch.isfinite(uncertainty_loss) else 0.0

            epoch_total_loss += total_loss_val
            epoch_data_loss += data_loss_val
            epoch_physics_loss += physics_loss_val
            epoch_wind_loss += lambda_wind * data_loss_val
            epoch_dir_loss += dir_loss_val
            epoch_mag_loss += mag_loss_val
            epoch_mag_rel_loss += mag_rel_loss_val
            epoch_mag_under_loss += mag_under_loss_val
            epoch_transition_mag_loss += transition_mag_loss_val
            epoch_reg_loss += reg_loss_val
            epoch_uncertainty_loss += uncertainty_loss_val

            pbar.set_postfix({
                'Loss': f'{total_loss_val:.4f}',
                'Data': f'{data_loss_val:.4f}',
                'CosD': f'{dir_loss_val:.2f}',
                'MagU': f'{mag_under_loss_val:.3f}',
                'PhyD': f'{phys_dir_loss.item():.2f}°' if torch.isfinite(phys_dir_loss) else 'nan',
                'PhyU': f'{phys_down_loss.item():.4f}' if torch.isfinite(phys_down_loss) else 'nan',
                'Unc': f'{uncertainty_loss_val:.4f}',
                'Skip': f'{epoch_nan_grad_count}'
            })

        num_batches = len(train_loader)

        # epoch 末聚合统计 NaN 跳过情况，便于诊断 GradScaler 是否进入稳定区
        total_skipped = epoch_nan_grad_count + epoch_nan_loss_count + epoch_nan_output_count
        if total_skipped > 0:
            skip_pct = 100.0 * total_skipped / max(num_batches, 1)
            note = ""
            if epoch == 0 and skip_pct < 10.0:
                note = "  (epoch 1 GradScaler 探测期，正常现象)"
            elif skip_pct >= 30.0:
                note = "  ⚠️ 跳过比例过高，建议下调 learning_rate 或 lambda_phys_dir"
            print(f"  [AMP 监控] 本 epoch 跳过 {total_skipped}/{num_batches} batch ({skip_pct:.1f}%): "
                  f"梯度异常={epoch_nan_grad_count}, loss NaN={epoch_nan_loss_count}, 输出 NaN={epoch_nan_output_count}{note}")

        return {
            'total': epoch_total_loss / num_batches,
            'data': epoch_data_loss / num_batches,
            'physics': epoch_physics_loss / num_batches,
            'wind': epoch_wind_loss / num_batches,
            'dir': epoch_dir_loss / num_batches,
            'mag': epoch_mag_loss / num_batches,
            'mag_relative': epoch_mag_rel_loss / num_batches,
            'mag_under': epoch_mag_under_loss / num_batches,
            'transition_mag': epoch_transition_mag_loss / num_batches,
            'reg': epoch_reg_loss / num_batches,
            'uncertainty': epoch_uncertainty_loss / num_batches,
            'grad_norm': epoch_grad_norm / num_batches
        }

    def validate(self, val_loader, lambda_physics, lambda_wind, w_actual=None, epoch=None):
        """验证一个epoch。

        w_actual: 可选，真实动态 sample_weight（torch.FloatTensor，shape=[N_val]，
            val_loader 必须 shuffle=False 时方可使用）。用于追踪 weighted_val_data_loss
            辅助指标；不参与 total_loss，不影响选模逻辑。
        """
        self.model.eval()

        epoch_total_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        epoch_wind_loss = 0.0
        epoch_dir_loss = 0.0
        epoch_mag_loss = 0.0
        epoch_mag_rel_loss = 0.0
        epoch_mag_under_loss = 0.0
        epoch_transition_mag_loss = 0.0
        epoch_reg_loss = 0.0
        epoch_uncertainty_loss = 0.0
        # 追踪加权 data_loss（辅助，不影响选模）
        epoch_weighted_data_loss = 0.0
        _w_actual_offset = 0  # 用于按顺序从 w_actual 切片 batch
        
        total_mae = 0.0
        total_rmse = 0.0
        total_wind_mag_error = 0.0
        total_wind_mag_rmse = 0.0
        total_high_wind_mag_rmse = 0.0
        total_high_wind_mag_bias = 0.0
        total_high_wind_under_bias = 0.0
        total_transition_mag_rmse = 0.0
        total_wind_direction_error = 0.0
        total_horizontal_direction_error = 0.0
        total_composite_score = 0.0
        
        all_q_scale = []
        all_r_scale = []
        all_angles = []
        
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                # 兼容 2/3 元组（验证不使用权重，保持 val_loss 跨实验可比）
                if len(batch) == 3:
                    X_batch, y_batch, _ = batch
                else:
                    X_batch, y_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                out = self.model(X_batch, return_dict=True)
                wind_estimate = out['wind_estimate']
                q_scale = out['q_scale']
                r_scale = out['r_scale']
                angles = out['angles']

                if epoch is not None and epoch < getattr(self, 'angles_warmup_epochs', 0):
                    angles = angles.detach().clone()
                    angles[:, 0:2] = 0.0
                    angles[:, 2] = 1.0

                # nan_to_num 比 where(isnan, ...) 更高效：单 kernel + 一次性处理 inf/nan
                wind_estimate = wind_estimate.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                angles = angles.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                # angles[:, 2] = s_tas，正常应该 ≈ 1.0；NaN→0 后修复回 1.0
                if angles.shape[1] >= 3:
                    angles[:, 2] = torch.where(angles[:, 2] == 0.0, torch.ones_like(angles[:, 2]), angles[:, 2])
                q_scale = q_scale.nan_to_num(nan=1.0, posinf=1.0, neginf=1.0)
                r_scale = r_scale.nan_to_num(nan=1.0, posinf=1.0, neginf=1.0)

                data_loss = self.calculate_data_loss(wind_estimate, y_batch)
                dir_loss = self.calculate_direction_loss(wind_estimate, y_batch)
                mag_loss = self.calculate_magnitude_loss(wind_estimate, y_batch)
                mag_tracking_losses = self.calculate_magnitude_tracking_losses(
                    wind_estimate, y_batch, sample_weight=None
                )
                phys_dir_loss = self.calculate_horizontal_direction_loss(wind_estimate, X_batch, y_batch)
                phys_down_loss = self.calculate_vertical_direction_loss(wind_estimate, X_batch, y_batch, angles)
                anti_collapse_loss = self.calculate_anti_collapse_loss(wind_estimate)

                # 追踪 weighted_val_data_loss（辅助指标，val_loader shuffle=False 保证顺序一致）
                if w_actual is not None:
                    b = X_batch.shape[0]
                    w_slice = w_actual[_w_actual_offset: _w_actual_offset + b].to(self.device)
                    _w_actual_offset += b
                    wind_truth_3 = y_batch[:, :3]
                    comp_w = self._wind_component_weights.to(wind_estimate.device)
                    per_dim = (wind_estimate - wind_truth_3).pow(2)
                    per_s = (per_dim * comp_w).sum(dim=1) / comp_w.sum().clamp(min=1e-6)
                    w_s = w_slice.view(-1).to(per_s.dtype)
                    wdl = (per_s * w_s).sum() / w_s.sum().clamp(min=1e-6)
                    epoch_weighted_data_loss += wdl.item() if torch.isfinite(wdl) else 0.0
                    mag_tracking_losses = self.calculate_magnitude_tracking_losses(
                        wind_estimate, y_batch, sample_weight=w_slice
                    )

                # 验证损失使用与训练相同的权重，确保 train/val loss 可直接对比
                supervised_loss = (
                    lambda_wind * data_loss
                    + self.lambda_dir * dir_loss
                    + self.lambda_mag * mag_loss
                    + self.lambda_mag_relative * mag_tracking_losses['relative']
                    + self.lambda_mag_under * mag_tracking_losses['under']
                    + self.lambda_transition_mag * mag_tracking_losses['transition']
                    + self.lambda_phys_dir * phys_dir_loss
                    + self.lambda_phys_down * phys_down_loss
                    + self.lambda_anti_collapse * anti_collapse_loss
                )

                if self.use_6dof_physics:
                    physics_loss = self.calculate_physics_loss_6dof(
                        wind_estimate, X_batch, y_batch, angles
                    )
                elif self.use_attitude_physics:
                    physics_loss = self.calculate_physics_loss_v2(
                        wind_estimate, X_batch, y_batch, angles
                    )
                else:
                    physics_loss = self.calculate_physics_loss_simple(
                        wind_estimate, y_batch
                    )

                reg_loss = self.calculate_scale_regularization(q_scale, r_scale)
                uncertainty_loss = self.calculate_uncertainty_loss(
                    wind_estimate, X_batch, y_batch, q_scale, r_scale, angles
                )
                total_loss = supervised_loss + lambda_physics * physics_loss + reg_loss + uncertainty_loss

                epoch_total_loss += (total_loss.item() if torch.isfinite(total_loss) else 0.0)
                epoch_data_loss += data_loss.item()
                epoch_physics_loss += physics_loss.item()
                epoch_wind_loss += (lambda_wind * data_loss).item() if torch.isfinite(data_loss) else 0.0
                epoch_dir_loss += dir_loss.item() if torch.isfinite(dir_loss) else 0.0
                epoch_mag_loss += mag_loss.item() if torch.isfinite(mag_loss) else 0.0
                epoch_mag_rel_loss += (
                    mag_tracking_losses['relative'].item()
                    if torch.isfinite(mag_tracking_losses['relative']) else 0.0
                )
                epoch_mag_under_loss += (
                    mag_tracking_losses['under'].item()
                    if torch.isfinite(mag_tracking_losses['under']) else 0.0
                )
                epoch_transition_mag_loss += (
                    mag_tracking_losses['transition'].item()
                    if torch.isfinite(mag_tracking_losses['transition']) else 0.0
                )
                epoch_reg_loss += reg_loss.item() if torch.isfinite(reg_loss) else 0.0
                epoch_uncertainty_loss += uncertainty_loss.item() if torch.isfinite(uncertainty_loss) else 0.0

                metrics = self.calculate_metrics(wind_estimate, y_batch)
                total_mae += metrics['mae'] if np.isfinite(metrics['mae']) else 0.0
                total_rmse += metrics['rmse'] if np.isfinite(metrics['rmse']) else 0.0
                total_wind_mag_error += metrics['wind_mag_error'] if np.isfinite(metrics['wind_mag_error']) else 0.0
                total_wind_mag_rmse += metrics['wind_mag_rmse'] if np.isfinite(metrics['wind_mag_rmse']) else 0.0
                total_high_wind_mag_rmse += metrics['high_wind_mag_rmse'] if np.isfinite(metrics['high_wind_mag_rmse']) else 0.0
                total_high_wind_mag_bias += metrics['high_wind_mag_bias'] if np.isfinite(metrics['high_wind_mag_bias']) else 0.0
                total_high_wind_under_bias += metrics['high_wind_under_bias'] if np.isfinite(metrics['high_wind_under_bias']) else 0.0
                total_transition_mag_rmse += (
                    float(np.sqrt(max(mag_tracking_losses['transition'].item(), 0.0)))
                    if torch.isfinite(mag_tracking_losses['transition']) else 0.0
                )
                total_wind_direction_error += metrics['wind_direction_error'] if np.isfinite(metrics['wind_direction_error']) else 0.0
                total_horizontal_direction_error += metrics['horizontal_direction_error'] if np.isfinite(metrics['horizontal_direction_error']) else 0.0
                total_composite_score += metrics['composite_score'] if np.isfinite(metrics['composite_score']) else 0.0

                all_q_scale.append(q_scale.cpu().numpy())
                all_r_scale.append(r_scale.cpu().numpy())
                all_angles.append(angles.cpu().numpy())

                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}' if torch.isfinite(total_loss) else 'nan',
                    'RMSE': f'{metrics["rmse"]:.3f}' if np.isfinite(metrics['rmse']) else 'nan',
                    'Dir': f'{metrics["wind_direction_error"]:.2f}',
                    'Unc': f'{uncertainty_loss.item():.4f}'
                })
        
        num_batches = len(val_loader)
        
        all_q_scale = np.vstack(all_q_scale)
        all_r_scale = np.vstack(all_r_scale)
        all_angles = np.vstack(all_angles)
        
        q_scale_mean = np.mean(all_q_scale, axis=0)
        r_scale_mean = np.mean(all_r_scale, axis=0)
        angle_mag = np.mean(np.abs(all_angles[:, :2]), axis=0)
        avg_rmse = total_rmse / num_batches
        avg_mag_rmse = total_wind_mag_rmse / num_batches
        avg_dir = total_wind_direction_error / num_batches
        avg_high_under = total_high_wind_under_bias / num_batches
        avg_transition_mag = total_transition_mag_rmse / num_batches
        mag_tracking_score = self.calculate_composite_score(
            rmse=avg_rmse,
            wind_mag_rmse=avg_mag_rmse,
            wind_direction_error=avg_dir,
            high_wind_under_bias=avg_high_under,
            transition_mag_rmse=avg_transition_mag,
        )
        
        return {
            'total': epoch_total_loss / num_batches,
            'data': epoch_data_loss / num_batches,
            'physics': epoch_physics_loss / num_batches,
            'wind': epoch_wind_loss / num_batches,
            'dir': epoch_dir_loss / num_batches,
            'mag': epoch_mag_loss / num_batches,
            'mag_relative': epoch_mag_rel_loss / num_batches,
            'mag_under': epoch_mag_under_loss / num_batches,
            'transition_mag': epoch_transition_mag_loss / num_batches,
            'reg': epoch_reg_loss / num_batches,
            'uncertainty': epoch_uncertainty_loss / num_batches,
            'weighted_data': epoch_weighted_data_loss / num_batches,  # 辅助指标
            'mae': total_mae / num_batches,
            'rmse': avg_rmse,
            'wind_mag_error': total_wind_mag_error / num_batches,
            'wind_mag_rmse': avg_mag_rmse,
            'high_wind_mag_rmse': total_high_wind_mag_rmse / num_batches,
            'high_wind_mag_bias': total_high_wind_mag_bias / num_batches,
            'high_wind_under_bias': avg_high_under,
            'transition_mag_rmse': avg_transition_mag,
            'wind_direction_error': avg_dir,
            'horizontal_direction_error': total_horizontal_direction_error / num_batches,
            'composite_score': total_composite_score / num_batches,
            'mag_tracking_score': mag_tracking_score,
            'q_scale_mean': q_scale_mean.tolist(),
            'r_scale_mean': r_scale_mean.tolist(),
            'angle_mag': angle_mag.tolist()
        }

    def train(self, X_train, y_train, X_val, y_val, w_train=None, w_val=None):
        """完整训练流程

        新增参数：
            w_train, w_val: 可选的逐样本权重（shape=[N]，float32）。当提供时与
                training.dynamic_sample_weight.enabled 共同生效；缺失则等价于均匀权重。
        """
        print(f"\n{'='*70}")
        print("开始训练")
        print('='*70)
        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"验证集: {X_val.shape[0]} 样本")
        print(f"输入维度: {X_train.shape[1:]} (seq_len, features)")
        print(f"输出维度: {y_train.shape[1]} (labels)")

        # ===== 构造样本权重张量（缺失或被禁用时退化为全 1） =====
        use_sample_weight = (
            self.dynamic_sample_weight_enabled
            and w_train is not None
        )
        if use_sample_weight:
            clamp_lo, clamp_hi = self.dynamic_sample_weight_clamp
            w_train_arr = np.clip(np.asarray(w_train, dtype=np.float32), clamp_lo, clamp_hi)
            mean_w = float(np.mean(w_train_arr))
            max_w = float(np.max(w_train_arr))
            n_dyn = int(np.sum(w_train_arr > 1.0 + 1e-6))
            print(f"【sample_weight】启用，train 权重: mean={mean_w:.3f} max={max_w:.3f} "
                  f"动态样本占比={100.0*n_dyn/max(len(w_train_arr),1):.1f}% (clamp={self.dynamic_sample_weight_clamp})")
        else:
            w_train_arr = np.ones(X_train.shape[0], dtype=np.float32)
            if not self.dynamic_sample_weight_enabled:
                print(f"【sample_weight】禁用：train_epoch 将使用均匀权重")
            else:
                print(f"【sample_weight】未提供 w_train.npy：退化为均匀权重")

        # ===== Wind-bin re-sampling: 按真值水平风强度做反频率重加权 =====
        # 设计动机（来自 evil-sample 分析）：训练集真值风强度分布严重不均
        # （0.5–1 m/s 的样本远少于 1.5–3 m/s），模型倾向于"忽略弱风段"。
        # 反频率加权强迫每个 wind-bin 对总损失贡献相近，弱风学习信号增强。
        if self.bin_rebalance_enabled:
            wm = self.wind_mean.detach().cpu().numpy()  # [3]
            ws = self.wind_std.detach().cpu().numpy()
            y_train_arr = np.asarray(y_train)
            wind_truth_train = y_train_arr[:, :3] * ws + wm
            truth_h_mag_train = np.linalg.norm(wind_truth_train[:, :2], axis=1)
            edge_max = max(float(np.percentile(truth_h_mag_train, 99.5)) * 1.05, 5.0)
            edges = np.linspace(0.0, edge_max, self.bin_rebalance_n_bins + 1)
            bin_idx = np.clip(
                np.digitize(truth_h_mag_train, edges) - 1,
                0,
                self.bin_rebalance_n_bins - 1,
            )
            counts = np.bincount(bin_idx, minlength=self.bin_rebalance_n_bins).astype(np.float32)
            counts = np.where(counts > 0, counts, 1.0)
            inv_freq = (counts.mean() / counts) ** self.bin_rebalance_alpha
            bin_w_per_sample = inv_freq[bin_idx].astype(np.float32)
            bin_w_per_sample = np.clip(bin_w_per_sample, 1.0 / self.bin_rebalance_max, self.bin_rebalance_max)
            scale_to_unit_mean = 1.0 / max(float(bin_w_per_sample.mean()), 1e-6)
            bin_w_per_sample = bin_w_per_sample * scale_to_unit_mean
            w_train_arr = w_train_arr * bin_w_per_sample
            print(
                f"【bin_rebalance】启用 n_bins={self.bin_rebalance_n_bins} "
                f"alpha={self.bin_rebalance_alpha:.2f} max={self.bin_rebalance_max:.1f}"
            )
            print(f"  {'bin (m/s)':>13s}  {'count':>8s}  {'pct':>6s}  {'inv_freq':>9s}  {'bin_w':>7s}")
            for i in range(self.bin_rebalance_n_bins):
                pct = counts[i] / counts.sum() * 100.0
                bw = inv_freq[i] * scale_to_unit_mean
                print(f"  [{edges[i]:>4.2f},{edges[i+1]:>5.2f})  {int(counts[i]):>8d}  {pct:>5.2f}%  "
                      f"{inv_freq[i]:>9.3f}  {bw:>7.3f}")
        else:
            print(f"【bin_rebalance】禁用 (bin_rebalance.enabled=false)")

        # 验证集：DataLoader 均使用 1.0 权重，保持 val_loss 跨实验可比；
        # 同时单独保存真实权重张量 _w_val_actual，供 validate() 追踪 weighted_data_loss 辅助指标。
        w_val_arr = np.ones(X_val.shape[0], dtype=np.float32)
        if use_sample_weight and w_val is not None:
            _clamp_lo, _clamp_hi = self.dynamic_sample_weight_clamp
            self._w_val_actual = torch.FloatTensor(
                np.clip(np.asarray(w_val, dtype=np.float32), _clamp_lo, _clamp_hi)
            )
        else:
            self._w_val_actual = None

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
            torch.FloatTensor(w_train_arr),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
            torch.FloatTensor(w_val_arr),
        )
        self._sample_weight_active = use_sample_weight
        
        num_workers = int(self.config['training'].get('num_workers', 0))
        pin_memory = True if self.device.type == 'cuda' else False
        persistent_workers = bool(self.config['training'].get('persistent_workers', False))
        prefetch_factor = int(self.config['training'].get('prefetch_factor', 4))

        if num_workers <= 0:
            print("【DataLoader】 使用单进程加载 (num_workers=0)，避免多进程 worker 清理告警")
        else:
            print(f"【DataLoader】 使用多进程加载 (num_workers={num_workers}, persistent_workers={persistent_workers})")

        train_loader_kwargs = {
            'dataset': train_dataset,
            'batch_size': self.config['training']['batch_size'],
            'shuffle': True,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }
        val_loader_kwargs = {
            'dataset': val_dataset,
            'batch_size': self.config['training']['batch_size'],
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }

        if num_workers > 0 and persistent_workers:
            train_loader_kwargs['persistent_workers'] = True
            train_loader_kwargs['prefetch_factor'] = prefetch_factor
            val_loader_kwargs['persistent_workers'] = True
            val_loader_kwargs['prefetch_factor'] = prefetch_factor

        train_loader = DataLoader(**train_loader_kwargs)
        val_loader = DataLoader(**val_loader_kwargs)

        # 训练循环（支持 --resume 续训：跳过已完成的 epoch）
        num_epochs = self.config['training']['num_epochs']
        start_epoch = int(getattr(self, '_resume_start_epoch', 0))
        if start_epoch >= num_epochs:
            print(f"\n⚠️  resume checkpoint 已完成 {start_epoch} epoch ≥ num_epochs={num_epochs}，无需续训。")
            self.writer.close()
            return
        if start_epoch > 0:
            print(f"\n🔁 续训模式：跳过前 {start_epoch} 轮，从 Epoch {start_epoch + 1} 开始")

        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('='*70)

            aux_heads_trainable = epoch >= self.aux_head_warmup_epochs
            self.set_aux_heads_trainable(aux_heads_trainable)
            if not aux_heads_trainable and self.aux_head_warmup_epochs > 0:
                print(f"🧊 辅助分支 warmup 中: {epoch+1}/{self.aux_head_warmup_epochs}")
            
            # 动态物理损失权重（预热结束后保留最小值）
            # 避免预热结束后 lambda 骤降为 0，导致物理约束完全消失
            min_physics_lambda = self.lambda_physics * 0.05
            if epoch < self.physics_warmup_epochs:
                current_lambda_physics = self.lambda_physics * (epoch + 1) / self.physics_warmup_epochs
                print(f"📍 物理损失预热中: {current_lambda_physics:.3f} / {self.lambda_physics:.3f}")
            else:
                current_lambda_physics = max(self.lambda_physics, min_physics_lambda)
            
            # 训练和验证
            train_losses = self.train_epoch(train_loader, current_lambda_physics, self.lambda_wind, epoch)
            ema_backup_state = None
            if self.use_ema_validation:
                ema_backup_state = self.apply_ema_weights()
                print(f"🪄 使用 EMA 权重进行验证 (decay={self.ema_decay:.4f})")
            val_losses = self.validate(
                val_loader, current_lambda_physics, self.lambda_wind,
                w_actual=self._w_val_actual, epoch=epoch
            )
            self.restore_model_weights(ema_backup_state)
            
            # 记录历史
            self.history['train_loss'].append(train_losses['total'])
            self.history['train_data_loss'].append(train_losses['data'])
            self.history['train_physics_loss'].append(train_losses['physics'])
            self.history['train_wind_loss'].append(train_losses['wind'])
            self.history['train_dir_loss'].append(train_losses['dir'])
            self.history['train_mag_loss'].append(train_losses['mag'])
            self.history['train_mag_relative_loss'].append(train_losses['mag_relative'])
            self.history['train_mag_under_loss'].append(train_losses['mag_under'])
            self.history['train_transition_mag_loss'].append(train_losses['transition_mag'])
            self.history['train_reg_loss'].append(train_losses['reg'])
            self.history['train_uncertainty_loss'].append(train_losses['uncertainty'])
            self.history['train_grad_norm'].append(train_losses.get('grad_norm', 0.0))
            
            self.history['val_loss'].append(val_losses['total'])
            self.history['val_data_loss'].append(val_losses['data'])
            self.history['val_physics_loss'].append(val_losses['physics'])
            self.history['val_wind_loss'].append(val_losses['wind'])
            self.history['val_dir_loss'].append(val_losses['dir'])
            self.history['val_mag_loss'].append(val_losses['mag'])
            self.history['val_mag_relative_loss'].append(val_losses['mag_relative'])
            self.history['val_mag_under_loss'].append(val_losses['mag_under'])
            self.history['val_transition_mag_loss'].append(val_losses['transition_mag'])
            self.history['val_reg_loss'].append(val_losses['reg'])
            self.history['val_uncertainty_loss'].append(val_losses['uncertainty'])
            self.history['val_weighted_data_loss'].append(val_losses['weighted_data'])
            
            self.history['val_mae'].append(val_losses['mae'])
            self.history['val_rmse'].append(val_losses['rmse'])
            self.history['val_wind_mag_error'].append(val_losses['wind_mag_error'])
            self.history['val_wind_mag_rmse'].append(val_losses['wind_mag_rmse'])
            self.history['val_high_wind_mag_rmse'].append(val_losses['high_wind_mag_rmse'])
            self.history['val_high_wind_mag_bias'].append(val_losses['high_wind_mag_bias'])
            self.history['val_high_wind_under_bias'].append(val_losses['high_wind_under_bias'])
            self.history['val_transition_mag_rmse'].append(val_losses['transition_mag_rmse'])
            self.history['val_wind_direction_error'].append(val_losses['wind_direction_error'])
            self.history['val_horizontal_direction_error'].append(val_losses['horizontal_direction_error'])
            self.history['val_composite_score'].append(val_losses['composite_score'])
            self.history['val_mag_tracking_score'].append(val_losses['mag_tracking_score'])
            self.history['val_q_scale_mean'].append(val_losses['q_scale_mean'])
            self.history['val_r_scale_mean'].append(val_losses['r_scale_mean'])
            self.history['val_angle_mag'].append(val_losses['angle_mag'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Total_train', train_losses['total'], epoch)
            self.writer.add_scalar('Loss/Total_val', val_losses['total'], epoch)
            self.writer.add_scalar('Loss/Data_train', train_losses['data'], epoch)
            self.writer.add_scalar('Loss/Data_val', val_losses['data'], epoch)
            self.writer.add_scalar('Loss/Data_val_weighted', val_losses['weighted_data'], epoch)
            self.writer.add_scalar('Loss/Direction_train', train_losses['dir'], epoch)
            self.writer.add_scalar('Loss/Direction_val', val_losses['dir'], epoch)
            self.writer.add_scalar('Loss/Magnitude_train', train_losses['mag'], epoch)
            self.writer.add_scalar('Loss/Magnitude_val', val_losses['mag'], epoch)
            self.writer.add_scalar('Loss/MagRelative_train', train_losses['mag_relative'], epoch)
            self.writer.add_scalar('Loss/MagRelative_val', val_losses['mag_relative'], epoch)
            self.writer.add_scalar('Loss/MagUnder_train', train_losses['mag_under'], epoch)
            self.writer.add_scalar('Loss/MagUnder_val', val_losses['mag_under'], epoch)
            self.writer.add_scalar('Loss/TransitionMag_train', train_losses['transition_mag'], epoch)
            self.writer.add_scalar('Loss/TransitionMag_val', val_losses['transition_mag'], epoch)
            self.writer.add_scalar('Loss/Physics_train', train_losses['physics'], epoch)
            self.writer.add_scalar('Loss/Physics_val', val_losses['physics'], epoch)
            self.writer.add_scalar('Loss/Reg_train', train_losses['reg'], epoch)
            self.writer.add_scalar('Loss/Reg_val', val_losses['reg'], epoch)
            self.writer.add_scalar('Loss/Uncertainty_train', train_losses['uncertainty'], epoch)
            self.writer.add_scalar('Loss/Uncertainty_val', val_losses['uncertainty'], epoch)
            
            self.writer.add_scalar('Metrics/MAE', val_losses['mae'], epoch)
            self.writer.add_scalar('Metrics/RMSE', val_losses['rmse'], epoch)
            self.writer.add_scalar('Metrics/WindMagError', val_losses['wind_mag_error'], epoch)
            self.writer.add_scalar('Metrics/WindMagRMSE', val_losses['wind_mag_rmse'], epoch)
            self.writer.add_scalar('Metrics/HighWindMagRMSE', val_losses['high_wind_mag_rmse'], epoch)
            self.writer.add_scalar('Metrics/HighWindUnderBias', val_losses['high_wind_under_bias'], epoch)
            self.writer.add_scalar('Metrics/TransitionMagRMSE', val_losses['transition_mag_rmse'], epoch)
            self.writer.add_scalar('Metrics/WindDirectionError', val_losses['wind_direction_error'], epoch)
            self.writer.add_scalar('Metrics/CompositeScore', val_losses['composite_score'], epoch)
            self.writer.add_scalar('Metrics/MagTrackingScore', val_losses['mag_tracking_score'], epoch)
            self.writer.add_scalar('Training/LearningRate', current_lr, epoch)
            
            q_scale_mean = val_losses['q_scale_mean']
            r_scale_mean = val_losses['r_scale_mean']
            self.writer.add_scalar('AdaptiveParams/QScale_N', q_scale_mean[0], epoch)
            self.writer.add_scalar('AdaptiveParams/QScale_E', q_scale_mean[1], epoch)
            self.writer.add_scalar('AdaptiveParams/QScale_D', q_scale_mean[2], epoch)
            self.writer.add_scalar('AdaptiveParams/RScale_GPS', r_scale_mean[0], epoch)
            self.writer.add_scalar('AdaptiveParams/RScale_TAS', r_scale_mean[1], epoch)
            self.writer.add_scalar('AdaptiveParams/RScale_ATT', r_scale_mean[2], epoch)
            
            print(f"\n【训练损失】 总={train_losses['total']:.4f}, "
                  f"数据={train_losses['data']:.4f}, 方向={train_losses['dir']:.4f}, 模值={train_losses['mag']:.4f}, "
                  f"物理={train_losses['physics']:.4f} (λ={current_lambda_physics:.3f}), 正则={train_losses['reg']:.4f}, "
                  f"不确定性={train_losses['uncertainty']:.4f}")
            print(f"【验证损失】 总={val_losses['total']:.4f}, "
                  f"数据={val_losses['data']:.4f}, 方向={val_losses['dir']:.4f}, 模值={val_losses['mag']:.4f}, "
                  f"物理={val_losses['physics']:.4f}, 正则={val_losses['reg']:.4f}, "
                  f"不确定性={val_losses['uncertainty']:.4f}")
            print(f"【验证指标】 MAE={val_losses['mae']:.3f} m/s, RMSE={val_losses['rmse']:.3f} m/s, "
                  f"风速大小MAE={val_losses['wind_mag_error']:.3f} m/s, 风速大小RMSE={val_losses['wind_mag_rmse']:.3f} m/s, "
                  f"高风速低估={val_losses['high_wind_under_bias']:.3f} m/s, 动态模值={val_losses['transition_mag_rmse']:.3f}, "
                  f"风向误差={val_losses['wind_direction_error']:.2f}°")
            print(f"【组合选模】 {self.monitor_metric_label}={val_losses[self.monitor_metric_name]:.4f}, "
                  f"Composite={val_losses['composite_score']:.4f}, MagTrack={val_losses['mag_tracking_score']:.4f}")
            print(f"【自适应参数】 q_scale=[{q_scale_mean[0]:.2f}, {q_scale_mean[1]:.2f}, {q_scale_mean[2]:.2f}], "
                  f"r_scale=[{r_scale_mean[0]:.2f}, {r_scale_mean[1]:.2f}, {r_scale_mean[2]:.2f}]")
            print(f"【学习率】 {current_lr:.6f}")
            
            self.scheduler.step(val_losses[self.monitor_metric_name])
            
            improved_primary = False
            improved_any_core = False   # 任一核心指标改善时重置 patience（防止主指标平台期提前踢出）
            primary_metric_value = val_losses[self.monitor_metric_name]
            primary_save_name_map = {
                'total': 'best_total_loss_model.pth',
                'rmse': 'best_rmse_model.pth',
                'wind_mag_error': 'best_mag_model.pth',
                'wind_mag_rmse': 'best_mag_rmse_model.pth',
                'wind_direction_error': 'best_dir_model.pth',
                'horizontal_direction_error': 'best_dir_model.pth',
                'composite_score': 'best_composite_model.pth',
                'mag_tracking_score': 'best_mag_tracking_model.pth',
                'high_wind_mag_rmse': 'best_high_wind_model.pth',
                'high_wind_under_bias': 'best_high_wind_under_model.pth',
                'transition_mag_rmse': 'best_transition_mag_model.pth',
            }
            if primary_metric_value < self.best_monitor_value:
                self.best_monitor_value = primary_metric_value
                self.best_epochs[self.monitor_metric_name] = epoch + 1
                self.patience_counter = 0
                improved_primary = True
                self.save_model('best_model.pth', use_ema_weights=self.use_ema_validation)
                primary_save_name = primary_save_name_map[self.monitor_metric_name]
                if primary_save_name != 'best_model.pth':
                    self.save_model(primary_save_name, use_ema_weights=self.use_ema_validation)
                print(f"✅ 保存主最佳模型 best_model.pth ({self.monitor_metric_label}: {primary_metric_value:.4f})")

            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.best_epochs['loss'] = epoch + 1
                self.save_model('best_total_loss_model.pth', use_ema_weights=self.use_ema_validation)
                print(f"💾 更新 best_total_loss_model.pth (验证总损失: {val_losses['total']:.4f})")

            if val_losses['rmse'] < self.best_val_rmse:
                self.best_val_rmse = val_losses['rmse']
                self.best_epochs['rmse'] = epoch + 1
                self.save_model('best_rmse_model.pth', use_ema_weights=self.use_ema_validation)
                improved_any_core = True
                print(f"💾 更新 best_rmse_model.pth (RMSE: {val_losses['rmse']:.4f})")

            if val_losses['wind_mag_error'] < self.best_val_wind_mag_error:
                self.best_val_wind_mag_error = val_losses['wind_mag_error']
                self.best_epochs['wind_mag_error'] = epoch + 1
                self.save_model('best_mag_model.pth', use_ema_weights=self.use_ema_validation)
                print(f"💾 更新 best_mag_model.pth (风速大小MAE: {val_losses['wind_mag_error']:.4f})")

            if val_losses['wind_mag_rmse'] < self.best_val_wind_mag_rmse:
                self.best_val_wind_mag_rmse = val_losses['wind_mag_rmse']
                self.best_epochs['wind_mag_rmse'] = epoch + 1
                self.save_model('best_mag_rmse_model.pth', use_ema_weights=self.use_ema_validation)
                print(f"💾 更新 best_mag_rmse_model.pth (风速大小RMSE: {val_losses['wind_mag_rmse']:.4f})")

            if val_losses['wind_direction_error'] < self.best_val_direction_error:
                self.best_val_direction_error = val_losses['wind_direction_error']
                self.best_val_horizontal_direction_error = val_losses['horizontal_direction_error']
                self.best_epochs['wind_direction_error'] = epoch + 1
                self.best_epochs['horizontal_direction_error'] = epoch + 1
                self.save_model('best_dir_model.pth', use_ema_weights=self.use_ema_validation)
                improved_any_core = True
                print(f"💾 更新 best_dir_model.pth (风向误差: {val_losses['wind_direction_error']:.4f}°)")

            if val_losses['composite_score'] < self.best_val_composite_score:
                self.best_val_composite_score = val_losses['composite_score']
                self.best_epochs['composite_score'] = epoch + 1
                self.save_model('best_composite_model.pth', use_ema_weights=self.use_ema_validation)
                improved_any_core = True
                print(f"💾 更新 best_composite_model.pth (组合分数: {val_losses['composite_score']:.4f})")

            if val_losses['mag_tracking_score'] < self.best_val_mag_tracking_score:
                self.best_val_mag_tracking_score = val_losses['mag_tracking_score']
                self.best_epochs['mag_tracking_score'] = epoch + 1
                self.save_model('best_mag_tracking_model.pth', use_ema_weights=self.use_ema_validation)
                print(f"💾 更新 best_mag_tracking_model.pth (幅值跟踪分数: {val_losses['mag_tracking_score']:.4f})")

            if improved_primary or improved_any_core:
                # 主指标改善，或核心三项（rmse/风向/composite）任一改善，均重置 patience
                if not improved_primary:
                    self.patience_counter = 0
                    print(f"↻ 核心指标（rmse/风向/composite）改善，patience 重置")
            else:
                self.patience_counter += 1
                print(f"❌ {self.monitor_metric_label} 及核心指标均未改善 ({self.patience_counter}/{self.early_stopping_patience})")
            
            checkpoint_interval = self.config['training'].get('checkpoint_save_interval', 10)
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
                print(f"💾 保存检查点: epoch_{epoch+1}")
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n{'='*70}")
                print("⚠️  早停触发，训练结束")
                print('='*70)
                break
        
        self.writer.close()
        self.plot_training_history()
        
        print(f"\n{'='*70}")
        print("✅ 训练完成！")
        print('='*70)
        print(f"主选模指标最佳值 ({self.monitor_metric_label}): {self.best_monitor_value:.4f} @ Epoch {self.best_epochs[self.monitor_metric_name]}")
        print(f"最佳验证总损失: {self.best_val_loss:.4f} @ Epoch {self.best_epochs['loss']}")
        print(f"最佳RMSE: {self.best_val_rmse:.4f} @ Epoch {self.best_epochs['rmse']}")
        print(f"最佳风速大小RMSE: {self.best_val_wind_mag_rmse:.4f} @ Epoch {self.best_epochs['wind_mag_rmse']}")
        print(f"最佳风向误差: {self.best_val_direction_error:.4f}° @ Epoch {self.best_epochs['wind_direction_error']}")
        print(f"最佳幅值跟踪分数: {self.best_val_mag_tracking_score:.4f} @ Epoch {self.best_epochs['mag_tracking_score']}")
        print(f"模型保存路径: {self.model_save_dir}")
    
    def save_model(self, filename, use_ema_weights=False):
        """保存模型"""
        save_path = os.path.join(self.model_save_dir, filename)
        backup_state = self.apply_ema_weights() if use_ema_weights else None
        try:
            payload = {
                'epoch': len(self.history['train_loss']),
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict() if hasattr(self, 'scaler') else None,
                'best_monitor_value': self.best_monitor_value,
                'best_val_loss': self.best_val_loss,
                'best_val_rmse': self.best_val_rmse,
                'best_val_wind_mag_error': self.best_val_wind_mag_error,
                'best_val_wind_mag_rmse': self.best_val_wind_mag_rmse,
                'best_val_direction_error': self.best_val_direction_error,
                'best_val_horizontal_direction_error': self.best_val_horizontal_direction_error,
                'best_val_composite_score': self.best_val_composite_score,
                'best_val_mag_tracking_score': self.best_val_mag_tracking_score,
                'best_epochs': self.best_epochs,
                'patience_counter': self.patience_counter,
                'selection_metric': self.monitor_metric_name,
                'config': self.config,
                'history': self.history,
                'saved_with_ema': bool(use_ema_weights),
            }
            if self.use_ema and self.ema_state is not None and not use_ema_weights:
                payload['ema_state'] = {k: v.detach().cpu() for k, v in self.ema_state.items()}
            torch.save(payload, save_path)
        finally:
            self.restore_model_weights(backup_state)

    def load_resume_state(self, resume_path: str):
        """从 checkpoint 恢复训练状态：模型/优化器/调度器/历史/best_*/EMA。

        - 复用 checkpoint 所在目录作为 model_save_dir，保留之前的 best_*.pth。
        - 不动当前的 TensorBoard writer：续训会写入新的 TB 子目录。
        - 设置 self._resume_start_epoch；train() 主循环据此跳过已完成的 epoch。
        """
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"--resume 指向的 checkpoint 不存在: {resume_path}")

        ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as exc:
                print(f"  ⚠ optimizer state 不兼容，跳过: {exc}")
        if 'scheduler_state_dict' in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception as exc:
                print(f"  ⚠ scheduler state 不兼容，跳过: {exc}")
        if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] is not None:
            try:
                self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            except Exception as exc:
                print(f"  ⚠ AMP scaler state 不兼容，跳过: {exc}")

        self.history = ckpt.get('history', self.history) or self.history
        self.best_monitor_value = ckpt.get('best_monitor_value', self.best_monitor_value)
        self.best_val_loss = ckpt.get('best_val_loss', self.best_val_loss)
        self.best_val_rmse = ckpt.get('best_val_rmse', self.best_val_rmse)
        self.best_val_wind_mag_error = ckpt.get('best_val_wind_mag_error', self.best_val_wind_mag_error)
        self.best_val_wind_mag_rmse = ckpt.get('best_val_wind_mag_rmse', self.best_val_wind_mag_rmse)
        self.best_val_direction_error = ckpt.get('best_val_direction_error', self.best_val_direction_error)
        self.best_val_horizontal_direction_error = ckpt.get(
            'best_val_horizontal_direction_error', self.best_val_horizontal_direction_error
        )
        self.best_val_composite_score = ckpt.get('best_val_composite_score', self.best_val_composite_score)
        self.best_val_mag_tracking_score = ckpt.get('best_val_mag_tracking_score', self.best_val_mag_tracking_score)
        self.best_epochs = ckpt.get('best_epochs', self.best_epochs) or self.best_epochs
        self.patience_counter = int(ckpt.get('patience_counter', 0))

        if self.use_ema and self.ema_state is not None:
            ema_saved = ckpt.get('ema_state')
            if ema_saved is not None:
                for k, v in ema_saved.items():
                    if k in self.ema_state:
                        self.ema_state[k].copy_(v.to(self.ema_state[k].device, dtype=self.ema_state[k].dtype))
            else:
                # checkpoint 里没有 ema_state（旧版或 saved_with_ema=True 时跳过保存），
                # 用当前模型权重重新作为 EMA 初值
                for k, v in self.model.state_dict().items():
                    if k in self.ema_state:
                        self.ema_state[k].copy_(v.detach())

        completed_epoch = int(ckpt.get('epoch', len(self.history.get('train_loss', []))))
        self._resume_start_epoch = completed_epoch

        prev_dir = os.path.dirname(os.path.abspath(resume_path))
        try:
            if os.path.isdir(self.model_save_dir) and not os.listdir(self.model_save_dir):
                os.rmdir(self.model_save_dir)
        except OSError:
            pass
        self.model_save_dir = prev_dir

        print(f"\n🔁 续训：从 epoch {completed_epoch + 1} 开始（已完成 {completed_epoch} 轮）")
        print(f"   checkpoint: {resume_path}")
        print(f"   model_save_dir 切换为: {self.model_save_dir}")
        print(f"   恢复主指标 best_{self.monitor_metric_name}={self.best_monitor_value:.6f}, "
              f"patience_counter={self.patience_counter}/{self.early_stopping_patience}")
        return completed_epoch

    def plot_training_history(self):
        """绘制训练历史（4x2布局）"""
        fig, axes = plt.subplots(4, 2, figsize=(15, 16))
        
        # 1. 总损失
        axes[0, 0].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('Total Loss')
        
        # 2. 数据损失
        axes[0, 1].plot(self.history['train_data_loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history['val_data_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Data Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Data Loss (Wind MSE)')
        
        # 3. 物理损失
        axes[1, 0].plot(self.history['train_physics_loss'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history['val_physics_loss'], label='Validation', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Physics Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Physics Loss')
        
        # 4. q/r 辅助损失
        axes[1, 1].plot(self.history['train_reg_loss'], label='Reg Train', linewidth=2)
        axes[1, 1].plot(self.history['val_reg_loss'], label='Reg Val', linewidth=2)
        axes[1, 1].plot(self.history['train_uncertainty_loss'], label='Unc Train', linewidth=2, linestyle='--')
        axes[1, 1].plot(self.history['val_uncertainty_loss'], label='Unc Val', linewidth=2, linestyle='--')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Auxiliary Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('q/r Auxiliary Losses')
        
        # 5. 学习率
        axes[2, 0].plot(self.history['learning_rate'], color='green', linewidth=2)
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Learning Rate')
        axes[2, 0].set_yscale('log')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_title('Learning Rate Schedule')
        
        # 6. MAE和RMSE
        axes[2, 1].plot(self.history['val_mae'], label='MAE', linewidth=2, color='blue')
        axes[2, 1].plot(self.history['val_rmse'], label='RMSE', linewidth=2, color='red')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Error (m/s)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_title('Validation Metrics')
        
        # 7. q_scale 演化
        q_scale_history = np.array(self.history['val_q_scale_mean'])  # [epochs, 3]
        if len(q_scale_history) > 0:
            axes[3, 0].plot(q_scale_history[:, 0], label='q_scale North', linewidth=2)
            axes[3, 0].plot(q_scale_history[:, 1], label='q_scale East', linewidth=2)
            axes[3, 0].plot(q_scale_history[:, 2], label='q_scale Down', linewidth=2)
            axes[3, 0].set_xlabel('Epoch')
            axes[3, 0].set_ylabel('q_scale Value')
            axes[3, 0].legend()
            axes[3, 0].grid(True, alpha=0.3)
            axes[3, 0].set_title('Process Noise q_scale Evolution')
        
        # 8. r_scale 演化
        r_scale_history = np.array(self.history['val_r_scale_mean'])  # [epochs, 3]
        if len(r_scale_history) > 0:
            axes[3, 1].plot(r_scale_history[:, 0], label='r_scale GPS', linewidth=2)
            axes[3, 1].plot(r_scale_history[:, 1], label='r_scale TAS', linewidth=2)
            axes[3, 1].plot(r_scale_history[:, 2], label='r_scale ATT', linewidth=2)
            axes[3, 1].set_xlabel('Epoch')
            axes[3, 1].set_ylabel('r_scale Value')
            axes[3, 1].legend()
            axes[3, 1].grid(True, alpha=0.3)
            axes[3, 1].set_title('Measurement Noise r_scale Evolution')
        
        plt.tight_layout()
        save_path = os.path.join(self.model_save_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 训练曲线已保存: {save_path}")
        
        try:
            backend = matplotlib.get_backend()
            if backend and backend.lower() != 'agg':
                plt.show()
        except:
            pass


def parse_lambda_list(lambda_list_str: str):
    """解析逗号分隔的 lambda 列表字符串。"""
    if not lambda_list_str:
        return []
    values = []
    for item in lambda_list_str.split(','):
        token = item.strip()
        if not token:
            continue
        values.append(float(token))
    return values




def setup_experiment(cfg, project_root):
    """
    从 config.yaml 的 experiment 段读取实验参数，
    自动创建输出目录并复制 norm_params.pkl。
    """
    import shutil

    exp = cfg.get('experiment', {})
    mode = exp.get('mode', 'sweep')
    processed_dir_rel = exp.get('processed_dir', cfg.get('data', {}).get('processed_dir', ''))
    output_base_rel = exp.get('output_base_dir', 'train_output')
    auto_ts = exp.get('auto_timestamp', True)
    lambda_list_str = str(exp.get('lambda_list', '0.1'))
    run_tag = exp.get('run_tag', '')
    physics_mode = exp.get('physics_mode', None)
    training_profile = exp.get('training_profile', 'auto')
    lambda_override = exp.get('lambda_physics_override', None)

    data_dir = resolve_project_path(project_root, processed_dir_rel)
    if not data_dir or not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"预处理数据目录不存在: {data_dir}\n"
            f"请检查 config.yaml 中 experiment.processed_dir 的设置"
        )

    output_base = resolve_project_path(project_root, output_base_rel)
    dir_name = f"train_lambda{lambda_list_str.replace(',', '_')}"
    if auto_ts:
        dir_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_save_dir = os.path.join(output_base, dir_name)
    os.makedirs(model_save_dir, exist_ok=True)

    norm_src = os.path.join(data_dir, 'norm_params.pkl')
    norm_dst = os.path.join(model_save_dir, 'norm_params.pkl')
    if os.path.exists(norm_src):
        shutil.copy2(norm_src, norm_dst)
        print(f"  ✓ norm_params.pkl 已复制到 {model_save_dir}")
    else:
        print(f"  ⚠ 未找到 {norm_src}，跳过复制")

    cfg.setdefault('data', {})['processed_dir'] = data_dir
    cfg.setdefault('training', {})['model_save_path'] = model_save_dir

    return {
        'mode': mode,
        'lambda_list_str': lambda_list_str,
        'lambda_override': lambda_override,
        'physics_mode': physics_mode,
        'training_profile': training_profile,
        'run_tag': run_tag,
        'data_dir': data_dir,
        'model_save_dir': model_save_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PI-GRU 训练脚本")
    parser.add_argument("--config_path", type=str, default=None,
                        help="配置文件路径，默认 config/config.yaml")
    parser.add_argument("--mode", type=str, default=None, help="覆盖 experiment.mode")
    parser.add_argument("--lambda_list", type=str, default=None, help="覆盖 experiment.lambda_list")
    parser.add_argument("--lambda_physics", type=float, default=None, help="覆盖 experiment.lambda_physics_override")
    parser.add_argument("--physics_mode", type=str, default=None, help="覆盖 experiment.physics_mode")
    parser.add_argument("--training_profile", type=str, default=None, help="覆盖 experiment.training_profile")
    parser.add_argument("--physics_warmup_epochs", type=int, default=None,
                        help="覆盖 training.physics_warmup_epochs")
    parser.add_argument("--angles_warmup_epochs", type=int, default=None,
                        help="覆盖 training.angles_warmup_epochs")
    parser.add_argument("--lambda_anti_collapse", type=float, default=None,
                        help="覆盖 training.lambda_anti_collapse")
    parser.add_argument("--snr_aware_enabled", type=str, default=None,
                        help="覆盖 training.snr_aware.enabled (true/false)")
    parser.add_argument("--run_tag", type=str, default=None, help="覆盖 experiment.run_tag")
    parser.add_argument("--processed_dir_override", type=str, default=None, help="覆盖 experiment.processed_dir")
    parser.add_argument("--model_save_path_override", type=str, default=None, help="覆盖 experiment.output_base_dir")
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 .pth checkpoint 续训（仅 single 模式）；保留原 model_save_dir")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    log_path = setup_file_logging(project_root, args)

    print("=" * 70)
    print(" PI-GRU 风场估计训练模块 v3.0 (EKF融合增强版)")
    print("=" * 70)
    print(f"[file-log] 日志同步写入: {log_path}")

    config_path = args.config_path or os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # CLI 参数可覆盖 config 中的 experiment 段
    exp_section = cfg.setdefault('experiment', {})
    if args.mode is not None:
        exp_section['mode'] = args.mode
    if args.lambda_list is not None:
        exp_section['lambda_list'] = args.lambda_list
    if args.lambda_physics is not None:
        exp_section['lambda_physics_override'] = args.lambda_physics
    if args.physics_mode is not None:
        exp_section['physics_mode'] = args.physics_mode
    if args.training_profile is not None:
        exp_section['training_profile'] = args.training_profile
    if args.physics_warmup_epochs is not None:
        cfg.setdefault('training', {})['physics_warmup_epochs'] = int(args.physics_warmup_epochs)
    if args.angles_warmup_epochs is not None:
        cfg.setdefault('training', {})['angles_warmup_epochs'] = int(args.angles_warmup_epochs)
    if args.lambda_anti_collapse is not None:
        cfg.setdefault('training', {})['lambda_anti_collapse'] = float(args.lambda_anti_collapse)
    if args.snr_aware_enabled is not None:
        cfg.setdefault('training', {}).setdefault('snr_aware', {})['enabled'] = args.snr_aware_enabled.lower() == 'true'
    if args.run_tag is not None:
        exp_section['run_tag'] = args.run_tag
    if args.processed_dir_override is not None:
        exp_section['processed_dir'] = args.processed_dir_override
    if args.model_save_path_override is not None:
        exp_section['output_base_dir'] = args.model_save_path_override

    print("\n【自动配置实验目录】")
    exp = setup_experiment(cfg, project_root)

    data_dir = exp['data_dir']
    selected_physics_mode = exp['physics_mode'] or infer_physics_mode(cfg.get('physics', {}))

    print(f"\n【实验配置】")
    print(f"  模式: {exp['mode']}")
    print(f"  物理损失: {selected_physics_mode}")
    print(f"  训练预设: {exp['training_profile']}")
    print(f"  数据目录: {data_dir}")
    print(f"  输出目录: {exp['model_save_dir']}")
    if exp['run_tag']:
        print(f"  实验标签: {exp['run_tag']}")

    print("\n正在加载数据...")

    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

    # 动态段加权用 sample weight：1_preprocessing_data.py 输出
    w_train_path = os.path.join(data_dir, 'w_train.npy')
    w_val_path = os.path.join(data_dir, 'w_val.npy')
    w_train = np.load(w_train_path) if os.path.exists(w_train_path) else None
    w_val = np.load(w_val_path) if os.path.exists(w_val_path) else None

    y_train, y_val, qvol_info = append_volatility_targets(y_train, y_val, cfg)
    if qvol_info is not None:
        print(f"【q_supervision=volatility】已追加风过程波动目标列 {Q_VOL_TARGET_START}:{Q_VOL_TARGET_START+3}")
        print(f"  目标映射 p10={[f'{v:.4f}' for v in qvol_info['p10']]} "
              f"p90={[f'{v:.4f}' for v in qvol_info['p90']]} -> [{qvol_info['lo']},{qvol_info['hi']}]")
        print(f"  目标 mean={[f'{v:.3f}' for v in qvol_info['target_mean']]} "
              f"std={[f'{v:.3f}' for v in qvol_info['target_std']]}")

    print(f"✓ 数据加载完成")
    print(f"  训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"  验证集: X={X_val.shape}, y={y_val.shape}")
    if w_train is not None:
        n_dyn = int(np.sum(w_train > 1.0 + 1e-6))
        print(f"  sample_weight: w_train={w_train.shape} mean={w_train.mean():.3f} max={w_train.max():.3f} "
              f"动态样本占比={100.0*n_dyn/max(len(w_train),1):.1f}%")
    else:
        print(f"  sample_weight: w_train.npy 不存在 → 训练将使用均匀权重")

    test_id_path = os.path.join(data_dir, 'X_test_id.npy')
    test_ood_path = os.path.join(data_dir, 'X_test_ood.npy')
    if os.path.exists(test_id_path):
        X_test_id = np.load(test_id_path)
        print(f"  测试集(ID): X={X_test_id.shape} (训练后评估用)")
    if os.path.exists(test_ood_path):
        X_test_ood = np.load(test_ood_path)
        print(f"  测试集(OOD): X={X_test_ood.shape} (泛化性评估用)")

    model_save_dir = exp['model_save_dir']

    if exp['mode'] == "single":
        trainer = Trainer(
            config_path=config_path,
            lambda_physics_override=exp['lambda_override'],
            physics_mode_override=exp['physics_mode'],
            run_tag=exp['run_tag'],
            processed_dir_override=data_dir,
            model_save_path_override=model_save_dir,
            training_profile=exp['training_profile'],
            physics_warmup_epochs_override=args.physics_warmup_epochs,
            angles_warmup_epochs_override=args.angles_warmup_epochs,
        )
        if args.resume:
            trainer.load_resume_state(args.resume)
        print(f"\n🎯 single 模式：lambda_physics = {trainer.lambda_physics}")
        trainer.train(X_train, y_train, X_val, y_val, w_train=w_train, w_val=w_val)
    else:
        lambda_values = parse_lambda_list(exp['lambda_list_str'])
        if len(lambda_values) == 0:
            raise ValueError("sweep模式下 lambda_list 不能为空")

        if args.resume:
            print("⚠️  --resume 仅在 single 模式下生效；sweep 模式将忽略")
        print(f"\n🎯 sweep 模式：共 {len(lambda_values)} 个lambda")
        print("   " + ", ".join([f"{v:g}" for v in lambda_values]))

        run_summaries = []
        for idx, lam in enumerate(lambda_values, start=1):
            print("\n" + "#" * 70)
            print(f"开始第 {idx}/{len(lambda_values)} 个实验: lambda_physics = {lam:g}")
            print("#" * 70)

            trainer = Trainer(
                config_path=config_path,
                lambda_physics_override=lam,
                physics_mode_override=exp['physics_mode'],
                run_tag=exp['run_tag'],
                processed_dir_override=data_dir,
                model_save_path_override=model_save_dir,
                training_profile=exp['training_profile'],
                physics_warmup_epochs_override=args.physics_warmup_epochs,
                angles_warmup_epochs_override=args.angles_warmup_epochs,
            )
            trainer.train(X_train, y_train, X_val, y_val, w_train=w_train, w_val=w_val)

            run_summaries.append({
                'lambda_physics': lam,
                'selection_metric': trainer.monitor_metric_name,
                'best_monitor_value': trainer.best_monitor_value,
                'best_val_rmse': trainer.best_val_rmse,
                'best_val_loss': trainer.best_val_loss,
                'best_val_wind_mag_error': trainer.best_val_wind_mag_error,
                'best_val_wind_mag_rmse': trainer.best_val_wind_mag_rmse,
                'best_val_direction_error': trainer.best_val_direction_error,
                'best_val_composite_score': trainer.best_val_composite_score,
                'best_epochs': trainer.best_epochs,
                'model_save_dir': trainer.model_save_dir
            })

        run_summaries = sorted(run_summaries, key=lambda x: x['best_monitor_value'])
        print("\n" + "=" * 70)
        print(f"✅ sweep 完成，按 {run_summaries[0]['selection_metric']} 从小到大排序：")
        print("=" * 70)
        for rank, item in enumerate(run_summaries, start=1):
            print(
                f"[{rank:02d}] λ={item['lambda_physics']:g} | "
                f"best_monitor={item['best_monitor_value']:.6f} | "
                f"best_val_rmse={item['best_val_rmse']:.6f} (epoch {item['best_epochs']['rmse']}) | "
                f"best_mag_rmse={item['best_val_wind_mag_rmse']:.6f} | "
                f"best_dir={item['best_val_direction_error']:.6f}° | "
                f"best_composite={item['best_val_composite_score']:.6f}"
            )
            print(f"     dir: {item['model_save_dir']}")

    print("\n" + "=" * 70)
    print("✅ 训练流程全部完成！")
    print("=" * 70)
    print(f"\n输出目录: {model_save_dir}")
    print("\n下一步:")
    print("  1. 查看 training_history.png")
    print("  2. 启动TensorBoard: tensorboard --logdir=./tensorboard_logs")
    print("  3. 进行模型评估: python src/4_eval_pigru.py")
    print("=" * 70)
