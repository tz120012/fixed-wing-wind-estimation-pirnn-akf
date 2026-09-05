"""
PI-GRU模型定义 v3.0 (EKF融合增强版)
新增：
    - 多通道 q_scale/r_scale 输出 (N/E/D独立 + GPS/TAS/ATT独立)
  - 小角修正输出 [Δα, Δβ, s_tas]
  - 字典输出模式 (向后兼容旧接口)
  - 在线推理辅助 (对数域EMA平滑)
  - Sigmoid + 线性映射保证正值范围
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
# 特征下标常量（须与 src/1_preprocessing_data.py FEATURE_IDX 一致）
# 当前数据集（240→50 Hz 下采样后）= 完整 45 维。历史阶段：
#   阶段 1：input_size=32（仅使用 0–31，不含 target_v / 实际舵面 / IMU 加速度）
#   阶段 2：input_size=45（完整布局，当前默认）
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


# ─────────────────────────────────────────────
# 输入特征组划分（用于输入特征组消融实验）。
# 消融方式：在归一化域将某组特征列置 0（= 该特征恒为均值，无信息），
# 训练与评估须置零相同组以保持分布一致。索引须与上方 FEATURE_IDX 严格对应。
# ─────────────────────────────────────────────
FEATURE_GROUPS = {
    "kinematics": [0, 1, 2, 3, 4, 5, 6, 7, 8],        # 地速 NED / 体速 / 机体加速度
    "attitude": [9, 10, 11, 12, 13, 14],              # 姿态角 + 机体角速度
    "control_cmd": [15, 16, 17, 18],                   # 舵面指令 + 油门指令
    "airspeed": [19],                                  # 空速（速度三角闭合关键量）
    "px4_attitude_target": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],  # PX4 目标姿态/误差/目标角速度/误差
    "velocity_target": [32, 33, 34, 35, 36, 37],       # 目标速度 + 速度误差
    "actuator": [38, 39, 40, 41],                      # 实际舵面（飞控补偿信息）
    "imu_accel": [42, 43, 44],                         # IMU 机体加速度
}


def resolve_feature_mask_indices(group_names):
    """把特征组名列表解析为需置零的特征列索引（升序去重）。未知组名抛错。"""
    idx = set()
    for name in group_names:
        key = name.strip()
        if not key:
            continue
        if key not in FEATURE_GROUPS:
            raise KeyError(f"Unknown feature group '{key}'; valid: {sorted(FEATURE_GROUPS)}")
        idx.update(FEATURE_GROUPS[key])
    return sorted(idx)


class PIGRU(nn.Module):
    """
    Physics-Informed GRU for UAV Wind Estimation (EKF-Enhanced)
    
    输入: [batch, seq_len, input_size]，input_size 由 config.yaml 控制（**当前数据 = 45**）
        特征顺序见模块顶层 FEATURE_IDX 常量；详细布局参考 src/1_preprocessing_data.py FEATURE_IDX。
        历史阶段保留供回滚：
          阶段 1 (32维)：原 20 维 + 目标姿态(3) + 姿态误差(3) + 目标角速度(3) + 角速度误差(3)
          阶段 2 (45维)：阶段 1 + 目标速度(3) + 速度误差(3) + 实际舵面(4) + IMU加速度(3) ← 当前默认
    
    输出 (字典模式):
        - wind_estimate: [B, 3]   风速估计 (N/E/D) - 训练监督/可选伪量测
        - q_scale: [B, 3]         过程噪声缩放 (N/E/D 独立) - 正值
        - r_scale: [B, 3]         量测噪声缩放 (GPS/TAS/ATT) - 正值
        - angles: [B, 3]          [Δα, Δβ, s_tas] - 迎角/侧滑修正 + 空速尺度
        - confidence: [B, 1]      置信度 s_k ∈ [0, 1] - 用于自适应遗忘因子调制
    """
    
    def __init__(self, 
                 input_size=45, 
                 hidden_size=128, 
                 num_layers=2, 
                 dropout=0.2,
                 rnn_type='gru',             # 循环骨干类型：'gru' | 'lstm'（骨干对比消融用；LSTM output 接口与 GRU 一致）
                 enable_wind_head=True,      # 是否启用风速回归头（部署时可关闭节省计算）
                 enable_noise_heads=True,    # 是否启用 q_scale / r_scale 辅助头
                 enable_angles_head=True,    # 是否启用角度修正头
                 enable_confidence_head=True,# 是否启用置信度头
                 angle_limit_deg=5.0,        # 小角限幅（度）
                 s_tas_range=(0.9, 1.1),     # 空速尺度范围
                 qr_scale_range=(0.3, 10.0), # q_scale/r_scale 输出范围
                 alpha_beta_range=None,      # 兼容旧配置键（已弃用）
                 yaw_invariant=False,        # 是否启用 yaw-invariant 表示
                 norm_params=None            # yaw_invariant=True 时必传：含 X_mean/X_scale/y_mean/y_scale
                 ):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: GRU隐藏层维度
            num_layers: GRU层数
            dropout: Dropout比例
            enable_wind_head: 是否启用风速估计头（训练时True，部署可选False）
            angle_limit_deg: 小角修正的限幅（角度）
            s_tas_range: 空速尺度因子的范围 (lo, hi)
            qr_scale_range: q_scale/r_scale 的输出范围 (lo, hi)
            alpha_beta_range: 旧参数名，等价于 qr_scale_range
        """
        super(PIGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.enable_wind_head = enable_wind_head
        self.enable_noise_heads = enable_noise_heads
        self.enable_angles_head = enable_angles_head
        self.enable_confidence_head = enable_confidence_head
        self.angle_limit = math.radians(angle_limit_deg)
        self.s_tas_lo, self.s_tas_hi = s_tas_range
        if alpha_beta_range is not None:
            qr_scale_range = alpha_beta_range
        self.alpha_lo, self.alpha_hi = qr_scale_range
        
        # ===== 循环主干（GRU 默认；LSTM 供骨干对比消融，forward 只取 output 序列，接口一致）=====
        self.rnn_type = str(rnn_type).lower()
        _rnn_cls = {'gru': nn.GRU, 'lstm': nn.LSTM}.get(self.rnn_type)
        if _rnn_cls is None:
            raise ValueError(f"Unsupported rnn_type={rnn_type!r}; choose 'gru' or 'lstm'")
        self.gru = _rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # ===== 风速估计头（训练监督/可选伪量测） =====
        if enable_wind_head:
            self.fc_wind = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 3)  # [N, E, D]
            )
        
        # ===== 过程噪声缩放 q_scale (N/E/D 三轴独立) =====
        if self.enable_noise_heads:
            self.head_alphaQ = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 3)  # 输出 raw logits
            )
        
        # ===== 量测噪声缩放 β (GPS/TAS/ATT 三通道独立) =====
        if self.enable_noise_heads:
            self.head_beta = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 3)
            )
        
        # ===== 小角与空速尺度修正 [Δα, Δβ, s_tas] =====
        if self.enable_angles_head:
            self.head_angles = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 3)
            )
        
        # ===== 置信度输出头 s_k ∈ [0, 1] =====
        # 论文 Section 3.1.2: 网络输出置信度用于调制自适应遗忘因子
        if self.enable_confidence_head:
            self.head_confidence = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()  # 输出 [0, 1]
            )
        
        # ===== Yaw-invariant 表示（B 方案）=====
        # 强制模型对绝对航向角不变：内部坐标变换到 track 系（沿/垂直机体航向），
        # 输出风速时再反变换回 NED。配合 A（rotation augmentation）形成双重保险。
        # track 系下 vel_along（沿航向）≈ 空速，vel_cross（垂直航向）≈ 0。
        self.yaw_invariant = bool(yaw_invariant)
        if self.yaw_invariant:
            if norm_params is None:
                raise ValueError("yaw_invariant=True 必须提供 norm_params (含 X_mean, X_scale, y_mean, y_scale)")
            X_mean = torch.as_tensor(norm_params['X_mean'], dtype=torch.float32)
            X_scale = torch.as_tensor(norm_params['X_scale'], dtype=torch.float32)
            y_mean = torch.as_tensor(norm_params['y_mean'], dtype=torch.float32)
            y_scale = torch.as_tensor(norm_params['y_scale'], dtype=torch.float32)
            # 用 buffer 注册，自动跟随 .to(device) 和 state_dict 保存
            self.register_buffer('_X_mean', X_mean)
            self.register_buffer('_X_scale', X_scale)
            self.register_buffer('_y_mean', y_mean)
            self.register_buffer('_y_scale', y_scale)
            # track 系下的 vel 物理量级与 NED 完全不同（vel_along ~14 m/s 而非 ~0），
            # 直接用 NED 归一化参数会让模型见到分布外输入。这里给 track 系字段
            # 单独的归一化常数（基于 UAV 飞行物理常识，无需重新预处理数据）。
            self.register_buffer('_track_along_mean', torch.tensor(15.05, dtype=torch.float32))
            self.register_buffer('_track_along_std',  torch.tensor(4.84, dtype=torch.float32))
            self.register_buffer('_track_cross_mean', torch.tensor(0.0, dtype=torch.float32))
            self.register_buffer('_track_cross_std',  torch.tensor(4.55, dtype=torch.float32))
            self.register_buffer('_yaw_diff_std',     torch.tensor(0.137, dtype=torch.float32))

        # 参数初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        权重初始化（Xavier/Kaiming）
        对输出层使用较小的初始化，避免训练初期的数值不稳定
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.orthogonal_(param)  # GRU用正交初始化
                else:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # 输出层特殊初始化
        # 目标：让初始 q_scale/r_scale 接近 1.0（在物理上合理的中间值）
        # 策略：调整 bias 让 sigmoid 映射后输出 1.0
        # 计算: target=1.0, range=[0.3, 10.0]
        #       ratio = (1.0-0.3)/(10.0-0.3) = 0.0722
        #       bias = sigmoid^-1(0.0722) = -2.554
        import math
        target_init = 1.0  # 目标初始值
        lo, hi = self.alpha_lo, self.alpha_hi
        target_ratio = (target_init - lo) / (hi - lo)
        init_bias = math.log(target_ratio / (1 - target_ratio + 1e-8))
        
        if hasattr(self, 'head_alphaQ'):
            # 增大权重初始化方差，引入通道差异性
            nn.init.normal_(self.head_alphaQ[-1].weight, mean=0, std=0.01)  # 0.001 → 0.01
            # 关键：bias 加入随机扰动，打破通道对称性
            # 让三个通道从不同的起点开始学习
            bias_noise = torch.randn(3) * 0.3  # ±0.3 的随机扰动
            nn.init.constant_(self.head_alphaQ[-1].bias, 0)
            self.head_alphaQ[-1].bias.data += init_bias + bias_noise
        
        if hasattr(self, 'head_beta'):
            nn.init.normal_(self.head_beta[-1].weight, mean=0, std=0.01)
            bias_noise = torch.randn(3) * 0.3
            nn.init.constant_(self.head_beta[-1].bias, 0)
            self.head_beta[-1].bias.data += init_bias + bias_noise
        
        if hasattr(self, 'head_angles'):
            # 角度修正初始化为 0，尺度初始化为 1.0
            nn.init.normal_(self.head_angles[-1].weight, mean=0, std=0.001)
            nn.init.constant_(self.head_angles[-1].bias, 0)
        
        if hasattr(self, 'head_confidence'):
            # 置信度初始化：让初始输出接近 0.5（中等置信度）
            nn.init.normal_(self.head_confidence[-2].weight, mean=0, std=0.01)
            nn.init.constant_(self.head_confidence[-2].bias, 0)  # sigmoid(0) = 0.5
    
    def _map_positive(self, raw, lo, hi):
        """
        将原始输出映射到正值范围 [lo, hi]
        **最终方案**：Sigmoid 软限幅 + 线性缩放（但简化偏置）
          1. 直接 Sigmoid，保证输出在 [0, 1]
          2. 线性缩放到 [lo, hi]
        
        核心思想：
          - Sigmoid 始终有梯度（不会为0）
          - 范围确保在 [lo, hi] 内
          - 初始化时 raw≈0 → sigmoid(0)=0.5 → 输出接近中点
        
        Args:
            raw: [B, 3] 原始输出
            lo, hi: 目标范围
        
        Returns:
            [B, 3] 映射到 [lo, hi] 的值
        """
        # Sigmoid: 输出 [0, 1]
        normed = torch.sigmoid(raw)
        
        # 线性映射到 [lo, hi]
        # 当 raw=0 时，normed=0.5，输出 = (lo+hi)/2
        return lo + normed * (hi - lo)
    
    def _map_angles_scale(self, raw_angles):
        """
        映射小角和尺度因子
        
        Args:
            raw_angles: [B, 3] 原始输出
        
        Returns:
            [B, 3] [Δα, Δβ, s_tas]
              - Δα, Δβ: 弧度，范围 [-angle_limit, +angle_limit]
              - s_tas: 尺度因子，范围 [s_tas_lo, s_tas_hi]
        """
        # Δα, Δβ: tanh映射到 [-1, 1]，再乘以限幅
        d_alpha = torch.tanh(raw_angles[:, 0:1]) * self.angle_limit
        d_beta  = torch.tanh(raw_angles[:, 1:2]) * self.angle_limit
        
        # s_tas: sigmoid映射到 [0, 1]，再映射到 [s_lo, s_hi]
        s_raw = torch.sigmoid(raw_angles[:, 2:3])
        s_tas = self.s_tas_lo + s_raw * (self.s_tas_hi - self.s_tas_lo)
        
        return torch.cat([d_alpha, d_beta, s_tas], dim=1)  # [B, 3]

    def _default_noise_scale(self, h):
        """辅助头关闭时，返回 1.0 的默认 q/r scale。"""
        return torch.ones(h.shape[0], 3, device=h.device, dtype=h.dtype)

    def _default_angles(self, h):
        """角度头关闭时，返回 [0, 0, 1]。"""
        angles = torch.zeros(h.shape[0], 3, device=h.device, dtype=h.dtype)
        angles[:, 2] = 1.0
        return angles

    def _default_confidence(self, h):
        """置信度头关闭时，返回中性置信度 0.5。"""
        return torch.full((h.shape[0], 1), 0.5, device=h.device, dtype=h.dtype)
    
    def _to_track_frame(self, x):
        """把 NED 输入序列变换到 track（航向）系。

        参考航向：序列最后一时间步的 yaw（即预测时刻的飞机航向）。
        变换：
            vel_along = cos(yaw) * vel_n + sin(yaw) * vel_e   (沿机体航向)
            vel_cross = -sin(yaw) * vel_n + cos(yaw) * vel_e  (机体右翼方向)
            yaw_diff  = wrap(yaw - yaw_ref)                   (相对参考的航向偏差)

        归一化用 track 系的专用常数（vs NED 完全不同的分布）。

        Returns:
            x_track: [B, T, input_size] 变换后的输入（track 系归一化空间）
            cos_y, sin_y: [B] 反变换需要的旋转角
        """
        IDX_VEL_N = FEATURE_IDX["vel_n"]
        IDX_VEL_E = FEATURE_IDX["vel_e"]
        IDX_YAW = FEATURE_IDX["yaw"]

        # 1) 反归一化 yaw → 物理角度
        yaw_phys = x[..., IDX_YAW] * self._X_scale[IDX_YAW] + self._X_mean[IDX_YAW]  # [B, T]
        yaw_ref = yaw_phys[:, -1]  # [B] 序列最后一时间步的 yaw
        cos_y = torch.cos(yaw_ref).unsqueeze(1)  # [B, 1]，广播到时间维
        sin_y = torch.sin(yaw_ref).unsqueeze(1)

        # 2) 反归一化 vel_n/e → 物理速度
        vel_n_phys = x[..., IDX_VEL_N] * self._X_scale[IDX_VEL_N] + self._X_mean[IDX_VEL_N]
        vel_e_phys = x[..., IDX_VEL_E] * self._X_scale[IDX_VEL_E] + self._X_mean[IDX_VEL_E]

        # 3) 旋转到 track 系
        vel_along = cos_y * vel_n_phys + sin_y * vel_e_phys
        vel_cross = -sin_y * vel_n_phys + cos_y * vel_e_phys

        # 4) yaw → yaw_diff (相对参考航向的偏差，wrap 到 [-π, π])
        yaw_diff = yaw_phys - yaw_ref.unsqueeze(1)
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))

        # 5) 用 track 系专用归一化常数
        x_track = x.clone()
        x_track[..., IDX_VEL_N] = (vel_along - self._track_along_mean) / self._track_along_std
        x_track[..., IDX_VEL_E] = (vel_cross - self._track_cross_mean) / self._track_cross_std
        x_track[..., IDX_YAW] = yaw_diff / self._yaw_diff_std  # mean=0

        return x_track, cos_y.squeeze(1), sin_y.squeeze(1)

    def _wind_track_to_ned(self, wind_pred_norm, cos_y, sin_y):
        """把模型在 track 系预测的 wind 反变换回 NED 归一化空间。

        模型 fc_wind 在 track 系学习，输出 wind_pred_norm 视为
        "track 系风速 [along, cross, down] 在 NED-wind 归一化空间的表达"。
        训练损失在 NED 归一化空间计算，所以需要旋转回 NED。

        反变换路径：track 归一化空间 → 物理空间 → 旋转回 NED → NED 归一化空间

        旋转关系：
            wind_n_phys = cos(yaw) * wind_along_phys - sin(yaw) * wind_cross_phys
            wind_e_phys = sin(yaw) * wind_along_phys + cos(yaw) * wind_cross_phys
        """
        # 反归一化：将归一化值视为 track 系物理风速（用 NED-wind 归一化参数）
        wind_along_phys = wind_pred_norm[:, 0] * self._y_scale[0] + self._y_mean[0]
        wind_cross_phys = wind_pred_norm[:, 1] * self._y_scale[1] + self._y_mean[1]

        # 物理空间旋转回 NED
        wind_n_phys = cos_y * wind_along_phys - sin_y * wind_cross_phys
        wind_e_phys = sin_y * wind_along_phys + cos_y * wind_cross_phys

        # 重归一化到 NED-wind 空间，使输出与训练标签 y_batch[:, :3] 同空间
        wind_n_norm = (wind_n_phys - self._y_mean[0]) / self._y_scale[0]
        wind_e_norm = (wind_e_phys - self._y_mean[1]) / self._y_scale[1]

        out = wind_pred_norm.clone()
        out[:, 0] = wind_n_norm
        out[:, 1] = wind_e_norm
        # wind_d (out[:, 2]) 不变 —— 垂直分量与 yaw 旋转无关
        return out

    def forward(self, x, return_dict=True):
        """
        前向传播
        
        Args:
            x: [B, T, input_size] 输入序列
            return_dict: True -> 返回字典; False -> 返回元组（向后兼容）
        
        Returns (return_dict=True):
            {
                'wind_estimate': [B, 3],  # 风速估计 (N/E/D)
                                'q_scale': [B, 3],         # 过程噪声缩放 (N/E/D)
                                'r_scale': [B, 3],         # 量测噪声缩放 (GPS/TAS/ATT)
                'angles': [B, 3]           # [Δα, Δβ, s_tas]
            }
        
        Returns (return_dict=False，向后兼容):
                        (wind_estimate, q_scale_scalar, r_scale_scalar)
              - wind_estimate: [B, 3]
                            - q_scale_scalar: [B, 1] q_scale 的均值
                            - r_scale_scalar: [B, 1] r_scale 的均值
        """
        # ===== Yaw-invariant 输入变换（B 方案）=====
        cos_y, sin_y = None, None
        if self.yaw_invariant:
            x, cos_y, sin_y = self._to_track_frame(x)

        # ===== GRU 特征提取 =====
        gru_out, _ = self.gru(x)           # [B, T, H]
        h = gru_out[:, -1, :]              # 取最后一步 [B, H]
        
        out = {}
        
        # ===== 风速估计 =====
        if self.enable_wind_head:
            wind_pred = self.fc_wind(h)
            # Yaw-invariant 时模型在 track 系预测，需反变换回 NED 归一化空间
            if self.yaw_invariant:
                wind_pred = self._wind_track_to_ned(wind_pred, cos_y, sin_y)
            out['wind_estimate'] = wind_pred
        else:
            out['wind_estimate'] = torch.zeros(h.shape[0], 3, device=h.device)
        
        # ===== 过程噪声 q_scale [B, 3] =====
        if self.enable_noise_heads:
            raw_aq = self.head_alphaQ(h)
            out['q_scale'] = self._map_positive(raw_aq, self.alpha_lo, self.alpha_hi)
        else:
            out['q_scale'] = self._default_noise_scale(h)
        
        # ===== 量测噪声 β [B, 3] =====
        if self.enable_noise_heads:
            raw_b = self.head_beta(h)
            out['r_scale'] = self._map_positive(raw_b, self.alpha_lo, self.alpha_hi)
        else:
            out['r_scale'] = self._default_noise_scale(h)
        
        # ===== 小角修正 [B, 3] =====
        if self.enable_angles_head:
            raw_ang = self.head_angles(h)
            out['angles'] = self._map_angles_scale(raw_ang)
        else:
            out['angles'] = self._default_angles(h)
        
        # ===== 置信度 s_k [B, 1] =====
        # 论文 Section 3.2.2: α_k = sigmoid(δ * s_k + ε)
        if self.enable_confidence_head:
            out['confidence'] = self.head_confidence(h)  # [B, 1], 范围 [0, 1]
        else:
            out['confidence'] = self._default_confidence(h)
        
        if return_dict:
            return out
        
        # ===== 向后兼容模式（不推荐） =====
        wind = out['wind_estimate']
        q_scale_scalar = out['q_scale'].mean(dim=1, keepdim=True)  # [B, 1]
        r_scale_scalar = out['r_scale'].mean(dim=1, keepdim=True)  # [B, 1]
        return wind, q_scale_scalar, r_scale_scalar
    
    def forward_sequence(self, x):
        """
        输出序列内每一帧的 wind_estimate，供 AKF 逐帧融合使用

        Args:
            x: [B, T, input_size]

        Returns:
            dict:
              'wind_seq':  [B, T, 3]  每帧风速估计
              'q_scale':   [B, 3]     最后帧的过程噪声缩放
              'r_scale':   [B, 3]     最后帧的测量噪声缩放
              'angles':    [B, 3]     最后帧的小角修正
              'confidence':[B, 1]     最后帧的置信度
        """
        # ===== Yaw-invariant 输入变换 =====
        cos_y, sin_y = None, None
        if self.yaw_invariant:
            x, cos_y, sin_y = self._to_track_frame(x)

        gru_out, _ = self.gru(x)                    # [B, T, H]
        h_last = gru_out[:, -1, :]                  # [B, H]

        # 每帧 wind estimate：把 fc_wind 应用到全序列
        B, T, H = gru_out.shape
        h_all = gru_out.reshape(B * T, H)           # [B*T, H]
        if self.enable_wind_head:
            wind_seq = self.fc_wind(h_all).reshape(B, T, 3)  # [B, T, 3]
            if self.yaw_invariant:
                # 整个序列每帧用同一 yaw_ref（最后一时间步）反变换
                wind_seq_flat = wind_seq.reshape(B * T, 3)
                cos_y_rep = cos_y.repeat_interleave(T)  # [B*T]
                sin_y_rep = sin_y.repeat_interleave(T)
                wind_seq_flat = self._wind_track_to_ned(wind_seq_flat, cos_y_rep, sin_y_rep)
                wind_seq = wind_seq_flat.reshape(B, T, 3)
        else:
            wind_seq = torch.zeros(B, T, 3, device=gru_out.device, dtype=gru_out.dtype)

        # 其余输出只取最后帧
        out = {}
        out['wind_seq']   = wind_seq
        out['wind_estimate'] = wind_seq[:, -1, :]   # 向后兼容

        if self.enable_noise_heads:
            raw_aq = self.head_alphaQ(h_last)
            out['q_scale'] = self._map_positive(raw_aq, self.alpha_lo, self.alpha_hi)

            raw_b = self.head_beta(h_last)
            out['r_scale'] = self._map_positive(raw_b, self.alpha_lo, self.alpha_hi)
        else:
            out['q_scale'] = self._default_noise_scale(h_last)
            out['r_scale'] = self._default_noise_scale(h_last)

        if self.enable_angles_head:
            raw_ang = self.head_angles(h_last)
            out['angles'] = self._map_angles_scale(raw_ang)
        else:
            out['angles'] = self._default_angles(h_last)

        if self.enable_confidence_head:
            out['confidence'] = self.head_confidence(h_last)
        else:
            out['confidence'] = self._default_confidence(h_last)

        return out

    def predict_online(self, x, prev_log_q=None, prev_log_r=None,
                      ema_alpha=0.1, clamp=True):
        """
        在线推理辅助函数（带对数域 EMA 平滑 + 限幅）
        
        用途：在实际部署中，对 q_scale/r_scale 进行时序平滑，避免突变
        
        Args:
            x: [B, T, input_size] 输入序列
            prev_log_q: [B, 3] 上一步的 log(q_scale)，首次为None
            prev_log_r: [B, 3] 上一步的 log(r_scale)，首次为None
            ema_alpha: 平滑系数 ∈ (0, 1)，越小越平滑
                      新值权重 = ema_alpha，旧值权重 = 1 - ema_alpha
            clamp: 是否二次限幅（防止EMA导致的范围溢出）
        
        Returns:
            out: 字典 + 额外的 'log_q_scale', 'log_r_scale' 用于下次EMA
        """
        # 前向传播
        out = self.forward(x, return_dict=True)
        
        # 对数域 EMA 平滑（在log空间更稳定）
        log_q = torch.log(out['q_scale'] + 1e-8)
        log_r = torch.log(out['r_scale'] + 1e-8)
        
        if prev_log_q is not None:
            log_q = (1 - ema_alpha) * prev_log_q + ema_alpha * log_q
        if prev_log_r is not None:
            log_r = (1 - ema_alpha) * prev_log_r + ema_alpha * log_r
        
        # 指数恢复到线性空间
        q_scale_smooth = torch.exp(log_q)
        r_scale_smooth = torch.exp(log_r)
        
        # 二次限幅（可选，防止数值漂移）
        if clamp:
            q_scale_smooth = torch.clamp(q_scale_smooth, self.alpha_lo, self.alpha_hi)
            r_scale_smooth = torch.clamp(r_scale_smooth, self.alpha_lo, self.alpha_hi)
        
        # 更新输出
        out['q_scale'] = q_scale_smooth
        out['r_scale'] = r_scale_smooth
        
        # 保存log值供下次使用（需要detach避免梯度累积）
        out['log_q_scale'] = log_q.detach()
        out['log_r_scale'] = log_r.detach()
        
        return out
    
    def get_model_info(self):
        """返回模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'enable_wind_head': self.enable_wind_head,
            'enable_noise_heads': self.enable_noise_heads,
            'enable_angles_head': self.enable_angles_head,
            'enable_confidence_head': self.enable_confidence_head,
            'angle_limit_deg': math.degrees(self.angle_limit),
            's_tas_range': (self.s_tas_lo, self.s_tas_hi),
            'qr_scale_range': (self.alpha_lo, self.alpha_hi)
        }
        return info


# ===== 测试代码 =====
if __name__ == "__main__":
    print("="*70)
    print(" PI-GRU v3.0 模型测试 (EKF融合增强版)")
    print("="*70)
    print()
    
    configs = [
        {'hidden_size': 64, 'num_layers': 2, 'name': '轻量级配置'},
        {'hidden_size': 128, 'num_layers': 2, 'name': '标准配置 ⭐'},
        {'hidden_size': 256, 'num_layers': 3, 'name': '大容量配置'}
    ]
    
    for config in configs:
        print(f"\n【{config['name']}】")
        print(f"  Hidden Size: {config['hidden_size']}")
        print(f"  Num Layers: {config['num_layers']}")
        
        model = PIGRU(
            input_size=45,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            enable_wind_head=True,
            angle_limit_deg=5.0,
            s_tas_range=(0.9, 1.1),
            qr_scale_range=(0.3, 10.0)
        )
        
        info = model.get_model_info()
        print(f"  总参数量: {info['total_params']:,}")
        print(f"  可训练参数: {info['trainable_params']:,}")
        
        # 测试前向传播
        batch_size, seq_len = 32, 100
        x = torch.randn(batch_size, seq_len, 45)
        
        print(f"\n  输入形状: {x.shape}")
        
        with torch.no_grad():
            # ===== 字典模式 =====
            out = model(x, return_dict=True)
            
            print(f"\n  【字典输出模式】")
            print(f"    wind_estimate: {out['wind_estimate'].shape} | "
                  f"范围: [{out['wind_estimate'].min():.3f}, {out['wind_estimate'].max():.3f}]")
            print(f"    q_scale: {out['q_scale'].shape} | "
                  f"范围: [{out['q_scale'].min():.3f}, {out['q_scale'].max():.3f}]")
            print(f"    r_scale: {out['r_scale'].shape} | "
                  f"范围: [{out['r_scale'].min():.3f}, {out['r_scale'].max():.3f}]")
            print(f"    angles: {out['angles'].shape}")
            print(f"      Δα: [{out['angles'][:, 0].min()*180/math.pi:.2f}°, "
                  f"{out['angles'][:, 0].max()*180/math.pi:.2f}°]")
            print(f"      Δβ: [{out['angles'][:, 1].min()*180/math.pi:.2f}°, "
                  f"{out['angles'][:, 1].max()*180/math.pi:.2f}°]")
            print(f"      s_tas: [{out['angles'][:, 2].min():.3f}, "
                  f"{out['angles'][:, 2].max():.3f}]")
            print(f"    confidence: {out['confidence'].shape} | "
                  f"范围: [{out['confidence'].min():.3f}, {out['confidence'].max():.3f}]")
            
            # ===== 兼容模式 =====
            wind, q_scale_scalar, r_scale_scalar = model(x, return_dict=False)
            print(f"\n  【兼容模式（元组输出）】")
            print(f"    wind: {wind.shape}, q_scale: {q_scale_scalar.shape}, r_scale: {r_scale_scalar.shape}")
            
            # ===== 在线推理测试 =====
            print(f"\n  【在线推理模式（EMA平滑）】")
            prev_log_q = None
            prev_log_r = None
            
            for step in range(3):
                out_online = model.predict_online(
                    x[:2, :, :],  # 模拟小批量
                    prev_log_q=prev_log_q,
                    prev_log_r=prev_log_r,
                    ema_alpha=0.1,
                    clamp=True
                )
                
                # 更新状态
                prev_log_q = out_online['log_q_scale']
                prev_log_r = out_online['log_r_scale']
                
                print(f"    Step {step+1}: q_scale={out_online['q_scale'][0].cpu().numpy()}, "
                      f"r_scale={out_online['r_scale'][0].cpu().numpy()}, "
                      f"confidence={out_online['confidence'][0].item():.3f}")
        
        print("  ✓ 测试通过")
    
    print("\n" + "="*70)
    print("✅ 所有配置测试完成")
    print("="*70)
    print("\n主要特性:")
    print("  ✓ 多通道 q_scale/r_scale 输出 (N/E/D + GPS/TAS/ATT)")
    print("  ✓ 小角修正输出 [Δα, Δβ, s_tas]")
    print("  ✓ 置信度输出 s_k ∈ [0, 1] ← 新增")
    print("  ✓ Sigmoid + 线性映射保证正值范围")
    print("  ✓ 字典输出模式（向后兼容元组）")
    print("  ✓ 在线推理EMA平滑")
    print("  ✓ 参数初始化优化")