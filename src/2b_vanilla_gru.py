"""
Vanilla GRU 模型定义 - 论文对比基线
特点：
  - 与PI-GRU结构相同，但只输出风速估计
  - 作为纯数据驱动的对比基线（无物理损失、无自适应参数）
  - 用于证明物理信息损失的有效性
"""

import torch
import torch.nn as nn
import math


class VanillaGRU(nn.Module):
    """
    Vanilla GRU for UAV Wind Estimation (Data-Driven Baseline)
    
    纯数据驱动的GRU模型，作为论文Table 1中的对比基线。
    与PIGRU的主要区别：
    1. 只输出风速估计 [B, 3]，不输出 q_scale/r_scale/angles
      2. 训练时只使用MSE损失，不使用物理约束损失
    
    输入: [batch, seq_len, input_size]，修回实验使用 input_size=41
        特征顺序与 PIGRU.FEATURE_IDX 一致（src/2_pigru_module.py）：
          0-2   vel_n, vel_e, vel_d (地速 NED, 归一化)
          3-5   vel_x_body, vel_y_body, vel_z_body (机体速度)
          6-8   acc_x, acc_y, acc_z (机体/导航加速度，与预处理一致)
          9-11  roll, pitch, yaw (姿态角, rad)
          12-14 p_rate, q_rate, r_rate (机体角速度)
          15-18 aileron_cmd, elevator_cmd, rudder_cmd, throttle_cmd (控制律输出)
          19    airspeed (m/s)
          20-22 target_roll, target_pitch, target_yaw
          23-25 roll_err, pitch_err, yaw_err
          26-28 target_p, target_q, target_r
          29-31 p_err, q_err, r_err
          32-34 target_vn, target_ve, target_vd
          35-37 vn_err, ve_err, vd_err
          38-41 aileron_act, elevator_act, rudder_act, throttle_act (实际舵面)
          42-44 imu_ax, imu_ay, imu_az (机体系 IMU 加速度)
    
    输出:
        wind_estimate: [B, 3] 风速估计 (N/E/D)
    """
    
    def __init__(self, 
                 input_size=45, 
                 hidden_size=128, 
                 num_layers=2, 
                 dropout=0.2,
                 rnn_type='gru'):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: GRU隐藏层维度
            num_layers: GRU层数
            dropout: Dropout比例
        """
        super(VanillaGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = str(rnn_type).lower()
        rnn_cls = {'gru': nn.GRU, 'lstm': nn.LSTM}.get(self.rnn_type)
        if rnn_cls is None:
            raise ValueError(
                f"Unsupported rnn_type={rnn_type!r}; choose 'gru' or 'lstm'"
            )
        
        # Keep the historical attribute name for checkpoint compatibility.
        self.gru = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # ===== 风速估计头 =====
        self.fc_wind = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # [N, E, D]
        )
        
        # 参数初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x, return_dict=False):
        """
        前向传播
        
        Args:
            x: [B, T, input_size] 输入序列
            return_dict: 是否返回字典格式（为了与PIGRU接口兼容）
        
        Returns:
            if return_dict=False: wind_estimate [B, 3]
            if return_dict=True: {'wind_estimate': [B, 3]}
        """
        # GRU 特征提取
        gru_out, _ = self.gru(x)           # [B, T, H]
        h = gru_out[:, -1, :]              # 取最后一步 [B, H]
        
        # 风速估计
        wind_estimate = self.fc_wind(h)    # [B, 3]
        
        if return_dict:
            return {'wind_estimate': wind_estimate}
        return wind_estimate
    
    def get_model_info(self):
        """返回模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_type': f"Vanilla{self.rnn_type.upper()}",
            'rnn_type': self.rnn_type,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'has_physics_loss': False,
            'has_adaptive_params': False
        }
        return info


# ===== 测试代码 =====
if __name__ == "__main__":
    print("="*70)
    print(" Vanilla GRU 模型测试 (Data-Driven Baseline)")
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
        
        model = VanillaGRU(
            input_size=45,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        )
        
        info = model.get_model_info()
        print(f"  总参数量: {info['total_params']:,}")
        print(f"  可训练参数: {info['trainable_params']:,}")
        
        # 测试前向传播
        batch_size, seq_len = 32, 100
        x = torch.randn(batch_size, seq_len, 45)
        
        with torch.no_grad():
            # 直接输出模式
            wind = model(x, return_dict=False)
            print(f"\n  输入形状: {x.shape}")
            print(f"  输出形状: {wind.shape}")
            print(f"  输出范围: [{wind.min():.3f}, {wind.max():.3f}]")
            
            # 字典输出模式
            out = model(x, return_dict=True)
            print(f"  字典输出: {list(out.keys())}")
        
        print("  ✓ 测试通过")
    
    print("\n" + "="*70)
    print("✅ Vanilla GRU 模型测试完成")
    print("="*70)
    print("\n与PIGRU的区别:")
    print("  ✓ 只输出风速估计 [B, 3]")
    print("  ✓ 无 q_scale/r_scale 自适应噪声参数")
    print("  ✓ 无小角修正输出")
    print("  ✓ 训练时只使用MSE损失（无物理约束）")
