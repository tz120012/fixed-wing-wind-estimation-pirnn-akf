"""
模型评估模块 v3.0 (EKF融合增强版)
功能：全面评估PI-GRU模型性能
优化：
    - 适配字典输出和多通道 q_scale/r_scale
  - 新增自适应参数统计分析
  - 扩展物理一致性评估
    - 增强可视化（包含 q_scale/r_scale 分布）
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
from tqdm import tqdm
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None
from datetime import datetime

# 导入实际训练的模型定义
import importlib.util
model_file = os.path.join(os.path.dirname(__file__), '2_pigru_module.py')
spec = importlib.util.spec_from_file_location("model_definition", model_file)
if spec and spec.loader:
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    PIGRU = model_module.PIGRU
else:
    raise ImportError("Cannot load 2_pigru_module.py")


def resolve_project_path(project_root: str, path_value):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(project_root, path_value.lstrip('../')))


def parse_eval_splits(raw_value: str):
    alias_map = {
        'id': 'test_id',
        'test_id': 'test_id',
        'ood': 'test_ood',
        'test_ood': 'test_ood',
        'seq_id': 'test_sequential',
        'test_sequential': 'test_sequential',
        'seq_ood': 'test_seq_ood',
        'test_seq_ood': 'test_seq_ood',
    }
    valid_keys = ', '.join(sorted(alias_map.keys()))
    splits = []
    for raw_item in raw_value.split(','):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item not in alias_map:
            raise ValueError(f"不支持的 split: {raw_item}，可选值: {valid_keys}")
        canonical = alias_map[item]
        if canonical not in splits:
            splits.append(canonical)
    if not splits:
        raise ValueError('splits 不能为空')
    return splits


def load_evaluation_dataset(data_dir: str, split_name: str):
    split_map = {
        'test_id': ('X_test_id.npy', 'y_test_id.npy'),
        'test_ood': ('X_test_ood.npy', 'y_test_ood.npy'),
        'test_sequential': ('X_test_sequential.npy', 'y_test_sequential.npy'),
        'test_seq_ood': ('X_test_seq_ood.npy', 'y_test_seq_ood.npy'),
    }
    if split_name not in split_map:
        raise ValueError(f'未知 split: {split_name}')

    x_name, y_name = split_map[split_name]
    x_path = os.path.join(data_dir, x_name)
    y_path = os.path.join(data_dir, y_name)
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f'缺少评估数据文件: split={split_name}, x={x_path}, y={y_path}'
        )

    X_test = np.load(x_path)
    y_test = np.load(y_path)
    print(f"  ✓ {split_name}: X={X_test.shape}, y={y_test.shape}")
    return X_test, y_test


def plot_lambda_summary(summary, summary_dir: str, split_name: str):
    if not summary:
        print(f'  ⚠️ split={split_name} 没有可汇总的评估结果')
        return

    print(f"\n{'='*70}")
    print(f"  生成汇总对比图: {split_name}")

    summary.sort(key=lambda x: x[1])
    labels = [s[0] for s in summary]
    rmse_vals = [s[2]['rmse'] for s in summary]
    mae_vals = [s[2]['mae'] for s in summary]
    rmse_n = [s[2]['north_rmse'] for s in summary]
    rmse_e = [s[2]['east_rmse'] for s in summary]
    rmse_d = [s[2]['down_rmse'] for s in summary]
    rmse_mag = [s[2]['magnitude_rmse'] for s in summary]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'PI-GRU: Effect of λ_physics on Wind Estimation Performance ({split_name})',
        fontsize=13,
        fontweight='bold'
    )

    def _bar(ax, vals, title, ylabel='RMSE (m/s)', color='#4472C4'):
        bars = ax.bar(x, vals, color=color, alpha=0.8, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        best_idx = int(np.argmin(vals))
        for i, bar in enumerate(bars):
            color_bar = 'red' if i == best_idx else 'black'
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f'{bar.get_height():.3f}',
                ha='center',
                va='bottom',
                fontsize=7,
                color=color_bar,
                fontweight='bold' if i == best_idx else 'normal'
            )
        ax.get_children()[best_idx].set_edgecolor('red')
        ax.get_children()[best_idx].set_linewidth(2)

    _bar(axes[0, 0], rmse_vals, 'Overall RMSE', color='#4472C4')
    _bar(axes[0, 1], mae_vals, 'Overall MAE', ylabel='MAE (m/s)', color='#ED7D31')
    _bar(axes[1, 0], rmse_mag, 'Magnitude RMSE', color='#5B9BD5')

    ax = axes[1, 1]
    ax.plot(labels, rmse_n, 'o-', color='#4472C4', lw=1.5, ms=5, label='North')
    ax.plot(labels, rmse_e, 's-', color='#ED7D31', lw=1.5, ms=5, label='East')
    ax.plot(labels, rmse_d, '^-', color='#A9D18E', lw=1.5, ms=5, label='Down')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_title('Per-Component RMSE vs λ_physics', fontsize=10, fontweight='bold')
    ax.set_ylabel('RMSE (m/s)', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    summary_path = os.path.join(summary_dir, 'lambda_comparison.svg')
    fig.savefig(summary_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ 汇总对比图已保存: {summary_path}")

    print(f"\n{'='*70}")
    print(f"  各模型 RMSE 排名（split={split_name}，升序）")
    print(f"{'='*70}")
    ranked = sorted(summary, key=lambda x: x[2]['rmse'])
    for rank, (lbl, lam, metrics) in enumerate(ranked, 1):
        print(f"  #{rank}  {lbl:20s}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}")
    print(f"\n  最佳模型: {ranked[0][0]}  (RMSE={ranked[0][2]['rmse']:.4f} m/s)")


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config_path=None, model_dir=None, processed_dir_override=None,
                 model_save_path_override=None, checkpoint_name='best_model.pth'):
        """
        Args:
            config_path: 配置文件路径
            model_dir: 指定模型目录（如果为None，则使用最新的训练目录）
            processed_dir_override: 可选测试数据目录覆盖
            model_save_path_override: 可选模型根目录覆盖
            checkpoint_name: 要加载的 checkpoint 文件名，默认 `best_model.pth`
        """
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        if processed_dir_override is not None:
            self.config.setdefault('data', {})['processed_dir'] = processed_dir_override
        if model_save_path_override is not None:
            self.config.setdefault('training', {})['model_save_path'] = model_save_path_override
        
        # 创建带时间戳的评估输出目录
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_dir = model_dir  # 保存模型目录参数
        self.checkpoint_name = checkpoint_name
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 先加载归一化参数（load_model 在 yaw_invariant 模式下需要 scaler_X/scaler_y）
        self.load_normalization_params()

        # 加载模型
        self.model = self.load_model()
        self.model.eval()
        
        # 评估结果存储
        self.results = {}
    
    def get_evaluation_dir(self, sub: str = None):
        """获取评估结果保存目录（带时间戳，可选子目录）"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        eval_base_dir = os.path.join(project_root, 'data', 'evaluation')
        base = os.path.join(eval_base_dir, f'eval_pigru_{self.timestamp}')
        if sub:
            d = os.path.join(base, sub)
        else:
            d = base
        os.makedirs(d, exist_ok=True)
        return d
    
    def load_model(self):
        """加载训练好的模型"""
        # 处理相对路径
        model_save_path = self.config['training']['model_save_path']
        if not os.path.isabs(model_save_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_save_path = os.path.join(project_root, model_save_path.lstrip('../'))
        
        # 如果指定了model_dir，使用指定的目录；否则寻找最新的训练目录
        if self.model_dir:
            model_dir = self.model_dir
        else:
            # 查找最新的 train_* 目录
            train_dirs = [d for d in os.listdir(model_save_path) 
                         if d.startswith('train_') and os.path.isdir(os.path.join(model_save_path, d))]
            if not train_dirs:
                raise FileNotFoundError(f"在 {model_save_path} 中未找到训练目录（train_*）")
            train_dirs.sort(reverse=True)  # 按时间戳降序排序
            model_dir = os.path.join(model_save_path, train_dirs[0])
            print(f"\n自动选择最新的训练目录: {train_dirs[0]}")
        
        model_path = os.path.join(model_dir, self.checkpoint_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"\n加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 创建模型实例
        model = PIGRU(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=0.0,  # 评估时不使用dropout
            enable_wind_head=True
        ).to(self.device)

        # Yaw-invariant 模型需要在 load_state_dict 之前注册 buffer，
        # 否则 strict=True 加载时会因为缺少 _X_mean 等键报错
        if bool(self.config.get('model', {}).get('yaw_invariant', False)):
            model.yaw_invariant = True
            model.register_buffer('_X_mean', torch.tensor(self.scaler_X.mean_, dtype=torch.float32, device=self.device))
            model.register_buffer('_X_scale', torch.tensor(self.scaler_X.scale_, dtype=torch.float32, device=self.device))
            model.register_buffer('_y_mean', torch.tensor(self.scaler_y.mean_, dtype=torch.float32, device=self.device))
            model.register_buffer('_y_scale', torch.tensor(self.scaler_y.scale_, dtype=torch.float32, device=self.device))
            model.register_buffer('_track_along_mean', torch.tensor(15.05, device=self.device))
            model.register_buffer('_track_along_std',  torch.tensor(4.84, device=self.device))
            model.register_buffer('_track_cross_mean', torch.tensor(0.0, device=self.device))
            model.register_buffer('_track_cross_std',  torch.tensor(4.55, device=self.device))
            model.register_buffer('_yaw_diff_std',     torch.tensor(0.137, device=self.device))
            print(f"  ✓ Yaw-invariant 模式已启用")

        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 打印模型信息
        model_info = model.get_model_info()
        print(f"  ✓ 模型参数量: {model_info['total_params']:,}")
        print(f"  ✓ 输入维度: {model_info['input_size']}")
        print(f"  ✓ 隐藏层维度: {model_info['hidden_size']}")
        print(f"  ✓ GRU层数: {model_info['num_layers']}")
        
        if 'epoch' in checkpoint:
            print(f"  ✓ 训练轮数: {checkpoint['epoch']}")
        if 'selection_metric' in checkpoint:
            print(f"  ✓ 选模指标: {checkpoint['selection_metric']}")
        if 'best_val_rmse' in checkpoint:
            print(f"  ✓ 最佳验证RMSE: {checkpoint['best_val_rmse']:.4f}")
        if 'best_val_loss' in checkpoint:
            print(f"  ✓ 最佳验证损失: {checkpoint['best_val_loss']:.4f}")
        if 'best_val_wind_mag_error' in checkpoint:
            print(f"  ✓ 最佳风速大小误差: {checkpoint['best_val_wind_mag_error']:.4f}")
        
        return model
    
    def load_normalization_params(self):
        """加载归一化参数"""
        # 处理相对路径
        model_save_path = self.config['training']['model_save_path']
        if not os.path.isabs(model_save_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_save_path = os.path.join(project_root, model_save_path.lstrip('../'))
        
        # 归一化参数保存在基础目录（不是带时间戳的子目录）
        norm_path = os.path.join(model_save_path, 'norm_params.pkl')
        
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"归一化参数文件不存在: {norm_path}")
        
        print(f"\n加载归一化参数: {norm_path}")
        
        with open(norm_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler_X = metadata['scaler_X']
        self.scaler_y = metadata['scaler_y']
        
        # 提取反归一化参数
        self.y_mean = self.scaler_y.mean_
        self.y_std = self.scaler_y.scale_
        
        # 分离各部分的归一化参数
        self.wind_mean = self.y_mean[0:3]
        self.wind_std = self.y_std[0:3]
        self.vel_mean = self.y_mean[3:6]
        self.vel_std = self.y_std[3:6]
        self.airspeed_mean = self.y_mean[6]
        self.airspeed_std = self.y_std[6]
        
        print(f"  ✓ 输入维度: {metadata.get('input_size', 'Unknown')}")
        print(f"  ✓ 输出维度: {metadata.get('output_size', 'Unknown')}")
        print(f"  ✓ 风速均值: [{self.wind_mean[0]:.2f}, {self.wind_mean[1]:.2f}, {self.wind_mean[2]:.2f}] m/s")
        print(f"  ✓ 风速标准差: [{self.wind_std[0]:.2f}, {self.wind_std[1]:.2f}, {self.wind_std[2]:.2f}] m/s")
    
    def predict(self, X_test):
        """模型预测（适配字典输出）"""
        print("\n执行预测...")
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        all_wind_preds = []
        all_q_scales = []
        all_r_scales = []
        all_angles = []
        
        batch_size = 256
        
        with torch.no_grad():
            for i in tqdm(range(0, len(X_test_tensor), batch_size), desc='预测进度'):
                batch = X_test_tensor[i:i+batch_size]
                
                # 字典输出
                out = self.model(batch, return_dict=True)
                
                all_wind_preds.append(out['wind_estimate'].cpu().numpy())
                all_q_scales.append(out['q_scale'].cpu().numpy())
                all_r_scales.append(out['r_scale'].cpu().numpy())
                all_angles.append(out['angles'].cpu().numpy())
        
        wind_pred = np.vstack(all_wind_preds)  # [N, 3]
        q_scales = np.vstack(all_q_scales)     # [N, 3]
        r_scales = np.vstack(all_r_scales)     # [N, 3]
        angles = np.vstack(all_angles)         # [N, 3]
        
        print(f"  ✓ 预测完成: {len(wind_pred)} 样本")
        print(f"  ✓ 风速预测形状: {wind_pred.shape}")
        print(f"  ✓ q_scale 范围: [{q_scales.min():.3f}, {q_scales.max():.3f}]")
        print(f"  ✓ r_scale 范围: [{r_scales.min():.3f}, {r_scales.max():.3f}]")
        
        return wind_pred, q_scales, r_scales, angles
    
    def denormalize(self, wind_pred_norm, y_test_norm):
        """反归一化"""
        print("\n反归一化数据...")
        
        # 反归一化风速预测 (只有前3维)
        wind_pred = wind_pred_norm * self.wind_std + self.wind_mean
        
        # 反归一化真值 (7维)
        y_test_denorm = self.scaler_y.inverse_transform(y_test_norm)
        
        # 提取各部分
        wind_true = y_test_denorm[:, 0:3]   # 风速真值
        vel_gps = y_test_denorm[:, 3:6]     # GPS地速
        airspeed_true = y_test_denorm[:, 6] # 空速真值
        
        print(f"  ✓ 风速预测范围: [{wind_pred.min():.2f}, {wind_pred.max():.2f}] m/s")
        print(f"  ✓ 风速真值范围: [{wind_true.min():.2f}, {wind_true.max():.2f}] m/s")
        
        return wind_pred, wind_true, vel_gps, airspeed_true
    
    def calculate_metrics(self, wind_pred, wind_true, q_scales=None):
        """计算评估指标
        
        新增（vs 论文 v3）：除了整体 dir_MAE，额外报告
          1. 按真值 |w_h| 分箱的 dir_MAE 表（dir 平台诊断）
          2. dir_MAE @ |w_h| ≥ 1.5  m/s（工程实用区间）
          3. dir_MAE @ confidence top-90%（拒绝预测指标，需传 q_scales）
        其设计动机：dir_MAE 在弱风段(|w_h|<1.5)受 arctan2 几何不可解性影响，
        被一小撮 evil samples 拉高均值；细化指标用于把"模型真实能力" 与
        "几何噪声地板" 拆开，与 paper §4.5 的 evil-sample 诊断一脉相承。
        """
        print("\n计算评估指标...")
        
        metrics = {}
        
        # 整体指标
        metrics['rmse'] = np.sqrt(mean_squared_error(wind_true, wind_pred))
        metrics['mae'] = mean_absolute_error(wind_true, wind_pred)
        
        # 各分量指标
        component_names = ['north', 'east', 'down']
        for i, name in enumerate(component_names):
            metrics[f'{name}_rmse'] = np.sqrt(mean_squared_error(wind_true[:, i], wind_pred[:, i]))
            metrics[f'{name}_mae'] = mean_absolute_error(wind_true[:, i], wind_pred[:, i])
            metrics[f'{name}_r2'] = r2_score(wind_true[:, i], wind_pred[:, i])
        
        # 风速大小
        wind_mag_true = np.linalg.norm(wind_true, axis=1)
        wind_mag_pred = np.linalg.norm(wind_pred, axis=1)
        
        metrics['magnitude_rmse'] = np.sqrt(mean_squared_error(wind_mag_true, wind_mag_pred))
        metrics['magnitude_mae'] = mean_absolute_error(wind_mag_true, wind_mag_pred)
        metrics['magnitude_r2'] = r2_score(wind_mag_true, wind_mag_pred)
        
        # 风向误差 (水平分量)
        wind_dir_true = np.arctan2(wind_true[:, 1], wind_true[:, 0]) * 180 / np.pi
        wind_dir_pred = np.arctan2(wind_pred[:, 1], wind_pred[:, 0]) * 180 / np.pi
        
        # 处理角度差异 (-180 to 180)
        dir_error = wind_dir_pred - wind_dir_true
        dir_error = (dir_error + 180) % 360 - 180
        dir_err_abs = np.abs(dir_error)
        
        metrics['direction_mae'] = float(np.mean(dir_err_abs))
        metrics['direction_std'] = float(np.std(dir_error))
        
        # ============================================================
        # 【dir 细化诊断 v4】把 dir_MAE 拆成几何下界 vs 模型能力
        # ============================================================
        wind_h_true = np.sqrt(wind_true[:, 0] ** 2 + wind_true[:, 1] ** 2)
        rmse_h_per = np.sqrt((wind_pred[:, 0] - wind_true[:, 0]) ** 2 +
                             (wind_pred[:, 1] - wind_true[:, 1]) ** 2)

        # ---- (1) 按 |w_h| 分箱的 dir_MAE 表 ----
        bin_edges = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf]
        bin_labels = ['<0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0',
                      '2.0-2.5', '2.5-3.0', '>3.0']
        dir_by_bin = []
        for i, lb in enumerate(bin_labels):
            mask = (wind_h_true >= bin_edges[i]) & (wind_h_true < bin_edges[i + 1])
            n = int(mask.sum())
            if n == 0:
                continue
            seg_rmse_h = float(np.sqrt(np.mean(rmse_h_per[mask] ** 2)))
            seg_mean_w = float(wind_h_true[mask].mean())
            seg_dir = float(dir_err_abs[mask].mean())
            geom_lower = float(np.degrees(np.arctan(seg_rmse_h / max(seg_mean_w, 1e-6))))
            dir_by_bin.append(dict(
                label=lb, n=n, frac=n / len(wind_h_true),
                mean_w=seg_mean_w, rmse_h=seg_rmse_h,
                dir_mae=seg_dir, geom_lower=geom_lower,
            ))
        metrics['direction_mae_by_bin'] = dir_by_bin

        # ---- (2) |w_h| ≥ 1.5 m/s 的工程实用区间 ----
        mask_eng = wind_h_true >= 1.5
        if mask_eng.any():
            metrics['direction_mae_w_ge_1.5'] = float(dir_err_abs[mask_eng].mean())
            metrics['direction_mae_w_ge_1.5_n'] = int(mask_eng.sum())
            metrics['direction_mae_w_ge_1.5_frac'] = float(mask_eng.mean())

        # ---- (3) confidence top-90% (q_scale_h 越小越自信) ----
        if q_scales is not None and len(q_scales) == len(wind_pred):
            q_h = np.sqrt(q_scales[:, 0] ** 2 + q_scales[:, 1] ** 2)  # 水平方向不确定度
            keep_thr = np.percentile(q_h, 90)  # 拒掉 q_h 最高的 10%
            mask_keep = q_h <= keep_thr
            if mask_keep.any():
                metrics['direction_mae_conf_top90'] = float(dir_err_abs[mask_keep].mean())
                metrics['direction_mae_conf_top90_n'] = int(mask_keep.sum())
                metrics['direction_mae_conf_top90_frac'] = float(mask_keep.mean())
                metrics['direction_mae_conf_top90_q_thr'] = float(keep_thr)

        # ---- (4) 90% "正常样本" vs 10% "evil 样本" ----
        evil_mask = dir_err_abs > 30.0
        if evil_mask.any():
            metrics['direction_mae_evil_pct'] = float(evil_mask.mean())
            metrics['direction_mae_evil_avg'] = float(dir_err_abs[evil_mask].mean())
            metrics['direction_mae_clean'] = float(dir_err_abs[~evil_mask].mean())
            metrics['direction_mae_evil_w_med'] = float(np.median(wind_h_true[evil_mask]))

        # 统计信息
        metrics['wind_mag_true_mean'] = float(wind_mag_true.mean())
        metrics['wind_mag_true_std'] = float(wind_mag_true.std())
        metrics['wind_mag_pred_mean'] = float(wind_mag_pred.mean())
        metrics['wind_mag_pred_std'] = float(wind_mag_pred.std())
        
        print(f"  ✓ 整体RMSE: {metrics['rmse']:.3f} m/s")
        print(f"  ✓ 整体MAE: {metrics['mae']:.3f} m/s")
        print(f"  ✓ 风速大小RMSE: {metrics['magnitude_rmse']:.3f} m/s")
        print(f"  ✓ 风向MAE: {metrics['direction_mae']:.2f}°")
        if 'direction_mae_w_ge_1.5' in metrics:
            print(f"  ✓ 风向MAE @ |w|≥1.5: {metrics['direction_mae_w_ge_1.5']:.2f}° "
                  f"(覆盖 {100 * metrics['direction_mae_w_ge_1.5_frac']:.1f}% 样本)")
        if 'direction_mae_conf_top90' in metrics:
            print(f"  ✓ 风向MAE @ conf top-90%: {metrics['direction_mae_conf_top90']:.2f}°")
        if 'direction_mae_evil_pct' in metrics:
            print(f"  ✓ Evil 样本 (dir>30°) 占比: {100 * metrics['direction_mae_evil_pct']:.2f}%, "
                  f"平均 dir={metrics['direction_mae_evil_avg']:.1f}°; "
                  f"剔除后 clean dir_MAE={metrics['direction_mae_clean']:.2f}°")
        
        return metrics
    
    def analyze_adaptive_params(self, q_scales, r_scales, angles):
        """
        分析自适应参数的统计特性
        
        Args:
            q_scales: [N, 3] - 过程噪声缩放 (N/E/D)
            r_scales: [N, 3] - 量测噪声缩放 (GPS/TAS/ATT)
            angles: [N, 3] - [Δα, Δβ, s_tas]
        """
        print("\n" + "="*70)
        print("【自适应参数分析】")
        print("="*70)
        
        # q_scale 统计 (N/E/D)
        print("\n过程噪声 q_scale (N/E/D):")
        for i, axis in enumerate(['North', 'East', 'Down']):
            print(f"  {axis:5s}: μ={q_scales[:, i].mean():.3f} ± σ={q_scales[:, i].std():.3f} | "
                  f"范围=[{q_scales[:, i].min():.3f}, {q_scales[:, i].max():.3f}]")
        
        # r_scale 统计 (GPS/TAS/ATT)
        print("\n量测噪声 r_scale (GPS/TAS/ATT):")
        for i, sensor in enumerate(['GPS', 'TAS', 'ATT']):
            print(f"  {sensor:3s}: μ={r_scales[:, i].mean():.3f} ± σ={r_scales[:, i].std():.3f} | "
                  f"范围=[{r_scales[:, i].min():.3f}, {r_scales[:, i].max():.3f}]")
        
        # 小角修正统计
        print("\n小角修正 [Δα, Δβ, s_tas]:")
        d_alpha_deg = angles[:, 0] * 180 / np.pi
        d_beta_deg = angles[:, 1] * 180 / np.pi
        s_tas = angles[:, 2]
        
        print(f"  Δα (迎角修正):   μ={d_alpha_deg.mean():.3f}° ± σ={d_alpha_deg.std():.3f}° | "
              f"范围=[{d_alpha_deg.min():.2f}°, {d_alpha_deg.max():.2f}°]")
        print(f"  Δβ (侧滑修正):   μ={d_beta_deg.mean():.3f}° ± σ={d_beta_deg.std():.3f}° | "
              f"范围=[{d_beta_deg.min():.2f}°, {d_beta_deg.max():.2f}°]")
        print(f"  s_tas (空速尺度): μ={s_tas.mean():.4f} ± σ={s_tas.std():.4f} | "
              f"范围=[{s_tas.min():.4f}, {s_tas.max():.4f}]")
        
        # 相关性分析
        print("\nq_scale 各轴相关性矩阵:")
        alpha_corr = np.corrcoef(q_scales.T)
        print(f"         N      E      D")
        for i, axis in enumerate(['North', 'East', 'Down']):
            print(f"  {axis:5s} {alpha_corr[i, 0]:6.3f} {alpha_corr[i, 1]:6.3f} {alpha_corr[i, 2]:6.3f}")
        
        print("\nr_scale 各通道相关性矩阵:")
        r_scale_corr = np.corrcoef(r_scales.T)
        print(f"         GPS    TAS    ATT")
        for i, sensor in enumerate(['GPS', 'TAS', 'ATT']):
            print(f"  {sensor:3s} {r_scale_corr[i, 0]:6.3f} {r_scale_corr[i, 1]:6.3f} {r_scale_corr[i, 2]:6.3f}")
        
        # 存储统计结果
        adaptive_stats = {
            'q_scale_mean': q_scales.mean(axis=0).tolist(),
            'q_scale_std': q_scales.std(axis=0).tolist(),
            'q_scale_corr': alpha_corr.tolist(),
            'r_scale_mean': r_scales.mean(axis=0).tolist(),
            'r_scale_std': r_scales.std(axis=0).tolist(),
            'r_scale_corr': r_scale_corr.tolist(),
            'd_alpha_mean_deg': d_alpha_deg.mean(),
            'd_beta_mean_deg': d_beta_deg.mean(),
            's_tas_mean': s_tas.mean()
        }
        
        print("="*70)
        
        return adaptive_stats
    
    def evaluate_physics_consistency(self, wind_pred, vel_gps, airspeed_true):
        """
        评估物理一致性
        验证: V_air = V_ground - V_wind
        """
        print("\n评估物理一致性...")
        
        # 计算理论空速
        vel_air_theory = vel_gps - wind_pred
        airspeed_theory = np.linalg.norm(vel_air_theory, axis=1)
        
        # 空速误差
        airspeed_error = airspeed_theory - airspeed_true
        airspeed_rmse = np.sqrt(mean_squared_error(airspeed_true, airspeed_theory))
        airspeed_mae = mean_absolute_error(airspeed_true, airspeed_theory)
        
        # 相对误差
        relative_error = np.abs(airspeed_error) / (airspeed_true + 1e-3)
        relative_error_mean = np.mean(relative_error)
        
        physics_metrics = {
            'airspeed_rmse': airspeed_rmse,
            'airspeed_mae': airspeed_mae,
            'airspeed_relative_error': relative_error_mean,
            'airspeed_error_std': np.std(airspeed_error),
            'airspeed_theory_mean': airspeed_theory.mean(),
            'airspeed_true_mean': airspeed_true.mean()
        }
        
        print(f"  ✓ 空速RMSE: {airspeed_rmse:.3f} m/s")
        print(f"  ✓ 空速MAE: {airspeed_mae:.3f} m/s")
        print(f"  ✓ 空速相对误差: {relative_error_mean*100:.2f}%")
        print(f"  ✓ 理论空速: {airspeed_theory.mean():.2f} ± {airspeed_theory.std():.2f} m/s")
        print(f"  ✓ 测量空速: {airspeed_true.mean():.2f} ± {airspeed_true.std():.2f} m/s")
        
        return physics_metrics, airspeed_theory, airspeed_error
    
    def print_summary(self, metrics, physics_metrics, adaptive_stats):
        """打印评估摘要"""
        print("\n" + "="*70)
        print("  评估结果摘要")
        print("="*70)
        
        print("\n【整体性能】")
        print(f"  RMSE (均方根误差):        {metrics['rmse']:.3f} m/s")
        print(f"  MAE  (平均绝对误差):      {metrics['mae']:.3f} m/s")
        
        print("\n【各分量性能】")
        components = [('North (北)', 'north'), ('East (东)', 'east'), ('Down (下)', 'down')]
        for name, key in components:
            print(f"\n  {name}:")
            print(f"    RMSE:  {metrics[f'{key}_rmse']:.3f} m/s")
            print(f"    MAE:   {metrics[f'{key}_mae']:.3f} m/s")
            print(f"    R²:    {metrics[f'{key}_r2']:.3f}")
        
        print("\n【风速大小性能】")
        print(f"  RMSE:  {metrics['magnitude_rmse']:.3f} m/s")
        print(f"  MAE:   {metrics['magnitude_mae']:.3f} m/s")
        print(f"  R²:    {metrics['magnitude_r2']:.3f}")
        
        print("\n【风向性能】")
        print(f"  MAE:   {metrics['direction_mae']:.2f}°")
        print(f"  STD:   {metrics['direction_std']:.2f}°")

        # ===== v4 新增：dir 细化诊断 =====
        if 'direction_mae_w_ge_1.5' in metrics:
            print(f"\n  ◇ MAE @ |w_h|≥1.5 m/s (工程实用区间): "
                  f"{metrics['direction_mae_w_ge_1.5']:.2f}°  "
                  f"({100 * metrics['direction_mae_w_ge_1.5_frac']:.1f}% 样本)")
        if 'direction_mae_conf_top90' in metrics:
            print(f"  ◇ MAE @ confidence top-90% (拒掉 q_h 最高 10%): "
                  f"{metrics['direction_mae_conf_top90']:.2f}°  "
                  f"(q_h_thr={metrics['direction_mae_conf_top90_q_thr']:.3f})")
        if 'direction_mae_evil_pct' in metrics:
            print(f"  ◇ Evil (dir>30°): {100 * metrics['direction_mae_evil_pct']:.2f}% 样本; "
                  f"avg={metrics['direction_mae_evil_avg']:.1f}°; "
                  f"|w|_med={metrics['direction_mae_evil_w_med']:.2f} m/s; "
                  f"clean MAE={metrics['direction_mae_clean']:.2f}°")
        if metrics.get('direction_mae_by_bin'):
            print("\n  ◇ 按真值 |w_h| 分箱 dir_MAE：")
            print(f"    {'区间':<10s}  {'样本数':>9s}  {'占比':>6s}  {'均|w|':>7s}  "
                  f"{'RMSE_h':>8s}  {'实测dir':>9s}  {'几何下界':>9s}  {'差距':>7s}")
            print("    " + "-" * 80)
            for r in metrics['direction_mae_by_bin']:
                diff = r['dir_mae'] - r['geom_lower']
                tag = '✅' if abs(diff) < 2 else ('⚠↑' if diff > 0 else '🚀↓')
                print(f"    {r['label']:<10s}  {r['n']:>9,}  {100 * r['frac']:>5.2f}%  "
                      f"{r['mean_w']:>7.3f}  {r['rmse_h']:>8.3f}  "
                      f"{r['dir_mae']:>8.2f}°  {r['geom_lower']:>8.2f}°  "
                      f"{diff:>+6.2f}° {tag}")

        print("\n【风速统计】")
        print(f"  真值: {metrics['wind_mag_true_mean']:.2f} ± {metrics['wind_mag_true_std']:.2f} m/s")
        print(f"  预测: {metrics['wind_mag_pred_mean']:.2f} ± {metrics['wind_mag_pred_std']:.2f} m/s")
        
        print("\n【物理一致性】")
        print(f"  空速RMSE:      {physics_metrics['airspeed_rmse']:.3f} m/s")
        print(f"  空速MAE:       {physics_metrics['airspeed_mae']:.3f} m/s")
        print(f"  空速相对误差:  {physics_metrics['airspeed_relative_error']*100:.2f}%")
        
        print("\n【自适应噪声参数】")
        q_scale_mean = adaptive_stats['q_scale_mean']
        r_scale_mean = adaptive_stats['r_scale_mean']
        print(f"  q_scale (N/E/D):   [{q_scale_mean[0]:.2f}, {q_scale_mean[1]:.2f}, {q_scale_mean[2]:.2f}]")
        print(f"  r_scale (GPS/TAS/ATT): [{r_scale_mean[0]:.2f}, {r_scale_mean[1]:.2f}, {r_scale_mean[2]:.2f}]")
        print(f"  Δα: {adaptive_stats['d_alpha_mean_deg']:.2f}°")
        print(f"  Δβ: {adaptive_stats['d_beta_mean_deg']:.2f}°")
        print(f"  s_tas: {adaptive_stats['s_tas_mean']:.4f}")
        
        print("\n" + "="*70)
    
    def plot_results(self, wind_pred, wind_true, airspeed_theory, airspeed_true,
                    airspeed_error, q_scales, r_scales, angles, eval_dir=None):
        """
        主评估图 — 3×3 学术风格
          行1: North / East / Down 分量 True vs Predicted 散点图
          行2: 风速大小散点 / 水平风向散点 / 物理一致性（空速）散点
          行3: North / East / Down 误差分布直方图（含 KDE）
        """
        print("\n生成可视化图表...")
        from scipy.stats import gaussian_kde as _kde
        if eval_dir is None:
            eval_dir = self.get_evaluation_dir()

        wind_mag_true = np.linalg.norm(wind_true, axis=1)
        wind_mag_pred = np.linalg.norm(wind_pred, axis=1)
        wind_dir_true = np.arctan2(wind_true[:, 1], wind_true[:, 0]) * 180 / np.pi
        wind_dir_pred = np.arctan2(wind_pred[:, 1], wind_pred[:, 0]) * 180 / np.pi
        dir_error     = (wind_dir_pred - wind_dir_true + 180) % 360 - 180

        COLORS = {'north': '#4472C4', 'east': '#ED7D31', 'down': '#A9D18E',
                  'mag': '#5B9BD5', 'dir': '#7030A0', 'air': '#FF8C00'}

        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        fig.suptitle('PI-GRU Wind Estimation Evaluation',
                     fontsize=14, fontweight='bold', y=0.98)

        comp_names  = ['North', 'East', 'Down']
        comp_colors = [COLORS['north'], COLORS['east'], COLORS['down']]

        for i, (name, color) in enumerate(zip(comp_names, comp_colors)):
            ax = axes[0, i]
            ax.scatter(wind_true[:, i], wind_pred[:, i],
                       alpha=0.2, s=2, c=color, rasterized=True, label='Samples')
            lims = [min(wind_true[:, i].min(), wind_pred[:, i].min()),
                    max(wind_true[:, i].max(), wind_pred[:, i].max())]
            ax.plot(lims, lims, 'r--', lw=1.5, label='Ideal', zorder=10)
            rmse = np.sqrt(mean_squared_error(wind_true[:, i], wind_pred[:, i]))
            r2   = r2_score(wind_true[:, i], wind_pred[:, i])
            ax.set_title(f'{name} Wind\nRMSE={rmse:.3f} m/s, R\u00b2={r2:.3f}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel(f'True {name} (m/s)', fontsize=9)
            ax.set_ylabel(f'Predicted {name} (m/s)', fontsize=9)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        ax = axes[1, 0]
        ax.scatter(wind_mag_true, wind_mag_pred, alpha=0.2, s=2, c=COLORS['mag'], rasterized=True)
        lims = [wind_mag_true.min(), wind_mag_true.max()]
        ax.plot(lims, lims, 'r--', lw=1.5, label='Ideal', zorder=10)
        rmse_m = np.sqrt(mean_squared_error(wind_mag_true, wind_mag_pred))
        r2_m   = r2_score(wind_mag_true, wind_mag_pred)
        ax.set_title(f'Wind Magnitude\nRMSE={rmse_m:.3f} m/s, R\u00b2={r2_m:.3f}', fontsize=10, fontweight='bold')
        ax.set_xlabel('True Magnitude (m/s)', fontsize=9)
        ax.set_ylabel('Predicted Magnitude (m/s)', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        ax = axes[1, 1]
        ax.scatter(wind_dir_true, wind_dir_pred, alpha=0.2, s=2, c=COLORS['dir'], rasterized=True)
        ax.plot([-180, 180], [-180, 180], 'r--', lw=1.5, label='Ideal', zorder=10)
        ax.set_title(f'Horizontal Wind Direction\nMAE={np.mean(np.abs(dir_error)):.2f}\u00b0', fontsize=10, fontweight='bold')
        ax.set_xlabel('True Direction (\u00b0)', fontsize=9)
        ax.set_ylabel('Predicted Direction (\u00b0)', fontsize=9)
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        ax = axes[1, 2]
        ax.scatter(airspeed_true, airspeed_theory, alpha=0.2, s=2, c=COLORS['air'], rasterized=True)
        lims = [min(airspeed_true.min(), airspeed_theory.min()),
                max(airspeed_true.max(), airspeed_theory.max())]
        ax.plot(lims, lims, 'r--', lw=1.5, label='Ideal', zorder=10)
        rmse_a = np.sqrt(mean_squared_error(airspeed_true, airspeed_theory))
        r2_a   = r2_score(airspeed_true, airspeed_theory)
        ax.set_title(f'Physics Consistency (Airspeed)\nRMSE={rmse_a:.3f} m/s, R\u00b2={r2_a:.3f}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Measured Airspeed (m/s)', fontsize=9)
        ax.set_ylabel('Theory Airspeed (m/s)', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        for i, (name, color) in enumerate(zip(comp_names, comp_colors)):
            ax = axes[2, i]
            err = wind_pred[:, i] - wind_true[:, i]
            ax.hist(err, bins=60, alpha=0.65, color=color, edgecolor='white', linewidth=0.3, density=True)
            x = np.linspace(err.min(), err.max(), 200)
            ax.plot(x, _kde(err)(x), color='navy', lw=1.5, label='KDE')
            ax.axvline(0,          color='red',  lw=1.5, linestyle='--', label='Zero')
            ax.axvline(err.mean(), color='black', lw=1.5, linestyle='-', label=f'\u03bc={err.mean():.3f}')
            ax.set_title(f'{name} Error Distribution\n\u03bc={err.mean():.3f}, \u03c3={err.std():.3f} m/s', fontsize=10, fontweight='bold')
            ax.set_xlabel('Error (m/s)', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        path = os.path.join(eval_dir, 'evaluation_results.svg')
        plt.savefig(path, format='svg', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  \u2713 主评估图已保存: {path}")

        self.plot_waveform(wind_pred, wind_true, eval_dir)
        self.plot_metrics_bar(wind_pred, wind_true, eval_dir)
        self.plot_adaptive_params(q_scales, r_scales, angles, eval_dir)
        self.plot_angle_corrections(angles, eval_dir)
        self.plot_correlation_heatmaps(q_scales, r_scales, eval_dir)

    def plot_waveform(self, wind_pred, wind_true, eval_dir):
        """N/E/D + 风速大小 波形对比图"""
        print("  生成波形对比图...")
        dt   = 1.0 / self.config['data']['sampling_rate']
        N    = len(wind_pred)
        t    = np.arange(N) * dt
        step = max(1, N // 4000)
        mag_t = np.linalg.norm(wind_true, axis=1)
        mag_p = np.linalg.norm(wind_pred, axis=1)
        COLORS = {'north': '#4472C4', 'east': '#ED7D31', 'down': '#A9D18E', 'mag': '#5B9BD5'}
        comp_cfg = [
            ('North',     wind_true[:, 0], wind_pred[:, 0], COLORS['north']),
            ('East',      wind_true[:, 1], wind_pred[:, 1], COLORS['east']),
            ('Down',      wind_true[:, 2], wind_pred[:, 2], COLORS['down']),
            ('Magnitude', mag_t,           mag_p,           COLORS['mag']),
        ]
        fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)
        fig.suptitle('PI-GRU \u2014 Estimated vs True Wind (Time Series)', fontsize=13, fontweight='bold')
        for ax, (label, true, pred, color) in zip(axes, comp_cfg):
            ax.plot(t[::step], true[::step], color='#555555', lw=0.7, alpha=0.8, label='True')
            ax.plot(t[::step], pred[::step], color=color,     lw=0.9, alpha=0.9, label='Estimated')
            rmse = np.sqrt(np.mean((pred - true) ** 2))
            ax.set_ylabel(f'{label}\n(m/s)', fontsize=9)
            ax.set_title(f'{label}  RMSE={rmse:.3f} m/s', fontsize=9, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right', ncol=2)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        axes[-1].set_xlabel('Time (s)', fontsize=10)
        plt.tight_layout()
        path = os.path.join(eval_dir, 'wind_waveform.svg')
        fig.savefig(path, format='svg', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  \u2713 波形图已保存: {path}")

    def plot_metrics_bar(self, wind_pred, wind_true, eval_dir):
        """各分量 RMSE/MAE 柱状图"""
        print("  生成指标柱状图...")
        names = ['North', 'East', 'Down', 'Magnitude']
        mag_t = np.linalg.norm(wind_true, axis=1)
        mag_p = np.linalg.norm(wind_pred, axis=1)
        all_t = [wind_true[:, 0], wind_true[:, 1], wind_true[:, 2], mag_t]
        all_p = [wind_pred[:, 0], wind_pred[:, 1], wind_pred[:, 2], mag_p]
        rmse_vals = [np.sqrt(mean_squared_error(t, p)) for t, p in zip(all_t, all_p)]
        mae_vals  = [mean_absolute_error(t, p)          for t, p in zip(all_t, all_p)]
        x = np.arange(len(names)); w = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        b1 = ax.bar(x - w/2, rmse_vals, w, label='RMSE', color='#4472C4', alpha=0.85)
        b2 = ax.bar(x + w/2, mae_vals,  w, label='MAE',  color='#ED7D31', alpha=0.85)
        ax.set_title('PI-GRU \u2014 Per-Component Error Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error (m/s)', fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        path = os.path.join(eval_dir, 'metrics_bar.svg')
        fig.savefig(path, format='svg', bbox_inches='tight')
        plt.close()
        print(f"  \u2713 指标柱状图已保存: {path}")

    def plot_adaptive_params(self, q_scales, r_scales, angles, eval_dir):
        """PI-GRU 自适应参数综合图 — 2×3 布局"""
        print("  生成自适应参数图...")
        from scipy.stats import gaussian_kde as _kde
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle('PI-GRU Adaptive Parameters Distribution\n'
                     '(q_scale: Process Noise Scale  |  r_scale: Measurement Noise Scale)',
                     fontsize=12, fontweight='bold')
        alpha_labels = ['q_scale North', 'q_scale East', 'q_scale Down']
        beta_labels  = ['r_scale GPS',   'r_scale TAS',  'r_scale ATT']
        alpha_colors = ['#4472C4', '#ED7D31', '#A9D18E']
        beta_colors  = ['#5B9BD5', '#FF8C00', '#7030A0']
        def _safe_kde_plot(ax, data, color, label):
            x = np.linspace(data.min() - 0.1, data.max() + 0.1, 200)
            ax.hist(data, bins=50, alpha=0.6, color=color, edgecolor='white', linewidth=0.3, density=True)
            if data.std() > 1e-6:
                try:
                    ax.plot(x, _kde(data)(x), color='navy', lw=1.5, label='KDE')
                except Exception:
                    ax.axvline(data.mean(), color='navy', lw=1.5, linestyle='-', label='KDE(退化)')
            else:
                ax.axvline(data.mean(), color='navy', lw=1.5, linestyle='-', label='常数(KDE不适用)')
            ax.axvline(data.mean(), color='red', lw=1.5, linestyle='--', label=f'\u03bc={data.mean():.3f}')
            ax.set_title(f'{label}\n\u03bc={data.mean():.3f} \u00b1 \u03c3={data.std():.3f}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Scale Factor', fontsize=9); ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        for i, (label, color) in enumerate(zip(alpha_labels, alpha_colors)):
            _safe_kde_plot(axes[0, i], q_scales[:, i], color, label)
        for i, (label, color) in enumerate(zip(beta_labels, beta_colors)):
            _safe_kde_plot(axes[1, i], r_scales[:, i], color, label)
        plt.tight_layout()
        path = os.path.join(eval_dir, 'adaptive_params.svg')
        fig.savefig(path, format='svg', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  \u2713 自适应参数图已保存: {path}")

        
        # ===== 额外绘图：小角修正和相关性 =====
        self.plot_angle_corrections(angles, eval_dir)
        self.plot_correlation_heatmaps(q_scales, r_scales, eval_dir)
    
    def plot_angle_corrections(self, angles, eval_dir=None):
        """绘制小角修正分析"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        d_alpha_deg = angles[:, 0] * 180 / np.pi
        d_beta_deg = angles[:, 1] * 180 / np.pi
        s_tas = angles[:, 2]
        
        # Δα 分布
        axes[0, 0].hist(d_alpha_deg, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(d_alpha_deg.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'μ={d_alpha_deg.mean():.3f}°')
        axes[0, 0].set_xlabel('Δα (degrees)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title(f'Attack Angle Correction\nμ={d_alpha_deg.mean():.3f}° ± σ={d_alpha_deg.std():.3f}°',
                            fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].legend(fontsize=10, loc='upper right')
        
        # Δβ 分布
        axes[0, 1].hist(d_beta_deg, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(d_beta_deg.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'μ={d_beta_deg.mean():.3f}°')
        axes[0, 1].set_xlabel('Δβ (degrees)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title(f'Sideslip Angle Correction\nμ={d_beta_deg.mean():.3f}° ± σ={d_beta_deg.std():.3f}°',
                            fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].legend(fontsize=10, loc='upper right')
        
        # s_tas 分布
        axes[1, 0].hist(s_tas, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(s_tas.mean(), color='red', linestyle='--', linewidth=2,
                          label=f'μ={s_tas.mean():.4f}')
        axes[1, 0].axvline(1.0, color='blue', linestyle=':', linewidth=2, label='Nominal=1.0')
        axes[1, 0].set_xlabel('s_tas (scale)', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title(f'Airspeed Scale Correction\nμ={s_tas.mean():.4f} ± σ={s_tas.std():.4f}',
                            fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].legend(fontsize=10, loc='upper right')
        
        # 时序图（采样）
        sample_indices = np.arange(0, len(angles), max(1, len(angles)//1000))
        axes[1, 1].plot(sample_indices, d_alpha_deg[sample_indices], 
                       alpha=0.6, linewidth=0.8, label='Δα', color='blue')
        axes[1, 1].plot(sample_indices, d_beta_deg[sample_indices], 
                       alpha=0.6, linewidth=0.8, label='Δβ', color='green')
        axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero')
        axes[1, 1].set_xlabel('Sample Index', fontsize=11)
        axes[1, 1].set_ylabel('Angle (degrees)', fontsize=11)
        axes[1, 1].set_title('Angle Corrections Time Series (Sampled)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=10, loc='upper right')
        
        plt.tight_layout()
        
        # 保存到 data/evaluation 目录
        if eval_dir is None:
            eval_dir = self.get_evaluation_dir()
        save_path = os.path.join(eval_dir, 'angle_corrections.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"  ✓ 小角修正图已保存: {save_path}")
        plt.close()
    
    def plot_correlation_heatmaps(self, q_scales, r_scales, eval_dir=None):
        """绘制相关性热力图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        def draw_heatmap(ax, corr, labels):
            if sns is not None:
                sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',
                            xticklabels=labels, yticklabels=labels,
                            vmin=-1, vmax=1, center=0, ax=ax,
                            cbar_kws={'label': 'Correlation'})
            else:
                im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_xticks(np.arange(len(labels)))
                ax.set_yticks(np.arange(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
                for i in range(corr.shape[0]):
                    for j in range(corr.shape[1]):
                        ax.text(j, i, f'{corr[i, j]:.3f}', ha='center', va='center', color='black')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Correlation')

        # q_scale 相关性
        alpha_corr = np.corrcoef(q_scales.T)
        draw_heatmap(axes[0], alpha_corr, ['N', 'E', 'D'])
        axes[0].set_title('q_scale Correlation Matrix\n(Process Noise N/E/D)', 
                         fontsize=12, fontweight='bold')
        
        # r_scale 相关性
        r_scale_corr = np.corrcoef(r_scales.T)
        draw_heatmap(axes[1], r_scale_corr, ['GPS', 'TAS', 'ATT'])
        axes[1].set_title('r_scale Correlation Matrix\n(Measurement Noise GPS/TAS/ATT)', 
                         fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if eval_dir is None:
            eval_dir = self.get_evaluation_dir()
        save_path = os.path.join(eval_dir, 'correlation_heatmaps.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"  ✓ 相关性热力图已保存: {save_path}")
        plt.close()
    
    def save_results(self, metrics, physics_metrics, adaptive_stats, eval_dir=None, eval_split=None):
        """保存评估结果"""
        if eval_dir is None:
            eval_dir = self.get_evaluation_dir()
        save_path = os.path.join(eval_dir, 'evaluation_metrics.pkl')

        results = {
            'metrics': metrics,
            'physics_metrics': physics_metrics,
            'adaptive_stats': adaptive_stats,
            'config': self.config,
            'eval_split': eval_split,
            'model_dir': self.model_dir,
        }

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"\n  ✓ 评估结果已保存: {save_path}")
    
    def run(self, X_test, y_test, eval_dir=None, eval_split='test_id'):
        """执行完整评估流程

        Args:
            X_test: 测试输入数据
            y_test: 测试标签数据
            eval_dir: 指定输出目录；若为 None 则自动创建 eval_pigru_<timestamp>
            eval_split: 当前评估数据集名称
        """
        print("\n" + "="*70)
        print("  PI-GRU 风速估计模型评估 v3.0")
        print("="*70)

        print(f"\n测试集信息:")
        print(f"  split:    {eval_split}")
        print(f"  样本数:   {X_test.shape[0]}")
        print(f"  输入形状: {X_test.shape}")
        print(f"  标签形状: {y_test.shape}")

        if eval_dir is None:
            eval_dir = self.get_evaluation_dir()

        wind_pred_norm, q_scales, r_scales, angles = self.predict(X_test)
        wind_pred, wind_true, vel_gps, airspeed_true = self.denormalize(
            wind_pred_norm, y_test
        )
        metrics = self.calculate_metrics(wind_pred, wind_true, q_scales=q_scales)
        physics_metrics, airspeed_theory, airspeed_error = self.evaluate_physics_consistency(
            wind_pred, vel_gps, airspeed_true
        )
        adaptive_stats = self.analyze_adaptive_params(q_scales, r_scales, angles)

        self.print_summary(metrics, physics_metrics, adaptive_stats)
        self.plot_results(
            wind_pred,
            wind_true,
            airspeed_theory,
            airspeed_true,
            airspeed_error,
            q_scales,
            r_scales,
            angles,
            eval_dir=eval_dir,
        )
        self.save_results(
            metrics,
            physics_metrics,
            adaptive_stats,
            eval_dir=eval_dir,
            eval_split=eval_split,
        )

        print("\n" + "="*70)
        print("  ✅ 评估完成！")
        print("="*70)

        return metrics, physics_metrics, adaptive_stats


if __name__ == "__main__":
    import re

    parser = argparse.ArgumentParser(description="PI-GRU 多 lambda_physics 批量评估")
    parser.add_argument("--config_path", type=str, default=None,
                        help="可选配置文件路径，默认使用 config/config.yaml")
    parser.add_argument("--processed_dir_override", type=str, default=None,
                        help="覆盖评估使用的数据目录")
    parser.add_argument("--model_save_path_override", type=str, default=None,
                        help="覆盖模型根目录")
    parser.add_argument("--checkpoint_name", type=str, default="best_model.pth",
                        help="要评估的 checkpoint 文件名，如 best_model.pth 或 best_dir_model.pth")
    parser.add_argument("--splits", type=str, default="test_id",
                        help="逗号分隔的评估数据集，如 test_id,test_ood 或 test_sequential,test_seq_ood")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="评估结果根目录；默认 data/evaluation/eval_pigru_<timestamp>")
    args = parser.parse_args()

    print("=" * 70)
    print(" PI-GRU 多 lambda_physics 批量评估")
    print("=" * 70)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = args.config_path or os.path.join(project_root, 'config', 'config.yaml')

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        processed_dir_override = resolve_project_path(project_root, args.processed_dir_override)
        model_save_path_override = resolve_project_path(project_root, args.model_save_path_override)
        if processed_dir_override is not None:
            config.setdefault('data', {})['processed_dir'] = processed_dir_override
        if model_save_path_override is not None:
            config.setdefault('training', {})['model_save_path'] = model_save_path_override

        eval_splits = parse_eval_splits(args.splits)
        data_dir = resolve_project_path(project_root, config['data']['processed_dir'])
        print(f"\n评估数据目录: {data_dir}")
        print(f"评估 splits: {', '.join(eval_splits)}")
        print(f"评估 checkpoint: {args.checkpoint_name}")

        model_save_path = resolve_project_path(project_root, config['training']['model_save_path'])
        train_dirs = sorted([
            d for d in os.listdir(model_save_path)
            if d.startswith('train_') and
               os.path.isdir(os.path.join(model_save_path, d)) and
               os.path.exists(os.path.join(model_save_path, d, args.checkpoint_name))
        ])

        if not train_dirs:
            raise FileNotFoundError(f"未找到任何 train_* 目录: {model_save_path}")

        print(f"\n找到 {len(train_dirs)} 个训练目录:")
        for d in train_dirs:
            print(f"  {d}")

        from datetime import datetime as _dt
        run_timestamp = _dt.now().strftime('%Y%m%d_%H%M%S')
        if args.output_dir:
            top_eval_dir = resolve_project_path(project_root, args.output_dir)
        else:
            top_eval_dir = os.path.join(project_root, 'data', 'evaluation', f'eval_pigru_{run_timestamp}')
        os.makedirs(top_eval_dir, exist_ok=True)
        print(f"\n评估结果根目录: {top_eval_dir}")

        for eval_split in eval_splits:
            print(f"\n{'#' * 70}")
            print(f"开始评估 split: {eval_split}")
            print(f"{'#' * 70}")

            X_test, y_test = load_evaluation_dataset(data_dir, eval_split)
            split_eval_dir = os.path.join(top_eval_dir, eval_split)
            os.makedirs(split_eval_dir, exist_ok=True)
            summary = []

            for train_dir_name in train_dirs:
                model_dir = os.path.join(model_save_path, train_dir_name)

                import torch as _torch
                ckpt = _torch.load(
                    os.path.join(model_dir, args.checkpoint_name),
                    map_location='cpu',
                    weights_only=False,
                )

                lam = None
                m = re.search(r'train_lambda([0-9.]+)', train_dir_name)
                if m:
                    lam = float(m.group(1))

                lam_ckpt = ckpt.get('config', {}).get('training', {}).get('lambda_physics', None)
                try:
                    lam_ckpt_f = float(lam_ckpt) if lam_ckpt is not None else None
                except (TypeError, ValueError):
                    lam_ckpt_f = None

                if lam is None and lam_ckpt_f is not None:
                    lam = lam_ckpt_f
                elif lam is not None and lam_ckpt_f is not None and abs(lam - lam_ckpt_f) > 1e-9:
                    print(f"  ⚠️ 检测到 lambda 记录不一致: dir={lam:g}, ckpt={lam_ckpt_f:g}，采用目录名")

                lam_display = f"{lam:g}" if lam is not None else "unknown"
                label = f"lambda_{lam_display}"

                print(f"\n{'=' * 70}")
                print(f"  评估: {train_dir_name} | split={eval_split} | lambda_physics={lam_display}")
                print(f"{'=' * 70}")

                evaluator = ModelEvaluator(
                    config_path=config_path,
                    model_dir=model_dir,
                    processed_dir_override=processed_dir_override,
                    model_save_path_override=model_save_path_override,
                    checkpoint_name=args.checkpoint_name,
                )

                checkpoint_tag = os.path.splitext(args.checkpoint_name)[0]
                eval_dir = os.path.join(split_eval_dir, f"{label}__{train_dir_name}__{checkpoint_tag}")
                os.makedirs(eval_dir, exist_ok=True)
                metrics, _, _ = evaluator.run(
                    X_test,
                    y_test,
                    eval_dir=eval_dir,
                    eval_split=eval_split,
                )

                summary.append((label, lam if lam is not None else -1, metrics))
                print(f"  ✓ 结果已保存到子目录: {eval_dir}")

            plot_lambda_summary(summary, split_eval_dir, eval_split)

    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
