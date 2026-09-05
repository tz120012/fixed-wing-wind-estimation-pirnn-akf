"""
PIRNN-AKF 评估模块 - 论文核心方法评估
与其他方法保持相同接口，用于多方法对比
"""

import numpy as np
import pickle
import yaml
import os
import sys
import re
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 导入PIRNN-AKF
# 修复导入路径
import importlib.util
module_path = os.path.join(os.path.dirname(__file__), '5_pigru_akf_fusion.py')
spec = importlib.util.spec_from_file_location("pirnn_akf_fusion", module_path)
if spec and spec.loader:
    pirnn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pirnn_module)
    PIRNN_AKF = pirnn_module.PIRNN_AKF
else:
    raise ImportError("Cannot load 5_pigru_akf_fusion.py")

preproc_module_path = os.path.join(os.path.dirname(__file__), '1_data_preprocessing_csv.py')
preproc_spec = importlib.util.spec_from_file_location("data_preprocessing_csv", preproc_module_path)
if os.path.exists(preproc_module_path) and preproc_spec and preproc_spec.loader:
    preproc_module = importlib.util.module_from_spec(preproc_spec)
    preproc_spec.loader.exec_module(preproc_module)
    CSVDataPreprocessor = preproc_module.CSVDataPreprocessor
else:
    CSVDataPreprocessor = None


def resolve_project_path(project_root: str, path_value):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(project_root, path_value.lstrip('../')))


class PIRNNAKFEvaluator:

    """PIRNN-AKF 评估器"""
    
    def __init__(self, config_path=None, model_path=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        else:
            config_path = os.path.abspath(config_path)

        self.config_path = config_path
        self.project_root = project_root

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建PIRNN-AKF估计器
        self.estimator = PIRNN_AKF(config_path, model_path)

        
        print(f"\n【PIRNN-AKF 评估器初始化】")
        model_info = self.estimator.get_model_info()
        for k, v in model_info.items():
            print(f"  {k}: {v}")
    
    def get_evaluation_dir(self):
        """获取评估结果保存目录"""
        eval_base_dir = os.path.join(self.project_root, 'data', 'evaluation')
        eval_dir = os.path.join(eval_base_dir, f'eval_pigru_akf_{self.timestamp}')
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir

    def _resolve_project_path(self, path_value):
        if os.path.isabs(path_value):
            return path_value
        return os.path.normpath(os.path.join(self.project_root, path_value.lstrip('../')))

    def load_or_build_sequential_test_set(self, split_name='test_id', force_rebuild=False):
        """优先加载连续测试集缓存；若不存在则从原始CSV按飞行段构建并缓存。"""
        data_dir = self._resolve_project_path(self.config['data']['processed_dir'])
        seq_X = os.path.join(data_dir, 'X_test_sequential.npy')
        seq_y = os.path.join(data_dir, 'y_test_sequential.npy')
        seq_s = os.path.join(data_dir, 'seg_test_sequential.npy')

        if (not force_rebuild) and all(os.path.exists(p) for p in [seq_X, seq_y, seq_s]):
            X_test = np.load(seq_X)
            y_test = np.load(seq_y)
            segments = np.load(seq_s)
            print(f"  ✓ X_test_sequential: {X_test.shape}  ({len(segments)} 段时序缓存)")
            return X_test, y_test, segments

        csv_root = self._resolve_project_path(self.config['data']['csv_dir'])
        split_dir = os.path.join(csv_root, split_name)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"未找到连续测试集目录: {split_dir}")
        if CSVDataPreprocessor is None:
            raise ImportError("Cannot rebuild sequential test set because src/1_data_preprocessing_csv.py is unavailable")

        print(f"  未找到连续测试集缓存，正在从 {split_dir} 构建按段连续测试集...")
        preprocessor = CSVDataPreprocessor(config_path=self.config_path)
        csv_files = sorted([
            os.path.join(split_dir, f) for f in os.listdir(split_dir)
            if f.endswith('.csv')
        ])
        if not csv_files:
            raise FileNotFoundError(f"连续测试集目录为空: {split_dir}")

        X_segments = []
        y_segments = []
        segments = []
        sample_cursor = 0

        for csv_path in tqdm(csv_files, desc='  构建连续测试集'):
            X_raw, y_raw = preprocessor.process_single_csv(csv_path)
            if X_raw is None or len(X_raw) == 0:
                continue

            n_samples, seq_len, n_features = X_raw.shape
            X_norm = self.estimator.scaler_X.transform(
                X_raw.reshape(-1, n_features)
            ).reshape(n_samples, seq_len, n_features).astype(np.float32)
            y_norm = self.estimator.scaler_y.transform(y_raw).astype(np.float32)

            X_segments.append(X_norm)
            y_segments.append(y_norm)
            segments.append([sample_cursor, sample_cursor + n_samples])
            sample_cursor += n_samples

        if not X_segments:
            raise ValueError('未能从原始CSV构建任何连续测试序列')

        X_test = np.vstack(X_segments).astype(np.float32)
        y_test = np.vstack(y_segments).astype(np.float32)
        segments = np.asarray(segments, dtype=np.int64)

        os.makedirs(data_dir, exist_ok=True)
        np.save(seq_X, X_test)
        np.save(seq_y, y_test)
        np.save(seq_s, segments)
        print(f"  ✓ 已生成连续测试集缓存: {X_test.shape}  ({len(segments)} 段时序)")
        print(f"    保存到: {data_dir}")
        return X_test, y_test, segments

    
    def predict(self, X_test, segments=None):
        """
        批量预测
        segments: [n_segs, 2] 每段的 [start, end)，若提供则按段 reset AKF
        """
        print("\n执行PIRNN-AKF估计...")
        if segments is not None:
            # 时序模式：按段 reset，段内连续
            N = len(X_test)
            wind_pred = np.zeros((N, 3))
            all_additional = {k: [] for k in [
                'wind_nn','wind_akf','wind_kin','q_scale','r_scale','confidence',
                'nn_weight','akf_weight','innovation','innovation_norm',
                'nis','P_diag','Q_diag','R_diag','maneuver_score','measurement_gap']}

            for seg_idx, (start, end) in enumerate(segments):
                for i in tqdm(range(start, end),
                              desc=f'  段{seg_idx+1}/{len(segments)}', leave=False):
                    result = self.estimator.estimate_sequence(
                        X_test[i],
                        reset_filter=(i == start)
                    )
                    wind_pred[i] = result['wind_estimate']
                    for k in all_additional:
                        all_additional[k].append(result.get(k, 0))

            additional = {k: np.array(v) for k, v in all_additional.items()}
        else:
            # 乱序模式（原有行为）
            wind_pred, additional = self.estimator.estimate_batch(X_test)

        print(f"  ✓ 估计完成: {len(wind_pred)} 样本")
        return wind_pred, additional
    
    def denormalize(self, wind_pred_norm, y_test_norm):
        """反归一化"""
        wind_pred = wind_pred_norm * self.estimator.wind_std + self.estimator.wind_mean
        y_test_denorm = self.estimator.scaler_y.inverse_transform(y_test_norm)
        wind_true = y_test_denorm[:, 0:3]
        return wind_pred, wind_true

    def _denormalize_additional_winds(self, additional):
        """将中间风场输出反归一化为 m/s，便于诊断绘图。"""
        wind_keys = ('wind_nn', 'wind_akf', 'wind_kin')
        outputs = {}
        for key in wind_keys:
            if key not in additional:
                continue
            arr = np.asarray(additional[key])
            if arr.ndim != 2 or arr.shape[1] < 3:
                continue
            outputs[key] = arr[:, :3] * self.estimator.wind_std + self.estimator.wind_mean
        return outputs

    def _build_plot_axis(self, n_samples, is_sequential=True):
        """根据数据模式构造横轴。"""
        step = max(1, n_samples // 4000)
        if is_sequential:
            dt = 1.0 / self.config['data']['sampling_rate']
            x = np.arange(n_samples) * dt
            x_label = 'Time (s)'
            title_suffix = 'Time Series'
        else:
            x = np.arange(n_samples)
            x_label = 'Sample Index'
            title_suffix = 'Sample-Order View (Non-sequential Test Set)'
        return x, step, x_label, title_suffix
    
    def calculate_metrics(self, wind_pred, wind_true):

        """计算评估指标"""
        print("\n计算评估指标...")
        
        metrics = {}
        
        metrics['rmse'] = np.sqrt(mean_squared_error(wind_true, wind_pred))
        metrics['mae'] = mean_absolute_error(wind_true, wind_pred)
        
        component_names = ['north', 'east', 'down']
        for i, name in enumerate(component_names):
            metrics[f'{name}_rmse'] = np.sqrt(mean_squared_error(wind_true[:, i], wind_pred[:, i]))
            metrics[f'{name}_mae'] = mean_absolute_error(wind_true[:, i], wind_pred[:, i])
            metrics[f'{name}_r2'] = r2_score(wind_true[:, i], wind_pred[:, i])
        
        wind_mag_true = np.linalg.norm(wind_true, axis=1)
        wind_mag_pred = np.linalg.norm(wind_pred, axis=1)
        
        metrics['magnitude_rmse'] = np.sqrt(mean_squared_error(wind_mag_true, wind_mag_pred))
        metrics['magnitude_mae'] = mean_absolute_error(wind_mag_true, wind_mag_pred)
        metrics['magnitude_r2'] = r2_score(wind_mag_true, wind_mag_pred)
        
        # 风向误差
        wind_dir_true = np.arctan2(wind_true[:, 1], wind_true[:, 0]) * 180 / np.pi
        wind_dir_pred = np.arctan2(wind_pred[:, 1], wind_pred[:, 0]) * 180 / np.pi
        dir_error = wind_dir_pred - wind_dir_true
        dir_error = (dir_error + 180) % 360 - 180
        metrics['direction_mae'] = np.mean(np.abs(dir_error))
        
        print(f"  ✓ 整体RMSE: {metrics['rmse']:.3f} m/s")
        print(f"  ✓ 整体MAE: {metrics['mae']:.3f} m/s")
        print(f"  ✓ 风向MAE: {metrics['direction_mae']:.2f}°")
        for name in component_names:
            print(f"  ✓ {name.capitalize():5s} RMSE: {metrics[f'{name}_rmse']:.3f} m/s  "
                  f"R²={metrics[f'{name}_r2']:.3f}")
        
        return metrics

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    COLORS = {'north': '#4472C4', 'east': '#ED7D31', 'down': '#A9D18E',
              'mag': '#5B9BD5', 'dir': '#7030A0'}

    def plot_results(self, wind_pred, wind_true, eval_dir):
        """3×3 学术散点图：各分量 + 风速大小 + 风向 + 误差直方图"""
        print("  生成散点评估图...")
        from scipy.stats import gaussian_kde as _kde

        wind_mag_t = np.linalg.norm(wind_true, axis=1)
        wind_mag_p = np.linalg.norm(wind_pred, axis=1)
        wind_dir_t = np.degrees(np.arctan2(wind_true[:, 1], wind_true[:, 0]))
        wind_dir_p = np.degrees(np.arctan2(wind_pred[:, 1], wind_pred[:, 0]))
        dir_err    = (wind_dir_p - wind_dir_t + 180) % 360 - 180

        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        fig.suptitle('PIRNN-AKF Wind Estimation Evaluation',
                     fontsize=14, fontweight='bold', y=0.98)
        comp_names  = ['North', 'East', 'Down']
        comp_colors = [self.COLORS['north'], self.COLORS['east'], self.COLORS['down']]

        # 行1: 各分量散点
        for i, (name, color) in enumerate(zip(comp_names, comp_colors)):
            ax = axes[0, i]
            ax.scatter(wind_true[:, i], wind_pred[:, i],
                       alpha=0.2, s=2, c=color, rasterized=True, label='Samples')
            lims = [min(wind_true[:, i].min(), wind_pred[:, i].min()),
                    max(wind_true[:, i].max(), wind_pred[:, i].max())]
            ax.plot(lims, lims, 'r--', lw=1.5, label='Ideal', zorder=10)
            rmse = np.sqrt(mean_squared_error(wind_true[:, i], wind_pred[:, i]))
            r2   = r2_score(wind_true[:, i], wind_pred[:, i])
            ax.set_title(f'{name} Wind\nRMSE={rmse:.3f} m/s, R²={r2:.3f}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel(f'True {name} (m/s)', fontsize=9)
            ax.set_ylabel(f'Predicted {name} (m/s)', fontsize=9)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # 行2左: 风速大小
        ax = axes[1, 0]
        ax.scatter(wind_mag_t, wind_mag_p, alpha=0.2, s=2,
                   c=self.COLORS['mag'], rasterized=True)
        lims = [wind_mag_t.min(), wind_mag_t.max()]
        ax.plot(lims, lims, 'r--', lw=1.5, label='Ideal', zorder=10)
        rmse_m = np.sqrt(mean_squared_error(wind_mag_t, wind_mag_p))
        r2_m   = r2_score(wind_mag_t, wind_mag_p)
        ax.set_title(f'Wind Magnitude\nRMSE={rmse_m:.3f} m/s, R²={r2_m:.3f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('True Magnitude (m/s)', fontsize=9)
        ax.set_ylabel('Predicted Magnitude (m/s)', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # 行2中: 风向
        ax = axes[1, 1]
        ax.scatter(wind_dir_t, wind_dir_p, alpha=0.2, s=2, c=self.COLORS['dir'], rasterized=True)
        ax.plot([-180, 180], [-180, 180], 'r--', lw=1.5, label='Ideal', zorder=10)
        ax.set_title(f'Horizontal Wind Direction\nMAE={np.mean(np.abs(dir_err)):.2f}°',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('True Direction (°)', fontsize=9)
        ax.set_ylabel('Predicted Direction (°)', fontsize=9)
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # 行2右: 水平误差时序
        ax = axes[1, 2]
        n   = len(wind_pred)
        idx = np.arange(0, n, max(1, n // 2000))
        ax.plot(idx, (wind_pred - wind_true)[idx, 0], lw=0.8,
                color=self.COLORS['north'], alpha=0.8, label='North error')
        ax.plot(idx, (wind_pred - wind_true)[idx, 1], lw=0.8,
                color=self.COLORS['east'],  alpha=0.8, label='East error')
        ax.axhline(0, color='black', lw=1.0, linestyle='--')
        ax.set_title('Horizontal Wind Error (Time Series)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=9)
        ax.set_ylabel('Error (m/s)', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # 行3: 误差直方图 + KDE
        for i, (name, color) in enumerate(zip(comp_names, comp_colors)):
            ax = axes[2, i]
            err = wind_pred[:, i] - wind_true[:, i]
            ax.hist(err, bins=60, alpha=0.65, color=color,
                    edgecolor='white', linewidth=0.3, density=True)
            x = np.linspace(err.min(), err.max(), 200)
            ax.plot(x, _kde(err)(x), color='navy', lw=1.5, label='KDE')
            ax.axvline(0,          color='red',   linestyle='--', lw=1.5, label='Zero')
            ax.axvline(err.mean(), color='black',  linestyle='-',  lw=1.5,
                       label=f'μ={err.mean():.3f}')
            ax.set_title(f'{name} Error Distribution\nμ={err.mean():.3f}, σ={err.std():.3f} m/s',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Error (m/s)', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        path = os.path.join(eval_dir, 'evaluation_results.svg')
        fig.savefig(path, format='svg', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  ✓ 散点评估图已保存: {path}")

    def plot_metrics_bar(self, metrics, eval_dir):
        """各分量 RMSE/MAE 柱状图"""
        print("  生成指标柱状图...")
        keys  = ['north', 'east', 'down', 'magnitude']
        names = ['North', 'East', 'Down', 'Magnitude']
        rmse_vals = [metrics[f'{k}_rmse'] for k in keys]
        mae_vals  = [metrics[f'{k}_mae']  for k in keys]
        x = np.arange(len(names)); w = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        b1 = ax.bar(x - w/2, rmse_vals, w, label='RMSE', color='#4472C4', alpha=0.85)
        b2 = ax.bar(x + w/2, mae_vals,  w, label='MAE',  color='#ED7D31', alpha=0.85)
        ax.set_title('PIRNN-AKF — Per-Component Error Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error (m/s)', fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        path = os.path.join(eval_dir, 'metrics_bar.svg')
        fig.savefig(path, format='svg', bbox_inches='tight')
        plt.close()
        print(f"  ✓ 指标柱状图已保存: {path}")

    def plot_waveform(self, wind_pred, wind_true, eval_dir, is_sequential=True):
        """N/E/D + 风速大小 波形对比图"""
        print("  生成波形对比图...")
        n_samples = len(wind_pred)
        x, step, x_label, title_suffix = self._build_plot_axis(n_samples, is_sequential)
        mag_t = np.linalg.norm(wind_true, axis=1)
        mag_p = np.linalg.norm(wind_pred, axis=1)

        comp_cfg = [
            ('North',     wind_true[:, 0], wind_pred[:, 0], self.COLORS['north']),
            ('East',      wind_true[:, 1], wind_pred[:, 1], self.COLORS['east']),
            ('Down',      wind_true[:, 2], wind_pred[:, 2], self.COLORS['down']),
            ('Magnitude', mag_t,           mag_p,           self.COLORS['mag']),
        ]

        fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)
        fig.suptitle(f'PIRNN-AKF — Estimated vs True Wind ({title_suffix})',
                     fontsize=13, fontweight='bold')

        for ax, (label, true, pred, color) in zip(axes, comp_cfg):
            ax.plot(x[::step], true[::step], color='#555555', lw=0.7, alpha=0.8, label='True')
            ax.plot(x[::step], pred[::step], color=color,     lw=0.9, alpha=0.9, label='Estimated')
            rmse = np.sqrt(np.mean((pred - true) ** 2))
            ax.set_ylabel(f'{label}\n(m/s)', fontsize=9)
            ax.set_title(f'{label}  RMSE={rmse:.3f} m/s', fontsize=9, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right', ncol=2)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[-1].set_xlabel(x_label, fontsize=10)
        if not is_sequential:
            fig.text(0.5, 0.01,
                     'Note: current test set is shuffled; the x-axis reflects sample order only.',
                     ha='center', fontsize=9, color='#A61C00')
        plt.tight_layout(rect=[0, 0.02, 1, 1])
        path = os.path.join(eval_dir, 'wind_waveform.svg')
        fig.savefig(path, format='svg', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  ✓ 波形图已保存: {path}")

    def plot_fusion_diagnostics(self, wind_pred, wind_true, additional, eval_dir, is_sequential=True):
        """绘制 wind_true / wind_nn / wind_kin / wind_akf / wind_fused 五路诊断图。"""
        print("  生成五路融合诊断图...")
        diagnostic_winds = self._denormalize_additional_winds(additional)
        required_keys = ['wind_nn', 'wind_kin', 'wind_akf']
        missing = [key for key in required_keys if key not in diagnostic_winds]
        if missing:
            print(f"  ⚠️ 缺少中间量 {missing}，跳过五路融合诊断图")
            return

        n_samples = len(wind_pred)
        x, step, x_label, title_suffix = self._build_plot_axis(n_samples, is_sequential)
        component_info = [
            ('North', 0, self.COLORS['north']),
            ('East', 1, self.COLORS['east']),
            ('Down', 2, self.COLORS['down']),
        ]
        curves = [
            ('True', wind_true, '#404040', 1.0, '-'),
            ('PI-GRU', diagnostic_winds['wind_nn'], '#5B9BD5', 0.9, '--'),
            ('Wind-kin', diagnostic_winds['wind_kin'], '#ED7D31', 0.9, '-'),
            ('AKF', diagnostic_winds['wind_akf'], '#70AD47', 0.9, '-'),
            ('Fused', wind_pred, '#C00000', 1.2, '-'),
        ]

        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        fig.suptitle(f'PIRNN-AKF Fusion Diagnostics ({title_suffix})',
                     fontsize=13, fontweight='bold')

        for ax, (name, idx, _) in zip(axes, component_info):
            for curve_name, values, color, lw, ls in curves:
                ax.plot(x[::step], values[::step, idx], color=color, lw=lw,
                        linestyle=ls, alpha=0.92 if curve_name == 'Fused' else 0.85,
                        label=curve_name)

            rmse_nn = np.sqrt(np.mean((diagnostic_winds['wind_nn'][:, idx] - wind_true[:, idx]) ** 2))
            rmse_kin = np.sqrt(np.mean((diagnostic_winds['wind_kin'][:, idx] - wind_true[:, idx]) ** 2))
            rmse_akf = np.sqrt(np.mean((diagnostic_winds['wind_akf'][:, idx] - wind_true[:, idx]) ** 2))
            rmse_fused = np.sqrt(np.mean((wind_pred[:, idx] - wind_true[:, idx]) ** 2))

            ax.set_ylabel(f'{name}\n(m/s)', fontsize=9)
            ax.set_title(
                f'{name}  RMSE: fused={rmse_fused:.3f}, nn={rmse_nn:.3f}, '
                f'kin={rmse_kin:.3f}, akf={rmse_akf:.3f}',
                fontsize=9,
                fontweight='bold'
            )
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[0].legend(fontsize=8, loc='upper right', ncol=5)
        axes[-1].set_xlabel(x_label, fontsize=10)
        if not is_sequential:
            fig.text(0.5, 0.01,
                     'Note: current test set is shuffled; the x-axis reflects sample order only.',
                     ha='center', fontsize=9, color='#A61C00')
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        path = os.path.join(eval_dir, 'wind_fusion_diagnostics.svg')
        fig.savefig(path, format='svg', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  ✓ 五路融合诊断图已保存: {path}")

    def plot_wind_magnitude_comparison(self, wind_pred, wind_true, additional, eval_dir, is_sequential=True):
        """绘制各路风场合成风大小对比图。"""
        print("  生成合成风大小对比图...")
        diagnostic_winds = self._denormalize_additional_winds(additional)

        n_samples = len(wind_pred)
        x, step, x_label, title_suffix = self._build_plot_axis(n_samples, is_sequential)
        mag_true = np.linalg.norm(wind_true, axis=1)
        mag_fused = np.linalg.norm(wind_pred, axis=1)

        curves = [
            ('True', mag_true, '#404040', 1.0, '-'),
        ]
        if 'wind_nn' in diagnostic_winds:
            curves.append(('PI-GRU', np.linalg.norm(diagnostic_winds['wind_nn'], axis=1), '#5B9BD5', 0.9, '--'))
        if 'wind_kin' in diagnostic_winds:
            curves.append(('Wind-kin', np.linalg.norm(diagnostic_winds['wind_kin'], axis=1), '#ED7D31', 0.9, '-'))
        if 'wind_akf' in diagnostic_winds:
            curves.append(('AKF', np.linalg.norm(diagnostic_winds['wind_akf'], axis=1), '#70AD47', 0.9, '-'))
        curves.append(('Fused', mag_fused, '#C00000', 1.2, '-'))

        fig, ax = plt.subplots(1, 1, figsize=(16, 4.8))
        fig.suptitle(f'PIRNN-AKF Wind Magnitude Comparison ({title_suffix})',
                     fontsize=13, fontweight='bold')

        for curve_name, values, color, lw, ls in curves:
            rmse = np.sqrt(np.mean((values - mag_true) ** 2)) if curve_name != 'True' else 0.0
            label = curve_name if curve_name == 'True' else f'{curve_name} (RMSE={rmse:.3f})'
            ax.plot(x[::step], values[::step], color=color, lw=lw, linestyle=ls,
                    alpha=0.92 if curve_name == 'Fused' else 0.85, label=label)

        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel('Wind Magnitude (m/s)', fontsize=10)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(fontsize=8, loc='upper right', ncol=min(5, len(curves)))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if not is_sequential:
            fig.text(0.5, 0.01,
                     'Note: current test set is shuffled; the x-axis reflects sample order only.',
                     ha='center', fontsize=9, color='#A61C00')
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        else:
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])

        path = os.path.join(eval_dir, 'wind_magnitude_comparison.svg')
        fig.savefig(path, format='svg', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  ✓ 合成风大小对比图已保存: {path}")

    def run(self, X_test, y_test, eval_dir=None, segments=None):

        """执行完整评估流程
        
        Args:
            X_test: 测试输入
            y_test: 测试标签
            eval_dir: 指定输出目录；若为 None 则自动创建
            segments: [n_segs, 2] 时序分段信息，若提供则按段 reset AKF
        """
        print("\n" + "="*70)
        print("  PIRNN-AKF 评估 (Physics-Informed Neural-Augmented AKF)")
        print("="*70)
        
        # 1. 预测
        wind_pred_norm, additional = self.predict(X_test, segments=segments)
        
        # 2. 反归一化
        wind_pred, wind_true = self.denormalize(wind_pred_norm, y_test)
        
        # 3. 计算指标
        metrics = self.calculate_metrics(wind_pred, wind_true)
        
        # 4. 保存结果 + 绘图
        if eval_dir is None:
            eval_dir = self.get_evaluation_dir()
        os.makedirs(eval_dir, exist_ok=True)
        
        results = {
            'metrics': metrics,
            'model_type': 'PIRNN-AKF',
            'data_mode': 'test_id_sequential' if segments is not None else 'id_shuffled',

            'wind_pred': wind_pred,
            'wind_true': wind_true,
            'additional_outputs': additional,
            'config': self.config
        }


        
        with open(os.path.join(eval_dir, 'evaluation_metrics.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        print("\n生成可视化图表...")
        is_sequential = (segments is not None)
        self.plot_results(wind_pred, wind_true, eval_dir)
        self.plot_metrics_bar(metrics, eval_dir)
        self.plot_waveform(wind_pred, wind_true, eval_dir, is_sequential=is_sequential)
        self.plot_fusion_diagnostics(wind_pred, wind_true, additional, eval_dir,
                                     is_sequential=is_sequential)
        self.plot_wind_magnitude_comparison(wind_pred, wind_true, additional, eval_dir,
                                            is_sequential=is_sequential)


        print(f"\n  ✓ 结果已保存: {eval_dir}")

        print("\n" + "="*70)
        print("  ✅ PIRNN-AKF 评估完成！")
        print("="*70)
        
        return metrics, wind_pred, wind_true, additional


if __name__ == "__main__":
    import torch as _torch

    parser = argparse.ArgumentParser(description='PIRNN-AKF 评估（支持按lambda筛选）')
    parser.add_argument('--lambda-physics', type=float, default=None,
                        help='仅评估指定 lambda_physics（例如 0.2）；不传则评估全部 train_*')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='仅评估指定训练目录名（例如 train_lambda0.2_20260419_030924）')
    parser.add_argument('--config_path', type=str, default=None,
                        help='配置文件路径，默认 config/config.yaml')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='评估结果根目录，默认 data/evaluation/eval_pigru_akf_<timestamp>')
    args = parser.parse_args()

    print("="*70)
    print(" PIRNN-AKF 多 lambda_physics 批量评估")
    if args.lambda_physics is not None:
        print(f"  (已启用筛选: lambda_physics={args.lambda_physics:g})")
    if args.model_dir:
        print(f"  (已启用筛选: model_dir={args.model_dir})")
    print("="*70)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = resolve_project_path(project_root, args.config_path) if args.config_path else os.path.join(project_root, 'config', 'config.yaml')

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 加载测试数据（优先使用时序化测试集）
        print("\n加载测试数据...")
        data_dir = config['data']['processed_dir']
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(project_root, data_dir.lstrip('../'))

        seq_X = os.path.join(data_dir, 'X_test_sequential.npy')
        seq_y = os.path.join(data_dir, 'y_test_sequential.npy')
        seq_s = os.path.join(data_dir, 'seg_test_sequential.npy')

        if os.path.exists(seq_X) and os.path.exists(seq_s):
            X_test   = np.load(seq_X)
            y_test   = np.load(seq_y)
            segments = np.load(seq_s)   # [n_segs, 2]
            print(f"  ✓ X_test_sequential: {X_test.shape}  ({len(segments)} 段时序)")
        else:
            X_test   = np.load(os.path.join(data_dir, 'X_test_id.npy'))
            y_test   = np.load(os.path.join(data_dir, 'y_test_id.npy'))
            segments = None
            print(f"  ✓ X_test_id (乱序): {X_test.shape}")
            print("  ⚠️ 未找到 sequential 测试集，波形图仅反映样本顺序，不代表真实时间连续轨迹")


        # 扫描所有 train_* 目录
        model_save_path = config['training']['model_save_path']
        if not os.path.isabs(model_save_path):
            model_save_path = os.path.join(project_root, model_save_path.lstrip('../'))

        train_dirs = sorted([
            d for d in os.listdir(model_save_path)
            if d.startswith('train_') and
               os.path.isdir(os.path.join(model_save_path, d)) and
               os.path.exists(os.path.join(model_save_path, d, 'best_model.pth'))
        ])

        # 可选：按目录名筛选
        if args.model_dir:
            train_dirs = [d for d in train_dirs if d == args.model_dir]

        # 可选：按 lambda 筛选（优先从目录名解析，兼容旧 ckpt）
        if args.lambda_physics is not None:
            target = float(args.lambda_physics)
            selected = []
            for d in train_dirs:
                m = re.search(r'train_lambda([0-9.]+)', d)
                if m and abs(float(m.group(1)) - target) < 1e-9:
                    selected.append(d)
            train_dirs = selected

        if not train_dirs:
            raise FileNotFoundError(
                f"筛选后未找到可评估目录。model_save_path={model_save_path}, "
                f"lambda={args.lambda_physics}, model_dir={args.model_dir}"
            )

        print(f"\n找到 {len(train_dirs)} 个训练目录:")
        for d in train_dirs:
            print(f"  {d}")

        # 创建本次批量评估的顶层目录（所有 lambda 共用）
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.output_dir:
            top_eval_dir = resolve_project_path(project_root, args.output_dir)
        else:
            top_eval_dir = os.path.join(project_root, 'data', 'evaluation',
                                        f'eval_pigru_akf_{run_timestamp}')
        os.makedirs(top_eval_dir, exist_ok=True)
        print(f"\n评估结果根目录: {top_eval_dir}")

        # 汇总结果
        summary = []   # list of (label, lambda_val, metrics)

        for train_dir_name in train_dirs:
            model_dir = os.path.join(model_save_path, train_dir_name)
            model_path = os.path.join(model_dir, 'best_model.pth')

            # 读取 lambda：优先目录名（兼容旧 checkpoint 记录错误）
            ckpt = _torch.load(model_path, map_location='cpu', weights_only=False)
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

            print(f"\n{'='*70}")
            print(f"  评估: {train_dir_name}  (lambda_physics={lam_display})")
            print(f"{'='*70}")

            evaluator = PIRNNAKFEvaluator(config_path=config_path,
                                          model_path=model_path)

            # 避免同一lambda多次实验覆盖
            eval_dir = os.path.join(top_eval_dir, f"{label}__{train_dir_name}")

            metrics, wind_pred, wind_true, additional = evaluator.run(
                X_test, y_test, eval_dir=eval_dir, segments=segments)

            summary.append((label, lam if lam is not None else -1, metrics))
            print(f"  ✓ 结果已保存到子目录: {os.path.basename(eval_dir)}")

        # ── 汇总对比图 ────────────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print("  生成汇总对比图...")

        summary.sort(key=lambda x: x[1])
        labels     = [s[0] for s in summary]
        rmse_vals  = [s[2]['rmse']           for s in summary]
        mae_vals   = [s[2]['mae']            for s in summary]
        rmse_n     = [s[2]['north_rmse']     for s in summary]
        rmse_e     = [s[2]['east_rmse']      for s in summary]
        rmse_d     = [s[2]['down_rmse']      for s in summary]
        rmse_mag   = [s[2]['magnitude_rmse'] for s in summary]

        x = np.arange(len(labels))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PIRNN-AKF: Effect of λ_physics on Wind Estimation Performance',
                     fontsize=13, fontweight='bold')

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
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{bar.get_height():.3f}', ha='center', va='bottom',
                        fontsize=7, color=color_bar,
                        fontweight='bold' if i == best_idx else 'normal')
            ax.get_children()[best_idx].set_edgecolor('red')
            ax.get_children()[best_idx].set_linewidth(2)

        _bar(axes[0, 0], rmse_vals, 'Overall RMSE',   color='#4472C4')
        _bar(axes[0, 1], mae_vals,  'Overall MAE',    color='#ED7D31')
        _bar(axes[1, 0], rmse_mag,  'Magnitude RMSE', color='#5B9BD5')

        # 各分量 RMSE 折线图
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
        summary_path = os.path.join(top_eval_dir, 'lambda_comparison.svg')
        fig.savefig(summary_path, format='svg', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  ✓ 汇总对比图已保存: {summary_path}")

        # 打印排名
        print(f"\n{'='*70}")
        print("  各模型 RMSE 排名（升序）")
        print(f"{'='*70}")
        ranked = sorted(summary, key=lambda x: x[2]['rmse'])
        for rank, (lbl, lam, m) in enumerate(ranked, 1):
            print(f"  #{rank}  {lbl:20s}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")
        print(f"\n  最佳模型: {ranked[0][0]}  (RMSE={ranked[0][2]['rmse']:.4f} m/s)")

    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
