"""
系统级证据论文图绘制脚本（顶刊风格版）

输入目录需包含：
- smoothness_metrics.csv
- anomaly_robustness.csv
- system_diagnostics_clean.npz

输出图（pdf/svg/png）：
1) figure_system_smoothness
2) figure_system_anomaly_robustness
3) figure_system_diagnostics
4) figure_system_pareto
"""

import argparse
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _apply_publication_style() -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8.5,
        'axes.titlesize': 9,
        'axes.labelsize': 8.5,
        'axes.linewidth': 0.7,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7.8,
        'legend.frameon': False,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.3,
        'lines.markersize': 3.5,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'figure.dpi': 180,
        'axes.unicode_minus': False
    })


def _save_figure(fig: plt.Figure, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'svg'):
        fig.savefig(os.path.join(out_dir, f'{stem}.{ext}'), format=ext, facecolor='white')
    fig.savefig(os.path.join(out_dir, f'{stem}.png'), format='png', dpi=600, facecolor='white')


def _load_inputs(evidence_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    smooth_csv = os.path.join(evidence_dir, 'smoothness_metrics.csv')
    anomaly_csv = os.path.join(evidence_dir, 'anomaly_robustness.csv')
    diag_npz = os.path.join(evidence_dir, 'system_diagnostics_clean.npz')

    if not os.path.exists(smooth_csv):
        raise FileNotFoundError(f'缺少文件: {smooth_csv}')
    if not os.path.exists(anomaly_csv):
        raise FileNotFoundError(f'缺少文件: {anomaly_csv}')
    if not os.path.exists(diag_npz):
        raise FileNotFoundError(f'缺少文件: {diag_npz}')

    smooth_df = pd.read_csv(smooth_csv)
    anomaly_df = pd.read_csv(anomaly_csv)
    diag_data = dict(np.load(diag_npz, allow_pickle=False))
    return smooth_df, anomaly_df, diag_data


def _chunk_metrics(w_true: np.ndarray, w_pred: np.ndarray, chunk: int) -> Dict[str, np.ndarray]:
    n = len(w_true)
    m = n // chunk
    if m < 2:
        raise ValueError('样本过少，无法进行分块统计')

    rmse, tv, jitter = [], [], []
    for i in range(m):
        s = i * chunk
        e = (i + 1) * chunk
        yt = w_true[s:e]
        yp = w_pred[s:e]

        diff = yp - yt
        rmse.append(float(np.sqrt(np.mean(np.sum(diff * diff, axis=1)))))

        d1 = np.diff(yp, axis=0)
        d2 = np.diff(d1, axis=0)
        tv.append(float(np.mean(np.linalg.norm(d1, axis=1))))
        jitter.append(float(np.mean(np.linalg.norm(d2, axis=1))))

    return {
        'rmse': np.asarray(rmse),
        'tv_mean': np.asarray(tv),
        'jitter_mean': np.asarray(jitter)
    }


def _minimal_axes(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.22)


def _plot_smoothness(diag: Dict[str, np.ndarray], out_dir: str, chunk_size: int) -> None:
    if not {'wind_true', 'wind_bare', 'wind_system'}.issubset(set(diag.keys())):
        raise ValueError('system_diagnostics_clean.npz 缺少 wind_true/wind_bare/wind_system')

    w_true = np.asarray(diag['wind_true'])
    w_bare = np.asarray(diag['wind_bare'])
    w_sys = np.asarray(diag['wind_system'])

    bare = _chunk_metrics(w_true, w_bare, chunk=chunk_size)
    sys = _chunk_metrics(w_true, w_sys, chunk=chunk_size)

    metrics = ['rmse', 'tv_mean', 'jitter_mean']
    titles = ['(a) RMSE distribution', '(b) TV distribution', '(c) Jitter distribution']
    ylabels = ['RMSE (m/s)', 'TV mean', 'Jitter mean']
    colors = ['#4C78A8', '#F58518']

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), constrained_layout=True)

    for i, m in enumerate(metrics):
        ax = axes[i]
        parts = ax.boxplot(
            [bare[m], sys[m]],
            labels=['Bare', 'System'],
            patch_artist=True,
            widths=0.55,
            medianprops={'color': '#222222', 'linewidth': 1.0},
            boxprops={'linewidth': 0.8},
            whiskerprops={'linewidth': 0.8},
            capprops={'linewidth': 0.8},
            showfliers=False
        )
        for patch, c in zip(parts['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)

        gain = (np.median(bare[m]) - np.median(sys[m])) / np.median(bare[m]) * 100.0
        ax.text(0.03, 0.94, f'Δmedian={gain:+.2f}%', transform=ax.transAxes, fontsize=7.5, va='top')
        ax.set_title(titles[i])
        ax.set_ylabel(ylabels[i])
        _minimal_axes(ax)

    fig.suptitle('ID performance from chunk-level distributions (lower is better)', y=1.03)
    _save_figure(fig, out_dir, 'figure_system_smoothness')
    plt.close(fig)


def _plot_anomaly_robustness(anomaly_df: pd.DataFrame, out_dir: str) -> None:
    needed = {'anomaly_type', 'method', 'peak_error', 'window_error_mean', 'rmse'}
    if not needed.issubset(set(anomaly_df.columns)):
        raise ValueError('anomaly_robustness.csv 缺少必要列')

    order = ['gps_spike', 'tas_spike', 'attitude_spike', 'sensor_dropout', 'gaussian_burst']
    label_map = {'gps_spike': 'GPS', 'tas_spike': 'TAS', 'attitude_spike': 'Att', 'sensor_dropout': 'Drop', 'gaussian_burst': 'Burst'}

    bare = anomaly_df[anomaly_df['method'] == 'bare_network'].set_index('anomaly_type')
    sys = anomaly_df[anomaly_df['method'] == 'system_level'].set_index('anomaly_type')
    present = [a for a in order if a in bare.index and a in sys.index]
    if not present:
        raise ValueError('anomaly_robustness.csv 中无可对齐的两方法结果')

    mat = np.zeros((3, len(present)), dtype=float)
    for j, a in enumerate(present):
        mat[0, j] = (float(bare.loc[a, 'peak_error']) - float(sys.loc[a, 'peak_error'])) * 100.0
        mat[1, j] = (float(bare.loc[a, 'window_error_mean']) - float(sys.loc[a, 'window_error_mean'])) * 100.0
        mat[2, j] = (float(bare.loc[a, 'rmse']) - float(sys.loc[a, 'rmse'])) * 100.0

    fig, ax = plt.subplots(figsize=(6.9, 2.9), constrained_layout=True)
    vmax = max(0.1, float(np.max(np.abs(mat))))
    im = ax.imshow(mat, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(present)))
    ax.set_xticklabels([label_map[a] for a in present])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Peak error gain', 'Window error gain', 'RMSE gain'])
    ax.set_title('Anomaly robustness gain map (Bare - System, unit: cm/s)')

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.3, color='black')

    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label('Gain (cm/s)')

    _save_figure(fig, out_dir, 'figure_system_anomaly_robustness')
    plt.close(fig)


def _window_around_peak(x: np.ndarray, center: int, half: int) -> Tuple[np.ndarray, int, int]:
    n = len(x)
    s = max(0, center - half)
    e = min(n, center + half)
    return x[s:e], s, e


def _plot_diagnostics(diag: Dict[str, np.ndarray], out_dir: str, fs: float, half_window: int) -> None:
    required = ['innovation_norm', 'nis', 'Q_diag', 'R_diag', 'confidence', 'nn_weight', 'akf_weight']
    for k in required:
        if k not in diag:
            raise ValueError(f'system_diagnostics_clean.npz 缺少键: {k}')

    nis = np.asarray(diag['nis'])
    peak_idx = int(np.nanargmax(nis))

    innovation_w, s, e = _window_around_peak(np.asarray(diag['innovation_norm']), peak_idx, half_window)
    nis_w, _, _ = _window_around_peak(np.asarray(diag['nis']), peak_idx, half_window)
    q_w, _, _ = _window_around_peak(np.asarray(diag['Q_diag'])[:, 1], peak_idx, half_window)
    r_w, _, _ = _window_around_peak(np.asarray(diag['R_diag'])[:, 2], peak_idx, half_window)
    conf_w, _, _ = _window_around_peak(np.asarray(diag['confidence']), peak_idx, half_window)
    nn_w, _, _ = _window_around_peak(np.asarray(diag['nn_weight']), peak_idx, half_window)
    akf_w, _, _ = _window_around_peak(np.asarray(diag['akf_weight']), peak_idx, half_window)

    t = np.arange(s, e) / fs
    q_ratio = q_w / np.median(np.asarray(diag['Q_diag'])[:, 1])
    r_ratio = r_w / np.median(np.asarray(diag['R_diag'])[:, 2])

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True, constrained_layout=True)

    axes[0].plot(t, nis_w, color='#E45756', label='NIS')
    axes[0].axhline(7.815, color='k', ls='--', lw=0.9, label='χ²(3,0.95)')
    axes[0].set_ylabel('NIS')
    axes[0].set_title('(a) Innovation consistency around strongest event')
    axes[0].legend(loc='upper right')
    _minimal_axes(axes[0])

    axes[1].plot(t, innovation_w, color='#4C78A8', label='||innovation||')
    axes[1].plot(t, q_ratio, color='#F58518', label='Q_E / median(Q_E)')
    axes[1].plot(t, r_ratio, color='#72B7B2', label='R_D / median(R_D)')
    axes[1].set_ylabel('Relative scale')
    axes[1].set_title('(b) Innovation and adaptive covariance response')
    axes[1].legend(loc='upper right', ncol=2)
    _minimal_axes(axes[1])

    axes[2].plot(t, conf_w, color='#54A24B', label='Confidence')
    axes[2].plot(t, nn_w, color='#B279A2', label='NN weight')
    axes[2].plot(t, akf_w, color='#FF9DA6', label='AKF weight')
    axes[2].set_ylabel('Value')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_title('(c) Confidence-driven fusion adaptation')
    axes[2].legend(loc='upper right', ncol=3)
    _minimal_axes(axes[2])

    fig.suptitle('System diagnostics (event-centered view)', y=1.02)
    _save_figure(fig, out_dir, 'figure_system_diagnostics')
    plt.close(fig)


def _plot_pareto(diag: Dict[str, np.ndarray], out_dir: str, chunk_size: int) -> None:
    if not {'wind_true', 'wind_bare', 'wind_system'}.issubset(set(diag.keys())):
        raise ValueError('system_diagnostics_clean.npz 缺少 wind_true/wind_bare/wind_system')

    w_true = np.asarray(diag['wind_true'])
    w_bare = np.asarray(diag['wind_bare'])
    w_sys = np.asarray(diag['wind_system'])

    bare = _chunk_metrics(w_true, w_bare, chunk=chunk_size)
    sys = _chunk_metrics(w_true, w_sys, chunk=chunk_size)

    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)

    ax.scatter(bare['rmse'], bare['jitter_mean'], s=16, alpha=0.25, color='#4C78A8', label='Bare chunks')
    ax.scatter(sys['rmse'], sys['jitter_mean'], s=16, alpha=0.25, color='#F58518', label='System chunks')

    mu_b = (float(np.mean(bare['rmse'])), float(np.mean(bare['jitter_mean'])))
    mu_s = (float(np.mean(sys['rmse'])), float(np.mean(sys['jitter_mean'])))
    ax.scatter([mu_b[0]], [mu_b[1]], s=80, color='#1F4E79', marker='o', label='Bare mean')
    ax.scatter([mu_s[0]], [mu_s[1]], s=90, color='#B85C1E', marker='D', label='System mean')
    ax.annotate('', xy=mu_s, xytext=mu_b, arrowprops=dict(arrowstyle='->', lw=1.1, color='#333333'))

    gain_rmse = (mu_b[0] - mu_s[0]) / mu_b[0] * 100.0
    gain_jitter = (mu_b[1] - mu_s[1]) / mu_b[1] * 100.0
    ax.text(0.03, 0.97, f'ΔRMSE={gain_rmse:+.2f}%, ΔJitter={gain_jitter:+.2f}%',
            transform=ax.transAxes, va='top', fontsize=8)

    ax.set_xlabel('RMSE (m/s)')
    ax.set_ylabel('Jitter mean')
    ax.set_title('Chunk-level Pareto cloud with mean shift')
    ax.grid(alpha=0.22)
    ax.legend(loc='upper right')

    _save_figure(fig, out_dir, 'figure_system_pareto')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='绘制系统级证据论文图（顶刊风格版）')
    parser.add_argument('--evidence-dir', type=str, required=True, help='证据目录（含 CSV/NPZ）')
    parser.add_argument('--output-dir', type=str, default='', help='图像输出目录，默认 evidence_dir/figures')
    parser.add_argument('--sample-rate', type=float, default=20.0, help='采样率 Hz')
    parser.add_argument('--diag-half-window', type=int, default=3000, help='诊断窗口半宽（样本点）')
    parser.add_argument('--chunk-size', type=int, default=2000, help='分块统计窗口长度')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence_dir = os.path.abspath(args.evidence_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(evidence_dir, 'figures')

    _apply_publication_style()
    smooth_df, anomaly_df, diag = _load_inputs(evidence_dir)

    _ = smooth_df
    _plot_smoothness(diag, output_dir, chunk_size=args.chunk_size)
    _plot_anomaly_robustness(anomaly_df, output_dir)
    _plot_diagnostics(diag, output_dir, fs=args.sample_rate, half_window=args.diag_half_window)
    _plot_pareto(diag, output_dir, chunk_size=args.chunk_size)

    print('=' * 70)
    print('系统证据论文图已生成（顶刊风格版）')
    print(f'输入目录: {evidence_dir}')
    print(f'输出目录: {output_dir}')
    print('输出文件:')
    print('  - figure_system_smoothness.(pdf/svg/png)')
    print('  - figure_system_anomaly_robustness.(pdf/svg/png)')
    print('  - figure_system_diagnostics.(pdf/svg/png)')
    print('  - figure_system_pareto.(pdf/svg/png)')
    print('=' * 70)


if __name__ == '__main__':
    main()
