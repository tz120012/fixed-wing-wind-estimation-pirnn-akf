#!/usr/bin/env python3
"""
测试集风场数据可视化工具

从 data/data_csv_processed/y_test_id.npy 加载测试集真值风场数据，
生成多维度专业图表，用于论文数据集描述章节。

运行方式:
    cd wind-estimation-main_260322
    python utils/plot_test_wind_field.py
    python utils/plot_test_wind_field.py --ood          # 绘制 OOD 测试集
    python utils/plot_test_wind_field.py --out my_dir   # 指定输出目录
"""

import argparse
import os
import sys
import pickle
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import yaml
from scipy.stats import gaussian_kde

# ── 路径设置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


def load_data(config: dict, use_ood: bool = False):
    """加载测试集并反归一化，返回真实物理单位的风速数组"""
    data_dir = config['data']['processed_dir']
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, data_dir.lstrip('../'))

    model_dir = config['training']['model_save_path']
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(PROJECT_ROOT, model_dir.lstrip('../'))

    # 选择数据集
    if use_ood:
        y_path = os.path.join(data_dir, 'y_test_ood.npy')
        tag = 'OOD'
    else:
        y_path = os.path.join(data_dir, 'y_test_id.npy')
        tag = 'ID'

    if not os.path.exists(y_path):
        raise FileNotFoundError(f"测试集文件不存在: {y_path}")

    y_norm = np.load(y_path)          # [N, 7]  归一化

    # 加载归一化参数
    norm_path = os.path.join(model_dir, 'norm_params.pkl')
    with open(norm_path, 'rb') as f:
        meta = pickle.load(f)
    scaler_y = meta['scaler_y']

    y = scaler_y.inverse_transform(y_norm)   # 反归一化 → 真实单位

    wind = y[:, 0:3]   # [N, 3]  wind_N, wind_E, wind_D  (m/s)
    print(f"  ✓ 加载 {tag} 测试集: {len(wind):,} 个样本")
    print(f"    wind_N: {wind[:,0].mean():.3f} ± {wind[:,0].std():.3f} m/s  "
          f"[{wind[:,0].min():.2f}, {wind[:,0].max():.2f}]")
    print(f"    wind_E: {wind[:,1].mean():.3f} ± {wind[:,1].std():.3f} m/s  "
          f"[{wind[:,1].min():.2f}, {wind[:,1].max():.2f}]")
    print(f"    wind_D: {wind[:,2].mean():.3f} ± {wind[:,2].std():.3f} m/s  "
          f"[{wind[:,2].min():.2f}, {wind[:,2].max():.2f}]")
    wind_mag = np.linalg.norm(wind, axis=1)
    print(f"    |wind|: {wind_mag.mean():.3f} ± {wind_mag.std():.3f} m/s  "
          f"[{wind_mag.min():.2f}, {wind_mag.max():.2f}]")
    return wind, tag


# ── 绘图函数 ──────────────────────────────────────────────────────────────────

COLORS = {
    'north': '#4472C4',
    'east':  '#ED7D31',
    'down':  '#A9D18E',
    'mag':   '#5B9BD5',
    'dir':   '#7030A0',
}


def plot_overview(wind: np.ndarray, tag: str, out_dir: str):
    """
    主图：3×3 布局
      行1: North / East / Down 分量 KDE 分布
      行2: 风速大小分布 / 水平风向玫瑰图 / N-E 散点图
      行3: 各分量箱线图 / 风速大小 CDF / 统计摘要文字
    """
    wind_mag = np.linalg.norm(wind, axis=1)
    wind_dir = np.degrees(np.arctan2(wind[:, 1], wind[:, 0]))  # [-180, 180]

    fig = plt.figure(figsize=(15, 13))
    fig.suptitle(f'Test Wind Field Distribution  ({tag}, N={len(wind):,})',
                 fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    comp_labels = ['North', 'East', 'Down']
    comp_colors = [COLORS['north'], COLORS['east'], COLORS['down']]

    # ── 行1: 各分量 KDE ──────────────────────────────────────────────────────
    for i, (label, color) in enumerate(zip(comp_labels, comp_colors)):
        ax = fig.add_subplot(gs[0, i])
        data = wind[:, i]
        x = np.linspace(data.min() - 0.5, data.max() + 0.5, 300)
        kde = gaussian_kde(data, bw_method='scott')
        ax.fill_between(x, kde(x), alpha=0.3, color=color)
        ax.plot(x, kde(x), color=color, lw=2)
        ax.axvline(data.mean(), color='red', lw=1.5, linestyle='--',
                   label=f'μ={data.mean():.2f}')
        ax.axvline(0, color='black', lw=1.0, linestyle=':', alpha=0.5)
        ax.set_title(f'{label} Wind Component', fontsize=10, fontweight='bold')
        ax.set_xlabel('Wind Speed (m/s)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # 标注统计
        ax.text(0.97, 0.95,
                f'σ={data.std():.2f}\n[{data.min():.1f}, {data.max():.1f}]',
                transform=ax.transAxes, fontsize=7.5, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # ── 行2左: 风速大小分布 ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(wind_mag, bins=60, color=COLORS['mag'], alpha=0.75,
            edgecolor='white', linewidth=0.3, density=True)
    x = np.linspace(wind_mag.min(), wind_mag.max(), 300)
    kde_mag = gaussian_kde(wind_mag, bw_method='scott')
    ax.plot(x, kde_mag(x), color='navy', lw=2, label='KDE')
    ax.axvline(wind_mag.mean(), color='red', lw=1.5, linestyle='--',
               label=f'μ={wind_mag.mean():.2f}')
    ax.set_title('Wind Magnitude Distribution', fontsize=10, fontweight='bold')
    ax.set_xlabel('|Wind| (m/s)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── 行2中: 水平风向极坐标玫瑰图 ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1], projection='polar')
    bins_dir = np.linspace(-np.pi, np.pi, 37)
    dir_rad = np.radians(wind_dir)
    counts, _ = np.histogram(dir_rad, bins=bins_dir)
    theta = (bins_dir[:-1] + bins_dir[1:]) / 2
    width = bins_dir[1] - bins_dir[0]
    bars = ax.bar(theta, counts, width=width, bottom=0,
                  color=COLORS['dir'], alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Horizontal Wind Direction', fontsize=10,
                 fontweight='bold', pad=15)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=8)
    ax.tick_params(axis='y', labelsize=7)

    # ── 行2右: N-E 散点图（密度着色）────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    # 下采样避免过密
    idx = np.random.choice(len(wind), min(8000, len(wind)), replace=False)
    sc = ax.scatter(wind[idx, 1], wind[idx, 0],
                    c=wind_mag[idx], cmap='viridis',
                    s=3, alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax, label='|Wind| (m/s)', shrink=0.85)
    ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.4)
    ax.axvline(0, color='black', lw=0.8, linestyle='--', alpha=0.4)
    ax.set_xlabel('East Wind (m/s)', fontsize=9)
    ax.set_ylabel('North Wind (m/s)', fontsize=9)
    ax.set_title('N-E Wind Vector Distribution', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── 行3左: 各分量箱线图 ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    bp = ax.boxplot([wind[:, 0], wind[:, 1], wind[:, 2]],
                    labels=['North', 'East', 'Down'],
                    patch_artist=True, notch=False,
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, color in zip(bp['boxes'], comp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color='black', lw=1.0, linestyle='--', alpha=0.5)
    ax.set_title('Wind Component Box Plot', fontsize=10, fontweight='bold')
    ax.set_ylabel('Wind Speed (m/s)', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── 行3中: 风速大小 CDF ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    sorted_mag = np.sort(wind_mag)
    cdf = np.arange(1, len(sorted_mag) + 1) / len(sorted_mag)
    ax.plot(sorted_mag, cdf, color=COLORS['mag'], lw=2)
    # 标注百分位
    for pct in [25, 50, 75, 90]:
        val = np.percentile(wind_mag, pct)
        ax.axvline(val, color='gray', lw=0.8, linestyle='--', alpha=0.7)
        ax.text(val + 0.05, pct / 100 - 0.04, f'P{pct}={val:.1f}',
                fontsize=7, color='gray')
    ax.set_title('Wind Magnitude CDF', fontsize=10, fontweight='bold')
    ax.set_xlabel('|Wind| (m/s)', fontsize=9)
    ax.set_ylabel('Cumulative Probability', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── 行3右: 统计摘要文字 ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    stats_lines = [
        f"Dataset: {tag} Test Set",
        f"Samples: {len(wind):,}",
        "",
        "Wind Components (m/s):",
        f"  North  μ={wind[:,0].mean():+.3f}  σ={wind[:,0].std():.3f}",
        f"  East   μ={wind[:,1].mean():+.3f}  σ={wind[:,1].std():.3f}",
        f"  Down   μ={wind[:,2].mean():+.3f}  σ={wind[:,2].std():.3f}",
        "",
        "Wind Magnitude (m/s):",
        f"  Mean   = {wind_mag.mean():.3f}",
        f"  Std    = {wind_mag.std():.3f}",
        f"  Min    = {wind_mag.min():.3f}",
        f"  Max    = {wind_mag.max():.3f}",
        f"  P50    = {np.percentile(wind_mag, 50):.3f}",
        f"  P90    = {np.percentile(wind_mag, 90):.3f}",
        "",
        "Horizontal Direction:",
        f"  Mean   = {wind_dir.mean():.1f}°",
        f"  Std    = {wind_dir.std():.1f}°",
    ]
    ax.text(0.05, 0.97, '\n'.join(stats_lines),
            transform=ax.transAxes, fontsize=9,
            va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f8f8',
                      edgecolor='#cccccc', alpha=0.9))

    save_path = os.path.join(out_dir, f'wind_field_overview_{tag.lower()}.svg')
    fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✓ 主图已保存: {save_path}")


def plot_waveform(wind: np.ndarray, tag: str, out_dir: str, sampling_rate: int = 20):
    """
    N / E / D 三分量风速波形图（时序）
    X 轴为时间（秒），三个子图上下排列共享 X 轴
    """
    N = len(wind)
    time_sec = np.arange(N) / sampling_rate

    wind_mag = np.linalg.norm(wind, axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'Wind Field Time Series  ({tag}, {N:,} samples @ {sampling_rate} Hz)',
                 fontsize=13, fontweight='bold')

    comp_cfg = [
        ('North', wind[:, 0], COLORS['north']),
        ('East',  wind[:, 1], COLORS['east']),
        ('Down',  wind[:, 2], COLORS['down']),
        ('Magnitude', wind_mag, COLORS['mag']),
    ]

    for ax, (label, data, color) in zip(axes, comp_cfg):
        ax.plot(time_sec, data, color=color, lw=0.6, alpha=0.85)
        ax.axhline(data.mean(), color='red', lw=1.0, linestyle='--',
                   label=f'μ={data.mean():.3f} m/s')
        ax.fill_between(time_sec, data.mean() - data.std(),
                        data.mean() + data.std(),
                        color=color, alpha=0.12, label=f'±σ={data.std():.3f}')
        ax.set_ylabel(f'{label}\n(m/s)', fontsize=9)
        ax.legend(fontsize=8, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(out_dir, f'wind_field_waveform_{tag.lower()}.svg')
    fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✓ 波形图已保存: {save_path}")


def plot_3d_wind(wind: np.ndarray, tag: str, out_dir: str):
    """
    三维风速分布图：两个视角的 3D 散点图
      - 点颜色按风速大小着色
      - 三个坐标轴投影面显示密度轮廓
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    wind_mag = np.linalg.norm(wind, axis=1)

    # 下采样，3D 散点不宜太多
    n_plot = min(6000, len(wind))
    idx = np.random.choice(len(wind), n_plot, replace=False)
    wn, we, wd = wind[idx, 0], wind[idx, 1], wind[idx, 2]
    mag = wind_mag[idx]

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f'3D Wind Vector Distribution  ({tag}, N={len(wind):,})',
                 fontsize=13, fontweight='bold')

    for col, (elev, azim, subtitle) in enumerate(
            [(25, -60, 'View 1  (Elev=25°, Azim=-60°)'),
             (15,  30, 'View 2  (Elev=15°, Azim=30°)')]):

        ax = fig.add_subplot(1, 2, col + 1, projection='3d')
        sc = ax.scatter(we, wn, -wd,          # X=East, Y=North, Z=Up(=-Down)
                        c=mag, cmap='plasma',
                        s=4, alpha=0.5, rasterized=True)

        # 三个坐标面的投影（密度轮廓）
        z_floor = -wd.min() - 0.5
        y_back  =  wn.max() + 0.5
        x_right =  we.max() + 0.5

        ax.scatter(we,      wn,      np.full_like(wd, z_floor),
                   c=mag, cmap='plasma', s=1, alpha=0.08, rasterized=True)
        ax.scatter(we,      np.full_like(wn, y_back), -wd,
                   c=mag, cmap='plasma', s=1, alpha=0.08, rasterized=True)
        ax.scatter(np.full_like(we, x_right), wn, -wd,
                   c=mag, cmap='plasma', s=1, alpha=0.08, rasterized=True)

        # 原点参考线
        for xs, ys, zs, xe, ye, ze in [
            (we.min(), 0, 0, we.max(), 0, 0),
            (0, wn.min(), 0, 0, wn.max(), 0),
            (0, 0, -wd.min(), 0, 0, -wd.max()),
        ]:
            ax.plot([xs, xe], [ys, ye], [zs, ze],
                    'k--', lw=0.6, alpha=0.4)

        ax.set_xlabel('East (m/s)',  fontsize=8, labelpad=4)
        ax.set_ylabel('North (m/s)', fontsize=8, labelpad=4)
        ax.set_zlabel('Up (m/s)',    fontsize=8, labelpad=4)
        ax.set_title(subtitle, fontsize=9)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=7)

        if col == 1:
            cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
            cb.set_label('|Wind| (m/s)', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'wind_field_3d_{tag.lower()}.svg')
    fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✓ 三维风场图已保存: {save_path}")


def plot_component_detail(wind: np.ndarray, tag: str, out_dir: str):
    """
    各分量详细对比图：violin + strip + 统计标注
    适合直接放入论文
    """
    wind_mag = np.linalg.norm(wind, axis=1)

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.suptitle(f'Wind Field Component Detail  ({tag})',
                 fontsize=12, fontweight='bold')

    data_list  = [wind[:, 0], wind[:, 1], wind[:, 2], wind_mag]
    labels     = ['North', 'East', 'Down', 'Magnitude']
    colors     = [COLORS['north'], COLORS['east'], COLORS['down'], COLORS['mag']]
    units      = ['m/s'] * 4

    for ax, data, label, color, unit in zip(axes, data_list, labels, colors, units):
        parts = ax.violinplot(data, positions=[0], showmedians=True,
                              showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(2)

        # 叠加箱线图
        ax.boxplot(data, positions=[0], widths=0.15,
                   patch_artist=False,
                   medianprops=dict(color='red', linewidth=0),
                   whiskerprops=dict(linewidth=1.0, linestyle='--'),
                   capprops=dict(linewidth=1.0),
                   flierprops=dict(marker='.', markersize=1, alpha=0.3))

        ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.4)
        ax.set_title(f'{label} Wind', fontsize=10, fontweight='bold')
        ax.set_ylabel(unit, fontsize=9)
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 标注统计
        txt = (f"μ={data.mean():+.3f}\n"
               f"σ={data.std():.3f}\n"
               f"[{data.min():.2f},\n {data.max():.2f}]")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                fontsize=8, va='top', ha='right', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'wind_field_violin_{tag.lower()}.svg')
    fig.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ 小提琴图已保存: {save_path}")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='测试集风场数据可视化')
    parser.add_argument('--ood', action='store_true', help='使用 OOD 测试集')
    parser.add_argument('--out', type=str, default=None, help='输出目录（默认 data/evaluation/wind_field_plots）')
    args = parser.parse_args()

    print("=" * 60)
    print("  测试集风场数据可视化")
    print("=" * 60)

    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 输出目录：带时间戳
    if args.out:
        out_dir = args.out
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(PROJECT_ROOT, 'data', 'evaluation',
                               f'plot_test_wind_field_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    # 加载数据
    wind, tag = load_data(config, use_ood=args.ood)

    # 绘图
    print("\n生成图表...")
    plot_overview(wind, tag, out_dir)
    plot_waveform(wind, tag, out_dir, sampling_rate=config['data']['sampling_rate'])
    plot_3d_wind(wind, tag, out_dir)
    plot_component_detail(wind, tag, out_dir)

    print(f"\n✅ 全部图表已保存到: {out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
