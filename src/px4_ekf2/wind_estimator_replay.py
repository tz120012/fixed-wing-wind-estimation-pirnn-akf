#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
    风速估计器回放脚本 - 从 PX4 ULog 文件中提取数据并重新运行风速估计算法
    该脚本使用扩展卡尔曼滤波器(EKF)融合空速传感器和速度数据来估计风速和空速缩放因子

    主回放脚本，从 ULog 抽取速度/空速/高度数据，运行风速+空速缩放 EKF，并保存结果图像。
"""

import matplotlib.pylab as plt
from pyulog import ULog
import numpy as np
import pandas as pd
import os

from fuse_airspeed import fuse_airspeed


def plot_wind_csv(csv_path, output_dir, timestamp):
    """绘制 wind_data_ekf2.csv 中的真值与估计风场对比图并保存到文件。"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Failed to load wind data from {csv_path}: {exc}")
        return None

    required_truth = {'time_sec', 'wind_north_ms', 'wind_east_ms', 'wind_down_ms'}
    required_est = {'wind_est_north_ms', 'wind_est_east_ms', 'airspeed_scale_est'}
    if not required_truth.issubset(df.columns):
        missing = ', '.join(sorted(required_truth - set(df.columns)))
        print(f"Skip wind_data_ekf2 plot: missing truth columns: {missing}")
        return None
    if not required_est.issubset(df.columns):
        missing = ', '.join(sorted(required_est - set(df.columns)))
        print(f"Skip wind_data_ekf2 plot: missing estimate columns: {missing}")
        return None

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    fig.suptitle('Wind Truth vs Estimate')

    axes[0].plot(df['time_sec'], df['wind_north_ms'], label='Truth North', linewidth=1.0)
    axes[0].plot(df['time_sec'], df['wind_est_north_ms'], label='Estimate North', linewidth=1.0)
    axes[0].set_ylabel('North (m/s)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    axes[1].plot(df['time_sec'], df['wind_east_ms'], label='Truth East', linewidth=1.0)
    axes[1].plot(df['time_sec'], df['wind_est_east_ms'], label='Estimate East', linewidth=1.0)
    axes[1].set_ylabel('East (m/s)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    axes[2].plot(df['time_sec'], df['wind_down_ms'], label='Truth Down', linewidth=1.0)
    axes[2].set_ylabel('Down (m/s)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    axes[3].plot(df['time_sec'], df['airspeed_scale_est'], label='Airspeed Scale', linewidth=1.0, color='tab:orange')
    axes[3].set_ylabel('Scale (-)')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path = os.path.join(output_dir, f'wind_data_comparison_{timestamp}.svg')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Wind data comparison plot saved to: {out_path}")
    return out_path

def getData(log, topic_name, variable_name, instance=0):
    """
    从 ULog 文件中提取指定话题的变量数据

    参数:
        log: ULog 对象，已加载的日志文件
        topic_name: 话题名称 (如 'vehicle_local_position')
        variable_name: 变量名称 (如 'vx')
        instance: 多实例ID，默认为0（用于区分同一话题的多个实例）

    返回:
        variable_data: numpy数组，包含提取的变量数据
    """
    variable_data = np.array([])
    for elem in log.data_list:
        if elem.name == topic_name:
            if instance == elem.multi_id:
                variable_data = elem.data[variable_name]
                break

    return variable_data

def us2s(time_ms):
    """
    将时间从微秒转换为秒

    参数:
        time_ms: 微秒时间戳

    返回:
        秒为单位的时间
    """
    return time_ms * 1e-6

def run(logfile, use_gnss):
    """
    主运行函数 - 执行风速估计器的回放

    参数:
        logfile: ULog 文件路径
        use_gnss: 布尔值，True则使用GNSS速度，False则使用局部位置估计速度

    功能:
        1. 从日志文件中提取速度、空速和高度数据
        2. 使用扩展卡尔曼滤波器估计风速（北向和东向）以及空速缩放因子
        3. 绘制估计结果
    """
    # 加载 ULog 文件
    log = ULog(logfile)

    # ========================================================================
    # 数据提取 - 根据配置选择速度数据源
    # ========================================================================

    if use_gnss:
        # 使用 GNSS 速度数据（来自GPS接收器）
        v_local = np.array([getData(log, 'vehicle_gps_position', 'vel_n_m_s'),  # 北向速度
                  getData(log, 'vehicle_gps_position', 'vel_e_m_s'),            # 东向速度
                  getData(log, 'vehicle_gps_position', 'vel_d_m_s')])           # 下向速度
        t_v_local = us2s(getData(log, 'vehicle_gps_position', 'timestamp'))

    else:
        # 使用局部位置估计的速度数据（来自融合估计器）
        v_local = np.array([getData(log, 'vehicle_local_position', 'vx'),  # X轴速度（北）
                  getData(log, 'vehicle_local_position', 'vy'),            # Y轴速度（东）
                  getData(log, 'vehicle_local_position', 'vz')])           # Z轴速度（下）
        t_v_local = us2s(getData(log, 'vehicle_local_position', 'timestamp'))

    # 提取真空速数据（来自空速传感器）
    true_airspeed = getData(log, 'airspeed', 'true_airspeed_m_s')
    t_true_airspeed = us2s(getData(log, 'airspeed', 'timestamp'))


    # 提取距地高度数据（用于判断是否离地足够高以进行风速估计）
    dist_bottom = getData(log, 'vehicle_local_position', 'dist_bottom')
    t_dist_bottom = us2s(getData(log, 'vehicle_local_position', 'timestamp'))

    # ========================================================================
    # 初始化卡尔曼滤波器状态和协方差
    # ========================================================================

    # 状态向量初始化 [wind_north, wind_east, airspeed_scale]
    state = np.array([0.0, 0.0, 1.0])  # 初始风速为0，空速缩放因子为1

    # 状态协方差矩阵初始化（对角矩阵）
    P = np.diag([1.0, 1.0, 1e-4])  # 风速不确定性较大，缩放因子不确定性较小

    # 过程噪声参数
    wind_nsd = 1e-2      # 风速过程噪声谱密度 (m/s/sqrt(s))
    scale_nsd = 1e-4     # 空速缩放因子过程噪声谱密度 (1/sqrt(s))

    # 过程噪声协方差矩阵 Q
    Q = np.diag([wind_nsd**2, wind_nsd**2, scale_nsd**2])

    # 测量噪声方差 R（空速测量的标准差为1.4 m/s）
    R = 1.4**2

    # 数值稳定性常数，防止除零
    epsilon = 1e-8

    # 初始化时间
    t_now = t_v_local[0]

    # ========================================================================
    # 准备输出数组
    # ========================================================================

    n = len(t_v_local)  # 数据点数量
    wind_est_n = np.zeros(n)  # 北向风速估计
    wind_est_e = np.zeros(n)  # 东向风速估计
    scale_est = np.zeros(n)   # 空速缩放因子估计

    # 数据索引初始化
    i_airspeed = 0      # 空速数据索引
    i_dist_bottom = 0   # 距地高度数据索引

    # ========================================================================
    # 主循环 - 对每个速度测量时刻进行处理
    # ========================================================================

    for i in range(n):
        # 计算时间步长
        dt = t_v_local[i] - t_now
        t_now = t_v_local[i]  # 基于局部位置更新时刻运行

        # 同步距地高度数据
        while i_dist_bottom < len(t_dist_bottom) and t_dist_bottom[i_dist_bottom] <= t_now:
            i_dist_bottom += 1
        i_dist_bottom -= 1

        # 高度检查：只在离地超过20米时进行风速估计
        # 这样可以避免地面效应和低空时不准确的空速测量
        if dist_bottom[i_dist_bottom] > 20.0:

            # 预测步骤：基于过程噪声更新协方差矩阵
            # P_k|k-1 = P_k-1 + Q * dt
            P += Q * dt

            # 检查是否有新的空速测量可用
            if t_true_airspeed[i_airspeed] < t_now:
                # 同步到最新的空速测量
                while i_airspeed < len(t_true_airspeed) and t_true_airspeed[i_airspeed] < t_now:
                    i_airspeed += 1
                i_airspeed -= 1

                # 测量更新步骤：融合空速测量
                # 调用自动生成的空速融合函数
                (H, K, innov_var, innov) = fuse_airspeed(
                    np.asarray(v_local[:,i]),           # 当前速度向量
                    state,                              # 当前状态估计
                    P, # P.flatten(),                        # 展平的协方差矩阵
                    true_airspeed[i_airspeed],         # 空速测量值
                    R,                                  # 测量噪声方差
                    epsilon                             # 数值稳定性常数
                )

                # 状态更新：x_k = x_k|k-1 + K * innov
                state += np.array(K) * innov

                # 协方差更新：P_k = P_k|k-1 - K * H * P_k|k-1
                P -= K * H * P

                # 移动到下一个空速测量
                i_airspeed += 1

        # 保存当前估计值
        wind_est_n[i] = state[0]  # 北向风速
        wind_est_e[i] = state[1]  # 东向风速
        scale_est[i] = state[2]   # 空速缩放因子

    # ========================================================================
    # 结果可视化
    # ========================================================================

    fig_estimate, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    axes[0].plot(t_v_local, wind_est_n, label='north')
    axes[0].plot(t_v_local, wind_est_e, label='east')
    axes[0].set_ylabel("wind speed (m/s)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_v_local, scale_est)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("airspeed scale (-)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_true_airspeed, true_airspeed)
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("true airspeed (m/s)")
    axes[2].grid(True, alpha=0.3)

    fig_estimate.tight_layout()

    # 保存图像到 Dataset 目录
    import datetime
    dataset_dir = '../Dataset'
    os.makedirs(dataset_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(dataset_dir, f'wind_estimation_{timestamp}.svg')
    fig_estimate.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig_estimate)
    print(f"Wind estimation plot saved to: {save_path}")

    # 将估计风速和缩放因子写入 wind_data_ekf2.csv，便于与真值对比
    estimates = pd.DataFrame({
        'time_sec': t_v_local,
        'wind_est_north_ms': wind_est_n,
        'wind_est_east_ms': wind_est_e,
        'airspeed_scale_est': scale_est,
    })
    estimates.sort_values('time_sec', inplace=True)

    wind_csv_path = os.path.join(dataset_dir, 'wind_data_ekf2.csv')
    if os.path.exists(wind_csv_path):
        try:
            truth_df = pd.read_csv(wind_csv_path)
            drop_cols = [
                'wind_est_north_ms',
                'wind_est_east_ms',
                'airspeed_scale_est',
            ]
            truth_sorted = truth_df.sort_values('time_sec').drop(
                columns=[col for col in drop_cols if col in truth_df.columns]
            )
            merged = pd.merge_asof(
                truth_sorted,
                estimates,
                on='time_sec',
                direction='nearest',
            )
            merged.to_csv(wind_csv_path, index=False)
            print(f"Wind estimates merged into: {wind_csv_path}")
            plot_wind_csv(wind_csv_path, dataset_dir, timestamp)
        except Exception as exc:
            fallback_path = os.path.join(dataset_dir, f'wind_estimates_only_{timestamp}.csv')
            estimates.to_csv(fallback_path, index=False)
            print(f"Failed to merge with existing wind data: {exc}")
            print(f"Saved estimates separately to: {fallback_path}")
    else:
        estimates.to_csv(wind_csv_path, index=False)
        print(f"Wind estimate export saved to: {wind_csv_path}")
        plot_wind_csv(wind_csv_path, dataset_dir, timestamp)

    # plt.show()

if __name__ == '__main__':
    import argparse

    # 获取脚本路径（不包含文件名）
    script_path = os.path.split(os.path.realpath(__file__))[0]

    # ========================================================================
    # 命令行参数解析
    # ========================================================================

    parser = argparse.ArgumentParser(
        description='Wind estimator with airspeed scale factor')

    # 必需参数：日志文件路径
    # python3 wind_estimator_replay.py ../Dataset/log_8_2025-10-20-22-55-18.ulg
    parser.add_argument('logfile', help='Full ulog file path, name and extension', type=str)

    # 可选参数：使用GNSS速度代替局部速度估计
    parser.add_argument('--gnss', help='Use GNSS velocity instead of local velocity estimate',
                        action='store_true')
    args = parser.parse_args()

    # 转换为绝对路径
    logfile = os.path.abspath(args.logfile)

    # 运行风速估计器回放
    run(logfile, args.gnss)
