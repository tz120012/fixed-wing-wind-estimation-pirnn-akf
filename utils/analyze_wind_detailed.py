#!/usr/bin/env python3
# 保存为: analyze_wind_detailed.py
# 分析gazebo风插件输出的风速数据，生成详细统计和图表
# 运行：python3 analyze_wind_detailed.py

import pandas as pd
import numpy as np
import os
import matplotlib
# Use a non-interactive backend (Agg) to avoid Qt / XDG_RUNTIME_DIR issues in headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 读取原始列名并映射为易读变量名
RAW_DATA_PATH = '../Dataset/flight_data_all_in_one.csv'
RAW_COLUMN_MAP = {
    'Time': 'timestamp_sec',
    '/fdm/jsbsim/simulation/sim-time-sec': 'sim_time_sec',
    '/fdm/jsbsim/atmosphere/wind-north-fps': 'wind_north_fps',
    '/fdm/jsbsim/atmosphere/wind-east-fps': 'wind_east_fps',
    '/fdm/jsbsim/atmosphere/wind-down-fps': 'wind_down_fps',
    '/fdm/jsbsim/velocities/vc-fps': 'calibrated_airspeed_fps',
    '/fdm/jsbsim/velocities/vtrue-fps': 'true_airspeed_fps',
    '/fdm/jsbsim/velocities/vg-fps': 'ground_speed_fps',
    '/fdm/jsbsim/position/h-agl-ft': 'height_agl_ft',
    '/fdm/jsbsim/position/lat-geod-deg': 'latitude_deg',
    '/fdm/jsbsim/position/long-gc-deg': 'longitude_deg',
    '/fdm/jsbsim/velocities/v-north-fps': 'body_velocity_north_fps',
    '/fdm/jsbsim/velocities/v-east-fps': 'body_velocity_east_fps',
    '/fdm/jsbsim/velocities/v-down-fps': 'body_velocity_down_fps',
    '/fdm/jsbsim/attitude/pitch-rad': 'pitch_rad',
    '/fdm/jsbsim/attitude/roll-rad': 'roll_rad',
    '/fdm/jsbsim/attitude/psi-rad': 'yaw_rad',
    '/fdm/jsbsim/velocities/p-rad_sec': 'roll_rate_rps',
    '/fdm/jsbsim/velocities/q-rad_sec': 'pitch_rate_rps',
    '/fdm/jsbsim/velocities/r-rad_sec': 'yaw_rate_rps',
    '/fdm/jsbsim/fcs/aileron-cmd-norm': 'aileron_cmd_norm',
    '/fdm/jsbsim/fcs/elevator-cmd-norm': 'elevator_cmd_norm',
    '/fdm/jsbsim/fcs/throttle-cmd-norm': 'throttle_cmd_norm',
    '/fdm/jsbsim/fcs/rudder-cmd-norm': 'rudder_cmd_norm',
}

df_raw = pd.read_csv(RAW_DATA_PATH)
df = df_raw.rename(columns=RAW_COLUMN_MAP)

FTPS_TO_MPS = 0.3048
df['wind_north_ms'] = df['wind_north_fps'] * FTPS_TO_MPS
df['wind_east_ms'] = df['wind_east_fps'] * FTPS_TO_MPS
df['wind_down_ms'] = df['wind_down_fps'] * FTPS_TO_MPS

horizontal_speed = np.hypot(df['wind_north_ms'], df['wind_east_ms'])
df['wind_speed_ms'] = np.hypot(horizontal_speed, df['wind_down_ms'])
df['wind_direction_deg'] = np.degrees(np.arctan2(df['wind_east_ms'], df['wind_north_ms']))
df['wind_elevation_deg'] = np.degrees(np.arctan2(-df['wind_down_ms'], horizontal_speed))

wind_export = df[['sim_time_sec', 'wind_north_ms', 'wind_east_ms', 'wind_down_ms']].copy()
wind_export = wind_export.rename(columns={'sim_time_sec': 'time_sec'})
wind_export_path = os.path.abspath('../Dataset/wind_data_ekf2.csv')
wind_export.to_csv(wind_export_path, index=False)

print("=" * 60)
print("Wind Data Detailed Analysis")
print("=" * 60)

# 基本统计
print(f"\n📊 Dataset Summary:")
print(f"  Total records: {len(df)}")
print(f"  Simulation duration: {df['sim_time_sec'].iloc[-1]:.2f} seconds")
print(f"  Actual duration: {df['timestamp_sec'].iloc[-1] - df['timestamp_sec'].iloc[0]:.2f} seconds")

print(f"\n💾 Wind component export saved to: {wind_export_path}")

# 风速分量统计
print(f"\n🌬️  Wind Components (m/s):")
for component in ['wind_north_ms', 'wind_east_ms', 'wind_down_ms']:
    print(f"  {component:15s}: mean={df[component].mean():7.4f}, std={df[component].std():7.4f}, "
          f"min={df[component].min():7.4f}, max={df[component].max():7.4f}")

# 总风速统计
print(f"\n💨 Wind Speed (m/s):")
print(f"  Mean:   {df['wind_speed_ms'].mean():.4f}")
print(f"  Median: {df['wind_speed_ms'].median():.4f}")
print(f"  Std:    {df['wind_speed_ms'].std():.4f}")
print(f"  Min:    {df['wind_speed_ms'].min():.4f}")
print(f"  Max:    {df['wind_speed_ms'].max():.4f}")

# 风向统计
print(f"\n🧭 Wind Direction (degrees):")
print(f"  Mean:   {df['wind_direction_deg'].mean():.2f}°")
print(f"  Std:    {df['wind_direction_deg'].std():.2f}°")
print(f"  Range:  [{df['wind_direction_deg'].min():.2f}°, {df['wind_direction_deg'].max():.2f}°]")

# 仰角统计
print(f"\n📐 Wind Elevation (degrees):")
print(f"  Mean:   {df['wind_elevation_deg'].mean():.2f}°")
print(f"  Std:    {df['wind_elevation_deg'].std():.2f}°")
updraft_pct = (df['wind_elevation_deg'] > 0).sum() / len(df) * 100
print(f"  Updraft: {updraft_pct:.1f}% of time")

# 绘图
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 1. 风速分量
axes[0, 0].plot(df['sim_time_sec'], df['wind_east_ms'], label='East', alpha=0.8)
axes[0, 0].plot(df['sim_time_sec'], df['wind_north_ms'], label='North', alpha=0.8)
axes[0, 0].plot(df['sim_time_sec'], -df['wind_down_ms'], label='Up', alpha=0.8)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Velocity (m/s)')
axes[0, 0].set_title('Wind Velocity Components')
axes[0, 0].legend(loc='upper right')
axes[0, 0].grid(True, alpha=0.3)

# 2. 总风速
axes[0, 1].plot(df['sim_time_sec'], df['wind_speed_ms'], linewidth=0.8)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Speed (m/s)')
axes[0, 1].set_title('Total Wind Speed')
axes[0, 1].grid(True, alpha=0.3)

# 3. 水平风向
axes[1, 0].plot(df['sim_time_sec'], df['wind_direction_deg'], linewidth=0.8)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Direction (degrees)')
axes[1, 0].set_title('Horizontal Wind Direction')
axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3, label='East')
axes[1, 0].axhline(y=90, color='g', linestyle='--', alpha=0.3, label='North')
axes[1, 0].legend(loc='upper right')
axes[1, 0].grid(True, alpha=0.3)

# 4. 仰角
axes[1, 1].plot(df['sim_time_sec'], df['wind_elevation_deg'], linewidth=0.8)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Elevation (degrees)')
axes[1, 1].set_title('Wind Elevation Angle')
axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Horizontal')
axes[1, 1].legend(loc='upper right')
axes[1, 1].fill_between(df['sim_time_sec'], 0, df['wind_elevation_deg'], 
                         where=df['wind_elevation_deg']>0, alpha=0.2, color='green', label='Updraft')
axes[1, 1].fill_between(df['sim_time_sec'], 0, df['wind_elevation_deg'], 
                         where=df['wind_elevation_deg']<0, alpha=0.2, color='red', label='Downdraft')
axes[1, 1].grid(True, alpha=0.3)

# 5. 风速分布
axes[2, 0].hist(df['wind_speed_ms'], bins=50, edgecolor='black', alpha=0.7)
axes[2, 0].set_xlabel('Wind Speed (m/s)')
axes[2, 0].set_ylabel('Frequency')
axes[2, 0].set_title('Wind Speed Distribution')
axes[2, 0].grid(True, alpha=0.3)

# 6. 风向-仰角散点图
scatter = axes[2, 1].scatter(df['wind_direction_deg'], df['wind_elevation_deg'], 
                             c=df['wind_speed_ms'], cmap='viridis', alpha=0.6, s=2)
axes[2, 1].set_xlabel('Direction (degrees)')
axes[2, 1].set_ylabel('Elevation (degrees)')
axes[2, 1].set_title('Direction vs Elevation (colored by speed)')
plt.colorbar(scatter, ax=axes[2, 1], label='Speed (m/s)')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.abspath('../Dataset/wind_detailed_analysis.svg')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n📈 Plot saved to: {out_path}")
backend = matplotlib.get_backend().lower()
if 'agg' not in backend and os.environ.get('DISPLAY'):
    plt.show()
else:
    plt.close()