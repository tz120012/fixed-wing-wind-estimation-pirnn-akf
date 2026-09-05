# PX4 风场分析与回放工具

本仓库提供两个主要脚本，用于从 PX4 ULog 日志和仿真数据中分析并回放风场估计结果：

- `wind_estimator_replay.py`：读取 PX4 ULog 日志，重放风速 + 空速缩放 EKF，生成估计曲线并将结果写回 `Dataset/wind_data.csv`。
- `analyze_wind_detailed.py`：针对 `Dataset/flight_data_all_in_one.csv` 进行真值风场统计分析、绘图，并生成对比所需的基础 CSV。

## 环境依赖

- Python 3.10 及以上
- 依赖库：`numpy`、`pandas`、`matplotlib`、`pyulog`

安装示例：

```bash
python3 -m pip install --user numpy pandas matplotlib pyulog
```

## 数据准备

1. 将 PX4 ULog 日志放在 `Dataset/` 目录，例如 `Dataset/log_mission_all_in_one_2025-11-14-20-22-43.ulg`。
2. 将仿真导出的 JSBSim CSV（包含 `/fdm/jsbsim/...` 字段）放在 `Dataset/flight_data_all_in_one.csv`。

## 分析流程

### 1. 生成真值风场统计

```bash
cd /path/to/px4_ekf2

python3 analyze_wind_detailed.py
```

运行后会：

- 生成/覆盖 `Dataset/wind_data.csv`，其中包含真值风速的北/东/下分量（单位 m/s）。
- 输出统计信息，并在 `Dataset/wind_detailed_analysis.svg` 中保存可视化图表。

### 2. 回放 PX4 EKF 估计

```bash
python3 wind_estimator_replay.py Dataset/log_mission_all_in_one_2025-11-14-20-22-43.ulg
```


脚本执行后会：

- 运行 EKF，得到风速北/东分量与空速缩放因子的估计序列。
- 生成 `Dataset/wind_estimation_<时间戳>.svg` 回放图。
- 将估计结果按照时间对齐合并进 `Dataset/wind_data.csv`，方便与真值做对比分析。
  - 若原 CSV 中已有旧的估计列，脚本会自动清理后再写入。
  - 若合并失败，会在 `Dataset/` 下生成 `wind_estimates_only_<时间戳>.csv` 作为备份。

## 注意事项

- `wind_estimator_replay.py` 使用空速测量更新风估计，需要日志中包含有效的 `airspeed` 话题。
- 若看到 `dist_bottom` 导致估计始终为 0，可考虑调整/移除 20 m 高度门限或改用其它高度源。
- Matplotlib 若因环境冲突无法导入 `Axes3D`，通常是系统/用户双重安装导致；保持单一安装即可。
- 若出现 `QStandardPaths: wrong permissions on runtime directory`，可以执行 `chmod 700 /run/user/<UID>` 修复。

## 结果对比

最终的 `Dataset/wind_data.csv` 将包含以下列：

- `time_sec`：仿真/日志时间（秒）
- `wind_north_ms`、`wind_east_ms`、`wind_down_ms`：真值风速（m/s）
- `wind_est_north_ms`、`wind_est_east_ms`、`airspeed_scale_est`：EKF 估计输出

可将该文件导入 Excel、matplotlib 或其他工具进行误差分析、绘图等进一步处理。
