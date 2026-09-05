# PX4-SITL + JSBSim 无人值守数据采集系统

根据 [PX4-SITL+JSBSim数据集生成完整指南](../doc/PX4-SITL+JSBSim数据集生成完整指南.md) 实现的**无人值守**数据采集脚本。当前为论文版动态风设计：160 轮 SITL，每轮 5 段，共 **800 条**（训练 600 + 验证 100 + 测试-ID 70 + 测试-OOD 30）。

## 🎯 核心特性

### 1. **可重现数据生成**
- 使用固定随机种子（默认 26）确保可重复性
- 每轮使用 `seed = 26 + run_index` 生成一致的风场和机动参数
- 支持多次运行生成完全相同的数据集

### 2. **断点续传**
- 自动检测已存在的数据文件，跳过已完成轮次
- 无人值守模式下失败后自动继续后续轮次
- 支持分批次采集，随时中断和恢复

### 3. **缺失数据报告**
- 160轮完成后自动生成缺失清单
- 列出所有未采集的轮次和段
- 提供精确的补采命令

### 4. **灵活补采**
- 支持补采整轮（5段）或单段
- 使用相同种子确保补采数据参数一致性
- 精确定位和修复缺失数据

### 5. **纯 Offboard 控制**
- 盘旋和8字机动使用 Offboard 速度控制，消除 `goto_location` 依赖
- 大幅减少模式切换，降低 COMMAND_DENIED 错误
- 提高采集成功率（从 30-40% 提升到 70-85%）

### 6. **自动故障恢复**
- **MAVSDK 连接监控**：自动检测 mavsdk_server 进程崩溃
- **自动重启机制**：mavsdk_server 崩溃后自动重启并重连
- **连接健康检查**：每段执行前验证 gRPC 连接状态
- **快速失败策略**：检测到不可恢复错误时立即终止当前段
- **日志记录**：mavsdk_server 输出重定向到 `logs/mavsdk_server.log`

---

## 📁 目录结构

> 2026-05 完成大规模重构，新增 `lib/` 库代码包；`scripts/` 仅保留入口与 shim，向后兼容 100%。

```
dataset_generation/
├── configs/
│   ├── dataset_config.json     # 数据集级配置（飞行/风/机动参数）
│   └── runtime.yaml            # 运行期参数：所有 sleep / timeout / retry / 阈值 / 端口
├── lib/                         # ★新增：库代码包，按职责拆分（每文件 < 400 行）
│   ├── errors.py               # 异常层级（SitlStartFailed / GpsTimeout / ...）
│   ├── cli.py                  # 命令行 dispatcher
│   ├── config/
│   │   ├── runtime.py          # RuntimeConfig dataclass + YAML 加载
│   │   └── dataset.py          # DEFAULT_DATASET_CONFIG + lru_cache 加载
│   ├── obs/
│   │   ├── logger.py           # 结构化 JSONL 日志（logs/runtime.jsonl）
│   │   └── failure.py          # FailureRecord + 分类器
│   ├── infra/
│   │   ├── px4.py              # Px4SitlProcess (async ctx mgr)
│   │   ├── mavsdk_server.py    # MavsdkServerProcess
│   │   └── jsbsim_bridge.py    # 风场写入 + wind_truth 查找
│   ├── planner/
│   │   ├── schedule.py         # build_paper_run_schedule
│   │   ├── wind_factory.py     # generate_wind_config / 阵风采样
│   │   └── segment_factory.py  # generate_one_segment_config / 转弯标签
│   ├── pipeline/
│   │   ├── stages.py           # Stage / Pipeline / PipelineCtx 基类
│   │   ├── segment.py          # SegmentPipeline (3 个 Stage)
│   │   ├── sortie.py           # SortiePipeline (cleanup-always 不变量)
│   │   └── session.py          # SessionPipeline
│   ├── validation/
│   │   └── quality_gates.py    # validate_segment_records → GateResult
│   └── recovery/
│       └── runner.py           # generate_missing_report / run_*_recovery
├── scripts/
│   ├── jsbsim_wind_config.py   # JSBSim 风场配置（写 XML）
│   ├── flight_controller.py   # MAVSDK 飞行控制（纯 Offboard 机动）
│   ├── data_logger.py          # 遥测记录（含逐行真值风）
│   ├── generate_dataset.py     # 主脚本：CLI 入口（瘦身 ~750 行，原 1609 行）
│   ├── postprocess_to_jsbsim_csv.py  # JSON → JSBSim 格式 CSV
│   ├── inspect_wind_dataset.py       # 风场验收：统计 + 打分 + 可视化
│   └── tests/                  # 单元/集成测试（共 15 个）
├── data/                        # 输出：train/val/test_id/test_ood
│   ├── train/                   # 120 轮，共 600 段
│   ├── val/                     # 20 轮，共 100 段
│   ├── test_id/                 # 10 轮，共 50 段
│   └── test_ood/                # 10 轮，共 50 段
├── logs/                        # 运行日志
│   ├── multi_segment_160runs.log     # 主运行日志
│   ├── missing_data_report.txt       # 缺失数据报告
│   ├── runtime.jsonl                 # ★新增：结构化事件流（每行一个 JSON）
│   ├── segment_failures.jsonl        # ★新增：段级失败结构化记录
│   ├── px4_last_run.log              # PX4 输出日志
│   └── mavsdk_server.log             # MAVSDK server 日志
├── run_unattended.sh            # 一键无人值守启动
└── README.md
```

### 重构后的关键改进

- **集中配置**：所有 sleep / timeout / retry / 阈值移到 [`configs/runtime.yaml`](configs/runtime.yaml)，改阈值不再翻代码。
- **结构化日志**：除原有 `multi_segment_160runs.log` 外，新增 `logs/runtime.jsonl`（每个 Stage start/end）与 `logs/segment_failures.jsonl`（每个 FailureRecord）。
- **异常分类**：原 30+ 处 `try/except: pass` 替换为 `FailureRecord`（按 `process / flight / telemetry / validation / cleanup` 分类）。
- **资源管理**：PX4 SITL / mavsdk_server 改为 `async with` 上下文管理，保证日志文件、子进程组、TCP 端口在异常路径上也能正确清理。
- **质量门可重写**：阈值改为构造时注入而不是 `raise RuntimeError`。`GateResult.failures` 列表暴露所有违规原因。
- **向后兼容**：`from data_logger import DataLogger` / `from generate_dataset import build_paper_run_schedule` 等导入路径保持不变；`run_unattended.sh`、`tests/run_all.sh`、`--mode multi_segment_160` 等 CLI 参数零修改。

## 🔧 环境要求

- **PX4-Autopilot**（或 PX4-Autopilot-v133）已编译，支持 `make px4_sitl jsbsim_rascal`
- **Python 3.7+**：`mavsdk`, `numpy`
- **可选**：`pandas`（后处理 CSV 批量用）

```bash
pip install mavsdk numpy pandas
```

## 📋 快速开始

### 方式 1: 使用 Shell 脚本（推荐）

```bash
cd dataset_generation
chmod +x run_unattended.sh

# 方式 A：后台运行，断 SSH 不中断（使用默认 seed=26）
./run_unattended.sh nohup
tail -f logs/multi_segment_160runs.log

# 指定自定义种子
./run_unattended.sh nohup 26     # 使用 seed=26
./run_unattended.sh nohup 123    # 使用 seed=123

# 🚀 仿真加速：使用 --speed 参数（推荐 2-4 倍速）
./run_unattended.sh nohup 26 --speed 2    # 2倍速，采集时间减半
./run_unattended.sh nohup 26 --speed 4    # 4倍速，采集时间缩短至1/4

# 方式 B：tmux 会话内运行，可随时 attach
./run_unattended.sh tmux         # 默认 seed=26
./run_unattended.sh tmux 26      # 指定 seed=26
./run_unattended.sh tmux 26 --speed 3     # 3倍速仿真

# 方式 C：前台运行（测试用）
./run_unattended.sh foreground
./run_unattended.sh foreground 26 --speed 2  # 2倍速前台运行

# 用这个来将远程服务器的PX4-SITL 连接到本地QGC
mavproxy.py --master=udp:127.0.0.1:14550 --out=tcpin:0.0.0.0:5790
启动QGC,建立tcp连接：127.0.0.1:5790
```

**特性**：
- ✅ 自动支持断点续传（跳过已存在文件）
- ✅ 默认使用 seed=26，可自定义
- ✅ 自动检测 PX4 路径
- ✅ 支持仿真加速（`--speed` 参数，默认 1:1 实时）

### 方式 2: 直接调用 Python（支持更多参数）

```bash
cd dataset_generation/scripts

# 全自动 160 轮采集（支持断点续传）
python3 generate_dataset.py --mode multi_segment_160 --seed 26

# 指定 PX4 根目录与输出目录
export PX4_ROOT=/path/to/PX4-Autopilot   # 可选
python3 generate_dataset.py --mode multi_segment_160 \
  --output-dir ../data --log ../logs/multi_segment_160runs.log --seed 26
```

---

## 🚀 使用模式详解

### 模式 1: 全自动 160 轮采集（断点续传）

```bash
cd scripts

# 首次运行或继续未完成的采集
python3 generate_dataset.py --mode multi_segment_160 --seed 26

# 强制重新采集所有数据（不跳过已存在）
python3 generate_dataset.py --mode multi_segment_160 --seed 26 --no-skip
```

**特性**：
- ✅ 自动跳过已存在的数据文件
- ✅ 失败后继续下一轮，不从头开始
- ✅ 采集完成后自动生成缺失报告
- ✅ 使用固定种子确保可重现

**输出文件**：
```
logs/multi_segment_160runs.log         # 运行日志
logs/missing_data_report.txt          # 缺失数据清单
data/train/datasets-X-Y.json          # 采集的数据
data/val/datasets-X-Y.json
data/test_id/datasets-X-Y.json
data/test_ood/datasets-X-Y.json
```

---

### 模式 2: 生成缺失数据报告

```bash
python3 generate_dataset.py --mode report
```

**输出示例**：
```
============================================================
缺失数据报告 (Missing Data Report)
============================================================

总计: 15/800 段缺失

缺失明细:
------------------------------------------------------------
Run  2 | Seg 3 | train    | datasets-2-3.json
Run  7 | Seg 1 | train    | datasets-7-1.json
...

补采命令示例 (补采单轮):
------------------------------------------------------------
python generate_dataset.py --mode recover --run 2
python generate_dataset.py --mode recover --run 7

补采单段命令示例:
------------------------------------------------------------
python generate_dataset.py --mode recover --run 2 --segment 3
python generate_dataset.py --mode recover --run 7 --segment 1
```

---

### 模式 3: 补采整轮（5段）

```bash
# 补采第 7 轮的所有 5 段数据
python3 generate_dataset.py --mode recover --run 7 --seed 26

# 批量补采多轮
for run in 2 7 15; do
    python3 generate_dataset.py --mode recover --run $run --seed 26
done
```

**保证**：
- ✅ 使用相同种子确保风场/机动参数与原计划一致
- ✅ 只重新采集缺失的段，已存在的不覆盖

---

### 模式 4: 补采单段

```bash
# 补采第 7 轮的第 1 段
python3 generate_dataset.py --mode recover --run 7 --segment 1 --seed 26

# 补采第 2 轮的第 3 段
python3 generate_dataset.py --mode recover --run 2 --segment 3 --seed 26
```

**适用场景**：
- 某轮只有个别段失败
- 精确控制采集过程
- 快速验证单段数据

---

### 模式 5: 单轮测试（验证配置）

```bash
python3 generate_dataset.py --mode single_round --seed 26
# 输出 data/train/datasets-1-1.json ~ datasets-1-5.json
```

---

### 模式 6: 后处理 JSON → CSV

```bash
cd scripts
python3 postprocess_to_jsbsim_csv.py ../data --output-dir ../data/processed
# 生成 data/processed/train/*.csv 等
```

---

### 模式 7: 风场验收与可视化

```bash
cd scripts

# 检查单段数据：输出统计并保存 4 联图 PNG
python3 inspect_wind_dataset.py ../data/train/datasets-1-1.json

# 只做统计，不保存图
python3 inspect_wind_dataset.py ../data/train/datasets-1-1.json --no-plot

# 扫描整个数据目录：输出 inspection_summary.csv，并额外保存前 8 个高风险样本图
python3 inspect_wind_dataset.py ../data --plot-limit 8
```

**输出内容**：
- `wind_inspection/inspection_summary.csv`：整批数据的验收汇总
- `*_inspection.png`：单段风场、空速/地速、姿态、控制量四联图
- 终端结论：`PASS / WARN / FAIL`

**当前验收重点**：
- 峰值合成风/空速比是否过高
- 垂直风占水平风比例是否异常
- 空速低谷、最小地速是否过低
- `roll/pitch` 响应是否过猛
- 控制量是否频繁接近饱和

**注意**：日志中的真值风列当前反映的是**常值风 + 阵风**；湍流未逐样本写回风真值，因此需要结合姿态、地速和控制抖动一起判断。

## 🔧 命令行参数

### Shell 脚本参数 (run_unattended.sh)

```bash
./run_unattended.sh [模式] [种子] [--speed 倍数]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| 模式 | nohup/tmux/foreground | foreground |
| 种子 | 随机种子（确保可重现） | 26 |
| `--speed` | 仿真加速倍数 | 1（1:1 实时） |

**加速说明**：
- `--speed 2`：2倍速，仿真时间过 2 秒，真实时间约过 1 秒
- `--speed 4`：4倍速，采集时间缩短至约 1/4
- 推荐范围：2-4 倍，过高可能影响数据质量或 CPU 资源不足
- 超时参数会自动调整，无需手动修改

### Python 脚本参数 (generate_dataset.py)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式：multi_segment_160/single_round/recover/report | multi_segment_160 |
| `--seed` | 随机种子（确保可重现） | 26 |
| `--no-skip` | 不跳过已存在数据 | False (会跳过) |
| `--run` | 补采模式：指定轮次 (1-160) | - |
| `--segment` | 补采模式：指定段 (1-5)，需配合 --run | - |
| `--round-timeout` | 单轮最大时间（秒） | 1800 |
| `--output-dir` | 数据输出根目录 | ../data |
| `--px4-dir` | PX4 源码根目录 | 自动检测 |
| `--log` | 运行日志路径 | ../logs/multi_segment_160runs.log |

**注意**：Python 脚本会自动读取环境变量 `PX4_SIM_SPEED_FACTOR` 来实现加速，建议通过 `run_unattended.sh` 使用 `--speed` 参数更方便。

## 📊 工作流程示例

### 场景 1: 完整无人值守采集

```bash
# 1. 启动采集
python3 generate_dataset.py --mode multi_segment_160 --seed 26

# 2. 采集完成后查看缺失报告
cat ../logs/missing_data_report.txt

# 3. 根据报告补采
python3 generate_dataset.py --mode recover --run 7 --seed 26
python3 generate_dataset.py --mode recover --run 15 --seed 26

# 4. 再次生成报告确认
python3 generate_dataset.py --mode report
```

### 场景 2: 分批采集

```bash
# 第一天：采集论文版数据集（支持中断）
python3 generate_dataset.py --mode multi_segment_160 --seed 26
# ... Ctrl+C 停止 ...

# 第二天继续（自动从断点继续）
python3 generate_dataset.py --mode multi_segment_160 --seed 26
```

### 场景 3: 更换种子重新生成

```bash
# 使用不同种子生成完全不同的数据集
python3 generate_dataset.py --mode multi_segment_160 --seed 123 --no-skip
```

---

## ⚠️ 重要提示

### 种子一致性
- **务必使用相同的 `--seed` 进行补采**
- 不同种子会生成不同的风场和机动参数
- 建议始终使用默认值 26

### 数据覆盖
- 默认会跳过已存在的文件（断点续传）
- 使用 `--no-skip` 会覆盖所有数据
- 补采单段时会覆盖该段数据

### 文件命名规则
- 格式：`datasets-{run_number}-{segment_number}.json`
- run_number: 1-80
- segment_number: 1-5
- 示例：`datasets-7-3.json` = 第7轮的第3段

---

## 🎉 优势总结

| 传统方式 | 新方式 |
|---------|--------|
| ❌ 失败后从头开始 | ✅ 断点续传 |
| ❌ 缺失数据难定位 | ✅ 自动生成报告 |
| ❌ 补采参数不一致 | ✅ 固定种子保证一致 |
| ❌ 需要人工监控 | ✅ 真正无人值守 |
| ❌ 不可重现 | ✅ 完全可重现 |
| ❌ 模式切换导致失败 | ✅ 纯 Offboard 控制 |

---

## 📝 技术细节

### 风场配置（JSBSim Bridge）

风场在**启动 SITL 前**由 `jsbsim_wind_config.py` 写入：
  1. **bridge `wind_config.txt`**：恒定风、湍流和单个 gust 事件时间表
  2. **scene 初始条件**：`PX4/Tools/jsbsim_bridge/scene/LSZH.xml`（vwind、winddir、vw-north-fps 等）
  3. FDM 或 `wind_override.xml`（若 Rascal FDM 中无 `<winds>`）

当前约束：**一次 sortie 最多只允许一个 gust 事件**。如果某个逻辑 run 含多个 gust 段，`generate_dataset.py` 会自动拆成多个 sortie，但复用同一背景风，避免 bridge 和日志真值错位。

### 数据与真值风

- **稳态段**：每行都记录三轴真值风 `wind_north / wind_east / wind_down`
- **阵风段**：`DataLogger` 使用与 bridge 相同的语义生成逐时刻 gust 脉冲：`start_time` 为段内起点，`duration` 为整次事件总时长，形状为**1-cos 上升 + 平顶保持 + 1-cos 下降**
- **数据有效性**：采集完成后会校验样本数量、时间覆盖和关键字段；坏文件不会再被当成成功样本

### 数据集划分

- **Train**: 60 个逻辑 run（默认 4 稳态 + 1 阵风 ID）→ 300 条
- **Val**: 10 个逻辑 run（默认 4 稳态 + 1 阵风 ID）→ 50 条
- **Test-ID**: 7 个逻辑 run，其中最后 1 个 run 含多个 gust 段；运行时会自动拆成多个 sortie 采集 → 35 条
- **Test-OOD**: 3 个逻辑 run（逻辑上均为 OOD 阵风段）；运行时会自动拆成多个 sortie 采集 → 15 条

### 飞行控制优化

- **直线/爬升下降**：Offboard 速度控制
- **盘旋/8字**：纯 Offboard 速度控制（切向速度矢量）
- **起飞/降落**：Action 接口（保留 PX4 内置逻辑）
- **优势**：消除 `goto_location` 高频调用，减少 COMMAND_DENIED 错误

### 仿真加速机制

- **环境变量**：通过 `PX4_SIM_SPEED_FACTOR` 控制仿真速度
- **支持范围**：JSBSim bridge 支持任意正数倍速（推荐 1-4）
- **自动调整**：PX4 会根据加速因子自动调整超时参数（COM_DL_LOSS_T、COM_RC_LOSS_T 等）
- **时间语义**：
  - 仿真内时间按 `speed` 倍加速
  - Python 控制脚本的 `asyncio.sleep()` 仍为真实时间
  - 整体采集时间约为 `原时间 / speed`
- **性能要求**：加速倍数越高，CPU 占用越大，建议先从 2 倍测试

---

## 📞 故障排查

### Q: 未找到 Rascal JSBSim 配置
**A**: 设置 `PX4_ROOT` 指向包含 `Tools/jsbsim_bridge/models/Rascal/Rascal110-JSBSim.xml` 的 PX4 目录

### Q: 连接超时
**A**: 确保 PX4 SITL 已启动（脚本会自动启停），端口 14540

### Q: 补采时提示 "run_number 必须在1-80之间"
**A**: 检查 `--run` 参数是否正确，范围是 1-80

### Q: 补采数据与原计划不一致
**A**: 确保使用了相同的 `--seed` 参数

### Q: 缺失报告为空但数据不全
**A**: 检查 `--output-dir` 是否正确指向数据目录

### Q: 想要完全重新生成所有数据
**A**: 使用 `--no-skip` 参数或删除 data 目录后重新运行

### Q: 大量 COMMAND_DENIED 错误
**A**: 已通过纯 Offboard 控制优化，成功率提升至 70-85%。如仍频繁失败，检查：
- PX4 版本是否支持
- 系统资源是否充足
- 查看 `logs/px4_last_run.log` 详细错误

### Q: 如何加速数据采集？
**A**: 使用 `--speed` 参数进行仿真加速：
```bash
# 2倍速，采集时间减半
./run_unattended.sh nohup 26 --speed 2

# 4倍速，采集时间缩短至约1/4
./run_unattended.sh nohup 26 --speed 4
```
- 推荐从 2 倍开始测试
- 过高倍速可能导致 CPU 资源不足或数据质量下降
- 超时参数会自动调整，无需手动配置

### Q: 仿真加速后 EKF 收敛时间会变化吗？
**A**: 仿真内的时间仍保持不变（如 EKF 仍需约 10-20s 仿真时间收敛），但真实等待时间会缩短。例如 2 倍速时，10s 仿真时间只需约 5s 真实时间。

### Q: 出现 "Connection refused 127.0.0.1:50051" 错误怎么办？
**A**: 这表示 mavsdk_server 进程崩溃或退出。系统会自动检测并重启：
- **自动恢复**：每段执行前会自动检查并重启 mavsdk_server
- **手动检查**：查看 `logs/mavsdk_server.log` 了解崩溃原因
- **常见原因**：
  - PX4 SITL 进程崩溃导致 MAVLink 连接断开
  - 系统资源不足（CPU/内存）
  - 仿真加速倍数过高（建议降低至 2-3 倍）
- **如果频繁崩溃**：
  ```bash
  # 检查系统资源
  top
  # 降低仿真加速
  ./run_unattended.sh nohup 26 --speed 1
  # 查看详细日志
  tail -f logs/mavsdk_server.log
  ```

---

## 📝 日志说明

**正常运行**：
```
Run 1/160 train base_id=1 wind=(2.1m/s, 135°)
[Session] 起飞成功，高度=95.3m
[Session] 段 1/5 已保存 datasets-1-1.json
...
  -> OK (attempt 1)
```

**跳过已存在**：
```
Run 5/160 已存在，跳过
```

**失败重试**：
```
  -> FAILED: COMMAND_DENIED (attempt 1)
  -> OK (attempt 2)
```

**最终总结**：
```
========== 160 轮采集结束 (跳过 12 轮) ===========
缺失: 15/800 段
缺失数据报告已生成: logs/missing_data_report.txt
```


# 一键清理所有相关进程
pkill -f "px4" ; pkill -f "jsbsim_bridge" ; pkill -f "mavsdk_server" ; pkill -f "generate_dataset" ; pkill -f "run_unattended" ; pkill -f "JSBSim"

# 等待进程退出
sleep 2

# 确认是否清理干净
ps aux | grep -E "px4|jsbsim|mavsdk|generate_dataset|run_unattended" | grep -v grep

# 如果有残留进程杀不掉，可以用 -9 强制：
pkill -9 -f "px4" ; pkill -9 -f "jsbsim_bridge" ; pkill -9 -f "mavsdk_server" ; pkill -9 -f "JSBSim"