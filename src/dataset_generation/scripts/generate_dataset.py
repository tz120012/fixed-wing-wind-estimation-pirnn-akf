"""
generate_dataset.py
无人值守数据采集：论文版 160 轮 SITL，每轮 5 段，共 800 条。
支持 --mode multi_segment_160 全自动运行；单轮内多段共享同一背景风，动态阵风段会拆分为独立 sortie。
"""

import argparse
import asyncio
import copy
import datetime
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

# =============== gRPC / 代理隔离（必须在 grpc/mavsdk import 之前）===============
#
# 数据采集所有连接都是本地（PX4↔mavsdk_server↔Python，全部走 127.0.0.1），
# 不应经过任何 HTTP/HTTPS 代理。但 WSL 经常继承 Windows 系统代理（Clash/V2Ray
# 把 HTTP(S)_PROXY 指向 127.0.0.1:7897），观察到的故障：
#
#   AioRpcError: ipv4:127.0.0.1:7897: Socket closed   ← 走错代理端口
#
# 之前的修复仅追加 no_proxy=localhost,127.0.0.1,::1。但实测仍有 3 个回退风险：
#   1) Windows 风格的 NO_PROXY 含 "<local>" 等非法 token，老版本 gRPC C-core
#      解析失败时会忽略整个 NO_PROXY；
#   2) "127.*" 通配符在旧版 gRPC 不被支持；
#   3) HTTP_PROXY 仍会被 mavsdk_server 子进程继承。
#
# 因此这里采用三重保险：① 显式空字符串覆盖 HTTP_PROXY 等 ② gRPC 专用 env
# ③ 重写 no_proxy 为干净 token（不再依赖系统继承值）。
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
# ① 把所有 HTTP(S) 代理变量在本进程及其子进程中强制清空。
for _proxy_var in (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
):
    if os.environ.get(_proxy_var):
        os.environ[_proxy_var] = ""
# ② gRPC 自身识别的代理控制变量：grpc_proxy 留空 + no_grpc_proxy 通配。
os.environ["grpc_proxy"] = ""
os.environ["no_grpc_proxy"] = "*"
# ③ 用干净 token 重写 no_proxy（移除 "<local>"、"127.*" 等可能让 gRPC 解析
# 失败的非标准片段；显式列出本机所有可能形态）。
_clean_no_proxy = "127.0.0.1,localhost,::1,0.0.0.0"
os.environ["no_proxy"] = _clean_no_proxy
os.environ["NO_PROXY"] = _clean_no_proxy

import numpy as np

# 脚本同目录下导入
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
# 让 `lib.*` 包可被导入：把 dataset_generation 根加到 sys.path
_DSGEN_ROOT = SCRIPT_DIR.parent
if str(_DSGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_DSGEN_ROOT))

try:
    from flight_controller import FlightController
    _FLIGHT_CONTROLLER_IMPORT_ERROR = None
except Exception as e:
    FlightController = None
    _FLIGHT_CONTROLLER_IMPORT_ERROR = e

from data_logger import DataLogger
from jsbsim_wind_config import (
    get_jsbsim_rascal_xml_path,
    get_jsbsim_bridge_root,
    wind_from_speed_direction,
    write_jsbsim_bridge_wind_config,
    set_aircraft_initial_heading,
)

# Phase 1: 纯函数迁移到 lib/。这里以 re-export 形式保留模块级名字，向后兼容。
from lib.config.dataset import (
    DEFAULT_DATASET_CONFIG,
    deep_merge_dict as _deep_merge_dict,
    get_split_profile as _get_split_profile,
    load_dataset_config as _load_dataset_config,
)
from lib.config.runtime import load_runtime_config
from lib.planner.schedule import (
    SEGMENTS_PER_RUN,
    build_paper_run_schedule,
    build_80_run_schedule,
)
from lib.planner.wind_factory import (
    generate_wind_config,
    sample_gust_start_time as _sample_gust_start_time,
)
from lib.planner.segment_factory import (
    derive_turn_labels,
    generate_one_segment_config,
    generate_segment_configs_for_run,
    split_segment_configs_for_sorties as _split_segment_configs_for_sorties,
    _TURNING_MANEUVERS,
    _NON_TURNING_MANEUVERS,
)
# Phase 2: infra 抽离。委托 PX4 / mavsdk_server / bridge 到独立 async ctx mgr。
from lib.infra import (
    Px4SitlProcess,
    MavsdkServerProcess,
    JsbsimBridgeConfig,
    find_latest_wind_truth_csv as _infra_find_latest_wind_truth_csv,
    clear_px4_lock_files,
    clear_px4_rootfs_state as _infra_clear_px4_rootfs_state,
)
from lib.errors import (
    SitlStartFailed, SitlCrashed, MavsdkServerDead, JsbsimBridgeMissing, GpsTimeout,
    TakeoffFailed, FlightControllerError,
)
from lib.obs import configure_logging, get_logger as _get_struct_logger
from lib.obs.failure import from_exception as _failure_from
from lib.validation import (
    REQUIRED_LOG_FIELDS,
    cleanup_segment_outputs as _cleanup_segment_outputs,
    is_finite_number as _is_finite_number,
    segment_metadata_path as _segment_metadata_path,
    segment_output_is_valid,
    validate_airspeed_filled_ratio,
    validate_segment_records,
)
from lib import recovery as _recovery


def find_latest_wind_truth_csv():
    """向后兼容包装：在 jsbsim_bridge 目录下查找最新的 wind_truth_*.csv。"""
    try:
        bridge_root = get_jsbsim_bridge_root()
    except Exception as e:
        print(f"[generate] 查找 wind_truth CSV 失败: {e}")
        return None
    return _infra_find_latest_wind_truth_csv(Path(bridge_root))


def clear_px4_rootfs_state():
    """向后兼容包装：清理 PX4 SITL rootfs 中的持久化状态。"""
    _infra_clear_px4_rootfs_state(get_px4_root())


def require_mavsdk_runtime():
    """在真正需要飞控/MAVSDK 时再强制检查依赖，允许 report 模式脱离 mavsdk 运行。"""
    if FlightController is None:
        raise RuntimeError(
            "MAVSDK Python 运行环境不可用（未安装 mavsdk，或 grpcio / mavsdk 版本不兼容）。"
            "请先执行 `python3 -m pip install --user --upgrade grpcio mavsdk numpy`，"
            "或切换到依赖已正确安装的 Python 环境。"
        ) from _FLIGHT_CONTROLLER_IMPORT_ERROR

    return FlightController


# clear_px4_lock_files / clear_px4_rootfs_state / get_mavsdk_server_path /
# start_mavsdk_server / check_mavsdk_server_alive 已迁移到 lib/infra/。
# 仍以向后兼容的方式可用：
#   - clear_px4_lock_files: 直接 from lib.infra import 顶部已 re-export
#   - clear_px4_rootfs_state: 上面薄包装已就位
#   - mavsdk_server 相关由 MavsdkServerProcess 内部处理
from lib.infra.mavsdk_server import find_mavsdk_server_executable as get_mavsdk_server_path  # noqa: E402


def get_px4_root():
    if os.environ.get("PX4_ROOT"):
        return Path(os.environ["PX4_ROOT"])
    for rel in [
        "wind_datasets/PX4-Autopilot-v133",
        "wind_datasets/PX4-Autopilot",
        "PX4-Autopilot-v133",
        "PX4-Autopilot",
    ]:
        p = Path.home() / rel
        if p.exists():
            return p
    return Path.home() / "PX4-Autopilot"


# ---------- 计划表 / 数据集配置 / 风场段配置 ----------
# 已迁移到 lib/planner/、lib/config/dataset.py。
# 顶层 import 区已 re-export 这些符号（SEGMENTS_PER_RUN / build_paper_run_schedule /
# build_80_run_schedule / DEFAULT_DATASET_CONFIG / generate_wind_config /
# generate_one_segment_config / generate_segment_configs_for_run /
# _split_segment_configs_for_sorties / _sample_gust_start_time / _deep_merge_dict /
# _get_split_profile / _load_dataset_config / derive_turn_labels）。
# 旧的本地实现已删除以避免重复维护与意外覆盖。

# Phase 3: 质量门已迁移到 lib/validation/。下面是向后兼容包装，保持原本签名不变。
# REQUIRED_LOG_FIELDS、_is_finite_number、_segment_metadata_path、_cleanup_segment_outputs
# 均通过顶部 import 直接 re-export。MIN_VALID_LOG_HZ 改为读 runtime.yaml。
MIN_VALID_LOG_HZ = load_runtime_config().quality_gates.min_valid_log_hz


def _validate_logged_records(records, requested_duration):
    """向后兼容包装：返回 ``(passed, issues)`` 元组（与原签名一致）。"""
    rc = load_runtime_config()
    res = validate_segment_records(records, requested_duration, rc)
    return res.passed, res.issues


def _segment_output_is_valid(data_path):
    """向后兼容包装。"""
    return segment_output_is_valid(Path(data_path), load_runtime_config())


class DatasetGenerator:
    """数据集采集编排器（Phase 2 起委托基础设施给 lib.infra）。"""

    def __init__(self, output_dir=None, px4_dir=None, airframe="rascal"):
        self.base_dir = Path(output_dir or (SCRIPT_DIR.parent / "data"))
        self.output_dir = self.base_dir
        self.px4_dir = Path(px4_dir or get_px4_root())
        self.airframe = str(airframe).strip().lower()
        self.fc = None
        self.jsbsim_xml_path = None
        self.runtime = load_runtime_config()
        self._px4_ctx = None      # Optional[Px4SitlProcess]
        self._mavsdk_ctx = None   # Optional[MavsdkServerProcess]
        # 启用结构化日志（脚本入口的副作用：让所有 lib.* 模块共享同一 sink）
        configure_logging(
            runtime_jsonl=self.runtime.logging.runtime_jsonl_path,
            failure_jsonl=self.runtime.logging.failure_jsonl_path,
            console_level=self.runtime.logging.console_level,
            base_dir=str(SCRIPT_DIR.parent),
        )
        self._log = _get_struct_logger("dataset_generator")

    # ---------- 兼容旧字段：以 property 形式暴露当前 PX4 / mavsdk_server 进程 ----------
    @property
    def px4_process(self):
        """已运行的 PX4 SITL ``subprocess.Popen``；未启动时为 ``None``。"""
        return self._px4_ctx.proc if self._px4_ctx is not None else None

    @property
    def _mavsdk_server_process(self):
        """已运行的 mavsdk_server ``subprocess.Popen``；未启动时为 ``None``。"""
        return self._mavsdk_ctx.proc if self._mavsdk_ctx is not None else None

    @property
    def _px4_log_file(self):
        """PX4 SITL 重定向日志文件句柄（read-only 兼容字段）。"""
        return self._px4_ctx.log_file if self._px4_ctx is not None else None

    # ---------- PX4 SITL 生命周期（委托给 Px4SitlProcess） ----------
    async def start_px4_sitl(self, redirect_log_to=None):
        """启动 PX4 SITL + JSBSim Rascal。"""
        if self._px4_ctx is not None and await self._px4_ctx.is_alive():
            self._log.warning("start_px4_sitl 被重复调用，先停止旧实例")
            await self.stop_px4_sitl()
        log_target = (
            Path(redirect_log_to)
            if redirect_log_to
            else Path(__file__).resolve().parents[1] / "logs" / "px4_last_run.log"
        )
        self._px4_ctx = Px4SitlProcess(
            px4_root=self.px4_dir,
            runtime=self.runtime,
            redirect_log_to=log_target,
            airframe=self.airframe,
        )
        try:
            await self._px4_ctx.start()
        except (SitlStartFailed, JsbsimBridgeMissing) as e:
            self._log.failure(_failure_from(e))
            # 保持原行为：不向上传播，让外层 GPS 验证 / 后续重试逻辑兜底
            print(f"[DatasetGenerator] 警告: {e}")

    async def _verify_jsbsim_running(self, max_retries=3):
        """向后兼容：委托 ``Px4SitlProcess.verify_bridge_ready``。"""
        if self._px4_ctx is None:
            return False
        return await self._px4_ctx.verify_bridge_ready(retries=max_retries)

    async def check_jsbsim_health(self):
        """向后兼容：委托 ``Px4SitlProcess.check_jsbsim_health``。"""
        if self._px4_ctx is None:
            return False
        return await self._px4_ctx.check_jsbsim_health()

    async def stop_px4_sitl(self):
        """停止 PX4 SITL（同时收尾 mavsdk_server 子进程）。"""
        # 先停 mavsdk_server，避免 PX4 退出时 mavsdk 客户端 hang 住
        if self._mavsdk_ctx is not None:
            await self._mavsdk_ctx.stop()
            self._mavsdk_ctx = None
        if self._px4_ctx is not None:
            await self._px4_ctx.stop()
            self._px4_ctx = None

    async def initialize_controllers(self):
        """连接 MAVSDK：启动/重启 mavsdk_server，再让 FlightController 连接。"""
        FlightControllerCls = require_mavsdk_runtime()

        if self.fc is not None:
            await self.fc.disconnect()
            await asyncio.sleep(self.runtime.sleeps.between_segments_s + 3)

        # 重启 mavsdk_server（旧的若存在会被 stop()）
        if self._mavsdk_ctx is not None:
            await self._mavsdk_ctx.stop()
        self._mavsdk_ctx = MavsdkServerProcess(
            runtime=self.runtime,
            log_dir=Path("logs"),
        )
        try:
            await self._mavsdk_ctx.start()
        except MavsdkServerDead as e:
            self._log.failure(_failure_from(e))
            # 回退到 SDK 自启
            self._mavsdk_ctx = None

        if self._mavsdk_ctx is not None and self._mavsdk_ctx.is_running():
            self.fc = FlightControllerCls(
                mavsdk_server_address="127.0.0.1",
                ned_altitude_sign=0.0 if self.airframe == "malolo" else -1.0,
            )
            await self.fc.connect()
        else:
            self.fc = FlightControllerCls(
                ned_altitude_sign=0.0 if self.airframe == "malolo" else -1.0
            )
            await self.fc.connect(system_address=f"udpin://0.0.0.0:{self.runtime.ports.mavsdk_udp}")

        # 应用 PX4 关键参数（统一在 _apply_px4_params 中维护）
        await self._apply_px4_params(reconnect=False)

        # 给 EKF 额外时间预热（让 GPS/IMU/气压计数据流稳定）
        self._log.info("等待 JSBSim 完全初始化", wait_s=self.runtime.sleeps.ekf_warmup_s)
        await asyncio.sleep(self.runtime.sleeps.ekf_warmup_s)

        # 验证 GPS 数据是否正常
        await self._verify_gps_data()
        self._log.info("控制器已初始化")

    async def _verify_gps_data(self, max_retries=None):
        """验证 GPS 数据有效（坐标在范围内、未未初始化），支持多次重试。

        失败后抛 ``GpsTimeout``（替换原 ``RuntimeError``，便于上层分类捕获）。
        若订阅本身异常（不是数据无效），仅记 warning 不阻断（保持原行为）。
        """
        if max_retries is None:
            max_retries = self.runtime.retries.gps_verify
        INT32_MIN = -2147483648

        async def _get_gps():
            async for gps in self.fc.drone.telemetry.raw_gps():
                return gps

        last_gps = None
        for attempt in range(max_retries):
            try:
                gps = await asyncio.wait_for(
                    _get_gps(), timeout=self.runtime.timeouts.gps_subscribe_s
                )
            except asyncio.TimeoutError as e:
                self._log.warning(
                    "GPS 订阅超时", attempt=attempt + 1, retries=max_retries,
                    timeout_s=self.runtime.timeouts.gps_subscribe_s,
                )
                if attempt == max_retries - 1:
                    self._log.failure(_failure_from(
                        GpsTimeout("raw_gps 订阅持续超时", context={"attempts": max_retries}),
                    ))
                    return
                await asyncio.sleep(3 * (attempt + 1))
                continue
            except Exception as e:
                # 订阅本身异常（如 mavsdk 内部错误）：保持旧行为不阻断
                self._log.warning("GPS 验证出现异常，继续执行", error=str(e))
                return

            last_gps = gps
            is_uninitialized = (
                gps.latitude_deg == INT32_MIN or gps.longitude_deg == INT32_MIN
            )
            is_out_of_range = abs(gps.latitude_deg) > 90 or abs(gps.longitude_deg) > 180

            alt = (
                getattr(gps, "altitude_mmsl_m", None)
                or getattr(gps, "altitude_ellipsoid_m", None)
                or getattr(gps, "alt", None)
            )

            if not (is_uninitialized or is_out_of_range):
                self._log.info(
                    "GPS 数据正常",
                    lat=round(gps.latitude_deg, 6),
                    lon=round(gps.longitude_deg, 6),
                )
                return

            wait_time = 3 * (attempt + 1)
            self._log.warning(
                "GPS 数据异常，准备重试",
                attempt=attempt + 1, retries=max_retries,
                lat=gps.latitude_deg, lon=gps.longitude_deg, alt=alt,
                wait_s=wait_time,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(wait_time)

        # 重试次数耗尽
        ctx = {"attempts": max_retries}
        if last_gps is not None:
            ctx.update({"lat": last_gps.latitude_deg, "lon": last_gps.longitude_deg})
        raise GpsTimeout(
            "GPS 数据持续无效（未初始化或超界），需要重启 SITL",
            context=ctx,
        )

    async def check_and_restart_mavsdk_server(self):
        """检查 mavsdk_server 健康；崩溃时尝试重启并重连 FlightController。"""
        # 1) 进程存活性检查
        is_alive = self._mavsdk_ctx is not None and self._mavsdk_ctx.is_running()
        if not is_alive:
            self._log.warning("检测到 mavsdk_server 进程已退出，尝试重启")
            # 重启 mavsdk_server
            if self._mavsdk_ctx is not None:
                await self._mavsdk_ctx.stop()
            self._mavsdk_ctx = MavsdkServerProcess(runtime=self.runtime, log_dir=Path("logs"))
            try:
                await self._mavsdk_ctx.start()
            except MavsdkServerDead as e:
                self._log.failure(_failure_from(e))
                self._mavsdk_ctx = None
                return False

            # 重连 FlightController + 重设关键参数
            try:
                if self.fc is not None:
                    await self.fc.disconnect()
                await asyncio.sleep(self.runtime.sleeps.pkill_grace_s)

                FlightControllerCls = require_mavsdk_runtime()
                self.fc = FlightControllerCls(
                    mavsdk_server_address="127.0.0.1",
                    ned_altitude_sign=(
                        0.0 if self.airframe == "malolo" else -1.0
                    ),
                )
                await self.fc.connect()
                await self._apply_px4_params(reconnect=True)
                self._log.info("FlightController 已重新连接并配置")
                return True
            except Exception as e:
                self._log.error("FlightController 重连失败", error=str(e))
                return False

        # 2) 进程存活，检查连接健康
        if self.fc is not None and self.fc.is_connected:
            is_healthy = await self.fc.check_connection_health()
            if not is_healthy:
                self._log.warning("MAVSDK 连接不健康，触发重启")
                # 标记 ctx 为 dead，递归走重启路径
                if self._mavsdk_ctx is not None:
                    await self._mavsdk_ctx.stop()
                    self._mavsdk_ctx = None
                return await self.check_and_restart_mavsdk_server()

        return True

    async def _apply_px4_params(self, reconnect: bool = False) -> None:
        """统一设置一组 PX4 关键参数（避免 init / restart 各处重复）。

        每个参数失败仅 warning，不阻塞流程（与原行为一致）。
        """
        params_int = [
            ("COM_RCL_EXCEPT", 7),    # 禁用 RC 丢失 failsafe
            ("NAV_RCL_ACT", 0),
            ("COM_RC_IN_MODE", 4),    # SITL: disable manual-control input checks
            ("COM_ARM_WO_GPS", 1),    # allow arming while simulated GPS settles
            ("EKF2_GPS_CHECK", 0),    # SITL 加速 EKF 收敛
            ("COM_OBL_ACT", -1),      # 禁用 offboard 丢失 failsafe 动作
            ("ASPD_DO_CHECKS", 0),    # 禁用空速健康检查（防 SITL airspeed_selector 误报）
            ("SYS_HAS_NUM_ASPD", 1),  # 显式告知 PX4 有 1 个空速管，防止 eeprom 清空后 selector 未激活
        ]
        params_float = [
            ("COM_POS_FS_EPH", 100.0),
            ("COM_POS_FS_EPV", 100.0),
            ("COM_VEL_FS_EVH", 10.0),
            ("COM_OF_LOSS_T", 10.0),
            ("COM_LKDOWN_TKO", 0.0),
        ]
        if self.airframe == "malolo":
            # The replacement electric powerplant otherwise drives this light
            # model beyond the tabulated aerodynamic envelope (~37 m/s).
            params_float.extend([
                ("FW_THR_MAX", 0.65),
                ("FW_AIRSPD_MAX", 22.0),
            ])
        for name, val in params_int:
            try:
                await self.fc.drone.param.set_param_int(name, val)
            except Exception as e:
                self._log.warning(
                    "set_param_int 失败", name=name, value=val, error=str(e),
                )
        for name, val in params_float:
            try:
                await self.fc.drone.param.set_param_float(name, val)
            except Exception as e:
                self._log.warning(
                    "set_param_float 失败", name=name, value=val, error=str(e),
                )
        suffix = "（重连后重设）" if reconnect else ""
        self._log.info(f"PX4 关键参数已配置{suffix}")

    async def _prepare_sortie_environment(self, wind_config, segment_configs, log_fn=None):
        """统一的 sortie 启动流程：写风场 -> 启动 SITL -> 连接控制器 -> 起飞。

        SITL 偶发故障（jsbsim 仿真停滞 / PX4 TECS 异常 / NED 流卡死）通过重试整套
        启动流程恢复。每次重试都会 stop_px4_sitl + 清 rootfs + 重新启动，
        最多尝试 ``runtime.retries.sortie_prepare`` 次（默认 3）。
        """
        _log = log_fn or print
        max_attempts = max(1, int(self.runtime.retries.sortie_prepare))
        last_err = None

        for attempt in range(1, max_attempts + 1):
            try:
                await self._prepare_sortie_environment_once(
                    wind_config, segment_configs, log_fn=_log,
                )
                if attempt > 1:
                    _log(f"[Session] sortie 准备已在第 {attempt}/{max_attempts} 次尝试成功")
                return
            except (TakeoffFailed, FlightControllerError) as e:
                last_err = e
                _log(
                    f"[Session] sortie 准备失败（{type(e).__name__}, "
                    f"尝试 {attempt}/{max_attempts}）: {e}"
                )
                self._log.warning(
                    "sortie_prepare_failed",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error_type=type(e).__name__,
                    error=str(e),
                    context=getattr(e, "context", {}) or {},
                )
                if attempt >= max_attempts:
                    break
                _log(
                    f"[Session] {self.runtime.sleeps.segment_retry_backoff_s}s 后清理 SITL "
                    f"并重试 sortie 准备..."
                )
                try:
                    await self.stop_px4_sitl()
                except Exception as cleanup_e:
                    _log(f"[Session] 警告: stop_px4_sitl 异常已忽略: {cleanup_e}")
                if getattr(self, "fc", None) is not None:
                    try:
                        await self.fc.disconnect()
                    except Exception:
                        pass
                await asyncio.sleep(self.runtime.sleeps.segment_retry_backoff_s)

        assert last_err is not None
        raise last_err

    async def _prepare_sortie_environment_once(self, wind_config, segment_configs, log_fn=None):
        """单次 sortie 准备实现（不含重试逻辑）。"""
        _log = log_fn or print

        await self.stop_px4_sitl()
        await asyncio.sleep(self.runtime.sleeps.pre_sortie_warmup_s)
        clear_px4_rootfs_state()

        try:
            write_jsbsim_bridge_wind_config(
                wind_config,
                segment_configs,
                wind_delay_alt_m=50.0,
            )
        except Exception as e:
            _log(f"[DatasetGenerator] 错误: 写入 bridge wind_config 失败: {e}")
            raise

        _log("[DatasetGenerator] 风场已就绪，正在启动 PX4 SITL")
        await self.start_px4_sitl(redirect_log_to=getattr(self, "_px4_log_path", None))
        await self.initialize_controllers()

        altitude = segment_configs[0].get("altitude", 100)
        try:
            await self.fc.arm_and_takeoff(
                altitude=altitude,
                takeoff_timeout_s=self.runtime.timeouts.takeoff_total_s,
                airspeed_ready_timeout_s=self.runtime.timeouts.airspeed_ready_s,
                airspeed_stable_s=self.runtime.timeouts.airspeed_stable_s,
                position_stream_max_failures=self.runtime.timeouts.position_stream_max_failures,
            )
        except (TakeoffFailed, FlightControllerError) as e:
            _log(f"[Session] 起飞失败（{type(e).__name__}）: {e}")
            self._log.warning(
                "takeoff_failed",
                error_type=type(e).__name__,
                error=str(e),
                context=getattr(e, "context", {}) or {},
            )
            raise
        _log(f"[Session] 起飞成功，高度={altitude:.1f}m")

    async def _cleanup_after_sortie(self, log_fn=None, land_first=True):
        """统一的 sortie 收尾流程，避免异常时残留 PX4/MAVSDK 进程。"""
        _log = log_fn or print
        try:
            if land_first and getattr(self, "fc", None) is not None and self.fc.is_connected:
                await self.fc.land()
                _log("[Session] 降落完成，本 sortie 成功")
        except Exception as e:
            _log(f"[Session] 警告: 降落阶段异常: {e}")
        finally:
            await self.stop_px4_sitl()
            if getattr(self, "fc", None) is not None:
                try:
                    await self.fc.disconnect()
                except Exception:
                    pass
            await asyncio.sleep(self.runtime.sleeps.cleanup_post_stop_s)

    async def execute_single_segment(self, segment_config, output_file):
        """
        执行单段机动并记录，不起飞不降落。
        segment_config 须已含 wind_north/wind_east/wind_down（本轮统一）。
        """
        # 在开始执行前检查 mavsdk_server 健康状态
        is_healthy = await self.check_and_restart_mavsdk_server()
        if not is_healthy:
            raise RuntimeError("mavsdk_server 不健康且无法恢复，终止段执行")
        
        # 检查 JSBSim 健康状态
        if not await self.check_jsbsim_health():
            print("[DatasetGenerator] 警告: jsbsim_bridge 进程已退出，GPS 数据可能无效")
            print("[DatasetGenerator] 尝试重启整个 SITL 环境...")
            raise RuntimeError("jsbsim_bridge 进程已退出，需要重启 SITL")
        
        # 定位本 sortie 对应的 wind_truth CSV（bridge 启动时创建，名含时间戳）
        wind_truth_path = find_latest_wind_truth_csv()
        if wind_truth_path:
            print(f"[DatasetGenerator] wind_truth CSV: {Path(wind_truth_path).name}")
        else:
            print("[DatasetGenerator] 警告: 未找到 wind_truth CSV，将使用解析式真值风（不含湍流）")

        # 从机动类型派生转弯标签（由机动计划确定，不依赖实时状态）
        maneuver_type = segment_config["maneuver_type"]
        turn_state, turn_class = derive_turn_labels(maneuver_type)

        logger = DataLogger(
            self.fc.drone,
            wind_north=segment_config["wind_north"],
            wind_east=segment_config["wind_east"],
            wind_down=segment_config.get("wind_down", 0.0),
            gust_params=segment_config.get("gust"),
            wind_truth_csv_path=wind_truth_path,
            turn_state=turn_state,
            turn_class=turn_class,
            jsbsim_telnet_port=self.runtime.ports.jsbsim_telnet,
        )
        duration = segment_config["duration"]
        logger.set_maneuver(maneuver_type)
        altitude = segment_config.get("altitude", 100)

        if maneuver_type == "straight_line":
            logging_task = asyncio.create_task(logger.start_logging(duration, output_file))
            flight_task = asyncio.create_task(
                self.fc.fly_straight_line(
                    heading=segment_config["heading"],
                    altitude=altitude,
                    speed=segment_config["speed"],
                    duration=duration,
                )
            )
        elif maneuver_type == "orbit":
            logging_task = asyncio.create_task(logger.start_logging(duration, output_file))
            flight_task = asyncio.create_task(
                self.fc.fly_orbit(
                    radius=segment_config["radius"],
                    altitude=altitude,
                    direction=segment_config.get("direction", "cw"),
                    duration=duration,
                )
            )
        elif maneuver_type == "figure_eight":
            logging_task = asyncio.create_task(logger.start_logging(duration, output_file))
            flight_task = asyncio.create_task(
                self.fc.fly_figure_eight(
                    lobe_radius=segment_config["radius"],
                    orientation=segment_config["heading"],
                    altitude=altitude,
                    duration=duration,
                )
            )
        elif maneuver_type == "climb_descent":
            # climb_descent：先飞爬升，等到目标高度后才启动 DataLogger，
            # 避免爬升/下降过程中的非稳态数据进入训练集。
            altitude_reached = asyncio.Event()
            h_end = segment_config.get("target_altitude", altitude + 20)
            climb_rate = segment_config.get("climb_rate", 1.5)

            climb_time_est = abs(h_end - altitude) / max(abs(climb_rate), 0.1)
            level_duration = max(duration - climb_time_est, 10.0)

            flight_task = asyncio.create_task(
                self.fc.fly_climb_descent(
                    h_start=altitude,
                    h_end=h_end,
                    climb_rate=climb_rate,
                    heading=segment_config["heading"],
                    duration=duration,
                    altitude_reached_event=altitude_reached,
                )
            )

            # 等待目标高度事件：理论上应在 climb_time_est 内触发，
            # 给 grace_s 容忍 NED 流延迟；若仍未触发说明 fly_climb_descent
            # 中的位置流断了，应及早降级到时间兜底，避免空等整段时长。
            grace_s = float(self.runtime.timeouts.climb_descent_event_grace_s)
            event_timeout = min(duration, climb_time_est + grace_s)
            try:
                await asyncio.wait_for(altitude_reached.wait(), timeout=event_timeout)
                print(
                    f"[DatasetGenerator] 目标高度已达到，开始记录平飞阶段"
                    f"（约 {level_duration:.1f}s）"
                )
            except asyncio.TimeoutError:
                # 兜底：用估算爬升时间 + grace 之后剩余的整段时长，且至少占段时长 60%
                # （与 quality_gates.min_coverage_ratio 对齐，保证不必然 coverage_too_low）。
                min_ratio = self.runtime.quality_gates.min_coverage_ratio
                level_duration = max(
                    duration - event_timeout,
                    duration * min_ratio + 1.0,
                )
                print(
                    f"[DatasetGenerator] 警告: 爬升事件 {event_timeout:.1f}s 内未触发，"
                    f"按 {level_duration:.1f}s 记录平飞段（min_coverage_ratio={min_ratio}）"
                )
                self._log.warning(
                    "climb_descent_event_timeout",
                    event_timeout_s=event_timeout,
                    fallback_level_duration_s=level_duration,
                    min_coverage_ratio=min_ratio,
                )

            logging_task = asyncio.create_task(logger.start_logging(level_duration, output_file))
        else:
            logging_task = asyncio.create_task(logger.start_logging(duration, output_file))
            flight_task = asyncio.create_task(asyncio.sleep(duration))

        try:
            _, logging_summary = await asyncio.gather(flight_task, logging_task)
        except Exception:
            # asyncio.gather 默认不会取消其它任务，手动取消防止残留后台任务
            for t in (flight_task, logging_task):
                if not t.done():
                    t.cancel()
            # 等待取消完成，防止 "was destroyed but it is pending" 警告
            await asyncio.gather(flight_task, logging_task, return_exceptions=True)
            _cleanup_segment_outputs(output_file)
            raise

        gate_result = validate_segment_records(logger.data_buffer, duration, self.runtime)
        is_valid = gate_result.passed
        issues = list(gate_result.issues)
        # 额外质量门：airspeed 前向填充比例（阈值见 runtime.yaml: quality_gates.airspeed_filled_ratio_threshold）
        filled_ratio = logging_summary.get("filled_invalid_airspeed_ratio", 0.0)
        airspeed_failure = validate_airspeed_filled_ratio(filled_ratio, self.runtime)
        if airspeed_failure is not None:
            issues.append(airspeed_failure.message)
            self._log.failure(airspeed_failure)
            is_valid = False
        # 把所有失败结构化输出到 segment_failures.jsonl，便于后续分析
        for f in gate_result.failures:
            self._log.failure(f)
        if not is_valid:
            _cleanup_segment_outputs(output_file)
            raise RuntimeError("采样结果无效: " + "; ".join(issues))

        metadata_file = output_file.replace(".json", "_metadata.json")
        metadata = copy.deepcopy(segment_config)
        metadata["logging_summary"] = logging_summary
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    async def run_multi_segment_session(
        self, wind_config, segment_configs, output_subdir, base_flight_id, run_number=None, log_fn=None
    ):
        """
        单轮逻辑采集入口。
        当一轮中包含多个阵风段时，会自动拆成多个 sortie，以保证 bridge 与日志真值始终只对应一个 gust 事件。
        """
        _log = log_fn or print
        if self.airframe == "malolo":
            # The bundled Malolo model needs a sustained runway climb.  A
            # 60--70 m operating band clears the 40 m safety floor while
            # remaining reachable inside the simulator's 120 s takeoff limit.
            # This flight-envelope adaptation does not alter the frozen
            # estimator weights, scaler or decision thresholds.
            for segment in segment_configs:
                segment["altitude"] = float(np.clip(
                    segment.get("altitude", 65.0), 60.0, 70.0
                ))
                if "target_altitude" in segment:
                    segment["target_altitude"] = float(np.clip(
                        segment["target_altitude"], 60.0, 80.0
                    ))
                    # Clipping can move the target across the clipped start
                    # altitude; preserve the intended vertical direction.
                    rate = abs(float(segment.get("climb_rate", 1.2)))
                    segment["climb_rate"] = (
                        rate
                        if segment["target_altitude"] >= segment["altitude"]
                        else -rate
                    )
                if segment.get("dataset_type") == "test_ood":
                    # Strong-wind Malolo turns can exceed this small airframe's
                    # controllable envelope.  Preserve the OOD wind/gust shift
                    # while using a reproducible straight-line trajectory.
                    segment["maneuver_type"] = "straight_line"
                    gust = segment.get("gust")
                    if gust:
                        gust["magnitude"] = min(
                            float(gust["magnitude"]), 2.5
                        )
                segment["airframe"] = "Malolo"
            if any(
                segment.get("dataset_type") == "test_ood"
                for segment in segment_configs
            ):
                horizontal = float(np.hypot(
                    wind_config["wind_north"],
                    wind_config["wind_east"],
                ))
                if horizontal > 3.5:
                    scale = 3.5 / horizontal
                    wind_config["wind_north"] *= scale
                    wind_config["wind_east"] *= scale
                    wind_config["wind_speed"] = 3.5
        sortie_groups = _split_segment_configs_for_sorties(segment_configs)
        if len(sortie_groups) > 1:
            _log(
                f"[Session] 检测到本轮包含多个阵风段，已拆分为 {len(sortie_groups)} 次 sortie，"
                "各 sortie 复用同一背景风。"
            )

        for sortie_idx, group in enumerate(sortie_groups, start=1):
            group_segment_indices = [idx for idx, _ in group]
            sortie_segment_configs = [seg for _, seg in group]
            _log(
                f"[Session] 准备 sortie {sortie_idx}/{len(sortie_groups)}，"
                f"覆盖逻辑段 {[idx + 1 for idx in group_segment_indices]}"
            )
            try:
                await self._prepare_sortie_environment(
                    wind_config, sortie_segment_configs, log_fn=_log,
                )
            except (TakeoffFailed, FlightControllerError) as e:
                _log(
                    f"[Session] sortie {sortie_idx}/{len(sortie_groups)} 准备最终失败，"
                    f"跳过本 sortie 覆盖的 {len(group)} 段（已采集段不受影响）：{e}"
                )
                self._log.error(
                    "sortie_skipped",
                    sortie_index=sortie_idx,
                    sortie_count=len(sortie_groups),
                    skipped_segments=[idx + 1 for idx in group_segment_indices],
                    error_type=type(e).__name__,
                    error=str(e),
                )
                try:
                    await self.stop_px4_sitl()
                except Exception:
                    pass
                if getattr(self, "fc", None) is not None:
                    try:
                        await self.fc.disconnect()
                    except Exception:
                        pass
                await asyncio.sleep(self.runtime.sleeps.between_sorties_s)
                continue

            try:
                for original_idx, seg_cfg in group:
                    logical_segment_index = int(
                        seg_cfg.get("_original_segment_index", original_idx + 1)
                    )
                    seg_cfg["wind_north"] = wind_config["wind_north"]
                    seg_cfg["wind_east"] = wind_config["wind_east"]
                    seg_cfg["wind_down"] = wind_config.get("wind_down", 0.0)
                    seg_cfg["sortie_index"] = sortie_idx
                    seg_cfg["sortie_count"] = len(sortie_groups)
                    seg_cfg["segment_index_in_run"] = logical_segment_index

                    if seg_cfg.get("gust"):
                        _log(
                            "[JSBSim] 阵风已添加: "
                            f"start={seg_cfg['gust']['start_time']:.1f}s, "
                            f"duration={seg_cfg['gust']['duration']:.1f}s"
                        )

                    if run_number is not None:
                        filename = (
                            f"datasets-{run_number}-{logical_segment_index}.json"
                        )
                    else:
                        filename = f"flight_{base_flight_id + original_idx:04d}.json"

                    out_path = Path(output_subdir) / filename
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    network_retried = False
                    sample_retries_left = 2
                    while True:
                        try:
                            await self.execute_single_segment(seg_cfg, str(out_path))
                            break
                        except Exception as e:
                            error_msg = str(e)
                            if "jsbsim_bridge 进程已退出" in error_msg:
                                _log("[Session] jsbsim_bridge 崩溃，终止本轮采集")
                                raise RuntimeError("jsbsim_bridge 进程异常退出") from e
                            elif ("Socket closed" in error_msg or "UNAVAILABLE" in error_msg) and not network_retried:
                                network_retried = True
                                _log(
                                    f"[Session] 段连接断开(Socket closed/UNAVAILABLE)，"
                                    f"{self.runtime.sleeps.segment_retry_backoff_s}s 后重试本段一次..."
                                )
                                _cleanup_segment_outputs(out_path)
                                await asyncio.sleep(self.runtime.sleeps.segment_retry_backoff_s)
                            elif "采样结果无效" in error_msg and sample_retries_left > 0:
                                sample_retries_left -= 1
                                _log(
                                    f"[Session] 段采样失败 ({error_msg})；"
                                    f"重启 sortie 仅重做本段（剩余重试 {sample_retries_left}）..."
                                )
                                _cleanup_segment_outputs(out_path)
                                try:
                                    await self._cleanup_after_sortie(log_fn=_log, land_first=True)
                                except Exception as cleanup_e:
                                    _log(f"[Session] 警告: cleanup 异常已忽略: {cleanup_e}")
                                await asyncio.sleep(self.runtime.sleeps.segment_retry_backoff_s)
                                await self._prepare_sortie_environment(
                                    wind_config, sortie_segment_configs, log_fn=_log
                                )
                            else:
                                _cleanup_segment_outputs(out_path)
                                raise
                    _log(f"[Session] 逻辑段 {original_idx + 1}/5 已保存 {out_path.name}")
                    await asyncio.sleep(self.runtime.sleeps.between_segments_s)
            finally:
                await self._cleanup_after_sortie(log_fn=_log, land_first=True)
                await asyncio.sleep(
                    self.runtime.sleeps.between_segments_s
                    if sortie_idx < len(sortie_groups)
                    else self.runtime.sleeps.between_sorties_s
                )

    # ---------- 恢复 / 补采（薄包装，实现见 lib/recovery/runner.py） ----------
    async def generate_missing_report(self, report_path):
        await _recovery.generate_missing_report(self, Path(report_path))

    async def run_single_run_recovery(self, run_number, base_seed=26, log_path=None):
        await _recovery.run_single_run_recovery(self, run_number, base_seed=base_seed, log_path=log_path)

    async def run_single_segment_recovery(self, run_number, segment_number, base_seed=26):
        await _recovery.run_single_segment_recovery(self, run_number, segment_number, base_seed=base_seed)

    async def run_full_paper_automated(self, log_path=None, round_timeout=None, max_retries=1, base_seed=26, skip_existing=True):
        await _recovery.run_full_paper_automated(
            self,
            log_path=Path(log_path) if log_path else None,
            round_timeout=round_timeout,
            max_retries=max_retries,
            base_seed=base_seed,
            skip_existing=skip_existing,
        )

    async def run_full_400_automated(self, *args, **kwargs):
        """兼容旧调用名；当前执行论文版 160 轮 / 800 段采集。"""
        return await self.run_full_paper_automated(*args, **kwargs)


def main():
    """超薄 CLI 入口；具体逻辑见 :mod:`lib.cli`。"""
    from lib.cli import run as cli_run
    return cli_run(
        generator_factory=DatasetGenerator,
        require_mavsdk_runtime=require_mavsdk_runtime,
        default_output_dir=SCRIPT_DIR.parent / "data",
    )


if __name__ == "__main__":
    raise SystemExit(main())
