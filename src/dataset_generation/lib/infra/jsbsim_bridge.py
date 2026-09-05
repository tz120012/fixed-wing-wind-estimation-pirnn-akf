"""JSBSim bridge 配置写入 / wind_truth 查找。

不启动 / 停止 bridge 进程本身（bridge 是 PX4 SITL 的子进程），仅负责
本次重构关心的两个职责：
    1. 写入风/起始航向到 ``wind_config.txt`` 与 ``LSZH.xml``；
    2. 在 SITL 启动后查找最新生成的 ``wind_truth_*.csv``。

把原 ``DatasetGenerator._prepare_sortie_environment`` 中调用
``write_jsbsim_bridge_wind_config`` / ``set_aircraft_initial_heading`` /
``find_latest_wind_truth_csv`` 的散落片段集中到一处，方便测试与 mock。
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from ..errors import WindTruthMissing
from ..obs import FailureCategory, FailureRecord, get_logger


_LOGGER = get_logger("infra.jsbsim_bridge")


def find_latest_wind_truth_csv(bridge_root: Path) -> Optional[str]:
    """在 ``bridge_root`` 目录下查找最新生成的 ``wind_truth_*.csv``。

    bridge 在启动时按时间戳命名该文件，一次 SITL 对应一个文件。
    返回路径字符串；未找到时返回 ``None``。
    """
    bridge_root = Path(bridge_root)
    try:
        candidates = list(bridge_root.glob("wind_truth_*.csv"))
    except OSError as e:
        _LOGGER.failure(FailureRecord(
            category=FailureCategory.TELEMETRY,
            code="wind_truth_glob_failed",
            message=f"扫描 {bridge_root} 失败: {e}",
            context={"bridge_root": str(bridge_root)},
        ))
        return None
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def require_wind_truth_csv(bridge_root: Path) -> str:
    """硬要求 wind_truth CSV 存在，否则抛 ``WindTruthMissing``。"""
    path = find_latest_wind_truth_csv(bridge_root)
    if path is None:
        raise WindTruthMissing(
            f"在 {bridge_root} 未找到 wind_truth_*.csv（jsbsim_bridge 是否未启动？）",
            context={"bridge_root": str(bridge_root)},
        )
    return path


class JsbsimBridgeConfig:
    """风场 / 起始航向写入器。

    与 ``Px4SitlProcess`` 配对使用：在 ``Px4SitlProcess.start()`` 之前写好风场，
    SITL 启动后会把这些值传入 JSBSim。

    Parameters
    ----------
    bridge_root
        ``$PX4_ROOT/Tools/jsbsim_bridge`` 路径。
    rascal_xml_path
        ``Rascal110-JSBSim.xml`` 路径。
    """

    def __init__(self, bridge_root: Path, rascal_xml_path: Path) -> None:
        self.bridge_root = Path(bridge_root)
        self.rascal_xml_path = Path(rascal_xml_path)

    def write_wind_config(self, wind_config: Mapping[str, float], gust_params: Optional[dict] = None) -> None:
        """委托原 ``jsbsim_wind_config.write_jsbsim_bridge_wind_config`` 写入。"""
        # 延迟 import 避免顶层 import 循环 / mavsdk 缺依赖问题
        from jsbsim_wind_config import write_jsbsim_bridge_wind_config
        write_jsbsim_bridge_wind_config(wind_config, gust_params=gust_params)
        _LOGGER.info(
            "已写入 wind_config",
            wind_speed=round(wind_config.get("wind_speed", 0.0), 3),
            wind_direction=round(wind_config.get("wind_direction", 0.0), 1),
            turb_gain=wind_config.get("turbulence_gain"),
            has_gust=bool(gust_params),
        )

    def write_initial_heading(self, heading_deg: float) -> None:
        """委托 ``jsbsim_wind_config.set_aircraft_initial_heading``。"""
        from jsbsim_wind_config import set_aircraft_initial_heading
        set_aircraft_initial_heading(self.rascal_xml_path, heading_deg)
        _LOGGER.info("已写入起始航向", heading_deg=round(heading_deg, 1))

    def latest_wind_truth_csv(self) -> Optional[str]:
        return find_latest_wind_truth_csv(self.bridge_root)
