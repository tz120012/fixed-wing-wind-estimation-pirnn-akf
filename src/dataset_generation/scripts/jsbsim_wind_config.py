"""
jsbsim_wind_config.py
在 JSBSim 中设置风场：修改 XML 配置文件（启动 SITL 前调用）。
不依赖 PX4 的 SIM_WIND_* 参数。
支持 PX4-Autopilot 或 PX4-Autopilot-v133 的 Tools/jsbsim_bridge 目录结构。

重要：PX4 的 jsbsim_bridge 通过 initial_condition->Load(init_script_path) 加载「场景初始条件」，
风场需写在 scene 初始条件文件中（如 scene/LSZH.xml），否则仿真内风场不生效。
"""

import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import random
from pathlib import Path


# JSBSim 使用 英尺/秒 (ft/s)，转换系数
MPS_TO_FPS = 3.28084

# ---- dataset_config.json 辅助加载 ----
_SCRIPT_DIR = Path(__file__).resolve().parent

def _load_turb_intensities():
    """从 dataset_config.json 加载湍流增益映射，如 {"light": 1.0, "moderate": 1.5}。"""
    if not hasattr(_load_turb_intensities, "_cache"):
        cfg_path = _SCRIPT_DIR.parent / "configs" / "dataset_config.json"
        mapping = {"light": 1.0, "moderate": 2.0}  # 默认值
        if cfg_path.exists():
            try:
                turb_cfg = json.loads(cfg_path.read_text(encoding="utf-8")).get("turbulence", {})
                for t in turb_cfg.get("types", []):
                    key = f"{t}_intensity"
                    if key in turb_cfg:
                        mapping[t] = turb_cfg[key]
            except Exception:
                pass
        _load_turb_intensities._cache = mapping
    return _load_turb_intensities._cache


def get_px4_root():
    """返回 PX4 根目录：优先环境变量 PX4_ROOT，其次 PX4-Autopilot-v133，再 home 下 PX4-Autopilot。"""
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


def get_jsbsim_bridge_root():
    """返回 jsbsim_bridge 根目录（即 PX4/Tools/jsbsim_bridge）。"""
    return get_px4_root() / "Tools" / "jsbsim_bridge"


def get_jsbsim_scene_ic_path():
    """
    返回 bridge 加载的「场景初始条件」文件路径。
    bridge 在 main 中通过 --scene 或默认使用 scene/LSZH.xml（相对 JSBSIM_ROOT_DIR=jsbsim_bridge）。
    修改此文件中的风场后，再启动 SITL，仿真内风场才会生效。
    """
    root = get_jsbsim_bridge_root()
    if not root.exists():
        raise FileNotFoundError(f"未找到 jsbsim_bridge 目录: {root}")
    path = root / "scene" / "LSZH.xml"
    if not path.exists():
        raise FileNotFoundError(f"未找到场景初始条件文件: {path}")
    return str(path)


def get_jsbsim_rascal_xml_path():
    """返回 Rascal 的 JSBSim FDM 路径（jsbsim_bridge/models/Rascal/Rascal110-JSBSim.xml）。"""
    px4_dir = get_px4_root()
    candidates = [
        px4_dir / "Tools/jsbsim_bridge/models/Rascal/Rascal110-JSBSim.xml",
        px4_dir / "Tools/simulation/jsbsim/models/rascal/rascal_jsbsim.xml",
        px4_dir / "Tools/sitl_gazebo/models/rascal/rascal_jsbsim.xml",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"未找到 Rascal JSBSim 配置文件，已检查: {[str(c) for c in candidates]}。"
        "请设置环境变量 PX4_ROOT 或确保 PX4 目录下存在 jsbsim_bridge/models/Rascal/Rascal110-JSBSim.xml"
    )


def set_jsbsim_wind_in_scene_ic(wind_north_mps, wind_east_mps, wind_down_mps=0.0):
    """
    在 jsbsim_bridge 的「场景初始条件」文件（scene/LSZH.xml）中设置风场。
    bridge 通过 initial_condition->Load(init_script_path) 只加载此文件，风场必须写在这里才生效。

    JSBSim IC 支持 vwind（风速幅值 ft/s）+ winddir（风向「来自」角度，度），或部分版本支持 NED 分量。
    此处同时写入 vwind/winddir 与 vw-north-fps/vw-east-fps/vw-down-fps（若 loader 支持）。
    """
    scene_path = Path(get_jsbsim_scene_ic_path())
    tree = ET.parse(scene_path)
    root = tree.getroot()
    if root.tag != "initialize":
        root = root.find(".//initialize")
    if root is None:
        raise ValueError(f"scene 文件中未找到 <initialize> 根元素: {scene_path}")

    wn_fps = wind_north_mps * MPS_TO_FPS
    we_fps = wind_east_mps * MPS_TO_FPS
    wd_fps = wind_down_mps * MPS_TO_FPS
    speed_fps = (wn_fps * wn_fps + we_fps * we_fps + wd_fps * wd_fps) ** 0.5
    # 气象风向：风「来自」的角度 (0=北)
    if speed_fps < 1e-6:
        wind_from_deg = 0.0
    else:
        wind_from_deg = (np.degrees(np.arctan2(-we_fps, -wn_fps)) + 360) % 360

    def set_or_add(parent, tag, text, attrib=None):
        el = parent.find(tag)
        if el is not None:
            el.text = text
        else:
            el = ET.SubElement(parent, tag, attrib or {})
            el.text = text
        return el

    set_or_add(root, "vwind", f"{speed_fps:.4f}", {"unit": "FT/SEC"})
    set_or_add(root, "winddir", f"{wind_from_deg:.2f}", {"unit": "DEG"})
    set_or_add(root, "vw-north-fps", f"{wn_fps:.4f}", {"unit": "FT/SEC"})
    set_or_add(root, "vw-east-fps", f"{we_fps:.4f}", {"unit": "FT/SEC"})
    set_or_add(root, "vw-down-fps", f"{wd_fps:.4f}", {"unit": "FT/SEC"})

    tree.write(scene_path, encoding="unicode", default_namespace="")
    print(
        f"[JSBSim] 已设置 scene IC 风场 ({scene_path.name}): "
        f"N={wind_north_mps:.2f}, E={wind_east_mps:.2f}, D={wind_down_mps:.2f} m/s"
    )
    return (wind_north_mps, wind_east_mps, wind_down_mps)


def set_aircraft_initial_heading(heading_deg=None, wind_direction_deg=None):
    """
    设置飞机初始航向（psi），避免强逆风导致 EKF 无法收敛。
    
    策略：
    - 如果提供 wind_direction_deg，则将飞机航向设为与风向垂直（侧风起飞），
      避免正逆风导致地面姿态扰动。
    - 如果提供 heading_deg，则直接使用该航向。
    - 两者都不提供时，随机生成航向。
    
    Args:
        heading_deg: 直接指定的航向（0-360度，0=北）
        wind_direction_deg: 风向（0-360度），飞机将设为与风向垂直
    
    Returns:
        实际设置的航向角（度）
    """
    scene_path = Path(get_jsbsim_scene_ic_path())
    tree = ET.parse(scene_path)
    root = tree.getroot()
    if root.tag != "initialize":
        root = root.find(".//initialize")
    if root is None:
        raise ValueError(f"scene 文件中未找到 <initialize> 根元素: {scene_path}")

    if heading_deg is not None:
        psi = heading_deg
    elif wind_direction_deg is not None:
        # 将航向设为与风向垂直（+90度），形成侧风而非逆风
        # 风向是风"来自"的方向，飞机侧对风向
        psi = (wind_direction_deg + 90) % 360
    else:
        # 随机航向
        psi = random.uniform(0, 360)
    
    # 更新 psi 元素
    psi_elem = root.find("psi")
    if psi_elem is not None:
        psi_elem.text = f" {psi:.1f} "
    else:
        psi_elem = ET.SubElement(root, "psi", {"unit": "DEG"})
        psi_elem.text = f" {psi:.1f} "

    tree.write(scene_path, encoding="unicode", default_namespace="")
    print(f"[JSBSim] 已设置飞机初始航向: psi={psi:.1f}° (场景: {scene_path.name})")
    return psi


def set_jsbsim_wind(xml_path, wind_north_mps, wind_east_mps, wind_down_mps=0.0, turb_gain=1.0):
    """
    在 JSBSim XML 中设置风场与湍流强度。
    若文件中无 <winds>，会在合适位置创建（部分 FDM 文件无 winds，则写入同目录下 wind_override.xml 供后续加载）。

    Args:
        xml_path: rascal JSBSim 配置或 FDM 路径
        wind_north_mps: 北向风速 (m/s)
        wind_east_mps: 东向风速 (m/s)
        wind_down_mps: 垂向风速 (m/s)，通常 0
        turb_gain: 湍流增益（如 1.0=轻度, 2.0=中度）

    Returns:
        (wind_north, wind_east, wind_down) 用于写入元数据/后处理
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML 不存在: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    winds = root.find(".//winds")
    if winds is None:
        # FDM 常无 winds，写单独 override 文件，与 FDM 同目录
        override_path = xml_path.parent / "wind_override.xml"
        if override_path.exists():
            tree_ov = ET.parse(override_path)
            root_ov = tree_ov.getroot()
            winds = root_ov.find(".//winds")
        else:
            root_ov = ET.Element("PropertyList")
            winds = ET.SubElement(root_ov, "winds")
            for tag in ["wind_north", "wind_east", "wind_down"]:
                ET.SubElement(winds, tag).text = "0"

        for tag, val in [
            ("wind_north", wind_north_mps * MPS_TO_FPS),
            ("wind_east", wind_east_mps * MPS_TO_FPS),
            ("wind_down", wind_down_mps * MPS_TO_FPS),
        ]:
            el = winds.find(tag)
            if el is not None:
                el.text = f"{val:.4f}"
            else:
                ET.SubElement(winds, tag).text = f"{val:.4f}"
        tree_ov = ET.ElementTree(root_ov)
        try:
            ET.indent(tree_ov, space="  ")
        except AttributeError:
            pass
        tree_ov.write(override_path, encoding="unicode", default_namespace="")
        print(
            f"[JSBSim] 风场已写入 override: N={wind_north_mps:.2f}, E={wind_east_mps:.2f}, D={wind_down_mps:.2f} m/s"
        )
        try:
            set_jsbsim_wind_in_scene_ic(wind_north_mps, wind_east_mps, wind_down_mps)
        except Exception as e:
            print(f"[JSBSim] 警告: 写入 scene IC 风场失败: {e}")
        return (wind_north_mps, wind_east_mps, wind_down_mps)

    for tag, val in [
        ("wind_north", wind_north_mps * MPS_TO_FPS),
        ("wind_east", wind_east_mps * MPS_TO_FPS),
        ("wind_down", wind_down_mps * MPS_TO_FPS),
    ]:
        el = winds.find(tag)
        if el is not None:
            el.text = f"{val:.4f}"
        else:
            ET.SubElement(winds, tag).text = f"{val:.4f}"

    turb = root.find(".//turbulence")
    if turb is not None:
        tg = turb.find("turbulence_gain")
        if tg is not None:
            tg.text = f"{turb_gain:.2f}"

    tree.write(xml_path, encoding="unicode", default_namespace="")
    print(
        f"[JSBSim] 已设置 FDM/override 风场: N={wind_north_mps:.2f}, E={wind_east_mps:.2f}, D={wind_down_mps:.2f} m/s, turb_gain={turb_gain}"
    )
    # 同时写入 jsbsim_bridge 加载的 scene 初始条件，否则仿真内风场不生效
    try:
        set_jsbsim_wind_in_scene_ic(wind_north_mps, wind_east_mps, wind_down_mps)
    except Exception as e:
        print(f"[JSBSim] 警告: 写入 scene IC 风场失败（bridge 可能仍用默认 0 风）: {e}")
    return (wind_north_mps, wind_east_mps, wind_down_mps)


def write_jsbsim_bridge_wind_config(wind_config, segment_configs=None, wind_delay_alt_m=50.0):
    """
    写入 jsbsim_bridge 的 wind_config.txt，供 bridge 在 RunIC() 后读取，设置恒定风、湍流与可选阵风。

    注意：bridge 侧一次 sortie 仅支持一个 gust 事件，因此调用方若有多个 gust 段，
    必须先拆成多个 sortie；这里会在发现多个 gust 段时直接报错，避免静默写错真值。
    """
    import math

    root = get_jsbsim_bridge_root()
    if not root.exists():
        raise FileNotFoundError(f"未找到 jsbsim_bridge 目录: {root}")
    path = root / "wind_config.txt"
    phases_path = root / "wind_config_phases.txt"
    if phases_path.exists():
        disabled_path = root / "wind_config_phases.disabled"
        try:
            if disabled_path.exists():
                disabled_path.unlink()
            phases_path.rename(disabled_path)
            print(
                "[JSBSim] 已禁用 wind_config_phases.txt，"
                "数据集采集将使用 wind_config.txt 单 sortie 风场配置"
            )
        except OSError as e:
            raise RuntimeError(
                f"无法禁用 {phases_path}；jsbsim_bridge 会优先读取该文件并忽略 wind_config.txt"
            ) from e

    wn = wind_config.get("wind_north", 0.0)
    we = wind_config.get("wind_east", 0.0)
    wd = wind_config.get("wind_down", 0.0)
    turb_gain = wind_config.get("turbulence_gain", 1.0)
    turb_type = 3  # 0=None, 1=Standard, 2=Culp, 3=Milspec(Dryden), 4=Tustin

    lines = [
        f"WIND_NORTH_FPS={wn * MPS_TO_FPS:.4f}",
        f"WIND_EAST_FPS={we * MPS_TO_FPS:.4f}",
        f"WIND_DOWN_FPS={wd * MPS_TO_FPS:.4f}",
        f"TURB_GAIN={turb_gain:.2f}",
        f"TURB_TYPE={turb_type}",
        f"WIND_DELAY_ALT_M={wind_delay_alt_m:.1f}",
    ]

    gust_start_time_sec = -1.0
    gust_mag_fps = 0.0
    gust_startup = 0.0
    gust_steady = 0.0
    gust_end = 0.0
    gust_n = 0.0
    gust_e = 0.0
    gust_d = 0.0

    gust_segments = []
    if segment_configs:
        gust_segments = [(idx, seg) for idx, seg in enumerate(segment_configs) if seg.get("gust")]

    if len(gust_segments) > 1:
        raise ValueError("一次 sortie 仅支持一个 gust 段，请先拆分 segment_configs")

    if gust_segments:
        idx, seg = gust_segments[0]
        g = seg["gust"]
        mag_mps = max(0.0, float(g.get("magnitude", 5.0)))
        dur = max(0.5, float(g.get("duration", 5.0)))
        direction_deg = float(g.get("direction", 0.0))
        segment_duration = max(0.5, float(seg.get("duration", dur)))
        segment_start_time = max(0.0, float(g.get("start_time", 0.0)))
        latest_start = max(0.0, segment_duration - dur - 0.5)
        segment_start_time = min(segment_start_time, latest_start)

        gust_mag_fps = mag_mps * MPS_TO_FPS
        rad = math.radians(direction_deg)
        gust_n = math.cos(rad)
        gust_e = math.sin(rad)
        gust_d = 0.0
        gust_startup = max(0.5, dur * 0.25)
        gust_steady = max(0.5, dur * 0.5)
        gust_end = max(0.5, dur * 0.25)
        time_before_gust_segment = sum(float(s.get("duration", 0.0) or 0.0) for s in segment_configs[:idx])
        gust_start_time_sec = time_before_gust_segment + segment_start_time

    lines.extend([
        f"GUST_MAGNITUDE_FPS={gust_mag_fps:.4f}",
        f"GUST_STARTUP_SEC={gust_startup:.4f}",
        f"GUST_STEADY_SEC={gust_steady:.4f}",
        f"GUST_END_SEC={gust_end:.4f}",
        f"GUST_NORTH_FPS={gust_n:.4f}",
        f"GUST_EAST_FPS={gust_e:.4f}",
        f"GUST_DOWN_FPS={gust_d:.4f}",
        f"GUST_START_TIME_SEC={gust_start_time_sec:.4f}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 恒定风显示：优先用 wind_speed/wind_direction，否则由 N/E 分量反算
    speed = wind_config.get("wind_speed")
    direction = wind_config.get("wind_direction")
    if speed is None or direction is None:
        speed = (wn * wn + we * we + wd * wd) ** 0.5
        direction = (np.degrees(np.arctan2(-we, -wn)) + 360) % 360 if speed >= 1e-6 else 0.0
    const_wind_str = f"恒定风({speed:.1f}m/s, {direction:.0f}°)+湍流(turb_gain={turb_gain})"
    gust_str = (
        f", 阵风(mag={gust_mag_fps / MPS_TO_FPS:.1f}m/s @ t={gust_start_time_sec:.1f}s)"
        if gust_mag_fps > 0
        else ""
    )
    print(f"[JSBSim] 已写入 bridge wind_config.txt: {const_wind_str}{gust_str}")


def wind_from_speed_direction(speed_mps, direction_deg):
    """由风速(m/s)和风向(度, 0=北)得到 N/E/D 分量 (m/s)。"""
    rad = np.deg2rad(direction_deg)
    wn = speed_mps * np.cos(rad)
    we = speed_mps * np.sin(rad)
    return wn, we, 0.0


def set_random_wind_in_jsbsim(xml_path=None, speed_range=(2, 8), direction_range=(0, 360), turb_gain=None):
    """
    随机生成风场并写入 JSBSim XML。用于每架次前调用。

    speed_range 默认 (2, 8) m/s —— Rascal 仅 13 磅，原先 (6,14) 会在地面导致 JSBSim
    数值发散（NaN）；8 m/s 上限在延迟施加＋NaN 兜底下可安全运行。

    Returns:
        dict: wind_north, wind_east, wind_down (m/s), wind_speed, wind_direction
    """
    if xml_path is None:
        xml_path = get_jsbsim_rascal_xml_path()

    speed = random.uniform(*speed_range)
    direction = random.uniform(*direction_range)
    wn, we, wd = wind_from_speed_direction(speed, direction)
    if turb_gain is None:
        turb_gain = random.choice([1.0, 1.5])

    set_jsbsim_wind(xml_path, wn, we, wd, turb_gain=turb_gain)
    return {
        "wind_north": wn,
        "wind_east": we,
        "wind_down": wd,
        "wind_speed": speed,
        "wind_direction": direction,
        "turbulence_gain": turb_gain,
    }


if __name__ == "__main__":
    xml_path = get_jsbsim_rascal_xml_path()
    info = set_random_wind_in_jsbsim(xml_path)
    print("当前随机风场(用于元数据):", info)
