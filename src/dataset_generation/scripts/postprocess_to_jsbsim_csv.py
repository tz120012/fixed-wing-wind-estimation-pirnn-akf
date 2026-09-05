"""
postprocess_to_jsbsim_csv.py
将 JSON（遥测 + 元数据）转为 JSBSim 风格 CSV，与 utils/jsbsim_csv_parser 及
src/1_data_preprocessing_csv.py 对接。每行含逐行真值风（稳态恒定；阵风由 1-cos 在 JSON 中已写入 wind_true）。
"""

import json
import math
import argparse
from pathlib import Path

# 与 JSBSim 单位一致：速度/风 用 ft/s，姿态用 rad
MPS_TO_FPS = 3.28084

# 与 Dataset/flight_data_all_in_one.csv 列名一致
CSV_HEADER = [
    "Time",
    "/fdm/jsbsim/simulation/sim-time-sec",
    "/fdm/jsbsim/atmosphere/wind-north-fps",
    "/fdm/jsbsim/atmosphere/wind-east-fps",
    "/fdm/jsbsim/atmosphere/wind-down-fps",
    "/fdm/jsbsim/velocities/vc-fps",
    "/fdm/jsbsim/velocities/vtrue-fps",
    "/fdm/jsbsim/velocities/vg-fps",
    "/fdm/jsbsim/position/h-agl-ft",
    "/fdm/jsbsim/position/lat-geod-deg",
    "/fdm/jsbsim/position/long-gc-deg",
    "/fdm/jsbsim/velocities/v-north-fps",
    "/fdm/jsbsim/velocities/v-east-fps",
    "/fdm/jsbsim/velocities/v-down-fps",
    "/fdm/jsbsim/attitude/pitch-rad",
    "/fdm/jsbsim/attitude/roll-rad",
    "/fdm/jsbsim/attitude/psi-rad",
    "/fdm/jsbsim/velocities/p-rad_sec",
    "/fdm/jsbsim/velocities/q-rad_sec",
    "/fdm/jsbsim/velocities/r-rad_sec",
    "/fdm/jsbsim/fcs/aileron-cmd-norm",
    "/fdm/jsbsim/fcs/elevator-cmd-norm",
    "/fdm/jsbsim/fcs/throttle-cmd-norm",
    "/fdm/jsbsim/fcs/rudder-cmd-norm",
    "wind_regime",
    "gust_phase",
    "gust_factor",
    "base_wind_north_mps",
    "base_wind_east_mps",
    "base_wind_down_mps",
    "gust_delta_north_mps",
    "gust_delta_east_mps",
    "gust_delta_down_mps",
    "maneuver_regime",
    "turn_state",
    "turn_class",
    # ============ 阶段 1：PX4 目标量 ============
    "target_roll_rad",
    "target_pitch_rad",
    "target_yaw_rad",
    "target_p_rad_s",
    "target_q_rad_s",
    "target_r_rad_s",
    # ============ 阶段 2：目标速度 / 实际舵面 / IMU 加速度 ============
    "target_vn",
    "target_ve",
    "target_vd",
    "aileron_actual",
    "elevator_actual",
    "rudder_actual",
    "throttle_actual",
    "imu_ax",
    "imu_ay",
    "imu_az",
]

def compute_projected_wind(vn, ve, vd, tas, wn_macro, we_macro, wd_macro):
    """通过真实的飞行动力学速度三角闭环，反解出包含瞬时高频湍流的精确相对风"""
    vg_sq = vn*vn + ve*ve + vd*vd
    w_mag = math.sqrt(wn_macro**2 + we_macro**2 + wd_macro**2)
    if w_mag < 1e-6:
        vg_mag = math.sqrt(vg_sq)
        if vg_mag < 1e-6:
            return 0.0, 0.0, 0.0
        amount = max(vg_mag - tas, 0.0)
        return (vn/vg_mag)*amount, (ve/vg_mag)*amount, (vd/vg_mag)*amount
    
    un, ue, ud = wn_macro/w_mag, we_macro/w_mag, wd_macro/w_mag
    dot = vn*un + ve*ue + vd*ud
    c = vg_sq - tas*tas
    disc = dot*dot - c
    
    if disc >= 0:
        sqrt_disc = math.sqrt(disc)
        r1 = dot - sqrt_disc
        r2 = dot + sqrt_disc
        
        cands = [r for r in (r1, r2) if r >= 0.0]
        if not cands:
            amount = max(dot, 0.0)
        else:
            amount = min(cands, key=lambda x: abs(x - w_mag))
    else:
        amount = max(dot, 0.0)
        
    return un*amount, ue*amount, ud*amount

def row_from_entry(entry, t0, fallback_wind=None, use_projected_wind=False):
    """
    从单条 JSON 记录生成 CSV 一行（列表）。
    支持扁平 JSON 格式（DataLogger 输出）：
      wind_north/east/down, velocity_north/east/down, roll_deg/pitch_deg/yaw_deg, 
      roll_rate_rad_s/pitch_rate_rad_s/yaw_rate_rad_s, roll_ctrl/pitch_ctrl/yaw_ctrl/throttle_ctrl
    """
    t = entry.get("timestamp", 0)
    time_rel = t - t0 if t0 is not None else t

    # 真值风 - 支持扁平格式或嵌套格式
    if "wind_north" in entry:
        # 扁平格式（DataLogger 输出）
        wn_fps = entry.get("wind_north", 0) * MPS_TO_FPS
        we_fps = entry.get("wind_east", 0) * MPS_TO_FPS
        wd_fps = entry.get("wind_down", 0) * MPS_TO_FPS
    else:
        # 嵌套格式或回退
        wind = entry.get("wind_true") or fallback_wind or {}
        wn_fps = wind.get("north", 0) * MPS_TO_FPS
        we_fps = wind.get("east", 0) * MPS_TO_FPS
        wd_fps = wind.get("down", 0) * MPS_TO_FPS

    # NED 地速 - 支持扁平格式或嵌套格式
    if "velocity_north" in entry:
        # 扁平格式（DataLogger 输出）
        vn = entry.get("velocity_north", 0) * MPS_TO_FPS
        ve = entry.get("velocity_east", 0) * MPS_TO_FPS
        vd = entry.get("velocity_down", 0) * MPS_TO_FPS
    else:
        # 嵌套格式
        vel = entry.get("velocity_ned", {})
        vn = vel.get("north", 0) * MPS_TO_FPS
        ve = vel.get("east", 0) * MPS_TO_FPS
        vd = vel.get("down", 0) * MPS_TO_FPS
    
    vg = math.sqrt(vn * vn + ve * ve)  # 地速只含水平分量
    
    # 空速 - 优先使用 airspeed_m_s，否则用地速近似
    if "airspeed_m_s" in entry and entry["airspeed_m_s"] is not None:
        vtrue = entry["airspeed_m_s"] * MPS_TO_FPS
    else:
        vtrue = vg
    vc = vtrue
    
    # 默认使用采集端写入的真值风（DataLogger 已按 wind_truth 对齐总风，含湍流）。
    # 仅在显式传入 --use-projected-wind 时才启用速度三角反投影（兼容历史数据）。
    if use_projected_wind:
        wn_fps, we_fps, wd_fps = compute_projected_wind(
            vn, ve, vd, vtrue, wn_fps, we_fps, wd_fps
        )

    # 位置 - 支持扁平格式或嵌套格式（可选字段）
    if "position" in entry:
        pos = entry["position"]
        alt_agl_m = pos.get("alt_rel", 0)
        lat = pos.get("lat", 0)
        lon = pos.get("lon", 0)
    else:
        alt_agl_m = entry.get("alt_rel", 0)
        lat = entry.get("lat", 0)
        lon = entry.get("lon", 0)
    alt_agl_ft = alt_agl_m / 0.3048

    # 姿态 - 支持扁平格式（度）或嵌套格式
    if "roll_deg" in entry:
        # 扁平格式（DataLogger 输出，单位为度）
        roll_rad = math.radians(entry.get("roll_deg", 0))
        pitch_rad = math.radians(entry.get("pitch_deg", 0))
        psi_rad = math.radians(entry.get("yaw_deg", 0))
    else:
        # 嵌套格式
        att = entry.get("attitude", {})
        roll_rad = math.radians(att.get("roll", 0))
        pitch_rad = math.radians(att.get("pitch", 0))
        psi_rad = math.radians(att.get("yaw", 0))

    # 角速度 - 支持扁平格式或嵌套格式
    if "roll_rate_rad_s" in entry:
        # 扁平格式（DataLogger 输出，单位为 rad/s）
        p = entry.get("roll_rate_rad_s", 0)
        q = entry.get("pitch_rate_rad_s", 0)
        r = entry.get("yaw_rate_rad_s", 0)
    else:
        # 嵌套格式
        imu = entry.get("imu", {})
        p = imu.get("gyro_x", 0)
        q = imu.get("gyro_y", 0)
        r = imu.get("gyro_z", 0)

    # 控制输入 - 支持扁平格式
    aileron = entry.get("roll_ctrl", 0)
    elevator = entry.get("pitch_ctrl", 0)
    throttle = entry.get("throttle_ctrl", 0)
    rudder = entry.get("yaw_ctrl", 0)

    # 提取刚注入的 maneuver_regime
    maneuver_regime = entry.get("maneuver_regime", "unknown")
    turn_state = entry.get("turn_state", "unknown")
    turn_class = int(entry.get("turn_class", -1))

    # ============ 阶段 1：PX4 目标量（rad 单位，与 jsbsim 姿态列一致）============
    # JSON 中保存的 target_*_deg 是度数（与 roll_deg 一致），CSV 列以 rad 输出
    target_roll_rad = math.radians(entry.get("target_roll_deg", 0.0) or 0.0)
    target_pitch_rad = math.radians(entry.get("target_pitch_deg", 0.0) or 0.0)
    target_yaw_rad = math.radians(entry.get("target_yaw_deg", 0.0) or 0.0)
    target_p = entry.get("target_roll_rate_rad_s", 0.0) or 0.0
    target_q = entry.get("target_pitch_rate_rad_s", 0.0) or 0.0
    target_r = entry.get("target_yaw_rate_rad_s", 0.0) or 0.0

    # ============ 阶段 2：目标速度 NED / 实际舵面 / IMU 机体加速度 ============
    target_vn = entry.get("target_velocity_north", 0.0) or 0.0
    target_ve = entry.get("target_velocity_east", 0.0) or 0.0
    target_vd = entry.get("target_velocity_down", 0.0) or 0.0
    aileron_actual = entry.get("aileron_actual_rad", 0.0) or 0.0
    elevator_actual = entry.get("elevator_actual_rad", 0.0) or 0.0
    rudder_actual = entry.get("rudder_actual_rad", 0.0) or 0.0
    throttle_actual = entry.get("throttle_actual_norm", 0.0) or 0.0
    imu_ax = entry.get("imu_accel_body_x", 0.0) or 0.0
    imu_ay = entry.get("imu_accel_body_y", 0.0) or 0.0
    imu_az = entry.get("imu_accel_body_z", 0.0) or 0.0

    return [
        time_rel,
        time_rel,
        wn_fps,
        we_fps,
        wd_fps,
        vc,
        vtrue,
        vg,
        alt_agl_ft,
        lat,
        lon,
        vn,
        ve,
        vd,
        pitch_rad,
        roll_rad,
        psi_rad,
        p,
        q,
        r,
        aileron,
        elevator,
        throttle,
        rudder,
        entry.get("wind_regime", "unknown"),
        entry.get("gust_phase", "unknown"),
        entry.get("gust_factor", 0.0),
        entry.get("base_wind_north", 0.0),
        entry.get("base_wind_east", 0.0),
        entry.get("base_wind_down", 0.0),
        entry.get("gust_delta_north", 0.0),
        entry.get("gust_delta_east", 0.0),
        entry.get("gust_delta_down", 0.0),
        maneuver_regime,
        turn_state,
        turn_class,
        # ============ 阶段 1：PX4 目标量 ============
        target_roll_rad,
        target_pitch_rad,
        target_yaw_rad,
        target_p,
        target_q,
        target_r,
        # ============ 阶段 2：目标速度 / 实际舵面 / IMU 加速度 ============
        target_vn,
        target_ve,
        target_vd,
        aileron_actual,
        elevator_actual,
        rudder_actual,
        throttle_actual,
        imu_ax,
        imu_ay,
        imu_az,
    ]


def convert_one_flight(json_path, metadata_path, csv_path, use_projected_wind=False):
    """将单架次 JSON + metadata 转为单 CSV。metadata 中的恒定风作为 wind_true 缺失时的回退。"""
    with open(json_path) as f:
        data = json.load(f)
    if not data:
        return False

    # 加载 metadata 中的恒定风作为回退（部分 entry 可能缺少 wind_true）
    fallback_wind = {"north": 0, "east": 0, "down": 0}
    if metadata_path is not None:
        try:
            with open(metadata_path) as f:
                meta = json.load(f)
            fallback_wind = {
                "north": meta.get("wind_north", 0),
                "east": meta.get("wind_east", 0),
                "down": meta.get("wind_down", 0),
            }
        except Exception:
            pass

    t0 = data[0].get("timestamp")
    rows = [row_from_entry(e, t0, fallback_wind, use_projected_wind=use_projected_wind) for e in data]
    with open(csv_path, "w") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    return True


def run_batch(data_dir, output_dir=None, splits=None, use_projected_wind=False):
    """
    data_dir: 根目录，下有 train/val/test_id/test_ood，每子目录有 datasets-XX-X.json + datasets-XX-X_metadata.json
    output_dir: 输出 CSV 根目录，默认 data_dir/processed；下同样子目录 datasets-XX-X.csv
    """
    if splits is None:
        splits = ["train", "val", "test_id", "test_ood"]
    data_dir = Path(data_dir)
    output_dir = Path(output_dir or data_dir / "processed")
    converted = 0
    for split in splits:
        src = data_dir / split
        if not src.exists():
            print(f"  跳过不存在的目录: {split}")
            continue
        dst = output_dir / split
        dst.mkdir(parents=True, exist_ok=True)
        # 支持两种文件命名格式：datasets-XX-X.json 和 flight_*.json
        json_files = list(src.glob("datasets-*.json")) + list(src.glob("flight_*.json"))
        for j in json_files:
            if "_metadata" in j.name:
                continue
            base = j.stem
            meta = src / f"{base}_metadata.json"
            csv_path = dst / f"{base}.csv"
            # metadata 是可选的
            meta_path = str(meta) if meta.exists() else None
            if convert_one_flight(
                str(j),
                meta_path,
                str(csv_path),
                use_projected_wind=use_projected_wind,
            ):
                converted += 1
                print(f"  {split}/{base}.csv")
    print(f"共转换 {converted} 个 CSV")
    return converted


def main():
    parser = argparse.ArgumentParser(description="JSON + metadata → JSBSim 格式 CSV")
    parser.add_argument("data_dir", type=str, nargs="?", default=None, help="数据根目录（含 train/val/test_id/test_ood）")
    parser.add_argument("--output-dir", type=str, default=None, help="CSV 输出根目录，默认 data_dir/processed")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test_id", "test_ood"])
    parser.add_argument(
        "--use-projected-wind",
        action="store_true",
        help="启用速度三角反投影覆盖风真值（默认关闭，优先使用采集端总风真值）",
    )
    args = parser.parse_args()
    data_dir = args.data_dir or (Path(__file__).resolve().parent.parent / "data")
    run_batch(
        data_dir,
        output_dir=args.output_dir,
        splits=args.splits,
        use_projected_wind=args.use_projected_wind,
    )


if __name__ == "__main__":
    main()
