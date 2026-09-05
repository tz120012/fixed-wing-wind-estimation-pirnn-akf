"""
8_dataset_replay_eval.py  —  数据集重播闭环评估（规避 HIGHRES_IMU/OOD 问题）
===========================================================================

背景
----
直接 SITL 部署时存在两个特征分布偏移问题：
  1. PX4 SITL MAVLink 流中 HIGHRES_IMU 不稳定发送（特征 6-8,42-44 全为 0）
  2. HOLD/Loiter 飞行模式与训练用 OFFBOARD PVA Orbit 模式的 OOD 偏差

本脚本采用"数据集重播"方式，完全规避上述问题：
  - 直接从测试集预处理好的 X_test_*.npy（45 维序列）流式输入模型
  - 使用与离线评估 (5b_eval_pigru_akf.py) 完全相同的 PIRNN_AKF.estimate_sequence()
    （包含完整 AKF update + kinematic fusion 步）
  - 在真实 50 Hz 节奏下运行推理，测量实际延迟
  - 可选：将估计风速通过 WIND_COV 注入运行中的 PX4 SITL
  - 输出 RMSE/MAE 与推理延迟，与论文离线结果对比

支持的评估模式
--------------
  --dataset test_id   : 同分布测试集（目标 RMSE ≈ 0.219 m/s）
  --dataset test_ood  : 分布外强阵风测试集（目标 RMSE ≈ 0.580 m/s）
  --inject-wind-cov   : 同时将估计风速发送到 PX4 SITL（需要 --connection）

运行示例
--------
  # 纯重播评估（无需 PX4）
  python3 8_dataset_replay_eval.py --dataset test_ood

  # 重播 + 注入 PX4 SITL
  python3 8_dataset_replay_eval.py --dataset test_ood --inject-wind-cov \\
      --connection udpin:0.0.0.0:14550

  # 在树莓派 5 上跑（HITL 延迟验证）
  python3 8_dataset_replay_eval.py --dataset test_ood --realtime-pace
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# ── 路径常量 ────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CONFIG_PATH  = _PROJECT_ROOT / "config" / "config_sitl.yaml"
_DATA_DIR     = _PROJECT_ROOT / "data" / "dataset_new_processed"


def _load_pirnn_akf_class():
    """动态加载 PIRNN_AKF，避免顶层循环导入。"""
    akf_path = _SCRIPT_DIR / "5_pigru_akf_fusion.py"
    spec = importlib.util.spec_from_file_location("pigru_akf_fusion", akf_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {akf_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PIRNN_AKF


def _setup_logging(out_dir: Path, label: str = "replay_eval") -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(label)
    logger.info(f"日志: {log_file}")
    return logger


# ────────────────────────────────────────────────────────────────────────────
class DatasetReplayEvaluator:
    """
    用预处理好的测试集（X_test_*.npy + y_test_*.npy）流式评估 PIRNN-AKF。

    核心：直接使用 PIRNN_AKF.estimate_sequence()，与论文离线评估完全等价，
    确保 AKF 运动学融合步骤与离线实验一致，避免手工复现 AKF 的错误。
    """

    def __init__(
        self,
        dataset: str = "test_ood",
        config_path: Optional[str] = None,
        inject_wind_cov: bool = False,
        connection_str: Optional[str] = None,
        realtime_pace: bool = True,
        inference_rate: float = 50.0,
        output_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        segment_reset: bool = True,
    ):
        self.dataset        = dataset
        self.inject_wind    = inject_wind_cov
        self.connection_str = connection_str
        self.realtime_pace  = realtime_pace
        self.inference_rate = inference_rate
        self.max_samples    = max_samples
        self.segment_reset  = segment_reset

        cfg_path = Path(config_path) if config_path else _CONFIG_PATH
        if not cfg_path.exists():
            cfg_path = _PROJECT_ROOT / "config" / "config.yaml"

        out_dir = Path(output_dir) if output_dir else _PROJECT_ROOT / "SITL" / "results"
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = "inject" if inject_wind_cov else "replay"
        self.out_csv = out_dir / f"dataset_replay_{dataset}_{suffix}_{ts}.csv"

        self.logger = _setup_logging(out_dir)

        # ── 路径加入 sys.path 供 PIRNN_AKF 内部导入 ──
        sys.path.insert(0, str(_SCRIPT_DIR))

        # ── 加载 PIRNN_AKF（与离线评估完全相同的类）──
        self.logger.info("加载 PIRNN-AKF 模型（完整融合估计器）...")
        PIRNN_AKF = _load_pirnn_akf_class()
        self.estimator = PIRNN_AKF(config_path=str(cfg_path))
        self.logger.info(
            f"模型加载完成: device={self.estimator.device}"
        )

        # ── 用于反归一化标签 ──
        self.wind_mean = self.estimator.wind_mean
        self.wind_std  = self.estimator.wind_std

        # ── 可选 MAVLink 连接 ──
        self.mav_conn = None
        if self.inject_wind and connection_str:
            self._connect_mavlink()

        # ── 加载测试集 ──
        self._load_dataset()

    # ── MAVLink 连接 ──────────────────────────────────────────────────────────
    def _connect_mavlink(self):
        try:
            from pymavlink import mavutil
            self.logger.info(f"连接 MAVLink: {self.connection_str}")
            self.mav_conn = mavutil.mavlink_connection(self.connection_str)
            self.mav_conn.wait_heartbeat(timeout=15)
            self.logger.info(
                f"✓ MAVLink 连接成功 (sysid={self.mav_conn.target_system})"
            )
        except Exception as e:
            self.logger.warning(f"MAVLink 连接失败（将跳过 WIND_COV 注入）: {e}")
            self.mav_conn = None

    def _send_wind_cov(self, wind_mps: np.ndarray):
        if self.mav_conn is None:
            return
        try:
            self.mav_conn.mav.wind_cov_send(
                int(time.time() * 1_000_000),
                float(wind_mps[0]), float(wind_mps[1]), float(wind_mps[2]),
                0.0, 0.0, 0.0, 0.0, 0.0,
            )
        except Exception as e:
            self.logger.debug(f"WIND_COV 发送失败: {e}")

    # ── 加载数据集 ────────────────────────────────────────────────────────────
    def _load_dataset(self):
        X_path = _DATA_DIR / f"X_{self.dataset}.npy"
        y_path = _DATA_DIR / f"y_{self.dataset}.npy"
        if not X_path.exists():
            raise FileNotFoundError(f"未找到特征文件: {X_path}")
        self.logger.info(f"加载数据集: {X_path.name}")
        # X 在预处理时已 StandardScaler 归一化后保存 → 直接用
        self.X = np.load(X_path).astype(np.float32)      # (N, T, 45)
        y_norm = np.load(y_path).astype(np.float32)       # (N, 7) 归一化
        if self.max_samples:
            self.X = self.X[:self.max_samples]
            y_norm = y_norm[:self.max_samples]
        # y 也已归一化，用 estimator 的 scaler_y 反归一化
        self.y_true = self.estimator.scaler_y.inverse_transform(
            y_norm.astype(np.float64)
        ).astype(np.float32)   # (N, 7)  原始单位

        wind_n = self.y_true[:, 0]
        self.logger.info(
            f"  X: {self.X.shape}  y: {self.y_true.shape}  "
            f"风场N: {wind_n.mean():.2f}±{wind_n.std():.2f} m/s  "
            f"范围: [{wind_n.min():.1f}, {wind_n.max():.1f}] m/s"
        )

    # ── 主评估循环 ────────────────────────────────────────────────────────────
    def run(self):
        n = len(self.X)
        interval = 1.0 / self.inference_rate

        latencies_ms: list[float] = []
        errors_n: list[float] = []
        errors_e: list[float] = []
        errors_d: list[float] = []
        errors_2d: list[float] = []
        valid_count = 0

        self.logger.info(
            f"\n{'='*60}\n"
            f"  数据集重播评估  —  {self.dataset.upper()}\n"
            f"  样本数: {n}  推理频率: {self.inference_rate} Hz\n"
            f"  实时节奏: {'是' if self.realtime_pace else '否'}\n"
            f"  WIND_COV 注入: {'是' if self.mav_conn else '否'}\n"
            f"{'='*60}"
        )

        # CSV 输出
        (self.out_csv.parent).mkdir(parents=True, exist_ok=True)
        csv_file = open(self.out_csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow([
            "sample_idx",
            "wind_true_n", "wind_true_e", "wind_true_d",
            "wind_est_n", "wind_est_e", "wind_est_d",
            "err_n", "err_e", "err_d", "err_2d", "err_3d",
            "latency_ms", "confidence",
        ])

        # 第一个样本强制 reset AKF
        first = True
        start_wall = time.time()
        for i in range(n):
            t_loop = time.time()

            X_seq    = self.X[i]           # (100, 45) 已归一化
            true_wind = self.y_true[i, :3]  # (3,) m/s

            t0 = time.time()
            try:
                result = self.estimator.estimate_sequence(
                    X_seq,
                    reset_filter=(first or (self.segment_reset and i % 500 == 0)),
                )
                first = False
            except Exception as e:
                self.logger.debug(f"推理失败 [sample {i}]: {e}")
                if self.realtime_pace:
                    elapsed = time.time() - t_loop
                    if elapsed < interval:
                        time.sleep(interval - elapsed)
                continue

            lat_ms = (time.time() - t0) * 1000.0

            # result['wind_estimate'] 是归一化后的融合风速 → 反归一化
            w_norm = np.asarray(result["wind_estimate"], dtype=np.float32).reshape(-1)[:3]
            wind_est = w_norm * self.wind_std + self.wind_mean

            if not np.all(np.isfinite(wind_est)):
                continue

            err_n  = float(wind_est[0] - true_wind[0])
            err_e  = float(wind_est[1] - true_wind[1])
            err_d  = float(wind_est[2] - true_wind[2])
            err_2d = float(np.sqrt(err_n**2 + err_e**2))
            err_3d = float(np.sqrt(err_n**2 + err_e**2 + err_d**2))
            conf   = float(result.get("confidence", 1.0))

            latencies_ms.append(lat_ms)
            errors_n.append(err_n)
            errors_e.append(err_e)
            errors_d.append(err_d)
            errors_2d.append(err_2d)
            valid_count += 1

            writer.writerow([
                i,
                f"{true_wind[0]:.4f}", f"{true_wind[1]:.4f}", f"{true_wind[2]:.4f}",
                f"{wind_est[0]:.4f}", f"{wind_est[1]:.4f}", f"{wind_est[2]:.4f}",
                f"{err_n:.4f}", f"{err_e:.4f}", f"{err_d:.4f}",
                f"{err_2d:.4f}", f"{err_3d:.4f}",
                f"{lat_ms:.3f}", f"{conf:.4f}",
            ])

            # 注入 PX4
            if self.mav_conn is not None:
                self._send_wind_cov(wind_est)

            # 进度
            if (i + 1) % 500 == 0 or i == n - 1:
                elapsed = time.time() - start_wall
                rmse_n  = np.sqrt(np.mean(np.array(errors_n)**2))  if errors_n  else float("nan")
                rmse_e  = np.sqrt(np.mean(np.array(errors_e)**2))  if errors_e  else float("nan")
                rmse_2d = np.sqrt(np.mean(np.array(errors_2d)**2)) if errors_2d else float("nan")
                avg_lat = np.mean(latencies_ms)                    if latencies_ms else float("nan")
                print(
                    f"\r[{i+1:6d}/{n}] "
                    f"RMSE_N={rmse_n:.3f}  RMSE_E={rmse_e:.3f}  RMSE_2D={rmse_2d:.3f} m/s | "
                    f"lat={avg_lat:.2f}ms | t={elapsed:.0f}s",
                    end="", flush=True,
                )

            # 实时节奏
            if self.realtime_pace:
                elapsed_loop = time.time() - t_loop
                if elapsed_loop < interval:
                    time.sleep(interval - elapsed_loop)

        csv_file.close()
        print()

        # ── 最终统计 ──────────────────────────────────────────────────────────
        if not latencies_ms:
            self.logger.error("所有推理均失败，无有效结果！")
            return {"rmse_3d": float("nan"), "valid_rate": 0.0}

        errs_n  = np.array(errors_n)
        errs_e  = np.array(errors_e)
        errs_d  = np.array(errors_d)
        errs_2d = np.array(errors_2d)
        lats    = np.array(latencies_ms)

        rmse_n   = float(np.sqrt(np.mean(errs_n**2)))
        rmse_e   = float(np.sqrt(np.mean(errs_e**2)))
        rmse_d   = float(np.sqrt(np.mean(errs_d**2)))
        rmse_2d  = float(np.sqrt(np.mean(errs_2d**2)))
        rmse_3d  = float(np.sqrt(np.mean(errs_n**2 + errs_e**2 + errs_d**2)))
        mae_2d   = float(np.mean(np.abs(errs_2d)))
        avg_lat  = float(np.mean(lats))
        p95_lat  = float(np.percentile(lats, 95))
        p99_lat  = float(np.percentile(lats, 99))
        max_lat  = float(np.max(lats))
        total_s  = time.time() - start_wall
        actual_hz = valid_count / total_s

        print(f"\n{'='*60}")
        print(f"  数据集重播评估完成  —  {self.dataset.upper()}")
        print(f"{'='*60}")
        print(f"  有效推理: {valid_count}/{n} ({100*valid_count/n:.1f}%)")
        print(f"  实际推理频率: {actual_hz:.1f} Hz")
        print(f"")
        print(f"  ━━ 风速估计精度 ━━")
        print(f"  RMSE_N  : {rmse_n:.4f} m/s")
        print(f"  RMSE_E  : {rmse_e:.4f} m/s")
        print(f"  RMSE_D  : {rmse_d:.4f} m/s")
        mean_comp_rmse = (rmse_n + rmse_e + rmse_d) / 3.0
        print(f"  均值RMSE: {mean_comp_rmse:.4f} m/s  ← 论文报告指标（三分量平均）")
        print(f"  RMSE_2D : {rmse_2d:.4f} m/s")
        print(f"  RMSE_3D : {rmse_3d:.4f} m/s  (3D 向量误差，供参考)")
        print(f"  MAE_2D  : {mae_2d:.4f} m/s")
        print(f"")
        print(f"  ━━ 推理延迟 ━━")
        print(f"  平均    : {avg_lat:.3f} ms")
        print(f"  p95     : {p95_lat:.3f} ms")
        print(f"  p99     : {p99_lat:.3f} ms")
        print(f"  max     : {max_lat:.3f} ms")
        print(f"")
        print(f"  论文离线结果对比（均值RMSE）:")
        ref = {"test_id": 0.219, "test_ood": 0.580}
        ref_val = ref.get(self.dataset)
        if ref_val:
            delta = mean_comp_rmse - ref_val
            match_ok = abs(delta) < ref_val * 0.25  # 25% 以内
            tag = "✓ 匹配" if match_ok else "✗ 偏差"
            print(f"  论文 均值RMSE ({self.dataset}): {ref_val:.3f} m/s")
            print(f"  本次 均值RMSE            : {mean_comp_rmse:.3f} m/s  {tag} (差值 {delta:+.4f})")
        print(f"")
        print(f"  输出 CSV: {self.out_csv}")
        print(f"{'='*60}")

        self.logger.info(
            f"评估完成: 均值RMSE={mean_comp_rmse:.4f} RMSE_3D={rmse_3d:.4f} "
            f"lat_avg={avg_lat:.2f}ms p95={p95_lat:.2f}ms"
        )

        if self.mav_conn is not None:
            self.mav_conn.close()

        return {
            "mean_comp_rmse": mean_comp_rmse,
            "rmse_3d": rmse_3d, "rmse_2d": rmse_2d,
            "rmse_n": rmse_n, "rmse_e": rmse_e, "rmse_d": rmse_d,
            "lat_avg_ms": avg_lat, "lat_p95_ms": p95_lat,
            "valid_rate": valid_count / n,
        }


# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="数据集重播闭环评估（规避 HIGHRES_IMU/OOD 问题）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", default="test_ood", choices=["test_id", "test_ood"],
        help="使用哪个测试集（默认 test_ood）",
    )
    parser.add_argument(
        "--config", default=None,
        help="config.yaml 路径",
    )
    parser.add_argument(
        "--inject-wind-cov", action="store_true",
        help="同时向 PX4 SITL 发送 WIND_COV（需要 --connection）",
    )
    parser.add_argument(
        "--connection", default="udpin:0.0.0.0:14550",
        help="MAVLink 连接（仅 --inject-wind-cov 时使用）",
    )
    parser.add_argument(
        "--no-realtime", action="store_true",
        help="不限速（尽可能快，用于批量精度评估）",
    )
    parser.add_argument(
        "--rate", type=float, default=50.0,
        help="模拟推理频率 Hz（默认 50）",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="限制评估样本数（调试用）",
    )
    parser.add_argument(
        "--no-segment-reset", action="store_true",
        help="禁用每 500 步 AKF 重置（默认每 500 步重置一次以模拟飞行段切换）",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="输出目录（默认 SITL/results/）",
    )
    args = parser.parse_args()

    evaluator = DatasetReplayEvaluator(
        dataset=args.dataset,
        config_path=args.config,
        inject_wind_cov=args.inject_wind_cov,
        connection_str=args.connection if args.inject_wind_cov else None,
        realtime_pace=not args.no_realtime,
        inference_rate=args.rate,
        max_samples=args.max_samples,
        segment_reset=not args.no_segment_reset,
        output_dir=args.output_dir,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
