#!/usr/bin/env python3
"""聚合 4 个模型 × 2 个 test split 的多视角 dir 指标对比表。

新指标（与 4_eval_pigru.py 中 calculate_metrics v4 一致）：
  ① raw dir_MAE                  整体（与论文公平对比）
  ② dir_MAE @ |w_h|≥1.5 m/s     工程实用区间（飞控前馈用）
  ③ dir_MAE @ confidence top-90% 拒绝预测视角
  ④ Clean dir_MAE (剔 dir>30°)  模型真实能力上限
  ⑤ 按 |w_h| 分箱表             几何下界 vs 实测穿透量
"""
import os
import sys
import pickle
import yaml
import json
import numpy as np
import torch
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))
sys.path.insert(0, str(PROJ_ROOT / "src" / "px4_ekf2"))
from importlib import import_module
PIGRU = import_module("2_pigru_module").PIGRU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = PROJ_ROOT / "data" / "dataset_stratified"
PX4_CACHE = PROJ_ROOT / "data" / "evaluation" / "px4_ekf2_metrics_cache.json"

# 4 个模型规格（路径 + 配置文件）
MODELS = [
    dict(name="Exp-C-v1", cfg=PROJ_ROOT / "config/config_strat_expC.yaml",
         ckpt=PROJ_ROOT / "train_data_stratified/train_lambda0.1_20260510_225251/train_lambda0.02_20260510_225305/best_model.pth",
         dataloader="旧(b=1024, lr=1e-4)", lambda_dir=0.5, lambda_ac=1.0,
         note="Ctrl+C 中断 24 ep"),
    dict(name="Exp-C",    cfg=PROJ_ROOT / "config/config_strat_expC.yaml",
         ckpt=PROJ_ROOT / "train_data_stratified/train_lambda0.1_20260511_002215/train_lambda0.02_20260511_002250/best_model.pth",
         dataloader="新(b=2048, lr=2e-4)", lambda_dir=0.5, lambda_ac=1.0,
         note="early-stop 31 ep"),
    dict(name="Exp-A",    cfg=PROJ_ROOT / "config/config_strat_expA.yaml",
         ckpt=PROJ_ROOT / "train_data_stratified/train_lambda0.1_20260511_010319/train_lambda0.02_20260511_010325/best_model.pth",
         dataloader="新(b=2048, lr=2e-4)", lambda_dir=0.1, lambda_ac=1.0,
         note="early-stop 28 ep"),
    dict(name="Exp-B",    cfg=PROJ_ROOT / "config/config_strat_expB.yaml",
         ckpt=PROJ_ROOT / "train_data_stratified/train_lambda0.1_20260511_013913/train_lambda0.02_20260511_013918/best_model.pth",
         dataloader="新(b=2048, lr=2e-4)", lambda_dir=0.5, lambda_ac=5.0,
         note="early-stop 30 ep"),
]


def load_split(split: str):
    X = np.load(DATA_DIR / f"X_{split}.npy")
    y = np.load(DATA_DIR / f"y_{split}.npy").astype(np.float64)
    return X, y


def load_norm():
    with open(DATA_DIR / "norm_params.pkl", "rb") as f:
        mn = pickle.load(f)
    return dict(scaler_X=mn["scaler_X"], scaler_y=mn["scaler_y"],
                norm_for_model=dict(
                    X_mean=mn["scaler_X"].mean_, X_scale=mn["scaler_X"].scale_,
                    y_mean=mn["scaler_y"].mean_, y_scale=mn["scaler_y"].scale_))


def build_model(cfg_path: Path, norm_for_model):
    cfg = yaml.safe_load(open(cfg_path))["model"]
    model = PIGRU(
        input_size=cfg["input_size"], hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"], dropout=cfg["dropout"],
        enable_wind_head=True,
        enable_noise_heads=cfg["enable_noise_heads"],
        enable_angles_head=cfg["enable_angles_head"],
        enable_confidence_head=cfg["enable_confidence_head"],
        yaw_invariant=cfg["yaw_invariant"],
        norm_params=norm_for_model,
    ).to(DEVICE)
    return model


def predict(model, X, B=4096):
    preds_w, preds_q = [], []
    with torch.no_grad():
        for i in range(0, len(X), B):
            xb = torch.from_numpy(X[i:i + B]).float().to(DEVICE)
            out = model(xb, return_dict=True)
            preds_w.append(out["wind_estimate"].cpu().numpy())
            preds_q.append(out["q_scale"].cpu().numpy())
    return np.vstack(preds_w).astype(np.float64), np.vstack(preds_q).astype(np.float64)


def compute_metrics(y_norm, yp_norm, q_scales, sy):
    y_real = y_norm * sy.scale_ + sy.mean_
    yp_real = yp_norm * sy.scale_[:3] + sy.mean_[:3]

    wn_t, we_t, wd_t = y_real[:, 0], y_real[:, 1], y_real[:, 2]
    wn_p, we_p = yp_real[:, 0], yp_real[:, 1]
    wd_p = yp_real[:, 2]

    rmse_3d = float(np.sqrt(np.mean((y_real[:, :3] - yp_real) ** 2)))
    rmse_per = np.sqrt(np.mean((y_real[:, :3] - yp_real) ** 2, axis=0))
    mag_t = np.sqrt(wn_t ** 2 + we_t ** 2 + wd_t ** 2)
    mag_p = np.sqrt(wn_p ** 2 + we_p ** 2 + wd_p ** 2)
    mag_rmse = float(np.sqrt(np.mean((mag_t - mag_p) ** 2)))

    wh_t = np.sqrt(wn_t ** 2 + we_t ** 2)
    rmse_h_per = np.sqrt((wn_p - wn_t) ** 2 + (we_p - we_t) ** 2)
    dir_t = np.degrees(np.arctan2(we_t, wn_t))
    dir_p = np.degrees(np.arctan2(we_p, wn_p))
    dir_diff = ((dir_p - dir_t + 180) % 360) - 180
    dir_err = np.abs(dir_diff)

    raw_dir = float(dir_err.mean())

    # |w|≥1.5
    m_eng = wh_t >= 1.5
    eng_dir = float(dir_err[m_eng].mean()) if m_eng.any() else float("nan")
    eng_frac = float(m_eng.mean())

    # confidence top-90
    q_h = np.sqrt(q_scales[:, 0] ** 2 + q_scales[:, 1] ** 2)
    thr = float(np.percentile(q_h, 90))
    m_conf = q_h <= thr
    conf_dir = float(dir_err[m_conf].mean())

    # evil
    m_evil = dir_err > 30
    evil_pct = float(m_evil.mean())
    clean_dir = float(dir_err[~m_evil].mean()) if (~m_evil).any() else float("nan")
    evil_dir = float(dir_err[m_evil].mean()) if m_evil.any() else float("nan")
    evil_w_med = float(np.median(wh_t[m_evil])) if m_evil.any() else float("nan")

    # 分箱
    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf]
    labels = ['<0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0',
              '2.0-2.5', '2.5-3.0', '>3.0']
    by_bin = []
    for i, lb in enumerate(labels):
        mask = (wh_t >= bins[i]) & (wh_t < bins[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        seg_rmse_h = float(np.sqrt(np.mean(rmse_h_per[mask] ** 2)))
        seg_mean_w = float(wh_t[mask].mean())
        seg_dir = float(dir_err[mask].mean())
        geom = float(np.degrees(np.arctan(seg_rmse_h / max(seg_mean_w, 1e-6))))
        by_bin.append(dict(label=lb, n=n, frac=n / len(wh_t),
                           mean_w=seg_mean_w, rmse_h=seg_rmse_h,
                           dir_mae=seg_dir, geom_lower=geom))

    return dict(rmse_3d=rmse_3d, rmse_per=rmse_per.tolist(),
                mag_rmse=mag_rmse, raw_dir=raw_dir,
                eng_dir=eng_dir, eng_frac=eng_frac,
                conf_dir=conf_dir, conf_thr=thr,
                evil_pct=evil_pct, evil_dir=evil_dir,
                clean_dir=clean_dir, evil_w_med=evil_w_med,
                by_bin=by_bin, n_total=int(len(wh_t)))


def predict_px4_ekf2(X, scaler_X):
    """PX4-EKF2 baseline 预测 (慢，每样本独立热身 50 步)。
    返回 (yp_real_3d, q_dummy_zero) — 无 confidence 信息，q 用 0 占位。
    """
    from eval_px4_ekf2 import PX4EKF2WindEstimator
    from tqdm import tqdm
    ekf = PX4EKF2WindEstimator(dt=0.05)
    N, T, _ = X.shape
    yp_real = np.zeros((N, 3))
    for i in tqdm(range(N), desc="  PX4-EKF2 预测"):
        ekf.reset()
        Xd = scaler_X.inverse_transform(X[i])  # [T, 20]
        last = None
        for t in range(T):
            v_ground = Xd[t, 0:3]
            airspeed = Xd[t, 19]
            last = ekf.step(v_ground, airspeed)
        yp_real[i] = last
    # PX4-EKF2 直接输出物理单位风速；返回 (yp_real, dummy q)
    q_dummy = np.zeros((N, 3), dtype=np.float64)
    return yp_real, q_dummy


def compute_metrics_from_real(y_real, yp_real, q_scales):
    """与 compute_metrics 同样的口径，但输入已是反归一化后的 (m/s)。"""
    wn_t, we_t, wd_t = y_real[:, 0], y_real[:, 1], y_real[:, 2]
    wn_p, we_p = yp_real[:, 0], yp_real[:, 1]
    wd_p = yp_real[:, 2]

    rmse_3d = float(np.sqrt(np.mean((y_real[:, :3] - yp_real) ** 2)))
    rmse_per = np.sqrt(np.mean((y_real[:, :3] - yp_real) ** 2, axis=0))
    mag_t = np.sqrt(wn_t ** 2 + we_t ** 2 + wd_t ** 2)
    mag_p = np.sqrt(wn_p ** 2 + we_p ** 2 + wd_p ** 2)
    mag_rmse = float(np.sqrt(np.mean((mag_t - mag_p) ** 2)))

    wh_t = np.sqrt(wn_t ** 2 + we_t ** 2)
    rmse_h_per = np.sqrt((wn_p - wn_t) ** 2 + (we_p - we_t) ** 2)
    dir_t = np.degrees(np.arctan2(we_t, wn_t))
    dir_p = np.degrees(np.arctan2(we_p, wn_p))
    dir_diff = ((dir_p - dir_t + 180) % 360) - 180
    dir_err = np.abs(dir_diff)

    raw_dir = float(dir_err.mean())
    m_eng = wh_t >= 1.5
    eng_dir = float(dir_err[m_eng].mean()) if m_eng.any() else float("nan")
    eng_frac = float(m_eng.mean())

    # confidence top-90: 仅在 q_scales 非全 0 时计算
    if np.any(q_scales):
        q_h = np.sqrt(q_scales[:, 0] ** 2 + q_scales[:, 1] ** 2)
        thr = float(np.percentile(q_h, 90))
        m_conf = q_h <= thr
        conf_dir = float(dir_err[m_conf].mean())
    else:
        thr = float("nan"); conf_dir = float("nan")

    m_evil = dir_err > 30
    evil_pct = float(m_evil.mean())
    clean_dir = float(dir_err[~m_evil].mean()) if (~m_evil).any() else float("nan")
    evil_dir = float(dir_err[m_evil].mean()) if m_evil.any() else float("nan")
    evil_w_med = float(np.median(wh_t[m_evil])) if m_evil.any() else float("nan")

    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf]
    labels = ['<0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0',
              '2.0-2.5', '2.5-3.0', '>3.0']
    by_bin = []
    for i, lb in enumerate(labels):
        mask = (wh_t >= bins[i]) & (wh_t < bins[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        seg_rmse_h = float(np.sqrt(np.mean(rmse_h_per[mask] ** 2)))
        seg_mean_w = float(wh_t[mask].mean())
        seg_dir = float(dir_err[mask].mean())
        geom = float(np.degrees(np.arctan(seg_rmse_h / max(seg_mean_w, 1e-6))))
        by_bin.append(dict(label=lb, n=n, frac=n / len(wh_t),
                           mean_w=seg_mean_w, rmse_h=seg_rmse_h,
                           dir_mae=seg_dir, geom_lower=geom))

    return dict(rmse_3d=rmse_3d, rmse_per=rmse_per.tolist(),
                mag_rmse=mag_rmse, raw_dir=raw_dir,
                eng_dir=eng_dir, eng_frac=eng_frac,
                conf_dir=conf_dir, conf_thr=thr,
                evil_pct=evil_pct, evil_dir=evil_dir,
                clean_dir=clean_dir, evil_w_med=evil_w_med,
                by_bin=by_bin, n_total=int(len(wh_t)))


def evaluate_px4_ekf2(splits, cache_data, norm):
    """跑 PX4-EKF2，结果缓存到磁盘；缓存命中时直接读。"""
    if PX4_CACHE.exists():
        try:
            data = json.loads(PX4_CACHE.read_text(encoding="utf-8"))
            if all(sp in data.get("per_split", {}) for sp in splits):
                print(f"  ✓ PX4-EKF2 缓存命中: {PX4_CACHE.name}")
                return data
        except Exception:
            pass

    print("\n  → 评估 PX4-EKF2 (无缓存，~8 分钟，每样本独立 50 步热身)")
    sy = norm["scaler_y"]; sx = norm["scaler_X"]
    per_split = {}
    for sp in splits:
        X, y_norm = cache_data[sp]
        y_real = y_norm * sy.scale_ + sy.mean_
        yp_real, q = predict_px4_ekf2(X, sx)
        per_split[sp] = compute_metrics_from_real(y_real, yp_real, q)

    info = dict(meta=dict(name="PX4-EKF2", lambda_dir=None, lambda_ac=None,
                          dataloader="—", note="symforce 实现，仅水平风"),
                per_split=per_split)
    PX4_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PX4_CACHE.write_text(json.dumps(info, ensure_ascii=False, indent=2,
                                    default=float), encoding="utf-8")
    print(f"  ✓ PX4-EKF2 结果已缓存到: {PX4_CACHE}")
    return info


def main():
    print("=" * 110)
    print("  4 PI-GRU 模型 + PX4-EKF2 baseline × 2 split  多视角 dir 指标汇总")
    print("=" * 110)

    norm = load_norm()
    sy = norm["scaler_y"]

    splits = ["test_id", "test_ood"]
    cache_data = {sp: load_split(sp) for sp in splits}
    print(f"\n  数据已加载: " +
          ", ".join(f"{sp}={cache_data[sp][0].shape[0]:,}" for sp in splits))

    all_results = {}
    # PX4-EKF2 baseline 优先（带缓存）
    px4_info = evaluate_px4_ekf2(splits, cache_data, norm)
    all_results["PX4-EKF2"] = px4_info
    for m in MODELS:
        ckpt = m["ckpt"]
        if not ckpt.exists():
            print(f"\n  ⚠ 跳过 {m['name']}: 模型文件不存在 {ckpt}")
            continue
        print(f"\n  → 评估 {m['name']:<10s} (ac={m['lambda_ac']:.1f}, dir={m['lambda_dir']:.1f}, {m['dataloader']})")
        sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        model = build_model(m["cfg"], norm["norm_for_model"])
        model.load_state_dict(sd, strict=False)
        model.eval()

        per_split = {}
        for sp in splits:
            X, y_norm = cache_data[sp]
            yp_norm, q = predict(model, X)
            per_split[sp] = compute_metrics(y_norm, yp_norm, q, sy)
        all_results[m["name"]] = dict(meta=m, per_split=per_split)
        del model
        torch.cuda.empty_cache()

    # ---- 主表：raw_dir / eng_dir / conf_dir / clean_dir ----
    print("\n" + "=" * 130)
    print("  ★ 主表 — 多视角 dir 指标对比 (单位: °)")
    print("=" * 130)
    for sp in splits:
        print(f"\n  ▼ split = {sp}")
        print(f"  {'模型':<10s}  {'lambda':<14s}  {'DL':<22s}  "
              f"{'RMSE_3D':>8s}  {'mag_RMSE':>9s}  "
              f"{'raw_dir':>8s}  {'eng_dir':>8s}  {'conf_dir':>9s}  "
              f"{'clean_dir':>10s}  {'evil%':>7s}")
        print("  " + "-" * 120)
        for name, info in all_results.items():
            r = info["per_split"][sp]; m = info["meta"]
            if m.get("lambda_ac") is not None and m.get("lambda_dir") is not None:
                cfg = f"ac={m['lambda_ac']:.1f},dir={m['lambda_dir']:.1f}"
            else:
                cfg = "(baseline)"
            conf_str = f"{r['conf_dir']:>8.2f}°" if not (isinstance(r['conf_dir'], float) and (r['conf_dir'] != r['conf_dir'])) else f"{'N/A':>9s}"
            print(f"  {name:<10s}  {cfg:<14s}  {m.get('dataloader','—'):<22s}  "
                  f"{r['rmse_3d']:>8.4f}  {r['mag_rmse']:>9.4f}  "
                  f"{r['raw_dir']:>7.2f}°  {r['eng_dir']:>7.2f}°  "
                  f"{conf_str}  {r['clean_dir']:>9.2f}°  "
                  f"{100 * r['evil_pct']:>6.2f}%")

    # ---- 分箱表 (test_id 一份) ----
    print("\n" + "=" * 130)
    print("  ★ 分箱 dir_MAE 表 — split=test_id（4 模型并列；几何下界 = arctan(RMSE_h / mean|w|)）")
    print("=" * 130)
    bins = ['<0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '>3.0']
    print(f"\n  {'区间':<10s}  ", end="")
    for name in all_results:
        print(f"{name + ' dir':>14s}  {name + ' geom':>14s}  ", end="")
    print()
    print("  " + "-" * (10 + 2 + 32 * len(all_results)))
    for b in bins:
        print(f"  {b:<10s}  ", end="")
        for name, info in all_results.items():
            r = info["per_split"]["test_id"]
            byb = {x['label']: x for x in r['by_bin']}
            if b in byb:
                d, g = byb[b]['dir_mae'], byb[b]['geom_lower']
                print(f"{d:>13.2f}°  {g:>13.2f}°  ", end="")
            else:
                print(f"{'-':>13s}   {'-':>13s}   ", end="")
        print()

    # ---- 保存 JSON 结果 ----
    out_path = PROJ_ROOT / "data" / "evaluation" / "dir_metrics_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for name, info in all_results.items():
        m = info["meta"].copy()
        m = {k: (str(v) if isinstance(v, Path) else v) for k, v in m.items()}
        serializable[name] = dict(meta=m, per_split=info["per_split"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n\n  ✓ 完整指标已保存为 JSON: {out_path}")


if __name__ == "__main__":
    main()
