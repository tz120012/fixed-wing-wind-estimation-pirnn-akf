"""
系统级证据评估：
1) 时序平滑性（TV/Jitter）
2) 异常注入鲁棒性（峰值误差/恢复步数）
3) 导出系统级诊断量（P/Q/R/innovation/confidence）
"""

import argparse
import importlib.util
import os
import pickle
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def _load_module(module_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'无法加载模块: {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(PROJECT_ROOT, path_str.lstrip('../'))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(np.square(diff))))


def _vector_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y_pred - y_true, axis=1)


def _temporal_metrics(wind: np.ndarray) -> Dict[str, float]:
    if len(wind) < 3:
        return {'tv_mean': np.nan, 'tv_std': np.nan, 'jitter_mean': np.nan, 'jitter_std': np.nan}

    d1 = np.diff(wind, axis=0)
    d2 = np.diff(d1, axis=0)
    tv = np.linalg.norm(d1, axis=1)
    jitter = np.linalg.norm(d2, axis=1)
    return {
        'tv_mean': float(np.mean(tv)),
        'tv_std': float(np.std(tv)),
        'jitter_mean': float(np.mean(jitter)),
        'jitter_std': float(np.std(jitter))
    }


def _find_latest_pigru_model(model_save_path: str) -> str:
    if not os.path.isdir(model_save_path):
        raise FileNotFoundError(f'模型目录不存在: {model_save_path}')

    train_dirs = [
        d for d in os.listdir(model_save_path)
        if d.startswith('train_') and os.path.isdir(os.path.join(model_save_path, d))
    ]
    if not train_dirs:
        raise FileNotFoundError(f'未找到 train_* 目录: {model_save_path}')

    train_dirs.sort(reverse=True)
    for d in train_dirs:
        p = os.path.join(model_save_path, d, 'best_model.pth')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'未找到 best_model.pth: {model_save_path}')


def _load_split(data_dir: str, scene: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载测试集，优先使用时序版本（保持飞行段连续性）。
    返回 (X, y, segments)，segments 为 [n_segs, 2] 或 None（乱序时）。
    """
    s = scene.strip()

    # 时序版本映射
    SEQ_MAP = {
        'id':             ('X_test_sequential.npy', 'y_test_sequential.npy', 'seg_test_sequential.npy'),
        'test_id':        ('X_test_sequential.npy', 'y_test_sequential.npy', 'seg_test_sequential.npy'),
        'ood':            ('X_test_seq_ood.npy',    'y_test_seq_ood.npy',    'seg_test_seq_ood.npy'),
        'test_ood':       ('X_test_seq_ood.npy',    'y_test_seq_ood.npy',    'seg_test_seq_ood.npy'),
        'ood_turb_heavy': ('X_test_seq_ood.npy',    'y_test_seq_ood.npy',    'seg_test_seq_ood.npy'),
    }

    if s in SEQ_MAP:
        xf, yf, sf = SEQ_MAP[s]
        x_path = os.path.join(data_dir, xf)
        y_path = os.path.join(data_dir, yf)
        s_path = os.path.join(data_dir, sf)
        if os.path.exists(x_path) and os.path.exists(y_path):
            segs = np.load(s_path) if os.path.exists(s_path) else None
            print(f'  [时序] 加载 {xf}  shape={np.load(x_path).shape}  '
                  f'段数={len(segs) if segs is not None else "N/A"}')
            return np.load(x_path), np.load(y_path), segs

    # 回退：乱序版本
    if s in ('id', 'test_id', 'X_test_id'):
        x_path = os.path.join(data_dir, 'X_test_id.npy')
        y_path = os.path.join(data_dir, 'y_test_id.npy')
    elif s in ('ood', 'test_ood', 'X_test_ood'):
        x_path = os.path.join(data_dir, 'X_test_ood.npy')
        y_path = os.path.join(data_dir, 'y_test_ood.npy')
    else:
        key = s if s.startswith('ood_') else f'ood_{s}'
        x_path = os.path.join(data_dir, f'X_{key}.npy')
        y_path = os.path.join(data_dir, f'y_{key}.npy')

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f'找不到数据: {x_path} / {y_path}')

    print(f'  [乱序] 加载 {os.path.basename(x_path)}')
    return np.load(x_path), np.load(y_path), None


def _inject_anomaly(
    X_norm: np.ndarray,
    scaler_X,
    anomaly_type: str,
    strength: float,
    start_ratio: float,
    end_ratio: float,
    seed: int
) -> Tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    n, seq, feat = X_norm.shape
    flat = X_norm.reshape(-1, feat)
    X_denorm = scaler_X.inverse_transform(flat).reshape(n, seq, feat)

    start = max(1, int(n * start_ratio))
    end = max(start + 1, int(n * end_ratio))
    end = min(end, n)

    if anomaly_type == 'gps_spike':
        X_denorm[start:end, -1, 0:3] += strength
    elif anomaly_type == 'tas_spike':
        X_denorm[start:end, -1, 19] += strength
    elif anomaly_type == 'attitude_spike':
        X_denorm[start:end, -1, 9:12] += np.deg2rad(strength)
    elif anomaly_type == 'sensor_dropout':
        X_denorm[start:end, -1, 0:3] = 0.0
        X_denorm[start:end, -1, 19] = 0.0
    elif anomaly_type == 'gaussian_burst':
        X_denorm[start:end, -1, 0:3] += rng.normal(0.0, strength, size=(end - start, 3))
        X_denorm[start:end, -1, 19] += rng.normal(0.0, strength, size=(end - start,))
    else:
        raise ValueError(f'未知异常类型: {anomaly_type}')

    X_norm_corrupt = scaler_X.transform(X_denorm.reshape(-1, feat)).reshape(n, seq, feat)
    return X_norm_corrupt.astype(np.float32), start, end


def _first_recovery_step(err: np.ndarray, threshold: float, start_idx: int, min_consecutive: int = 10) -> int:
    if start_idx >= len(err):
        return -1

    count = 0
    for i in range(start_idx, len(err)):
        if err[i] <= threshold:
            count += 1
            if count >= min_consecutive:
                return i - min_consecutive + 1
        else:
            count = 0
    return -1


def _load_pigru_predictor(config: Dict, model_path: str, scaler_X=None, scaler_y=None):
    try:
        import torch
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError('当前环境缺少 torch，无法运行该评估脚本。') from e

    module = _load_module(os.path.join(PROJECT_ROOT, 'src', '2_pigru_module.py'), 'model_definition_evidence')
    PIGRU = module.PIGRU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_cfg = checkpoint.get('config', {}).get('model', config.get('model', {}))
    yaw_invariant = bool(model_cfg.get('yaw_invariant', False))
    norm_params = None
    if yaw_invariant:
        if scaler_X is None or scaler_y is None:
            raise ValueError('yaw_invariant=True 的模型加载需要 scaler_X/scaler_y')
        norm_params = {
            'X_mean': scaler_X.mean_,
            'X_scale': scaler_X.scale_,
            'y_mean': scaler_y.mean_,
            'y_scale': scaler_y.scale_,
        }

    model = PIGRU(
        input_size=model_cfg.get('input_size', config['model']['input_size']),
        hidden_size=model_cfg.get('hidden_size', config['model']['hidden_size']),
        num_layers=model_cfg.get('num_layers', config['model']['num_layers']),
        dropout=0.0,
        enable_wind_head=True,
        enable_noise_heads=model_cfg.get('enable_noise_heads', True),
        enable_angles_head=model_cfg.get('enable_angles_head', True),
        enable_confidence_head=model_cfg.get('enable_confidence_head', True),
        yaw_invariant=yaw_invariant,
        norm_params=norm_params,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    def predict(X_test: np.ndarray) -> np.ndarray:
        X_tensor = torch.FloatTensor(X_test).to(device)
        out_list = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), 256):
                out = model(X_tensor[i:i + 256], return_dict=True)
                out_list.append(out['wind_estimate'].cpu().numpy())
        return np.vstack(out_list)

    return predict


def _to_denorm(wind_norm: np.ndarray, scaler_y) -> np.ndarray:
    wind_mean = scaler_y.mean_[0:3]
    wind_std = scaler_y.scale_[0:3]
    return wind_norm * wind_std + wind_mean


def _evaluate_pair(wind_true: np.ndarray, wind_bare: np.ndarray, wind_sys: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, pred in [('bare_network', wind_bare), ('system_level', wind_sys)]:
        e = _vector_error(wind_true, pred)
        t = _temporal_metrics(pred)
        rows.append({
            'method': name,
            'rmse': _rmse(wind_true, pred),
            'vector_error_mean': float(np.mean(e)),
            'vector_error_std': float(np.std(e)),
            **t
        })
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description='系统级证据评估（平滑性+异常鲁棒+诊断量导出）')
    parser.add_argument('--config', type=str, default=os.path.join(PROJECT_ROOT, 'config', 'config.yaml'))
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--scene', type=str, default='id',
                        help='id（同分布时序）或 ood / ood_turb_heavy（分布外时序）')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--output-dir', type=str, default='')
    parser.add_argument('--anomaly-types', type=str,
                        default='gps_spike,tas_spike,attitude_spike,sensor_dropout,gaussian_burst')
    parser.add_argument('--anomaly-strength', type=float, default=2.0, help='spike/burst 强度')
    parser.add_argument('--anomaly-start-ratio', type=float, default=0.45)
    parser.add_argument('--anomaly-end-ratio', type=float, default=0.55)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(_resolve_path(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_dir = _resolve_path(config['data']['processed_dir'])
    model_save_path = _resolve_path(config['training']['model_save_path'])
    model_path = _resolve_path(args.model_path) if args.model_path else _find_latest_pigru_model(model_save_path)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = _resolve_path(args.output_dir) if args.output_dir else os.path.join(
        PROJECT_ROOT, 'data', 'system_evidence', f'evidence_{timestamp}'
    )
    os.makedirs(out_dir, exist_ok=True)

    X_test, y_test, segments = _load_split(data_dir, args.scene)
    if args.max_samples and args.max_samples > 0:
        X_test = X_test[:args.max_samples]
        y_test = y_test[:args.max_samples]
        # 截断 segments
        if segments is not None:
            segs_trunc = []
            for s, e in segments:
                if s >= args.max_samples:
                    break
                segs_trunc.append([s, min(e, args.max_samples)])
            segments = np.array(segs_trunc, dtype=np.int32) if segs_trunc else None

    # 归一化参数
    with open(os.path.join(model_save_path, 'norm_params.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    scaler_y = metadata['scaler_y']
    scaler_X = metadata['scaler_X']
    y_denorm = scaler_y.inverse_transform(y_test)
    wind_true = y_denorm[:, 0:3]

    # 裸网络预测
    bare_predict = _load_pigru_predictor(config, model_path, scaler_X=scaler_X, scaler_y=scaler_y)
    wind_bare_norm = bare_predict(X_test)
    wind_bare = _to_denorm(wind_bare_norm, scaler_y)

    # 系统级预测 + 诊断量（按段 reset AKF，保持时序连续性）
    pirnn_module = _load_module(os.path.join(PROJECT_ROOT, 'src', '5_pigru_akf_fusion.py'), 'pirnn_akf_fusion_evidence')
    PIRNN_AKF = pirnn_module.PIRNN_AKF
    estimator = PIRNN_AKF(config_path=_resolve_path(args.config), model_path=model_path)

    if segments is not None:
        # 时序模式：按段 reset
        wind_sys_norm = np.zeros((len(X_test), 3), dtype=np.float32)
        all_additional: Dict[str, list] = {k: [] for k in [
            'q_scale','r_scale','confidence','nn_weight','akf_weight',
            'innovation','innovation_norm','nis','P_diag','Q_diag','R_diag']}
        for seg_start, seg_end in segments:
            estimator.reset()
            for i in range(seg_start, seg_end):
                result = estimator.estimate_sequence(X_test[i])
                wind_sys_norm[i] = result['wind_estimate']
                for k in all_additional:
                    all_additional[k].append(result.get(k, 0))
        additional = {k: np.array(v) for k, v in all_additional.items()}
    else:
        wind_sys_norm, additional = estimator.estimate_batch(X_test)

    wind_sys = _to_denorm(wind_sys_norm, scaler_y)

    # 清洁场景指标
    smooth_df = _evaluate_pair(wind_true, wind_bare, wind_sys)
    smooth_csv = os.path.join(out_dir, 'smoothness_metrics.csv')
    smooth_df.to_csv(smooth_csv, index=False)

    # 导出系统级诊断量
    diag_npz = os.path.join(out_dir, 'system_diagnostics_clean.npz')
    np.savez_compressed(
        diag_npz,
        wind_true=wind_true,
        wind_bare=wind_bare,
        wind_system=wind_sys,
        q_scale=additional.get('q_scale'),
        r_scale=additional.get('r_scale'),
        confidence=additional.get('confidence'),
        nn_weight=additional.get('nn_weight'),
        akf_weight=additional.get('akf_weight'),
        innovation=additional.get('innovation'),
        innovation_norm=additional.get('innovation_norm'),
        nis=additional.get('nis'),
        P_diag=additional.get('P_diag'),
        Q_diag=additional.get('Q_diag'),
        R_diag=additional.get('R_diag')
    )

    # 异常鲁棒性
    clean_err_bare = _vector_error(wind_true, wind_bare)
    clean_err_sys = _vector_error(wind_true, wind_sys)
    th_bare = float(np.percentile(clean_err_bare, 95))
    th_sys = float(np.percentile(clean_err_sys, 95))

    anomaly_rows: List[Dict] = []
    anomaly_types = [v.strip() for v in args.anomaly_types.split(',') if v.strip()]

    for idx, anomaly in enumerate(anomaly_types):
        X_anom, start_idx, end_idx = _inject_anomaly(
            X_test,
            scaler_X,
            anomaly_type=anomaly,
            strength=args.anomaly_strength,
            start_ratio=args.anomaly_start_ratio,
            end_ratio=args.anomaly_end_ratio,
            seed=args.seed + idx
        )

        wind_bare_anom = _to_denorm(bare_predict(X_anom), scaler_y)

        estimator_anom = PIRNN_AKF(config_path=_resolve_path(args.config), model_path=model_path)
        if segments is not None:
            wind_sys_anom_norm = np.zeros((len(X_anom), 3), dtype=np.float32)
            for seg_start, seg_end in segments:
                estimator_anom.reset()
                for i in range(seg_start, seg_end):
                    wind_sys_anom_norm[i] = estimator_anom.estimate_sequence(X_anom[i])['wind_estimate']
        else:
            wind_sys_anom_norm, _ = estimator_anom.estimate_batch(X_anom)
        wind_sys_anom = _to_denorm(wind_sys_anom_norm, scaler_y)

        for method, pred, threshold in [
            ('bare_network', wind_bare_anom, th_bare),
            ('system_level', wind_sys_anom, th_sys)
        ]:
            err = _vector_error(wind_true, pred)
            peak_error = float(np.max(err))
            window_error = float(np.mean(err[start_idx:end_idx]))
            rec_idx = _first_recovery_step(err, threshold=threshold, start_idx=end_idx)
            recovery_steps = int(rec_idx - end_idx) if rec_idx >= 0 else -1

            anomaly_rows.append({
                'anomaly_type': anomaly,
                'method': method,
                'rmse': _rmse(wind_true, pred),
                'peak_error': peak_error,
                'window_error_mean': window_error,
                'recovery_steps': recovery_steps,
                'threshold_p95_clean': threshold,
                'window_start': start_idx,
                'window_end': end_idx
            })

    anomaly_df = pd.DataFrame(anomaly_rows)
    anomaly_csv = os.path.join(out_dir, 'anomaly_robustness.csv')
    anomaly_df.to_csv(anomaly_csv, index=False)

    print('=' * 70)
    print('系统级证据评估完成')
    print(f'模型: {model_path}')
    print(f'数据场景: {args.scene} | 样本数: {len(X_test)}')
    print(f'平滑性结果: {smooth_csv}')
    print(f'异常鲁棒性: {anomaly_csv}')
    print(f'诊断量导出: {diag_npz}')
    print('=' * 70)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'❌ 运行失败: {exc}')
        raise
