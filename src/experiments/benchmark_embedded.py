"""
嵌入式平台推理基准脚本（PI-GRU）

用途：
  1) 统一测量模型前向推理延迟（ms）与吞吐（Hz）
  2) 导出可追溯结果：metrics.json / latency_samples.csv / benchmark_report.md
  3) 为论文 Section 3.2.3 / 4.5 提供可复现实测入口（不限 Jetson）

示例：
  python3 src/experiments/benchmark_embedded.py \
      --input-source test_id \
      --warmup 200 --iters 2000 --batch-size 1
"""

import argparse
import csv
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference_backends import create_backend


def _resolve_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(PROJECT_ROOT, path_value.lstrip('../'))


def _find_default_model_path(config: Dict) -> str:
    model_save_base = _resolve_path(config['training']['model_save_path'])
    direct_best = os.path.join(model_save_base, 'best_model.pth')
    if os.path.exists(direct_best):
        return direct_best

    train_dirs = [
        d for d in os.listdir(model_save_base)
        if d.startswith('train_') and os.path.isdir(os.path.join(model_save_base, d))
    ]
    if not train_dirs:
        raise FileNotFoundError(f'未找到可用模型目录: {model_save_base}')

    train_dirs.sort(reverse=True)
    candidate = os.path.join(model_save_base, train_dirs[0], 'best_model.pth')
    if not os.path.exists(candidate):
        raise FileNotFoundError(f'未找到模型文件: {candidate}')
    return candidate


def _load_inputs(config: Dict, input_source: str, batch_size: int) -> np.ndarray:
    data_dir = _resolve_path(config['data']['processed_dir'])
    seq_len = int(config['data']['sequence_length'])
    input_size = int(config['model']['input_size'])

    if input_source == 'random':
        return np.random.randn(batch_size, seq_len, input_size).astype(np.float32)

    test_path = os.path.join(data_dir, 'X_test_id.npy')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f'未找到测试集输入: {test_path}')

    x = np.load(test_path)
    if len(x) < batch_size:
        raise ValueError(f'X_test_id 样本不足，batch_size={batch_size}, 可用={len(x)}')
    return x[:batch_size].astype(np.float32)


def _sync_if_cuda(backend):
    device = getattr(backend, 'device', None)
    if isinstance(device, torch.device) and device.type == 'cuda':
        torch.cuda.synchronize(device)


def _latency_stats(samples_ms: np.ndarray) -> Dict[str, float]:
    return {
        'count': int(len(samples_ms)),
        'mean_ms': float(np.mean(samples_ms)),
        'std_ms': float(np.std(samples_ms)),
        'min_ms': float(np.min(samples_ms)),
        'p50_ms': float(np.percentile(samples_ms, 50)),
        'p90_ms': float(np.percentile(samples_ms, 90)),
        'p95_ms': float(np.percentile(samples_ms, 95)),
        'p99_ms': float(np.percentile(samples_ms, 99)),
        'max_ms': float(np.max(samples_ms)),
    }


def _read_first_line_if_exists(path: str) -> str:
    if not os.path.exists(path):
        return ''
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.readline().strip('\x00\r\n\t ')
    except OSError:
        return ''


def _hardware_platform_name(device, backend_name: str) -> str:
    dt_model = _read_first_line_if_exists('/proc/device-tree/model')
    if dt_model:
        return dt_model

    dmi_fields = [
        '/sys/devices/virtual/dmi/id/product_name',
        '/sys/devices/virtual/dmi/id/board_name',
        '/sys/devices/virtual/dmi/id/modalias',
    ]
    for field_path in dmi_fields:
        value = _read_first_line_if_exists(field_path)
        if value:
            return value

    if isinstance(device, torch.device) and device.type == 'cuda' and torch.cuda.is_available():
        gpu_index = device.index if device.index is not None else 0
        return torch.cuda.get_device_name(gpu_index)

    if backend_name == 'ascend':
        return 'Ascend NPU'

    machine = platform.machine()
    system = platform.system()
    return f'{system} {machine}'.strip()


def _compute_form(device, backend_name: str) -> str:
    if backend_name == 'ascend':
        return 'NPU'
    if isinstance(device, torch.device):
        if device.type == 'cuda':
            return 'GPU (CUDA)'
        if device.type == 'cpu':
            return 'CPU'
        return device.type.upper()
    return str(device) if device is not None else 'unknown'


def _device_info(backend, backend_name: str) -> Dict[str, str]:
    device = getattr(backend, 'device', None)
    info = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'hardware_platform': _hardware_platform_name(device, backend_name),
        'compute_form': _compute_form(device, backend_name),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'backend': backend_name,
        'device': getattr(backend, 'device_name', str(device)),
        'cuda_available': str(torch.cuda.is_available()),
    }
    if isinstance(device, torch.device) and device.type == 'cuda':
        gpu_index = device.index if device.index is not None else 0
        info['gpu_name'] = torch.cuda.get_device_name(gpu_index)
        total_mem_gb = torch.cuda.get_device_properties(gpu_index).total_memory / 1024**3
        info['gpu_total_mem_gb'] = f'{total_mem_gb:.2f}'
    return info


def _write_outputs(output_dir: str,
                   metrics: Dict,
                   samples_ms: np.ndarray,
                   args: argparse.Namespace):
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, 'latency_samples.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'latency_ms'])
        for i, v in enumerate(samples_ms, start=1):
            writer.writerow([i, float(v)])

    report_path = os.path.join(output_dir, 'benchmark_report.md')
    s = metrics['stats']
    device_info = metrics.get('device_info', {})
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# Embedded Benchmark Report\n\n')
        f.write(f'- Timestamp: {metrics["timestamp"]}\n')
        f.write(f'- Backend: {metrics["backend"]}\n')
        f.write(f'- Configured GPU: {metrics.get("configured_use_gpu", "unknown")}\n')
        f.write(f'- CUDA visible: {device_info.get("cuda_available", "unknown")}\n')
        f.write(f'- Hardware platform: {device_info.get("hardware_platform", "unknown")}\n')
        f.write(f'- Compute form: {device_info.get("compute_form", "unknown")}\n')
        f.write(f'- Runtime device: {device_info.get("device", "unknown")}\n')
        if device_info.get('gpu_name'):
            f.write(f'- Accelerator: {device_info["gpu_name"]}\n')
        f.write(f'- Model: {metrics["model_path"]}\n')
        f.write(f'- Input source: {args.input_source}\n')
        f.write(f'- Batch size: {args.batch_size}\n')
        f.write(f'- Warmup iters: {args.warmup}\n')
        f.write(f'- Measure iters: {args.iters}\n\n')

        f.write('## Latency Summary\n\n')
        f.write('| Metric | Value |\n')
        f.write('|---|---|\n')
        f.write(f'| mean_ms | {s["mean_ms"]:.4f} |\n')
        f.write(f'| p95_ms | {s["p95_ms"]:.4f} |\n')
        f.write(f'| p99_ms | {s["p99_ms"]:.4f} |\n')
        f.write(f'| max_ms | {s["max_ms"]:.4f} |\n')
        f.write(f'| throughput_hz | {metrics["throughput_hz"]:.2f} |\n')


def main():
    parser = argparse.ArgumentParser(description='PI-GRU 嵌入式平台推理基准测试')
    parser.add_argument('--config', default=os.path.join(PROJECT_ROOT, 'config', 'config.yaml'),
                        help='配置文件路径')
    parser.add_argument('--backend', choices=['torch', 'ascend', 'torchscript', 'torch_compile', 'compile'], default=None,
                        help='推理后端（默认读取 config.deployment.backend）')
    parser.add_argument('--model-path', default=None,
                        help='模型路径（torch: best_model.pth；ascend: .om）')
    parser.add_argument('--input-source', choices=['test_id', 'random'], default='test_id',
                        help='输入来源：test_id 使用真实测试输入，random 使用随机输入')
    parser.add_argument('--batch-size', type=int, default=1, help='批量大小')
    parser.add_argument('--warmup', type=int, default=200, help='预热迭代次数')
    parser.add_argument('--iters', type=int, default=2000, help='正式计时迭代次数')
    parser.add_argument('--output-dir', default=os.path.join(PROJECT_ROOT, 'benchmark_logs'),
                        help='输出目录基路径')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    deploy_cfg = config.setdefault('deployment', {})
    if args.backend:
        deploy_cfg['backend'] = args.backend
    backend_name = str(deploy_cfg.get('backend', 'torch')).lower()

    _torch_backends = ('torch', 'torchscript', 'torch_compile', 'compile')
    if backend_name in _torch_backends:
        model_path = args.model_path if args.model_path else _find_default_model_path(config)
        deploy_cfg['model_path_torch'] = _resolve_path(model_path)
    else:
        if not args.model_path:
            raise ValueError('ascend 后端需要显式传入 --model-path 指向 .om 模型文件')
        model_path = _resolve_path(args.model_path)
        deploy_cfg['model_path_ascend'] = model_path

    backend = create_backend(config)
    backend_info = backend.load()

    x_np = _load_inputs(config, args.input_source, args.batch_size)
    prev_log_q = None
    prev_log_r = None

    for _ in range(args.warmup):
        out = backend.infer(
            x_np,
            prev_log_q=prev_log_q,
            prev_log_r=prev_log_r,
            ema_alpha=0.1,
            clamp=True
        )
        prev_log_q = out.get('log_q_scale')
        prev_log_r = out.get('log_r_scale')
    _sync_if_cuda(backend)

    latency_ms = np.zeros(args.iters, dtype=np.float64)
    for i in range(args.iters):
        _sync_if_cuda(backend)
        t0 = time.perf_counter()
        out = backend.infer(
            x_np,
            prev_log_q=prev_log_q,
            prev_log_r=prev_log_r,
            ema_alpha=0.1,
            clamp=True
        )
        _sync_if_cuda(backend)
        latency_ms[i] = (time.perf_counter() - t0) * 1000.0
        prev_log_q = out.get('log_q_scale')
        prev_log_r = out.get('log_r_scale')

    backend.close()

    stats = _latency_stats(latency_ms)
    throughput_hz = 1000.0 / stats['mean_ms'] if stats['mean_ms'] > 0 else float('nan')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'embedded_benchmark_{timestamp}')

    metrics = {
        'timestamp': timestamp,
        'backend': backend_name,
        'configured_use_gpu': bool(deploy_cfg.get('use_gpu', True)),
        'device_info': _device_info(backend, backend_name),
        'model_path': backend_info.get('model_path', model_path),
        'input_source': args.input_source,
        'batch_size': args.batch_size,
        'warmup_iters': args.warmup,
        'measure_iters': args.iters,
        'stats': stats,
        'throughput_hz': float(throughput_hz),
    }

    _write_outputs(output_dir, metrics, latency_ms, args)

    print('✅ Embedded Benchmark 完成')
    print(f'  输出目录: {output_dir}')
    print(f'  mean: {stats["mean_ms"]:.4f} ms | p95: {stats["p95_ms"]:.4f} ms | p99: {stats["p99_ms"]:.4f} ms')
    print(f'  throughput: {throughput_hz:.2f} Hz')


if __name__ == '__main__':
    main()
