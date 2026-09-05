"""
硬件性能结果回填脚本

功能：
1) 自动读取最新（或指定）benchmark 日志中的 metrics.json
2) 生成论文可粘贴段落草稿（含 mean/p95/p99/throughput）
3) 可选直接替换 paper/FCGJ-v2.2.md 的 4.5 段落

示例：
    python3 src/experiments/backfill_hardware_metrics.py
    python3 src/experiments/backfill_hardware_metrics.py --benchmark-dir benchmark_logs/embedded_benchmark_20260306_110053
    python3 src/experiments/backfill_hardware_metrics.py --apply-paper
"""

import argparse
import glob
import json
import os
import re
from datetime import datetime
from typing import Dict, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_PAPER = os.path.join(PROJECT_ROOT, 'paper', 'FCGJ-v2.2.md')
DEFAULT_BENCHMARK_ROOT = os.path.join(PROJECT_ROOT, 'benchmark_logs')
DEFAULT_DRAFT_DIR = os.path.join(PROJECT_ROOT, 'paper', 'drafts')


def _is_jetson_device(device_info: Dict[str, str]) -> bool:
    text = ' '.join([
        str(device_info.get('platform', '')),
        str(device_info.get('hostname', '')),
        str(device_info.get('gpu_name', '')),
        str(device_info.get('device', '')),
    ]).lower()
    keywords = ['jetson', 'tegra', 'orin', 'xavier', 'nano']
    return any(k in text for k in keywords)


def _find_latest_benchmark_dir(root: str) -> str:
    patterns = [
        os.path.join(root, 'embedded_benchmark_*'),
        os.path.join(root, 'jetson_benchmark_*'),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend([d for d in glob.glob(pattern) if os.path.isdir(d)])
    if not candidates:
        raise FileNotFoundError(f'未找到基准目录: {patterns[0]} 或 {patterns[1]}')
    return max(candidates, key=os.path.getmtime)


def _load_metrics(benchmark_dir: str) -> Dict:
    metrics_path = os.path.join(benchmark_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f'未找到 metrics.json: {metrics_path}')
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _build_paragraph(metrics: Dict, benchmark_dir: str) -> Tuple[str, bool]:
    s = metrics['stats']
    device_info = metrics.get('device_info', {})
    is_jetson = _is_jetson_device(device_info)

    device_label = device_info.get('gpu_name') or device_info.get('platform') or '目标硬件'
    tag = '嵌入式板端实测' if is_jetson else '非Jetson主机实测（方法验证）'

    paragraph = (
        f"PIRNN-AKF 在 {device_label} 上的推理性能已完成{tag}："
        f"平均时延 {s['mean_ms']:.3f} ms/step，P95 {s['p95_ms']:.3f} ms，"
        f"P99 {s['p99_ms']:.3f} ms，吞吐 {metrics['throughput_hz']:.2f} Hz "
        f"（batch={metrics['batch_size']}，input={metrics['input_source']}，"
        f"warmup={metrics['warmup_iters']}，iters={metrics['measure_iters']}）。"
        f"原始证据文件位于 `{benchmark_dir}`（含 `metrics.json`、`latency_samples.csv`、`benchmark_report.md`）。"
    )

    if not is_jetson:
        paragraph += ' 该结果可用于流程与脚本有效性验证；最终投稿口径仍建议补充目标嵌入式平台板端同口径实测。'

    return paragraph, is_jetson


def _build_selfcheck_note(is_jetson: bool) -> str:
    if is_jetson:
        return (
            '硬件性能结论可复现性建议状态：已满足（证据充分）。\n'
            '理由：已具备板端原始日志，且包含 mean/P95/P99 与原始样本。'
        )
    return (
        '硬件性能结论可复现性建议状态：部分满足。\n'
        '理由：已具备自动化脚本与主机实测日志，仍缺目标嵌入式平台板端同口径原始日志。'
    )


def _write_draft(paragraph: str, selfcheck_note: str, benchmark_dir: str) -> str:
    os.makedirs(DEFAULT_DRAFT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(DEFAULT_DRAFT_DIR, f'hardware_metrics_backfill_{ts}.md')

    content = (
        '# 硬件性能回填草稿\n\n'
        f'- 基准目录：`{benchmark_dir}`\n\n'
        '## 4.5 段落替换建议\n\n'
        f'{paragraph}\n\n'
        '## 投稿前自检状态建议\n\n'
        f'{selfcheck_note}\n'
    )

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return out_path


def _apply_to_paper(paper_path: str, paragraph: str) -> bool:
    with open(paper_path, 'r', encoding='utf-8') as f:
        text = f.read()

    pattern = (
        r"PIRNN-AKF 在 .*?上的推理时间与 CPU 占用[\s\S]*?"
        r"(?=\n\n5\. Conclusion and Future Work)"
    )

    if not re.search(pattern, text):
        return False

    new_text = re.sub(pattern, paragraph, text, count=1)

    with open(paper_path, 'w', encoding='utf-8') as f:
        f.write(new_text)
    return True


def main():
    parser = argparse.ArgumentParser(description='硬件性能结果回填脚本')
    parser.add_argument('--benchmark-root', default=DEFAULT_BENCHMARK_ROOT,
                        help='benchmark 根目录（默认 benchmark_logs）')
    parser.add_argument('--benchmark-dir', default=None,
                        help='指定某次 benchmark 目录，不指定则自动取最新')
    parser.add_argument('--paper-file', default=DEFAULT_PAPER,
                        help='论文文件路径')
    parser.add_argument('--apply-paper', action='store_true',
                        help='将 4.5 段落直接替换为当前草稿段落')
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir or _find_latest_benchmark_dir(args.benchmark_root)
    metrics = _load_metrics(benchmark_dir)
    paragraph, is_jetson = _build_paragraph(metrics, benchmark_dir)
    selfcheck_note = _build_selfcheck_note(is_jetson)

    draft_path = _write_draft(paragraph, selfcheck_note, benchmark_dir)

    print('✅ 已生成回填草稿')
    print(f'  草稿文件: {draft_path}')
    print(f'  基准目录: {benchmark_dir}')
    print(f'  设备类型: {"Jetson" if is_jetson else "非Jetson"}')

    if args.apply_paper:
        ok = _apply_to_paper(args.paper_file, paragraph)
        if ok:
            print(f'✅ 已更新论文段落: {args.paper_file}')
        else:
            print('⚠️ 未能自动匹配论文 4.5 段落，请手动粘贴草稿内容。')


if __name__ == '__main__':
    main()
