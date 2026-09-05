"""
实验结果汇总脚本

用途：
  1. 读取 compare_methods.py 输出的 comparison_results.pkl
  2. 重新计算 Table 1 风格统计
  3. 计算 PIRNN-AKF 与基线的配对 t 检验
  4. 导出 summary_table.csv / summary_ttest.csv / summary_report.md
"""

import argparse
import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


METHOD_ORDER = ['PX4-EKF2', 'Vanilla GRU', 'PIRNN-AKF (Ours)']
BASELINE = 'PIRNN-AKF (Ours)'


def _vector_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - truth, axis=1)


def load_results(pkl_path: str) -> Dict:
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def build_table(predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    truth = predictions['Ground Truth']
    rows = []

    ekf_err = _vector_error(predictions['PX4-EKF2'], truth)
    ekf_mean = float(np.mean(ekf_err))

    for method in METHOD_ORDER:
        if method not in predictions:
            continue

        pred = predictions[method]
        vec_err = _vector_error(pred, truth)

        row = {
            'Method': method,
            'North RMSE (m/s)': float(np.sqrt(np.mean((pred[:, 0] - truth[:, 0]) ** 2))),
            'East RMSE (m/s)': float(np.sqrt(np.mean((pred[:, 1] - truth[:, 1]) ** 2))),
            'Down RMSE (m/s)': float(np.sqrt(np.mean((pred[:, 2] - truth[:, 2]) ** 2))),
            'Total Vector Error (m/s)': float(np.mean(vec_err)),
            'Vector Error STD (m/s)': float(np.std(vec_err)),
        }
        if method == 'PX4-EKF2':
            row['Improvement vs EKF2 (%)'] = np.nan
        else:
            row['Improvement vs EKF2 (%)'] = (1.0 - row['Total Vector Error (m/s)'] / ekf_mean) * 100.0

        rows.append(row)

    return pd.DataFrame(rows)


def build_ttest(predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
    truth = predictions['Ground Truth']
    if BASELINE not in predictions:
        return pd.DataFrame()

    ours_err = _vector_error(predictions[BASELINE], truth)

    def _cohens_dz(diff: np.ndarray) -> float:
        if len(diff) < 2:
            return float('nan')
        std = np.std(diff, ddof=1)
        if std == 0:
            return float('nan')
        return float(np.mean(diff) / std)

    def _bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 2000,
                      alpha: float = 0.05, seed: int = 26):
        if len(values) < 2:
            return float('nan'), float('nan')
        rng = np.random.default_rng(seed)
        n = len(values)
        stats = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            sample = values[rng.integers(0, n, size=n)]
            stats[i] = stat_fn(sample)
        low = float(np.percentile(stats, 100 * (alpha / 2)))
        high = float(np.percentile(stats, 100 * (1 - alpha / 2)))
        return low, high

    rows = []
    for method in ['PX4-EKF2', 'Vanilla GRU']:
        if method not in predictions:
            continue
        other_err = _vector_error(predictions[method], truth)
        if len(other_err) != len(ours_err):
            continue

        t_stat, p_val = ttest_rel(other_err, ours_err)
        diff = other_err - ours_err
        mean_diff = float(np.mean(diff))
        cohens_d = _cohens_dz(diff)
        mean_ci_low, mean_ci_high = _bootstrap_ci(diff, np.mean)
        d_ci_low, d_ci_high = _bootstrap_ci(diff, _cohens_dz)
        rows.append({
            'Comparison': f'{BASELINE} vs {method}',
            'Mean Error Diff (m/s)': mean_diff,
            'Mean Diff 95% CI Low': mean_ci_low,
            'Mean Diff 95% CI High': mean_ci_high,
            't-statistic': float(t_stat),
            'p-value': float(p_val),
            "Cohen's d_z": cohens_d,
            "d_z 95% CI Low": d_ci_low,
            "d_z 95% CI High": d_ci_high,
            'Significant (p<0.001)': bool(p_val < 0.001),
        })

    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return ''

    headers = [str(col) for col in df.columns]
    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')

    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                cells.append(f'{val:.6g}')
            else:
                cells.append(str(val))
        lines.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join(lines)


def write_report(output_dir: str, table_df: pd.DataFrame, ttest_df: pd.DataFrame) -> str:
    report_path = os.path.join(output_dir, 'summary_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# Experiment Summary\n\n')
        f.write('## Table 1 Summary\n\n')
        if len(table_df) > 0:
            f.write(_df_to_markdown(table_df))
            f.write('\n\n')
        else:
            f.write('No method table could be generated.\n\n')

        f.write('## Paired t-test Summary\n\n')
        if len(ttest_df) > 0:
            f.write(_df_to_markdown(ttest_df))
            f.write('\n')
        else:
            f.write('No paired t-test results were generated.\n')

    return report_path


def resolve_pkl_path(input_path: str) -> str:
    if os.path.isdir(input_path):
        candidate = os.path.join(input_path, 'comparison_results.pkl')
        if not os.path.exists(candidate):
            raise FileNotFoundError(f'未找到 comparison_results.pkl: {candidate}')
        return candidate
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'路径不存在: {input_path}')
    return input_path


def main():
    parser = argparse.ArgumentParser(description='汇总 compare_methods 实验结果')
    parser.add_argument(
        '--input',
        required=True,
        help='comparison_results.pkl 文件路径，或其所在目录'
    )
    args = parser.parse_args()

    pkl_path = resolve_pkl_path(args.input)
    output_dir = os.path.dirname(pkl_path)

    results = load_results(pkl_path)
    predictions = results.get('predictions', {})
    if 'Ground Truth' not in predictions:
        raise ValueError('结果文件缺少 Ground Truth，无法汇总。')

    table_df = build_table(predictions)
    ttest_df = build_ttest(predictions)

    table_csv = os.path.join(output_dir, 'summary_table.csv')
    ttest_csv = os.path.join(output_dir, 'summary_ttest.csv')
    table_df.to_csv(table_csv, index=False)
    ttest_df.to_csv(ttest_csv, index=False)

    report_path = write_report(output_dir, table_df, ttest_df)

    print('✅ 汇总完成')
    print(f'  Table CSV : {table_csv}')
    print(f'  T-test CSV: {ttest_csv}')
    print(f'  Report MD : {report_path}')


if __name__ == '__main__':
    main()
