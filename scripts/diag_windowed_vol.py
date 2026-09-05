"""Quick check: is windowed turbulence intensity (rolling std of true wind)
persistent/learnable, unlike the unpredictable single-step increment?

We compare, on y_train wind labels (normalized):
  - single-step |dw|  : lag-1 autocorr (should be ~0 -> unpredictable noise)
  - windowed std (W)  : lag-1 autocorr (should be high -> persistent, learnable)
and report dynamic range so we know it carries a usable signal for q_scale.
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def rolling_std(w, win):
    n = len(w)
    out = np.zeros_like(w)
    c = np.cumsum(np.insert(w, 0, 0.0, axis=0), axis=0)
    c2 = np.cumsum(np.insert(w * w, 0, 0.0, axis=0), axis=0)
    for i in range(n):
        lo = max(0, i - win + 1)
        k = i + 1 - lo
        mean = (c[i + 1] - c[lo]) / k
        var = (c2[i + 1] - c2[lo]) / k - mean ** 2
        out[i] = np.sqrt(np.maximum(var, 0.0))
    return out


def autocorr1(x):
    x = x - x.mean()
    return float(np.sum(x[1:] * x[:-1]) / (np.sum(x * x) + 1e-12))


def main():
    win = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    y = np.load(PROJECT_ROOT / "data/dataset_new_processed/y_train.npy")
    w = y[:, 0:3].astype(np.float64)
    # subsample contiguous chunk to keep it fast
    w = w[:40000]
    step = np.abs(np.diff(w, axis=0))
    step = np.vstack([step[0:1], step])
    wstd = rolling_std(w, win)
    for ax, lbl in enumerate(["N", "E"]):
        s = step[:, ax]
        v = wstd[:, ax]
        print(f"axis {lbl}: single-step  autocorr={autocorr1(s):.3f} cv={s.std()/(s.mean()+1e-9):.3f}")
        print(f"axis {lbl}: windowed(W={win}) autocorr={autocorr1(v):.3f} cv={v.std()/(v.mean()+1e-9):.3f} "
              f"p10={np.percentile(v,10):.4f} p90={np.percentile(v,90):.4f} range={np.percentile(v,90)/(np.percentile(v,10)+1e-9):.2f}")


if __name__ == "__main__":
    main()
