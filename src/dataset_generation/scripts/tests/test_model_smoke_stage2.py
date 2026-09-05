#!/usr/bin/env python3
"""模型 forward+backward 烟测（阶段 2，input_size=45）"""

import sys
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
MOD_PATH = ROOT / "src" / "2_pigru_module.py"
spec = importlib.util.spec_from_file_location("pigru_mod", str(MOD_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PIGRU = mod.PIGRU


def test_forward_input45():
    torch.manual_seed(0)
    model = PIGRU(input_size=45, hidden_size=64, num_layers=1)
    model.eval()
    x = torch.randn(2, 100, 45)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, dict):
        wind = out["wind_estimate"]
    else:
        wind = out
    assert wind.shape == (2, 3), wind.shape


def test_backward_input45():
    torch.manual_seed(0)
    model = PIGRU(input_size=45, hidden_size=64, num_layers=1)
    model.train()
    x = torch.randn(2, 100, 45)
    out = model(x)
    wind = out["wind_estimate"] if isinstance(out, dict) else out
    loss = (wind ** 2).mean()
    loss.backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad


if __name__ == "__main__":
    tests = [test_forward_input45, test_backward_input45]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  [OK]   {fn.__name__}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [ERR]  {fn.__name__}: {type(e).__name__}: {e}")
            failures.append(fn.__name__)
    print()
    if failures:
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")
