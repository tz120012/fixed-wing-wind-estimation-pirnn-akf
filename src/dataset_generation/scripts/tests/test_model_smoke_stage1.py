#!/usr/bin/env python3
"""模型 forward 烟测（阶段 1，input_size=32）"""

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


def test_forward_input32():
    torch.manual_seed(0)
    model = PIGRU(input_size=32, hidden_size=64, num_layers=1)
    model.eval()
    B, T, F = 2, 100, 32
    x = torch.randn(B, T, F)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, dict):
        wind = out.get("wind_estimate", out.get("wind", None))
        assert wind is not None, "model 输出需含 wind_estimate"
        assert wind.shape == (B, 3), wind.shape
    else:
        # 旧接口：直接是 tensor
        assert out.shape[-1] == 3
        assert out.shape[0] == B


def test_backward_input32():
    """反向传播能跑通，梯度非 None。"""
    torch.manual_seed(0)
    model = PIGRU(input_size=32, hidden_size=64, num_layers=1)
    model.train()
    B, T, F = 2, 100, 32
    x = torch.randn(B, T, F)
    out = model(x)
    if isinstance(out, dict):
        wind = out.get("wind_estimate", out.get("wind", None))
    else:
        wind = out
    # 简单 dummy loss
    loss = (wind ** 2).mean()
    loss.backward()
    # 检查至少一个可学习参数有梯度
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad, "反向传播没有有效梯度"


if __name__ == "__main__":
    tests = [test_forward_input32, test_backward_input32]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  [OK]   {fn.__name__}")
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failures.append(fn.__name__)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERR]  {fn.__name__}: {type(e).__name__}: {e}")
            failures.append(fn.__name__)
    print()
    if failures:
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")
