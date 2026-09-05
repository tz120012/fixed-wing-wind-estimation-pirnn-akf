#!/usr/bin/env python3
"""Export the frozen 41-D PI-GRU checkpoint and verify ONNX parity."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x, return_dict=True)
        return (
            output["wind_estimate"],
            output["q_scale"],
            output["r_scale"],
            output["angles"],
            output["confidence"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--norm-params",
        type=Path,
        default=ROOT / "data/dataset_revision_41d/norm_params.pkl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--test-array",
        type=Path,
        default=ROOT / "data/dataset_revision_41d/X_test_id.npy",
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    evidence = load_module(
        ROOT / "src/experiments/paper_evidence_chain_eval.py", "onnx_evidence"
    )
    with args.norm_params.open("rb") as stream:
        norm = pickle.load(stream)
    scaler_X, scaler_y = norm["scaler_X"], norm["scaler_y"]
    if int(scaler_X.n_features_in_) != 41:
        raise ValueError("Only the frozen 41-D deployment model may be exported")
    model = evidence.load_pigru_model(
        args.checkpoint, scaler_X, scaler_y, torch.device("cpu")
    )
    wrapper = ExportWrapper(model).eval()
    test = np.load(args.test_array, mmap_mode="r")
    if test.shape[1:] != (100, 41):
        raise ValueError(f"Expected test shape (*,100,41), got {test.shape}")
    indices = np.linspace(0, len(test) - 1, num=min(8, len(test)), dtype=int)
    sample = np.ascontiguousarray(test[indices], dtype=np.float32)
    dummy = torch.from_numpy(sample[:1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_names = [
        "wind_estimate",
        "q_scale",
        "r_scale",
        "angles",
        "confidence",
    ]
    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["features"],
        output_names=output_names,
        dynamic_axes={
            "features": {0: "batch"},
            **{name: {0: "batch"} for name in output_names},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(args.output))
    session = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    with torch.no_grad():
        torch_outputs = [
            value.numpy() for value in wrapper(torch.from_numpy(sample))
        ]
    ort_outputs = session.run(None, {"features": sample})
    errors = {
        name: float(np.max(np.abs(expected - actual)))
        for name, expected, actual in zip(
            output_names, torch_outputs, ort_outputs
        )
    }
    if max(errors.values()) > 1e-4:
        raise RuntimeError(f"ONNX parity failed: {errors}")
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "norm_params": str(args.norm_params.resolve()),
        "onnx": str(args.output.resolve()),
        "input_shape": ["batch", 100, 41],
        "opset": args.opset,
        "max_absolute_errors": errors,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
