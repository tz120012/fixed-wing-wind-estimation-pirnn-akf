#!/usr/bin/env python3
"""Freeze deployment artifacts and provenance before the first HITL session."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HITL.common.manifest import artifact_record, software_manifest, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--norm-params", type=Path, required=True)
    parser.add_argument(
        "--deployment-config",
        type=Path,
        default=PROJECT_ROOT / "config/config_hitl.yaml",
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=PROJECT_ROOT / "HITL/config/campaign.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "HITL/revision_protocol",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _validate_onnx(path: Path) -> dict:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to validate the deployment model") from exc
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"expected one ONNX input, got {len(inputs)}")
    shape = inputs[0].shape
    if len(shape) != 3 or list(shape[-2:]) != [100, 41]:
        raise ValueError(f"expected ONNX input [batch,100,41], got {shape}")
    return {
        "input_name": inputs[0].name,
        "input_shape": [str(item) for item in shape],
        "providers": session.get_providers(),
        "outputs": [value.name for value in session.get_outputs()],
    }


def _validate_norm(path: Path) -> dict:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    scaler = value.get("scaler_X") if isinstance(value, dict) else None
    features = int(getattr(scaler, "n_features_in_", -1))
    if features != 41:
        raise ValueError(f"expected 41-D scaler_X, got {features}")
    return {"scaler_x_features": features}


def _validate_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    checks = {
        "input_size": int(config.get("model", {}).get("input_size", -1)),
        "sequence_length": int(config.get("data", {}).get("sequence_length", -1)),
        "backend": str(config.get("deployment", {}).get("backend", "")),
        "inference_rate": float(
            config.get("deployment", {}).get("inference_rate", 0)
        ),
    }
    expected = {
        "input_size": 41,
        "sequence_length": 100,
        "backend": "onnx",
        "inference_rate": 50.0,
    }
    if checks != expected:
        raise ValueError(f"deployment config mismatch: {checks}; expected {expected}")
    return checks


def main() -> int:
    args = parse_args()
    sources = {
        "model.onnx": args.onnx.expanduser().resolve(),
        "norm_params.pkl": args.norm_params.expanduser().resolve(),
        "config_hitl_frozen.yaml": args.deployment_config.expanduser().resolve(),
        "campaign_frozen.yaml": args.campaign.expanduser().resolve(),
    }
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)

    validations = {
        "onnx": _validate_onnx(sources["model.onnx"]),
        "normalization": _validate_norm(sources["norm_params.pkl"]),
        "deployment": _validate_config(sources["config_hitl_frozen.yaml"]),
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        destination = output / name
        if destination.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {destination}; use --force")
        shutil.copy2(source, destination)

    manifest = {
        **software_manifest(),
        "schema_version": 1,
        "validations": validations,
        "artifacts": {
            name: artifact_record(output / name) for name in sorted(sources)
        },
    }
    write_json_atomic(output / "artifact_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
