#!/usr/bin/env python3
"""Validation-locked AKF sensitivity and causal smoothing baselines."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "prediction_increment": ["prediction_delta_gain"],
    "maneuver_score": [
        "maneuver_gyro_weight",
        "maneuver_acc_weight",
        "maneuver_control_weight",
        "maneuver_throttle_weight",
        "q_maneuver_gain",
        "r_maneuver_gain",
    ],
    "disagreement_shrinkage": [
        "disagreement_scale",
        "kinematic_trust_slope",
    ],
    "mahalanobis_gate": ["mahalanobis_gate"],
    "dynamic_r_modulation": [
        "r_disagreement_gain",
        "r_outlier_multiplier",
    ],
    "final_fusion_weight": [
        "fusion_base",
        "fusion_confidence_gain",
        "fusion_disagreement_gain",
        "fusion_covariance_gain",
    ],
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_contiguous_slices(x: np.ndarray) -> list[slice]:
    """Recover sliding-window boundaries without crossing independent flights."""
    if len(x) == 0:
        return []
    if len(x) == 1:
        return [slice(0, 1)]
    consecutive = np.all(x[1:, 0, :] == x[:-1, 1, :], axis=1)
    starts = np.r_[0, np.flatnonzero(~consecutive) + 1]
    stops = np.r_[starts[1:], len(x)]
    return [slice(int(start), int(stop)) for start, stop in zip(starts, stops)]


def metrics(
    method: str,
    split: str,
    truth: np.ndarray,
    pred: np.ndarray,
    segments: list[slice],
    **tags,
):
    error = pred - truth
    horizontal_valid = np.linalg.norm(truth[:, :2], axis=1) >= 0.5
    true_dir = np.degrees(np.arctan2(truth[:, 1], truth[:, 0]))
    pred_dir = np.degrees(np.arctan2(pred[:, 1], pred[:, 0]))
    direction_error = np.abs((pred_dir - true_dir + 180.0) % 360.0 - 180.0)
    delta_parts = [np.diff(pred[item], axis=0) for item in segments if item.stop - item.start >= 2]
    second_parts = [
        np.diff(pred[item], n=2, axis=0)
        for item in segments
        if item.stop - item.start >= 3
    ]
    delta = np.concatenate(delta_parts) if delta_parts else np.empty((0, 3))
    second_delta = (
        np.concatenate(second_parts) if second_parts else np.empty((0, 3))
    )
    return {
        "method": method,
        "split": split,
        "n_samples": len(truth),
        "rmse_3d": float(np.sqrt(np.mean(error**2))),
        "rmse_n": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rmse_e": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "rmse_d": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "direction_mae_deg": (
            float(np.mean(direction_error[horizontal_valid]))
            if horizontal_valid.any()
            else float("nan")
        ),
        "jitter": (
            float(np.mean(np.linalg.norm(second_delta, axis=1)))
            if len(second_delta)
            else float("nan")
        ),
        "max_jump": (
            float(np.max(np.linalg.norm(delta, axis=1)))
            if len(delta)
            else float("nan")
        ),
        **tags,
    }


def fixed_ema(wind: np.ndarray, alpha: float) -> np.ndarray:
    output = np.empty_like(wind)
    output[0] = wind[0]
    for index in range(1, len(wind)):
        output[index] = alpha * wind[index] + (1.0 - alpha) * output[index - 1]
    return output


def confidence_complementary(
    wind: np.ndarray,
    confidence: np.ndarray,
    alpha_min: float,
    alpha_max: float,
) -> np.ndarray:
    output = np.empty_like(wind)
    output[0] = wind[0]
    for index in range(1, len(wind)):
        alpha = alpha_min + (alpha_max - alpha_min) * float(confidence[index])
        output[index] = alpha * wind[index] + (1.0 - alpha) * output[index - 1]
    return output


def fixed_covariance_kf(wind: np.ndarray, q: float, r: float) -> np.ndarray:
    state = wind[0].astype(np.float64).copy()
    covariance = np.ones(3, dtype=np.float64)
    output = np.empty_like(wind, dtype=np.float64)
    output[0] = state
    for index in range(1, len(wind)):
        covariance += q
        gain = covariance / (covariance + r)
        state += gain * (wind[index] - state)
        covariance = (1.0 - gain) * covariance
        output[index] = state
    return output


def apply_segmented(function, wind: np.ndarray, segments: list[slice], *args) -> np.ndarray:
    output = np.empty_like(wind)
    for item in segments:
        output[item] = function(wind[item], *args)
    return output


def confidence_complementary_segmented(
    wind: np.ndarray,
    confidence: np.ndarray,
    segments: list[slice],
    alpha_min: float,
    alpha_max: float,
) -> np.ndarray:
    output = np.empty_like(wind)
    for item in segments:
        output[item] = confidence_complementary(
            wind[item], confidence[item], alpha_min, alpha_max
        )
    return output


def run_akf_segmented(
    evidence,
    config: dict,
    model_output: dict,
    x: np.ndarray,
    scaler_x,
    scaler_y,
    segments: list[slice],
) -> np.ndarray:
    prediction = np.empty((len(x), 3), dtype=np.float64)
    for item in segments:
        split_output = {
            key: value[item] for key, value in model_output.items()
        }
        prediction[item], _ = evidence.run_fast_pirnn_akf(
            config,
            split_output,
            x[item],
            scaler_x,
            scaler_y,
            continuous=True,
        )
    return prediction


def selection_score(
    truth: np.ndarray, pred: np.ndarray, segments: list[slice]
) -> float:
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    second_parts = [
        np.diff(pred[item], n=2, axis=0)
        for item in segments
        if item.stop - item.start >= 3
    ]
    jitter = float(np.mean(np.linalg.norm(
        np.concatenate(second_parts), axis=1
    )))
    return rmse + 0.05 * jitter


def scaled_config(base: dict, group: str, multiplier: float) -> dict:
    config = copy.deepcopy(base)
    constants = config.setdefault("akf", {}).setdefault("constants", {})
    for key in GROUPS[group]:
        constants[key] = float(constants[key]) * multiplier
    return config


def source_run(path: Path) -> int | None:
    match = re.search(r"datasets-(\d+)-(\d+)", path.name)
    return int(match.group(1)) if match else None


def source_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.search(r"datasets-(\d+)-(\d+)", path.name)
    if not match:
        return (10**9, 10**9, str(path))
    return (int(match.group(1)), int(match.group(2)), str(path))


def load_csv_session_cache(
    csv_root: Path,
    metadata: dict,
    preprocessing,
    evidence,
    model,
    scaler_x,
    scaler_y,
    device,
    batch_size: int,
    downsample_factor: int,
    max_samples: int,
) -> dict:
    """Build full, ordered sessions directly from the source CSV records."""
    split_meta = metadata.get("stratified_meta") or {}
    val_runs = set(split_meta.get("val_runs", []))
    test_runs = set(split_meta.get("test_runs", []))
    if not val_runs or not test_runs:
        raise RuntimeError(
            "norm_params.pkl lacks stratified val/test run provenance"
        )
    all_paths = sorted(csv_root.rglob("*.csv"), key=source_sort_key)
    paths_by_split = {
        "val": [
            path for path in all_paths
            if path.parent.name != "test_ood" and source_run(path) in val_runs
        ],
        "test_id": [
            path for path in all_paths
            if path.parent.name != "test_ood" and source_run(path) in test_runs
        ],
        "test_ood": [
            path for path in all_paths if path.parent.name == "test_ood"
        ],
    }
    threshold = float(metadata.get("velocity_triangle_threshold_fps", 3.0))
    clip_sigma = float(metadata.get("clip_sigma", 0.0))
    keep = [index for index in range(45) if index not in (38, 39, 40, 41)]
    cache = {}
    for split, paths in paths_by_split.items():
        xs, truths, winds = [], [], []
        outputs: dict[str, list[np.ndarray]] = {}
        segments = []
        offset = 0
        for path in paths:
            passed, _, _ = preprocessing.check_velocity_triangle(
                str(path), threshold
            )
            if not passed:
                continue
            x45, y, _, _ = preprocessing.build_features_labels_from_csv(
                str(path),
                seq_len=int(metadata.get("sequence_length", 100)),
                weight_config={"enabled": False},
                sampling_rate=50,
                downsample_factor=downsample_factor,
            )
            if not len(x45):
                continue
            x = x45[..., keep].astype(np.float32, copy=False)
            flat = x.reshape(-1, x.shape[-1])
            flat = scaler_x.transform(flat).astype(np.float32, copy=False)
            if clip_sigma > 0:
                np.clip(flat, -clip_sigma, clip_sigma, out=flat)
            x = flat.reshape(x.shape)
            output = evidence.predict_pigru(model, x, batch_size, device)
            wind = evidence.denorm_wind(output["wind"], scaler_y)
            stop = offset + len(x)
            xs.append(x)
            truths.append(y[:, :3].astype(np.float64))
            winds.append(wind)
            for key, value in output.items():
                outputs.setdefault(key, []).append(value)
            segments.append(slice(offset, stop))
            offset = stop
            if max_samples and offset >= max_samples:
                break
        if not xs:
            raise RuntimeError(f"No valid source CSV sessions for {split}")
        x = np.concatenate(xs)
        truth = np.concatenate(truths)
        wind = np.concatenate(winds)
        output = {
            key: np.concatenate(values) for key, values in outputs.items()
        }
        if max_samples and len(x) > max_samples:
            x, truth, wind = x[:max_samples], truth[:max_samples], wind[:max_samples]
            output = {key: value[:max_samples] for key, value in output.items()}
            segments = [
                slice(item.start, min(item.stop, max_samples))
                for item in segments if item.start < max_samples
            ]
        cache[split] = (x, truth, output, wind, segments)
        print(
            f"{split}: {len(segments)} source CSV sessions, "
            f"{len(truth):,} contiguous samples"
        )
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data-dir", default="data/dataset_revision_41d")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="data/revision_akf")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--csv-root",
        default="data/data_csv",
        help="Source CSV root; used to preserve session boundaries and time order.",
    )
    parser.add_argument("--downsample-factor", type=int, default=5)
    args = parser.parse_args()

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = (ROOT / args.data_dir).resolve()
    with (ROOT / args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with (data_dir / "norm_params.pkl").open("rb") as handle:
        metadata = pickle.load(handle)
    scaler_x, scaler_y = metadata["scaler_X"], metadata["scaler_y"]

    evidence = load_module(
        ROOT / "src/experiments/paper_evidence_chain_eval.py",
        "revision_akf_evidence",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = (ROOT / args.checkpoint).resolve()
    if checkpoint.is_dir():
        candidates = list(checkpoint.rglob("best_model.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"No best_model.pth found below {checkpoint}"
            )
        checkpoint = max(
            candidates, key=lambda item: item.stat().st_mtime
        )
    model = evidence.load_pigru_model(
        checkpoint, scaler_x, scaler_y, device
    )

    preprocessing = load_module(
        ROOT / "src/1_preprocessing_data.py", "revision_akf_preprocessing"
    )
    cache = load_csv_session_cache(
        (ROOT / args.csv_root).resolve(),
        metadata,
        preprocessing,
        evidence,
        model,
        scaler_x,
        scaler_y,
        device,
        args.batch_size,
        args.downsample_factor,
        args.max_samples,
    )

    _, val_truth, val_output, val_wind, val_segments = cache["val"]
    ema_alpha = min(
        np.linspace(0.05, 1.0, 20),
        key=lambda value: selection_score(
            val_truth,
            apply_segmented(fixed_ema, val_wind, val_segments, value),
            val_segments,
        ),
    )
    confidence_grid = list(
        itertools.product(np.linspace(0.05, 0.45, 9), np.linspace(0.50, 1.0, 11))
    )
    confidence_grid = [(low, high) for low, high in confidence_grid if low < high]
    confidence_params = min(
        confidence_grid,
        key=lambda pair: selection_score(
            val_truth,
            confidence_complementary_segmented(
                val_wind,
                val_output["confidence"].reshape(-1),
                val_segments,
                pair[0],
                pair[1],
            ),
            val_segments,
        ),
    )
    kf_grid = list(itertools.product([1e-4, 3e-4, 1e-3, 3e-3, 1e-2], [0.01, 0.03, 0.1, 0.3, 1.0]))
    kf_params = min(
        kf_grid,
        key=lambda pair: selection_score(
            val_truth,
            apply_segmented(
                fixed_covariance_kf, val_wind, val_segments, pair[0], pair[1]
            ),
            val_segments,
        ),
    )

    rows = []
    for split, (x, truth, output, wind, segments) in cache.items():
        rows.append(metrics("PI-GRU", split, truth, wind, segments))
        rows.append(metrics(
            "fixed_EMA",
            split,
            truth,
            apply_segmented(fixed_ema, wind, segments, ema_alpha),
            segments,
            alpha=ema_alpha,
        ))
        rows.append(
            metrics(
                "confidence_complementary",
                split,
                truth,
                confidence_complementary_segmented(
                    wind,
                    output["confidence"].reshape(-1),
                    segments,
                    *confidence_params,
                ),
                segments,
                alpha_min=confidence_params[0],
                alpha_max=confidence_params[1],
            )
        )
        rows.append(
            metrics(
                "fixed_covariance_KF",
                split,
                truth,
                apply_segmented(fixed_covariance_kf, wind, segments, *kf_params),
                segments,
                q=kf_params[0],
                r=kf_params[1],
            )
        )
        nominal = run_akf_segmented(
            evidence, config, output, x, scaler_x, scaler_y, segments
        )
        rows.append(metrics(
            "PIRNN-AKF",
            split,
            truth,
            nominal,
            segments,
            group="nominal",
            multiplier=1.0,
        ))
        if split in {"test_id", "test_ood"}:
            for group, multiplier in itertools.product(
                GROUPS, [0.6, 0.8, 1.2, 1.4]
            ):
                prediction = run_akf_segmented(
                    evidence,
                    scaled_config(config, group, multiplier),
                    output,
                    x,
                    scaler_x,
                    scaler_y,
                    segments,
                )
                rows.append(
                    metrics(
                        "PIRNN-AKF_sensitivity",
                        split,
                        truth,
                        prediction,
                        segments,
                        group=group,
                        multiplier=multiplier,
                    )
                )

    pd.DataFrame(rows).to_csv(output_dir / "akf_sensitivity_and_baselines.csv", index=False)
    selection = {
        "selection_split": "validation_only",
        "ema_alpha": float(ema_alpha),
        "confidence_alpha_min": float(confidence_params[0]),
        "confidence_alpha_max": float(confidence_params[1]),
        "fixed_kf_q": float(kf_params[0]),
        "fixed_kf_r": float(kf_params[1]),
        "sensitivity_multipliers": [0.6, 0.8, 1.0, 1.2, 1.4],
        "groups": GROUPS,
        "source": "ordered_source_csv_sessions",
        "csv_root": str((ROOT / args.csv_root).resolve()),
        "downsample_factor": args.downsample_factor,
        "reset_filter_at_each_csv": True,
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
