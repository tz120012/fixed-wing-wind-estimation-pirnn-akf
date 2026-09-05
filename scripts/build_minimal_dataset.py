#!/usr/bin/env python3
"""Build the compact CSV package for the current Drones revision manuscript."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "Paper_2/MDPI_template_APA/submission"
DESTINATION = SUBMISSION / "minimal_dataset"
SEEDS = (26, 42, 2024, 2025, 2026)
SPLITS = ("test_id", "test_ood")
METHODS = {
    "vanilla_gru": "Vanilla GRU",
    "vanilla_lstm": "Vanilla LSTM",
    "pigru": "PI-GRU",
    "pirnn_akf": "PIRNN-AKF",
}
KALMANNET_DIR = ROOT / "data/revision_kalmannet_41d"
AKF_RESULTS = ROOT / "data/revision_akf/akf_sensitivity_and_baselines.csv"
AKF_SELECTION = ROOT / "data/revision_akf/selection_manifest.json"
MAIN_STATISTICS = ROOT / "data/revision_main_41d/statistics"


RETAINED_CSVS = (
    "Figure03_Table06_physics_weight_sweep.csv",
    "Table07_feature_group_ablation.csv",
    "Figure04_Table08_weak_wind_subset.csv",
    "RobustSmoothing_regime_windows.csv",
    "RobustSmoothing_fixed_EMA_vs_AKF_cross_scenario.csv",
    "Figure06_Table10_transient_tracking_vs_jitter.csv",
    "Table10_paired_wilcoxon_tests.csv",
    "Figure05_gps_spike_timeseries_panels.csv",
    "Figure05_anomaly_window_diagnostics_timeseries.csv",
    "Predecessor_edge_latency_by_backend.csv",
    "Predecessor_latency_per_frame_by_backend.csv",
    "Figure08_replay_test_ID_perframe.csv",
    "Figure08_replay_test_OOD_perframe.csv",
    "Archived_online_SITL_perframe.csv",
    "Archived_online_SITL_reference_wind.csv",
    "TableA1_dynamic_QR_ablation.csv",
    "FigureA1_dynamic_Q_volatility_regime_summary.csv",
    "FigureA1_dynamic_Q_volatility_per_window.csv",
    "TableB1_FigureB1_multi_anomaly_robustness.csv",
    "TableB1_PX4_EKF2_per_window_rows.csv",
    "TableB1_KalmanNet_per_window_rows.csv",
    "FigureD1_anomaly_strength_sweep.csv",
    "TableE1_cross_platform_forward_latency.csv",
)


def direction_metrics(truth: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    mask = np.hypot(truth[:, 0], truth[:, 1]) >= 0.5
    true_angle = np.degrees(np.arctan2(truth[mask, 1], truth[mask, 0]))
    estimate_angle = np.degrees(
        np.arctan2(estimate[mask, 1], estimate[mask, 0])
    )
    error = np.abs((estimate_angle - true_angle + 180.0) % 360.0 - 180.0)
    return float(error.mean()), float(np.quantile(error, 0.95))


def metric_row(
    method: str,
    split: str,
    seed: int | str,
    truth: np.ndarray,
    estimate: np.ndarray,
    provenance: str,
) -> dict[str, object]:
    error = estimate.astype(np.float64) - truth.astype(np.float64)
    axis_rmse = np.sqrt(np.mean(error**2, axis=0))
    direction_mae, direction_p95 = direction_metrics(truth, estimate)
    return {
        "method": method,
        "split": split,
        "seed": seed,
        "n_samples": len(truth),
        "rmse_3d_mps": float(np.sqrt(np.mean(error**2))),
        "rmse_n_mps": float(axis_rmse[0]),
        "rmse_e_mps": float(axis_rmse[1]),
        "rmse_d_mps": float(axis_rmse[2]),
        "direction_mae_deg": direction_mae,
        "direction_p95_deg": direction_p95,
        "provenance": provenance,
    }


def build_main_comparison() -> None:
    prediction_dir = ROOT / "data/revision_main_41d/predictions"
    rows: list[dict[str, object]] = []
    truths: dict[str, np.ndarray] = {}
    kalmannet_predictions: dict[str, dict[int, np.ndarray]] = {
        split: {} for split in SPLITS
    }
    for seed in SEEDS:
        for split in SPLITS:
            archive_path = prediction_dir / f"seed{seed}_{split}.npz"
            archive = np.load(archive_path)
            truth = archive["wind_true"].astype(np.float64)
            truths.setdefault(split, truth)
            if not np.array_equal(truths[split], truth):
                raise ValueError(f"Truth arrays differ across seeds for {split}")
            for key, label in METHODS.items():
                rows.append(metric_row(
                    label,
                    split,
                    seed,
                    truth,
                    archive[key],
                    str(archive_path.relative_to(ROOT)),
                ))
            kalmannet_path = (
                KALMANNET_DIR / f"seed{seed}_{split}_kalmannet.npy"
            )
            kalmannet = np.load(kalmannet_path)
            if kalmannet.shape != truth.shape:
                raise ValueError(
                    f"KalmanNet shape mismatch for seed {seed} {split}: "
                    f"{kalmannet.shape} != {truth.shape}"
                )
            kalmannet_predictions[split][seed] = kalmannet
            rows.append(metric_row(
                "KalmanNet",
                split,
                seed,
                truth,
                kalmannet,
                str(kalmannet_path.relative_to(ROOT)),
            ))

    ekf = np.load(ROOT / "data/revision_main_41d/ekf_predictions.npz")
    for split in SPLITS:
        rows.append(metric_row(
            "PX4-EKF2",
            split,
            "deterministic",
            truths[split],
            ekf[split],
            "data/revision_main_41d/ekf_predictions.npz",
        ))

    axis_names = ("n", "e", "d")
    for split in SPLITS:
        truth = truths[split]
        columns: dict[str, np.ndarray] = {
            "sample_index": np.arange(len(truth), dtype=np.int64),
        }
        for axis_index, axis_name in enumerate(axis_names):
            columns[f"truth_{axis_name}_mps"] = truth[:, axis_index]
        for seed in SEEDS:
            estimate = kalmannet_predictions[split][seed]
            for axis_index, axis_name in enumerate(axis_names):
                columns[f"seed{seed}_{axis_name}_mps"] = estimate[:, axis_index]
        pd.DataFrame(columns).to_csv(
            DESTINATION / f"Figure02_KalmanNet_predictions_{split}.csv",
            index=False,
            float_format="%.9g",
        )

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(
        DESTINATION / "Figure02_Table04_main_metrics_per_seed.csv",
        index=False,
    )

    aggregate_rows: list[dict[str, object]] = []
    metric_columns = [
        "rmse_3d_mps",
        "rmse_n_mps",
        "rmse_e_mps",
        "rmse_d_mps",
        "direction_mae_deg",
        "direction_p95_deg",
    ]
    for (method, split), group in per_seed.groupby(["method", "split"]):
        row: dict[str, object] = {
            "method": method,
            "split": split,
            "n_seeds": 1 if method == "PX4-EKF2" else len(group),
            "source_level": "derived_from_archived_predictions",
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_sd"] = (
                0.0 if len(group) == 1 else float(group[column].std(ddof=1))
            )
        aggregate_rows.append(row)

    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(
        DESTINATION / "Figure02_Table04_main_metrics_aggregated.csv",
        index=False,
    )


def write_configuration_tables() -> None:
    pd.DataFrame([
        ("Prediction increment", "gain/component clip", "0.25 / +/-1 m/s"),
        ("Maneuver score", "gyro/acceleration/control/throttle weights",
         "0.45/0.30/0.20/0.05"),
        ("Disagreement shrinkage", "scale/trust range", "3.0 m/s / [0.20,0.90]"),
        ("Innovation gate", "squared Mahalanobis threshold", "9.0"),
        ("Dynamic-R modulation", "maneuver/disagreement/outlier", "0.20/1.50/2.0"),
        ("Final fusion weight", "formula/range/outlier cap",
         "0.58+0.10c-0.28d-0.10u / [0.18,0.72] / 0.28"),
    ], columns=["functional_group", "item", "nominal_setting"]).to_csv(
        DESTINATION / "Table01_AKF_constants.csv", index=False
    )

    pd.DataFrame([
        ("Geometry", "wing area/span/mean chord", "0.982/2.795/0.351 m"),
        ("Mass properties", "empty mass", "5.897 kg"),
        ("Mass properties", "Ixx/Iyy/Izz", "2.644/2.102/2.590 kg m^2"),
        ("Propulsion", "nominal electric-engine power", "1050 W"),
        ("Sampling", "rate/window", "50 Hz/100 samples"),
        ("Atmosphere", "wind model", "constant + 1-cos gust + Dryden"),
    ], columns=["group", "parameter", "frozen_value"]).to_csv(
        DESTINATION / "Table02_Rascal_parameters.csv", index=False
    )

    pd.DataFrame([
        ("Input", "dimension", "41"),
        ("Sampling", "rate_hz", "50"),
        ("Sampling", "sequence_steps", "100"),
        ("Architecture", "GRU hidden size", "128"),
        ("Architecture", "LSTM hidden size", "110"),
        ("Optimization", "learning rate", "1e-4"),
        ("Optimization", "batch size", "1024"),
        ("Optimization", "maximum epochs", "150"),
        ("Replicates", "random seeds", "26,42,2024,2025,2026"),
        ("Deployment", "latency budget ms", "20"),
    ], columns=["category", "item", "setting"]).to_csv(
        DESTINATION / "Table03_model_configuration.csv", index=False
    )


def build_cross_configuration() -> None:
    # This object archive was written with NumPy 2. Alias its private module
    # path when the release builder runs in the retained NumPy 1.24 environment.
    sys.modules.setdefault("numpy._core", np.core)
    for module_name in ("multiarray", "numeric", "_multiarray_umath"):
        module = getattr(np.core, module_name, None)
        if module is not None:
            sys.modules.setdefault(f"numpy._core.{module_name}", module)
    archive_path = (
        ROOT / "data/revision_cross_airframe/cross_airframe_predictions.npz"
    )
    records = np.load(archive_path, allow_pickle=True)["records"]
    session_rows: list[dict[str, object]] = []
    seed_positions: dict[int, int] = {}
    for record in records:
        seed = int(record["seed"])
        position = seed_positions.get(seed, 0)
        if position >= 10:
            raise ValueError(f"More than ten Malolo records for seed {seed}")
        condition = "ID wind" if position < 5 else "Stronger wind/gust"
        seed_positions[seed] = position + 1
        truth = np.asarray(record["truth"], dtype=np.float64)
        for prediction_key, method in (
            ("pi_gru", "PI-GRU"),
            ("pirnn_akf", "PIRNN-AKF"),
        ):
            estimate = np.asarray(record[prediction_key], dtype=np.float64)
            error = estimate - truth
            true_magnitude = np.linalg.norm(truth, axis=1)
            estimate_magnitude = np.linalg.norm(estimate, axis=1)
            true_angle = np.degrees(np.arctan2(truth[:, 1], truth[:, 0]))
            estimate_angle = np.degrees(
                np.arctan2(estimate[:, 1], estimate[:, 0])
            )
            direction_error = np.abs(
                (estimate_angle - true_angle + 180.0) % 360.0 - 180.0
            )
            second_difference = np.diff(estimate, n=2, axis=0)
            jumps = np.diff(estimate, axis=0)
            axis_rmse = np.sqrt(np.mean(error**2, axis=0))
            session_rows.append({
                "condition": condition,
                "method": method,
                "seed": seed,
                "session": record["session"],
                "n": len(truth),
                "rmse_3d_mps": float(np.sqrt(np.mean(error**2))),
                "rmse_n_mps": float(axis_rmse[0]),
                "rmse_e_mps": float(axis_rmse[1]),
                "rmse_d_mps": float(axis_rmse[2]),
                "mae_mps": float(np.mean(np.abs(error))),
                "magnitude_rmse_mps": float(np.sqrt(np.mean(
                    (estimate_magnitude - true_magnitude) ** 2
                ))),
                "magnitude_mae_mps": float(np.mean(np.abs(
                    estimate_magnitude - true_magnitude
                ))),
                "direction_mae_deg": float(direction_error.mean()),
                "direction_p95_deg": float(
                    np.quantile(direction_error, 0.95)
                ),
                "jitter_mps": float(np.mean(
                    np.linalg.norm(second_difference, axis=1)
                )),
                "max_jump_mps": float(np.max(
                    np.linalg.norm(jumps, axis=1)
                )),
                "nonfinite_failure_rate": float(
                    1.0 - np.isfinite(estimate).all(axis=1).mean()
                ),
            })
    if seed_positions != {seed: 10 for seed in SEEDS}:
        raise ValueError(f"Unexpected Malolo record counts: {seed_positions}")

    by_session = pd.DataFrame(session_rows)
    by_session.to_csv(
        DESTINATION / "Table05_cross_configuration_by_session.csv",
        index=False,
    )

    pooled_rows = []
    rmse_fields = [
        "rmse_3d_mps", "rmse_n_mps", "rmse_e_mps", "rmse_d_mps",
        "magnitude_rmse_mps",
    ]
    mean_fields = [
        "mae_mps", "magnitude_mae_mps", "direction_mae_deg",
        "direction_p95_deg", "jitter_mps", "nonfinite_failure_rate",
    ]
    for (condition, method, seed), group in by_session.groupby(
        ["condition", "method", "seed"], sort=True
    ):
        weights = group["n"].to_numpy(np.float64)
        total = float(weights.sum())
        row: dict[str, object] = {
            "condition": condition,
            "method": method,
            "seed": seed,
            "n": int(total),
        }
        for field in rmse_fields:
            values = group[field].to_numpy(np.float64)
            row[field] = float(
                np.sqrt(np.sum(weights * values**2) / total)
            )
        for field in mean_fields:
            values = group[field].to_numpy(np.float64)
            row[field] = float(np.sum(weights * values) / total)
        row["max_jump_mps"] = float(group["max_jump_mps"].max())
        pooled_rows.append(row)
    by_seed = pd.DataFrame(pooled_rows)
    by_seed.to_csv(
        DESTINATION / "Table05_cross_configuration_by_seed.csv",
        index=False,
    )

    closure = {
        ("ID wind", "PI-GRU"): (0.693, 0.040),
        ("ID wind", "PIRNN-AKF"): (0.686, 0.039),
        ("Stronger wind/gust", "PI-GRU"): (0.664, 0.079),
        ("Stronger wind/gust", "PIRNN-AKF"): (0.655, 0.078),
    }
    summary_rows = []
    for (condition, method), group in by_seed.groupby(
        ["condition", "method"], sort=False
    ):
        closure_mean, closure_sd = closure[(condition, method)]
        summary_rows.append({
            "condition": condition,
            "method": method,
            "rmse_3d_mean_mps": float(group["rmse_3d_mps"].mean()),
            "rmse_3d_sd_mps": float(group["rmse_3d_mps"].std(ddof=1)),
            "direction_mae_mean_deg": float(
                group["direction_mae_deg"].mean()
            ),
            "direction_mae_sd_deg": float(
                group["direction_mae_deg"].std(ddof=1)
            ),
            "closure_rmse_mean_mps": closure_mean,
            "closure_rmse_sd_mps": closure_sd,
            "jitter_mean_mps": float(group["jitter_mps"].mean()),
            "jitter_sd_mps": float(group["jitter_mps"].std(ddof=1)),
            "nonfinite_failure_rate": float(
                group["nonfinite_failure_rate"].mean()
            ),
            "provenance": (
                "prediction metrics derived from cross_airframe_predictions.npz; "
                "closure RMSE retained from audited manuscript aggregate"
            ),
        })
    pd.DataFrame(summary_rows).to_csv(
        DESTINATION / "Table05_cross_configuration_summary.csv",
        index=False,
    )


def write_revision_summaries() -> None:
    source = pd.read_csv(AKF_RESULTS)
    provenance = str(AKF_RESULTS.relative_to(ROOT))
    method_labels = {
        "PI-GRU": "PI-GRU",
        "fixed_EMA": "Fixed EMA (alpha=1.0)",
        "confidence_complementary": "Confidence complementary",
        "fixed_covariance_KF": "Fixed-covariance KF",
        "PIRNN-AKF": "PIRNN-AKF",
    }
    baseline = source[source["method"].isin(method_labels)].copy()
    expected = {
        (method, split)
        for method in method_labels
        for split in ("val", "test_id", "test_ood")
    }
    observed = set(zip(baseline["method"], baseline["split"]))
    if observed != expected:
        raise ValueError(
            "Unexpected AKF baseline rows: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    baseline.insert(
        1, "method_label", baseline["method"].map(method_labels)
    )
    baseline["provenance"] = provenance
    baseline.to_csv(
        DESTINATION / "Table09_causal_smoothing_rows.csv", index=False
    )

    comparison_rows = []
    for source_method, label in method_labels.items():
        id_row = baseline[
            (baseline["method"] == source_method)
            & (baseline["split"] == "test_id")
        ].iloc[0]
        ood_row = baseline[
            (baseline["method"] == source_method)
            & (baseline["split"] == "test_ood")
        ].iloc[0]
        comparison_rows.append({
            "method": label,
            "id_rmse_mps": id_row["rmse_3d"],
            "id_jitter_mps": id_row["jitter"],
            "ood_rmse_mps": ood_row["rmse_3d"],
            "ood_jitter_mps": ood_row["jitter"],
            "ood_max_jump_mps": ood_row["max_jump"],
            "provenance": provenance,
        })
    pd.DataFrame(comparison_rows).to_csv(
        DESTINATION / "Table09_causal_smoothing_comparison.csv", index=False
    )

    sensitivity = source[
        source["method"] == "PIRNN-AKF_sensitivity"
    ].copy()
    expected_sensitivity_rows = (
        2
        * 6
        * 4
    )
    if len(sensitivity) != expected_sensitivity_rows:
        raise ValueError(
            f"Expected {expected_sensitivity_rows} sensitivity rows, "
            f"found {len(sensitivity)}"
        )
    sensitivity["provenance"] = provenance
    sensitivity.to_csv(
        DESTINATION / "AKF_grouped_sensitivity_rows.csv", index=False
    )

    nominal = baseline[baseline["method"] == "PIRNN-AKF"].set_index("split")
    id_sensitivity = sensitivity[sensitivity["split"] == "test_id"]
    ood_sensitivity = sensitivity[sensitivity["split"] == "test_ood"].copy()
    ood_nominal_jump = float(nominal.loc["test_ood", "max_jump"])
    ood_sensitivity["jump_change_pct"] = (
        100.0
        * (ood_sensitivity["max_jump"] / ood_nominal_jump - 1.0)
    )
    gate = sensitivity[sensitivity["group"] == "mahalanobis_gate"]
    gate_max_change = 0.0
    for _, row in gate.iterrows():
        reference = nominal.loc[row["split"]]
        gate_max_change = max(
            gate_max_change,
            abs(float(row["rmse_3d"]) - float(reference["rmse_3d"])),
            abs(float(row["jitter"]) - float(reference["jitter"])),
            abs(float(row["max_jump"]) - float(reference["max_jump"])),
        )
    largest_jump = ood_sensitivity.loc[
        ood_sensitivity["jump_change_pct"].idxmax()
    ]
    pd.DataFrame([
        (
            "Test-ID RMSE",
            "m/s",
            id_sensitivity["rmse_3d"].min(),
            id_sensitivity["rmse_3d"].max(),
            nominal.loc["test_id", "rmse_3d"],
            "all six groups, multipliers 0.6/0.8/1.2/1.4",
        ),
        (
            "Test-OOD RMSE",
            "m/s",
            ood_sensitivity["rmse_3d"].min(),
            ood_sensitivity["rmse_3d"].max(),
            nominal.loc["test_ood", "rmse_3d"],
            "all six groups, multipliers 0.6/0.8/1.2/1.4",
        ),
        (
            "Maximum OOD jump increase",
            "%",
            0.0,
            ood_sensitivity["jump_change_pct"].max(),
            0.0,
            (
                f"largest change: {largest_jump['group']} at "
                f"x{largest_jump['multiplier']:.1f}"
            ),
        ),
        (
            "Mahalanobis-gate effect",
            "maximum absolute metric change",
            0.0,
            gate_max_change,
            0.0,
            "gate not activated in retained sessions",
        ),
    ], columns=[
        "quantity", "unit", "observed_min", "observed_max", "nominal",
        "scope",
    ]).assign(provenance=provenance).to_csv(
        DESTINATION / "AKF_grouped_sensitivity_summary.csv", index=False
    )
    shutil.copy2(AKF_SELECTION, DESTINATION / "AKF_selection_manifest.json")


def copy_main_statistics() -> None:
    mappings = {
        "seed_level_rmse.csv": "MainStatistics_seed_level_rmse.csv",
        "seed_level_rmse_summary.csv":
            "MainStatistics_seed_level_rmse_summary.csv",
        "moving_block_bootstrap_ci.csv":
            "MainStatistics_moving_block_bootstrap_ci.csv",
        "paired_window_wilcoxon.csv":
            "MainStatistics_paired_window_wilcoxon.csv",
        "statistics_manifest.json": "MainStatistics_manifest.json",
    }
    missing = [
        source_name for source_name in mappings
        if not (MAIN_STATISTICS / source_name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Run scripts/analyze_revision_main_statistics.py before building "
            f"the release; missing: {', '.join(missing)}"
        )
    for source_name, destination_name in mappings.items():
        shutil.copy2(
            MAIN_STATISTICS / source_name,
            DESTINATION / destination_name,
        )


def build_hitl_sources() -> None:
    campaign = ROOT / "HITL/sessions/campaign_summary"
    session = pd.read_csv(campaign / "session_metrics.csv")
    overview = pd.read_csv(
        campaign / "figureF1_hitl_session_overview_source_data.csv"
    )
    extra = overview[[
        "session_id", "truth_horizontal_wind_mps", "magnitude_ratio"
    ]]
    session = session.merge(extra, on="session_id", how="left", validate="one_to_one")
    session.to_csv(DESTINATION / "Table11_HITL_session_metrics.csv", index=False)
    overview.to_csv(
        DESTINATION / "FigureA5_HITL_session_overview.csv", index=False
    )

    numeric = [
        column for column in session.columns
        if column not in ("session_id", "condition")
    ]
    summary_rows = []
    for condition, group in session.groupby("condition", sort=False):
        row: dict[str, object] = {
            "condition": condition,
            "session_count": len(group),
        }
        for column in numeric:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_sd"] = float(group[column].std(ddof=1))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        DESTINATION / "Table11_HITL_condition_summary.csv", index=False
    )

    frame_parts = []
    for session_id in [
        *(f"id_{index:02d}" for index in range(1, 6)),
        *(f"ood_{index:02d}" for index in range(1, 6)),
    ]:
        source = ROOT / f"HITL/sessions/{session_id}/aligned/aligned.csv"
        frame = pd.read_csv(source, low_memory=False)
        monotonic = frame["pi_monotonic_ns"].to_numpy(np.float64)
        keep = monotonic - monotonic[0] >= 20.0e9
        compact = frame.loc[keep, [
            "session_id", "condition", "pi_monotonic_ns",
            "inference_latency_ms", "companion_processing_latency_ms",
            "deadline_missed",
        ]].copy()
        compact["session_time_after_warmup_s"] = (
            compact["pi_monotonic_ns"].to_numpy(np.float64)
            - monotonic[0] - 20.0e9
        ) / 1.0e9
        frame_parts.append(compact)
    pd.concat(frame_parts, ignore_index=True).to_csv(
        DESTINATION / "Figure07_HITL_latency_per_frame.csv",
        index=False,
        float_format="%.9g",
    )


def write_provenance() -> None:
    rows = [
        (
            "Figure02_Table04_main_metrics_per_seed.csv and "
            "Figure02_KalmanNet_predictions_*.csv",
            "data/revision_main_41d/predictions/*.npz, ekf_predictions.npz, "
            "and data/revision_kalmannet_41d/*.npy",
            "derived",
        ),
        (
            "Table05_cross_configuration_*.csv",
            "data/revision_cross_airframe/cross_airframe_predictions.npz",
            "prediction metrics derived; closure RMSE remains aggregate-only",
        ),
        (
            "Table09_causal_smoothing_*.csv and "
            "AKF_grouped_sensitivity_*.csv",
            "data/revision_akf/akf_sensitivity_and_baselines.csv",
            "summary and all 63 source rows included",
        ),
        (
            "MainStatistics_*.csv",
            "data/revision_main_41d/statistics",
            "five-seed, moving-block bootstrap and paired-window outputs",
        ),
        (
            "Table11/Figure07/FigureA5 HITL CSV files",
            "HITL/sessions/* and campaign_summary",
            "derived from ten aligned sessions",
        ),
        (
            "Retained experiment CSV files",
            "submission/minimal_dataset",
            "unchanged experiments retained in current manuscript",
        ),
    ]
    pd.DataFrame(rows, columns=["artifact", "local_source", "status"]).to_csv(
        DESTINATION / "SOURCE_PROVENANCE.csv", index=False
    )


def normalize_csv_line_endings() -> None:
    """Keep committed CSVs platform-neutral and friendly to Git tooling."""
    for path in DESTINATION.glob("*.csv"):
        content = path.read_bytes()
        normalized = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if normalized != content:
            path.write_bytes(normalized)


def verify_and_package() -> None:
    verifier = DESTINATION / "verify_reported_values.py"
    completed = subprocess.run(
        [sys.executable, str(verifier)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    verification_output = completed.stdout
    (DESTINATION / "verification_output.txt").write_text(
        verification_output, encoding="utf-8"
    )

    generated_files = {
        "Figure02_Table04_main_metrics_per_seed.csv",
        "Figure02_Table04_main_metrics_aggregated.csv",
        "Figure02_KalmanNet_predictions_test_id.csv",
        "Figure02_KalmanNet_predictions_test_ood.csv",
        "Table01_AKF_constants.csv",
        "Table02_Rascal_parameters.csv",
        "Table03_model_configuration.csv",
        "Table05_cross_configuration_by_session.csv",
        "Table05_cross_configuration_by_seed.csv",
        "Table05_cross_configuration_summary.csv",
        "Table09_causal_smoothing_comparison.csv",
        "Table09_causal_smoothing_rows.csv",
        "AKF_grouped_sensitivity_summary.csv",
        "AKF_grouped_sensitivity_rows.csv",
        "AKF_selection_manifest.json",
        "Table11_HITL_session_metrics.csv",
        "Table11_HITL_condition_summary.csv",
        "Figure07_HITL_latency_per_frame.csv",
        "FigureA5_HITL_session_overview.csv",
        "MainStatistics_seed_level_rmse.csv",
        "MainStatistics_seed_level_rmse_summary.csv",
        "MainStatistics_moving_block_bootstrap_ci.csv",
        "MainStatistics_paired_window_wilcoxon.csv",
        "MainStatistics_manifest.json",
        "SOURCE_PROVENANCE.csv",
        "README.md",
        "verify_reported_values.py",
        "verification_output.txt",
    }
    package_files = sorted(set(RETAINED_CSVS) | generated_files)
    missing = [
        name for name in package_files
        if not (DESTINATION / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot package incomplete minimal dataset: "
            + ", ".join(missing)
        )
    archive_path = SUBMISSION / "minimal_dataset.zip"
    with zipfile.ZipFile(
        archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in package_files:
            archive.write(
                DESTINATION / name,
                arcname=f"minimal_dataset/{name}",
            )
    print(verification_output.rstrip())
    print(
        f"Packaged {len(package_files)} files in "
        f"{archive_path.relative_to(ROOT)}"
    )


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    missing = [
        name for name in RETAINED_CSVS
        if not (DESTINATION / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "The committed retained experiment CSVs are incomplete: "
            + ", ".join(missing)
        )
    build_main_comparison()
    build_cross_configuration()
    write_configuration_tables()
    write_revision_summaries()
    copy_main_statistics()
    build_hitl_sources()
    write_provenance()
    normalize_csv_line_endings()
    shutil.copy2(
        ROOT / "scripts/verify_minimal_dataset.py",
        DESTINATION / "verify_reported_values.py",
    )
    verify_and_package()
    print(f"Built compact CSV package in {DESTINATION}")


if __name__ == "__main__":
    main()
