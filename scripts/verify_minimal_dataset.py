#!/usr/bin/env python3
"""Verify principal reported values from the current compact CSV package."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = (
    HERE
    if (HERE / "Figure02_Table04_main_metrics_per_seed.csv").is_file()
    else HERE.parents[0]
    / "Paper_2/MDPI_template_APA/submission/minimal_dataset"
)


class Verifier:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.rows: list[tuple[str, str, str, str, bool]] = []

    def load(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.root / name, low_memory=False)

    def check(
        self,
        element: str,
        quantity: str,
        reported: float,
        computed: float,
        decimals: int = 3,
        tolerance: float | None = None,
    ) -> None:
        scale = 10**decimals
        shown_reported = f"{reported:.{decimals}f}"
        shown_computed = f"{computed:.{decimals}f}"
        if tolerance is None:
            ok = abs(round(computed * scale) / scale - reported) < 0.5 / scale
        else:
            ok = abs(computed - reported) <= tolerance
        self.rows.append(
            (element, quantity, shown_reported, shown_computed, ok)
        )

    def verify_main(self) -> None:
        per_seed = self.load(
            "Figure02_Table04_main_metrics_per_seed.csv"
        )
        aggregate = self.load(
            "Figure02_Table04_main_metrics_aggregated.csv"
        ).set_index(["method", "split"])
        reported = {
            ("PX4-EKF2", "test_id"): (0.649, 0.0, 25.24),
            ("PX4-EKF2", "test_ood"): (1.387, 0.0, 24.12),
            ("KalmanNet", "test_id"): (0.386, 0.011, 13.13),
            ("KalmanNet", "test_ood"): (0.485, 0.055, 5.91),
            ("Vanilla GRU", "test_id"): (0.256, 0.008, 3.82),
            ("Vanilla GRU", "test_ood"): (0.870, 0.041, 3.22),
            ("Vanilla LSTM", "test_id"): (0.244, 0.005, 3.75),
            ("Vanilla LSTM", "test_ood"): (0.813, 0.027, 4.19),
            ("PI-GRU", "test_id"): (0.208, 0.003, 3.22),
            ("PI-GRU", "test_ood"): (0.546, 0.007, 1.94),
            ("PIRNN-AKF", "test_id"): (0.215, 0.003, 3.29),
            ("PIRNN-AKF", "test_ood"): (0.544, 0.007, 1.97),
        }
        for key, (rmse, rmse_sd, direction) in reported.items():
            row = aggregate.loc[key]
            self.check("Figure 2 / Table 4", f"{key} RMSE", rmse,
                       row["rmse_3d_mps_mean"])
            self.check("Figure 2 / Table 4", f"{key} RMSE SD", rmse_sd,
                       row["rmse_3d_mps_sd"])
            self.check("Figure 2 / Table 4", f"{key} direction MAE",
                       direction, row["direction_mae_deg_mean"], 2)

        derived = per_seed[
            per_seed["method"].isin(
                [
                    "KalmanNet", "Vanilla GRU", "Vanilla LSTM",
                    "PI-GRU", "PIRNN-AKF",
                ]
            )
        ]
        grouped = derived.groupby(["method", "split"])["rmse_3d_mps"]
        for key, values in grouped:
            row = aggregate.loc[key]
            self.check("Figure 2 source", f"{key} per-seed mean",
                       row["rmse_3d_mps_mean"], values.mean(), 9)
            self.check("Figure 2 source", f"{key} per-seed SD",
                       row["rmse_3d_mps_sd"], values.std(ddof=1), 9)

        indexed = per_seed.set_index(["method", "split", "seed"])
        for split in ("test_id", "test_ood"):
            raw = self.load(
                f"Figure02_KalmanNet_predictions_{split}.csv"
            )
            truth = raw[
                ["truth_n_mps", "truth_e_mps", "truth_d_mps"]
            ].to_numpy(np.float64)
            for seed in (26, 42, 2024, 2025, 2026):
                estimate = raw[
                    [
                        f"seed{seed}_n_mps",
                        f"seed{seed}_e_mps",
                        f"seed{seed}_d_mps",
                    ]
                ].to_numpy(np.float64)
                error = estimate - truth
                rmse = float(np.sqrt(np.mean(error**2)))
                mask = np.hypot(truth[:, 0], truth[:, 1]) >= 0.5
                true_angle = np.degrees(
                    np.arctan2(truth[mask, 1], truth[mask, 0])
                )
                estimate_angle = np.degrees(
                    np.arctan2(estimate[mask, 1], estimate[mask, 0])
                )
                direction = float(np.mean(np.abs(
                    (estimate_angle - true_angle + 180.0) % 360.0 - 180.0
                )))
                source_row = indexed.loc[("KalmanNet", split, str(seed))]
                self.check(
                    "KalmanNet raw source",
                    f"{split} seed {seed} RMSE",
                    source_row["rmse_3d_mps"],
                    rmse,
                    7,
                )
                self.check(
                    "KalmanNet raw source",
                    f"{split} seed {seed} direction MAE",
                    source_row["direction_mae_deg"],
                    direction,
                    7,
                    tolerance=1e-6,
                )

        ood = aggregate.loc[("Vanilla GRU", "test_ood"), "rmse_3d_mps_mean"]
        pi = aggregate.loc[("PI-GRU", "test_ood"), "rmse_3d_mps_mean"]
        akf = aggregate.loc[("PIRNN-AKF", "test_ood"), "rmse_3d_mps_mean"]
        self.check("Results", "PI-GRU OOD reduction (%)", 37.2,
                   100.0 * (ood - pi) / ood, 1)
        self.check("Results", "PIRNN-AKF OOD reduction (%)", 37.4,
                   100.0 * (ood - akf) / ood, 1)

        statistics = self.load(
            "MainStatistics_paired_window_wilcoxon.csv"
        )
        paired = statistics[
            (statistics["split"] == "test_ood")
            & (
                statistics["comparison"]
                == "PIRNN-AKF vs PI-GRU"
            )
        ].iloc[0]
        self.check(
            "Main statistics",
            "Test-OOD PIRNN-AKF vs PI-GRU paired-window p-value",
            0.485,
            paired["p_value"],
            3,
        )

    def verify_cross_configuration(self) -> None:
        data = self.load(
            "Table05_cross_configuration_summary.csv"
        ).set_index(["condition", "method"])
        by_seed = self.load(
            "Table05_cross_configuration_by_seed.csv"
        )
        expected = {
            ("ID wind", "PI-GRU"):
                (0.756, 0.077, 19.40, 7.75, 0.693, 0.040,
                 0.00206, 0.00012),
            ("ID wind", "PIRNN-AKF"):
                (0.751, 0.077, 19.21, 7.77, 0.686, 0.039,
                 0.00154, 0.00008),
            ("Stronger wind/gust", "PI-GRU"):
                (0.602, 0.065, 10.01, 1.30, 0.664, 0.079,
                 0.00158, 0.00008),
            ("Stronger wind/gust", "PIRNN-AKF"):
                (0.599, 0.065, 9.96, 1.29, 0.655, 0.078,
                 0.00116, 0.00005),
        }
        for key, values in expected.items():
            row = data.loc[key]
            for name, reported, decimals in zip(
                [
                    "rmse_3d_mean_mps", "rmse_3d_sd_mps",
                    "direction_mae_mean_deg", "direction_mae_sd_deg",
                    "closure_rmse_mean_mps", "closure_rmse_sd_mps",
                    "jitter_mean_mps", "jitter_sd_mps",
                ],
                values,
                [3, 3, 2, 2, 3, 3, 5, 5],
            ):
                self.check("Table 5", f"{key} {name}", reported, row[name],
                           decimals)
            group = by_seed[
                (by_seed["condition"] == key[0])
                & (by_seed["method"] == key[1])
            ]
            for source_field, mean_field, sd_field in [
                ("rmse_3d_mps", "rmse_3d_mean_mps", "rmse_3d_sd_mps"),
                (
                    "direction_mae_deg",
                    "direction_mae_mean_deg",
                    "direction_mae_sd_deg",
                ),
                ("jitter_mps", "jitter_mean_mps", "jitter_sd_mps"),
            ]:
                self.check("Table 5 source", f"{key} {source_field} mean",
                           row[mean_field], group[source_field].mean(), 9)
                self.check("Table 5 source", f"{key} {source_field} SD",
                           row[sd_field],
                           group[source_field].std(ddof=1), 9)

    def verify_ablations(self) -> None:
        physics = self.load(
            "Figure03_Table06_physics_weight_sweep.csv"
        ).set_index("lambda_physics")
        self.check("Figure 3 / Table 6", "direction MAE at lambda=0.09",
                   3.75, physics.loc[0.09, "dir_mae"], 2)
        self.check("Figure 3 / Table 6", "closure RMSE at lambda=0.13",
                   0.345, physics.loc[0.13, "physics_residual"])

        feature = self.load("Table07_feature_group_ablation.csv")
        expected_feature = [(0.221, 0.574), (0.210, 0.558), (0.223, 0.610)]
        for index, (id_rmse, ood_rmse) in enumerate(expected_feature):
            self.check("Table 7", f"row {index + 1} ID RMSE", id_rmse,
                       feature.iloc[index]["test_id_rmse_mps"])
            self.check("Table 7", f"row {index + 1} OOD RMSE", ood_rmse,
                       feature.iloc[index]["test_ood_rmse_mps"])

        weak = self.load(
            "Figure04_Table08_weak_wind_subset.csv"
        )
        for index, values in enumerate([
            (0.188, 20.36, 0.50),
            (0.176, 20.20, 0.24),
        ]):
            row = weak.iloc[index]
            self.check("Figure 4 / Table 8", f"row {index + 1} RMSE",
                       values[0], row["weak_rmse_mps"])
            self.check("Figure 4 / Table 8", f"row {index + 1} direction P95",
                       values[1], row["direction_p95_deg"], 2)
            self.check("Figure 4 / Table 8", f"row {index + 1} collapse (%)",
                       values[2], row["collapse_ratio_pct"], 2)

    def verify_smoothing(self) -> None:
        data = self.load(
            "Table09_causal_smoothing_comparison.csv"
        ).set_index("method")
        expected = {
            "PI-GRU": (0.2048, 0.002694, 0.5393, 0.002465, 0.1581),
            "Fixed EMA (alpha=1.0)":
                (0.2048, 0.002694, 0.5393, 0.002465, 0.1581),
            "Confidence complementary":
                (0.2049, 0.002120, 0.5393, 0.001970, 0.1571),
            "Fixed-covariance KF":
                (0.2049, 0.001890, 0.5393, 0.001768, 0.1566),
            "PIRNN-AKF":
                (0.2116, 0.002051, 0.5360, 0.001855, 0.1473),
        }
        columns = [
            "id_rmse_mps", "id_jitter_mps", "ood_rmse_mps",
            "ood_jitter_mps", "ood_max_jump_mps",
        ]
        for method, values in expected.items():
            for column, reported in zip(columns, values):
                self.check("Table 9", f"{method} {column}", reported,
                           data.loc[method, column], 6 if "jitter" in column
                           else 4)
        pi = data.loc["PI-GRU"]
        akf = data.loc["PIRNN-AKF"]
        self.check("Table 9", "OOD jitter reduction (%, rounded inputs)", 24.8,
                   100 * (pi["ood_jitter_mps"] - akf["ood_jitter_mps"])
                   / pi["ood_jitter_mps"], 1, tolerance=0.06)
        self.check("Table 9", "OOD max-jump reduction (%)", 6.8,
                   100 * (pi["ood_max_jump_mps"] - akf["ood_max_jump_mps"])
                   / pi["ood_max_jump_mps"], 1)

        source_rows = self.load("Table09_causal_smoothing_rows.csv")
        for method in expected:
            for split, prefix in (("test_id", "id"), ("test_ood", "ood")):
                source = source_rows[
                    (source_rows["method_label"] == method)
                    & (source_rows["split"] == split)
                ].iloc[0]
                self.check(
                    "Table 9 raw source",
                    f"{method} {split} RMSE",
                    data.loc[method, f"{prefix}_rmse_mps"],
                    source["rmse_3d"],
                    9,
                )
                self.check(
                    "Table 9 raw source",
                    f"{method} {split} jitter",
                    data.loc[method, f"{prefix}_jitter_mps"],
                    source["jitter"],
                    9,
                )
                if split == "test_ood":
                    self.check(
                        "Table 9 raw source",
                        f"{method} {split} max jump",
                        data.loc[method, "ood_max_jump_mps"],
                        source["max_jump"],
                        9,
                    )

        sensitivity = self.load(
            "AKF_grouped_sensitivity_summary.csv"
        ).set_index("quantity")
        self.check("AKF sensitivity", "ID RMSE minimum", 0.209,
                   sensitivity.loc["Test-ID RMSE", "observed_min"])
        self.check("AKF sensitivity", "ID RMSE maximum", 0.214,
                   sensitivity.loc["Test-ID RMSE", "observed_max"])
        self.check("AKF sensitivity", "OOD RMSE minimum", 0.535,
                   sensitivity.loc["Test-OOD RMSE", "observed_min"])
        self.check("AKF sensitivity", "OOD RMSE maximum", 0.537,
                   sensitivity.loc["Test-OOD RMSE", "observed_max"])
        self.check("AKF sensitivity", "maximum jump increase (%)", 37.2,
                   sensitivity.loc[
                       "Maximum OOD jump increase", "observed_max"
                   ], 1)

        sensitivity_rows = self.load("AKF_grouped_sensitivity_rows.csv")
        self.check(
            "AKF sensitivity raw source",
            "row count",
            48,
            len(sensitivity_rows),
            0,
        )
        for split, summary_name in (
            ("test_id", "Test-ID RMSE"),
            ("test_ood", "Test-OOD RMSE"),
        ):
            source = sensitivity_rows[
                sensitivity_rows["split"] == split
            ]
            self.check(
                "AKF sensitivity raw source",
                f"{split} RMSE minimum",
                sensitivity.loc[summary_name, "observed_min"],
                source["rmse_3d"].min(),
                9,
            )
            self.check(
                "AKF sensitivity raw source",
                f"{split} RMSE maximum",
                sensitivity.loc[summary_name, "observed_max"],
                source["rmse_3d"].max(),
                9,
            )
        nominal_jump = data.loc["PIRNN-AKF", "ood_max_jump_mps"]
        ood_rows = sensitivity_rows[
            sensitivity_rows["split"] == "test_ood"
        ]
        jump_increase = float(
            100.0 * (ood_rows["max_jump"] / nominal_jump - 1.0).max()
        )
        self.check(
            "AKF sensitivity raw source",
            "maximum OOD jump increase",
            sensitivity.loc[
                "Maximum OOD jump increase", "observed_max"
            ],
            jump_increase,
            7,
        )

        transient = self.load(
            "Figure06_Table10_transient_tracking_vs_jitter.csv"
        ).set_index("method")
        for method, rmse, jitter in [
            ("PI-GRU (Raw)", 0.585, 0.0511),
            ("EMA a=0.1", 0.806, 0.0046),
            ("EMA a=0.5", 0.609, 0.0233),
            ("EMA a=0.9", 0.587, 0.0448),
            ("PIRNN-AKF", 0.585, 0.0433),
        ]:
            self.check("Figure 6 / Table 10", f"{method} RMSE", rmse,
                       transient.loc[method, "transient_rmse_mean"])
            self.check("Figure 6 / Table 10", f"{method} jitter", jitter,
                       transient.loc[method, "anomaly_jitter_mean"], 4)

    def verify_timeseries_and_replay(self) -> None:
        spike = self.load("Figure05_gps_spike_timeseries_panels.csv")
        self.check("Figure 5", "rows", 1000, len(spike), 0)
        self.check("Figure 5", "sampling interval", 0.02,
                   np.diff(spike["time_s"]).mean(), 2)
        window = spike["in_anomaly_window"].astype(bool)
        self.check("Figure 5", "anomaly start", 9.00,
                   spike.loc[window, "time_s"].min(), 2)
        self.check("Figure 5", "anomaly end", 11.00,
                   spike.loc[window, "time_s"].max(), 2)

        for split, filename, count, overall in [
            ("Test-ID", "Figure08_replay_test_ID_perframe.csv", 60369, 0.226),
            ("Test-OOD", "Figure08_replay_test_OOD_perframe.csv", 42574, 0.570),
        ]:
            frame = self.load(filename)
            truth = frame[
                ["wind_true_n", "wind_true_e", "wind_true_d"]
            ].to_numpy(float)
            estimate = frame[
                ["wind_est_n", "wind_est_e", "wind_est_d"]
            ].to_numpy(float)
            self.check("Figure 8", f"{split} rows", count, len(frame), 0)
            self.check("Figure 8", f"{split} RMSE", overall,
                       np.sqrt(np.mean((estimate - truth) ** 2)))

    def verify_hitl(self) -> None:
        session = self.load(
            "Table11_HITL_session_metrics.csv"
        )
        summary = self.load(
            "Table11_HITL_condition_summary.csv"
        ).set_index("condition")
        expected = {
            "id": [
                ("post_warmup_rows", 25188, 778, 0, 1.0),
                ("rmse_n_mps", 0.614, 0.081, 3, 1.0),
                ("rmse_e_mps", 0.760, 0.102, 3, 1.0),
                ("rmse_d_mps", 0.109, 0.007, 3, 1.0),
                ("rmse_3d_mps", 0.569, 0.054, 3, 1.0),
                ("direction_mae_deg", 19.0, 6.0, 1, 1.0),
                ("magnitude_ratio", 0.967, 0.118, 3, 1.0),
                ("inference_latency_mean_ms", 9.62, 0.32, 2, 1.0),
                ("inference_latency_p95_ms", 18.3, 0.4, 1, 1.0),
                ("companion_latency_mean_ms", 12.18, 0.39, 2, 1.0),
                ("companion_latency_p95_ms", 20.3, 0.5, 1, 1.0),
                ("loop_hz", 83.2, 2.0, 1, 1.0),
                ("deadline_miss_ratio", 5.5, 0.7, 1, 100.0),
            ],
            "ood": [
                ("post_warmup_rows", 25047, 616, 0, 1.0),
                ("rmse_n_mps", 0.945, 0.637, 3, 1.0),
                ("rmse_e_mps", 0.948, 0.496, 3, 1.0),
                ("rmse_d_mps", 0.085, 0.014, 3, 1.0),
                ("rmse_3d_mps", 0.832, 0.318, 3, 1.0),
                ("direction_mae_deg", 6.19, 2.46, 2, 1.0),
                ("magnitude_ratio", 0.818, 0.096, 3, 1.0),
                ("inference_latency_mean_ms", 9.69, 0.25, 2, 1.0),
                ("inference_latency_p95_ms", 18.6, 0.5, 1, 1.0),
                ("companion_latency_mean_ms", 12.26, 0.31, 2, 1.0),
                ("companion_latency_p95_ms", 20.5, 0.5, 1, 1.0),
                ("loop_hz", 83.2, 1.4, 1, 1.0),
                ("deadline_miss_ratio", 5.8, 0.8, 1, 100.0),
            ],
        }
        for condition, checks in expected.items():
            row = summary.loc[condition]
            self.check("Table 11", f"{condition} sessions", 5,
                       row["session_count"], 0)
            for field, mean, standard_deviation, decimals, scale in checks:
                self.check("Table 11", f"{condition} {field} mean", mean,
                           scale * row[f"{field}_mean"], decimals)
                self.check("Table 11", f"{condition} {field} SD",
                           standard_deviation,
                           scale * row[f"{field}_sd"], decimals)

        self.check("HITL aggregate", "session count", 10, len(session), 0)
        self.check("HITL aggregate", "inference mean (ms)", 9.65,
                   session["inference_latency_mean_ms"].mean(), 2)
        self.check("HITL aggregate", "companion mean (ms)", 12.22,
                   session["companion_latency_mean_ms"].mean(), 2)
        self.check("HITL aggregate", "companion p95 (ms)", 20.4,
                   session["companion_latency_p95_ms"].mean(), 1)
        self.check("HITL aggregate", "loop rate (Hz)", 83.2,
                   session["loop_hz"].mean(), 1)
        self.check("HITL aggregate", "deadline misses (%)", 5.6,
                   100 * session["deadline_miss_ratio"].mean(), 1)

        latency = self.load("Figure07_HITL_latency_per_frame.csv")
        self.check("Figure 7", "post-warm-up rows",
                   int(session["post_warmup_rows"].sum()), len(latency), 0)
        self.check("Figure 7", "session count", 10,
                   latency["session_id"].nunique(), 0)

    def verify_appendices(self) -> None:
        appendix_a = self.load(
            "TableA1_dynamic_QR_ablation.csv"
        ).set_index("method")
        for method, rmse, jitter in [
            ("AKF fixed Q/R", 0.596, 0.0730),
            ("AKF dynamic Q only", 0.596, 0.0730),
            ("AKF dynamic R only", 0.601, 0.0712),
            ("AKF dynamic Q/R", 0.601, 0.0712),
        ]:
            self.check("Table A1", f"{method} RMSE", rmse,
                       appendix_a.loc[method, "h_rmse_mean"])
            self.check("Table A1", f"{method} jitter", jitter,
                       appendix_a.loc[method, "jitter_mean"], 4)

        appendix_b = self.load(
            "TableB1_FigureB1_multi_anomaly_robustness.csv"
        ).set_index("method")
        self.check("Table B1 / Figure B1", "PIRNN-AKF jitter reduction (%)",
                   21.5,
                   appendix_b.loc[
                       "PIRNN-AKF",
                       "jitter_reduction_vs_pigru_pct_mean",
                   ], 1)

        strength = self.load("FigureD1_anomaly_strength_sweep.csv")
        gps8 = strength[
            (strength["anomaly_type"] == "gps_spike")
            & (strength["strength"] == 8)
        ].set_index("method")
        self.check("Figure D1", "GPS-8 PIRNN-AKF RMSE", 0.715,
                   gps8.loc["PIRNN-AKF", "h_rmse_mean"])

        platform = self.load("TableE1_cross_platform_forward_latency.csv")
        if not (platform["mean_per_step_latency_ms"] < 20).all():
            self.rows.append(
                ("Table E1", "all means below 20 ms", "yes", "no", False)
            )

    def finish(self) -> int:
        widths = [
            max(len(row[index]) for row in self.rows + [
                ("Element", "Quantity", "Reported", "Computed", True)
            ])
            for index in range(4)
        ]
        print(
            f"{'Element':<{widths[0]}}  {'Quantity':<{widths[1]}}  "
            f"{'Reported':>{widths[2]}}  {'Computed':>{widths[3]}}  Status"
        )
        print("-" * (sum(widths) + 12))
        for element, quantity, reported, computed, ok in self.rows:
            print(
                f"{element:<{widths[0]}}  {quantity:<{widths[1]}}  "
                f"{reported:>{widths[2]}}  {computed:>{widths[3]}}  "
                f"{'OK' if ok else 'DIFF'}"
            )
        differences = [row for row in self.rows if not row[-1]]
        print("-" * (sum(widths) + 12))
        print(f"{len(self.rows) - len(differences)} checks agree, "
              f"{len(differences)} differ.")
        return 1 if differences else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    verifier = Verifier(args.dataset)
    verifier.verify_main()
    verifier.verify_cross_configuration()
    verifier.verify_ablations()
    verifier.verify_smoothing()
    verifier.verify_timeseries_and_replay()
    verifier.verify_hitl()
    verifier.verify_appendices()
    return verifier.finish()


if __name__ == "__main__":
    raise SystemExit(main())
