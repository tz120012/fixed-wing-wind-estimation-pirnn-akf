# Minimal Reproducibility Dataset — drones-4507054 Revision

Supporting CSV data for:

**Local Three-Dimensional Wind Estimation for Fixed-Wing UAVs via a
Physics-Informed GRU with Measurement-Noise-Adaptive Kalman Smoothing**

All CSV files are UTF-8 with a header row. The package is aligned to the
current 41-input, five-seed manuscript rather than the superseded three-seed
submission.

## Verification

Run from the repository root:

```bash
python Paper_2/MDPI_template_APA/submission/minimal_dataset/verify_reported_values.py
```

The retained `verification_output.txt` records the current result:
`302 checks agree, 0 differ`.

The six-decimal jitter values printed in Table 9 give a 24.7% reduction when
used as already-rounded inputs; the manuscript's 24.8% value is compatible
with rounding of the underlying unprinted values. The verifier therefore uses
an explicit 0.06 percentage-point tolerance for this one derived percentage.

## Current-revision sources

- `Figure02_Table04_main_metrics_per_seed.csv`: seed-level Test-ID/Test-OOD
  metrics recalculated from the archived five-seed Vanilla GRU, Vanilla LSTM,
  PI-GRU and PIRNN-AKF prediction arrays, plus deterministic PX4-EKF2.
- `Figure02_Table04_main_metrics_aggregated.csv`: Figure 2 and Table 4 plotting
  values derived from the archived predictions for every method.
- `Figure02_KalmanNet_predictions_test_id.csv` and
  `Figure02_KalmanNet_predictions_test_ood.csv`: the full five-seed,
  per-sample 41-input KalmanNet predictions together with the shared truth
  arrays. These files independently reproduce the KalmanNet rows in the
  per-seed and aggregate Figure 2/Table 4 sources.
- `Table05_cross_configuration_by_session.csv`,
  `Table05_cross_configuration_by_seed.csv` and
  `Table05_cross_configuration_summary.csv`: Table 5 Rascal-to-Malolo metrics
  recovered from the archived prediction arrays. RMSE, direction error,
  jitter and maximum jump are derived; closure RMSE remains an audited
  aggregate because the archive lacks the physical-feature array.
- `Table09_causal_smoothing_comparison.csv`: Table 9 causal-smoother summary
  derived from `Table09_causal_smoothing_rows.csv`, which includes the
  validation, Test-ID and Test-OOD result row for every smoother.
- `AKF_grouped_sensitivity_summary.csv`: reported nominal values, ranges and
  largest transient change derived from all 48 perturbation rows in
  `AKF_grouped_sensitivity_rows.csv`. `AKF_selection_manifest.json` records the
  validation-locked baseline parameters and grouped perturbation protocol.
- `MainStatistics_seed_level_rmse.csv`,
  `MainStatistics_seed_level_rmse_summary.csv`,
  `MainStatistics_moving_block_bootstrap_ci.csv` and
  `MainStatistics_paired_window_wilcoxon.csv`: the five-seed and sample-order
  uncertainty outputs used in the statistical statements.
- `Table11_HITL_session_metrics.csv`: all ten independent HITL sessions,
  augmented with truth-wind magnitude and estimate/truth magnitude ratio.
- `Table11_HITL_condition_summary.csv`: condition-level mean and unbiased
  sample SD recomputed with session as the replicate.
- `Figure07_HITL_latency_per_frame.csv`: 251,176 post-warm-up frames used for
  the final-model latency CDF and deadline results.
- `FigureA5_HITL_session_overview.csv`: exact ten points plotted in Appendix F.
- `Table01_AKF_constants.csv`, `Table02_Rascal_parameters.csv` and
  `Table03_model_configuration.csv`: compact machine-readable copies of the
  frozen settings reported in Tables 1–3.

## Retained experiments used by the revision

- `Figure03_Table06_physics_weight_sweep.csv`: physics-loss sweep.
- `Table07_feature_group_ablation.csv`: input-feature-group ablation.
- `Figure04_Table08_weak_wind_subset.csv`: weak-wind diagnostics.
- `Figure05_gps_spike_timeseries_panels.csv`: exact plotted anomaly series.
- `Figure05_anomaly_window_diagnostics_timeseries.csv`: AKF diagnostics.
- `Figure06_Table10_transient_tracking_vs_jitter.csv`: transient/jitter means.
- `Table10_paired_wilcoxon_tests.csv`: paired window tests.
- `RobustSmoothing_regime_windows.csv` and
  `RobustSmoothing_fixed_EMA_vs_AKF_cross_scenario.csv`: high/low-dynamics and
  deliberately strong fixed-EMA comparisons.
- `Figure08_replay_test_ID_perframe.csv` and
  `Figure08_replay_test_OOD_perframe.csv`: representative replay records.
- `Archived_online_SITL_perframe.csv` and
  `Archived_online_SITL_reference_wind.csv`: predecessor desktop online SITL
  accuracy record explicitly identified as such in the manuscript.
- `Predecessor_edge_latency_by_backend.csv` and
  `Predecessor_latency_per_frame_by_backend.csv`: predecessor 20-input,
  50-step compute reference retained only for the historical comparison.
- `TableA1_dynamic_QR_ablation.csv`,
  `FigureA1_dynamic_Q_volatility_regime_summary.csv` and
  `FigureA1_dynamic_Q_volatility_per_window.csv`: dynamic-Q/R ablations.
- `TableB1_FigureB1_multi_anomaly_robustness.csv`,
  `TableB1_PX4_EKF2_per_window_rows.csv` and
  `TableB1_KalmanNet_per_window_rows.csv`: multi-anomaly robustness. The
  KalmanNet anomaly rows belong to the retained legacy seed-26 anomaly
  experiment and are distinct from the final five-seed 41-input main
  comparison sources above.
- `FigureD1_anomaly_strength_sweep.csv`: anomaly-strength sweep.
- `TableE1_cross_platform_forward_latency.csv`: early cross-platform
  forward-pass benchmark.

## Provenance limits

`SOURCE_PROVENANCE.csv` distinguishes values recomputed from local
predictions/logs from aggregate-only records. No artificial seed-level rows
were generated.

The final five-seed KalmanNet predictions and every row of the grouped AKF
sensitivity and causal-smoother experiment are included. The remaining
aggregate-only item is the Malolo airspeed-closure residual: the package
contains the cross-configuration predictions and session/seed metrics, but the
physical-feature array required to recompute that residual is not present in
the retained archive. The complete Rascal data lineage---38.54 GiB
(41.38 GB) acquisition JSON, 9.34 GiB (10.03 GB) deterministic training CSV,
and 10.76 GiB (11.56 GB) final 41-input arrays---and the training checkpoints
are also outside this compact supporting-data file. The full data are publicly available as six independently downloadable
archives under the Zenodo data DOI <https://doi.org/10.5281/zenodo.22338389>. The code and
checkpoints are publicly available at
<https://github.com/tz120012/fixed-wing-wind-estimation-pirnn-akf> and archived under the
Zenodo software DOI <https://doi.org/10.5281/zenodo.22463526>.
