from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from HITL.postprocess.align_session import (
    _probe_rows,
    align_session,
    fit_clock_model,
    pi_to_pc_wall_ns,
)
from HITL.postprocess.analyze_campaign import analyze_campaign
from HITL.postprocess.validate_session import validate_session


def campaign_config(root: Path, *, duration: float = 2.0, warmup: float = 0.2) -> dict:
    return {
        "protocol": {
            "name": "test",
            "estimator_rate_hz": 50,
            "total_duration_s": duration,
            "warmup_exclusion_s": warmup,
            "minimum_valid_post_warmup_s": duration - warmup,
            "direction_min_horizontal_wind_mps": 0.5,
            "required_sessions_per_condition": 5,
        },
        "conditions": {
            "id": {
                "horizontal_wind_range_mps": [1.0, 3.0],
                "speeds_mps": [2.0] * 5,
                "directions_deg": [0, 30, 60, 90, 120],
            },
            "ood": {
                "horizontal_wind_range_mps": [4.0, 8.0],
                "speeds_mps": [5.0] * 5,
                "directions_deg": [0, 30, 60, 90, 120],
            },
        },
        "alignment": {
            "max_clock_uncertainty_ms": 20,
            "max_truth_match_residual_ms": 50,
            "minimum_truth_match_ratio": 0.98,
            "max_fc_match_residual_ms": 60,
            "minimum_fc_match_ratio": 0.95,
        },
        "quality": {
            "minimum_achieved_loop_hz": 45,
            "maximum_deadline_miss_ratio": 0.05,
            "maximum_nonfinite_ratio": 0,
            "enforce_exact_session_count": True,
        },
        "logging": {
            "sessions_root": str(root),
            "pi_csv_name": "pi_estimator.csv",
            "pc_truth_name": "jsbsim_truth.csv",
            "fc_ulog_name": "px4.ulg",
            "aligned_csv_name": "aligned.csv",
        },
    }


def write_campaign(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def aligned_frame(session_id: str, *, rate_hz: float = 50.0, duration_s: float = 2.0, wind: float = 2.0) -> pd.DataFrame:
    count = int(duration_s * rate_hz) + 1
    seconds = np.arange(count) / rate_hz
    frame = pd.DataFrame(
        {
            "session_id": session_id,
            "condition": session_id.split("_")[0],
            "pi_wall_time_ns": 1.7e18 + seconds * 1e9,
            "pi_monotonic_ns": 1e12 + seconds * 1e9,
            "fc_boot_time_us": 1e6 + seconds * 1e6,
            "estimated_wind_n_mps": wind,
            "estimated_wind_e_mps": 0.0,
            "estimated_wind_d_mps": 0.0,
            "nn_wind_n_mps": wind,
            "nn_wind_e_mps": 0.0,
            "nn_wind_d_mps": 0.0,
            "inference_latency_ms": 2.0,
            "companion_processing_latency_ms": 3.0,
            "deadline_missed": 0,
            "truth_time_us": 1.7e15 + seconds * 1e6,
            "truth_wind_n_mps": wind,
            "truth_wind_e_mps": 0.0,
            "truth_wind_d_mps": 0.0,
            "truth_alignment_residual_ms": 1.0,
        }
    )
    return frame


class ClockModelTests(unittest.TestCase):
    def test_actual_four_timestamp_probe_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "marker_start.jsonl"
            records = []
            for sample in range(4):
                t1 = 1_700_000_000_000_000_000 + sample * 100_000_000
                records.append(
                    {
                        "protocol_version": 1,
                        "session_id": "id_01",
                        "sample": sample,
                        "t1_pc_send_ns": t1,
                        "t2_pi_recv_ns": t1 + 8_500_000,
                        "t3_pi_send_ns": t1 + 8_600_000,
                        "t4_pc_recv_ns": t1 + 1_100_000,
                        "ok": True,
                    }
                )
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            probes = _probe_rows([path], "id_01")
            self.assertEqual(len(probes), 4)
            np.testing.assert_allclose(probes["pi_minus_pc_ns"], 8_000_000)
            np.testing.assert_allclose(probes["rtt_ns"], 1_000_000, atol=256)

    def test_known_offset_and_drift_recovery(self) -> None:
        pc = 1.7e18 + np.concatenate((np.arange(10) * 1e8, 100e9 + np.arange(10) * 1e8))
        reference = np.median(pc)
        offset_ns = 8_000_000.0
        slope = 12e-6
        rtt = np.tile([1e6, 2e6], 10)
        probes = pd.DataFrame(
            {
                "pc_mid_ns": pc,
                "pi_minus_pc_ns": offset_ns + slope * (pc - reference),
                "rtt_ns": rtt,
            }
        )
        model = fit_clock_model(probes, minimum_span_s=30)
        self.assertEqual(model["kind"], "linear")
        self.assertAlmostEqual(model["drift_ppm"], 12.0, places=5)
        test_pc = np.array([1.7e18 + 20e9, 1.7e18 + 80e9])
        test_pi = test_pc + offset_ns + slope * (test_pc - reference)
        np.testing.assert_allclose(pi_to_pc_wall_ns(test_pi, model), test_pc, atol=256)


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.campaign = self.root / "campaign.yaml"
        write_campaign(self.campaign, campaign_config(self.root / "sessions"))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_session(self, frame: pd.DataFrame) -> None:
        session = self.root / "sessions/id_01"
        (session / "aligned").mkdir(parents=True)
        (session / "summary").mkdir()
        frame.to_csv(session / "aligned/aligned.csv", index=False)
        (session / "aligned/alignment_report.json").write_text(
            json.dumps({"session_id": "id_01", "clock_model": {"uncertainty_ms": 1.0}, "fc": None}),
            encoding="utf-8",
        )

    def test_nan_is_rejected_without_row_drop(self) -> None:
        frame = aligned_frame("id_01")
        frame.loc[10, "estimated_wind_n_mps"] = np.nan
        self._write_session(frame)
        result = validate_session("id_01", campaign_path=self.campaign, local_path=self.root / "missing.yaml")
        self.assertFalse(result["valid"])
        self.assertTrue(any("non-finite" in error and "never dropped" in error for error in result["errors"]))

    def test_wrong_condition_wind_range_is_rejected(self) -> None:
        self._write_session(aligned_frame("id_01", wind=5.0))
        result = validate_session("id_01", campaign_path=self.campaign, local_path=self.root / "missing.yaml")
        self.assertFalse(result["valid"])
        self.assertTrue(any("wind mismatches" in error for error in result["errors"]))


class EndToEndAlignmentTests(unittest.TestCase):
    def test_align_then_validate_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign_path = root / "campaign.yaml"
            write_campaign(campaign_path, campaign_config(root / "sessions"))
            session = root / "sessions/id_01"
            for child in ("pc", "pi", "fc", "aligned", "summary"):
                (session / child).mkdir(parents=True)

            count = 101
            seconds = np.arange(count) / 50.0
            pc_wall_ns = 1_700_000_000_000_000_000 + seconds * 1e9
            pi_wall_ns = pc_wall_ns + 8_000_000
            pi = aligned_frame("id_01").drop(
                columns=[
                    "truth_time_us",
                    "truth_wind_n_mps",
                    "truth_wind_e_mps",
                    "truth_wind_d_mps",
                    "truth_alignment_residual_ms",
                ]
            )
            pi["pi_wall_time_ns"] = pi_wall_ns
            pi.to_csv(session / "pi/pi_estimator.csv", index=False)
            truth = pd.DataFrame(
                {
                    "wall_time_usec": (pc_wall_ns / 1000).astype(np.int64),
                    "total_wind_north_ms": 2.0,
                    "total_wind_east_ms": 0.0,
                    "total_wind_down_ms": 0.0,
                }
            )
            truth.to_csv(session / "pc/jsbsim_truth.csv", index=False)

            for phase, base in (("start", int(pc_wall_ns[0])), ("end", int(pc_wall_ns[-1]))):
                records = []
                for sample in range(6):
                    t1 = base + sample * 100_000
                    records.append(
                        {
                            "protocol_version": 1,
                            "session_id": "id_01",
                            "sample": sample,
                            "t1_pc_send_ns": t1,
                            "t2_pi_recv_ns": t1 + 8_500_000,
                            "t3_pi_send_ns": t1 + 8_600_000,
                            "t4_pc_recv_ns": t1 + 1_100_000,
                            "ok": True,
                        }
                    )
                (session / f"pc/marker_{phase}.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )

            aligned, report = align_session(
                "id_01",
                campaign_path=campaign_path,
                local_path=root / "missing.yaml",
            )
            self.assertEqual(len(aligned), count)
            self.assertGreater(report["truth"]["coverage"], 0.99)
            result = validate_session(
                "id_01",
                campaign_path=campaign_path,
                local_path=root / "missing.yaml",
            )
            self.assertTrue(result["valid"], result["errors"])


class CampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.campaign = self.root / "campaign.yaml"
        config = campaign_config(self.root / "sessions", duration=5.0, warmup=1.0)
        config["protocol"]["minimum_valid_post_warmup_s"] = 4.0
        config["quality"]["minimum_achieved_loop_hz"] = 0.5
        write_campaign(self.campaign, config)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create_campaign(self, count: int) -> None:
        session_ids = [f"{condition}_{index:02d}" for condition in ("id", "ood") for index in range(1, 6)]
        for session_id in session_ids[:count]:
            root = self.root / "sessions" / session_id
            (root / "aligned").mkdir(parents=True)
            (root / "summary").mkdir()
            wind = 2.0 if session_id.startswith("id") else 5.0
            frame = aligned_frame(session_id, rate_hz=1.0, duration_s=5.0, wind=wind)
            frame["estimated_wind_n_mps"] += 1.0
            frame["estimated_wind_e_mps"] += 2.0
            frame["estimated_wind_d_mps"] += 2.0
            frame.to_csv(root / "aligned/aligned.csv", index=False)
            (root / "summary/validation.json").write_text(
                json.dumps({"session_id": session_id, "valid": True}), encoding="utf-8"
            )

    def test_bad_session_count_is_rejected(self) -> None:
        self._create_campaign(9)
        with self.assertRaisesRegex(ValueError, "missing validation"):
            analyze_campaign(campaign_path=self.campaign, local_path=self.root / "missing.yaml")

    def test_session_aggregation_and_summary_values(self) -> None:
        self._create_campaign(10)
        sessions, conditions, payload = analyze_campaign(
            campaign_path=self.campaign,
            local_path=self.root / "missing.yaml",
            output_dir=self.root / "output",
        )
        self.assertEqual(len(sessions), 10)
        np.testing.assert_allclose(sessions["rmse_3d_mps"], np.sqrt(3.0))
        np.testing.assert_allclose(sessions["second_diff_jitter_mps"], 0.0, atol=1e-12)
        np.testing.assert_allclose(conditions["rmse_3d_mps_mean"], np.sqrt(3.0))
        np.testing.assert_allclose(conditions["rmse_3d_mps_sd"], 0.0)
        self.assertEqual(payload["aggregation_unit"], "session")
        self.assertTrue((self.root / "output/session_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
