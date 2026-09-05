"""Phase 3 单元测试：lib/validation/quality_gates.py。

验收点（来自 plan）：
    - 构造一段 coverage=10s, duration=44s 的假记录，喂给 validate_segment_records，
      得到 failures=[FailureRecord(code="coverage_too_low", ...)]。
    - 校验失败返回 GateResult 而非 raise。
    - 阈值通过 runtime.yaml 注入，可被 with_overrides 覆盖。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "dataset_generation"))

from lib.config import load_runtime_config  # noqa: E402
from lib.validation import (  # noqa: E402
    REQUIRED_LOG_FIELDS,
    GateResult,
    cleanup_segment_outputs,
    is_finite_number,
    segment_metadata_path,
    segment_output_is_valid,
    validate_airspeed_filled_ratio,
    validate_segment_records,
)


def _make_record(ts: float) -> dict:
    return {
        "timestamp": ts,
        "airspeed_m_s": 12.0,
        "wind_north": 1.5,
        "wind_east": 0.8,
        "wind_down": 0.0,
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "yaw_deg": 30.0,
    }


def test_coverage_too_low_emits_failure_record():
    rc = load_runtime_config()
    # 44s 的请求，但只采到 0..10s（约 500 个 50Hz 样本，足够通过 min_samples，
    # 但覆盖时长不足）
    records = [_make_record(t) for t in [i * 0.02 for i in range(500)]]
    res = validate_segment_records(records, requested_duration=44.0, runtime=rc)
    assert isinstance(res, GateResult)
    assert not res.passed
    codes = [f.code for f in res.failures]
    assert "coverage_too_low" in codes, f"期望 coverage_too_low, 实际 {codes}"
    f = next(x for x in res.failures if x.code == "coverage_too_low")
    assert f.context["min_coverage_s"] >= 26.0
    assert f.context["coverage_s"] < 11.0
    assert f.retriable is True
    print("[OK] test_coverage_too_low_emits_failure_record")


def test_low_sample_rate_emits_failure_record():
    rc = load_runtime_config()
    # 仅 30 个样本，远低于 80 floor
    records = [_make_record(i * 0.02) for i in range(30)]
    res = validate_segment_records(records, requested_duration=10.0, runtime=rc)
    codes = [f.code for f in res.failures]
    assert "low_sample_rate" in codes
    print("[OK] test_low_sample_rate_emits_failure_record")


def test_pass_when_records_valid():
    rc = load_runtime_config()
    records = [_make_record(i * 0.02) for i in range(2500)]  # 50s @ 50Hz
    res = validate_segment_records(records, requested_duration=44.0, runtime=rc)
    assert res.passed is True
    assert res.failures == []
    print("[OK] test_pass_when_records_valid")


def test_missing_field_emits_failure():
    rc = load_runtime_config()
    rec = _make_record(0.0)
    rec.pop("yaw_deg")
    records = [rec for _ in range(2500)]
    # 修复时间戳让 coverage 通过
    for i, r in enumerate(records):
        r2 = dict(r)
        r2["timestamp"] = i * 0.02
        records[i] = r2
    res = validate_segment_records(records, requested_duration=10.0, runtime=rc)
    codes = [f.code for f in res.failures]
    assert "missing_fields" in codes
    print("[OK] test_missing_field_emits_failure")


def test_airspeed_filled_ratio_below_threshold_returns_none():
    rc = load_runtime_config()
    assert validate_airspeed_filled_ratio(0.02, rc) is None
    print("[OK] test_airspeed_filled_ratio_below_threshold_returns_none")


def test_airspeed_filled_ratio_above_threshold_returns_failure():
    rc = load_runtime_config()
    f = validate_airspeed_filled_ratio(0.10, rc)
    assert f is not None
    assert f.code == "airspeed_degraded"
    assert f.context["filled_ratio"] == 0.10
    print("[OK] test_airspeed_filled_ratio_above_threshold_returns_failure")


def test_with_overrides_changes_thresholds():
    """验证 RuntimeConfig.with_overrides 可以放宽阈值（用于测试场景）。"""
    rc = load_runtime_config()
    rc_loose = rc.with_overrides(quality_gates={"min_coverage_ratio": 0.1})
    records = [_make_record(t) for t in [i * 0.02 for i in range(500)]]
    res_strict = validate_segment_records(records, 44.0, rc)
    res_loose = validate_segment_records(records, 44.0, rc_loose)
    assert not res_strict.passed
    # 放宽后 coverage_too_low 不再触发（10s > 0.1*44 = 4.4s）
    codes_loose = [f.code for f in res_loose.failures]
    assert "coverage_too_low" not in codes_loose
    print("[OK] test_with_overrides_changes_thresholds")


def test_segment_output_helpers(tmpdir):
    rc = load_runtime_config()
    tmp = Path(tmpdir)
    data = tmp / "seg.json"
    meta = segment_metadata_path(data)
    assert meta.name == "seg_metadata.json"
    # 不存在 → False
    assert segment_output_is_valid(data, rc) is False
    # 写一条不通过的（duration 太长）
    import json
    data.write_text(json.dumps([_make_record(0.0)]))
    meta.write_text(json.dumps({"duration": 100.0}))
    assert segment_output_is_valid(data, rc) is False
    # cleanup 后应当都不存在
    cleanup_segment_outputs(data)
    assert not data.exists()
    assert not meta.exists()
    print("[OK] test_segment_output_helpers")


def main():
    import tempfile
    test_coverage_too_low_emits_failure_record()
    test_low_sample_rate_emits_failure_record()
    test_pass_when_records_valid()
    test_missing_field_emits_failure()
    test_airspeed_filled_ratio_below_threshold_returns_none()
    test_airspeed_filled_ratio_above_threshold_returns_failure()
    test_with_overrides_changes_thresholds()
    with tempfile.TemporaryDirectory() as td:
        test_segment_output_helpers(td)
    print("\n[PASS] Phase 3 validation unit tests (8/8)")


if __name__ == "__main__":
    main()
