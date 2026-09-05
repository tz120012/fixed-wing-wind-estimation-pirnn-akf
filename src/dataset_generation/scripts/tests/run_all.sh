#!/usr/bin/env bash
# 批量跑所有 PIRNN-AKF 重构相关 unit/integration tests
# 用法：bash src/dataset_generation/scripts/tests/run_all.sh
set -e
cd "$(dirname "$0")/../../../.."  # 回到 wind-estimation-main 根
PY="${PY:-.venv/bin/python}"

echo "=========================================="
echo "PIRNN-AKF refactor — full test suite"
echo "=========================================="

declare -a TESTS=(
  "src/dataset_generation/scripts/tests/test_data_logger_stage1.py"
  "src/dataset_generation/scripts/tests/test_data_logger_stage2.py"
  "src/dataset_generation/scripts/tests/test_postprocess_stage1.py"
  "src/dataset_generation/scripts/tests/test_postprocess_stage2.py"
  "src/dataset_generation/scripts/tests/test_preprocess_stage1.py"
  "src/dataset_generation/scripts/tests/test_preprocess_stage2.py"
  "src/dataset_generation/scripts/tests/test_model_smoke_stage1.py"
  "src/dataset_generation/scripts/tests/test_model_smoke_stage2.py"
  "src/dataset_generation/scripts/tests/test_integration_stage1.py"
  "src/dataset_generation/scripts/tests/test_integration_stage2.py"
  "src/dataset_generation/scripts/tests/test_loss_dyn_stage3.py"
  "src/dataset_generation/scripts/tests/test_infra_phase2.py"
  "src/dataset_generation/scripts/tests/test_validation_phase3.py"
  "src/dataset_generation/scripts/tests/test_pipeline_phase4.py"
  "src/dataset_generation/scripts/tests/test_final_bulk_smoke.py"
)

PASS=0
FAIL=0
for t in "${TESTS[@]}"; do
  echo ""
  echo "▶ $(basename "$t")"
  if "$PY" "$t" >/tmp/_test_out 2>&1; then
    echo "  [PASS]"
    PASS=$((PASS + 1))
  else
    echo "  [FAIL] (last 30 lines):"
    tail -n 30 /tmp/_test_out | sed 's/^/    /'
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "=========================================="
echo "Total: $((PASS + FAIL))    Pass: $PASS    Fail: $FAIL"
echo "=========================================="

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
