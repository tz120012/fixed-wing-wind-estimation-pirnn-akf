#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV="${PROJECT_ROOT}/.venv"

if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
  printf 'Warning: expected a 64-bit Raspberry Pi OS, detected %s.\n' "$(uname -m)" >&2
fi

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip wheel
"${VENV}/bin/python" -m pip install \
  'numpy<2' 'pandas<3' scipy PyYAML 'scikit-learn==1.3.2' \
  pymavlink pyserial onnxruntime

if command -v getent >/dev/null 2>&1 && getent group dialout >/dev/null; then
  if id -nG "${USER:-$(id -un)}" | tr ' ' '\n' | awk '$0 == "dialout" { found=1 } END { exit !found }'; then
    printf 'Serial access: %s is already in the dialout group.\n' "${USER:-$(id -un)}"
  else
    printf '\nSerial access requires dialout membership. Run this yourself, then log out/in:\n' >&2
    printf '  sudo usermod -aG dialout %q\n' "${USER:-$(id -un)}" >&2
    printf 'No sudo command was run by this script.\n' >&2
  fi
else
  printf 'Note: no dialout group was found; verify serial-device permissions manually.\n' >&2
fi

chmod +x \
  "${SCRIPT_DIR}/setup.sh" \
  "${SCRIPT_DIR}/clock_marker_listener.py" \
  "${SCRIPT_DIR}/preflight.py" \
  "${SCRIPT_DIR}/run_estimator.py"

printf '\nPi environment ready. Activate with:\n  source %q\n' "${VENV}/bin/activate"
