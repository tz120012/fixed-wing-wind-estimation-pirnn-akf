#!/usr/bin/env python3
"""Prepare sanitized software and unified-data release staging directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import shutil
from collections.abc import Callable
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (26, 42, 2024, 2025, 2026)
PUBLIC_REPOSITORY = (
    "https://github.com/tz120012/fixed-wing-wind-estimation-pirnn-akf"
)

SOFTWARE_TREE_SOURCES = (
    "src",
    "scripts",
    "config",
    "utils",
    "artifacts",
)

SCRIPT_EXCLUSIONS = {
    "cleanup_markdown_math.py",
    "latex_to_unicode.py",
    "manual_math_fix.py",
    "robust_markdown_math.py",
}

IGNORED_NAMES = {
    ".DS_Store",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "_p0_tmp",
    "__pycache__",
    "data",
    "logs",
    "results",
    "sessions",
    "tensorboard",
    "wandb",
}

IGNORED_SUFFIXES = {
    ".aux",
    ".log",
    ".out",
    ".pyc",
    ".tlog",
    ".ulg",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

FEATURE_NAMES_45 = (
    "vel_n",
    "vel_e",
    "vel_d",
    "vel_x_body",
    "vel_y_body",
    "vel_z_body",
    "acc_x",
    "acc_y",
    "acc_z",
    "roll",
    "pitch",
    "yaw",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "aileron",
    "elevator",
    "rudder",
    "throttle",
    "airspeed",
    "target_roll",
    "target_pitch",
    "target_yaw",
    "roll_err",
    "pitch_err",
    "yaw_err",
    "target_p",
    "target_q",
    "target_r",
    "p_err",
    "q_err",
    "r_err",
    "target_vn",
    "target_ve",
    "target_vd",
    "vn_err",
    "ve_err",
    "vd_err",
    "aileron_act",
    "elevator_act",
    "rudder_act",
    "throttle_act",
    "imu_ax",
    "imu_ay",
    "imu_az",
)
REMOVED_FEATURE_INDICES = {38, 39, 40, 41}
FEATURE_NAMES_41 = tuple(
    name
    for index, name in enumerate(FEATURE_NAMES_45)
    if index not in REMOVED_FEATURE_INDICES
)
LABEL_NAMES = (
    "wind_north",
    "wind_east",
    "wind_down",
    "vel_n",
    "vel_e",
    "vel_d",
    "airspeed",
)

DATA_PRIVACY_REPLACEMENTS = {
    str(ROOT): "/opt/wind-estimation",
    "/path/to/PX4-Autopilot": "/opt/PX4-Autopilot",
    str(Path("/", "home", "pi", "wind-estimation-main")):
        "/opt/wind-estimation",
    str(Path("/", "home", "rasp", "wind-estimation")):
        "/opt/wind-estimation",
    "<PI_IP>": "192.0.2.20",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--software-dir",
        type=Path,
        default=ROOT.parent
        / "releases/fixed-wing-wind-estimation-pirnn-akf",
    )
    parser.add_argument(
        "--data-release-dir",
        type=Path,
        default=ROOT.parent / "releases/wind-estimation-data-v1.0.0",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--hash-source-files",
        action="store_true",
        help="Hash every source file (reads roughly 60 GB).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--software-only",
        action="store_true",
        help="Rebuild only the sanitized software snapshot.",
    )
    mode.add_argument(
        "--refresh-data-metadata",
        action="store_true",
        help="Refresh metadata/staging without deleting packaged archives.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def ignore_copy(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(directory) / name
        if name in IGNORED_NAMES:
            ignored.add(name)
        elif path.suffix.lower() in IGNORED_SUFFIXES:
            ignored.add(name)
        elif name in SCRIPT_EXCLUSIONS:
            ignored.add(name)
    return ignored


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=ignore_copy,
    )


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def replace_text_values(
    text: str, replacements: dict[str, str]
) -> str:
    updated = text
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    return updated


def replace_nested_values(
    value: object, replacements: dict[str, str]
) -> object:
    if isinstance(value, str):
        return replace_text_values(value, replacements)
    if isinstance(value, dict):
        return {
            replace_nested_values(key, replacements):
                replace_nested_values(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [replace_nested_values(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(
            replace_nested_values(item, replacements) for item in value
        )
    if isinstance(value, set):
        return {
            replace_nested_values(item, replacements) for item in value
        }
    return value


def sanitize_pickle(
    source: Path,
    destination: Path,
    replacements: dict[str, str],
) -> None:
    with source.open("rb") as handle:
        value = pickle.load(handle)
    sanitized = replace_nested_values(value, replacements)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(sanitized, handle, protocol=pickle.HIGHEST_PROTOCOL)


def sanitize_text_tree(root: Path) -> None:
    replacements = {
        f"{ROOT}/": "",
        str(ROOT): ".",
        "/path/to/PX4-Autopilot":
            "/path/to/PX4-Autopilot",
        "": "",
        "<PI_IP>": "<PI_IP>",
        "pi@<PI_IP>": "pi@<PI_IP>",
    }
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            had_crlf = b"\r\n" in path.read_bytes()
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        updated = replace_text_values(updated, replacements)
        if path.name == "run_sitl_experiment.py":
            updated = updated.replace(
                '_PX4_ROOT     = Path("/path/to/PX4-Autopilot")',
                '_PX4_ROOT = Path(os.environ.get('
                '"PX4_ROOT", str(Path.home() / "PX4-Autopilot")'
                ")).expanduser()",
            )
        if path.name == "sitl_takeoff.py":
            updated = updated.replace(
                "sys.path.insert(0, '')\n",
                "",
            )
        if path.name == "hitl_collect.sh":
            updated = updated.replace(
                'PX4_ROOT="${PX4_ROOT:-/path/to/PX4-Autopilot}"',
                'PX4_ROOT="${PX4_ROOT:-$HOME/PX4-Autopilot}"',
            )
        if path.name == "local.example.yaml":
            updated = re.sub(
                r"(?m)^  project_root: .+$",
                "  project_root: /opt/wind-estimation-main",
                updated,
            )
            updated = re.sub(
                r"(?m)^  pc_address: .+$",
                "  pc_address: 192.0.2.10",
                updated,
            )
        if updated != text or had_crlf:
            path.write_text(updated, encoding="utf-8")


def latest_checkpoint(base_dir: Path, prefix: str) -> Path:
    candidates = [
        path
        for path in base_dir.rglob("best_model.pth")
        if any(parent.name.startswith(prefix) for parent in path.parents)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No best_model.pth with prefix {prefix!r} below {base_dir}"
        )
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def software_readme() -> str:
    return """# Fixed-Wing 3D Wind Estimation with PI-GRU and AKF

This repository is the public software release for the associated *Drones*
manuscript. It contains the frozen 41-input PI-GRU/PIRNN-AKF pipeline,
parameter-matched recurrent baselines, KalmanNet comparison, PX4/JSBSim data
generation, zero-shot Rascal-to-Malolo evaluation, AKF sensitivity analysis,
ONNX deployment, and Raspberry Pi 5 + CUAV V5+ HITL tooling.

## Scope

- Five fixed seeds: 26, 42, 2024, 2025 and 2026.
- Input: 41 state channels, 100 time steps, 50 Hz nominal sampling.
- Final learned methods: Vanilla GRU, Vanilla LSTM, PI-GRU and KalmanNet.
- Post-processing: causal measurement-noise-adaptive Kalman smoother.
- Validation: PX4/JSBSim ID/OOD, conservative cross-configuration Malolo
  transfer, and ten hardware-in-the-loop sessions.

The public results do not constitute outdoor real-flight validation or prove
closed-loop control benefit.

## Quick verification

The compact verifier supports Python 3.8 or newer. Full model training and
evaluation were frozen with Python 3.13 and the versions in
`requirements-revision.txt`; the CUDA 12.8 environment is retained separately
as `requirements-revision-cu128.txt`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-minimal.txt
unzip evidence/minimal_dataset.zip -d evidence/
.venv/bin/python evidence/minimal_dataset/verify_reported_values.py
```

Expected result: `302 checks agree, 0 differ`.

## Repository layout

- `src/`: preprocessing, models, training, inference and data generation.
- `scripts/`: frozen revision evaluation and plotting workflows.
- `config/`: frozen experiment and deployment configurations.
- `models/`: canonical final checkpoints, ONNX model and normalization.
- `HITL/`: PC, Raspberry Pi and post-processing HITL toolchain.
- `SITL/`: retained PX4/JSBSim orchestration.
- `evidence/minimal_dataset.zip`: compact figure/table source data.

## Data

The complete data are distributed under one versioned dataset DOI. The record
contains six independently downloadable archives: acquisition JSON, training
CSV, processed 41-input arrays, main/cross-configuration results, HITL
campaign logs, and the compact minimal dataset.

Dataset DOI: **to be inserted after deposition**.

## Reproduction

See `REVISION_REPRODUCIBILITY.md`, `HITL/README.md`,
`CHECKPOINT_MANIFEST.csv`, and `RELEASE_MANIFEST.md`.

## Licence and citation

Original project software is MIT licensed. The Rascal model under
`src/Rascal/` retains its GPL-3.0 licence; see `THIRD_PARTY_NOTICES.md`.
Please cite the associated article and the archived software/data records.
"""


def third_party_notices() -> str:
    return """# Third-party notices

## Rascal 110 aircraft model

Files under `src/Rascal/` retain the GNU General Public License version 3
distributed in `src/Rascal/LICENSE`. The repository-level MIT licence does not
replace or weaken that subdirectory licence.

## PX4 and SymForce-derived material

The project interoperates with PX4-Autopilot and contains a generated
airspeed-fusion implementation under `src/px4_ekf2/`. Preserve upstream
attribution and applicable licence notices when redistributing derived files.
PX4-Autopilot itself is not vendored in this repository.

## JSBSim, MAVSDK, pymavlink and ONNX Runtime

These dependencies are not vendored. Their names identify interoperability
requirements only; each remains governed by its upstream licence.

## Manuscript template and literature

Publisher templates, reviewer correspondence, submission letters and
third-party article PDFs are intentionally excluded from this public software
snapshot.
"""


def public_release_manifest() -> str:
    return """# Public release manifest

Version: `v1.0.0`

## Included

- Frozen source code and configurations.
- Fifteen canonical recurrent-model checkpoints (three model families,
  five seeds).
- Five KalmanNet checkpoints.
- Final 41-input ONNX model and training-split normalization parameters.
- Compact 51-file supporting dataset and verifier.
- PC/Raspberry Pi HITL acquisition and alignment software.

## Excluded from Git

- Full acquisition JSON, training CSV and processed NumPy arrays.
- Full experiment predictions, Malolo source records and ten-session HITL
  logs (published in the unified data record).
- Intermediate checkpoints and training logs.
- Manuscript drafts, reviewer/editor correspondence and literature PDFs.

## Release gates

- [ ] Public GitHub URL resolves.
- [ ] Software DOI resolves.
- [ ] Unified data DOI resolves.
- [ ] `CHECKSUMS.sha256` passes in a clean clone.
- [ ] Minimal dataset reports 302 checks and zero differences.
- [ ] No absolute local paths, private hosts or submission correspondence.
"""


def citation_cff() -> str:
    return f"""cff-version: 1.2.0
message: "If you use this software, please cite the software and article."
title: "Fixed-Wing 3D Wind Estimation with PI-GRU and Adaptive Kalman Smoothing"
type: software
version: 1.0.0
date-released: 2026-09-05
authors:
  - family-names: Tian
    given-names: Zhong
    orcid: "https://orcid.org/0009-0000-6342-2874"
  - family-names: Song
    given-names: Mingli
  - family-names: Fu
    given-names: Jiahao
  - family-names: Zhu
    given-names: Weiyu
  - family-names: Zhang
    given-names: Bangchu
repository-code: "{PUBLIC_REPOSITORY}"
abstract: >-
  Frozen software, configurations, canonical checkpoints and compact evidence
  for fixed-wing local three-dimensional wind estimation with PI-GRU and
  measurement-noise-adaptive Kalman smoothing.
keywords:
  - fixed-wing UAV
  - wind estimation
  - physics-informed recurrent neural network
  - adaptive Kalman filter
license: MIT
"""


def zenodo_software_metadata() -> dict[str, object]:
    return {
        "title": (
            "Fixed-Wing 3D Wind Estimation with PI-GRU and "
            "Adaptive Kalman Smoothing"
        ),
        "description": (
            "Frozen code, configurations, canonical checkpoints and compact "
            "evidence accompanying the associated Drones manuscript."
        ),
        "creators": [
            {
                "name": "Tian, Zhong",
                "orcid": "0009-0000-6342-2874",
                "affiliation": (
                    "School of Aeronautics and Astronautics, "
                    "Sun Yat-sen University"
                ),
            },
            {"name": "Song, Mingli"},
            {"name": "Fu, Jiahao"},
            {"name": "Zhu, Weiyu"},
            {"name": "Zhang, Bangchu"},
        ],
        "license": "MIT",
        "upload_type": "software",
        "version": "1.0.0",
        "keywords": [
            "fixed-wing UAV",
            "wind estimation",
            "physics-informed recurrent neural network",
            "adaptive Kalman filter",
            "hardware-in-the-loop",
        ],
        "related_identifiers": [
            {
                "identifier": PUBLIC_REPOSITORY,
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ],
    }


def write_software_checksums(root: Path) -> None:
    output = root / "CHECKSUMS.sha256"
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path != output
        and ".git" not in path.parts
    )
    with output.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(
                f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            )


def build_software_release(destination: Path, overwrite: bool) -> None:
    reset_directory(destination, overwrite)
    for relative in SOFTWARE_TREE_SOURCES:
        copy_tree(ROOT / relative, destination / relative)

    hitl_destination = destination / "HITL"
    for relative in (
        "common",
        "config",
        "pc",
        "pi",
        "postprocess",
        "revision_protocol",
        "tests",
    ):
        copy_tree(ROOT / "HITL" / relative, hitl_destination / relative)
    for name in (
        "README.md",
        "__init__.py",
        "align_hitl_timestamps.py",
        "hitl_collect.sh",
        "prepare_artifacts.py",
    ):
        copy_file(ROOT / "HITL" / name, hitl_destination / name)
    for norm_path in hitl_destination.rglob("norm_params.pkl"):
        sanitize_pickle(
            norm_path,
            norm_path,
            DATA_PRIVACY_REPLACEMENTS,
        )

    sitl_destination = destination / "SITL"
    copy_tree(ROOT / "SITL" / "wind_configs", sitl_destination / "wind_configs")
    for name in (
        "sitl_launch_and_eval.py",
        "sitl_planB_control_eval.py",
        "sitl_takeoff.py",
    ):
        copy_file(ROOT / "SITL" / name, sitl_destination / name)

    for name in (
        "LICENSE",
        "REVISION_REPRODUCIBILITY.md",
        "HITL_REVISION_RUNBOOK.md",
    ):
        copy_file(ROOT / name, destination / name)
    full_requirements = (ROOT / "requirements-revision.txt").read_text(
        encoding="utf-8"
    )
    (destination / "requirements-revision-cu128.txt").write_text(
        full_requirements, encoding="utf-8"
    )
    portable_requirements = "\n".join(
        line.replace("torch==2.11.0+cu128", "torch==2.11.0")
        for line in full_requirements.splitlines()
        if not line.startswith("--extra-index-url")
    )
    (destination / "requirements-revision.txt").write_text(
        portable_requirements + "\n", encoding="utf-8"
    )
    (destination / "requirements-minimal.txt").write_text(
        "numpy>=1.24,<3\npandas>=2.0,<4\n",
        encoding="utf-8",
    )

    compact_source = (
        ROOT / "Paper_2/MDPI_template_APA/submission/minimal_dataset"
    )
    compact_destination = (
        destination
        / "Paper_2/MDPI_template_APA/submission/minimal_dataset"
    )
    copy_tree(compact_source, compact_destination)
    copy_file(
        ROOT / "Paper_2/MDPI_template_APA/submission/minimal_dataset.zip",
        destination
        / "Paper_2/MDPI_template_APA/submission/minimal_dataset.zip",
    )
    copy_file(
        ROOT / "Paper_2/MDPI_template_APA/submission/minimal_dataset.zip",
        destination / "evidence/minimal_dataset.zip",
    )

    checkpoint_rows: list[dict[str, object]] = []
    model_root = ROOT / "data/revision_main_41d/models"
    model_specs = {
        "vanilla_gru": ("vanilla", "vanilla_gru_"),
        "vanilla_lstm": ("lstm", "vanilla_gru_"),
        "pi_gru": ("pigru", "train_"),
    }
    for seed in SEEDS:
        for public_name, (source_name, prefix) in model_specs.items():
            source = latest_checkpoint(
                model_root / f"seed_{seed}" / source_name,
                prefix,
            )
            relative = (
                Path("models") / public_name / f"seed_{seed}"
                / "best_model.pth"
            )
            target = destination / relative
            copy_file(source, target)
            checkpoint_rows.append({
                "model": public_name,
                "seed": seed,
                "path": relative.as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "selection": "latest validation-selected best_model.pth",
            })

        kalmannet_source = (
            ROOT / "data/revision_kalmannet_41d"
            / f"kalmannet_seed{seed}.pth"
        )
        kalmannet_relative = (
            Path("models/kalmannet") / f"seed_{seed}" / "best_model.pth"
        )
        kalmannet_target = destination / kalmannet_relative
        copy_file(kalmannet_source, kalmannet_target)
        checkpoint_rows.append({
            "model": "kalmannet",
            "seed": seed,
            "path": kalmannet_relative.as_posix(),
            "size_bytes": kalmannet_target.stat().st_size,
            "sha256": sha256_file(kalmannet_target),
            "selection": "validation-selected KalmanNet checkpoint",
        })

    norm_relative = Path("models/norm_params_41d.pkl")
    sanitize_pickle(
        ROOT / "data/dataset_revision_41d/norm_params.pkl",
        destination / norm_relative,
        DATA_PRIVACY_REPLACEMENTS,
    )
    onnx_relative = Path("models/pigru/seed_26/model.onnx")
    copy_file(
        ROOT / "artifacts/pigru_41d_seed26.onnx",
        destination / onnx_relative,
    )
    for model, seed, relative in (
        ("normalization", "train", norm_relative),
        ("pi_gru_onnx", 26, onnx_relative),
    ):
        target = destination / relative
        checkpoint_rows.append({
            "model": model,
            "seed": seed,
            "path": relative.as_posix(),
            "size_bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "selection": "frozen deployment artifact",
        })

    with (destination / "CHECKPOINT_MANIFEST.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=checkpoint_rows[0].keys())
        writer.writeheader()
        writer.writerows(checkpoint_rows)

    (destination / "README.md").write_text(
        software_readme(), encoding="utf-8"
    )
    (destination / "THIRD_PARTY_NOTICES.md").write_text(
        third_party_notices(), encoding="utf-8"
    )
    (destination / "RELEASE_MANIFEST.md").write_text(
        public_release_manifest(), encoding="utf-8"
    )
    (destination / "CITATION.cff").write_text(
        citation_cff(), encoding="utf-8"
    )
    (destination / ".zenodo.json").write_text(
        json.dumps(zenodo_software_metadata(), indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / ".gitignore").write_text(
        "\n".join([
            ".venv/",
            "__pycache__/",
            "*.pyc",
            "*.log",
            "*.ulg",
            "*.tlog",
            "data/",
            "HITL/sessions/",
            "HITL/config/local.yaml",
            "",
        ]),
        encoding="utf-8",
    )
    (destination / ".gitattributes").write_text(
        "\n".join([
            "* text=auto eol=lf",
            "*.onnx binary",
            "*.pkl binary",
            "*.pth binary",
            "*.zip binary",
            "",
        ]),
        encoding="utf-8",
    )
    sanitize_text_tree(destination)
    write_software_checksums(destination)


def data_readme() -> str:
    return """# Unified Data Record for Fixed-Wing 3D Wind Estimation

Version: `v1.0.0`

This record contains the complete retained data lineage and experiment
evidence for the associated *Drones* manuscript. The files are separated into
six independently downloadable archives but share one dataset DOI.

## Archives

1. `01_rascal_acquisition_json_v1.0.0.tar.gz`: 800 high-rate telemetry JSON
   segments and 800 metadata sidecars generated by PX4 SITL + JSBSim.
2. `02_rascal_training_csv_v1.0.0.tar.gz`: 800 deterministic JSBSim-style CSV
   exports consumed by preprocessing and AKF sensitivity analysis.
3. `03_rascal_41d_windows_v1.0.0.tar.gz`: frozen 41-input, 100-step NumPy
   arrays, labels, sample weights, turn classes and normalization parameters.
4. `04_main_and_cross_configuration_results_v1.0.0.tar.gz`: five-seed
   predictions/statistics, KalmanNet, Malolo cross-configuration and AKF
   sensitivity evidence.
5. `05_hitl_campaign_v1.0.0.tar.gz`: ten ID/OOD Raspberry Pi 5 + CUAV V5+
   HITL sessions with separate PC, Pi, FC, aligned and validation records.
6. `06_minimal_dataset_v1.0.0.zip`: compact manuscript source data and
   verifier (`302 checks agree, 0 differ`).

## Data lineage

`01 acquisition JSON` -> `02 training CSV` -> `03 41-input windows` ->
`04 model predictions/statistics`.

Archive 05 is a separate HITL evidence chain. Archive 06 is a compact,
reviewer-facing source-data projection derived from archives 03--05 and
retained predecessor experiments.

## Splits

The Rascal corpus contains 550 train, 150 validation, 50 Test-ID and 50
Test-OOD segments. `SPLIT_MANIFEST.csv` maps every acquisition JSON pair to
its training CSV. Test data were not used for checkpoint or smoother
selection.

## Integrity

- `SOURCE_FILE_MANIFEST.csv` records source path, size and optional SHA-256.
- `ARCHIVE_MANIFEST.csv` records final archive size and SHA-256.
- `SPLIT_MANIFEST.csv` freezes segment membership.

## Release-only de-identification

Packaging replaces workstation filesystem prefixes with `/opt/...` and the
private Raspberry Pi LAN address with the RFC 5737 TEST-NET address
`192.0.2.20`. This affects provenance strings only; timestamps, telemetry,
truth wind, estimates, timing measurements and all numerical results are
unchanged. The private source files remain unmodified.

## Licence

The released data are licensed under CC BY 4.0. Software is distributed
separately under MIT with retained third-party notices.

## Citation

The final DataCite citation and DOI will be inserted after deposition.
"""


def data_dictionary() -> str:
    feature_lines = "\n".join(
        f"{index},{name},normalized model input,see source CSV metadata"
        for index, name in enumerate(FEATURE_NAMES_41)
    )
    label_lines = "\n".join(
        f"{index},{name},supervision/auxiliary target,see norm_params.pkl"
        for index, name in enumerate(LABEL_NAMES)
    )
    return f"""# Data dictionary

## Acquisition JSON

Each `datasets-*.json` is a JSON array of timestamped PX4/JSBSim telemetry
records. Its paired `*_metadata.json` records split, wind/maneuver setup,
duration, sample count, effective logging rate and simulator provenance.
Important fields include truth wind N/E/D, NED and body velocity, attitude,
body rates, airspeed, control commands, targets, IMU acceleration, manoeuvre
labels, simulator time and wall-clock time.

## Training CSV

Each CSV is the deterministic output of
`src/dataset_generation/scripts/postprocess_to_jsbsim_csv.py`. Column names
follow the JSBSim-oriented preprocessing schema. The header in each file is
authoritative; `SPLIT_MANIFEST.csv` maps it to the acquisition JSON.

## Processed arrays

- `X_<split>.npy`: `[samples, 100, 41]`, normalized float32 inputs.
- `y_<split>.npy`: labels whose first three columns are wind N/E/D.
- `w_<split>.npy`: sample weights.
- `turn_class_<split>.npy`: turn/maneuver class labels.
- `norm_params.pkl`: train-only fitted scalers and feature metadata.

### 41 model-input channels

index,name,role,unit
{feature_lines}

### Label channels

index,name,role,unit
{label_lines}

Wind and velocity metrics are reported in m/s after inverse transformation.
Angles and rates retain the units documented by preprocessing and
`norm_params.pkl`.

## HITL

Canonical required fields are defined in `HITL/common/schema.py`. PC truth,
Pi estimates and FC boot-time records remain separate before offline
alignment. `alignment_report.json` and `validation.json` document clock and
quality gates for each session.
"""


def data_zenodo_metadata() -> dict[str, object]:
    return {
        "title": (
            "Data for Fixed-Wing 3D Wind Estimation with PI-GRU and "
            "Adaptive Kalman Smoothing"
        ),
        "description": (
            "Unified acquisition, training, processed, five-seed, "
            "cross-configuration, AKF-sensitivity and ten-session HITL data "
            "supporting the associated Drones manuscript."
        ),
        "creators": [
            {
                "name": "Tian, Zhong",
                "orcid": "0009-0000-6342-2874",
                "affiliation": (
                    "School of Aeronautics and Astronautics, "
                    "Sun Yat-sen University"
                ),
            },
            {"name": "Song, Mingli"},
            {"name": "Fu, Jiahao"},
            {"name": "Zhu, Weiyu"},
            {"name": "Zhang, Bangchu"},
        ],
        "license": "cc-by-4.0",
        "upload_type": "dataset",
        "version": "1.0.0",
        "keywords": [
            "fixed-wing UAV",
            "wind estimation",
            "PX4 SITL",
            "JSBSim",
            "physics-informed neural network",
            "hardware-in-the-loop",
        ],
        "related_identifiers": [
            {
                "identifier": PUBLIC_REPOSITORY,
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ],
    }


def segment_identity(path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"datasets-(\d+)-(\d+)", path.stem)
    if not match:
        raise ValueError(f"Unexpected segment name: {path.name}")
    return int(match.group(1)), int(match.group(2))


def build_split_manifest(destination: Path) -> None:
    acquisition = ROOT / "src/dataset_generation/data"
    csv_root = ROOT / "data/data_csv"
    rows = []
    for split in ("train", "val", "test_id", "test_ood"):
        data_files = sorted(
            path
            for path in (acquisition / split).glob("datasets-*.json")
            if not path.name.endswith("_metadata.json")
        )
        for data_path in data_files:
            run, segment = segment_identity(data_path)
            metadata_path = data_path.with_name(
                f"{data_path.stem}_metadata.json"
            )
            csv_path = csv_root / split / f"{data_path.stem}.csv"
            if not metadata_path.is_file() or not csv_path.is_file():
                raise FileNotFoundError(
                    f"Incomplete JSON/metadata/CSV mapping for {data_path}"
                )
            rows.append({
                "split": split,
                "run": run,
                "segment": segment,
                "acquisition_json": (
                    f"rascal_acquisition_json/{split}/{data_path.name}"
                ),
                "metadata_json": (
                    f"rascal_acquisition_json/{split}/{metadata_path.name}"
                ),
                "training_csv": (
                    f"rascal_training_csv/{split}/{csv_path.name}"
                ),
            })
    expected = {"train": 550, "val": 150, "test_id": 50, "test_ood": 50}
    observed = {
        split: sum(row["split"] == split for row in rows)
        for split in expected
    }
    if observed != expected:
        raise ValueError(f"Unexpected corpus split counts: {observed}")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def stage_experiment_evidence(destination: Path) -> None:
    destination.mkdir(parents=True)
    main_source = ROOT / "data/revision_main_41d"
    main_destination = destination / "revision_main_41d"
    for relative in ("predictions", "statistics", "configs"):
        copy_tree(main_source / relative, main_destination / relative)
    for name in (
        "ekf_predictions.npz",
        "figure2_metrics_aggregated.csv",
        "figure2_metrics_per_seed.csv",
        "figure2_rmse_bar.pdf",
        "figure2_rmse_bar.svg",
        "manifest.json",
    ):
        source = main_source / name
        if source.is_file():
            copy_file(source, main_destination / name)

    for relative in (
        "revision_kalmannet_41d",
        "revision_cross_airframe",
        "revision_akf",
        "cross_airframe_malolo_raw",
        "cross_airframe_malolo_processed",
    ):
        copy_tree(
            ROOT / "data" / relative,
            destination / relative,
        )
    sanitize_text_tree(destination)


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_sanitized_data_tree(
    source: Path,
    destination: Path,
    *,
    sanitize_text: Callable[[Path], bool],
    sanitize_norm_pickle: bool = False,
) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if sanitize_norm_pickle and path.name == "norm_params.pkl":
            sanitize_pickle(path, target, DATA_PRIVACY_REPLACEMENTS)
        elif sanitize_text(path):
            target.parent.mkdir(parents=True, exist_ok=True)
            text = path.read_text(encoding="utf-8")
            target.write_text(
                replace_text_values(text, DATA_PRIVACY_REPLACEMENTS),
                encoding="utf-8",
            )
            shutil.copystat(path, target)
        else:
            hardlink_or_copy(path, target)


def archive_source_groups(
    experiment_staging: Path,
    acquisition_staging: Path,
    windows_staging: Path,
    hitl_staging: Path,
) -> list[dict[str, object]]:
    return [
        {
            "archive": "01_rascal_acquisition_json_v1.0.0.tar.gz",
            "group": "rascal_acquisition_json",
            "source": acquisition_staging,
        },
        {
            "archive": "02_rascal_training_csv_v1.0.0.tar.gz",
            "group": "rascal_training_csv",
            "source": ROOT / "data/data_csv",
        },
        {
            "archive": "03_rascal_41d_windows_v1.0.0.tar.gz",
            "group": "rascal_41d_windows",
            "source": windows_staging,
        },
        {
            "archive":
                "04_main_and_cross_configuration_results_v1.0.0.tar.gz",
            "group": "experiment_evidence",
            "source": experiment_staging,
        },
        {
            "archive": "05_hitl_campaign_v1.0.0.tar.gz",
            "group": "hitl_campaign",
            "source": hitl_staging,
        },
        {
            "archive": "06_minimal_dataset_v1.0.0.zip",
            "group": "minimal_dataset",
            "source": (
                ROOT
                / "Paper_2/MDPI_template_APA/submission/minimal_dataset.zip"
            ),
        },
    ]


def iter_source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.rglob("*") if path.is_file())


def build_source_manifest(
    groups: list[dict[str, object]],
    destination: Path,
    hash_files: bool,
) -> None:
    fieldnames = (
        "archive",
        "group",
        "relative_path",
        "size_bytes",
        "sha256",
    )
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            source = Path(group["source"])
            for path in iter_source_files(source):
                relative = (
                    path.name
                    if source.is_file()
                    else path.relative_to(source).as_posix()
                )
                writer.writerow({
                    "archive": group["archive"],
                    "group": group["group"],
                    "relative_path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path) if hash_files else "",
                })


def build_archive_manifest(
    groups: list[dict[str, object]], destination: Path
) -> None:
    rows = []
    for group in groups:
        source = Path(group["source"])
        files = iter_source_files(source)
        rows.append({
            "archive": group["archive"],
            "group": group["group"],
            "source_file_count": len(files),
            "uncompressed_bytes": sum(path.stat().st_size for path in files),
            "archive_bytes": "",
            "sha256": "",
            "status": "pending_packaging",
        })
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def populate_data_release(destination: Path, hash_files: bool) -> None:
    metadata = destination / "metadata"
    archives = destination / "archives"
    staging = destination / ".staging"
    metadata.mkdir()
    archives.mkdir(exist_ok=True)
    experiment_staging = staging / "experiment_evidence"
    stage_experiment_evidence(experiment_staging)
    acquisition_staging = staging / "rascal_acquisition_json"
    stage_sanitized_data_tree(
        ROOT / "src/dataset_generation/data",
        acquisition_staging,
        sanitize_text=lambda path: path.name.endswith("_metadata.json"),
    )
    windows_staging = staging / "rascal_41d_windows"
    stage_sanitized_data_tree(
        ROOT / "data/dataset_revision_41d",
        windows_staging,
        sanitize_text=lambda path: path.suffix.lower()
        in {".json", ".md", ".txt", ".yaml", ".yml"},
        sanitize_norm_pickle=True,
    )
    hitl_staging = staging / "hitl_campaign"
    stage_sanitized_data_tree(
        ROOT / "HITL/sessions",
        hitl_staging,
        sanitize_text=lambda path: path.suffix.lower()
        in {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"},
    )

    (destination / "README.md").write_text(
        data_readme(), encoding="utf-8"
    )
    (destination / "DATA_DICTIONARY.md").write_text(
        data_dictionary(), encoding="utf-8"
    )
    (destination / "LICENSE_DATA.txt").write_text(
        "Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        "https://creativecommons.org/licenses/by/4.0/\n",
        encoding="utf-8",
    )
    (destination / "zenodo_metadata.json").write_text(
        json.dumps(data_zenodo_metadata(), indent=2) + "\n",
        encoding="utf-8",
    )
    build_split_manifest(metadata / "SPLIT_MANIFEST.csv")
    groups = archive_source_groups(
        experiment_staging,
        acquisition_staging,
        windows_staging,
        hitl_staging,
    )
    build_source_manifest(
        groups,
        metadata / "SOURCE_FILE_MANIFEST.csv",
        hash_files,
    )
    build_archive_manifest(groups, metadata / "ARCHIVE_MANIFEST.csv")
    local_sources = {
        str(group["archive"]): str(Path(group["source"]).resolve())
        for group in groups
    }
    (staging / "PACKAGING_SOURCES.local.json").write_text(
        json.dumps(local_sources, indent=2) + "\n",
        encoding="utf-8",
    )


def build_data_release(destination: Path, overwrite: bool, hash_files: bool) -> None:
    reset_directory(destination, overwrite)
    populate_data_release(destination, hash_files)


def refresh_data_release(destination: Path, hash_files: bool) -> None:
    if not destination.is_dir():
        raise FileNotFoundError(destination)
    for relative in ("metadata", ".staging"):
        path = destination / relative
        if path.exists():
            shutil.rmtree(path)
    populate_data_release(destination, hash_files)


def main() -> None:
    args = parse_args()
    if args.refresh_data_metadata:
        refresh_data_release(
            args.data_release_dir.resolve(),
            args.hash_source_files,
        )
        print(f"Data metadata refreshed: {args.data_release_dir.resolve()}")
    else:
        build_software_release(args.software_dir.resolve(), args.overwrite)
        print(f"Software staging: {args.software_dir.resolve()}")
        if not args.software_only:
            build_data_release(
                args.data_release_dir.resolve(),
                args.overwrite,
                args.hash_source_files,
            )
            print(f"Data staging: {args.data_release_dir.resolve()}")
    print("No files were uploaded or published.")


if __name__ == "__main__":
    main()
