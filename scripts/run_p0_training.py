"""P0 ablation training driver: multi-seed & architecture-sensitivity sweeps.

These P0 experiments (extra random seeds, hidden-size / num-layers / seq-len sensitivity)
only need to override keys that ``Trainer`` reads directly from the config YAML. Rather than
touching the core training script, this driver clones ``config/config.yaml``, patches the
relevant keys, writes a temporary config, and launches ``src/3_train_pigru.py`` on it via
subprocess (using the SAME Python interpreter that runs this driver -- so run it with the
project ``.venv`` to get GPU: ``.venv/bin/python scripts/run_p0_training.py ...``).

Presets:
  * multiseed : train the main PI-GRU config under a list of extra seeds.
  * arch      : train under a given (hidden_size, num_layers) combo.

Backbone comparison (LSTM/TCN/...) and feature-group ablation need model/data-level changes
and are handled by separate scripts.

Examples:
  # smoke: 1 epoch to measure per-epoch time
  .venv/bin/python scripts/run_p0_training.py --preset multiseed --seeds 42 --epochs 1 \
      --out_base data/p0_smoke

  # real multi-seed run (uses config num_epochs / early stopping)
  .venv/bin/python scripts/run_p0_training.py --preset multiseed --seeds 42,7,123,2027,2028 \
      --out_base data/p0_multiseed

  # architecture sensitivity
  .venv/bin/python scripts/run_p0_training.py --preset arch --hidden 64 --layers 2 \
      --out_base data/p0_arch
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "3_train_pigru.py"
# main PI-GRU uses lambda_physics=0.1, single mode
MAIN_LAMBDA = 0.1


def load_base_config() -> dict:
    with open(BASE_CONFIG, "r") as f:
        return yaml.safe_load(f)


def patch_common(cfg: dict, out_base: str, run_tag: str, epochs: int | None) -> None:
    """Force single-run mode on the main lambda, isolate output dir, optional epoch cap."""
    exp = cfg.setdefault("experiment", {})
    exp["mode"] = "single"
    exp["lambda_list"] = str(MAIN_LAMBDA)
    exp["lambda_physics_override"] = MAIN_LAMBDA
    exp["output_base_dir"] = out_base
    exp["run_tag"] = run_tag
    exp["auto_timestamp"] = True
    if epochs is not None:
        tr = cfg.setdefault("training", {})
        tr["num_epochs"] = int(epochs)
        # keep early stopping from cutting a short smoke run
        tr["early_stopping_patience"] = max(int(epochs) + 1, tr.get("early_stopping_patience", 30))


def write_tmp_config(cfg: dict, tag: str) -> Path:
    tmp_dir = PROJECT_ROOT / "config" / "_p0_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = tmp_dir / f"config_{tag}_{ts}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def launch(cfg_path: Path, dry_run: bool) -> int:
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--config_path", str(cfg_path)]
    print(f"\n>>> {' '.join(cmd)}")
    if dry_run:
        print("    [dry-run] skipped")
        return 0
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def run_multiseed(args) -> None:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    print(f"[multiseed] seeds={seeds} out_base={args.out_base} epochs={args.epochs or 'config'}")
    for seed in seeds:
        cfg = load_base_config()
        patch_common(cfg, args.out_base, run_tag=f"p0_seed{seed}", epochs=args.epochs)
        cfg.setdefault("training", {})["seed"] = seed
        cfg_path = write_tmp_config(cfg, f"seed{seed}")
        rc = launch(cfg_path, args.dry_run)
        print(f"[multiseed] seed={seed} finished rc={rc}")


def run_arch(args) -> None:
    print(f"[arch] hidden={args.hidden} layers={args.layers} out_base={args.out_base} "
          f"epochs={args.epochs or 'config'}")
    cfg = load_base_config()
    patch_common(cfg, args.out_base, run_tag=f"p0_h{args.hidden}_l{args.layers}", epochs=args.epochs)
    model = cfg.setdefault("model", {})
    model["hidden_size"] = int(args.hidden)
    model["num_layers"] = int(args.layers)
    cfg_path = write_tmp_config(cfg, f"h{args.hidden}_l{args.layers}")
    rc = launch(cfg_path, args.dry_run)
    print(f"[arch] finished rc={rc}")


def main() -> None:
    p = argparse.ArgumentParser(description="P0 training driver (multiseed / arch).")
    p.add_argument("--preset", required=True, choices=["multiseed", "arch"])
    p.add_argument("--out_base", default="data/p0_runs", help="output base dir (relative to project root)")
    p.add_argument("--epochs", type=int, default=None, help="override num_epochs (smoke); default=config (500)")
    p.add_argument("--dry_run", action="store_true", help="print the launch command without training")
    # multiseed
    p.add_argument("--seeds", default="42,7,123", help="comma-separated extra seeds (multiseed)")
    # arch
    p.add_argument("--hidden", type=int, default=128, help="GRU hidden size (arch)")
    p.add_argument("--layers", type=int, default=2, help="GRU num layers (arch)")
    args = p.parse_args()

    if args.preset == "multiseed":
        run_multiseed(args)
    else:
        run_arch(args)


if __name__ == "__main__":
    main()
