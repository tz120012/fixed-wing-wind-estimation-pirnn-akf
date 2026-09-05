#!/usr/bin/env python3
"""Apply the reproducible electric-motor fix to PX4's JSBSim Malolo model.

The stock Malolo gasoline piston model can receive an impulsive propeller
torque during runway takeoff with current JSBSim builds, producing an immediate
gear-impact instability. This patch installs a 750 W electric engine so that
the lightweight model remains inside its tabulated airspeed domain while
leaving geometry, mass, aerodynamics and control allocation unchanged.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


OLD_ENGINE = '<engine file="Zenoah_G-26A">'
LEGACY_ENGINES = (
    '<engine file="electrical_engine_1050W">',
    '<engine file="revision_electrical_engine_450W">',
)
NEW_ENGINE = '<engine file="revision_electrical_engine_750W">'
ELECTRIC_ENGINE = """<?xml version="1.0"?>
<electric_engine name="revision_electrical_engine_750W">
  <power unit="WATTS"> 750 </power>
</electric_engine>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--px4-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = (
        args.px4_root
        / "Tools/jsbsim_bridge/models/ATI-Resolution"
    )
    model = model_dir / "Malolo1.xml"
    engine = model_dir / "Engines/revision_electrical_engine_750W.xml"
    if not model.exists():
        raise FileNotFoundError(model)

    text = model.read_text(encoding="utf-8")
    if NEW_ENGINE not in text:
        source_engine = next(
            (item for item in LEGACY_ENGINES if item in text),
            OLD_ENGINE,
        )
        if source_engine not in text:
            raise RuntimeError(
                f"No supported engine declaration found in {model}"
            )
        backup = model.with_suffix(".xml.revision-backup")
        if not backup.exists():
            shutil.copy2(model, backup)
        model.write_text(
            text.replace(source_engine, NEW_ENGINE, 1),
            encoding="utf-8",
        )
    engine.write_text(ELECTRIC_ENGINE, encoding="utf-8")
    print(f"Patched Malolo propulsion: {model}")
    print(f"Installed electric engine: {engine}")


if __name__ == "__main__":
    main()
