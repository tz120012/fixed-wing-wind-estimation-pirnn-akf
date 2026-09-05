"""Stage one unambiguous set of raw HITL artifacts into a canonical session."""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HITL.common.config import DEFAULT_CAMPAIGN, DEFAULT_LOCAL, SessionPaths, load_configs
from HITL.common.manifest import artifact_record, write_json_atomic
from HITL.common.schema import (
    PI_REQUIRED_COLUMNS,
    TRUTH_REQUIRED_COLUMNS,
    require_columns,
    require_constant_text,
)


def _resolve_one(value: str | Path, suffix: str) -> Path:
    source = Path(value).expanduser().resolve()
    if source.is_file():
        candidates = [source]
    elif source.is_dir():
        candidates = sorted(path for path in source.iterdir() if path.suffix.lower() == suffix)
    else:
        candidates = sorted(Path(path).resolve() for path in glob.glob(str(value)))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one {suffix} file from {value}, got {candidates}")
    return candidates[0].resolve()


def _filename_session(path: Path) -> str | None:
    matches = set(re.findall(r"(?:^|[^a-z])(id|ood)_(\d{2})(?:[^0-9]|$)", path.name.lower()))
    if len(matches) > 1:
        raise ValueError(f"{path}: filename contains ambiguous session IDs")
    if not matches:
        return None
    condition, index = matches.pop()
    return f"{condition}_{index}"


def _verify_csv(path: Path, session_id: str, *, pi: bool) -> None:
    frame = pd.read_csv(path)
    required = PI_REQUIRED_COLUMNS if pi else TRUTH_REQUIRED_COLUMNS
    require_columns(frame, required, source=str(path))
    if frame.empty:
        raise ValueError(f"{path}: CSV is empty")
    if pi:
        require_constant_text(frame, "session_id", session_id, source=str(path))
        require_constant_text(frame, "condition", session_id.split("_")[0], source=str(path))
    elif "session_id" in frame.columns:
        require_constant_text(frame, "session_id", session_id, source=str(path))
    embedded = _filename_session(path)
    if embedded is not None and embedded != session_id:
        raise ValueError(f"{path}: filename session {embedded} != requested {session_id}")


def prepare_session(
    session_id: str,
    truth_source: str | Path,
    pi_source: str | Path,
    *,
    ulog_source: str | Path | None = None,
    campaign_path: str | Path = DEFAULT_CAMPAIGN,
    local_path: str | Path = DEFAULT_LOCAL,
    copy_files: bool = True,
) -> dict[str, Any]:
    campaign, _ = load_configs(campaign_path, local_path, require_local=False)
    paths = SessionPaths.build(campaign, session_id, create=True)
    truth = _resolve_one(truth_source, ".csv")
    pi = _resolve_one(pi_source, ".csv")
    ulog = _resolve_one(ulog_source, ".ulg") if ulog_source is not None else None
    _verify_csv(truth, session_id, pi=False)
    _verify_csv(pi, session_id, pi=True)
    if ulog is not None:
        embedded = _filename_session(ulog)
        if embedded is not None and embedded != session_id:
            raise ValueError(f"{ulog}: filename session {embedded} != requested {session_id}")

    configured = campaign["logging"]
    destinations = {
        "pc_truth": paths.pc / configured["pc_truth_name"],
        "pi_estimator": paths.pi / configured["pi_csv_name"],
    }
    if ulog is not None:
        destinations["fc_ulog"] = paths.fc / configured["fc_ulog_name"]
    sources = {"pc_truth": truth, "pi_estimator": pi, "fc_ulog": ulog}

    artifacts: dict[str, Any] = {}
    for role, destination in destinations.items():
        source = sources[role]
        assert source is not None
        if destination.exists() and destination.resolve() != source:
            raise FileExistsError(f"refusing to replace existing canonical file: {destination}")
        if destination.resolve() != source:
            if copy_files:
                shutil.copy2(source, destination)
            else:
                destination.symlink_to(source)
        indexed = destination
        artifacts[role] = {
            "source": artifact_record(source),
            "canonical_path": str(destination.absolute()),
            "indexed": artifact_record(indexed),
        }

    inventory = {
        "schema_version": 1,
        "session_id": session_id,
        "condition": session_id.split("_")[0],
        "mode": "copy" if copy_files else "index",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    write_json_atomic(paths.root / "inventory.json", inventory)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id")
    parser.add_argument("--truth", required=True)
    parser.add_argument("--pi", required=True)
    parser.add_argument("--ulog")
    parser.add_argument("--campaign", default=str(DEFAULT_CAMPAIGN))
    parser.add_argument("--local", default=str(DEFAULT_LOCAL))
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    result = prepare_session(
        args.session_id,
        args.truth,
        args.pi,
        ulog_source=args.ulog,
        campaign_path=args.campaign,
        local_path=args.local,
        copy_files=not args.index_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
