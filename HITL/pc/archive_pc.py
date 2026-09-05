#!/usr/bin/env python3
"""Copy PC truth/log artifacts into a session directory with provenance."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from HITL.common.config import (  # noqa: E402
    DEFAULT_CAMPAIGN,
    SessionPaths,
    load_configs,
    parse_session_id,
)
from HITL.common.manifest import (  # noqa: E402
    artifact_record,
    sha256_file,
    software_manifest,
    write_json_atomic,
)


def _unique_destination(directory: Path, prefix: str, source: Path) -> Path:
    safe_name = source.name
    if safe_name in ("", ".", ".."):
        raise ValueError(f"unsafe source name: {safe_name!r}")
    candidate = directory / f"{prefix}_{safe_name}"
    index = 1
    while candidate.exists():
        candidate = directory / f"{prefix}_{source.stem}_{index}{source.suffix}"
        index += 1
    resolved = candidate.resolve()
    if resolved.parent != directory.resolve():
        raise ValueError("archive destination escaped the session PC directory")
    return resolved


def _copy(source_value: Path, destination_dir: Path, kind: str) -> dict[str, Any]:
    source = source_value.expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"{kind} source is not a regular file: {source}")
    if source.parent == destination_dir.resolve():
        raise ValueError(f"{kind} source is already in the destination directory: {source}")
    destination = _unique_destination(destination_dir, kind, source)
    source_hash = sha256_file(source)
    shutil.copy2(source, destination)
    copied_hash = sha256_file(destination)
    if copied_hash != source_hash:
        destination.unlink(missing_ok=True)
        raise IOError(f"hash mismatch after copying {source}")
    return {
        "kind": kind,
        "source": str(source),
        "source_sha256": source_hash,
        "artifact": artifact_record(destination),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id", help="session identifier, for example id_01")
    parser.add_argument("--truth-source", type=Path, help="wind_truth CSV to archive")
    parser.add_argument("--log-source", type=Path, help="PC-side launch log to archive")
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    session_id = args.session_id.strip().lower()
    parse_session_id(session_id)
    if args.truth_source is None and args.log_source is None:
        raise SystemExit("provide --truth-source and/or --log-source")
    campaign, _ = load_configs(args.campaign, require_local=False)
    destination = SessionPaths.build(campaign, session_id, create=True).pc.resolve()

    copied: list[dict[str, Any]] = []
    if args.truth_source is not None:
        copied.append(_copy(args.truth_source, destination, "truth"))
    if args.log_source is not None:
        copied.append(_copy(args.log_source, destination, "log"))

    manifest_path = destination / "archive_pc_manifest.json"
    history: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(previous, dict) and isinstance(previous.get("archives"), list):
                history = previous["archives"]
        except (OSError, ValueError):
            raise SystemExit(f"existing archive manifest is invalid: {manifest_path}")
    event = {
        "archived_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "files": copied,
    }
    history.append(event)
    manifest = {
        **software_manifest(),
        "session_id": session_id,
        "archives": history,
    }
    write_json_atomic(manifest_path, manifest)
    print(f"Archived {len(copied)} file(s) into {destination}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
