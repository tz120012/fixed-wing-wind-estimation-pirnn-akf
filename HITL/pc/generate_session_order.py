#!/usr/bin/env python3
"""Generate the frozen randomized order for the ten manual-QGC sessions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HITL.common.config import DEFAULT_CAMPAIGN, SessionPaths, load_configs
from HITL.common.manifest import sha256_file, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    campaign_path = args.campaign.expanduser().resolve()
    campaign, _ = load_configs(campaign_path, require_local=False)
    required = int(campaign["protocol"]["required_sessions_per_condition"])
    seed = (
        args.seed
        if args.seed is not None
        else int(campaign["protocol"]["randomization_seed"])
    )
    order = [
        f"{condition}_{index:02d}"
        for condition in ("id", "ood")
        for index in range(1, required + 1)
    ]
    random.Random(seed).shuffle(order)

    if args.output:
        output = args.output.expanduser().resolve()
    else:
        first = SessionPaths.build(campaign, "id_01")
        output = first.root.parent / "session_order.json"
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; use --force")
    payload = {
        "schema_version": 1,
        "campaign": str(campaign_path),
        "campaign_sha256": sha256_file(campaign_path),
        "seed": seed,
        "session_count": len(order),
        "order": order,
    }
    write_json_atomic(output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
