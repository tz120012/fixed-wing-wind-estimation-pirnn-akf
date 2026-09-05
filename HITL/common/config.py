"""Configuration and session-path helpers shared by HITL programs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = PROJECT_ROOT / "HITL/config/campaign.yaml"
DEFAULT_LOCAL = PROJECT_ROOT / "HITL/config/local.yaml"
SESSION_ID_RE = re.compile(r"^(id|ood)_[0-9]{2}$")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_configs(
    campaign_path: str | Path = DEFAULT_CAMPAIGN,
    local_path: str | Path = DEFAULT_LOCAL,
    *,
    require_local: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = _read_yaml(Path(campaign_path).expanduser().resolve())
    local_file = Path(local_path).expanduser().resolve()
    if require_local:
        local = _read_yaml(local_file)
    else:
        local = _read_yaml(local_file) if local_file.exists() else {}
    validate_campaign(campaign)
    return campaign, local


def validate_campaign(config: Mapping[str, Any]) -> None:
    protocol = config.get("protocol", {})
    conditions = config.get("conditions", {})
    if float(protocol.get("estimator_rate_hz", 0)) <= 0:
        raise ValueError("protocol.estimator_rate_hz must be positive")
    if float(protocol.get("total_duration_s", 0)) <= float(
        protocol.get("warmup_exclusion_s", 0)
    ):
        raise ValueError("total_duration_s must exceed warmup_exclusion_s")
    required = int(protocol.get("required_sessions_per_condition", 0))
    if required <= 0:
        raise ValueError("required_sessions_per_condition must be positive")
    for name in ("id", "ood"):
        condition = conditions.get(name)
        if not isinstance(condition, Mapping):
            raise ValueError(f"Missing condition: {name}")
        speeds = condition.get("speeds_mps", [])
        directions = condition.get("directions_deg", [])
        if len(speeds) != required or len(directions) != required:
            raise ValueError(
                f"{name} must define exactly {required} speeds and directions"
            )
        low, high = map(float, condition["horizontal_wind_range_mps"])
        if any(not low <= float(speed) <= high for speed in speeds):
            raise ValueError(f"{name} speed lies outside [{low}, {high}] m/s")


def parse_session_id(session_id: str) -> tuple[str, int]:
    normalized = session_id.strip().lower()
    if not SESSION_ID_RE.fullmatch(normalized):
        raise ValueError("session_id must match id_01..id_05 or ood_01..ood_05")
    condition, index = normalized.split("_")
    return condition, int(index)


def session_wind(campaign: Mapping[str, Any], session_id: str) -> dict[str, float]:
    condition, one_based_index = parse_session_id(session_id)
    cfg = campaign["conditions"][condition]
    index = one_based_index - 1
    return {
        "speed_mps": float(cfg["speeds_mps"][index]),
        "direction_deg": float(cfg["directions_deg"][index]),
        "turbulence_gain": float(cfg.get("turbulence_gain", 1.0)),
        "gust_magnitude_mps": float(cfg.get("gust_magnitude_mps", 0.0)),
    }


@dataclass(frozen=True)
class SessionPaths:
    root: Path
    pc: Path
    pi: Path
    fc: Path
    aligned: Path
    summary: Path

    @classmethod
    def build(
        cls, campaign: Mapping[str, Any], session_id: str, *, create: bool = False
    ) -> "SessionPaths":
        parse_session_id(session_id)
        configured = Path(campaign["logging"]["sessions_root"])
        base = configured if configured.is_absolute() else PROJECT_ROOT / configured
        root = base / session_id
        result = cls(
            root=root,
            pc=root / "pc",
            pi=root / "pi",
            fc=root / "fc",
            aligned=root / "aligned",
            summary=root / "summary",
        )
        if create:
            for path in (
                result.pc,
                result.pi,
                result.fc,
                result.aligned,
                result.summary,
            ):
                path.mkdir(parents=True, exist_ok=True)
        return result
