"""Configuration loading for the personal project scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def parse_timestamp_to_seconds(value: object) -> float:
    """Parse a timestamp value like 192 or '3:12' into seconds."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" not in text:
        return max(0.0, float(text))
    parts = [part.strip() for part in text.split(":")]
    if len(parts) != 2:
        raise ValueError(f"Unsupported timestamp format: {value}")
    minutes = int(parts[0])
    seconds = float(parts[1])
    return max(0.0, (minutes * 60.0) + seconds)


@dataclass(slots=True)
class AppConfig:
    """Small wrapper around raw YAML configuration."""

    raw: dict

    @classmethod
    def load(cls, config_path: str | Path) -> "AppConfig":
        """Load the scaffold configuration file."""
        path = Path(config_path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(raw=data)

    @property
    def capture(self) -> dict:
        capture = dict(self.raw.get("capture", {}))
        capture["start_time_seconds"] = parse_timestamp_to_seconds(
            capture.get("start_time", capture.get("start_time_seconds", 0.0))
        )
        return capture

    @property
    def scheduling(self) -> dict:
        return dict(self.raw.get("scheduling", {}))

    @property
    def scoreboard(self) -> dict:
        return dict(self.raw.get("scoreboard_registration", {}))

    @property
    def killfeed(self) -> dict:
        return dict(self.raw.get("killfeed", {}))

    @property
    def utility(self) -> dict:
        return dict(self.raw.get("utility", {}))

    @property
    def hardpoint(self) -> dict:
        return dict(self.raw.get("hardpoint", {}))

    @property
    def dashboard(self) -> dict:
        return dict(self.raw.get("dashboard", {}))

    @property
    def debug(self) -> dict:
        return dict(self.raw.get("debug", {}))
