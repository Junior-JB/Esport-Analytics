"""Shared models for the extraction and validation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModuleStatus(str, Enum):
    """High-level module lifecycle states for validation."""

    IDLE = "idle"
    WAITING = "waiting"
    ACTIVE = "active"
    COMPLETE = "complete"
    DISABLED = "disabled"


@dataclass(slots=True)
class FramePacket:
    """Sampled frame metadata passed through the pipeline."""

    frame_index: int
    timestamp_seconds: float
    source_path: str
    image: Any | None = None


@dataclass(slots=True)
class RegisteredPlayer:
    """One validated scoreboard player registration."""

    player_id: int
    name: str
    team: str
    confidence: float
    slot_index: int
    observation_count: int = 0


@dataclass(slots=True)
class FinalScoreSnapshot:
    """Authoritative final score, when detected."""

    blue: int | None = None
    red: int | None = None
    source: str | None = None


@dataclass(slots=True)
class MatchContext:
    """Shared match registration state used by downstream modules."""

    match_id: str = "001"
    players: list[RegisteredPlayer] = field(default_factory=list)
    player_registry: dict[str, str] = field(default_factory=dict)
    registration_complete: bool = False
    registry_locked: bool = False
    registry_confidence: float | None = None
    final_score: FinalScoreSnapshot = field(default_factory=FinalScoreSnapshot)

    @property
    def all_player_names(self) -> list[str]:
        """Return the registered 8-player roster."""
        return [player.name for player in sorted(self.players, key=lambda player: player.player_id)]

    @property
    def friendly_team_names(self) -> list[str]:
        """Return the authoritative friendly roster."""
        return [player.name for player in self.players if player.team == "friendly"]

    @property
    def enemy_team_names(self) -> list[str]:
        """Return the authoritative enemy roster."""
        return [player.name for player in self.players if player.team == "enemy"]


@dataclass(slots=True)
class ScoreboardRegistrationResult:
    """Validated scoreboard registration output."""

    players: list[RegisteredPlayer]
    registration_complete: bool
    confidence: float


@dataclass(slots=True)
class ScoreboardSnapshot:
    """Current scoreboard registration/debug state."""

    scoreboard_visible: bool = False
    current_state: str = "waiting"
    registered_players: list[str] = field(default_factory=list)
    registration_complete: bool = False
    registry_confidence: float | None = None
    final_score_detected: bool = False
    ocr_confidence: float | None = None
    player_registry: dict[str, str] = field(default_factory=dict)
    final_score_blue: int | None = None
    final_score_red: int | None = None


@dataclass(slots=True)
class KillfeedCountSnapshot:
    """Simple killfeed validation counts."""

    kills: int = 0
    deaths: int = 0
    trades: int = 0
    name_detections: int = 0
    resolved_name_detections: int = 0
    row_groups_detected: int = 0
    single_side_row_groups: int = 0
    event_threads_created: int = 0
    event_threads_reused: int = 0
    events_counted: int = 0
    events_blocked_victim_cooldown: int = 0
    events_blocked_low_confidence: int = 0
    events_blocked_same_side: int = 0
    player_detection_counts: dict[str, int] = field(default_factory=dict)
    player_kills: dict[str, int] = field(default_factory=dict)
    player_deaths: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class UtilityCountSnapshot:
    """Simple utility validation counts."""

    grenade: int = 0
    stun: int = 0
    specialty_equipment: int = 0


@dataclass(slots=True)
class HardpointSnapshot:
    """Validated Hardpoint state only."""

    blue_score: int | None = None
    red_score: int | None = None
    current_hill: int | None = None


@dataclass(slots=True)
class DashboardState:
    """Validation dashboard payload."""

    source_path: str | None = None
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    registered_players: list[str] = field(default_factory=list)
    scoreboard: ScoreboardSnapshot = field(default_factory=ScoreboardSnapshot)
    killfeed_counts: KillfeedCountSnapshot = field(default_factory=KillfeedCountSnapshot)
    utility_counts: UtilityCountSnapshot = field(default_factory=UtilityCountSnapshot)
    hardpoint: HardpointSnapshot = field(default_factory=HardpointSnapshot)
    ocr_confidence: float | None = None
    module_status: dict[str, ModuleStatus] = field(default_factory=dict)
