
from __future__ import annotations

from app.models import (
    DashboardState,
    HardpointSnapshot,
    KillfeedCountSnapshot,
    MatchContext,
    ModuleStatus,
    ScoreboardSnapshot,
    UtilityCountSnapshot,
)


class ValidationDashboard:
    """Assemble a validation snapshot from the current module state."""

    def __init__(self) -> None:
        self._state = DashboardState()

    def update_match_context(self, context: MatchContext) -> None:
        """Refresh the registered player list from the match context."""
        self._state.registered_players = context.all_player_names

    def update_scoreboard(self, snapshot: ScoreboardSnapshot) -> None:
        """Refresh scoreboard registration/debug state."""
        self._state.scoreboard = snapshot

    def update_frame_context(self, source_path: str, frame_index: int, timestamp_seconds: float) -> None:
        """Refresh the latest processed frame metadata."""
        self._state.source_path = source_path
        self._state.frame_index = frame_index
        self._state.timestamp_seconds = timestamp_seconds

    def update_killfeed(self, snapshot: KillfeedCountSnapshot) -> None:
        self._state.killfeed_counts = snapshot

    def update_utility(self, snapshot: UtilityCountSnapshot) -> None:
        self._state.utility_counts = snapshot

    def update_hardpoint(self, snapshot: HardpointSnapshot) -> None:
        self._state.hardpoint = snapshot

    def update_confidence(self, confidence: float | None) -> None:
        self._state.ocr_confidence = confidence

    def update_status(self, module_name: str, status: ModuleStatus) -> None:
        self._state.module_status[module_name] = status

    def snapshot(self) -> DashboardState:
        """Return the current validation dashboard state."""
        return self._state
