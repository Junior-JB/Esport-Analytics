"""Killfeed extraction scaffold."""

from __future__ import annotations

from app.models import FramePacket, KillfeedCountSnapshot, MatchContext, ModuleStatus
from app.modules.base import PipelineModule
from app.scheduler import FixedRateScheduler


class KillfeedModule(PipelineModule):
    """Primary telemetry source, gated on scoreboard registration."""

    def __init__(self, config: dict, scheduling_config: dict) -> None:
        super().__init__(name="killfeed")
        self.config = config
        self.enabled = bool(config.get("enabled", True))
        self.scheduler = FixedRateScheduler(scheduling_config.get("samples_per_second", 3.0))
        self.snapshot = KillfeedCountSnapshot()
        self.status = ModuleStatus.WAITING if self.enabled else ModuleStatus.DISABLED

    def process(self, frame_packet: FramePacket, match_context: MatchContext) -> None:
        """Run only after scoreboard registration succeeds."""
        if not self.enabled:
            self.status = ModuleStatus.DISABLED
            return
        if not match_context.registration_complete:
            self.status = ModuleStatus.WAITING
            return
        if not self.scheduler.should_run(frame_packet.timestamp_seconds):
            return
        self.status = ModuleStatus.ACTIVE
