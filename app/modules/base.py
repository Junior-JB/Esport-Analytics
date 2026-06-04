"""Base module contracts for the scaffold."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import FramePacket, MatchContext, ModuleStatus


class PipelineModule(ABC):
    """Shared interface for all ordered extraction modules."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = ModuleStatus.IDLE

    @abstractmethod
    def process(self, frame_packet: FramePacket, match_context: MatchContext) -> None:
        """Process one sampled frame."""

    def disable(self) -> None:
        """Disable the module once it should no longer run."""
        self.status = ModuleStatus.DISABLED
