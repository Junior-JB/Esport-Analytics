"""Fixed-rate scheduling helpers."""

from __future__ import annotations


class FixedRateScheduler:
    """Allow one module tick only when its next sample time is due."""

    def __init__(self, samples_per_second: float) -> None:
        self.samples_per_second = max(0.0, float(samples_per_second))
        self._last_timestamp_seconds: float | None = None

    def should_run(self, timestamp_seconds: float) -> bool:
        """Return whether the module should process this sampled frame."""
        if self.samples_per_second <= 0.0:
            return False
        if self._last_timestamp_seconds is None:
            self._last_timestamp_seconds = timestamp_seconds
            return True
        interval_seconds = 1.0 / self.samples_per_second
        if (timestamp_seconds - self._last_timestamp_seconds) >= interval_seconds:
            self._last_timestamp_seconds = timestamp_seconds
            return True
        return False
