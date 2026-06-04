"""Simple file-based frame sampling for validation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from app.models import FramePacket


class VideoSampler:
    """Yield sampled frames from a local video file at a fixed max rate."""

    def __init__(
        self,
        video_path: str,
        max_samples_per_second: float,
        start_time_seconds: float = 0.0,
    ) -> None:
        self.video_path = video_path
        self.max_samples_per_second = max(0.1, float(max_samples_per_second))
        self.start_time_seconds = max(0.0, float(start_time_seconds))

    def frames(self) -> Iterator[FramePacket]:
        """Yield sampled frame packets from the configured video."""
        path = Path(self.video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video file: {path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 29.97
        frame_interval = max(1, int(round(fps / self.max_samples_per_second)))
        start_frame_index = int(round(self.start_time_seconds * fps))

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index >= start_frame_index and frame_index % frame_interval == 0:
                yield FramePacket(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    source_path=str(path),
                    image=frame,
                )
            frame_index += 1
        capture.release()
