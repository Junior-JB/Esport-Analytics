"""Entry point for the personal project scaffold."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.capture import VideoSampler
from app.config import AppConfig
from app.pipeline import PipelineBundle, ValidationPipeline
from app.modules.hardpoint import HardpointModule
from app.modules.killfeed import KillfeedModule
from app.modules.scoreboard import ScoreboardRegistrationModule
from app.modules.utility import UtilityModule


def build_pipeline(config_path: str | Path) -> ValidationPipeline:
    """Build the ordered extraction-only pipeline."""
    config = AppConfig.load(config_path)
    bundle = PipelineBundle(
        scoreboard=ScoreboardRegistrationModule(
            config.scoreboard,
            config.scheduling.get("scoreboard_registration", {}),
        ),
        killfeed=KillfeedModule(config.killfeed, config.scheduling.get("killfeed", {})),
        utility=UtilityModule(config.utility, config.scheduling.get("utility", {})),
        hardpoint=HardpointModule(config.hardpoint, config.scheduling.get("hardpoint", {})),
    )
    return ValidationPipeline(bundle, debug_config=config.debug)


def main() -> int:
    """Run the current validation focus against the configured VOD."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "app_config.yaml"
    config = AppConfig.load(config_path)
    pipeline = build_pipeline(config_path)
    sampler = VideoSampler(
        video_path=config.capture.get("video_path", ""),
        max_samples_per_second=float(config.capture.get("max_capture_samples_per_second", 30.0)),
        start_time_seconds=float(config.capture.get("start_time_seconds", 0.0)),
    )

    frames_processed = 0
    for frame_packet in sampler.frames():
        pipeline.process_frame(frame_packet)
        frames_processed += 1

    dashboard = pipeline.dashboard.snapshot()
    final_summary_path = pipeline.finalize()
    print("Validation run complete.")
    print(f"Frames processed: {frames_processed}")
    print(f"Registered players: {dashboard.registered_players}")
    print(
        "Utility counts:",
        {
            "grenade": dashboard.utility_counts.grenade,
            "stun": dashboard.utility_counts.stun,
            "specialty_equipment": dashboard.utility_counts.specialty_equipment,
        },
    )
    print(f"Last OCR confidence: {dashboard.ocr_confidence}")
    print(f"Module status: {dashboard.module_status}")
    print(f"Final summary: {final_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
