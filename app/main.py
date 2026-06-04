"""Entry point for the personal project scaffold."""

from __future__ import annotations

import copy
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
from app.models import ModuleStatus


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


def run_pipeline_pass(config: AppConfig, pipeline: ValidationPipeline) -> tuple[int, Path]:
    """Run one full video pass through the provided pipeline."""
    sampler = VideoSampler(
        video_path=config.capture.get("video_path", ""),
        max_samples_per_second=float(config.capture.get("max_capture_samples_per_second", 30.0)),
        start_time_seconds=float(config.capture.get("start_time_seconds", 0.0)),
    )
    frames_processed = 0
    for frame_packet in sampler.frames():
        pipeline.process_frame(frame_packet)
        frames_processed += 1
    final_summary_path = pipeline.finalize()
    return frames_processed, final_summary_path


def main() -> int:
    """Run the current validation focus against the configured VOD."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "app_config.yaml"
    config = AppConfig.load(config_path)
    first_pass = build_pipeline(config_path)
    first_pass.bundle.killfeed.enabled = False
    first_pass.bundle.killfeed.status = ModuleStatus.DISABLED
    first_pass_frames_processed, first_pass_summary = run_pipeline_pass(config, first_pass)

    print("First pass complete.")
    print(f"Frames processed: {first_pass_frames_processed}")
    print(f"Registered players: {first_pass.dashboard.snapshot().registered_players}")
    print(f"First-pass summary: {first_pass_summary}")

    registry_ready = (
        first_pass.match_context.registration_complete
        and first_pass.match_context.registry_locked
        and len(first_pass.match_context.players) == 8
    )
    if not registry_ready:
        print("Killfeed second pass skipped: locked 8-player registry was not established.")
        return 0

    second_pass = build_pipeline(config_path)
    second_pass.match_context = copy.deepcopy(first_pass.match_context)
    second_pass.match_context.registration_complete = True
    second_pass.match_context.registry_locked = True
    second_pass.bundle.killfeed.enabled = True
    second_pass.bundle.killfeed.status = ModuleStatus.WAITING
    second_pass.bundle.utility.enabled = False
    second_pass.bundle.utility.status = ModuleStatus.DISABLED
    second_pass.bundle.hardpoint.enabled = False
    second_pass.bundle.hardpoint.status = ModuleStatus.DISABLED
    second_pass_frames_processed, final_summary_path = run_pipeline_pass(config, second_pass)

    dashboard = second_pass.dashboard.snapshot()
    print("Second pass complete.")
    print(f"Frames processed: {second_pass_frames_processed}")
    print(f"Registered players: {dashboard.registered_players}")
    print(
        "Utility counts:",
        {
            "grenade": dashboard.utility_counts.grenade,
            "stun": dashboard.utility_counts.stun,
            "specialty_equipment": dashboard.utility_counts.specialty_equipment,
        },
    )
    print(
        "Killfeed detections:",
        {
            "name_detections": dashboard.killfeed_counts.name_detections,
            "resolved_name_detections": dashboard.killfeed_counts.resolved_name_detections,
        },
    )
    print(f"Last OCR confidence: {dashboard.ocr_confidence}")
    print(f"Module status: {dashboard.module_status}")
    print(f"Final summary: {final_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
