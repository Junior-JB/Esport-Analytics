"""Create one ROI preview image for a configured module at a chosen timestamp."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml


DEFAULT_VIDEO_PATH = Path("/Users/juniorbenitez/Desktop/vod/2026-05-29 14-44-38.mkv")
DEFAULT_CONFIG_PATH = Path("/Users/juniorbenitez/Documents/main project/personal_project/config/app_config.yaml")
DEFAULT_OUTPUT_DIR = Path("/Users/juniorbenitez/Documents/main project/personal_project/debug_previews")

SECTION_ALIASES = {
    "scoreboard": "scoreboard_registration",
    "scoreboard_registration": "scoreboard_registration",
    "utility": "utility",
    "hardpoint": "hardpoint",
}

SECTION_COLORS = {
    "scoreboard_registration": (0, 255, 255),
    "utility": (0, 255, 0),
    "hardpoint": (255, 255, 0),
}


def parse_timestamp_to_seconds(value: str) -> float:
    """Parse a timestamp like 14:12 or 852 into seconds."""
    text = str(value).strip()
    if not text:
        raise ValueError("Timestamp is required")
    if ":" not in text:
        return float(text)
    minutes_text, seconds_text = text.split(":", 1)
    return (int(minutes_text) * 60.0) + float(seconds_text)


def sanitize_timestamp_label(value: str) -> str:
    """Convert a timestamp like 14:12 into a safe filename fragment."""
    text = value.strip()
    if ":" not in text:
        return f"{text}s"
    minutes_text, seconds_text = text.split(":", 1)
    return f"{minutes_text}m{seconds_text}s"


def resolve_regions(config_path: Path, section_name: str) -> dict:
    """Load normalized regions for one config section."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = data.get(section_name, {})
    regions = dict(section.get("absolute_regions", {}))
    if not regions:
        raise SystemExit(f"No absolute_regions found for section: {section_name}")
    return regions


def read_frame(video_path: Path, timestamp_seconds: float):
    """Read one frame from the video at the target timestamp."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise SystemExit(f"Could not read frame at {timestamp_seconds:.3f}s")
    return frame


def draw_regions(frame, regions: dict, color: tuple[int, int, int], title: str) -> None:
    """Draw all configured ROIs for one section onto the frame."""
    frame_height, frame_width = frame.shape[:2]
    cv2.putText(frame, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    for name, region in regions.items():
        x = int(region.get("x", 0.0) * frame_width)
        y = int(region.get("y", 0.0) * frame_height)
        width = int(region.get("width", 0.0) * frame_width)
        height = int(region.get("height", 0.0) * frame_height)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            frame,
            name,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def build_output_path(section_name: str, timestamp_label: str, output_name: str | None) -> Path:
    """Return the final preview output path."""
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_name:
        return DEFAULT_OUTPUT_DIR / output_name
    return DEFAULT_OUTPUT_DIR / f"{section_name}_preview_{timestamp_label}.jpg"


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", help="Config section: scoreboard, utility, or hardpoint")
    parser.add_argument("timestamp", help="Timestamp like 14:12 or 852")
    parser.add_argument("output_name", nargs="?", help="Optional output filename")
    parser.add_argument("--video", default=str(DEFAULT_VIDEO_PATH), help="Optional video path override")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Optional config path override")
    args = parser.parse_args()

    section_name = SECTION_ALIASES.get(args.section.strip().lower())
    if not section_name:
        raise SystemExit(f"Unsupported section: {args.section}")

    timestamp_seconds = parse_timestamp_to_seconds(args.timestamp)
    timestamp_label = sanitize_timestamp_label(args.timestamp)
    frame = read_frame(Path(args.video), timestamp_seconds)
    regions = resolve_regions(Path(args.config), section_name)
    color = SECTION_COLORS.get(section_name, (0, 255, 255))
    draw_regions(frame, regions, color, f"{section_name} preview {args.timestamp}")

    output_path = build_output_path(section_name, timestamp_label, args.output_name)
    cv2.imwrite(str(output_path), frame)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
