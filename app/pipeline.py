"""Ordered validation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import csv
import json
from collections import Counter

from app.dashboard import ValidationDashboard
from app.models import FramePacket, MatchContext, ModuleStatus
from app.modules.hardpoint import HardpointModule
from app.modules.killfeed import KillfeedModule
from app.modules.scoreboard import ScoreboardRegistrationModule
from app.modules.utility import UtilityModule


@dataclass(slots=True)
class PipelineBundle:
    """Ordered module set for the validation pipeline."""

    scoreboard: ScoreboardRegistrationModule
    killfeed: KillfeedModule
    utility: UtilityModule
    hardpoint: HardpointModule


class ValidationPipeline:
    """Execute modules in the required order and update the dashboard."""

    def __init__(self, bundle: PipelineBundle, debug_config: dict | None = None) -> None:
        self.bundle = bundle
        self.debug_config = dict(debug_config or {})
        self.match_context = MatchContext()
        self.dashboard = ValidationDashboard()
        self._active_source_path: str | None = None
        self._output_root = Path(__file__).resolve().parent.parent / "debug_validation"
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._cleanup_old_runs = bool(self.debug_config.get("cleanup_old_runs", True))
        self._keep_recent_runs = int(self.debug_config.get("keep_recent_runs", 1))
        self._run_output_dir = self._output_root
        self._live_json_path = self._output_root / "live_state.json"
        self._final_txt_path = self._output_root / "final_summary.txt"
        self._hardpoint_csv_path = self._output_root / "hardpoint_timeline.csv"
        self._hardpoint_analytics_csv_path = self._output_root / "hardpoint_timeline_final_analytics.csv"
        self._last_written_frame_index: int | None = None
        self._last_tangible_blue_score: int | None = None
        self._last_tangible_red_score: int | None = None

    def process_frame(self, frame_packet: FramePacket) -> None:
        """Run modules in the required fixed order."""
        self._prepare_run_context(frame_packet.source_path)
        self.bundle.scoreboard.process(frame_packet, self.match_context)

        self.bundle.killfeed.process(frame_packet, self.match_context)
        self.bundle.utility.process(frame_packet, self.match_context)
        self.bundle.hardpoint.process(frame_packet, self.match_context)
        self._refresh_dashboard(frame_packet)
        self._write_live_outputs()

    def finalize(self) -> Path:
        """Write one final text summary for the completed run."""
        if hasattr(self.bundle.scoreboard, "finalize"):
            self.bundle.scoreboard.finalize()
        self._write_hardpoint_analytics_csv()
        self._write_final_summary()
        return self._final_txt_path

    def _refresh_dashboard(self, frame_packet: FramePacket) -> None:
        """Mirror current module state into the validation dashboard."""
        self.dashboard.update_frame_context(
            source_path=frame_packet.source_path,
            frame_index=frame_packet.frame_index,
            timestamp_seconds=frame_packet.timestamp_seconds,
        )
        self.dashboard.update_match_context(self.match_context)
        self.dashboard.update_scoreboard(self.bundle.scoreboard.snapshot)
        self.dashboard.update_killfeed(self.bundle.killfeed.snapshot)
        self.dashboard.update_utility(self.bundle.utility.snapshot)
        self.dashboard.update_hardpoint(self.bundle.hardpoint.snapshot)
        dominant_confidence = (
            self.bundle.scoreboard.last_ocr_confidence
            or self.bundle.utility.last_confidence
            or self.bundle.hardpoint.last_confidence
        )
        self.dashboard.update_confidence(dominant_confidence)
        self.dashboard.update_status(self.bundle.scoreboard.name, self.bundle.scoreboard.status)
        self.dashboard.update_status(self.bundle.killfeed.name, self.bundle.killfeed.status)
        self.dashboard.update_status(self.bundle.utility.name, self.bundle.utility.status)
        self.dashboard.update_status(self.bundle.hardpoint.name, self.bundle.hardpoint.status)

    def _prepare_run_context(self, source_path: str) -> None:
        """Initialize one output folder per source clip."""
        if source_path == self._active_source_path:
            return
        self._active_source_path = source_path
        self._cleanup_debug_runs()
        source_stem = Path(source_path).stem.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_output_dir = self._output_root / f"{source_stem}_{timestamp}"
        self._run_output_dir.mkdir(parents=True, exist_ok=True)
        self._live_json_path = self._run_output_dir / "live_state.json"
        self._final_txt_path = self._run_output_dir / "final_summary.txt"
        self._last_written_frame_index = None
        self._last_tangible_blue_score = None
        self._last_tangible_red_score = None
        if hasattr(self.bundle.scoreboard, "set_debug_output_dir"):
            self.bundle.scoreboard.set_debug_output_dir(self._run_output_dir)
        self._initialize_hardpoint_csv()

    def _cleanup_debug_runs(self) -> None:
        """Keep only the configured number of recent validation runs."""
        if not self._cleanup_old_runs or not self._output_root.exists():
            return
        children = sorted(
            [path for path in self._output_root.iterdir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for child in children[self._keep_recent_runs :]:
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file():
                        nested.unlink(missing_ok=True)
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink(missing_ok=True)

    def _write_live_outputs(self) -> None:
        """Write one shared live JSON validation file."""
        state = self.dashboard.snapshot()
        serializable = asdict(state)
        serializable["module_status"] = {
            key: value.value for key, value in state.module_status.items()
        }
        self._live_json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        self._write_hardpoint_csv_row(state)

    def _initialize_hardpoint_csv(self) -> None:
        """Overwrite the single shared Hardpoint CSV for the new run."""
        self._hardpoint_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self._hardpoint_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "timestamp_seconds", "blue_score", "red_score", "hill_number"])

    def _write_hardpoint_csv_row(self, state) -> None:
        """Append one hardpoint row per sampled frame, filling forward blue/red values."""
        if state.timestamp_seconds is None or state.frame_index is None:
            return
        if self._last_written_frame_index == state.frame_index:
            return

        if state.hardpoint.blue_score is not None:
            self._last_tangible_blue_score = state.hardpoint.blue_score
        if state.hardpoint.red_score is not None:
            self._last_tangible_red_score = state.hardpoint.red_score

        blue_value = self._last_tangible_blue_score
        red_value = self._last_tangible_red_score
        hill_value = state.hardpoint.current_hill

        with self._hardpoint_csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    state.frame_index,
                    f"{state.timestamp_seconds:.3f}",
                    "" if blue_value is None else blue_value,
                    "" if red_value is None else red_value,
                    "" if hill_value is None else hill_value,
                ]
            )

        self._last_written_frame_index = state.frame_index

    def _write_final_summary(self) -> None:
        """Write one final text summary with all validation info."""
        state = self.dashboard.snapshot()
        lines = [
            "Validation Summary",
            f"Source: {state.source_path}",
            f"Last frame: {state.frame_index}",
            f"Last timestamp: {state.timestamp_seconds}",
            "",
            "Registered Players:",
        ]
        if state.registered_players:
            lines.extend(f"- {name}" for name in state.registered_players)
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "Scoreboard Registration:",
                f"Visible: {state.scoreboard.scoreboard_visible}",
                f"State: {state.scoreboard.current_state}",
                f"Registration complete: {state.scoreboard.registration_complete}",
                f"Registry confidence: {state.scoreboard.registry_confidence}",
                f"Final score detected: {state.scoreboard.final_score_detected}",
                f"Player registry: {state.scoreboard.player_registry}",
                "",
                "Killfeed Counts:",
                f"Kills: {state.killfeed_counts.kills}",
                f"Deaths: {state.killfeed_counts.deaths}",
                f"Trades: {state.killfeed_counts.trades}",
                "",
                "Utility Counts:",
                f"Grenade: {state.utility_counts.grenade}",
                f"Stun: {state.utility_counts.stun}",
                f"Specialty equipment: {state.utility_counts.specialty_equipment}",
                "",
                "Hardpoint:",
                f"Blue score: {state.hardpoint.blue_score}",
                f"Red score: {state.hardpoint.red_score}",
                f"Current hill: {state.hardpoint.current_hill}",
                "",
                f"OCR confidence: {state.ocr_confidence}",
                "",
                "Module Status:",
            ]
        )
        for module_name, status in state.module_status.items():
            lines.append(f"- {module_name}: {status.value}")
        self._final_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_hardpoint_analytics_csv(self) -> None:
        """Write a cleaned post-analysis CSV from the raw hardpoint timeline, aggregated by second."""
        if not self._hardpoint_csv_path.exists():
            return

        with self._hardpoint_csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return

        second_rows = self._aggregate_rows_by_second(rows)
        blue_raw = [self._parse_optional_int(row.get("blue_score_raw", "")) for row in second_rows]
        red_raw = [self._parse_optional_int(row.get("red_score_raw", "")) for row in second_rows]
        hill_raw = [self._parse_optional_int(row.get("hill_number_raw", "")) for row in second_rows]

        blue_clean = self._clean_score_series(blue_raw)
        red_clean = self._clean_score_series(red_raw)
        hill_clean = self._clean_mode_series(hill_raw, window_radius=2)

        with self._hardpoint_analytics_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timestamp_seconds",
                    "blue_score_raw",
                    "blue_score_cleaned",
                    "red_score_raw",
                    "red_score_cleaned",
                    "hill_number_raw",
                    "hill_number_cleaned",
                ]
            )
            for index, row in enumerate(second_rows):
                writer.writerow(
                    [
                        row.get("timestamp_seconds", ""),
                        "" if blue_raw[index] is None else blue_raw[index],
                        "" if blue_clean[index] is None else blue_clean[index],
                        "" if red_raw[index] is None else red_raw[index],
                        "" if red_clean[index] is None else red_clean[index],
                        "" if hill_raw[index] is None else hill_raw[index],
                        "" if hill_clean[index] is None else hill_clean[index],
                    ]
                )

    def _aggregate_rows_by_second(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """Collapse raw per-frame rows into one representative row per second."""
        buckets: list[dict[str, str]] = []
        current_second: int | None = None
        current_rows: list[dict[str, str]] = []

        for row in rows:
            timestamp = row.get("timestamp_seconds", "")
            if not timestamp:
                continue
            second = int(float(timestamp))
            if current_second is None:
                current_second = second
            if second != current_second:
                aggregated = self._aggregate_second_bucket(current_second, current_rows)
                if aggregated is not None:
                    buckets.append(aggregated)
                current_rows = [row]
                current_second = second
            else:
                current_rows.append(row)

        if current_rows and current_second is not None:
            aggregated = self._aggregate_second_bucket(current_second, current_rows)
            if aggregated is not None:
                buckets.append(aggregated)
        return buckets

    def _aggregate_second_bucket(self, second: int, rows: list[dict[str, str]]) -> dict[str, str] | None:
        """Choose one representative raw value per field for a second bucket."""
        if not rows:
            return None
        blue_value = self._pick_bucket_value([self._parse_optional_int(row.get("blue_score", "")) for row in rows])
        red_value = self._pick_bucket_value([self._parse_optional_int(row.get("red_score", "")) for row in rows])
        hill_value = self._pick_bucket_value([self._parse_optional_int(row.get("hill_number", "")) for row in rows])
        return {
            "timestamp_seconds": str(second),
            "blue_score_raw": "" if blue_value is None else str(blue_value),
            "red_score_raw": "" if red_value is None else str(red_value),
            "hill_number_raw": "" if hill_value is None else str(hill_value),
        }

    @staticmethod
    def _pick_bucket_value(values: list[int | None]) -> int | None:
        """Pick the strongest representative value inside one second bucket."""
        usable = [value for value in values if value is not None]
        if not usable:
            return None
        counts = Counter(usable)
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    @staticmethod
    def _parse_optional_int(value: str | None) -> int | None:
        """Parse one optional integer CSV cell."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(text)

    def _clean_score_series(self, values: list[int | None]) -> list[int | None]:
        """Use short-run correction plus local cluster consensus to smooth score output."""
        corrected = self._replace_short_runs(values, max_run_length=4, next_run_min_length=6, jump_tolerance=20)
        clustered = self._cluster_consensus_series(corrected, window_radius=4, cluster_tolerance=2, min_support=2)
        corrected_highs = self._remove_unsupported_high_runs(
            clustered,
            support_points=3,
            max_step_size=6,
            max_drop_to_allow=25,
        )
        return self._enforce_monotonic_increase(corrected_highs)

    def _replace_short_runs(
        self,
        values: list[int | None],
        max_run_length: int,
        next_run_min_length: int,
        jump_tolerance: int,
    ) -> list[int | None]:
        """Replace short suspicious runs when later evidence strongly supports a nearby value."""
        if not values:
            return []
        result = list(values)
        runs: list[tuple[int, int, int | None]] = []
        run_start = 0
        current = values[0]
        for index in range(1, len(values)):
            if values[index] != current:
                runs.append((run_start, index - 1, current))
                run_start = index
                current = values[index]
        runs.append((run_start, len(values) - 1, current))

        for run_index, (start, end, value) in enumerate(runs):
            if value is None:
                continue
            run_length = end - start + 1
            if run_length > max_run_length or run_index + 1 >= len(runs):
                continue
            next_start, next_end, next_value = runs[run_index + 1]
            if next_value is None:
                continue
            next_length = next_end - next_start + 1
            if next_length < next_run_min_length:
                continue
            if abs(int(value) - int(next_value)) > jump_tolerance:
                continue
            for replace_index in range(start, end + 1):
                result[replace_index] = next_value
        return result

    def _cluster_consensus_series(
        self,
        values: list[int | None],
        window_radius: int,
        cluster_tolerance: int,
        min_support: int,
    ) -> list[int | None]:
        """Choose a local value cluster around each row to build a more linear score path."""
        cleaned: list[int | None] = []
        for index, current in enumerate(values):
            if current is None:
                cleaned.append(cleaned[-1] if cleaned else None)
                continue
            start = max(0, index - window_radius)
            end = min(len(values), index + window_radius + 1)
            neighborhood = [value for value in values[start:end] if value is not None]
            if not neighborhood:
                cleaned.append(current)
                continue
            clusters: list[list[int]] = []
            for value in sorted(int(v) for v in neighborhood):
                if not clusters or abs(value - clusters[-1][-1]) > cluster_tolerance:
                    clusters.append([value])
                else:
                    clusters[-1].append(value)
            best_cluster = max(clusters, key=lambda cluster: (len(cluster), -abs(int(round(sum(cluster) / len(cluster))) - int(current))))
            if len(best_cluster) < min_support:
                cleaned.append(current)
                continue
            cleaned.append(int(round(sum(best_cluster) / len(best_cluster))))
        return cleaned

    def _clean_mode_series(self, values: list[int | None], window_radius: int) -> list[int | None]:
        """Apply a simple local mode smoother to the hill series."""
        cleaned: list[int | None] = []
        for index, current in enumerate(values):
            start = max(0, index - window_radius)
            end = min(len(values), index + window_radius + 1)
            neighborhood = [value for value in values[start:end] if value is not None]
            if not neighborhood:
                cleaned.append(current)
                continue
            counts = Counter(neighborhood)
            best_value = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
            cleaned.append(best_value)
        return cleaned

    def _remove_unsupported_high_runs(
        self,
        values: list[int | None],
        support_points: int,
        max_step_size: int,
        max_drop_to_allow: int,
    ) -> list[int | None]:
        """Rewrite an earlier high run when later consistent lower reads form a believable upward path.

        Example:
        17,17,17,7,8,9 -> 7,7,7,7,8,9
        """
        if not values:
            return []

        result = list(values)
        runs: list[tuple[int, int, int | None]] = []
        run_start = 0
        current = result[0]
        for index in range(1, len(result)):
            if result[index] != current:
                runs.append((run_start, index - 1, current))
                run_start = index
                current = result[index]
        runs.append((run_start, len(result) - 1, current))

        for run_index, (start, end, value) in enumerate(runs):
            if value is None or run_index + support_points >= len(runs):
                continue
            next_runs = runs[run_index + 1 : run_index + 1 + support_points]
            next_values = [next_value for _, _, next_value in next_runs]
            if any(next_value is None for next_value in next_values):
                continue
            first_lower = int(next_values[0])
            if first_lower >= int(value):
                continue
            if (int(value) - first_lower) > max_drop_to_allow:
                continue
            if not self._is_supported_increase_sequence(
                [int(v) for v in next_values],
                max_step_size=max_step_size,
            ):
                continue
            for replace_index in range(start, end + 1):
                result[replace_index] = first_lower

        return result

    @staticmethod
    def _is_supported_increase_sequence(values: list[int], max_step_size: int) -> bool:
        """Return True when values form a believable non-decreasing progression."""
        if not values:
            return False
        for left, right in zip(values, values[1:]):
            if right < left:
                return False
            if (right - left) > max_step_size:
                return False
        return True

    @staticmethod
    def _enforce_monotonic_increase(values: list[int | None]) -> list[int | None]:
        """Force the cleaned score path to be non-decreasing over time."""
        cleaned: list[int | None] = []
        last_value: int | None = None
        for value in values:
            if value is None:
                cleaned.append(last_value)
                continue
            current = int(value)
            if last_value is None:
                last_value = current
            elif current < last_value:
                current = last_value
            else:
                last_value = current
            cleaned.append(current)
            last_value = current
        return cleaned
