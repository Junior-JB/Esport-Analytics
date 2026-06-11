"""Ordered validation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import csv
import json
import re
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
        self._match_summary_json_path = self._output_root / "match_summary.json"
        self._match_summary_csv_path = self._output_root / "match_summary.csv"
        self._final_score_csv_path = self._output_root / "final_score.csv"
        self._telemetry_csv_path = self._output_root / "telemetry_timeseries.csv"
        self._last_written_telemetry_frame_index: int | None = None
        self._last_tangible_blue_score: int | None = None
        self._last_tangible_red_score: int | None = None
        self._telemetry_rows: list[dict[str, object]] = []
        self._telemetry_merge_base_rows: list[dict[str, object]] | None = None

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
        final_telemetry_rows = self._write_telemetry_csv()
        self._write_final_score_csv(final_telemetry_rows)
        self._write_match_summary_json()
        self._write_match_summary_csv()
        self._write_final_summary()
        return self._final_txt_path

    @property
    def run_output_dir(self) -> Path:
        """Expose the current run output directory."""
        return self._run_output_dir

    @property
    def telemetry_csv_path(self) -> Path:
        """Expose the dashboard-oriented telemetry CSV path."""
        return self._telemetry_csv_path

    @property
    def telemetry_rows(self) -> list[dict[str, object]]:
        """Expose collected telemetry rows for cross-pass export merging."""
        return [self._copy_telemetry_row(row) for row in self._telemetry_rows]

    def set_telemetry_merge_base_rows(self, rows: list[dict[str, object]]) -> None:
        """Inject first-pass telemetry rows for final merged CSV generation."""
        self._telemetry_merge_base_rows = [self._copy_telemetry_row(row) for row in rows]

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
            or self.bundle.killfeed.last_confidence
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
        self._match_summary_json_path = self._output_root / "match_summary.json"
        self._match_summary_csv_path = self._output_root / "match_summary.csv"
        self._final_score_csv_path = self._output_root / "final_score.csv"
        self._telemetry_csv_path = self._output_root / "telemetry_timeseries.csv"
        self._last_written_telemetry_frame_index = None
        self._last_tangible_blue_score = None
        self._last_tangible_red_score = None
        self._telemetry_rows = []
        self._telemetry_merge_base_rows = None
        if hasattr(self.bundle.scoreboard, "set_debug_output_dir"):
            self.bundle.scoreboard.set_debug_output_dir(self._run_output_dir)
        self._initialize_telemetry_csv()

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
        self._write_telemetry_csv_row(state)

    def _initialize_telemetry_csv(self) -> None:
        """Prepare the dashboard telemetry CSV path for the new run."""
        self._telemetry_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._telemetry_csv_path.write_text("", encoding="utf-8")

    def _write_telemetry_csv_row(self, state) -> None:
        """Collect one dashboard telemetry row per processed frame."""
        if state.timestamp_seconds is None or state.frame_index is None:
            return
        if self._last_written_telemetry_frame_index == state.frame_index:
            return

        if state.hardpoint.blue_score is not None:
            self._last_tangible_blue_score = state.hardpoint.blue_score
        if state.hardpoint.red_score is not None:
            self._last_tangible_red_score = state.hardpoint.red_score

        self._telemetry_rows.append(
            {
                "timestamp": f"{state.timestamp_seconds:.3f}",
                "timestamp_seconds": float(state.timestamp_seconds),
                "frame_index": state.frame_index,
                "blue_score": self._last_tangible_blue_score,
                "red_score": self._last_tangible_red_score,
                "hill_number": state.hardpoint.current_hill,
                "nades_used_to_that_point": int(state.utility_counts.grenade),
                "stuns_used_to_that_point": int(state.utility_counts.stun),
                "specialties_used_to_that_point": int(state.utility_counts.specialty_equipment),
                "player_kills": {
                    str(player_name): int(kill_count)
                    for player_name, kill_count in state.killfeed_counts.player_kills.items()
                },
            }
        )

        self._last_written_telemetry_frame_index = state.frame_index

    def _write_telemetry_csv(self) -> list[dict[str, object]]:
        """Write the final dashboard telemetry CSV, including cumulative player kill columns."""
        state = self.dashboard.snapshot()
        registered_names = list(state.registered_players)
        merged_rows = self._build_final_telemetry_rows(registered_names)
        second_rows = self._build_final_telemetry_second_rows(
            merged_rows,
            registered_names,
            final_score_blue=state.scoreboard.final_score_blue,
            final_score_red=state.scoreboard.final_score_red,
            bucket_size_seconds=10,
        )
        player_columns = self._player_kill_column_map(registered_names)

        with self._telemetry_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "timestamp",
                    "blue_score",
                    "red_score",
                    "hill_number",
                    "nades_used_to_that_point",
                    "stuns_used_to_that_point",
                    "specialties_used_to_that_point",
                    *player_columns.values(),
                ]
            )
            for row in second_rows:
                writer.writerow(
                    [
                        row.get("second", ""),
                        "" if row.get("blue_score") is None else row.get("blue_score"),
                        "" if row.get("red_score") is None else row.get("red_score"),
                        "" if row.get("hill_number") is None else row.get("hill_number"),
                        int(row.get("nades_used_to_that_point", 0)),
                        int(row.get("stuns_used_to_that_point", 0)),
                        int(row.get("specialties_used_to_that_point", 0)),
                        *[
                            row["player_kills"].get(player_name, 0)
                            for player_name in player_columns
                        ],
                    ]
                )
        return second_rows

    def _build_final_telemetry_rows(self, registered_names: list[str]) -> list[dict[str, object]]:
        """Merge pass-one score/utility telemetry with pass-two cumulative kill telemetry."""
        base_rows = self._telemetry_merge_base_rows if self._telemetry_merge_base_rows is not None else self._telemetry_rows
        base_rows_sorted = sorted(
            [self._copy_telemetry_row(row) for row in base_rows],
            key=lambda row: float(row.get("timestamp_seconds", 0.0)),
        )
        kill_rows_sorted = sorted(
            [self._copy_telemetry_row(row) for row in self._telemetry_rows],
            key=lambda row: float(row.get("timestamp_seconds", 0.0)),
        )

        merged_rows: list[dict[str, object]] = []
        latest_kills = {player_name: 0 for player_name in registered_names}
        kill_index = 0
        for base_row in base_rows_sorted:
            timestamp_seconds = float(base_row.get("timestamp_seconds", 0.0))
            while kill_index < len(kill_rows_sorted) and float(kill_rows_sorted[kill_index].get("timestamp_seconds", 0.0)) <= timestamp_seconds:
                latest_kills.update(
                    {
                        player_name: int(kill_count)
                        for player_name, kill_count in dict(kill_rows_sorted[kill_index].get("player_kills", {})).items()
                    }
                )
                kill_index += 1

            merged_row = dict(base_row)
            merged_row["player_kills"] = {player_name: int(latest_kills.get(player_name, 0)) for player_name in registered_names}
            merged_rows.append(merged_row)
        return merged_rows

    def _build_final_telemetry_second_rows(
        self,
        merged_rows: list[dict[str, object]],
        registered_names: list[str],
        final_score_blue: int | None,
        final_score_red: int | None,
        bucket_size_seconds: int,
    ) -> list[dict[str, object]]:
        """Convert merged frame rows into cleaned fixed-bucket telemetry."""
        if not merged_rows:
            return []

        origin_second = int(float(merged_rows[0].get("timestamp_seconds", 0.0)))
        buckets: dict[int, list[dict[str, object]]] = {}
        for row in merged_rows:
            absolute_seconds = float(row.get("timestamp_seconds", 0.0))
            relative_second = int(absolute_seconds) - origin_second + 1
            if relative_second < 1:
                relative_second = 1
            bucket_index = ((relative_second - 1) // bucket_size_seconds) + 1
            buckets.setdefault(bucket_index, []).append(row)

        second_rows: list[dict[str, object]] = []
        last_second_row: dict[str, object] | None = None
        for bucket_index in range(1, max(buckets) + 1):
            rows = buckets.get(bucket_index)
            bucket_timestamp = bucket_index * bucket_size_seconds
            if rows:
                blue_value = self._pick_bucket_value([self._parse_optional_int(row.get("blue_score")) for row in rows])
                red_value = self._pick_bucket_value([self._parse_optional_int(row.get("red_score")) for row in rows])
                hill_value = self._pick_bucket_value([self._parse_optional_int(row.get("hill_number")) for row in rows])
                last_row = rows[-1]
                current_row = {
                    "second": bucket_timestamp,
                    "blue_score": blue_value,
                    "red_score": red_value,
                    "hill_number": hill_value,
                    "nades_used_to_that_point": int(last_row.get("nades_used_to_that_point", 0)),
                    "stuns_used_to_that_point": int(last_row.get("stuns_used_to_that_point", 0)),
                    "specialties_used_to_that_point": int(last_row.get("specialties_used_to_that_point", 0)),
                    "player_kills": {
                        player_name: int(dict(last_row.get("player_kills", {})).get(player_name, 0))
                        for player_name in registered_names
                    },
                }
            elif last_second_row is not None:
                current_row = {
                    "second": bucket_timestamp,
                    "blue_score": last_second_row.get("blue_score"),
                    "red_score": last_second_row.get("red_score"),
                    "hill_number": last_second_row.get("hill_number"),
                    "nades_used_to_that_point": int(last_second_row.get("nades_used_to_that_point", 0)),
                    "stuns_used_to_that_point": int(last_second_row.get("stuns_used_to_that_point", 0)),
                    "specialties_used_to_that_point": int(last_second_row.get("specialties_used_to_that_point", 0)),
                    "player_kills": {
                        player_name: int(dict(last_second_row.get("player_kills", {})).get(player_name, 0))
                        for player_name in registered_names
                    },
                }
            else:
                current_row = {
                    "second": bucket_timestamp,
                    "blue_score": None,
                    "red_score": None,
                    "hill_number": None,
                    "nades_used_to_that_point": 0,
                    "stuns_used_to_that_point": 0,
                    "specialties_used_to_that_point": 0,
                    "player_kills": {player_name: 0 for player_name in registered_names},
                }
            second_rows.append(current_row)
            last_second_row = current_row

        blue_clean = self._clean_score_series([row.get("blue_score") for row in second_rows])
        red_clean = self._clean_score_series([row.get("red_score") for row in second_rows])
        hill_clean = self._clean_mode_series([row.get("hill_number") for row in second_rows], window_radius=2)

        cleaned_rows: list[dict[str, object]] = []
        for index, row in enumerate(second_rows):
            cleaned_row = dict(row)
            cleaned_row["blue_score"] = blue_clean[index]
            cleaned_row["red_score"] = red_clean[index]
            cleaned_row["hill_number"] = hill_clean[index]
            cleaned_rows.append(cleaned_row)

        return self._insert_forced_final_score_row(cleaned_rows, final_score_blue, final_score_red)

    def _copy_telemetry_row(self, row: dict[str, object]) -> dict[str, object]:
        """Clone one telemetry row, including nested cumulative kill maps."""
        cloned = dict(row)
        cloned["player_kills"] = {
            str(player_name): int(kill_count)
            for player_name, kill_count in dict(row.get("player_kills", {})).items()
        }
        return cloned

    def _player_kill_column_map(self, registered_names: list[str]) -> dict[str, str]:
        """Build stable CSV headers for cumulative player kill columns."""
        seen_headers: set[str] = set()
        column_map: dict[str, str] = {}
        for player_name in registered_names:
            candidate = self._csv_player_name(player_name)
            suffix = 2
            while candidate in seen_headers:
                candidate = f"{self._csv_player_name(player_name)}_{suffix}"
                suffix += 1
            seen_headers.add(candidate)
            column_map[player_name] = candidate
        return column_map

    @staticmethod
    def _csv_player_name(player_name: str) -> str:
        """Strip bracketed clan tags from a player name for CSV-facing exports."""
        raw_name = str(player_name)
        cleaned = re.sub(r"\[[^\]]*\]", "", raw_name)
        cleaned = cleaned.replace("[", "").replace("]", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        fallback = raw_name.replace("[", "").replace("]", "").strip()
        return cleaned or fallback or raw_name.strip()

    def _insert_forced_final_score_row(
        self,
        rows: list[dict[str, object]],
        final_score_blue: int | None,
        final_score_red: int | None,
    ) -> list[dict[str, object]]:
        """Insert a final scoreboard-derived score row when a 250 end-state is available."""
        if not rows:
            return rows
        if final_score_blue is None or final_score_red is None:
            return rows
        if 250 not in {final_score_blue, final_score_red}:
            return rows

        winner_key = "blue_score" if final_score_blue == 250 else "red_score"
        winner_final_score = final_score_blue if winner_key == "blue_score" else final_score_red
        losing_final_score = final_score_red if winner_key == "blue_score" else final_score_blue

        if rows[-1].get("blue_score") == final_score_blue and rows[-1].get("red_score") == final_score_red:
            return rows

        prior_winner_scores = [
            int(score)
            for score in (row.get(winner_key) for row in rows)
            if score is not None and int(score) < winner_final_score
        ]
        if prior_winner_scores:
            anchor_score = max(prior_winner_scores)
            insert_after_index = max(
                index
                for index, row in enumerate(rows)
                if row.get(winner_key) is not None and int(row.get(winner_key)) == anchor_score
            )
        else:
            insert_after_index = len(rows) - 1

        anchor_row = dict(rows[insert_after_index])
        forced_row = dict(anchor_row)
        forced_row["blue_score"] = int(final_score_blue)
        forced_row["red_score"] = int(final_score_red)
        rows = list(rows)
        rows.insert(insert_after_index + 1, forced_row)

        for index, row in enumerate(rows, start=1):
            row["second"] = index * 10
            if winner_key == "blue_score" and row.get("blue_score") == winner_final_score:
                row["red_score"] = int(losing_final_score)
            elif winner_key == "red_score" and row.get("red_score") == winner_final_score:
                row["blue_score"] = int(losing_final_score)
        return rows

    @staticmethod
    def _pick_bucket_value(values: list[int | None]) -> int | None:
        """Pick the strongest representative score/hill value inside one second bucket."""
        usable = [value for value in values if value is not None]
        if not usable:
            return None
        counts = Counter(usable)
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    @staticmethod
    def _parse_optional_int(value: object) -> int | None:
        """Parse one optional integer-like value."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))

    def _clean_score_series(self, values: list[object]) -> list[int | None]:
        """Smooth score series while allowing later consistent lower sequences to rewrite early spikes."""
        parsed_values = [self._parse_optional_int(value) for value in values]
        corrected = self._replace_short_runs(parsed_values, max_run_length=4, next_run_min_length=6, jump_tolerance=20)
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
            _, next_end, next_value = runs[run_index + 1]
            if next_value is None:
                continue
            next_start = runs[run_index + 1][0]
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
        """Choose a local value cluster around each second to build a more linear score path."""
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
            best_cluster = max(
                clusters,
                key=lambda cluster: (
                    len(cluster),
                    -abs(int(round(sum(cluster) / len(cluster))) - int(current)),
                ),
            )
            if len(best_cluster) < min_support:
                cleaned.append(current)
                continue
            cleaned.append(int(round(sum(best_cluster) / len(best_cluster))))
        return cleaned

    def _remove_unsupported_high_runs(
        self,
        values: list[int | None],
        support_points: int,
        max_step_size: int,
        max_drop_to_allow: int,
    ) -> list[int | None]:
        """Rewrite an earlier high run when later lower reads form a believable upward path."""
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
            if not self._is_supported_increase_sequence([int(v) for v in next_values], max_step_size=max_step_size):
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

    def _clean_mode_series(self, values: list[object], window_radius: int) -> list[int | None]:
        """Apply a simple local mode smoother to the hill series."""
        parsed_values = [self._parse_optional_int(value) for value in values]
        cleaned: list[int | None] = []
        for index, current in enumerate(parsed_values):
            start = max(0, index - window_radius)
            end = min(len(parsed_values), index + window_radius + 1)
            neighborhood = [value for value in parsed_values[start:end] if value is not None]
            if not neighborhood:
                cleaned.append(current)
                continue
            counts = Counter(neighborhood)
            best_value = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
            cleaned.append(best_value)
        return cleaned

    def _write_match_summary_json(self) -> None:
        """Write the compact dashboard-focused final match summary JSON."""
        state = self.dashboard.snapshot()
        registered_names = list(state.registered_players)
        kills_by_player = {name: int(state.killfeed_counts.player_kills.get(name, 0)) for name in registered_names}
        deaths_by_player = {name: int(state.killfeed_counts.player_deaths.get(name, 0)) for name in registered_names}

        player_kds: dict[str, float] = {}
        for name in registered_names:
            kills = kills_by_player[name]
            deaths = deaths_by_player[name]
            kd_value = float(kills) if deaths == 0 else float(kills / deaths)
            player_kds[name] = round(kd_value, 2)

        final_blue = state.scoreboard.final_score_blue
        final_red = state.scoreboard.final_score_red
        if final_blue is None:
            final_blue = state.hardpoint.blue_score
        if final_red is None:
            final_red = state.hardpoint.red_score

        payload = {
            "final_score": {
                "blue": final_blue,
                "red": final_red,
            },
            "player_registry": state.scoreboard.player_registry,
            "player_kds": player_kds,
            "total_kills_per_player": kills_by_player,
            "total_deaths_per_player": deaths_by_player,
            "total_nades_used": int(state.utility_counts.grenade),
            "total_stuns_used": int(state.utility_counts.stun),
            "total_specialties_used": int(state.utility_counts.specialty_equipment),
        }
        self._match_summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_final_score_csv(self, telemetry_rows: list[dict[str, object]]) -> None:
        """Write a small standalone final-score CSV for dashboard consumers."""
        state = self.dashboard.snapshot()
        final_blue, final_red = self._resolve_final_score_for_export(state, telemetry_rows)

        with self._final_score_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["blue_final_score", "red_final_score"])
            writer.writerow(
                [
                    "" if final_blue is None else int(final_blue),
                    "" if final_red is None else int(final_red),
                ]
            )

    def _resolve_final_score_for_export(
        self,
        state,
        telemetry_rows: list[dict[str, object]],
    ) -> tuple[int | None, int | None]:
        """Prefer scoreboard 250 detection, otherwise fall back to the final telemetry score."""
        scoreboard_blue = state.scoreboard.final_score_blue
        scoreboard_red = state.scoreboard.final_score_red
        if scoreboard_blue is not None and scoreboard_red is not None and 250 in {int(scoreboard_blue), int(scoreboard_red)}:
            return int(scoreboard_blue), int(scoreboard_red)

        for row in reversed(telemetry_rows):
            blue_score = self._parse_optional_int(row.get("blue_score"))
            red_score = self._parse_optional_int(row.get("red_score"))
            if blue_score is not None or red_score is not None:
                return blue_score, red_score

        final_blue = state.hardpoint.blue_score
        final_red = state.hardpoint.red_score
        return (
            None if final_blue is None else int(final_blue),
            None if final_red is None else int(final_red),
        )

    def _write_match_summary_csv(self) -> None:
        """Write a flat dashboard-oriented match summary CSV with players as columns."""
        state = self.dashboard.snapshot()
        registered_players = sorted(self.match_context.players, key=lambda player: player.player_id)
        player_names = [player.name for player in registered_players]
        csv_player_names = [self._csv_player_name(player_name) for player_name in player_names]
        teams_by_player = {
            player.name: ("ally" if player.team == "friendly" else "enemy")
            for player in registered_players
        }
        kills_by_player = {
            player_name: int(state.killfeed_counts.player_kills.get(player_name, 0))
            for player_name in player_names
        }
        deaths_by_player = {
            player_name: int(state.killfeed_counts.player_deaths.get(player_name, 0))
            for player_name in player_names
        }
        kd_by_player: dict[str, float] = {}
        for player_name in player_names:
            kills = kills_by_player[player_name]
            deaths = deaths_by_player[player_name]
            kd_by_player[player_name] = round(float(kills) if deaths == 0 else float(kills / deaths), 2)

        with self._match_summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", *csv_player_names, "all_match"])
            writer.writerow(
                [
                    "team",
                    *[teams_by_player.get(player_name, "") for player_name in player_names],
                    "",
                ]
            )
            writer.writerow(
                [
                    "kd",
                    *[kd_by_player.get(player_name, 0.0) for player_name in player_names],
                    "",
                ]
            )
            writer.writerow(
                [
                    "total_kills",
                    *[kills_by_player.get(player_name, 0) for player_name in player_names],
                    sum(kills_by_player.values()),
                ]
            )
            writer.writerow(
                [
                    "total_deaths",
                    *[deaths_by_player.get(player_name, 0) for player_name in player_names],
                    sum(deaths_by_player.values()),
                ]
            )
            writer.writerow(
                [
                    "total_nades_used",
                    *["" for _ in player_names],
                    int(state.utility_counts.grenade),
                ]
            )
            writer.writerow(
                [
                    "total_stuns_used",
                    *["" for _ in player_names],
                    int(state.utility_counts.stun),
                ]
            )
            writer.writerow(
                [
                    "total_specialties_used",
                    *["" for _ in player_names],
                    int(state.utility_counts.specialty_equipment),
                ]
            )

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
                f"Name detections: {state.killfeed_counts.name_detections}",
                f"Resolved name detections: {state.killfeed_counts.resolved_name_detections}",
                f"Row groups detected: {state.killfeed_counts.row_groups_detected}",
                f"Single-side row groups: {state.killfeed_counts.single_side_row_groups}",
                f"Event threads created: {state.killfeed_counts.event_threads_created}",
                f"Event threads reused: {state.killfeed_counts.event_threads_reused}",
                f"Events counted: {state.killfeed_counts.events_counted}",
                f"Events blocked by victim cooldown: {state.killfeed_counts.events_blocked_victim_cooldown}",
                f"Events blocked by low confidence: {state.killfeed_counts.events_blocked_low_confidence}",
                f"Events blocked by same-side pairing: {state.killfeed_counts.events_blocked_same_side}",
                "Player detection counts:",
            ]
        )
        if state.killfeed_counts.player_detection_counts:
            for player_name, count in sorted(
                state.killfeed_counts.player_detection_counts.items(),
                key=lambda item: (-item[1], item[0].lower()),
            ):
                lines.append(f"- {player_name}: {count}")
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "Player Kills:",
            ]
        )
        if state.killfeed_counts.player_kills:
            for player_name, count in sorted(
                state.killfeed_counts.player_kills.items(),
                key=lambda item: (-item[1], item[0].lower()),
            ):
                lines.append(f"- {player_name}: {count}")
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "Player Deaths:",
            ]
        )
        if state.killfeed_counts.player_deaths:
            for player_name, count in sorted(
                state.killfeed_counts.player_deaths.items(),
                key=lambda item: (-item[1], item[0].lower()),
            ):
                lines.append(f"- {player_name}: {count}")
        else:
            lines.append("- none")
        lines.extend(
            [
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
