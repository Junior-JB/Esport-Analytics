"""Killfeed name-detection module."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

import cv2
import numpy as np

from app.models import FramePacket, KillfeedCountSnapshot, MatchContext, ModuleStatus
from app.modules.base import PipelineModule
from app.scheduler import FixedRateScheduler


@dataclass(slots=True)
class PixelBounds:
    """Absolute integer bounds for one killfeed region."""

    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class KillfeedResolvedName:
    """One resolved registry-backed name found inside a killfeed row."""

    player_id: int
    name: str
    team: str
    confidence: float
    box: tuple[int, int, int, int]
    raw_text: str = ""


@dataclass(slots=True)
class KillfeedRowState:
    """Current row-level killfeed detection state."""

    row_index: int
    center_y: float = 0.0
    active: bool = False
    changed: bool = False
    change_score: float = 0.0
    readability_score: float = 0.0
    raw_texts: list[str] = field(default_factory=list)
    resolved_names: list[KillfeedResolvedName] = field(default_factory=list)


@dataclass(slots=True)
class KillfeedEventThread:
    """Short-lived memory for one on-screen killfeed event across adjacent frames."""

    center_y: float
    last_seen: float
    left_name: KillfeedResolvedName | None = None
    right_name: KillfeedResolvedName | None = None
    counted: bool = False


class KillfeedModule(PipelineModule):
    """Registry-only killfeed name detector without kill/death semantics yet."""

    def __init__(self, config: dict, scheduling_config: dict) -> None:
        super().__init__(name="killfeed")
        self.config = dict(config)
        self.enabled = bool(config.get("enabled", True))
        self.requires_registration = bool(config.get("requires_registration", True))
        self.scheduler = FixedRateScheduler(scheduling_config.get("samples_per_second", 3.0))
        self.snapshot = KillfeedCountSnapshot()
        self.absolute_regions = dict(self.config.get("absolute_regions", {}))
        self.row_count = int(self.config.get("row_count", 4))
        self.row_gap_ratio = float(self.config.get("row_gap_ratio", 0.02))
        self.row_change_threshold = float(self.config.get("row_change_threshold", 0.06))
        self.row_min_readability = float(self.config.get("row_min_readability", 0.10))
        self.text_min_confidence = float(self.config.get("text_min_confidence", 0.20))
        self.registry_match_threshold = float(self.config.get("registry_match_threshold", 0.60))
        self.name_min_length = int(self.config.get("name_min_length", 3))
        self.max_names_per_row = int(self.config.get("max_names_per_row", 2))
        self.event_name_min_confidence = float(self.config.get("event_name_min_confidence", 0.55))
        self.victim_cooldown_seconds = float(self.config.get("victim_cooldown_seconds", 3.0))
        self.thread_match_seconds = float(self.config.get("thread_match_seconds", 1.5))
        self.thread_forget_seconds = float(
            self.config.get("thread_forget_seconds", self.victim_cooldown_seconds)
        )
        self.thread_vertical_tolerance = float(self.config.get("thread_vertical_tolerance", 45.0))
        self._reader = None
        self._previous_binary: np.ndarray | None = None
        self._row_states: list[KillfeedRowState] = [KillfeedRowState(row_index=index) for index in range(self.row_count)]
        self._last_confidence: float | None = None
        self._victim_last_seen: dict[int, float] = {}
        self._event_threads: list[KillfeedEventThread] = []
        self.status = ModuleStatus.WAITING if self.enabled else ModuleStatus.DISABLED

    def process(self, frame_packet: FramePacket, match_context: MatchContext) -> None:
        """Detect active killfeed rows and resolve names only against the registry."""
        if not self.enabled:
            self.status = ModuleStatus.DISABLED
            return
        if self.requires_registration and (
            not match_context.registration_complete
            or not match_context.registry_locked
            or len(match_context.players) != 8
        ):
            self.status = ModuleStatus.WAITING
            return
        if not self.scheduler.should_run(frame_packet.timestamp_seconds):
            return
        if frame_packet.image is None:
            return

        self.status = ModuleStatus.ACTIVE
        killfeed_crop = self._crop_region(frame_packet.image, "killfeed_roi")
        if killfeed_crop is None or killfeed_crop.size == 0:
            return

        registry_players = {player.name: player for player in match_context.players}
        grayscale = cv2.cvtColor(killfeed_crop, cv2.COLOR_BGR2GRAY)
        current_binary = self._prepare_row(grayscale)
        roi_change_score = self._score_row_change(current_binary, self._previous_binary)
        roi_readability = self._score_row_readability(current_binary)
        self._previous_binary = current_binary

        new_states = self._detect_roi_rows(
            killfeed_crop=killfeed_crop,
            registry_players=registry_players,
            change_score=roi_change_score,
            readability_score=roi_readability,
        )
        resolved_confidences: list[float] = []
        name_detection_count = 0
        resolved_name_detection_count = 0
        player_detection_increments: dict[str, int] = {}

        for state in new_states:
            name_detection_count += len(state.raw_texts)
            resolved_name_detection_count += len(state.resolved_names)
            resolved_confidences.extend(name.confidence for name in state.resolved_names)
            for resolved_name in state.resolved_names:
                player_detection_increments[resolved_name.name] = (
                    player_detection_increments.get(resolved_name.name, 0) + 1
                )
            self._apply_row_event(state, frame_packet.timestamp_seconds)

        self._row_states = new_states
        self._last_confidence = self._average_confidence(resolved_confidences)
        self.snapshot.name_detections += name_detection_count
        self.snapshot.resolved_name_detections += resolved_name_detection_count
        for player_name, increment in player_detection_increments.items():
            self.snapshot.player_detection_counts[player_name] = (
                self.snapshot.player_detection_counts.get(player_name, 0) + increment
            )

    @property
    def last_confidence(self) -> float | None:
        """Expose the most recent name-resolution confidence."""
        return self._last_confidence

    @property
    def row_states(self) -> list[KillfeedRowState]:
        """Expose current row states for later debugging."""
        return self._row_states

    def _ensure_reader(self):
        """Lazily create the EasyOCR reader."""
        if self._reader is not None:
            return self._reader
        import easyocr

        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return self._reader

    def _detect_roi_rows(
        self,
        killfeed_crop: np.ndarray,
        registry_players: dict[str, object],
        change_score: float,
        readability_score: float,
    ) -> list[KillfeedRowState]:
        """Run EasyOCR on the whole ROI and group detected text boxes into row-like events."""
        if not registry_players:
            return []

        prepared = self._prepare_easyocr_crop(killfeed_crop)
        reader = self._ensure_reader()
        results = reader.readtext(prepared, detail=1)

        _, resolved_candidates = self._resolve_ocr_results(
            results=results,
            registry_players=registry_players,
            x_offset=0,
        )
        grouped_candidates = self._group_candidates_into_rows(resolved_candidates)

        row_states: list[KillfeedRowState] = []
        for row_index, group in enumerate(grouped_candidates):
            row_resolved = self._select_row_names(group)
            row_states.append(
                KillfeedRowState(
                    row_index=row_index,
                    center_y=self._group_center_y(group),
                    active=True,
                    changed=True,
                    change_score=change_score,
                    readability_score=readability_score,
                    raw_texts=[candidate.raw_text for candidate in group],
                    resolved_names=row_resolved,
                )
            )
        return row_states

    def _resolve_ocr_results(
        self,
        results,
        registry_players: dict[str, object],
        x_offset: int,
    ) -> tuple[list[str], list[KillfeedResolvedName]]:
        """Resolve one OCR result set into raw texts and registry-backed candidates."""
        raw_texts: list[str] = []
        resolved_candidates: list[KillfeedResolvedName] = []

        for box, text, confidence in results:
            candidate_text = self._clean_text(str(text))
            if not candidate_text or len(self._normalize_name(candidate_text)) < self.name_min_length:
                continue
            confidence_value = max(0.0, float(confidence))
            if confidence_value < self.text_min_confidence:
                continue

            raw_texts.append(candidate_text)
            best_player, best_score = self._resolve_to_registry(candidate_text, registry_players)
            if best_player is None:
                continue
            flattened_box = self._flatten_box(box)
            resolved_candidates.append(
                KillfeedResolvedName(
                    player_id=best_player.player_id,
                    name=best_player.name,
                    team=best_player.team,
                    confidence=self._blend_name_confidence(confidence_value, best_score),
                    box=(
                        flattened_box[0] + x_offset,
                        flattened_box[1],
                        flattened_box[2],
                        flattened_box[3],
                    ),
                    raw_text=candidate_text,
                )
            )

        return raw_texts, resolved_candidates

    def _group_candidates_into_rows(
        self,
        candidates: list[KillfeedResolvedName],
    ) -> list[list[KillfeedResolvedName]]:
        """Cluster OCR detections into row-like groups by vertical position."""
        if not candidates:
            return []

        sorted_candidates = sorted(
            candidates,
            key=lambda candidate: (candidate.box[1] + (candidate.box[3] / 2.0), candidate.box[0]),
        )
        groups: list[list[KillfeedResolvedName]] = []
        for candidate in sorted_candidates:
            candidate_center_y = candidate.box[1] + (candidate.box[3] / 2.0)
            candidate_height = max(1, candidate.box[3])
            placed = False
            for group in groups:
                group_centers = [item.box[1] + (item.box[3] / 2.0) for item in group]
                group_heights = [max(1, item.box[3]) for item in group]
                average_center_y = float(sum(group_centers) / len(group_centers))
                average_height = float(sum(group_heights) / len(group_heights))
                if abs(candidate_center_y - average_center_y) <= max(candidate_height, average_height) * 0.85:
                    group.append(candidate)
                    placed = True
                    break
            if not placed:
                groups.append([candidate])
        groups.sort(key=lambda group: min(item.box[1] for item in group))
        return groups

    def _select_row_names(
        self,
        group: list[KillfeedResolvedName],
    ) -> list[KillfeedResolvedName]:
        """Select the leftmost and rightmost resolved names from one grouped OCR row."""
        if not group:
            return []
        sorted_candidates = sorted(group, key=lambda item: (item.box[0], item.box[1]))
        left_name = sorted_candidates[0]
        if len(sorted_candidates) == 1:
            return [left_name]

        right_name = max(sorted_candidates, key=lambda item: (item.box[0] + item.box[2], item.confidence))
        if left_name.player_id == right_name.player_id:
            alternatives = [item for item in sorted_candidates if item.player_id != left_name.player_id]
            if not alternatives:
                return [left_name]
            right_name = max(alternatives, key=lambda item: (item.box[0] + item.box[2], item.confidence))
        return sorted([left_name, right_name], key=lambda item: (item.box[0], item.box[1]))

    def _apply_row_event(self, state: KillfeedRowState, timestamp_seconds: float) -> None:
        """Apply kill/death telemetry when a row yields two legible sides."""
        self._prune_event_threads(timestamp_seconds)
        thread = self._match_or_create_thread(state, timestamp_seconds)
        if thread is None:
            return
        if thread.counted:
            return
        left_name = thread.left_name
        right_name = thread.right_name
        if left_name is None or right_name is None:
            return
        if left_name.player_id == right_name.player_id:
            return
        if left_name.confidence < self.event_name_min_confidence or right_name.confidence < self.event_name_min_confidence:
            return

        victim_last_seen = self._victim_last_seen.get(right_name.player_id)
        if victim_last_seen is not None and (timestamp_seconds - victim_last_seen) < self.victim_cooldown_seconds:
            return

        self._victim_last_seen[right_name.player_id] = timestamp_seconds
        thread.counted = True

        self.snapshot.kills += 1
        self.snapshot.deaths += 1
        self.snapshot.player_kills[left_name.name] = self.snapshot.player_kills.get(left_name.name, 0) + 1
        self.snapshot.player_deaths[right_name.name] = self.snapshot.player_deaths.get(right_name.name, 0) + 1

    @staticmethod
    def _group_center_y(group: list[KillfeedResolvedName]) -> float:
        """Return the average vertical center for one grouped OCR row."""
        if not group:
            return 0.0
        centers = [item.box[1] + (item.box[3] / 2.0) for item in group]
        return float(sum(centers) / len(centers))

    def _prune_event_threads(self, timestamp_seconds: float) -> None:
        """Forget stale killfeed event threads after they have left the screen."""
        self._event_threads = [
            thread
            for thread in self._event_threads
            if (timestamp_seconds - thread.last_seen) <= self.thread_forget_seconds
        ]

    def _match_or_create_thread(
        self,
        state: KillfeedRowState,
        timestamp_seconds: float,
    ) -> KillfeedEventThread | None:
        """Merge a row state into an existing vertical-band thread or create a new one."""
        left_name = state.resolved_names[0] if state.resolved_names else None
        right_name = state.resolved_names[-1] if len(state.resolved_names) >= 2 else None
        if left_name is None and right_name is None:
            return None

        best_thread: KillfeedEventThread | None = None
        best_distance: float | None = None
        for thread in self._event_threads:
            if (timestamp_seconds - thread.last_seen) > self.thread_match_seconds:
                continue
            vertical_distance = abs(thread.center_y - state.center_y)
            if vertical_distance > self.thread_vertical_tolerance:
                continue
            if not self._thread_names_compatible(thread, left_name, right_name):
                continue
            if best_distance is None or vertical_distance < best_distance:
                best_thread = thread
                best_distance = vertical_distance

        if best_thread is None:
            best_thread = KillfeedEventThread(center_y=state.center_y, last_seen=timestamp_seconds)
            self._event_threads.append(best_thread)

        best_thread.last_seen = timestamp_seconds
        best_thread.center_y = state.center_y
        best_thread.left_name = self._prefer_name(best_thread.left_name, left_name)
        best_thread.right_name = self._prefer_name(best_thread.right_name, right_name)
        return best_thread

    @staticmethod
    def _thread_names_compatible(
        thread: KillfeedEventThread,
        left_name: KillfeedResolvedName | None,
        right_name: KillfeedResolvedName | None,
    ) -> bool:
        """Allow partial updates as long as they do not contradict an existing thread."""
        if left_name is not None and thread.left_name is not None and left_name.player_id != thread.left_name.player_id:
            return False
        if right_name is not None and thread.right_name is not None and right_name.player_id != thread.right_name.player_id:
            return False
        return True

    @staticmethod
    def _prefer_name(
        current: KillfeedResolvedName | None,
        incoming: KillfeedResolvedName | None,
    ) -> KillfeedResolvedName | None:
        """Keep the stronger resolved name when a thread sees repeated frames."""
        if incoming is None:
            return current
        if current is None or incoming.confidence >= current.confidence:
            return incoming
        return current

    def _resolve_to_registry(self, candidate_text: str, registry_players: dict[str, object]):
        """Return the closest registered player for one OCR candidate."""
        normalized_candidate = self._normalize_name(candidate_text)
        best_player = None
        best_score = 0.0
        for player in registry_players.values():
            score = self._name_similarity(normalized_candidate, self._normalize_name(player.name))
            if score > best_score:
                best_player = player
                best_score = score
        return best_player, best_score

    def _crop_region(self, frame: np.ndarray, region_name: str) -> np.ndarray | None:
        """Crop one configured killfeed region."""
        if region_name not in self.absolute_regions:
            return None
        bounds = self._resolve_bounds(frame.shape, self.absolute_regions[region_name])
        return frame[bounds.y : bounds.y + bounds.height, bounds.x : bounds.x + bounds.width]

    @staticmethod
    def _resolve_bounds(frame_shape: tuple[int, ...], relative_region: dict) -> PixelBounds:
        """Convert a normalized region into absolute pixel bounds."""
        frame_height, frame_width = frame_shape[:2]
        x = int(relative_region.get("x", 0.0) * frame_width)
        y = int(relative_region.get("y", 0.0) * frame_height)
        width = int(relative_region.get("width", 0.0) * frame_width)
        height = int(relative_region.get("height", 0.0) * frame_height)
        return PixelBounds(x=x, y=y, width=width, height=height)

    @staticmethod
    def _prepare_row(grayscale: np.ndarray) -> np.ndarray:
        """Prepare a row for activity/change detection."""
        blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(blurred, 50, 150)
        combined = cv2.bitwise_or(binary, edges)
        return combined

    @staticmethod
    def _prepare_easyocr_crop(row_crop: np.ndarray) -> np.ndarray:
        """Prepare a row for EasyOCR text-box detection."""
        grayscale = cv2.cvtColor(row_crop, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(grayscale, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary == 255) > 0.60:
            binary = cv2.bitwise_not(binary)
        return binary

    @staticmethod
    def _score_row_change(current_binary: np.ndarray, previous_binary: np.ndarray | None) -> float:
        """Return normalized change score between row states."""
        if previous_binary is None or previous_binary.shape != current_binary.shape:
            return 1.0
        difference = cv2.absdiff(current_binary, previous_binary)
        return float(np.count_nonzero(difference)) / float(difference.size)

    @staticmethod
    def _score_row_readability(binary: np.ndarray) -> float:
        """Estimate whether a row likely contains readable text/activity."""
        if binary.size <= 0:
            return 0.0
        foreground = (binary > 0).astype(np.uint8)
        fill_ratio = float(np.count_nonzero(foreground)) / float(foreground.size)
        if fill_ratio <= 0.0:
            return 0.0
        active_columns = float(np.count_nonzero(foreground.sum(axis=0) > 0)) / float(foreground.shape[1])
        active_rows = float(np.count_nonzero(foreground.sum(axis=1) > 0)) / float(foreground.shape[0])
        return min(1.0, (0.45 * min(1.0, fill_ratio / 0.20)) + (0.35 * active_columns) + (0.20 * active_rows))

    @staticmethod
    def _flatten_box(box) -> tuple[int, int, int, int]:
        """Convert an EasyOCR quadrilateral into a simple bounding box."""
        x_values = [int(point[0]) for point in box]
        y_values = [int(point[1]) for point in box]
        x0 = min(x_values)
        y0 = min(y_values)
        x1 = max(x_values)
        y1 = max(y_values)
        return x0, y0, x1 - x0, y1 - y0

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean OCR text while keeping gamer-tag friendly characters."""
        allowed = []
        for character in text.strip():
            if character.isalnum() or character in {"_", "-", "[", "]"}:
                allowed.append(character)
        return "".join(allowed)

    @staticmethod
    def _normalize_name(text: str) -> str:
        """Normalize an OCR/player name for matching."""
        return "".join(character for character in text.lower() if character.isalnum())

    @staticmethod
    def _name_similarity(left: str, right: str) -> float:
        """Return similarity between two normalized names."""
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        if left in right or right in left:
            return float(min(len(left), len(right)) / max(len(left), len(right)))
        return float(SequenceMatcher(a=left, b=right).ratio())

    @staticmethod
    def _blend_name_confidence(ocr_confidence: float, registry_score: float) -> float:
        """Blend OCR confidence and registry similarity into one resolution score."""
        return min(1.0, (0.35 * ocr_confidence) + (0.65 * registry_score))

    @staticmethod
    def _average_confidence(values: list[float]) -> float | None:
        """Average a non-empty list of confidence values."""
        usable = [value for value in values if value > 0.0]
        if not usable:
            return None
        return float(sum(usable) / len(usable))
