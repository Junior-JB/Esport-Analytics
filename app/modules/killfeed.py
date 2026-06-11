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
    source_variant: str = ""
    preferred_team: str | None = None


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
        self.team_anchor_confidence = float(self.config.get("team_anchor_confidence", 0.70))
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
        friendly_registry_players = {
            player.name: player for player in match_context.players if player.team == "friendly"
        }
        enemy_registry_players = {
            player.name: player for player in match_context.players if player.team == "enemy"
        }
        grayscale = cv2.cvtColor(killfeed_crop, cv2.COLOR_BGR2GRAY)
        current_binary = self._prepare_row(grayscale)
        roi_change_score = self._score_row_change(current_binary, self._previous_binary)
        roi_readability = self._score_row_readability(current_binary)
        self._previous_binary = current_binary

        new_states = self._detect_roi_rows(
            killfeed_crop=killfeed_crop,
            registry_players=registry_players,
            friendly_registry_players=friendly_registry_players,
            enemy_registry_players=enemy_registry_players,
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
        self.snapshot.row_groups_detected += len(new_states)
        self.snapshot.single_side_row_groups += sum(1 for state in new_states if len(state.resolved_names) < 2)
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
        friendly_registry_players: dict[str, object],
        enemy_registry_players: dict[str, object],
        change_score: float,
        readability_score: float,
    ) -> list[KillfeedRowState]:
        """Run EasyOCR on the whole ROI and group detected text boxes into row-like events."""
        if not registry_players:
            return []

        reader = self._ensure_reader()
        variant_candidates_by_name: dict[str, list[KillfeedResolvedName]] = {}
        for variant_name, prepared in self._prepare_easyocr_variants(killfeed_crop).items():
            results = reader.readtext(prepared, detail=1)
            if variant_name == "red":
                match_pool = enemy_registry_players
            elif variant_name == "blue":
                match_pool = friendly_registry_players
            else:
                match_pool = registry_players
            _, variant_candidates = self._resolve_ocr_results(
                results=results,
                registry_players=match_pool,
                x_offset=0,
                source_variant=variant_name,
            )
            variant_candidates_by_name[variant_name] = self._dedupe_resolved_candidates(variant_candidates)

        resolved_candidates: list[KillfeedResolvedName] = []
        for candidates in variant_candidates_by_name.values():
            resolved_candidates.extend(candidates)
        resolved_candidates = self._dedupe_resolved_candidates(resolved_candidates)
        red_candidates = variant_candidates_by_name.get("red", [])
        reference_candidates = self._build_reference_candidates(
            red_candidates=red_candidates,
            all_candidates=resolved_candidates,
        )
        grouped_candidates = self._group_candidates_into_rows(reference_candidates)

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
        source_variant: str,
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
                    source_variant=source_variant,
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
        left_name = self._prefer_team_on_side(sorted_candidates, side="left")
        if len(sorted_candidates) == 1:
            return [left_name]

        right_name = self._prefer_team_on_side(sorted_candidates, side="right")
        opposite_for_left = self._best_opposite_team_candidate(
            candidates=sorted_candidates,
            anchor=left_name,
            side="right",
        )
        opposite_for_right = self._best_opposite_team_candidate(
            candidates=sorted_candidates,
            anchor=right_name,
            side="left",
        )

        if left_name.confidence >= self.team_anchor_confidence and opposite_for_left is not None:
            right_name = opposite_for_left
        elif right_name.confidence >= self.team_anchor_confidence and opposite_for_right is not None:
            left_name = opposite_for_right
        elif left_name.team == right_name.team or left_name.player_id == right_name.player_id:
            if opposite_for_left is not None:
                right_name = opposite_for_left
            elif opposite_for_right is not None:
                left_name = opposite_for_right

        if left_name.player_id == right_name.player_id or left_name.team == right_name.team:
            alternatives = [item for item in sorted_candidates if item.player_id != left_name.player_id]
            if not alternatives:
                return [left_name]
            right_name = self._prefer_team_on_side(alternatives, side="right")
        return sorted([left_name, right_name], key=lambda item: (item.box[0], item.box[1]))

    @staticmethod
    def _prefer_team_on_side(
        candidates: list[KillfeedResolvedName],
        side: str,
    ) -> KillfeedResolvedName:
        """Prefer candidates whose team hint matches the side expectation, if present."""
        if side == "left":
            hinted = [candidate for candidate in candidates if candidate.preferred_team in {None, "friendly"}]
            pool = hinted or candidates
            return min(pool, key=lambda item: (item.box[0], item.box[1], -item.confidence))
        hinted = [candidate for candidate in candidates if candidate.preferred_team in {None, "enemy"}]
        pool = hinted or candidates
        return max(pool, key=lambda item: (item.box[0] + item.box[2], item.confidence))

    @staticmethod
    def _best_opposite_team_candidate(
        candidates: list[KillfeedResolvedName],
        anchor: KillfeedResolvedName,
        side: str,
    ) -> KillfeedResolvedName | None:
        """Pick the strongest candidate from the opposite team on the requested side."""
        pool = [
            candidate
            for candidate in candidates
            if candidate.player_id != anchor.player_id
            and candidate.team != anchor.team
        ]
        if side == "right":
            pool = [candidate for candidate in pool if candidate.box[0] >= anchor.box[0]]
            if not pool:
                return None
            return max(pool, key=lambda item: (item.box[0] + item.box[2], item.confidence))
        pool = [candidate for candidate in pool if candidate.box[0] <= anchor.box[0]]
        if not pool:
            return None
        return min(pool, key=lambda item: (-item.confidence, item.box[0]))

    def _apply_row_event(self, state: KillfeedRowState, timestamp_seconds: float) -> None:
        """Apply kill/death telemetry when a row yields two legible sides."""
        self._prune_event_threads(timestamp_seconds)
        thread, created = self._match_or_create_thread(state, timestamp_seconds)
        if thread is None:
            return
        if created:
            self.snapshot.event_threads_created += 1
        else:
            self.snapshot.event_threads_reused += 1
        if thread.counted:
            return
        left_name = thread.left_name
        right_name = thread.right_name
        if left_name is None or right_name is None:
            return
        if left_name.player_id == right_name.player_id:
            self.snapshot.events_blocked_same_side += 1
            return
        if left_name.team == right_name.team:
            self.snapshot.events_blocked_same_side += 1
            return
        if left_name.confidence < self.event_name_min_confidence or right_name.confidence < self.event_name_min_confidence:
            self.snapshot.events_blocked_low_confidence += 1
            return

        victim_last_seen = self._victim_last_seen.get(right_name.player_id)
        if victim_last_seen is not None and (timestamp_seconds - victim_last_seen) < self.victim_cooldown_seconds:
            self.snapshot.events_blocked_victim_cooldown += 1
            return

        self._victim_last_seen[right_name.player_id] = timestamp_seconds
        thread.counted = True

        self.snapshot.kills += 1
        self.snapshot.deaths += 1
        self.snapshot.events_counted += 1
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
    ) -> tuple[KillfeedEventThread | None, bool]:
        """Merge a row state into an existing vertical-band thread or create a new one."""
        left_name = state.resolved_names[0] if state.resolved_names else None
        right_name = state.resolved_names[-1] if len(state.resolved_names) >= 2 else None
        if left_name is None and right_name is None:
            return None, False

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

        created = False
        if best_thread is None:
            best_thread = KillfeedEventThread(center_y=state.center_y, last_seen=timestamp_seconds)
            self._event_threads.append(best_thread)
            created = True

        best_thread.last_seen = timestamp_seconds
        best_thread.center_y = state.center_y
        best_thread.left_name = self._prefer_name(best_thread.left_name, left_name)
        best_thread.right_name = self._prefer_name(best_thread.right_name, right_name)
        return best_thread, created

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

    def _prepare_easyocr_variants(self, killfeed_crop: np.ndarray) -> dict[str, np.ndarray]:
        """Build multiple OCR prep variants so colored text can stand out from the background."""
        variants = {
            "grayscale": self._prepare_easyocr_crop(killfeed_crop),
            "red": self._prepare_color_easyocr_crop(killfeed_crop, "red"),
            "blue": self._prepare_color_easyocr_crop(killfeed_crop, "blue"),
            "yellow": self._prepare_color_easyocr_crop(killfeed_crop, "yellow"),
        }
        return variants

    @staticmethod
    def _prepare_color_easyocr_crop(killfeed_crop: np.ndarray, color_name: str) -> np.ndarray:
        """Prepare a color-focused OCR view for one target text color."""
        hsv = cv2.cvtColor(killfeed_crop, cv2.COLOR_BGR2HSV)

        if color_name == "red":
            # Close to #D61A1A, with a small wraparound allowance.
            lower_1 = np.array([0, 100, 50], dtype=np.uint8)
            upper_1 = np.array([4, 255, 235], dtype=np.uint8)
            lower_2 = np.array([176, 100, 50], dtype=np.uint8)
            upper_2 = np.array([179, 255, 235], dtype=np.uint8)
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower_1, upper_1), cv2.inRange(hsv, lower_2, upper_2))
        elif color_name == "blue":
            # Close to #3DA5FF, but widened toward lighter cyan-adjacent HUD blues.
            lower = np.array([92, 80, 105], dtype=np.uint8)
            upper = np.array([114, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
        else:
            # Favor brighter, higher-saturation yellows so the text pops harder.
            lower = np.array([18, 110, 170], dtype=np.uint8)
            upper = np.array([40, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

        # Preserve hard text edges instead of smoothing them into the background.
        upscaled = cv2.resize(mask, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        if np.mean(upscaled == 255) > 0.60:
            upscaled = cv2.bitwise_not(upscaled)
        return upscaled

    @staticmethod
    def _dedupe_resolved_candidates(
        candidates: list[KillfeedResolvedName],
    ) -> list[KillfeedResolvedName]:
        """Keep the strongest candidate when multiple prep variants hit the same visual text box."""
        if not candidates:
            return []

        deduped: list[KillfeedResolvedName] = []
        for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
            duplicate = False
            for existing in deduped:
                if candidate.player_id != existing.player_id:
                    continue
                if KillfeedModule._boxes_similar(candidate.box, existing.box):
                    duplicate = True
                    break
            if not duplicate:
                deduped.append(candidate)
        return deduped

    def _build_reference_candidates(
        self,
        red_candidates: list[KillfeedResolvedName],
        all_candidates: list[KillfeedResolvedName],
    ) -> list[KillfeedResolvedName]:
        """Use red OCR boxes as positional anchors when available, but allow names from any prep."""
        if not all_candidates:
            return []
        if not red_candidates:
            return all_candidates

        anchored: list[KillfeedResolvedName] = []
        consumed_indices: set[int] = set()
        for red_candidate in red_candidates:
            best_index = self._find_best_reference_match(
                anchor=red_candidate,
                candidates=all_candidates,
                required_team="enemy",
            )
            if best_index is None:
                anchored.append(
                    KillfeedResolvedName(
                        player_id=red_candidate.player_id,
                        name=red_candidate.name,
                        team=red_candidate.team,
                        confidence=red_candidate.confidence,
                        box=red_candidate.box,
                        raw_text=red_candidate.raw_text,
                        source_variant=red_candidate.source_variant,
                        preferred_team="enemy",
                    )
                )
                continue
            consumed_indices.add(best_index)
            best_candidate = all_candidates[best_index]
            anchored.append(
                KillfeedResolvedName(
                    player_id=best_candidate.player_id,
                    name=best_candidate.name,
                    team=best_candidate.team,
                    confidence=best_candidate.confidence,
                    box=red_candidate.box,
                    raw_text=best_candidate.raw_text,
                    source_variant=best_candidate.source_variant,
                    preferred_team="enemy",
                )
            )

        for index, candidate in enumerate(all_candidates):
            if index in consumed_indices:
                continue
            if any(self._boxes_similar(candidate.box, anchored_candidate.box) for anchored_candidate in anchored):
                continue
            anchored.append(
                KillfeedResolvedName(
                    player_id=candidate.player_id,
                    name=candidate.name,
                    team=candidate.team,
                    confidence=candidate.confidence,
                    box=candidate.box,
                    raw_text=candidate.raw_text,
                    source_variant=candidate.source_variant,
                    preferred_team="friendly",
                )
            )
        return self._dedupe_resolved_candidates(anchored)

    @staticmethod
    def _find_best_reference_match(
        anchor: KillfeedResolvedName,
        candidates: list[KillfeedResolvedName],
        required_team: str | None = None,
    ) -> int | None:
        """Find the strongest OCR candidate that aligns closely with a red anchor box."""
        best_index: int | None = None
        best_score: tuple[float, float] | None = None
        for index, candidate in enumerate(candidates):
            if required_team is not None and candidate.team != required_team:
                continue
            if not KillfeedModule._boxes_similar(anchor.box, candidate.box):
                continue
            overlap = KillfeedModule._box_overlap_ratio(anchor.box, candidate.box)
            score = (overlap, candidate.confidence)
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        return best_index

    @staticmethod
    def _box_overlap_ratio(left_box: tuple[int, int, int, int], right_box: tuple[int, int, int, int]) -> float:
        """Return intersection-over-union style overlap for two OCR boxes."""
        lx, ly, lw, lh = left_box
        rx, ry, rw, rh = right_box
        left_x2 = lx + lw
        left_y2 = ly + lh
        right_x2 = rx + rw
        right_y2 = ry + rh

        inter_x1 = max(lx, rx)
        inter_y1 = max(ly, ry)
        inter_x2 = min(left_x2, right_x2)
        inter_y2 = min(left_y2, right_y2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = float(inter_w * inter_h)
        if intersection <= 0.0:
            return 0.0
        left_area = float(max(1, lw * lh))
        right_area = float(max(1, rw * rh))
        union = left_area + right_area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    @staticmethod
    def _boxes_similar(left_box: tuple[int, int, int, int], right_box: tuple[int, int, int, int]) -> bool:
        """Return whether two OCR boxes likely describe the same on-screen text region."""
        lx, ly, lw, lh = left_box
        rx, ry, rw, rh = right_box
        left_center = (lx + (lw / 2.0), ly + (lh / 2.0))
        right_center = (rx + (rw / 2.0), ry + (rh / 2.0))
        center_distance = abs(left_center[0] - right_center[0]) + abs(left_center[1] - right_center[1])
        size_distance = abs(lw - rw) + abs(lh - rh)
        return center_distance <= 40.0 and size_distance <= 35.0

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
        sequence_ratio = float(SequenceMatcher(a=left, b=right).ratio())
        substring_ratio = 0.0
        if left in right or right in left:
            substring_ratio = float(min(len(left), len(right)) / max(len(left), len(right)))

        # Weight distinctive character runs more heavily than broad fuzzy overlap.
        bigram_score = KillfeedModule._ngram_overlap_score(left, right, 2)
        trigram_score = KillfeedModule._ngram_overlap_score(left, right, 3)
        longest_run_score = KillfeedModule._longest_common_substring_score(left, right)
        length_score = KillfeedModule._length_similarity_score(left, right)

        return min(
            1.0,
            (0.17 * sequence_ratio)
            + (0.10 * substring_ratio)
            + (0.22 * bigram_score)
            + (0.24 * trigram_score)
            + (0.12 * longest_run_score)
            + (0.15 * length_score),
        )

    @staticmethod
    def _length_similarity_score(left: str, right: str) -> float:
        """Score how close two normalized name lengths are."""
        if not left or not right:
            return 0.0
        return float(min(len(left), len(right)) / max(len(left), len(right)))

    @staticmethod
    def _ngram_overlap_score(left: str, right: str, n: int) -> float:
        """Score overlap of contiguous character sequences."""
        if len(left) < n or len(right) < n:
            return 0.0
        left_ngrams = {left[index : index + n] for index in range(len(left) - n + 1)}
        right_ngrams = {right[index : index + n] for index in range(len(right) - n + 1)}
        if not left_ngrams or not right_ngrams:
            return 0.0
        shared = left_ngrams & right_ngrams
        return float((2 * len(shared)) / (len(left_ngrams) + len(right_ngrams)))

    @staticmethod
    def _longest_common_substring_score(left: str, right: str) -> float:
        """Score the strongest shared contiguous character run."""
        if not left or not right:
            return 0.0
        matcher = SequenceMatcher(a=left, b=right)
        match = matcher.find_longest_match(0, len(left), 0, len(right))
        if match.size <= 0:
            return 0.0
        return float((2 * match.size) / (len(left) + len(right)))

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
