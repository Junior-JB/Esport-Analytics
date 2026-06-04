"""Scoreboard registration and final-score validation module."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
import re

import cv2
import numpy as np
import pytesseract

from app.models import (
    FinalScoreSnapshot,
    FramePacket,
    MatchContext,
    ModuleStatus,
    RegisteredPlayer,
    ScoreboardRegistrationResult,
    ScoreboardSnapshot,
)
from app.modules.base import PipelineModule
from app.scheduler import FixedRateScheduler


@dataclass(slots=True)
class PixelBounds:
    """Absolute integer bounds for one configured scoreboard ROI."""

    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class SlotObservation:
    """Running consensus state for one fixed scoreboard slot."""

    player_id: int
    team: str
    slot_index: int
    total_observations: int = 0
    weighted_scores: dict[str, float] = field(default_factory=dict)
    observation_counts: dict[str, int] = field(default_factory=dict)
    display_variants: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))
    resolved_name: str | None = None
    resolved_confidence: float = 0.0
    last_raw_text: str | None = None

    def add(self, raw_text: str, normalized_text: str, confidence: float) -> None:
        """Record one OCR observation for the slot."""
        self.total_observations += 1
        self.last_raw_text = raw_text
        self.weighted_scores[normalized_text] = self.weighted_scores.get(normalized_text, 0.0) + confidence
        self.observation_counts[normalized_text] = self.observation_counts.get(normalized_text, 0) + 1
        variants = self.display_variants.setdefault(normalized_text, {})
        variants[raw_text] = variants.get(raw_text, 0) + 1
        self._resolve()

    def _resolve(self) -> None:
        """Resolve the current best name for the slot."""
        if not self.weighted_scores:
            self.resolved_name = None
            self.resolved_confidence = 0.0
            return
        best_key = max(
            self.weighted_scores,
            key=lambda key: (self.weighted_scores[key], self.observation_counts.get(key, 0), key),
        )
        variants = self.display_variants.get(best_key, {})
        best_display = max(variants, key=lambda value: (variants[value], len(value))) if variants else best_key
        total_weight = float(sum(self.weighted_scores.values()))
        confidence = 0.0 if total_weight <= 0.0 else (self.weighted_scores[best_key] / total_weight)
        self.resolved_name = best_display
        self.resolved_confidence = float(confidence)


@dataclass(slots=True)
class TeamNameObservation:
    """Running consensus for one player name on one side of the scoreboard."""

    team: str
    canonical_key: str
    total_observations: int = 0
    weighted_score: float = 0.0
    display_variants: dict[str, int] = field(default_factory=dict)
    last_seen_slot_index: int | None = None

    def add(self, display_name: str, confidence: float, slot_index: int) -> None:
        """Record one observation for this team-level player candidate."""
        self.total_observations += 1
        self.weighted_score += confidence
        self.display_variants[display_name] = self.display_variants.get(display_name, 0) + 1
        self.last_seen_slot_index = slot_index

    @property
    def resolved_name(self) -> str:
        """Return the most likely display form for this player."""
        if not self.display_variants:
            return self.canonical_key
        return max(self.display_variants, key=lambda value: (self.display_variants[value], len(value)))

    @property
    def confidence(self) -> float:
        """Return the aggregated confidence for this player candidate."""
        if self.total_observations <= 0:
            return 0.0
        base_confidence = float(self.weighted_score / self.total_observations)
        repeat_bonus = min(0.20, max(0, self.total_observations - 1) * 0.05)
        return min(1.0, base_confidence + repeat_bonus)


class ScoreboardRegistrationModule(PipelineModule):
    """Detect scoreboard visibility, register players, and validate final score."""

    _NAME_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-[]"

    def __init__(self, config: dict, scheduling_config: dict) -> None:
        super().__init__(name="scoreboard_registration")
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.requires_registration = False
        self.scheduler = FixedRateScheduler(scheduling_config.get("samples_per_second", 1.0))
        self.snapshot = ScoreboardSnapshot(
            current_state="waiting" if self.enabled else "disabled",
        )
        self.registration_result: ScoreboardRegistrationResult | None = None
        self.status = ModuleStatus.WAITING if self.enabled else ModuleStatus.DISABLED
        self.required_player_count = int(self.config.get("required_player_count", 8))
        self.disable_after_registration = bool(self.config.get("disable_after_registration", True))
        self.refine_registry_after_registration = bool(
            self.config.get("refine_registry_after_registration", True)
        )
        self.registration_min_slot_observations = int(
            self.config.get("registration_min_slot_observations", 2)
        )
        self.registration_min_slot_confidence = float(
            self.config.get("registration_min_slot_confidence", 0.72)
        )
        self.final_score_score_threshold = int(self.config.get("final_score_authoritative_value", 250))
        self.roi_preview_mode = bool(self.config.get("roi_preview_mode", True))
        self.preserve_debug_artifacts = bool(self.config.get("preserve_debug_artifacts", False))
        self.debug_every_visible_frame = bool(self.config.get("debug_every_visible_frame", False))
        self.visibility_config = dict(self.config.get("visibility", {}))
        self.absolute_regions = dict(self.config.get("absolute_regions", {}))
        self.header_template_path = Path(
            self.visibility_config.get(
                "header_template_path",
                Path(__file__).resolve().parent.parent.parent
                / "assets"
                / "scoreboard_visibility"
                / "scoreboard_header_presence_template.png",
            )
        )
        self.header_template_min_similarity = float(
            self.visibility_config.get("header_template_min_similarity", 0.70)
        )
        self._last_ocr_confidence: float | None = None
        self._debug_output_dir: Path | None = None
        self._debug_temp_dir: Path | None = None
        self._debug_artifacts: list[Path] = []
        self._slot_observations = self._build_slot_observations()
        self._team_observations: dict[str, list[TeamNameObservation]] = {
            "friendly": [],
            "enemy": [],
        }
        self.name_match_similarity = float(self.config.get("name_match_similarity", 0.82))
        self.name_min_length = int(self.config.get("name_min_length", 3))
        self.readability_min_fill_ratio = float(self.config.get("readability_min_fill_ratio", 0.02))
        self.readability_max_fill_ratio = float(self.config.get("readability_max_fill_ratio", 0.45))
        self.readability_min_foreground_columns = float(
            self.config.get("readability_min_foreground_columns", 0.20)
        )
        self.non_empty_name_floor_confidence = float(
            self.config.get("non_empty_name_floor_confidence", 0.55)
        )
        self._header_template = self._load_header_template()

    def set_debug_output_dir(self, output_dir: Path) -> None:
        """Attach a run-scoped debug directory."""
        self._debug_output_dir = output_dir
        self._debug_temp_dir = output_dir / "scoreboard_debug_temp"
        self._debug_temp_dir.mkdir(parents=True, exist_ok=True)

    def finalize(self) -> None:
        """Remove temporary scoreboard artifacts unless preservation is enabled."""
        if self.preserve_debug_artifacts or self._debug_temp_dir is None:
            return
        for path in sorted(self._debug_temp_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        if self._debug_temp_dir.exists():
            self._debug_temp_dir.rmdir()

    @property
    def last_ocr_confidence(self) -> float | None:
        """Expose the most recent scoreboard OCR confidence."""
        return self._last_ocr_confidence

    def process(self, frame_packet: FramePacket, match_context: MatchContext) -> None:
        """Process scoreboard registration or final validation on scheduled frames."""
        if not self.enabled:
            self.status = ModuleStatus.DISABLED
            self.snapshot.current_state = "disabled"
            return
        if frame_packet.image is None:
            self.status = ModuleStatus.WAITING
            return
        if not self.scheduler.should_run(frame_packet.timestamp_seconds):
            return

        scoreboard_visible = self._detect_scoreboard_visibility(frame_packet.image)
        self.snapshot.scoreboard_visible = scoreboard_visible
        self.snapshot.registration_complete = match_context.registration_complete
        self.snapshot.player_registry = dict(match_context.player_registry)
        self.snapshot.registered_players = match_context.all_player_names
        self.snapshot.registry_confidence = match_context.registry_confidence
        self.snapshot.final_score_detected = (
            match_context.final_score.blue is not None and match_context.final_score.red is not None
        )
        self.snapshot.final_score_blue = match_context.final_score.blue
        self.snapshot.final_score_red = match_context.final_score.red

        if self.roi_preview_mode and (scoreboard_visible or self.debug_every_visible_frame):
            self._write_roi_preview(frame_packet.image, frame_packet)

        if not scoreboard_visible:
            self._last_ocr_confidence = None
            self.snapshot.ocr_confidence = None
            self.status = ModuleStatus.COMPLETE if match_context.registration_complete else ModuleStatus.WAITING
            self.snapshot.current_state = "registered_monitoring" if match_context.registration_complete else "searching"
            return

        self.status = ModuleStatus.ACTIVE
        if not match_context.registration_complete or self.refine_registry_after_registration:
            self._extract_slot_observations(frame_packet.image)
            self._promote_registry(match_context)

        if not match_context.registration_complete and self._registration_ready():
            result = self._build_registration_result()
            self.complete_registration(result, match_context)

        if match_context.registration_complete:
            self._validate_final_score(frame_packet.image, match_context)

        self.snapshot.registration_complete = match_context.registration_complete
        self.snapshot.player_registry = dict(match_context.player_registry)
        self.snapshot.registered_players = match_context.all_player_names
        self.snapshot.registry_confidence = match_context.registry_confidence
        self.snapshot.final_score_detected = (
            match_context.final_score.blue is not None and match_context.final_score.red is not None
        )
        self.snapshot.final_score_blue = match_context.final_score.blue
        self.snapshot.final_score_red = match_context.final_score.red
        self.snapshot.current_state = (
            "registered_monitoring" if match_context.registration_complete else "registering"
        )

    def complete_registration(self, result: ScoreboardRegistrationResult, match_context: MatchContext) -> None:
        """Populate the authoritative match context."""
        players = sorted(result.players, key=lambda player: player.player_id)
        match_context.players = players
        match_context.player_registry = {str(player.player_id): player.name for player in players}
        match_context.registration_complete = result.registration_complete
        match_context.registry_confidence = result.confidence
        self.registration_result = result
        self.snapshot.registration_complete = result.registration_complete
        self.snapshot.registry_confidence = result.confidence
        self.snapshot.registered_players = [player.name for player in players]
        self.snapshot.player_registry = dict(match_context.player_registry)
        self.status = ModuleStatus.COMPLETE if self.disable_after_registration else ModuleStatus.ACTIVE

    @staticmethod
    def build_registration(players: list[RegisteredPlayer], confidence: float) -> ScoreboardRegistrationResult:
        """Create a validated registration payload."""
        return ScoreboardRegistrationResult(
            players=players,
            registration_complete=len(players) == 8,
            confidence=confidence,
        )

    def _build_slot_observations(self) -> dict[str, SlotObservation]:
        """Create the fixed 8 scoreboard slot trackers."""
        observations: dict[str, SlotObservation] = {}
        player_id = 1
        for team in ("friendly", "enemy"):
            for slot_index in range(4):
                key = self._slot_key(team, slot_index)
                observations[key] = SlotObservation(
                    player_id=player_id,
                    team=team,
                    slot_index=slot_index,
                )
                player_id += 1
        return observations

    def _detect_scoreboard_visibility(self, frame: np.ndarray) -> bool:
        """Use ROI structure checks before any OCR runs."""
        if "scoreboard_roi" not in self.absolute_regions:
            return False
        scoreboard_crop = self._crop_region(frame, "scoreboard_roi")
        if scoreboard_crop is None or scoreboard_crop.size == 0:
            return False
        scoreboard_gray = cv2.cvtColor(scoreboard_crop, cv2.COLOR_BGR2GRAY)
        _, scoreboard_binary = cv2.threshold(scoreboard_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        roi_fill_ratio = float(np.count_nonzero(scoreboard_binary)) / float(scoreboard_binary.size)

        header_similarity = 0.0
        if "scoreboard_header_roi" in self.absolute_regions:
            header_crop = self._crop_region(frame, "scoreboard_header_roi")
            if header_crop is not None and header_crop.size > 0 and self._header_template is not None:
                header_similarity = self._match_header_template(header_crop)

        min_roi_fill_ratio = float(self.visibility_config.get("scoreboard_min_fill_ratio", 0.03))
        min_header_similarity = self.header_template_min_similarity
        return (
            roi_fill_ratio >= min_roi_fill_ratio
            and header_similarity >= min_header_similarity
        )

    def _extract_slot_observations(self, frame: np.ndarray) -> None:
        """OCR all 8 fixed player slots."""
        confidences: list[float] = []
        for team in ("friendly", "enemy"):
            frame_candidates: list[tuple[int, str, str, float]] = []
            for slot_index in range(4):
                region_name = f"{team}_slot_{slot_index + 1}"
                crop = self._crop_region(frame, region_name)
                if crop is None or crop.size == 0:
                    continue
                name, confidence = self._ocr_name(crop)
                if not name:
                    continue
                normalized = self._normalize_name(name)
                if not normalized:
                    continue
                slot = self._slot_observations[self._slot_key(team, slot_index)]
                slot.add(name, normalized, confidence)
                frame_candidates.append((slot_index, name, normalized, confidence))
                confidences.append(confidence)
            self._merge_team_candidates(team, frame_candidates)

        self._last_ocr_confidence = self._average(confidences)
        self.snapshot.ocr_confidence = self._last_ocr_confidence

    def _promote_registry(self, match_context: MatchContext) -> None:
        """Push the current team-level best names into the shared registry."""
        players = self._build_team_registry_players()
        match_context.players = players
        match_context.player_registry = {str(player.player_id): player.name for player in players}
        match_context.registry_confidence = self._average([player.confidence for player in players])

    def _registration_ready(self) -> bool:
        """Return whether both teams have four stable unique names."""
        return (
            self._ready_team_count("friendly") >= 4
            and self._ready_team_count("enemy") >= 4
        )

    def _build_registration_result(self) -> ScoreboardRegistrationResult:
        """Create the authoritative registration payload from team-level consensus."""
        players = self._build_team_registry_players()
        confidence = self._average([player.confidence for player in players]) or 0.0
        return self.build_registration(players, confidence)

    def _merge_team_candidates(
        self,
        team: str,
        candidates: list[tuple[int, str, str, float]],
    ) -> None:
        """Merge one scoreboard opening's names into the team-level registry pool."""
        deduped_candidates = self._dedupe_frame_candidates(candidates)
        observations = self._team_observations[team]
        for slot_index, display_name, normalized_name, confidence in deduped_candidates:
            match = self._find_team_match(observations, normalized_name)
            if match is None:
                match = TeamNameObservation(team=team, canonical_key=normalized_name)
                observations.append(match)
            match.add(display_name, confidence, slot_index)

    def _dedupe_frame_candidates(
        self,
        candidates: list[tuple[int, str, str, float]],
    ) -> list[tuple[int, str, str, float]]:
        """Collapse near-duplicate names within one scoreboard opening per side."""
        deduped: list[tuple[int, str, str, float]] = []
        for slot_index, display_name, normalized_name, confidence in sorted(
            candidates,
            key=lambda item: item[3],
            reverse=True,
        ):
            existing_index = None
            for index, (_, _, existing_normalized, _) in enumerate(deduped):
                if self._names_match(existing_normalized, normalized_name):
                    existing_index = index
                    break
            if existing_index is None:
                deduped.append((slot_index, display_name, normalized_name, confidence))
                continue
            if confidence > deduped[existing_index][3]:
                deduped[existing_index] = (slot_index, display_name, normalized_name, confidence)
        return deduped

    def _find_team_match(
        self,
        observations: list[TeamNameObservation],
        normalized_name: str,
    ) -> TeamNameObservation | None:
        """Find the closest existing team candidate for a normalized name."""
        best_match: TeamNameObservation | None = None
        best_ratio = 0.0
        for observation in observations:
            ratio = self._name_similarity(observation.canonical_key, normalized_name)
            if ratio >= self.name_match_similarity and ratio > best_ratio:
                best_match = observation
                best_ratio = ratio
        return best_match

    def _build_team_registry_players(self) -> list[RegisteredPlayer]:
        """Build the current left/right team registry from the best unique candidates."""
        players: list[RegisteredPlayer] = []
        player_id = 1
        for team in ("friendly", "enemy"):
            team_candidates = self._top_team_candidates(team, limit=4)
            for team_slot_index, observation in enumerate(team_candidates):
                players.append(
                    RegisteredPlayer(
                        player_id=player_id,
                        name=observation.resolved_name,
                        team=team,
                        confidence=observation.confidence,
                        slot_index=team_slot_index,
                        observation_count=observation.total_observations,
                    )
                )
                player_id += 1
        return players

    def _ready_team_count(self, team: str) -> int:
        """Return how many stable unique names the given side has accumulated."""
        ready = 0
        for observation in self._top_team_candidates(team, limit=4):
            if (
                observation.total_observations >= self.registration_min_slot_observations
                and observation.confidence >= self.registration_min_slot_confidence
            ):
                ready += 1
        return ready

    def _top_team_candidates(self, team: str, limit: int) -> list[TeamNameObservation]:
        """Return the strongest current unique name candidates for one side."""
        observations = sorted(
            self._team_observations[team],
            key=lambda item: (item.total_observations, item.weighted_score, len(item.resolved_name)),
            reverse=True,
        )
        return observations[:limit]

    def _validate_final_score(self, frame: np.ndarray, match_context: MatchContext) -> None:
        """Validate the final scoreboard score when an authoritative 250 appears."""
        blue = self._ocr_numeric_region(frame, "final_blue_score_roi", allowlist="0123456789")
        red = self._ocr_numeric_region(frame, "final_red_score_roi", allowlist="0123456789")
        if blue is None or red is None:
            return
        if blue == self.final_score_score_threshold or red == self.final_score_score_threshold:
            match_context.final_score = FinalScoreSnapshot(blue=blue, red=red, source="scoreboard")

    def _ocr_numeric_region(self, frame: np.ndarray, region_name: str, allowlist: str) -> int | None:
        """Read a small numeric scoreboard region with Tesseract."""
        crop = self._crop_region(frame, region_name)
        if crop is None or crop.size == 0:
            return None
        binary = self._prepare_numeric_crop(crop)
        config = f"--oem 3 --psm 7 -c tessedit_char_whitelist={allowlist}"
        text = pytesseract.image_to_string(binary, config=config).strip()
        digits = "".join(character for character in text if character.isdigit())
        return int(digits) if digits else None

    def _ocr_name(self, crop: np.ndarray) -> tuple[str | None, float]:
        """Read one player-name slot with lightweight Tesseract OCR."""
        prepared = self._prepare_name_crop(crop)
        readability = self._score_name_readability(prepared)
        config = f"--oem 3 --psm 7 -c tessedit_char_whitelist={self._NAME_ALLOWLIST}"
        data = pytesseract.image_to_data(
            prepared,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        texts: list[str] = []
        confidences: list[float] = []
        for text, confidence_text in zip(data.get("text", []), data.get("conf", []), strict=False):
            candidate = str(text).strip()
            if not candidate:
                continue
            try:
                confidence = max(0.0, float(confidence_text)) / 100.0
            except (TypeError, ValueError):
                continue
            cleaned = self._clean_display_name(candidate)
            if not cleaned:
                continue
            texts.append(cleaned)
            confidences.append(confidence)
        if not texts:
            return None, 0.0
        merged = "".join(texts)
        if len(self._normalize_name(merged)) < self.name_min_length:
            return None, 0.0

        ocr_confidence = self._average(confidences) or 0.0
        if ocr_confidence <= 0.0:
            ocr_confidence = self.non_empty_name_floor_confidence
        elif readability >= 0.50:
            ocr_confidence = max(ocr_confidence, self.non_empty_name_floor_confidence)

        blended_confidence = (0.65 * readability) + (0.35 * ocr_confidence)
        blended_confidence = max(blended_confidence, self.non_empty_name_floor_confidence * 0.90)
        return merged, min(1.0, blended_confidence)

    def _prepare_name_crop(self, crop: np.ndarray) -> np.ndarray:
        """Prepare one player-slot crop for slot OCR and structure checks."""
        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(grayscale, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary == 255) > 0.55:
            binary = cv2.bitwise_not(binary)
        return binary

    def _prepare_numeric_crop(self, crop: np.ndarray) -> np.ndarray:
        """Prepare a scoreboard score crop for digit-only OCR."""
        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(grayscale, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary == 255) > 0.55:
            binary = cv2.bitwise_not(binary)
        return binary

    def _score_name_readability(self, binary: np.ndarray) -> float:
        """Estimate how readable a thresholded player-name crop is."""
        if binary.size <= 0:
            return 0.0
        foreground = (binary > 0).astype(np.uint8)
        fill_ratio = float(np.count_nonzero(foreground)) / float(foreground.size)
        if fill_ratio <= 0.0:
            return 0.0

        foreground_columns = np.count_nonzero(foreground.sum(axis=0) > 0)
        column_coverage = float(foreground_columns) / float(foreground.shape[1])
        fill_score = self._bounded_ratio_score(
            fill_ratio,
            self.readability_min_fill_ratio,
            self.readability_max_fill_ratio,
        )
        coverage_score = min(1.0, column_coverage / max(0.01, self.readability_min_foreground_columns))

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        significant_components = 0
        min_area = max(4, int(round(foreground.size * 0.003)))
        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                significant_components += 1
        component_score = 0.0
        if significant_components > 0:
            component_score = min(1.0, significant_components / 6.0)

        return min(1.0, (0.45 * fill_score) + (0.35 * coverage_score) + (0.20 * component_score))

    def _load_header_template(self) -> np.ndarray | None:
        """Load the configured scoreboard header template asset."""
        if not self.header_template_path.exists():
            return None
        image = cv2.imread(str(self.header_template_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return image

    def _match_header_template(self, crop: np.ndarray) -> float:
        """Return normalized similarity for the header-presence template."""
        if self._header_template is None:
            return 0.0
        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        template_height, template_width = self._header_template.shape[:2]
        resized = cv2.resize(grayscale, (template_width, template_height), interpolation=cv2.INTER_CUBIC)
        result = cv2.matchTemplate(resized, self._header_template, cv2.TM_CCOEFF_NORMED)
        return float(result.max()) if result.size > 0 else 0.0

    def _write_roi_preview(self, frame: np.ndarray, frame_packet: FramePacket) -> None:
        """Write one scoreboard ROI preview image for box placement/debugging."""
        if self._debug_temp_dir is None:
            return
        preview = frame.copy()
        label_map = {
            "scoreboard_roi": (255, 255, 0),
            "scoreboard_header_roi": (0, 255, 255),
            "final_blue_score_roi": (255, 128, 0),
            "final_red_score_roi": (255, 0, 255),
        }
        for region_name, color in label_map.items():
            if region_name not in self.absolute_regions:
                continue
            self._draw_region(preview, region_name, color)
        for region_name in self._slot_region_names():
            if region_name in self.absolute_regions:
                self._draw_region(preview, region_name, (180, 180, 180))

        lines = [
            f"scoreboard_visible: {self.snapshot.scoreboard_visible}",
            f"current_state: {self.snapshot.current_state}",
            f"registered_players: {len(self.snapshot.registered_players)}",
            f"registration_complete: {self.snapshot.registration_complete}",
            f"registry_confidence: {self.snapshot.registry_confidence}",
            f"final_score_detected: {self.snapshot.final_score_detected}",
            f"OCR confidence: {self.snapshot.ocr_confidence}",
            f"player_registry: {self.snapshot.player_registry}",
        ]
        y = 24
        for line in lines:
            cv2.putText(preview, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            y += 22

        path = self._debug_temp_dir / f"scoreboard_preview_{frame_packet.frame_index}.jpg"
        cv2.imwrite(str(path), preview)
        self._debug_artifacts.append(path)

    def _draw_region(self, frame: np.ndarray, region_name: str, color: tuple[int, int, int]) -> None:
        """Draw one named region onto a preview frame."""
        region = self._resolve_bounds(frame.shape, self.absolute_regions[region_name])
        cv2.rectangle(
            frame,
            (region.x, region.y),
            (region.x + region.width, region.y + region.height),
            color,
            2,
        )
        cv2.putText(
            frame,
            region_name,
            (region.x, max(12, region.y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    def _crop_region(self, frame: np.ndarray, region_name: str) -> np.ndarray | None:
        """Crop one configured scoreboard region."""
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

    def _slot_region_names(self) -> list[str]:
        """Return the fixed slot region names in registration order."""
        names: list[str] = []
        for team in ("friendly", "enemy"):
            for slot_index in range(4):
                names.append(f"{team}_slot_{slot_index + 1}")
        return names

    @staticmethod
    def _slot_key(team: str, slot_index: int) -> str:
        """Create a stable key for one scoreboard slot."""
        return f"{team}_{slot_index}"

    @staticmethod
    def _normalize_name(text: str) -> str:
        """Normalize OCR text for slot-based consensus matching."""
        return re.sub(r"[^a-z0-9]", "", text.lower())

    def _names_match(self, left: str, right: str) -> bool:
        """Return whether two normalized names are close enough to treat as one player."""
        return self._name_similarity(left, right) >= self.name_match_similarity

    @staticmethod
    def _name_similarity(left: str, right: str) -> float:
        """Return fuzzy similarity for two normalized names."""
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        if left in right or right in left:
            shorter = min(len(left), len(right))
            longer = max(len(left), len(right))
            if longer > 0:
                return float(shorter / longer)
        return float(SequenceMatcher(a=left, b=right).ratio())

    @staticmethod
    def _bounded_ratio_score(value: float, minimum: float, maximum: float) -> float:
        """Score a ratio from 0-1 based on whether it sits in a readable band."""
        if maximum <= minimum:
            return 0.0
        if value < minimum:
            return max(0.0, value / max(minimum, 1e-6))
        if value > maximum:
            overflow = min(1.0, (value - maximum) / max(1e-6, 1.0 - maximum))
            return max(0.0, 1.0 - overflow)
        return 1.0

    @staticmethod
    def _clean_display_name(text: str) -> str:
        """Clean OCR text while preserving display characters."""
        cleaned = re.sub(r"[^A-Za-z0-9_\-\[\]]", "", text)
        return cleaned.strip()

    @staticmethod
    def _average(values: list[float]) -> float | None:
        """Return the average of non-empty numeric values."""
        usable = [value for value in values if value is not None]
        if not usable:
            return None
        return float(sum(usable) / len(usable))
