"""Utility extraction scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.models import FramePacket, MatchContext, ModuleStatus, UtilityCountSnapshot
from app.modules.base import PipelineModule
from app.scheduler import FixedRateScheduler


@dataclass(slots=True)
class PixelBounds:
    """Absolute integer bounds for one utility subregion."""

    x: int
    y: int
    width: int
    height: int


class UtilityModule(PipelineModule):
    """Simple utility usage counting only."""

    def __init__(self, config: dict, scheduling_config: dict) -> None:
        super().__init__(name="utility")
        self.config = config
        self.enabled = bool(config.get("enabled", True))
        self.requires_registration = bool(config.get("requires_registration", False))
        self.scheduler = FixedRateScheduler(scheduling_config.get("samples_per_second", 2.0))
        self.snapshot = UtilityCountSnapshot()
        self.utility_roi = dict(config.get("utility_roi", {}))
        self.absolute_regions = dict(config.get("absolute_regions", {}))
        self.binary_state_template_size = tuple(
            int(v) for v in config.get("binary_state_template_size", [18, 24])
        )
        self.binary_state_min_score = float(config.get("binary_state_min_score", 0.24))
        self.binary_state_min_margin = float(config.get("binary_state_min_margin", 0.20))
        self.shape_center_min_for_one = float(config.get("shape_center_min_for_one", 0.45))
        self.shape_side_max_for_one = float(config.get("shape_side_max_for_one", 0.42))
        self.shape_width_ratio_max_for_one = float(config.get("shape_width_ratio_max_for_one", 0.45))
        self.shape_width_ratio_min_for_zero = float(config.get("shape_width_ratio_min_for_zero", 0.34))
        self.shape_fill_ratio_min_for_zero = float(config.get("shape_fill_ratio_min_for_zero", 0.22))
        self.shape_fill_ratio_max_for_zero = float(config.get("shape_fill_ratio_max_for_zero", 0.70))
        self.shape_hole_min_area_ratio = float(config.get("shape_hole_min_area_ratio", 0.05))
        self.shape_hole_strong_area_ratio = float(config.get("shape_hole_strong_area_ratio", 0.10))
        self.specialty_orange_hue_min = int(config.get("specialty_orange_hue_min", 6))
        self.specialty_orange_hue_max = int(config.get("specialty_orange_hue_max", 24))
        self.specialty_orange_saturation_min = int(config.get("specialty_orange_saturation_min", 90))
        self.specialty_orange_value_min = int(config.get("specialty_orange_value_min", 90))
        self.specialty_orange_ratio_threshold = float(config.get("specialty_orange_ratio_threshold", 0.45))
        self.specialty_absence_seconds = float(config.get("specialty_absence_seconds", 15.0))
        self.specialty_activation_confirmation_frames = int(
            config.get("specialty_activation_confirmation_frames", 2)
        )

        self._frag_state: int | None = None
        self._frag_pending_state: int | None = None
        self._frag_pending_count = 0
        self._frag_seen_one = False
        self._frag_seen_zero_after_one = False

        self._tactical_state: int | None = None
        self._tactical_pending_state: int | None = None
        self._tactical_pending_count = 0
        self._tactical_seen_one = False
        self._tactical_seen_zero_after_one = False

        self._specialty_active = False
        self._specialty_active_confirmation_hits = 0
        self._specialty_absence_started_at: float | None = None
        self._last_confidence: float | None = None
        self.status = ModuleStatus.WAITING if self.enabled else ModuleStatus.DISABLED

    def process(self, frame_packet: FramePacket, match_context: MatchContext) -> None:
        """Run on a fixed schedule after registration."""
        if not self.enabled:
            self.status = ModuleStatus.DISABLED
            return
        if self.requires_registration and not match_context.registration_complete:
            self.status = ModuleStatus.WAITING
            return
        if not self.scheduler.should_run(frame_packet.timestamp_seconds):
            return
        self.status = ModuleStatus.ACTIVE
        if frame_packet.image is None:
            return

        grenade_state, grenade_confidence = self._classify_binary_region(frame_packet.image, "grenade")
        stun_state, stun_confidence = self._classify_binary_region(frame_packet.image, "stun")
        specialty_active, specialty_confidence = self._detect_specialty_active(frame_packet.image)
        self._last_confidence = self._average_confidence(
            grenade_confidence,
            stun_confidence,
            specialty_confidence,
        )

        self._update_frag_usage(grenade_state)
        self._update_tactical_usage(stun_state)
        self._update_specialty_usage(specialty_active, frame_packet.timestamp_seconds)

    @property
    def last_confidence(self) -> float | None:
        """Expose the most recent utility confidence for dashboard use."""
        return self._last_confidence

    def _update_frag_usage(self, current_state: int | None) -> None:
        """Count one frag usage on a 1 -> 0 -> 1 cycle."""
        confirmed_state = self._confirm_state_change(
            current_state=current_state,
            accepted_state=self._frag_state,
            pending_state=self._frag_pending_state,
            pending_count=self._frag_pending_count,
        )
        self._frag_pending_state = confirmed_state["pending_state"]
        self._frag_pending_count = confirmed_state["pending_count"]
        current_state = confirmed_state["accepted_state"]
        if current_state is None:
            return
        self._frag_state = current_state
        if current_state == 1:
            if self._frag_seen_one and self._frag_seen_zero_after_one:
                self.snapshot.grenade += 1
                self._frag_seen_zero_after_one = False
            self._frag_seen_one = True
            return
        if current_state == 0 and self._frag_seen_one:
            self._frag_seen_zero_after_one = True

    def _update_tactical_usage(self, current_state: int | None) -> None:
        """Count one tactical usage on a 1 -> 0 -> 1 cycle."""
        confirmed_state = self._confirm_state_change(
            current_state=current_state,
            accepted_state=self._tactical_state,
            pending_state=self._tactical_pending_state,
            pending_count=self._tactical_pending_count,
        )
        self._tactical_pending_state = confirmed_state["pending_state"]
        self._tactical_pending_count = confirmed_state["pending_count"]
        current_state = confirmed_state["accepted_state"]
        if current_state is None:
            return
        self._tactical_state = current_state
        if current_state == 1:
            if self._tactical_seen_one and self._tactical_seen_zero_after_one:
                self.snapshot.stun += 1
                self._tactical_seen_zero_after_one = False
            self._tactical_seen_one = True
            return
        if current_state == 0 and self._tactical_seen_one:
            self._tactical_seen_zero_after_one = True

    def _update_specialty_usage(self, specialty_active: bool, timestamp_seconds: float) -> None:
        """Count one specialty use after 15 seconds of continuous orange absence."""
        if specialty_active:
            self._specialty_active_confirmation_hits += 1
            if self._specialty_active_confirmation_hits >= self.specialty_activation_confirmation_frames:
                self._specialty_active = True
            self._specialty_absence_started_at = None
            return

        self._specialty_active_confirmation_hits = 0

        if not self._specialty_active:
            return
        if self._specialty_absence_started_at is None:
            self._specialty_absence_started_at = timestamp_seconds
            return
        if (timestamp_seconds - self._specialty_absence_started_at) >= self.specialty_absence_seconds:
            self.snapshot.specialty_equipment += 1
            self._specialty_active = False
            self._specialty_absence_started_at = None

    @staticmethod
    def _confirm_state_change(
        current_state: int | None,
        accepted_state: int | None,
        pending_state: int | None,
        pending_count: int,
    ) -> dict[str, int | None]:
        """Require two consecutive identical non-null reads before changing accepted state."""
        if current_state is None:
            return {
                "accepted_state": accepted_state,
                "pending_state": None,
                "pending_count": 0,
            }
        if accepted_state is None:
            if pending_state == current_state:
                pending_count += 1
            else:
                pending_state = current_state
                pending_count = 1
            if pending_count >= 2:
                return {
                    "accepted_state": current_state,
                    "pending_state": None,
                    "pending_count": 0,
                }
            return {
                "accepted_state": None,
                "pending_state": pending_state,
                "pending_count": pending_count,
            }
        if current_state == accepted_state:
            return {
                "accepted_state": accepted_state,
                "pending_state": None,
                "pending_count": 0,
            }
        if pending_state == current_state:
            pending_count += 1
        else:
            pending_state = current_state
            pending_count = 1
        if pending_count >= 2:
            return {
                "accepted_state": current_state,
                "pending_state": None,
                "pending_count": 0,
            }
        return {
            "accepted_state": accepted_state,
            "pending_state": pending_state,
            "pending_count": pending_count,
        }

    def _classify_binary_region(self, frame: np.ndarray, region_name: str) -> tuple[int | None, float]:
        """Read a tiny utility counter as 0, 1, or null using shape structure."""
        crop = self._crop_region(frame, region_name)
        if crop is None or crop.size == 0:
            return None, 0.0
        binary = self._prepare_binary(crop)
        scores = self._shape_scores(binary)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_digit, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score
        if best_score < self.binary_state_min_score or margin < self.binary_state_min_margin:
            return None, float(best_score)
        return int(best_digit), float(best_score)

    def _detect_specialty_active(self, frame: np.ndarray) -> tuple[bool, float]:
        """Detect specialty activation from a strict orange pixel ratio."""
        crop = self._crop_region(frame, "specialty_equipment")
        if crop is None or crop.size == 0:
            return False, 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower = np.array(
            [
                self.specialty_orange_hue_min,
                self.specialty_orange_saturation_min,
                self.specialty_orange_value_min,
            ],
            dtype=np.uint8,
        )
        upper = np.array(
            [
                self.specialty_orange_hue_max,
                255,
                255,
            ],
            dtype=np.uint8,
        )
        mask = cv2.inRange(hsv, lower, upper)
        orange_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        return orange_ratio >= self.specialty_orange_ratio_threshold, orange_ratio

    def _crop_region(self, frame: np.ndarray, region_name: str) -> np.ndarray | None:
        """Crop one utility subregion from the frame."""
        if region_name not in self.absolute_regions:
            return None
        region = self._resolve_bounds(frame.shape, self.absolute_regions[region_name])
        return frame[region.y : region.y + region.height, region.x : region.x + region.width]

    @staticmethod
    def _resolve_bounds(
        frame_shape: tuple[int, ...],
        relative_region: dict,
        parent: PixelBounds | None = None,
    ) -> PixelBounds:
        """Convert a normalized region into absolute pixel bounds."""
        frame_height, frame_width = frame_shape[:2]
        if parent is None:
            x = int(relative_region.get("x", 0.0) * frame_width)
            y = int(relative_region.get("y", 0.0) * frame_height)
            width = int(relative_region.get("width", 0.0) * frame_width)
            height = int(relative_region.get("height", 0.0) * frame_height)
            return PixelBounds(x=x, y=y, width=width, height=height)

        x = parent.x + int(relative_region.get("x", 0.0) * parent.width)
        y = parent.y + int(relative_region.get("y", 0.0) * parent.height)
        width = int(relative_region.get("width", 0.0) * parent.width)
        height = int(relative_region.get("height", 0.0) * parent.height)
        return PixelBounds(x=x, y=y, width=width, height=height)

    def _prepare_binary(self, crop: np.ndarray) -> np.ndarray:
        """Normalize a tiny number crop into a binary digit image."""
        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grayscale, self.binary_state_template_size, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary == 255) > 0.55:
            binary = cv2.bitwise_not(binary)
        return (binary > 127).astype(np.uint8)

    def _shape_scores(self, binary: np.ndarray) -> dict[str, float]:
        """Score how strongly the crop resembles a 0-shape or 1-shape."""
        features = self._extract_shape_features(binary)
        if features is None:
            return {"0": 0.0, "1": 0.0}

        zero_score = 0.0
        one_score = 0.0

        if features["hole_area_ratio"] >= self.shape_hole_strong_area_ratio:
            zero_score += 0.78
            one_score -= 0.18
        elif features["hole_area_ratio"] >= self.shape_hole_min_area_ratio:
            zero_score += 0.62
            one_score -= 0.10
        else:
            one_score += 0.06
        if self.shape_width_ratio_min_for_zero <= features["width_ratio"] <= 0.85:
            zero_score += 0.18
        if self.shape_fill_ratio_min_for_zero <= features["fill_ratio"] <= self.shape_fill_ratio_max_for_zero:
            zero_score += 0.12
        if features["side_strength"] >= features["center_strength"] * 0.75:
            zero_score += 0.10

        if features["width_ratio"] <= self.shape_width_ratio_max_for_one:
            one_score += 0.20
        if features["center_strength"] >= self.shape_center_min_for_one:
            one_score += 0.42
        if features["side_strength"] <= self.shape_side_max_for_one:
            one_score += 0.22
        if features["hole_area_ratio"] <= 0.01:
            one_score += 0.08
        if features["fill_ratio"] <= 0.45:
            one_score += 0.08

        return {
            "0": min(1.0, max(0.0, zero_score)),
            "1": min(1.0, max(0.0, one_score)),
        }

    def _extract_shape_features(self, binary: np.ndarray) -> dict[str, Any] | None:
        """Measure simple structural features from the binary digit shape."""
        coords = np.argwhere(binary > 0)
        if coords.size == 0:
            return None

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        bbox = binary[y0 : y1 + 1, x0 : x1 + 1]
        bbox_h, bbox_w = bbox.shape
        if bbox_h <= 0 or bbox_w <= 0:
            return None

        fill_ratio = float(bbox.mean())
        width_ratio = float(bbox_w) / float(max(1, bbox_h))

        col_profile = bbox.mean(axis=0)
        third = max(1, bbox_w // 3)
        left_strength = float(col_profile[:third].mean())
        center_slice = col_profile[max(0, (bbox_w // 2) - 1) : min(bbox_w, (bbox_w // 2) + 2)]
        center_strength = float(center_slice.mean()) if center_slice.size else float(col_profile.mean())
        right_strength = float(col_profile[-third:].mean())
        side_strength = (left_strength + right_strength) / 2.0

        hole_area_ratio = self._estimate_hole_area_ratio(bbox)
        return {
            "fill_ratio": fill_ratio,
            "width_ratio": width_ratio,
            "center_strength": center_strength,
            "side_strength": side_strength,
            "hole_area_ratio": hole_area_ratio,
        }

    @staticmethod
    def _estimate_hole_area_ratio(bbox: np.ndarray) -> float:
        """Estimate whether the shape contains an interior hole like a 0."""
        mask = (bbox.astype(np.uint8) * 255)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or not contours:
            return 0.0
        total_area = float(mask.shape[0] * mask.shape[1])
        best_child_area = 0.0
        for index, contour in enumerate(contours):
            parent_index = hierarchy[0][index][3]
            if parent_index >= 0:
                best_child_area = max(best_child_area, float(cv2.contourArea(contour)))
        if total_area <= 0.0:
            return 0.0
        return best_child_area / total_area

    @staticmethod
    def _average_confidence(*values: float) -> float | None:
        """Return the average of non-zero confidence values."""
        usable = [value for value in values if value > 0.0]
        if not usable:
            return None
        return float(sum(usable) / len(usable))
