"""Hardpoint extraction scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.models import FramePacket, HardpointSnapshot, MatchContext, ModuleStatus
from app.modules.base import PipelineModule
from app.scheduler import FixedRateScheduler


@dataclass(slots=True)
class PixelBounds:
    """Absolute integer bounds for one hardpoint subregion."""

    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class DigitShapeFeatures:
    """Simple structural features for one segmented score digit."""

    hole_count: int
    top_mass: float
    bottom_mass: float
    left_mass: float
    right_mass: float
    center_drift: float


class HardpointModule(PipelineModule):
    """Validated Hardpoint score and hill extraction only."""

    def __init__(self, config: dict, scheduling_config: dict) -> None:
        super().__init__(name="hardpoint")
        self.config = config
        self.enabled = bool(config.get("enabled", True))
        self.requires_registration = bool(config.get("requires_registration", False))
        self.scheduler = FixedRateScheduler(scheduling_config.get("samples_per_second", 2.0))
        self.snapshot = HardpointSnapshot()
        self.absolute_regions = dict(config.get("absolute_regions", {}))
        self.score_min = int(config.get("score_min", 0))
        self.score_max = int(config.get("score_max", 250))
        self.hill_min = int(config.get("hill_min", 1))
        self.hill_max = int(config.get("hill_max", 5))
        self.score_min_confidence = float(config.get("score_min_confidence", 0.75))
        self.single_digit_score_min_confidence = float(
            config.get("single_digit_score_min_confidence", 0.80)
        )
        self.hill_min_confidence = float(config.get("hill_min_confidence", 0.80))
        self.score_max_delta_from_last = int(config.get("score_max_delta_from_last", 15))
        self.score_thickness_min_ratio = float(config.get("score_thickness_min_ratio", 0.08))
        self.score_thickness_max_ratio = float(config.get("score_thickness_max_ratio", 0.55))
        self.upscale_factor = float(config.get("upscale_factor", 4.0))
        self.score_left_mask_pixels = int(config.get("score_left_mask_pixels", 2))
        self.score_threshold_bias = float(config.get("score_threshold_bias", 18.0))
        self.template_size = tuple(int(v) for v in config.get("score_template_size", [18, 24]))
        self.score_digit_min_similarity = float(config.get("score_digit_min_similarity", 0.50))
        self.score_digit_candidates_per_segment = int(
            config.get("score_digit_candidates_per_segment", 2)
        )
        self.score_number_min_confidence = float(config.get("score_number_min_confidence", 0.60))
        self.score_segment_min_fill_ratio = float(config.get("score_segment_min_fill_ratio", 0.10))
        self.score_leading_artifact_max_width_ratio = float(
            config.get("score_leading_artifact_max_width_ratio", 0.20)
        )
        self.score_leading_artifact_max_confidence = float(
            config.get("score_leading_artifact_max_confidence", 0.55)
        )
        self.score_template_root = Path(
            config.get(
                "score_template_root",
                Path(__file__).resolve().parent.parent.parent / "assets" / "score_digit_templates",
            )
        )
        self._reader = None
        self._digit_templates = self._load_digit_templates()
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

        blue_score, blue_confidence = self._read_score(
            frame_packet.image,
            "blue_score",
            self.snapshot.blue_score,
        )
        red_score, red_confidence = self._read_score(
            frame_packet.image,
            "red_score",
            self.snapshot.red_score,
        )
        hill_number, hill_confidence = self._read_hill(frame_packet.image, "hill_number")

        if blue_score is not None:
            self.snapshot.blue_score = blue_score
        if red_score is not None:
            self.snapshot.red_score = red_score
        if hill_number is not None:
            self.snapshot.current_hill = hill_number
        self._last_confidence = self._average_confidence(
            blue_confidence,
            red_confidence,
            hill_confidence,
        )

    @property
    def last_confidence(self) -> float | None:
        """Expose the most recent Hardpoint OCR confidence."""
        return self._last_confidence

    def _ensure_reader(self):
        """Create the OCR reader only when the module is actually used."""
        if self._reader is not None:
            return self._reader
        import easyocr

        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        return self._reader

    def _read_score(
        self,
        frame: np.ndarray,
        region_name: str,
        last_value: int | None,
    ) -> tuple[int | None, float]:
        """Read one score box and restrict it to 0-250."""
        crop = self._crop_region(frame, region_name)
        if crop is None or crop.size == 0:
            return None, 0.0
        prepared = self._prepare_crop(crop)
        if not self._passes_score_thickness(prepared):
            return None, 0.0
        candidate, confidence = self._read_score_from_templates(prepared)
        if candidate is None:
            return None, 0.0
        if len(candidate) > 3:
            return None, float(confidence)
        value = int(candidate)
        if value < self.score_min or value > self.score_max:
            return None, float(confidence)
        required_confidence = self.score_min_confidence
        if len(candidate) == 1:
            required_confidence = self.single_digit_score_min_confidence
        if float(confidence) < required_confidence:
            return None, float(confidence)
        if (
            last_value is not None
            and self.score_max_delta_from_last > 0
            and abs(value - last_value) >= self.score_max_delta_from_last
        ):
            return None, float(confidence)
        return value, float(confidence)

    def _read_hill(self, frame: np.ndarray, region_name: str) -> tuple[int | None, float]:
        """Read one hill number box and restrict it to 1-5."""
        crop = self._crop_region(frame, region_name)
        if crop is None or crop.size == 0:
            return None, 0.0
        prepared = self._prepare_crop(crop)
        reader = self._ensure_reader()
        results = reader.readtext(prepared, detail=1, allowlist="12345")
        candidate, confidence = self._best_numeric_candidate(results)
        if candidate is None:
            return None, 0.0
        if len(candidate) != 1:
            return None, float(confidence)
        value = int(candidate)
        if value < self.hill_min or value > self.hill_max:
            return None, float(confidence)
        if float(confidence) < self.hill_min_confidence:
            return None, float(confidence)
        return value, float(confidence)

    def _crop_region(self, frame: np.ndarray, region_name: str) -> np.ndarray | None:
        """Crop one hardpoint subregion from the frame."""
        if region_name not in self.absolute_regions:
            return None
        region = self._resolve_bounds(frame.shape, self.absolute_regions[region_name])
        return frame[region.y : region.y + region.height, region.x : region.x + region.width]

    @staticmethod
    def _resolve_bounds(frame_shape: tuple[int, ...], relative_region: dict) -> PixelBounds:
        """Convert a normalized region into absolute pixel bounds."""
        frame_height, frame_width = frame_shape[:2]
        x = int(relative_region.get("x", 0.0) * frame_width)
        y = int(relative_region.get("y", 0.0) * frame_height)
        width = int(relative_region.get("width", 0.0) * frame_width)
        height = int(relative_region.get("height", 0.0) * frame_height)
        return PixelBounds(x=x, y=y, width=width, height=height)

    def _prepare_crop(self, crop: np.ndarray) -> np.ndarray:
        """Threshold a small HUD crop for numeric OCR without score upscaling."""
        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
        otsu_threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        strict_threshold = max(0.0, min(255.0, float(otsu_threshold) + self.score_threshold_bias))
        _, binary = cv2.threshold(blurred, strict_threshold, 255, cv2.THRESH_BINARY)
        if self.score_left_mask_pixels > 0 and binary.shape[1] > self.score_left_mask_pixels:
            binary[:, : self.score_left_mask_pixels] = 0
        return binary

    def _load_digit_templates(self) -> dict[str, list[np.ndarray]]:
        """Load real digit reference crops from the asset folders."""
        templates: dict[str, list[np.ndarray]] = {}
        for digit in "0123456789":
            digit_dir = self.score_template_root / digit
            if not digit_dir.exists():
                continue
            variants: list[np.ndarray] = []
            for path in sorted(digit_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                variants.append(self._prepare_template_image(image))
            if variants:
                templates[digit] = variants
        return templates

    def _prepare_template_image(self, grayscale: np.ndarray) -> np.ndarray:
        """Normalize one reference digit into the matcher template space."""
        resized = cv2.resize(grayscale, self.template_size, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary == 255) > 0.55:
            binary = cv2.bitwise_not(binary)
        return (binary > 127).astype(np.uint8)

    def _read_score_from_templates(self, binary: np.ndarray) -> tuple[str | None, float]:
        """Read a full 1-3 digit score by segmenting and matching digit templates."""
        if not self._digit_templates:
            return None, 0.0
        digit_images = self._segment_score_digits(binary)
        if not digit_images or len(digit_images) > 3:
            return None, 0.0

        digit_images = self._drop_leading_artifact_segment(digit_images)
        if not digit_images or len(digit_images) > 3:
            return None, 0.0

        candidate, confidence = self._choose_best_score_candidate(digit_images)
        if candidate is None:
            return None, 0.0
        if confidence < self.score_number_min_confidence:
            return None, confidence
        return candidate, confidence

    def _choose_best_score_candidate(self, digit_images: list[np.ndarray]) -> tuple[str | None, float]:
        """Evaluate multiple plausible digits per segment and keep the strongest whole-number match."""
        best_digits: str | None = None
        best_confidence = 0.0

        def search(index: int, chosen_digits: list[str], confidences: list[float]) -> None:
            nonlocal best_digits, best_confidence
            if index >= len(digit_images):
                if not chosen_digits:
                    return
                overall_confidence = float(sum(confidences) / len(confidences))
                candidate_digits = "".join(chosen_digits)
                if overall_confidence > best_confidence:
                    best_digits = candidate_digits
                    best_confidence = overall_confidence
                return

            allowed_digits = self._allowed_score_digits(
                digit_count=len(digit_images),
                index=index,
                chosen_digits=chosen_digits,
            )
            ranked_candidates = self._rank_digit_templates(digit_images[index], allowed_digits)
            if not ranked_candidates:
                return
            for digit, confidence in ranked_candidates[: self.score_digit_candidates_per_segment]:
                if confidence < self.score_digit_min_similarity:
                    continue
                chosen_digits.append(digit)
                confidences.append(confidence)
                search(index + 1, chosen_digits, confidences)
                confidences.pop()
                chosen_digits.pop()

        search(0, [], [])
        return best_digits, best_confidence

    def _drop_leading_artifact_segment(self, digit_images: list[np.ndarray]) -> list[np.ndarray]:
        """Ignore a very thin weak first segment when it looks like a border artifact."""
        if len(digit_images) < 3:
            return digit_images
        first = digit_images[0]
        fill_ratio = float(first.mean())
        width_ratio = self._digit_foreground_width_ratio(first)
        matched_digit, matched_score = self._match_digit_template(first, allowed_digits=None)
        del matched_digit
        if (
            fill_ratio < self.score_segment_min_fill_ratio
            or (
                width_ratio <= self.score_leading_artifact_max_width_ratio
                and matched_score <= self.score_leading_artifact_max_confidence
            )
        ):
            return digit_images[1:]
        return digit_images

    @staticmethod
    def _allowed_score_digits(
        digit_count: int,
        index: int,
        chosen_digits: list[str],
    ) -> set[str]:
        """Return the allowed digit set for a score position under the 0-250 range."""
        if digit_count <= 1:
            return set("0123456789")
        if digit_count == 2:
            if index == 0:
                return set("123456789")
            return set("0123456789")
        if index == 0:
            return {"1", "2"}
        if index == 1:
            if chosen_digits and chosen_digits[0] == "2":
                return set("012345")
            return set("0123456789")
        if index == 2:
            if len(chosen_digits) >= 2 and chosen_digits[0] == "2" and chosen_digits[1] == "5":
                return {"0"}
            return set("0123456789")
        return set("0123456789")

    def _segment_score_digits(self, binary: np.ndarray) -> list[np.ndarray]:
        """Split a score crop into up to three digit images using filtered connected components."""
        foreground = (binary > 0).astype(np.uint8)
        height, width = foreground.shape
        if height <= 0 or width <= 0:
            return []
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        candidates: list[tuple[int, int, int, int, int]] = []
        min_area = max(6, int(round(height * width * 0.015)))
        min_height = max(4, int(round(height * 0.35)))
        min_width = max(2, int(round(width * 0.04)))
        for label in range(1, component_count):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            touches_left = x <= 0
            touches_bottom = (y + h) >= (height - 1)
            if area < min_area or h < min_height or w < min_width:
                continue
            # Reject likely HUD border artifacts hugging the left or bottom edges.
            if touches_left and (w / max(1, h)) < 0.18:
                continue
            if touches_bottom and (h / max(1, height)) < 0.22:
                continue
            candidates.append((x, y, w, h, area))
        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 3:
            candidates = sorted(candidates, key=lambda item: item[4], reverse=True)[:3]
            candidates.sort(key=lambda item: item[0])

        digit_images: list[np.ndarray] = []
        for x, y, w, h, _ in candidates:
            digit = foreground[y : y + h, x : x + w]
            digit = cv2.copyMakeBorder(digit, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            digit_resized = cv2.resize(
                digit.astype(np.uint8),
                self.template_size,
                interpolation=cv2.INTER_NEAREST,
            )
            digit_images.append((digit_resized > 0).astype(np.uint8))
        return digit_images

    def _match_digit_template(
        self,
        digit_image: np.ndarray,
        allowed_digits: set[str] | None = None,
    ) -> tuple[str | None, float]:
        """Return the best matching template digit and its similarity."""
        ranked = self._rank_digit_templates(digit_image, allowed_digits)
        if not ranked:
            return None, 0.0
        return ranked[0]

    def _rank_digit_templates(
        self,
        digit_image: np.ndarray,
        allowed_digits: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ranked digit candidates for one segment."""
        features = self._extract_digit_shape_features(digit_image)
        digit_scores: list[tuple[str, float]] = []
        for digit, variants in self._digit_templates.items():
            if allowed_digits is not None and digit not in allowed_digits:
                continue
            digit_best_score = 0.0
            for template in variants:
                score = self._template_similarity(digit_image, template)
                if score > digit_best_score:
                    digit_best_score = score
            if digit_best_score > 0.0:
                adjusted_score = digit_best_score + self._shape_feature_adjustment(digit, features)
                digit_scores.append((digit, adjusted_score))
        digit_scores.sort(key=lambda item: item[1], reverse=True)
        return digit_scores

    @staticmethod
    def _extract_digit_shape_features(digit_image: np.ndarray) -> DigitShapeFeatures:
        """Measure simple shape cues from one binary digit image."""
        binary = (digit_image > 0).astype(np.uint8)
        height, width = binary.shape
        top_half = binary[: max(1, height // 2), :]
        bottom_half = binary[max(1, height // 2) :, :]
        left_half = binary[:, : max(1, width // 2)]
        right_half = binary[:, max(1, width // 2) :]

        total_foreground = float(np.count_nonzero(binary))
        if total_foreground <= 0.0:
            return DigitShapeFeatures(
                hole_count=0,
                top_mass=0.0,
                bottom_mass=0.0,
                left_mass=0.0,
                right_mass=0.0,
                center_drift=0.0,
            )

        top_mass = float(np.count_nonzero(top_half)) / total_foreground
        bottom_mass = float(np.count_nonzero(bottom_half)) / total_foreground
        left_mass = float(np.count_nonzero(left_half)) / total_foreground
        right_mass = float(np.count_nonzero(right_half)) / total_foreground
        center_drift = HardpointModule._center_drift(binary)
        hole_count = HardpointModule._count_digit_holes(binary)
        return DigitShapeFeatures(
            hole_count=hole_count,
            top_mass=top_mass,
            bottom_mass=bottom_mass,
            left_mass=left_mass,
            right_mass=right_mass,
            center_drift=center_drift,
        )

    @staticmethod
    def _center_drift(binary: np.ndarray) -> float:
        """Measure horizontal center drift from the top half to the bottom half."""
        height, width = binary.shape
        split = max(1, height // 2)
        top = binary[:split, :]
        bottom = binary[split:, :]
        top_center = HardpointModule._mean_foreground_x(top)
        bottom_center = HardpointModule._mean_foreground_x(bottom)
        if top_center is None or bottom_center is None or width <= 0:
            return 0.0
        return float(bottom_center - top_center) / float(width)

    @staticmethod
    def _mean_foreground_x(binary: np.ndarray) -> float | None:
        """Return the mean x-position of foreground pixels in one binary region."""
        coords = np.argwhere(binary > 0)
        if coords.size == 0:
            return None
        return float(coords[:, 1].mean())

    @staticmethod
    def _count_digit_holes(binary: np.ndarray) -> int:
        """Count meaningful enclosed holes inside a digit shape."""
        image = (binary * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or not contours:
            return 0
        hole_count = 0
        for index, contour in enumerate(contours):
            parent = hierarchy[0][index][3]
            if parent < 0:
                continue
            area = cv2.contourArea(contour)
            if area >= 4.0:
                hole_count += 1
        return hole_count

    @staticmethod
    def _shape_feature_adjustment(digit: str, features: DigitShapeFeatures) -> float:
        """Apply small shape-based bonuses and penalties on top of template similarity."""
        adjustment = 0.0
        has_hole = features.hole_count > 0
        top_minus_bottom = features.top_mass - features.bottom_mass
        bottom_minus_top = features.bottom_mass - features.top_mass
        right_minus_left = features.right_mass - features.left_mass
        left_minus_right = features.left_mass - features.right_mass
        drift = features.center_drift

        if has_hole:
            if digit in {"0", "6", "8", "9"}:
                adjustment += 0.055
            elif digit in {"5", "7", "1", "3"}:
                adjustment -= 0.045
        else:
            if digit in {"5", "7", "1", "3", "2", "4"}:
                adjustment += 0.015
            elif digit in {"0", "6", "8", "9"}:
                adjustment -= 0.02

        if bottom_minus_top > 0.08:
            if digit == "6":
                adjustment += 0.045
            if digit == "5":
                adjustment -= 0.04
        if top_minus_bottom > 0.08:
            if digit == "5":
                adjustment += 0.03
            if digit == "6":
                adjustment -= 0.03

        if left_minus_right > 0.08:
            if digit == "6":
                adjustment += 0.03
            if digit == "5":
                adjustment += 0.015
        if right_minus_left > 0.10 and not has_hole:
            if digit == "3":
                adjustment += 0.045
            if digit == "5":
                adjustment -= 0.02

        if abs(drift) <= 0.035:
            if digit == "1":
                adjustment += 0.05
            if digit == "7":
                adjustment -= 0.025
        if drift <= -0.045:
            if digit == "7":
                adjustment += 0.055
            if digit == "1":
                adjustment -= 0.035

        return adjustment

    @staticmethod
    def _digit_foreground_width_ratio(digit_image: np.ndarray) -> float:
        """Return the foreground bounding-box width/height ratio for one segmented digit."""
        coords = np.argwhere(digit_image > 0)
        if coords.size == 0:
            return 0.0
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        width = float((x1 - x0) + 1)
        height = float((y1 - y0) + 1)
        if height <= 0.0:
            return 0.0
        return width / height

    @staticmethod
    def _template_iou(observed: np.ndarray, template: np.ndarray) -> float:
        """Measure overlap between an observed digit and a reference template."""
        intersection = float(np.logical_and(observed > 0, template > 0).sum())
        union = float(np.logical_or(observed > 0, template > 0).sum())
        if union <= 0.0:
            return 0.0
        return intersection / union

    def _template_similarity(self, observed: np.ndarray, template: np.ndarray) -> float:
        """Combine overlap and projection-profile similarity for a more tolerant digit match."""
        iou = self._template_iou(observed, template)
        column_similarity = self._profile_similarity(
            observed.mean(axis=0),
            template.mean(axis=0),
        )
        row_similarity = self._profile_similarity(
            observed.mean(axis=1),
            template.mean(axis=1),
        )
        return (0.45 * iou) + (0.35 * column_similarity) + (0.20 * row_similarity)

    @staticmethod
    def _profile_similarity(observed_profile: np.ndarray, template_profile: np.ndarray) -> float:
        """Return a simple 0-1 similarity score between two normalized projection profiles."""
        if observed_profile.size != template_profile.size or observed_profile.size == 0:
            return 0.0
        difference = np.abs(observed_profile.astype(np.float32) - template_profile.astype(np.float32))
        score = 1.0 - float(np.mean(difference))
        return max(0.0, min(1.0, score))

    @staticmethod
    def _best_numeric_candidate(results: list) -> tuple[str | None, float]:
        """Return the strongest digits-only OCR candidate."""
        best_text: str | None = None
        best_confidence = 0.0
        for _, text, confidence in results:
            digits = "".join(ch for ch in str(text) if ch.isdigit())
            if not digits:
                continue
            if float(confidence) > best_confidence:
                best_text = digits
                best_confidence = float(confidence)
        return best_text, best_confidence

    def _passes_score_thickness(self, binary: np.ndarray) -> bool:
        """Reject score crops that are too thin or too thick overall."""
        if binary.size == 0:
            return False
        foreground_ratio = float(np.count_nonzero(binary)) / float(binary.size)
        return self.score_thickness_min_ratio <= foreground_ratio <= self.score_thickness_max_ratio

    @staticmethod
    def _average_confidence(*values: float) -> float | None:
        """Return the average of non-zero confidence values."""
        usable = [value for value in values if value > 0.0]
        if not usable:
            return None
        return float(sum(usable) / len(usable))
