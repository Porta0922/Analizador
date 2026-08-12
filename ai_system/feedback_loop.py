import logging
from datetime import datetime
from typing import List, Optional

from .database import Database
from .models import AnalysisResult, UserFeedback, Correction, LearnedPattern
from .config import LEARNING_RATE, MIN_CORRECTIONS_FOR_PATTERN, PATTERN_CONFIDENCE_THRESHOLD

logger = logging.getLogger("ai-system")

# All pattern types that represent individual field corrections
FIELD_PATTERN_TYPES = [
    "name_format", "document_type", "date_format",
    "number_format", "gender_format", "common_error",
]

# All pattern types that feed into success-rate tracking
TRACKED_PATTERN_TYPES = FIELD_PATTERN_TYPES + ["field_regex", "tampering_sign"]


class FeedbackLoop:
    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_analysis(self, result: AnalysisResult) -> int:
        """Persist an analysis result and return its DB id."""
        analysis_id = self.db.save_analysis(result)
        logger.info("[FEEDBACK] Analysis recorded: id=%d", analysis_id)
        return analysis_id

    def record_feedback(self, analysis_id: int, feedback: UserFeedback):
        """
        Persist user feedback, run inline learning, and capture an
        accuracy snapshot so the trend graph stays up-to-date.
        """
        self.db.save_feedback(analysis_id, feedback)
        logger.info("[FEEDBACK] Feedback recorded for analysis %d: confirmed=%s",
                    analysis_id, feedback.confirmed)

        if feedback.corrections:
            self._learn_from_corrections(analysis_id, feedback.corrections)

        self._update_pattern_success_rates(analysis_id, feedback.confirmed)

        # Persist accuracy snapshot after every feedback event (Phase 4)
        try:
            self.db.save_accuracy_snapshot(trigger_event="feedback")
        except Exception as exc:
            logger.warning("[FEEDBACK] Could not save accuracy snapshot: %s", exc)

    # ------------------------------------------------------------------
    # Internal learning logic
    # ------------------------------------------------------------------

    def _learn_from_corrections(self, analysis_id: int, corrections: List[Correction]):
        """Create or reinforce learned patterns from explicit user corrections."""
        for correction in corrections:
            error_type = self._classify_error(correction)
            existing = self.db.find_similar_pattern(error_type, correction)

            if existing:
                existing.times_applied += 1
                delta = LEARNING_RATE if correction.was_correct else -LEARNING_RATE
                existing.confidence = max(0.0, min(1.0, existing.confidence + delta))
                self.db.update_pattern(existing)
                logger.info("[LEARNING] Updated pattern id=%d type=%s confidence=%.2f",
                            existing.id, existing.pattern_type, existing.confidence)
            else:
                new_pattern = self._create_pattern_from_correction(correction)
                if new_pattern:
                    self.db.save_pattern(new_pattern)
                    logger.info("[LEARNING] New pattern created: type=%s field=%s",
                                new_pattern.pattern_type, correction.field_name)

    def _classify_error(self, correction: Correction) -> str:
        """Map a correction's field name to a pattern type stored in the DB."""
        field = correction.field_name.lower()
        if "fecha" in field or "date" in field:
            return "date_format"
        if "numero" in field or "num" in field or "doc" in field:
            return "number_format"
        if "nombre" in field or "apellido" in field or "name" in field:
            return "name_format"
        if "tipo" in field or "type" in field:
            return "document_type"
        if "sexo" in field or "gender" in field:
            return "gender_format"
        return "common_error"

    def _create_pattern_from_correction(self, correction: Correction) -> Optional[LearnedPattern]:
        """
        Create a new LearnedPattern only when we have accumulated enough
        similar corrections (MIN_CORRECTIONS_FOR_PATTERN threshold).
        """
        similar_count = self._count_similar_corrections(correction)
        if similar_count < MIN_CORRECTIONS_FOR_PATTERN:
            return None

        # Build a human-readable description used in prompt injection
        if correction.was_correct:
            desc = (f"'{correction.expected_value}' es un valor válido para el campo "
                    f"'{correction.field_name}'.")
        else:
            desc = (f"El campo '{correction.field_name}' fue corregido de "
                    f"'{correction.extracted_value}' a '{correction.expected_value}'.")

        pattern_data = {
            "field_name": correction.field_name,
            "expected_pattern": correction.expected_value,
            "extracted_pattern": correction.extracted_value,
            "description": desc,
            "correction_count": similar_count,
        }

        return LearnedPattern(
            pattern_type=self._classify_error(correction),
            pattern_data=pattern_data,
            confidence=0.3 if correction.was_correct else 0.1,
            times_applied=1,
            success_rate=1.0 if correction.was_correct else 0.0,
        )

    def _count_similar_corrections(self, correction: Correction) -> int:
        """Count existing corrections that match the same field + expected value."""
        all_corrections = self.db.get_all_corrections()
        return sum(
            1 for c in all_corrections
            if (c["field_name"] == correction.field_name
                and c["expected_value"] == correction.expected_value)
        )

    def _update_pattern_success_rates(self, analysis_id: int, was_confirmed: bool):
        """
        Apply exponential moving average to pattern success_rate for all
        patterns that were active at the time of this analysis.
        """
        analysis = self.db.get_analysis(analysis_id)
        if not analysis:
            return

        outcome = 1.0 if was_confirmed else 0.0
        for pattern_type in TRACKED_PATTERN_TYPES:
            for pattern in self.db.get_patterns_by_type(pattern_type):
                pattern.times_applied += 1
                pattern.success_rate = pattern.success_rate * 0.9 + outcome * 0.1
                self.db.update_pattern(pattern)

    # ------------------------------------------------------------------
    # Stats (Phase 1 — expanded response schema)
    # ------------------------------------------------------------------

    def get_learning_stats(self) -> dict:
        """
        Return the full telemetry payload consumed by /ai/stats and the
        extension dashboard widget.

        Schema:
          total_analyses         int
          confirmed_analyses     int
          rejected_analyses      int
          total_corrections      int
          learned_patterns       int
          accuracy_rate          float  (0–1)
          accuracy_pct           float  (0–100, rounded 1dp)
          patterns_by_type       dict[str, {count, avg_confidence, avg_success_rate}]
          top_error_fields       list[{field_name, error_count, error_rate_pct}]
          active_patterns        list[{id, pattern_type, description, confidence}]
          accuracy_trend         list[{snapshot_at, accuracy_pct, active_patterns}]
          learning_rate          float
          min_corrections_for_pattern  int
        """
        stats = self.db.get_statistics()

        # ---- Patterns by type ----
        all_types = TRACKED_PATTERN_TYPES
        patterns_by_type = {}
        for ptype in all_types:
            pts = self.db.get_patterns_by_type(ptype)
            if pts:
                patterns_by_type[ptype] = {
                    "count": len(pts),
                    "avg_confidence": round(
                        sum(p.confidence for p in pts) / len(pts), 3),
                    "avg_success_rate": round(
                        sum(p.success_rate for p in pts) / len(pts), 3),
                }

        # ---- Top error fields (Phase 1) ----
        top_error_fields = self.db.get_top_error_fields(limit=5)

        # ---- Active injected patterns (Phase 1) ----
        active_patterns = self.db.get_active_patterns_summary(
            min_confidence=PATTERN_CONFIDENCE_THRESHOLD, limit=10
        )

        # ---- Accuracy trend (Phase 4) ----
        accuracy_trend = self.db.get_accuracy_trend(limit=20)

        return {
            **stats,
            "accuracy_pct": round(stats["accuracy_rate"] * 100, 1),
            "patterns_by_type": patterns_by_type,
            "top_error_fields": top_error_fields,
            "active_patterns": active_patterns,
            "accuracy_trend": accuracy_trend,
            "learning_rate": LEARNING_RATE,
            "min_corrections_for_pattern": MIN_CORRECTIONS_FOR_PATTERN,
        }
