import logging
from datetime import datetime
from typing import List, Optional

from .database import Database
from .models import AnalysisResult, UserFeedback, Correction, LearnedPattern
from .config import LEARNING_RATE, MIN_CORRECTIONS_FOR_PATTERN, PATTERN_CONFIDENCE_THRESHOLD

logger = logging.getLogger("ai-system")


class FeedbackLoop:
    def __init__(self, db: Database):
        self.db = db
    
    def record_analysis(self, result: AnalysisResult) -> int:
        """Record an analysis result in the database."""
        analysis_id = self.db.save_analysis(result)
        logger.info("[FEEDBACK] Analysis recorded with ID: %d", analysis_id)
        return analysis_id
    
    def record_feedback(self, analysis_id: int, feedback: UserFeedback):
        """Record user feedback and trigger learning."""
        self.db.save_feedback(analysis_id, feedback)
        logger.info("[FEEDBACK] Feedback recorded for analysis %d: confirmed=%s", 
                    analysis_id, feedback.confirmed)
        
        # Learn from corrections if any
        if feedback.corrections:
            self._learn_from_corrections(analysis_id, feedback.corrections)
        
        # Update pattern success rates
        self._update_pattern_success_rates(analysis_id, feedback.confirmed)
    
    def _learn_from_corrections(self, analysis_id: int, corrections: List[Correction]):
        """Learn from user corrections to improve future analyses."""
        for correction in corrections:
            # Classify the error type
            error_type = self._classify_error(correction)
            
            # Find or create pattern
            existing_pattern = self.db.find_similar_pattern(error_type, correction)
            
            if existing_pattern:
                # Reinforce existing pattern
                existing_pattern.times_applied += 1
                if correction.was_correct:
                    existing_pattern.confidence = min(
                        1.0, 
                        existing_pattern.confidence + LEARNING_RATE
                    )
                else:
                    existing_pattern.confidence = max(
                        0.0,
                        existing_pattern.confidence - LEARNING_RATE
                    )
                self.db.update_pattern(existing_pattern)
                logger.info("[LEARNING] Updated pattern %d: confidence=%.2f", 
                          existing_pattern.id, existing_pattern.confidence)
            else:
                # Create new pattern if we have enough corrections
                new_pattern = self._create_pattern_from_correction(correction)
                if new_pattern:
                    self.db.save_pattern(new_pattern)
                    logger.info("[LEARNING] Created new pattern: %s", new_pattern.pattern_type)
    
    def _classify_error(self, correction: Correction) -> str:
        """Classify the type of error based on correction data."""
        field = correction.field_name.lower()
        
        # Date-related fields
        if "fecha" in field or "date" in field:
            return "date_format"
        
        # Number-related fields
        if "numero" in field or "num" in field or "doc" in field:
            return "number_format"
        
        # Name-related fields
        if "nombre" in field or "apellido" in field or "name" in field:
            return "name_format"
        
        # Document type
        if "tipo" in field or "type" in field:
            return "document_type"
        
        # Gender
        if "sexo" in field or "gender" in field:
            return "gender_format"
        
        # Default
        return "common_error"
    
    def _create_pattern_from_correction(self, correction: Correction) -> Optional[LearnedPattern]:
        """Create a new pattern from a correction if conditions are met."""
        # Check if we have enough similar corrections
        similar_count = self._count_similar_corrections(correction)
        
        if similar_count < MIN_CORRECTIONS_FOR_PATTERN:
            return None
        
        # Calculate initial confidence based on correction quality
        initial_confidence = 0.3 if correction.was_correct else 0.1
        
        pattern_data = {
            "field_name": correction.field_name,
            "expected_pattern": correction.expected_value,
            "extracted_pattern": correction.extracted_value,
            "description": f"Pattern for {correction.field_name}: {correction.expected_value}",
            "correction_count": similar_count
        }
        
        return LearnedPattern(
            pattern_type=self._classify_error(correction),
            pattern_data=pattern_data,
            confidence=initial_confidence,
            times_applied=1,
            success_rate=1.0 if correction.was_correct else 0.0
        )
    
    def _count_similar_corrections(self, correction: Correction) -> int:
        """Count how many similar corrections exist."""
        all_corrections = self.db.get_all_corrections()
        
        count = 0
        for c in all_corrections:
            if (c["field_name"] == correction.field_name and
                c["expected_value"] == correction.expected_value):
                count += 1
        
        return count
    
    def _update_pattern_success_rates(self, analysis_id: int, was_confirmed: bool):
        """Update success rates for all patterns applied in this analysis."""
        # Get analysis to find which patterns were used
        analysis = self.db.get_analysis(analysis_id)
        if not analysis:
            return
        
        # Update all patterns of relevant types
        for pattern_type in ["field_regex", "common_error", "tampering_sign"]:
            patterns = self.db.get_patterns_by_type(pattern_type)
            for pattern in patterns:
                pattern.times_applied += 1
                if was_confirmed:
                    # Calculate new success rate using exponential moving average
                    pattern.success_rate = (
                        pattern.success_rate * 0.9 + 1.0 * 0.1
                    )
                else:
                    pattern.success_rate = (
                        pattern.success_rate * 0.9 + 0.0 * 0.1
                    )
                self.db.update_pattern(pattern)
    
    def get_learning_stats(self) -> dict:
        """Get statistics about the learning process."""
        stats = self.db.get_statistics()
        
        # Get additional learning-specific stats
        patterns = {}
        for pattern_type in ["field_regex", "common_error", "tampering_sign"]:
            type_patterns = self.db.get_patterns_by_type(pattern_type)
            patterns[pattern_type] = {
                "count": len(type_patterns),
                "avg_confidence": sum(p.confidence for p in type_patterns) / len(type_patterns) if type_patterns else 0,
                "avg_success_rate": sum(p.success_rate for p in type_patterns) / len(type_patterns) if type_patterns else 0
            }
        
        return {
            **stats,
            "patterns_by_type": patterns,
            "learning_rate": LEARNING_RATE,
            "min_corrections_for_pattern": MIN_CORRECTIONS_FOR_PATTERN
        }
