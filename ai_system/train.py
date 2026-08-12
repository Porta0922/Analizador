import json
import logging
from datetime import datetime
from typing import List, Dict, Any

from .database import Database
from .models import LearnedPattern, Correction

logger = logging.getLogger("ai-system")


class AITrainer:
    def __init__(self, db: Database):
        self.db = db
    
    def train_patterns(self) -> List[LearnedPattern]:
        """Train new patterns based on accumulated data."""
        logger.info("[TRAIN] Starting pattern training")
        
        # Get all corrections
        corrections = self.db.get_all_corrections()
        
        if not corrections:
            logger.info("[TRAIN] No corrections found for training")
            return []
        
        # Identify patterns
        patterns = self._identify_patterns(corrections)
        
        # Create rules
        rules = self._create_rules(patterns)
        
        # Save to database
        saved_patterns = []
        for rule in rules:
            pattern_id = self.db.save_pattern(rule)
            rule.id = pattern_id
            saved_patterns.append(rule)
            logger.info("[TRAIN] Saved pattern: %s (confidence: %.2f)", 
                       rule.pattern_type, rule.confidence)
        
        logger.info("[TRAIN] Training complete: %d patterns created", len(saved_patterns))
        return saved_patterns
    
    def _identify_patterns(self, corrections: List[dict]) -> Dict[str, List[dict]]:
        """Identify patterns from corrections."""
        patterns = {}
        
        for correction in corrections:
            field_name = correction["field_name"]
            expected = correction["expected_value"]
            extracted = correction["extracted_value"]
            was_correct = correction["was_correct"]
            
            # Group by field name
            if field_name not in patterns:
                patterns[field_name] = []
            
            patterns[field_name].append({
                "expected": expected,
                "extracted": extracted,
                "was_correct": was_correct
            })
        
        return patterns
    
    def _create_rules(self, patterns: Dict[str, List[dict]]) -> List[LearnedPattern]:
        """Create rules from identified patterns."""
        rules = []
        
        for field_name, corrections in patterns.items():
            # Calculate success rate for this field
            correct_count = sum(1 for c in corrections if c["was_correct"])
            total_count = len(corrections)
            success_rate = correct_count / total_count if total_count > 0 else 0
            
            # Only create pattern if we have enough data
            if total_count >= 3:
                # Find common expected values
                expected_values = {}
                for c in corrections:
                    val = c["expected"]
                    if val not in expected_values:
                        expected_values[val] = {"count": 0, "correct": 0}
                    expected_values[val]["count"] += 1
                    if c["was_correct"]:
                        expected_values[val]["correct"] += 1
                
                # Create pattern for most common expected value
                if expected_values:
                    most_common = max(expected_values.items(), key=lambda x: x[1]["count"])
                    expected_val, stats = most_common
                    
                    pattern_data = {
                        "field_name": field_name,
                        "expected_pattern": expected_val,
                        "success_rate": stats["correct"] / stats["count"],
                        "occurrences": stats["count"],
                        "description": f"Pattern for {field_name}: {expected_val}"
                    }
                    
                    confidence = min(1.0, stats["count"] / 10)  # Scale confidence with occurrences
                    
                    rules.append(LearnedPattern(
                        pattern_type=self._get_pattern_type(field_name),
                        pattern_data=pattern_data,
                        confidence=confidence,
                        times_applied=stats["count"],
                        success_rate=stats["correct"] / stats["count"]
                    ))
        
        return rules
    
    def _get_pattern_type(self, field_name: str) -> str:
        """Determine pattern type based on field name."""
        field_lower = field_name.lower()
        
        if "fecha" in field_lower or "date" in field_lower:
            return "date_format"
        elif "numero" in field_lower or "num" in field_lower or "doc" in field_lower:
            return "number_format"
        elif "nombre" in field_lower or "apellido" in field_lower:
            return "name_format"
        elif "tipo" in field_lower:
            return "document_type"
        else:
            return "common_error"
    
    def generate_training_data(self) -> List[Dict[str, Any]]:
        """Generate training data for fine-tuning."""
        confirmed = self.db.get_confirmed_analyses()
        
        training_data = []
        for analysis in confirmed:
            training_data.append({
                "input": analysis.get("ocr_extracted", ""),
                "output": {
                    "coherence_score": analysis.get("ai_coherence_score", 0),
                    "tampering_score": analysis.get("ai_tampering_score", 0),
                    "extracted_data": json.loads(analysis.get("ai_extraction", "{}")),
                    "field_matches": json.loads(analysis.get("field_matches", "{}"))
                },
                "weight": 1.0 if analysis.get("user_confirmed") else 0.5
            })
        
        return training_data
    
    def evaluate_performance(self) -> Dict[str, Any]:
        """Evaluate system performance."""
        stats = self.db.get_statistics()
        
        # Calculate additional metrics
        recent_analyses = self.db.get_recent_analyses(days=30)
        
        if not recent_analyses:
            return {
                "total_analyses": 0,
                "accuracy_rate": 0,
                "corrections_received": 0,
                "avg_confidence": 0
            }
        
        # Calculate average confidence
        confidences = []
        for analysis in recent_analyses:
            if analysis.ai_analysis:
                confidences.append(analysis.ai_analysis.overall_confidence)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total_analyses": stats["total_analyses"],
            "accuracy_rate": stats["accuracy_rate"],
            "corrections_received": stats["total_corrections"],
            "learned_patterns": stats["learned_patterns"],
            "avg_confidence": round(avg_confidence, 2),
            "recent_analyses_count": len(recent_analyses)
        }
    
    def export_training_data(self, filepath: str):
        """Export training data to JSON file."""
        training_data = self.generate_training_data()
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.info("[TRAIN] Exported %d training samples to %s", len(training_data), filepath)
    
    def import_patterns(self, filepath: str):
        """Import patterns from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            patterns_data = json.load(f)
        
        imported_count = 0
        for pattern_data in patterns_data:
            pattern = LearnedPattern(
                pattern_type=pattern_data.get("pattern_type", "common_error"),
                pattern_data=pattern_data.get("pattern_data", {}),
                confidence=pattern_data.get("confidence", 0.5),
                times_applied=pattern_data.get("times_applied", 0),
                success_rate=pattern_data.get("success_rate", 0.5)
            )
            
            self.db.save_pattern(pattern)
            imported_count += 1
        
        logger.info("[TRAIN] Imported %d patterns from %s", imported_count, filepath)
