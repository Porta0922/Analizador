from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import json


@dataclass
class AIAnalysis:
    coherence_score: float  # 0-100
    coherence_issues: List[str] = field(default_factory=list)
    tampering_score: float = 100.0  # 100 = no tampering
    tampering_areas: List[str] = field(default_factory=list)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    overall_confidence: float = 0.0
    reasoning: str = ""
    
    def to_dict(self) -> dict:
        return {
            "coherence_score": self.coherence_score,
            "coherence_issues": self.coherence_issues,
            "tampering_score": self.tampering_score,
            "tampering_areas": self.tampering_areas,
            "extracted_data": self.extracted_data,
            "overall_confidence": self.overall_confidence,
            "reasoning": self.reasoning
        }


@dataclass
class Correction:
    field_name: str
    expected_value: str
    extracted_value: str
    was_correct: bool
    confidence_at_time: float = 0.0
    corrected_at: datetime = field(default_factory=datetime.now)


@dataclass
class UserFeedback:
    confirmed: bool
    corrections: List[Correction] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "confirmed": self.confirmed,
            "corrections": [
                {
                    "field_name": c.field_name,
                    "expected_value": c.expected_value,
                    "extracted_value": c.extracted_value,
                    "was_correct": c.was_correct,
                    "confidence_at_time": c.confidence_at_time,
                    "corrected_at": c.corrected_at.isoformat() if isinstance(c.corrected_at, datetime) else str(c.corrected_at),
                }
                for c in self.corrections
            ],
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
        }


@dataclass
class AnalysisResult:
    analysis_id: Optional[int] = None
    selfie_hash: str = ""
    doc_front_hash: str = ""
    doc_back_hash: str = ""
    face_similarity: float = 0.0
    ocr_text: str = ""
    field_matches: Dict[str, bool] = field(default_factory=dict)
    duplicate_score: Optional[float] = None
    ai_analysis: Optional[AIAnalysis] = None
    user_feedback: Optional[UserFeedback] = None
    processing_time_ms: int = 0
    model_version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "analysis_id": self.analysis_id,
            "selfie_hash": self.selfie_hash,
            "doc_front_hash": self.doc_front_hash,
            "doc_back_hash": self.doc_back_hash,
            "face_similarity": self.face_similarity,
            "ocr_text": self.ocr_text,
            "field_matches": self.field_matches,
            "duplicate_score": self.duplicate_score,
            "ai_analysis": self.ai_analysis.to_dict() if self.ai_analysis else None,
            "user_feedback": self.user_feedback.to_dict() if self.user_feedback else None,
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class LearnedPattern:
    id: Optional[int] = None
    pattern_type: str = ""  # "field_regex", "common_error", "tampering_sign"
    pattern_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    times_applied: int = 0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "pattern_data": self.pattern_data,
            "confidence": self.confidence,
            "times_applied": self.times_applied,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
