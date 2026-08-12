import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager

from .config import DATABASE_PATH
from .models import AnalysisResult, UserFeedback, Correction, LearnedPattern


class Database:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_PATH
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Analyses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    selfie_hash TEXT,
                    doc_front_hash TEXT,
                    doc_back_hash TEXT,
                    face_similarity REAL,
                    ocr_extracted TEXT,
                    field_matches JSON,
                    duplicate_score REAL,
                    ai_coherence_score REAL,
                    ai_tampering_score REAL,
                    ai_extraction JSON,
                    ai_overall_confidence REAL,
                    user_confirmed BOOLEAN DEFAULT FALSE,
                    user_corrections JSON,
                    final_verdict TEXT,
                    processing_time_ms INTEGER,
                    model_version TEXT DEFAULT '1.0.0'
                )
            """)
            
            # Corrections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER REFERENCES analyses(id),
                    field_name TEXT,
                    expected_value TEXT,
                    extracted_value TEXT,
                    was_correct BOOLEAN,
                    confidence_at_time REAL,
                    corrected_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Learned patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT,
                    pattern_data JSON,
                    confidence REAL DEFAULT 0.0,
                    times_applied INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def save_analysis(self, result: AnalysisResult) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analyses (
                    selfie_hash, doc_front_hash, doc_back_hash,
                    face_similarity, ocr_extracted, field_matches,
                    duplicate_score, ai_coherence_score, ai_tampering_score,
                    ai_extraction, ai_overall_confidence, processing_time_ms,
                    model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.selfie_hash,
                result.doc_front_hash,
                result.doc_back_hash,
                result.face_similarity,
                result.ocr_text,
                json.dumps(result.field_matches),
                result.duplicate_score,
                result.ai_analysis.coherence_score if result.ai_analysis else None,
                result.ai_analysis.tampering_score if result.ai_analysis else None,
                json.dumps(result.ai_analysis.extracted_data) if result.ai_analysis else None,
                result.ai_analysis.overall_confidence if result.ai_analysis else None,
                result.processing_time_ms,
                result.model_version
            ))
            return cursor.lastrowid
    
    def get_analysis(self, analysis_id: int) -> Optional[AnalysisResult]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return AnalysisResult(
                analysis_id=row["id"],
                selfie_hash=row["selfie_hash"],
                doc_front_hash=row["doc_front_hash"],
                doc_back_hash=row["doc_back_hash"],
                face_similarity=row["face_similarity"],
                ocr_text=row["ocr_extracted"],
                field_matches=json.loads(row["field_matches"]) if row["field_matches"] else {},
                duplicate_score=row["duplicate_score"],
                processing_time_ms=row["processing_time_ms"],
                model_version=row["model_version"]
            )
    
    def save_feedback(self, analysis_id: int, feedback: UserFeedback):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Serialize corrections with datetime handling
            corrections_data = []
            for c in feedback.corrections:
                corrections_data.append({
                    "field_name": c.field_name,
                    "expected_value": c.expected_value,
                    "extracted_value": c.extracted_value,
                    "was_correct": c.was_correct,
                    "confidence_at_time": c.confidence_at_time,
                    "corrected_at": c.corrected_at.isoformat() if isinstance(c.corrected_at, datetime) else str(c.corrected_at),
                })
            
            # Update analysis with feedback
            cursor.execute("""
                UPDATE analyses SET
                    user_confirmed = ?,
                    user_corrections = ?,
                    final_verdict = ?
                WHERE id = ?
            """, (
                feedback.confirmed,
                json.dumps(corrections_data),
                "approved" if feedback.confirmed else "rejected",
                analysis_id
            ))
            
            # Save individual corrections
            for correction in feedback.corrections:
                cursor.execute("""
                    INSERT INTO corrections (
                        analysis_id, field_name, expected_value,
                        extracted_value, was_correct, confidence_at_time
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    analysis_id,
                    correction.field_name,
                    correction.expected_value,
                    correction.extracted_value,
                    correction.was_correct,
                    correction.confidence_at_time
                ))
    
    def get_all_corrections(self) -> List[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.*, a.ai_overall_confidence
                FROM corrections c
                JOIN analyses a ON c.analysis_id = a.id
                ORDER BY c.corrected_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def save_pattern(self, pattern: LearnedPattern) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO learned_patterns (
                    pattern_type, pattern_data, confidence,
                    times_applied, success_rate
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                pattern.pattern_type,
                json.dumps(pattern.pattern_data),
                pattern.confidence,
                pattern.times_applied,
                pattern.success_rate
            ))
            return cursor.lastrowid
    
    def update_pattern(self, pattern: LearnedPattern):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE learned_patterns SET
                    confidence = ?,
                    times_applied = ?,
                    success_rate = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                pattern.confidence,
                pattern.times_applied,
                pattern.success_rate,
                pattern.id
            ))
    
    def get_patterns_by_type(self, pattern_type: str) -> List[LearnedPattern]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM learned_patterns
                WHERE pattern_type = ?
                ORDER BY confidence DESC
            """, (pattern_type,))
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append(LearnedPattern(
                    id=row["id"],
                    pattern_type=row["pattern_type"],
                    pattern_data=json.loads(row["pattern_data"]),
                    confidence=row["confidence"],
                    times_applied=row["times_applied"],
                    success_rate=row["success_rate"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    last_updated=datetime.fromisoformat(row["last_updated"])
                ))
            return patterns
    
    def find_similar_pattern(self, pattern_type: str, correction: Correction) -> Optional[LearnedPattern]:
        patterns = self.get_patterns_by_type(pattern_type)
        
        for pattern in patterns:
            if pattern.pattern_data.get("field_name") == correction.field_name:
                return pattern
        
        return None
    
    def get_recent_analyses(self, days: int = 30) -> List[AnalysisResult]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM analyses
                WHERE timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC
            """, (f"-{days} days",))
            
            results = []
            for row in cursor.fetchall():
                result = AnalysisResult(
                    analysis_id=row["id"],
                    face_similarity=row["face_similarity"],
                    ocr_text=row["ocr_extracted"],
                    field_matches=json.loads(row["field_matches"]) if row["field_matches"] else {},
                    duplicate_score=row["duplicate_score"],
                    processing_time_ms=row["processing_time_ms"]
                )
                result.user_feedback = UserFeedback(
                    confirmed=row["user_confirmed"],
                    corrections=[],
                    timestamp=datetime.fromisoformat(row["timestamp"])
                )
                results.append(result)
            return results
    
    def get_confirmed_analyses(self) -> List[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM analyses
                WHERE user_confirmed = TRUE
                ORDER BY timestamp DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> dict:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as total FROM analyses")
            total = cursor.fetchone()["total"]
            
            cursor.execute("SELECT COUNT(*) as confirmed FROM analyses WHERE user_confirmed = TRUE")
            confirmed = cursor.fetchone()["confirmed"]
            
            cursor.execute("SELECT COUNT(*) as corrections FROM corrections")
            corrections = cursor.fetchone()["corrections"]
            
            cursor.execute("SELECT COUNT(*) as patterns FROM learned_patterns")
            patterns = cursor.fetchone()["patterns"]
            
            return {
                "total_analyses": total,
                "confirmed_analyses": confirmed,
                "total_corrections": corrections,
                "learned_patterns": patterns,
                "accuracy_rate": confirmed / total if total > 0 else 0
            }
