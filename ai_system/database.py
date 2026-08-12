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

            # Accuracy snapshots table (Phase 4) — one row per training/feedback event
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_analyses INTEGER DEFAULT 0,
                    confirmed_analyses INTEGER DEFAULT 0,
                    accuracy_rate REAL DEFAULT 0.0,
                    active_patterns INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0.0,
                    trigger_event TEXT DEFAULT 'manual'
                )
            """)

    # ------------------------------------------------------------------
    # Analysis CRUD
    # ------------------------------------------------------------------

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
                result.model_version,
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
                model_version=row["model_version"],
            )

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def save_feedback(self, analysis_id: int, feedback: UserFeedback):
        with self._get_connection() as conn:
            cursor = conn.cursor()

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
                analysis_id,
            ))

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
                    correction.confidence_at_time,
                ))

    # ------------------------------------------------------------------
    # Corrections
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

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
                pattern.success_rate,
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
                pattern.id,
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
                    last_updated=datetime.fromisoformat(row["last_updated"]),
                ))
            return patterns

    def get_all_active_patterns(self, min_confidence: float = 0.5) -> List[LearnedPattern]:
        """Return all patterns above a confidence threshold, sorted by confidence."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM learned_patterns
                WHERE confidence >= ?
                ORDER BY confidence DESC
            """, (min_confidence,))
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
                    last_updated=datetime.fromisoformat(row["last_updated"]),
                ))
            return patterns

    def find_similar_pattern(self, pattern_type: str, correction: Correction) -> Optional[LearnedPattern]:
        patterns = self.get_patterns_by_type(pattern_type)
        for pattern in patterns:
            if pattern.pattern_data.get("field_name") == correction.field_name:
                return pattern
        return None

    # ------------------------------------------------------------------
    # Bulk queries
    # ------------------------------------------------------------------

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
                    processing_time_ms=row["processing_time_ms"],
                )
                result.user_feedback = UserFeedback(
                    confirmed=row["user_confirmed"],
                    corrections=[],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
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

    # ------------------------------------------------------------------
    # Statistics (Phase 1 — expanded)
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as total FROM analyses")
            total = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) as confirmed FROM analyses WHERE user_confirmed = TRUE")
            confirmed = cursor.fetchone()["confirmed"]

            cursor.execute("SELECT COUNT(*) as rejected FROM analyses WHERE final_verdict = 'rejected'")
            rejected = cursor.fetchone()["rejected"]

            cursor.execute("SELECT COUNT(*) as corrections FROM corrections")
            corrections_count = cursor.fetchone()["corrections"]

            cursor.execute("SELECT COUNT(*) as patterns FROM learned_patterns")
            patterns_count = cursor.fetchone()["patterns"]

            accuracy = confirmed / total if total > 0 else 0.0

            return {
                "total_analyses": total,
                "confirmed_analyses": confirmed,
                "rejected_analyses": rejected,
                "total_corrections": corrections_count,
                "learned_patterns": patterns_count,
                "accuracy_rate": round(accuracy, 4),
            }

    def get_top_error_fields(self, limit: int = 5) -> List[dict]:
        """
        Return the fields with the most incorrect corrections, ranked by error count.
        Used to populate the 'Top Error Fields' section of the dashboard.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    field_name,
                    COUNT(*) AS total_corrections,
                    SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) AS error_count,
                    ROUND(
                        100.0 * SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) / COUNT(*),
                        1
                    ) AS error_rate_pct
                FROM corrections
                GROUP BY field_name
                ORDER BY error_count DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_active_patterns_summary(self, min_confidence: float = 0.5, limit: int = 10) -> List[dict]:
        """
        Return a human-readable summary of the top active learned patterns.
        Used in the 'Active Injected Patterns' section of the dashboard.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, pattern_type, pattern_data, confidence, times_applied, success_rate
                FROM learned_patterns
                WHERE confidence >= ?
                ORDER BY confidence DESC
                LIMIT ?
            """, (min_confidence, limit))
            results = []
            for row in cursor.fetchall():
                data = json.loads(row["pattern_data"]) if row["pattern_data"] else {}
                results.append({
                    "id": row["id"],
                    "pattern_type": row["pattern_type"],
                    "description": data.get("description", ""),
                    "confidence": round(row["confidence"], 3),
                    "times_applied": row["times_applied"],
                    "success_rate": round(row["success_rate"], 3),
                })
            return results

    # ------------------------------------------------------------------
    # Accuracy snapshots (Phase 4)
    # ------------------------------------------------------------------

    def save_accuracy_snapshot(self, trigger_event: str = "feedback") -> int:
        """
        Persist a point-in-time accuracy snapshot to enable trend tracking.
        Called automatically after each training cycle.
        """
        stats = self.get_statistics()

        # Average confidence of all analyses with a score
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT AVG(ai_overall_confidence) as avg_conf
                FROM analyses
                WHERE ai_overall_confidence IS NOT NULL
            """)
            row = cursor.fetchone()
            avg_conf = round(row["avg_conf"] or 0.0, 2)

            cursor.execute("""
                INSERT INTO accuracy_snapshots (
                    total_analyses, confirmed_analyses, accuracy_rate,
                    active_patterns, avg_confidence, trigger_event
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                stats["total_analyses"],
                stats["confirmed_analyses"],
                stats["accuracy_rate"],
                stats["learned_patterns"],
                avg_conf,
                trigger_event,
            ))
            return cursor.lastrowid

    def get_accuracy_trend(self, limit: int = 20) -> List[dict]:
        """
        Return the most recent accuracy snapshots for charting / display.
        Each entry represents a model state at a point in time.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    id,
                    snapshot_at,
                    total_analyses,
                    confirmed_analyses,
                    ROUND(accuracy_rate * 100, 1) AS accuracy_pct,
                    active_patterns,
                    avg_confidence,
                    trigger_event
                FROM accuracy_snapshots
                ORDER BY snapshot_at DESC
                LIMIT ?
            """, (limit,))
            rows = [dict(row) for row in cursor.fetchall()]
            # Return chronological order for charting
            rows.reverse()
            return rows
