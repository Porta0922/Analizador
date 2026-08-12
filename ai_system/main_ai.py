import time
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import CORS_ORIGINS, OLLAMA_BASE_URL
from .ollama_client import OllamaClient
from .document_analyzer import DocumentAnalyzer
from .feedback_loop import FeedbackLoop
from .database import Database
from .train import AITrainer
from .seed_patterns import seed as seed_patterns
from .models import AIAnalysis, UserFeedback, Correction, AnalysisResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-system")

app = FastAPI(title="AI Document Analyzer", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Component initialization
# ---------------------------------------------------------------------------
db            = Database()
ollama        = OllamaClient()
analyzer      = DocumentAnalyzer(ollama, db)
feedback_loop = FeedbackLoop(db)
trainer       = AITrainer(db)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    selfie_b64: str = Field(min_length=1)
    doc_front_b64: str = Field(min_length=1)
    doc_back_b64: Optional[str] = Field(default=None)
    form_data: dict = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    confirmed: bool
    corrections: list = Field(default_factory=list)


class CorrectionData(BaseModel):
    field_name: str
    expected_value: str
    extracted_value: str
    was_correct: bool


# ---------------------------------------------------------------------------
# Background learning task (Phase 2b)
# ---------------------------------------------------------------------------

def _run_training_and_reload():
    """
    Background task: re-runs AITrainer.train_patterns() to extract new rules
    from accumulated corrections, then reloads them into the live analyzer so
    the very next inference benefits immediately.
    """
    try:
        logger.info("[TRAIN] Background training triggered by feedback event")
        new_patterns = trainer.train_patterns()
        logger.info("[TRAIN] Training complete — %d patterns upserted", len(new_patterns))

        # Reload patterns into the live DocumentAnalyzer instance
        analyzer.reload_patterns()
        logger.info("[TRAIN] DocumentAnalyzer patterns refreshed")
    except Exception as exc:
        logger.error("[TRAIN] Background training failed: %s", str(exc))


# ---------------------------------------------------------------------------
# Phase 2d: compute_field_diffs helper
# ---------------------------------------------------------------------------

def compute_field_diffs(expected: dict, actual: dict) -> list[dict]:
    """
    Compare two field dictionaries and return a list of Correction-compatible
    dicts describing each discrepancy found.

    Args:
        expected: dict of field_name -> value as provided / corrected by user
        actual:   dict of field_name -> value as extracted by AI

    Returns:
        List of dicts with keys: field_name, expected_value, extracted_value, was_correct
    """
    diffs = []
    all_keys = set(expected.keys()) | set(actual.keys())

    for key in all_keys:
        exp_val = str(expected.get(key, "")).strip()
        act_val = str(actual.get(key, "")).strip()

        was_correct = exp_val.lower() == act_val.lower()

        diffs.append({
            "field_name": key,
            "expected_value": exp_val,
            "extracted_value": act_val,
            "was_correct": was_correct,
        })

    return diffs


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check: Ollama status + learning stats."""
    models = ollama.get_available_models()
    return {
        "status": "ok",
        "ollama_url": OLLAMA_BASE_URL,
        "available_models": models,
        "stats": feedback_loop.get_learning_stats(),
    }


@app.post("/ai/analyze")
def analyze_document(request: AnalysisRequest):
    """Analyze a document using AI (coherence + tampering + confidence)."""
    import traceback
    start_time = time.time()

    try:
        logger.info("[AI] Received analysis request")
        ai_analysis = analyzer.analyze(
            request.selfie_b64,
            request.doc_front_b64,
            request.doc_back_b64,
            request.form_data,
        )
        logger.info("[AI] Analysis completed")

        result = AnalysisResult(
            face_similarity=0.0,
            ocr_text="",
            field_matches={},
            ai_analysis=ai_analysis,
            processing_time_ms=int((time.time() - start_time) * 1000),
        )

        analysis_id = feedback_loop.record_analysis(result)
        logger.info("[AI] Analysis saved: id=%d, confidence=%.1f%%",
                    analysis_id, ai_analysis.overall_confidence)

        return {
            "analysis_id": analysis_id,
            "result": ai_analysis.to_dict(),
            "processing_time_ms": result.processing_time_ms,
            "should_reject": analyzer.should_reject(ai_analysis),
            "summary": analyzer.get_analysis_summary(ai_analysis),
        }

    except Exception as e:
        logger.error("[AI] Analysis failed: %s\n%s", str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/ai/feedback/{analysis_id}")
def submit_feedback(
    analysis_id: int,
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit user feedback for a completed analysis.

    Workflow (Phase 2b + 2d):
      1. Convert raw correction dicts → Correction objects.
      2. If the user provided expected_data + actual_data at the top level,
         compute field diffs automatically and merge with manual corrections.
      3. Persist feedback via FeedbackLoop (updates patterns inline).
      4. Enqueue background training task so new rules are ready for the
         next inference without blocking this response.
    """
    try:
        corrections: list[Correction] = []

        for corr_data in request.corrections:
            # Support both flat dicts and nested expected/actual dicts
            if "expected_data" in corr_data and "actual_data" in corr_data:
                # Compute field-level diffs from high-level objects
                auto_diffs = compute_field_diffs(
                    corr_data["expected_data"],
                    corr_data["actual_data"],
                )
                for diff in auto_diffs:
                    corrections.append(Correction(**diff))
            else:
                corrections.append(Correction(
                    field_name=corr_data.get("field_name", "unknown"),
                    expected_value=str(corr_data.get("expected_value", "")),
                    extracted_value=str(corr_data.get("extracted_value", "")),
                    was_correct=bool(corr_data.get("was_correct", False)),
                ))

        feedback = UserFeedback(
            confirmed=request.confirmed,
            corrections=corrections,
        )

        feedback_loop.record_feedback(analysis_id, feedback)

        # --- Phase 2b: event-driven training trigger ---
        # Always re-train after feedback to keep patterns fresh.
        # Use BackgroundTasks so the HTTP response is not blocked.
        background_tasks.add_task(_run_training_and_reload)
        logger.info("[AI] Background training enqueued for analysis_id=%d", analysis_id)

        return {
            "status": "recorded",
            "analysis_id": analysis_id,
            "corrections_processed": len(corrections),
            "training_queued": True,
            "learning_stats": feedback_loop.get_learning_stats(),
        }

    except Exception as e:
        logger.error("[AI] Feedback recording failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai/stats")
def get_stats():
    """Get learning statistics (used by extension dashboard — Phase 1)."""
    return feedback_loop.get_learning_stats()


@app.get("/ai/patterns")
def get_patterns(pattern_type: Optional[str] = None):
    """Get learned patterns, optionally filtered by type."""
    all_types = [
        "field_regex", "common_error", "tampering_sign",
        "name_format", "document_type", "date_format",
        "number_format", "gender_format",
    ]
    if pattern_type:
        patterns = db.get_patterns_by_type(pattern_type)
        return {"patterns": [p.to_dict() for p in patterns]}
    else:
        all_patterns = {}
        for ptype in all_types:
            ps = db.get_patterns_by_type(ptype)
            if ps:
                all_patterns[ptype] = [p.to_dict() for p in ps]
        return {"patterns": all_patterns}


@app.post("/ai/train")
def trigger_training():
    """Manually trigger pattern training (useful for ops/debugging)."""
    new_patterns = trainer.train_patterns()
    analyzer.reload_patterns()
    return {
        "status": "ok",
        "patterns_created": len(new_patterns),
        "patterns": [p.to_dict() for p in new_patterns],
    }


@app.on_event("startup")
def startup():
    logger.info("[AI] Starting AI Document Analyzer v2.0")
    logger.info("[AI] Ollama URL: %s", OLLAMA_BASE_URL)
    logger.info("[AI] Database: %s", db.db_path)

    # Seed Paraguay domain patterns on first start (idempotent)
    try:
        inserted = seed_patterns(force=False)
        if inserted:
            analyzer.reload_patterns()
            logger.info("[AI] Seeded %d Paraguay domain patterns", inserted)
    except Exception as exc:
        logger.warning("[AI] Pattern seed failed (non-fatal): %s", exc)


@app.on_event("shutdown")
def shutdown():
    ollama.close()
    logger.info("[AI] Shutdown complete")
