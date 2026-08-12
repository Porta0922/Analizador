import time
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import CORS_ORIGINS, OLLAMA_BASE_URL
from .ollama_client import OllamaClient
from .document_analyzer import DocumentAnalyzer
from .feedback_loop import FeedbackLoop
from .database import Database
from .models import AIAnalysis, UserFeedback, Correction, AnalysisResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-system")

app = FastAPI(title="AI Document Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Initialize components
db = Database()
ollama = OllamaClient()
analyzer = DocumentAnalyzer(ollama, db)
feedback_loop = FeedbackLoop(db)


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


@app.get("/health")
def health():
    """Health check endpoint."""
    models = ollama.get_available_models()
    return {
        "status": "ok",
        "ollama_url": OLLAMA_BASE_URL,
        "available_models": models,
        "stats": feedback_loop.get_learning_stats()
    }


@app.post("/ai/analyze")
def analyze_document(request: AnalysisRequest):
    """Analyze a document using AI."""
    import traceback
    start_time = time.time()
    
    try:
        # Run AI analysis
        logger.info("[AI] Received analysis request")
        ai_analysis = analyzer.analyze(
            request.selfie_b64,
            request.doc_front_b64,
            request.doc_back_b64,
            request.form_data
        )
        logger.info("[AI] Analysis completed, creating result")
        
        # Create analysis result for storage
        result = AnalysisResult(
            face_similarity=0.0,
            ocr_text="",
            field_matches={},
            ai_analysis=ai_analysis,
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        
        # Record analysis
        analysis_id = feedback_loop.record_analysis(result)
        
        logger.info("[AI] Analysis saved: id=%d, confidence=%.1f%%", 
                    analysis_id, ai_analysis.overall_confidence)
        
        return {
            "analysis_id": analysis_id,
            "result": ai_analysis.to_dict(),
            "processing_time_ms": result.processing_time_ms,
            "should_reject": analyzer.should_reject(ai_analysis),
            "summary": analyzer.get_analysis_summary(ai_analysis)
        }
    
    except Exception as e:
        logger.error("[AI] Analysis failed: %s\n%s", str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/ai/feedback/{analysis_id}")
def submit_feedback(analysis_id: int, request: FeedbackRequest):
    """Submit user feedback for an analysis."""
    try:
        # Convert corrections to proper objects
        corrections = []
        for corr_data in request.corrections:
            corrections.append(Correction(
                field_name=corr_data["field_name"],
                expected_value=corr_data["expected_value"],
                extracted_value=corr_data["extracted_value"],
                was_correct=corr_data["was_correct"]
            ))
        
        feedback = UserFeedback(
            confirmed=request.confirmed,
            corrections=corrections
        )
        
        feedback_loop.record_feedback(analysis_id, feedback)
        
        return {
            "status": "recorded",
            "analysis_id": analysis_id,
            "learning_stats": feedback_loop.get_learning_stats()
        }
    
    except Exception as e:
        logger.error("[AI] Feedback recording failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai/stats")
def get_stats():
    """Get learning statistics."""
    return feedback_loop.get_learning_stats()


@app.get("/ai/patterns")
def get_patterns(pattern_type: Optional[str] = None):
    """Get learned patterns."""
    if pattern_type:
        patterns = db.get_patterns_by_type(pattern_type)
        return {"patterns": [p.to_dict() for p in patterns]}
    else:
        all_patterns = {}
        for ptype in ["field_regex", "common_error", "tampering_sign"]:
            patterns = db.get_patterns_by_type(ptype)
            all_patterns[ptype] = [p.to_dict() for p in patterns]
        return {"patterns": all_patterns}


@app.on_event("startup")
def startup():
    logger.info("[AI] Starting AI Document Analyzer")
    logger.info("[AI] Ollama URL: %s", OLLAMA_BASE_URL)
    logger.info("[AI] Database: %s", db.db_path)


@app.on_event("shutdown")
def shutdown():
    ollama.close()
    logger.info("[AI] Shutdown complete")
