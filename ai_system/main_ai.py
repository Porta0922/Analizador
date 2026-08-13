import time
import hashlib
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import (
    CORS_ORIGINS, OLLAMA_BASE_URL,
    APPROVAL_THRESHOLD, VERDICT_VERSION, AI_REQUIRED_FOR_APPROVAL,
    ENABLE_CROSS_REQUEST_FRAUD, FRAUD_RING_SIM_THRESHOLD,
    TAMPERING_THRESHOLD,
)
from .ollama_client import OllamaClient
from .document_analyzer import DocumentAnalyzer
from .feedback_loop import FeedbackLoop
from .database import Database
from .train import AITrainer
from .seed_patterns import seed as seed_patterns
from .decision_engine import DecisionEngine, DEFAULT_WEIGHTS
from .models import AIAnalysis, UserFeedback, Correction, AnalysisResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-system")

app = FastAPI(title="AI Document Analyzer", version="3.0.0")

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
engine        = DecisionEngine(
    weights=DEFAULT_WEIGHTS,
    approval_threshold=APPROVAL_THRESHOLD,
    verdict_version=VERDICT_VERSION,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    selfie_b64: str = Field(min_length=1)
    doc_front_b64: str = Field(min_length=1)
    doc_back_b64: Optional[str] = Field(default=None)
    form_data: dict = Field(default_factory=dict)
    # Etapa 2 — result de field_matches del backend principal (/verify):
    # dict {campo: True|False|None}. Los False se re-verifican con llava.
    ocr_field_matches: Optional[dict] = Field(default=None)


class DecisionRequest(AnalysisRequest):
    """Petición para el motor de decisión unificado (Fase 1/3)."""
    # Señales del /verify (backend principal, 8000)
    verify_signals: Optional[dict] = Field(default=None)
    # Embeddings faciales (Fase 3 — antifraude entre solicitudes)
    selfie_embedding: Optional[list] = Field(default=None)


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
# Fase 1/3 — Helpers del motor de decisión
# ---------------------------------------------------------------------------

def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _normalize_doc_number(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())


def _cosine_sim(a: list, b: list) -> Optional[float]:
    """Similitud coseno pura (sin numpy). Devuelve -1..1 o None."""
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


def _build_decision_inputs(verify: dict, ai_analysis: Optional[AIAnalysis]) -> dict:
    """
    Convierte las señales crudas de /verify + análisis IA en las entradas
    del DecisionEngine: {signals, notes, criticals, sufficient, reason}.
    """
    signals: dict = {}
    notes: dict = {}
    criticals = []

    # ── 1. Similitud facial determinista (FaceNet) ──────────────────────
    face_sim = verify.get("facial_similarity")
    face_state = verify.get("face_detected")
    if face_sim is not None:
        signals["face_similarity"] = face_sim
        notes["face_similarity"] = f"Similitud facial FaceNet: {face_sim}% (estado: {face_state})"
    if face_state == "none":
        criticals.append({"code": "FACE_NOT_DETECTED",
                          "message": "No se pudo detectar un rostro en la selfie o en el documento"})
    elif verify.get("is_same_person") is False and face_state in ("both", "one"):
        criticals.append({"code": "FACE_MISMATCH",
                          "message": f"Coincidencia facial negativa ({face_sim}%)"})

    # ── 2. Coincidencia de campos OCR ────────────────────────────────────
    field_matches = verify.get("field_matches") or {}
    verifiable = [(k, v) for k, v in field_matches.items() if v is not None]
    if verifiable:
        rate = sum(1 for _, v in verifiable if v is True) / len(verifiable) * 100
        signals["field_match_rate"] = round(rate, 2)
        notes["field_match_rate"] = f"{len(verifiable) - int(rate/100*len(verifiable))} de {len(verifiable)} campos no coinciden"
        if rate < 60:
            criticals.append({"code": "FIELD_MISMATCH",
                              "message": f"Solo {rate:.0f}% de los campos verificables coinciden con el documento"})

    # ── 3. Dorso del documento ───────────────────────────────────────────
    back_status = verify.get("back_document_status")
    back_map = {"ok": 100, "duplicate": 40, "same_as_front": 10,
                "decode_error": 30, "not_provided": None}
    if back_status in back_map and back_map[back_status] is not None:
        signals["back_document"] = back_map[back_status]
        notes["back_document"] = f"Dorso: {back_status}"
        if back_status == "same_as_front":
            criticals.append({"code": "DUPLICATE_DOCUMENT",
                              "message": "El dorso parece ser el frente del documento (imagen duplicada)"})
        elif back_status == "duplicate":
            criticals.append({"code": "DUPLICATE_DOCUMENT",
                              "message": "Frente y dorso del documento son casi idénticos"})

    # ── 4. Fraude selfie = documento ─────────────────────────────────────
    fraud_reason = verify.get("fraud_reason", "none")
    if fraud_reason == "identical_image":
        signals["selfie_doc_fraud"] = 0
        notes["selfie_doc_fraud"] = "Selfie idéntica al documento (misma foto)"
        criticals.append({"code": "SELFIE_DOC_FRAUD",
                          "message": "La selfie y el documento son la misma imagen"})
    elif fraud_reason == "photo_of_screen":
        signals["selfie_doc_fraud"] = 50
        notes["selfie_doc_fraud"] = "Selfie muy similar al documento (posible foto de pantalla)"
        criticals.append({"code": "PHOTO_OF_SCREEN",
                          "message": "La selfie parece una foto de pantalla del documento"})
    else:
        signals["selfie_doc_fraud"] = 100
        notes["selfie_doc_fraud"] = "Sin fraude de imagen detectado"

    # ── 5. Señales IA ────────────────────────────────────────────────────
    if ai_analysis is not None:
        signals["coherence"] = ai_analysis.coherence_score
        notes["coherence"] = f"Coherencia de datos: {ai_analysis.coherence_score}%"
        if ai_analysis.coherence_issues:
            notes["coherence"] += " · " + "; ".join(ai_analysis.coherence_issues[:3])

        signals["tampering"] = ai_analysis.tampering_score
        notes["tampering"] = f"Integridad del documento: {ai_analysis.tampering_score}%"
        if ai_analysis.tampering_areas:
            notes["tampering"] += " · áreas: " + ", ".join(
                str(a) if isinstance(a, str) else str(a.get("area", a))
                for a in ai_analysis.tampering_areas[:3]
            )
        if ai_analysis.tampering_score < TAMPERING_THRESHOLD:
            criticals.append({"code": "TAMPERING",
                              "message": f"Posible manipulación del documento (score {ai_analysis.tampering_score:.0f}%)"})

        if ai_analysis.face_match_score >= 0:
            signals["ai_face_match"] = ai_analysis.face_match_score
            notes["ai_face_match"] = f"Comparación facial por IA: {ai_analysis.face_match_score:.0f}%"

        if ai_analysis.back_analysis_score >= 0:
            signals["ai_back"] = ai_analysis.back_analysis_score
            notes["ai_back"] = f"Verificación del dorso por IA: {ai_analysis.back_analysis_score:.0f}%"

    sufficient = bool(verifiable) or ai_analysis is not None
    insufficient_reason = ""
    if not verifiable and ai_analysis is None:
        insufficient_reason = "No hay datos verificables (sin OCR y sin análisis IA)"

    return {
        "signals": signals,
        "notes": notes,
        "criticals": criticals,
        "sufficient": sufficient,
        "insufficient_reason": insufficient_reason,
    }


def _check_cross_request_fraud(request: DecisionRequest) -> list:
    """
    Fase 3 — Antifraude entre solicitudes:
      - Mismo CI con selfie de otra persona (cara distinta) → FRAUD_RING
      - Misma selfie reutilizada con otro documento → SELFIE_REUSED
    """
    reasons: list = []
    if not ENABLE_CROSS_REQUEST_FRAUD:
        return reasons
    if not request.selfie_embedding:
        return reasons

    selfie_embedding = request.selfie_embedding
    doc_number = _normalize_doc_number(request.form_data.get("numeroDoc", ""))
    selfie_hash = _md5(request.selfie_b64.split(",", 1)[-1])

    if doc_number:
        doc_hash = _md5(doc_number)
        prev = db.query_same_doc_embeddings(doc_hash)
        for record in prev:
            emb = record.get("selfie_embedding") or []
            sim = _cosine_sim(selfie_embedding, emb)
            if sim is None:
                continue
            sim_pct = sim * 100
            if sim_pct < FRAUD_RING_SIM_THRESHOLD:
                reasons.append({
                    "code": "FRAUD_RING",
                    "message": (f"El CI {doc_number} ya fue presentado con una cara distinta "
                                f"(similitud {sim_pct:.0f}%, análisis previo #{record['analysis_id']})"),
                    "severity": "critical",
                })
                break

    # Misma selfie usada con otro documento
    if doc_number:
        doc_hash = _md5(doc_number)
        reused = db.query_selfie_hashes_other_docs(selfie_hash, exclude_doc_hash=doc_hash)
        if reused:
            reasons.append({
                "code": "SELFIE_REUSED",
                "message": f"La misma selfie ya fue usada en el análisis #{reused[0]['analysis_id']} con otro documento",
                "severity": "warning",
            })

    return reasons


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
            ocr_field_matches=request.ocr_field_matches,
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
            # Opción D — flags rápidos para la extensión
            "face_match_available": ai_analysis.face_match_score >= 0,
            "face_match_score":     ai_analysis.face_match_score,
            "back_verified":        ai_analysis.back_analysis_score >= 0,
            "back_is_back":         ai_analysis.back_analysis_issues == [] and ai_analysis.back_analysis_score >= 50,
            # Etapa 2 — campos que llava re-verificó visualmente
            "visual_field_matches": ai_analysis.visual_field_matches,
        }

    except Exception as e:
        logger.error("[AI] Analysis failed: %s\n%s", str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.post("/ai/decision")
def ai_decision(request: DecisionRequest):
    """
    Motor de decisión unificado (Fase 1):
    orquesta el análisis IA + las señales del /verify + antifraude entre
    solicitudes, y devuelve un veredicto binario: approved / rejected.

    Sin revisión manual: si los datos son insuficientes o la IA está caída,
    el veredicto es RECHAZADO (nunca se aprueba en silencio).
    """
    import traceback
    start_time = time.time()
    verify = request.verify_signals or {}

    try:
        # ── ¿La IA está disponible? (circuit breaker + Ollama arriba) ────
        ai_available = not OllamaClient.is_degraded()
        if ai_available:
            try:
                ai_available = bool(ollama.get_available_models())
            except Exception:
                ai_available = False

        # ── Análisis IA (solo si está disponible) ─────────────────────────
        ai_analysis: Optional[AIAnalysis] = None
        if ai_available:
            ai_analysis = analyzer.analyze(
                request.selfie_b64,
                request.doc_front_b64,
                request.doc_back_b64,
                request.form_data,
                ocr_field_matches=request.ocr_field_matches,
            )
        else:
            logger.warning("[DECISION] Ollama degradado/caído — se omite análisis IA")

        # ── Construir entradas del motor de decisión ──────────────────────
        inputs = _build_decision_inputs(verify, ai_analysis)

        # Si la IA es obligatoria y no está → datos insuficientes → rechazo
        if AI_REQUIRED_FOR_APPROVAL and not ai_available:
            inputs["sufficient"] = False
            inputs["insufficient_reason"] = "El análisis IA no está disponible"
            inputs["criticals"].append({"code": "AI_UNAVAILABLE",
                                        "message": "Ollama está caído o degradado; no se puede auto-aprobar"})

        # ── Antifraude entre solicitudes (Fase 3) ─────────────────────────
        fraud_reasons = _check_cross_request_fraud(request)

        decision = engine.evaluate(
            signals=inputs["signals"],
            notes=inputs["notes"],
            criticals=inputs["criticals"],
            sufficient=inputs["sufficient"],
            insufficient_reason=inputs["insufficient_reason"],
            extra_reasons=fraud_reasons,
        )

        # ── Persistir análisis + decisión + embeddings ────────────────────
        result = AnalysisResult(
            face_similarity=verify.get("facial_similarity") or 0.0,
            ocr_text="",
            field_matches=verify.get("field_matches") or {},
            ai_analysis=ai_analysis,
            processing_time_ms=int((time.time() - start_time) * 1000),
        )
        analysis_id = feedback_loop.record_analysis(result)
        db.save_decision(analysis_id, decision)

        # Embeddings para antifraude futuro (solo si hay CI y embedding)
        doc_number = _normalize_doc_number(request.form_data.get("numeroDoc", ""))
        if doc_number and request.selfie_embedding:
            db.save_face_embeddings(
                analysis_id=analysis_id,
                numero_doc_hash=_md5(doc_number),
                selfie_embedding=request.selfie_embedding,
                selfie_hash=_md5(request.selfie_b64.split(",", 1)[-1]),
            )

        return {
            "analysis_id": analysis_id,
            "decision": decision.to_dict(),
            "ai_available": ai_available,
            "ai_analysis": ai_analysis.to_dict() if ai_analysis else None,
            "processing_time_ms": result.processing_time_ms,
            "summary": _decision_summary(decision),
        }

    except Exception as e:
        logger.error("[DECISION] Failed: %s\n%s", str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Decision error: {str(e)}")


def _decision_summary(decision) -> str:
    status = "APROBADO" if decision.approved else "RECHAZADO"
    lines = [f"Veredicto: {status}", f"Riesgo: {decision.risk_score:.1f}%"]
    for r in decision.reasons:
        lines.append(f"  - [{r.get('severity')}] {r.get('message')}")
    return "\n".join(lines)


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


# ---------------------------------------------------------------------------
# Auditoría (Fase 2)
# ---------------------------------------------------------------------------

@app.get("/audit/{analysis_id}")
def audit_analysis(analysis_id: int):
    """Registro completo de un análisis: señales, decisión, razones y correcciones."""
    record = db.get_decision_by_id(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Análisis no encontrado")
    return {"analysis": record}


@app.get("/audit/export")
def audit_export(limit: int = 500):
    """Exportación plana (JSON) de los análisis con sus decisiones."""
    return {"count": 0, "rows": db.export_analyses(limit=min(limit, 5000))}


# ---------------------------------------------------------------------------
# Dashboard de operación (Fase 6)
# ---------------------------------------------------------------------------

@app.get("/ai/dashboard")
def dashboard():
    """Métricas operativas para monitorear la automatización."""
    stats = db.get_dashboard_stats()
    stats["accuracy_trend"] = db.get_accuracy_trend(20)
    return stats


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
