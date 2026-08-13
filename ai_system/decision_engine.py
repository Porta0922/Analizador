"""
decision_engine.py — Motor de decisión unificado (Fase 1)

Combina las señales del backend principal (/verify, 8000) con las del
análisis IA (/ai/analyze, 8001) en:

    - risk_score : 0-100 (mayor = más confiable)
    - verdict    : "approved" | "rejected"  (binario, sin revisión manual)
    - reasons    : lista de {code, message, severity} para auditoría

Reglas:
    - Si hay señales críticas (fraude confirmado, cara no detectada,
      tampering, mismo CI con cara distinta, IA caída) → RECHAZADO siempre.
    - Si los datos son insuficientes (sin OCR verificable y sin IA) →
      RECHAZADO (nunca aprobar en silencio con datos insuficientes).
    - En caso contrario, se calcula el promedio ponderado de las señales
      presentes (los pesos se renomalizan) y se compara con
      APPROVAL_THRESHOLD.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ai-system")

SEV_CRITICAL = "critical"
SEV_WARNING = "warning"
SEV_INFO = "info"

REASON_OK = "OK"
REASON_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
REASON_AI_UNAVAILABLE = "AI_UNAVAILABLE"

# Pesos por señal (mayor = más peso en el risk score). Suma ≈ 1.0.
# Si una señal no está presente, los pesos se renomalizan automáticamente.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "face_similarity": 0.20,
    "field_match_rate": 0.15,
    "back_document": 0.10,
    "selfie_doc_fraud": 0.10,
    "coherence": 0.15,
    "tampering": 0.20,
    "ai_face_match": 0.10,
}


@dataclass
class DecisionResult:
    approved: bool = False
    verdict: str = "rejected"
    risk_score: float = 0.0
    verdict_version: str = "1.0.0"
    data_sufficient: bool = False
    reasons: List[Dict[str, Any]] = field(default_factory=list)
    signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "verdict": self.verdict,
            "risk_score": round(self.risk_score, 2),
            "verdict_version": self.verdict_version,
            "data_sufficient": self.data_sufficient,
            "reasons": self.reasons,
            "signals": self.signals,
        }


class DecisionEngine:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        approval_threshold: float = 70.0,
        verdict_version: str = "1.0.0",
    ):
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.approval_threshold = float(approval_threshold)
        self.verdict_version = verdict_version

    def evaluate(
        self,
        signals: Dict[str, Optional[float]],
        notes: Optional[Dict[str, str]] = None,
        criticals: Optional[List[Dict[str, Any]]] = None,
        sufficient: bool = True,
        insufficient_reason: str = "",
        extra_reasons: Optional[List[Dict[str, Any]]] = None,
    ) -> DecisionResult:
        """
        Evalúa el conjunto de señales (0-100, mayor = mejor).

        Args:
            signals:   {clave: score} — las que falten o sean None se ignoran
            notes:     {clave: nota legible} para mensajes de señal baja
            criticals: lista de {code, message} — cualquiera presente => rechazo
            sufficient: si los datos alcanzan para decidir
            insufficient_reason: mensaje cuando sufficient=False
            extra_reasons: razones informativas adicionales (ej. fraud ring)
        """
        notes = notes or {}
        criticals = criticals or []
        extra_reasons = extra_reasons or []

        result = DecisionResult(verdict_version=self.verdict_version)

        present = {k: v for k, v in signals.items() if v is not None}

        total_w = sum(self.weights.get(k, 0) for k in present)
        if total_w > 0:
            raw = sum(present[k] * self.weights.get(k, 0) for k in present) / total_w
        else:
            raw = 0.0
        risk = round(max(0.0, min(100.0, raw)), 2)
        result.risk_score = risk

        # Desglose por señal para auditoría / calibración
        for k, v in present.items():
            w = self.weights.get(k, 0)
            result.signals[k] = {
                "score": round(v, 2),
                "weight": round(w, 3),
                "contribution": round(v * w / total_w, 2) if total_w > 0 else 0.0,
                "note": notes.get(k, ""),
            }

        # Razones de advertencia para señales bajas
        for k, v in present.items():
            if v < 50:
                result.reasons.append({
                    "code": k.upper(),
                    "message": notes.get(k, f"Señal baja: {k} ({v:.0f}%)"),
                    "severity": SEV_WARNING,
                    "score": round(v, 2),
                })

        result.reasons.extend(criticals)
        result.reasons.extend(extra_reasons)

        if not sufficient:
            result.reasons.append({
                "code": REASON_INSUFFICIENT_DATA,
                "message": insufficient_reason or "Datos insuficientes para emitir un veredicto",
                "severity": SEV_CRITICAL,
            })

        if not result.reasons and risk >= self.approval_threshold:
            result.reasons.append({
                "code": REASON_OK,
                "message": "Todas las señales superan los umbrales de aprobación",
                "severity": SEV_INFO,
            })

        approved = sufficient and not criticals and risk >= self.approval_threshold
        result.approved = approved
        result.verdict = "approved" if approved else "rejected"
        result.data_sufficient = sufficient

        logger.info(
            "[DECISION] risk=%.1f threshold=%.1f sufficient=%s criticals=%d → %s",
            risk, self.approval_threshold, sufficient, len(criticals), result.verdict,
        )
        return result
