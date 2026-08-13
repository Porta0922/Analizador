"""
calibrate.py — Fase 5. Calibra el umbral de aprobación con los feedbacks reales.

Recomienda el mejor APPROVAL_THRESHOLD según precision/recall/F1 calculados
sobre las decisiones ya emitidas que tuvieron feedback del usuario.

Uso:
    python -m ai_system.calibrate
"""

import json
import logging
from typing import List, Dict

from .database import Database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-system")


def _get_feedback_rows(db: Database) -> List[Dict]:
    """Analyses con feedback explícito (user_corrections IS NOT NULL) y decisión."""
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, risk_score, verdict, decision_reasons, signal_scores, user_confirmed
            FROM analyses
            WHERE user_corrections IS NOT NULL
              AND risk_score IS NOT NULL
            ORDER BY id
        """)
        rows = []
        for r in cursor.fetchall():
            reasons = json.loads(r["decision_reasons"] or "[]")
            has_critical = any(x.get("severity") == "critical" for x in reasons)
            rows.append({
                "id": r["id"],
                "risk": float(r["risk_score"]),
                "confirmed": bool(r["user_confirmed"]),
                "has_critical": has_critical,
                "signals": json.loads(r["signal_scores"] or "{}"),
            })
        return rows


def _decision(rows: List[Dict], threshold: float) -> Dict:
    """Aplica el threshold binario + regla crítica a los datos históricos."""
    tp = fp = tn = fn = 0
    approved_count = 0
    for r in rows:
        auto_approved = (not r["has_critical"]) and r["risk"] >= threshold
        if auto_approved:
            approved_count += 1
            if r["confirmed"]:
                tp += 1
            else:
                fp += 1
        else:
            if not r["confirmed"]:
                tn += 1
            else:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "approved": approved_count,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def _signal_profile(rows: List[Dict]) -> List[Dict]:
    """Promedio por señal para aprobaciones correctas vs falsos positivos."""
    def avg(key, subset):
        vals = [s[key]["score"] for s in subset if s["signals"].get(key)]
        return round(sum(vals) / len(vals), 1) if vals else None

    # Falsos positivos: auto-aprobadas pero el usuario dijo que estaba mal
    fpos = [r for r in rows if not r["has_critical"] and r["risk"] >= 70 and not r["confirmed"]]
    confirmed = [r for r in rows if r["confirmed"]]

    keys = ("face_similarity", "field_match_rate", "back_document",
            "selfie_doc_fraud", "coherence", "tampering", "ai_face_match")
    profile = []
    for key in keys:
        profile.append({
            "signal": key,
            "avg_confirmed": avg(key, confirmed),
            "avg_false_positive": avg(key, fpos),
        })
    return profile


def run() -> Dict:
    db = Database()
    rows = _get_feedback_rows(db)

    if not rows:
        logger.info("No hay análisis con feedback (user_corrections) para calibrar.")
        return {"total_feedback": 0, "table": [], "best_f1": None, "best_safe": None}

    logger.info("Calibrando con %d análisis con feedback...", len(rows))

    table = []
    best_f1 = None
    best_safe = None  # máximo threshold con recall >= 0.9
    for t in range(50, 91, 5):
        row = _decision(rows, t)
        table.append(row)
        if best_f1 is None or row["f1"] > best_f1["f1"]:
            best_f1 = row
        if row["recall"] >= 0.9 and (best_safe is None or row["threshold"] > best_safe["threshold"]):
            best_safe = row

    result = {
        "total_feedback": len(rows),
        "confirmed_true": sum(1 for r in rows if r["confirmed"]),
        "table": table,
        "best_f1": best_f1,
        "best_safe": best_safe,
        "signal_profile": _signal_profile(rows),
    }

    print("\n=== CALIBRACIÓN DE UMBRAL DE APROBACIÓN ===")
    print(f"Análisis con feedback: {result['total_feedback']} (confirmados: {result['confirmed_true']})")
    print(f"{'thr':>4} {'aprob':>5} {'tp':>3} {'fp':>3} {'tn':>3} {'fn':>3} {'prec':>6} {'rec':>6} {'f1':>6}")
    for t in table:
        print(f"{t['threshold']:>4} {t['approved']:>5} {t['tp']:>3} {t['fp']:>3} {t['tn']:>3} {t['fn']:>3} "
              f"{t['precision']:>6} {t['recall']:>6} {t['f1']:>6}")
    if best_f1:
        print(f"\nMejor por F1: threshold={best_f1['threshold']} (f1={best_f1['f1']}, prec={best_f1['precision']}, rec={best_f1['recall']})")
    if best_safe:
        print(f"Mejor seguro (recall≥0.9): threshold={best_safe['threshold']}")

    print("\n=== PERFIL POR SEÑAL ===")
    print(f"{'signal':<18} {'avg confirmado':>14} {'avg falso +':>14}")
    for s in result["signal_profile"]:
        print(f"{s['signal']:<18} {str(s['avg_confirmed']):>14} {str(s['avg_false_positive']):>14}")

    return result


if __name__ == "__main__":
    run()
