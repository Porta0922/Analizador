import json
import hashlib
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import (
    COHERENCE_THRESHOLD,
    TAMPERING_THRESHOLD,
    OVERALL_CONFIDENCE_THRESHOLD,
    PROMPTS_DIR,
    PATTERN_CONFIDENCE_THRESHOLD,
)
from .ollama_client import OllamaClient
from .database import Database
from .models import AIAnalysis, LearnedPattern

logger = logging.getLogger("ai-system")

# ---------------------------------------------------------------------------
# Reglas de dominio: Paraguay / LATAM
# ---------------------------------------------------------------------------
LOCALE_RULES = """
## Reglas de Dominio — Paraguay / LATAM

- "CÉDULA DE IDENTIDAD POLICIAL" es un tipo de documento válido en Paraguay. No lo marques como sospechoso.
- Los apellidos compuestos son comunes (ej. "MORALES FERNANDEZ", "GIMÉNEZ BENÍTEZ"). No los marques como inconsistentes.
- Nombres como "BANILDO", "WILFRIDO", "SINDULFO", "NATANAEL" son nombres válidos en Paraguay.
- El número de CI paraguayo puede tener entre 6 y 8 dígitos, con o sin puntos separadores (ej. "1.234.567" o "1234567").
- El campo "sexo" puede aparecer como "M", "F", "MASCULINO" o "FEMENINO".
- Las fechas pueden estar en formato DD/MM/YYYY o DD-MM-YYYY.
- El texto OCR puede contener errores menores (ej. "0" por "O", "1" por "I"). No los trates como falsificación.
"""

# Tipos de patrón por categoría de uso en prompts
PROMPT_PATTERN_TYPES = {
    "coherence": ["name_format", "document_type", "date_format", "number_format", "gender_format", "common_error"],
    "tampering": ["tampering_sign"],
    "all":       ["name_format", "document_type", "date_format", "number_format",
                  "gender_format", "common_error", "tampering_sign", "field_regex"],
}


class DocumentAnalyzer:
    def __init__(self, ollama_client: OllamaClient, db: Database):
        self.ollama = ollama_client
        self.db = db
        self.patterns = self._load_learned_patterns()
        self._load_prompts()

    # ------------------------------------------------------------------
    # Prompt loading
    # ------------------------------------------------------------------

    def _load_prompts(self):
        """Load prompt templates from files."""
        self.prompts = {}
        prompt_files = {
            "analyze_document": "analyze_document.txt",
            "validate_coherence": "validate_coherence.txt",
            "detect_tampering": "detect_tampering.txt",
        }
        for key, filename in prompt_files.items():
            prompt_path = PROMPTS_DIR / filename
            if prompt_path.exists():
                self.prompts[key] = prompt_path.read_text(encoding="utf-8")
            else:
                self.prompts[key] = ""
                logger.warning("[ANALYZER] Prompt file not found: %s", filename)

    # ------------------------------------------------------------------
    # Pattern loading
    # ------------------------------------------------------------------

    def _load_learned_patterns(self) -> Dict[str, list]:
        """Load all known pattern types from database."""
        all_types = [
            "field_regex", "common_error", "tampering_sign",
            "name_format", "document_type", "date_format",
            "number_format", "gender_format",
        ]
        patterns = {}
        for pattern_type in all_types:
            patterns[pattern_type] = self.db.get_patterns_by_type(pattern_type)
        return patterns

    def reload_patterns(self):
        """Reload patterns from DB (called after training completes)."""
        self.patterns = self._load_learned_patterns()
        logger.info("[ANALYZER] Patterns reloaded from DB")

    # ------------------------------------------------------------------
    # 3-layer system prompt builder (Phase 2c)
    # ------------------------------------------------------------------

    def _build_system_prompt(self, base_prompt_key: str, form_data: Optional[dict] = None) -> str:
        """
        Build a layered system prompt:
          Layer 1 — Base prompt (from prompts/*.txt)
          Layer 2 — Locale / domain rules (Paraguay specifics)
          Layer 3 — Dynamic learned patterns injected from DB
        """
        # --- Layer 1: Base ---
        base = self.prompts.get(base_prompt_key, "")

        if base_prompt_key == "validate_coherence" and form_data:
            try:
                base = base.format(
                    nombre=form_data.get("primerNombre", ""),
                    apellido=form_data.get("primerApellido", ""),
                    numero=form_data.get("numeroDoc", ""),
                    tipo=form_data.get("tipoDoc", ""),
                    sexo=form_data.get("sexo", ""),
                    fecha=form_data.get("fechaNacimiento", ""),
                )
            except KeyError:
                pass  # Template may not have all placeholders

        # --- Layer 2: Locale rules ---
        prompt = base + "\n\n" + LOCALE_RULES.strip()

        # --- Layer 3: Dynamic pattern injection ---
        category = "coherence" if "coherence" in base_prompt_key else "tampering"
        prompt = self._inject_learned_patterns(prompt, category)

        return prompt

    def _inject_learned_patterns(self, prompt: str, category: str) -> str:
        """
        Inject top-N high-confidence learned patterns relevant to the category.
        Fixes the broken type mismatch from the original implementation.
        """
        relevant_types = PROMPT_PATTERN_TYPES.get(category, PROMPT_PATTERN_TYPES["all"])

        active_patterns: List[LearnedPattern] = []
        for ptype in relevant_types:
            for p in self.patterns.get(ptype, []):
                if p.confidence >= PATTERN_CONFIDENCE_THRESHOLD:
                    active_patterns.append(p)

        if not active_patterns:
            return prompt

        # Sort by confidence descending, take top 10
        top = sorted(active_patterns, key=lambda p: p.confidence, reverse=True)[:10]

        lines = []
        for p in top:
            desc = p.pattern_data.get("description", "")
            if desc:
                lines.append(f"- {desc}  [confianza: {p.confidence:.0%}]")

        if lines:
            section = "\n## Patrones Aprendidos de Correcciones Anteriores\n" + "\n".join(lines)
            prompt += section
            logger.info("[ANALYZER] Injected %d learned patterns (%s)", len(lines), category)

        return prompt

    # ------------------------------------------------------------------
    # Legacy helper kept for compatibility
    # ------------------------------------------------------------------

    def _apply_learned_patterns(self, prompt: str, category: str) -> str:
        """Compatibility shim — delegates to _inject_learned_patterns."""
        return self._inject_learned_patterns(prompt, category)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _calculate_image_hash(self, image_b64: str) -> str:
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        return hashlib.md5(image_b64.encode()).hexdigest()

    def _extract_json_from_response(self, response: dict) -> dict:
        """Normalize different Ollama response formats to a canonical dict."""
        if "error" in response:
            return {"score": 50, "issues": [response["error"]], "reasoning": response.get("error", "")}

        if "score" in response or "tampering_score" in response or "coherence_score" in response:
            return response

        if "coherencia" in response:
            is_valid = bool(response.get("coherencia", False))
            return {
                "score": 80 if is_valid else 30,
                "issues": [] if is_valid else ["Inconsistencia detectada"],
                "is_valid": is_valid,
            }

        if "valid" in response or "is_valid" in response:
            is_valid = response.get("valid", response.get("is_valid", False))
            return {
                "score": 80 if is_valid else 30,
                "issues": [] if is_valid else ["Documento no válido"],
                "is_valid": is_valid,
            }

        if "reasoning" in response or "raw_response" in response:
            reasoning = response.get("reasoning", response.get("raw_response", ""))
            return {
                "score": 50,
                "tampering_score": 50,
                "issues": [],
                "suspicious_areas": [],
                "overall_assessment": reasoning[:200] if reasoning else "Análisis no disponible",
                "reasoning": reasoning,
            }

        return response

    # ------------------------------------------------------------------
    # Main analysis pipeline
    # ------------------------------------------------------------------

    def analyze(
        self,
        selfie_b64: str,
        doc_front_b64: str,
        doc_back_b64: Optional[str],
        form_data: dict,
    ) -> AIAnalysis:
        """Main analysis: coherence check + tampering detection → overall confidence."""

        logger.info("[AI] Starting coherence analysis...")
        coherence = self._analyze_coherence(form_data)
        logger.info("[AI] Coherence result: %s", coherence)

        logger.info("[AI] Starting tampering detection...")
        tampering = self._detect_tampering(doc_front_b64)
        logger.info("[AI] Tampering result: %s", tampering)

        overall = self._calculate_overall_confidence(coherence, tampering)

        coherence_score  = coherence.get("score", 50) if isinstance(coherence, dict) else 50
        coherence_issues = coherence.get("issues", [])  if isinstance(coherence, dict) else []
        tampering_score  = tampering.get("tampering_score", 50) if isinstance(tampering, dict) else 50
        tampering_areas  = tampering.get("suspicious_areas", []) if isinstance(tampering, dict) else []
        extracted_data   = tampering.get("extracted_fields", {}) if isinstance(tampering, dict) else {}

        coherence_reasoning = coherence.get("reasoning", "") if isinstance(coherence, dict) else ""
        tampering_reasoning = tampering.get("overall_assessment", "") if isinstance(tampering, dict) else ""

        return AIAnalysis(
            coherence_score=coherence_score,
            coherence_issues=coherence_issues,
            tampering_score=tampering_score,
            tampering_areas=tampering_areas,
            extracted_data=extracted_data,
            overall_confidence=overall,
            reasoning=f"{coherence_reasoning} {tampering_reasoning}".strip(),
        )

    def _analyze_coherence(self, form_data: dict) -> dict:
        """
        Analyze data coherence.
        Runs Python sanity checks first; if Ollama text model is available,
        also sends the full 3-layer prompt for deeper semantic analysis.
        """
        try:
            # --- Python sanity layer (always runs, fast) ---
            issues = []
            values: Dict[str, str] = {}

            for key, value in form_data.items():
                if value and value.strip():
                    val = value.strip().upper()
                    if val in values:
                        issues.append(f"Valor duplicado '{val}' en {key} y {values[val]}")
                    else:
                        values[val] = key

            required = ["primerNombre", "primerApellido", "numeroDoc", "sexo", "fechaNacimiento"]
            for req_field in required:
                if not form_data.get(req_field, "").strip():
                    issues.append(f"Campo requerido vacío: {req_field}")

            # --- Ollama semantic layer (Phase 3: now active) ---
            prompt = self._build_system_prompt("validate_coherence", form_data)
            if prompt.strip():
                form_text = "\n".join(f"{k}: {v}" for k, v in form_data.items() if v)
                logger.info("[AI] Sending coherence prompt to Ollama text model...")
                ollama_result = self.ollama.analyze_text(form_text, prompt)
                parsed = self._extract_json_from_response(ollama_result)

                # Merge: Ollama issues + Python issues (deduplicated)
                ollama_issues = parsed.get("issues", [])
                for oi in ollama_issues:
                    if oi not in issues:
                        issues.append(oi)

                # Use Ollama score if available, otherwise derive from issues
                score = parsed.get("score", 80 if not issues else 40)
                reasoning = parsed.get("reasoning", "")
            else:
                score = 80 if not issues else 40
                reasoning = ""

            return {
                "score": score,
                "issues": issues[:5],
                "is_valid": len(issues) == 0,
                "reasoning": reasoning,
            }

        except Exception as e:
            logger.error("[AI] Coherence analysis error: %s", str(e))
            return {"score": 50, "issues": [], "is_valid": True, "reasoning": ""}

    def _detect_tampering(self, image_b64: str) -> dict:
        """Detect tampering using vision model with full 3-layer prompt."""
        try:
            prompt = self._build_system_prompt("detect_tampering")
            logger.info("[AI] Calling Ollama for tampering detection...")
            response = self.ollama.analyze_image(image_b64, prompt)
            logger.info("[AI] Raw tampering response: %s", str(response)[:200])
            result = self._extract_json_from_response(response)
            logger.info("[AI] Extracted tampering result: %s", result)
            return result
        except Exception as e:
            logger.error("[AI] Tampering detection error: %s", str(e))
            return {"tampering_score": 50, "suspicious_areas": [], "overall_assessment": f"Error: {str(e)}"}

    def _calculate_overall_confidence(self, coherence: dict, tampering: dict) -> float:
        """Weighted average: coherence 40% + tampering 60%, with per-issue penalty."""
        try:
            coherence_score = coherence.get("score", 50) if isinstance(coherence, dict) else 50
            tampering_score = tampering.get("tampering_score", 50) if isinstance(tampering, dict) else 50

            overall = (coherence_score * 0.4) + (tampering_score * 0.6)

            issues_count = len(coherence.get("issues", []) if isinstance(coherence, dict) else [])
            if issues_count > 0:
                overall *= (1 - (issues_count * 0.1))

            result = round(max(0.0, min(100.0, overall)), 2)
            logger.info("[AI] Confidence: coherence=%.1f, tampering=%.1f, overall=%.1f",
                        coherence_score, tampering_score, result)
            return result
        except Exception as e:
            logger.error("[AI] Error calculating confidence: %s", str(e))
            return 50.0

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def should_reject(self, analysis: AIAnalysis) -> bool:
        return (
            analysis.overall_confidence < OVERALL_CONFIDENCE_THRESHOLD
            or analysis.coherence_score < COHERENCE_THRESHOLD
            or analysis.tampering_score < TAMPERING_THRESHOLD
        )

    def get_analysis_summary(self, analysis: AIAnalysis) -> str:
        status = "APROBADO" if not self.should_reject(analysis) else "RECHAZADO"
        summary = f"Estado: {status}\n"
        summary += f"Confianza General: {analysis.overall_confidence:.1f}%\n"
        summary += f"Coherencia: {analysis.coherence_score:.1f}%\n"
        summary += f"Integridad: {analysis.tampering_score:.1f}%\n"

        if analysis.coherence_issues:
            summary += "\nProblemas de Coherencia:\n"
            for issue in analysis.coherence_issues:
                summary += f"  - {issue}\n"

        if analysis.tampering_areas:
            summary += "\nÁreas Sospechosas:\n"
            for area in analysis.tampering_areas:
                summary += f"  - {area}\n"

        if analysis.reasoning:
            summary += f"\nAnálisis:\n{analysis.reasoning}\n"

        return summary
