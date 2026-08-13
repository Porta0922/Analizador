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
            "analyze_document":      "analyze_document.txt",
            "validate_coherence":    "validate_coherence.txt",
            "detect_tampering":      "detect_tampering.txt",
            "verify_face_match":     "verify_face_match.txt",
            "verify_back_document":  "verify_back_document.txt",
            "extract_fields_visual": "extract_fields_visual.txt",
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

    # Prompts que son puramente visuales (comparan imágenes) — NO deben recibir
    # reglas de dominio de texto ni patrones aprendidos de campos de formulario.
    # Agregarles LOCALE_RULES confunde al modelo (llava devuelve esas reglas
    # en su reasoning y corrompe el JSON de respuesta).
    _VISUAL_ONLY_PROMPTS = {"verify_face_match", "verify_back_document", "extract_fields_visual"}

    def _build_system_prompt(self, base_prompt_key: str, form_data: Optional[dict] = None) -> str:
        """
        Build a layered system prompt:
          Layer 1 — Base prompt (from prompts/*.txt)
          Layer 2 — Locale / domain rules  [SOLO para prompts de texto]
          Layer 3 — Dynamic learned patterns [SOLO para prompts de texto]

        Los prompts visuales (verify_face_match, verify_back_document) usan
        solo el Layer 1 para evitar contaminar la respuesta del modelo de visión
        con reglas de texto que no son relevantes para la comparación facial.
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
                pass

        # Prompts visuales: devolver solo el base — sin reglas de dominio ni patrones
        if base_prompt_key in self._VISUAL_ONLY_PROMPTS:
            return base

        # --- Layer 2: Locale rules (solo prompts de texto/documento) ---
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
        ocr_field_matches: Optional[Dict[str, Optional[bool]]] = None,
    ) -> AIAnalysis:
        """
        Pipeline completo de análisis:
          1. Coherencia de datos del formulario (texto + Ollama)
          2. Detección de tampering en el frente (visión llava)
          3. Análisis facial: selfie vs foto del documento (visión llava)
          4. Verificación del dorso del documento (visión llava)
          5. [NUEVO] Etapa 2 de verificación de campos: llava extrae visualmente
             los campos que el OCR no pudo confirmar y los re-evalúa
          6. Cálculo de confianza global ponderando los 4 factores
        """
        logger.info("[AI] Starting coherence analysis...")
        coherence = self._analyze_coherence(form_data)
        logger.info("[AI] Coherence result: %s", coherence)

        logger.info("[AI] Starting tampering detection...")
        tampering = self._detect_tampering(doc_front_b64)
        logger.info("[AI] Tampering result: %s", tampering)

        # Análisis facial por IA
        logger.info("[AI] Starting face match analysis...")
        face_match = self._analyze_face_match(selfie_b64, doc_front_b64)
        logger.info("[AI] Face match result: %s", face_match)

        # Verificación del dorso
        back_analysis: Optional[dict] = None
        if doc_back_b64:
            logger.info("[AI] Starting back document verification...")
            back_analysis = self._verify_back_document(doc_back_b64)
            logger.info("[AI] Back document result: %s", back_analysis)

        # ── Etapa 2: extracción visual de campos que el OCR no confirmó ───────
        visual_field_matches: Dict[str, Optional[bool]] = {}
        visual_extracted: Dict[str, str] = {}

        if form_data and ocr_field_matches is not None:
            # Campos que OCR marcó como False (falló) o None (no aplicable)
            # Solo re-intentamos los que son False — los None son no-verificables
            failed_fields = {
                field: value
                for field, value in form_data.items()
                if ocr_field_matches.get(field) is False and value
            }

            if failed_fields:
                logger.info("[AI] Etapa 2: enviando %d campos fallidos a llava: %s",
                            len(failed_fields), list(failed_fields.keys()))
                visual_result = self._extract_fields_visual(doc_front_b64, failed_fields)
                visual_extracted = visual_result.get("extracted_fields", {})

                # Re-evaluar cada campo con lo que llava extrajo
                for field, expected_value in failed_fields.items():
                    extracted = visual_extracted.get(field)
                    if extracted is None:
                        visual_field_matches[field] = False  # llava tampoco lo vio
                    else:
                        # Comparación normalizada (sin tildes, mayúsculas, sin separadores)
                        match = self._fuzzy_field_match(field, expected_value, extracted)
                        visual_field_matches[field] = match
                        logger.info("[AI] Campo '%s': OCR=False → llava='%s' → match=%s",
                                    field, extracted, match)
            else:
                logger.info("[AI] Etapa 2: no hay campos fallidos para re-evaluar con llava")
        else:
            logger.info("[AI] Etapa 2: omitida (no hay ocr_field_matches)")

        overall = self._calculate_overall_confidence(coherence, tampering, face_match, back_analysis)

        # Extraer valores con defaults seguros
        coherence_score  = coherence.get("score", 50)            if isinstance(coherence, dict)  else 50
        coherence_issues = coherence.get("issues", [])           if isinstance(coherence, dict)  else []
        tampering_score  = tampering.get("tampering_score", 50)  if isinstance(tampering, dict)  else 50
        tampering_areas  = tampering.get("suspicious_areas", []) if isinstance(tampering, dict)  else []
        extracted_data   = tampering.get("extracted_fields", {}) if isinstance(tampering, dict)  else {}

        face_score       = face_match.get("face_match_score", -1) if isinstance(face_match, dict) else -1
        face_issues      = face_match.get("issues", [])           if isinstance(face_match, dict) else []
        face_reasoning   = face_match.get("reasoning", "")        if isinstance(face_match, dict) else ""

        back_score       = back_analysis.get("back_score", -1)   if isinstance(back_analysis, dict) else -1
        back_issues      = back_analysis.get("issues", [])        if isinstance(back_analysis, dict) else []

        coherence_reasoning = coherence.get("reasoning", "")         if isinstance(coherence, dict)  else ""
        tampering_reasoning = tampering.get("overall_assessment", "") if isinstance(tampering, dict) else ""

        # Merge extracted_data con lo que extrajo llava en etapa 2
        extracted_data.update(visual_extracted)

        return AIAnalysis(
            coherence_score=coherence_score,
            coherence_issues=coherence_issues,
            tampering_score=tampering_score,
            tampering_areas=tampering_areas,
            extracted_data=extracted_data,
            overall_confidence=overall,
            reasoning=f"{coherence_reasoning} {tampering_reasoning}".strip(),
            face_match_score=face_score,
            face_match_issues=face_issues,
            face_match_reasoning=face_reasoning,
            back_analysis_score=back_score,
            back_analysis_issues=back_issues,
            visual_field_matches=visual_field_matches,
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

    def _calculate_overall_confidence(
        self,
        coherence: dict,
        tampering: dict,
        face_match: Optional[dict] = None,
        back_analysis: Optional[dict] = None,
    ) -> float:
        """
        Confianza global ponderada:
          - Coherencia de datos:  25%
          - Integridad (tampering): 35%
          - Coincidencia facial IA: 25%  (si disponible, sino se distribuye entre los otros)
          - Verificación dorso:    15%  (si disponible)

        Si el análisis facial no está disponible (face_match_score == -1),
        los pesos se redistribuyen a coherencia 35% + tampering 65%.
        """
        try:
            coherence_score = coherence.get("score", 50) if isinstance(coherence, dict) else 50
            tampering_score = tampering.get("tampering_score", 50) if isinstance(tampering, dict) else 50

            face_score = face_match.get("face_match_score", -1) if isinstance(face_match, dict) else -1
            back_score = back_analysis.get("back_score", -1)    if isinstance(back_analysis, dict) else -1

            has_face = face_score >= 0
            has_back = back_score >= 0

            if has_face and has_back:
                # Todos disponibles: 25 + 35 + 25 + 15 = 100
                overall = (coherence_score * 0.25 + tampering_score * 0.35 +
                           face_score * 0.25 + back_score * 0.15)
            elif has_face and not has_back:
                # Sin dorso: 25 + 40 + 35 = 100
                overall = (coherence_score * 0.25 + tampering_score * 0.40 +
                           face_score * 0.35)
            elif has_back and not has_face:
                # Sin face match: 30 + 55 + 15 = 100
                overall = (coherence_score * 0.30 + tampering_score * 0.55 +
                           back_score * 0.15)
            else:
                # Solo coherencia + tampering
                overall = coherence_score * 0.40 + tampering_score * 0.60

            # Penalización por issues de coherencia
            issues_count = len(coherence.get("issues", []) if isinstance(coherence, dict) else [])
            if issues_count > 0:
                overall *= (1 - issues_count * 0.08)

            # Penalización severa si face_match indica fraude explícito
            if has_face and face_score < 20:
                overall = min(overall, 30.0)  # capear en 30 si la IA ve personas distintas

            # Penalización si el dorso parece ser el frente (is_back=False)
            if has_back and isinstance(back_analysis, dict):
                if not back_analysis.get("is_back", True):
                    overall *= 0.85  # -15% por dorso incorrecto

            result = round(max(0.0, min(100.0, overall)), 2)
            logger.info(
                "[AI] Confidence: coh=%.1f tam=%.1f face=%.1f back=%.1f → overall=%.1f",
                coherence_score, tampering_score,
                face_score if has_face else -1,
                back_score if has_back else -1,
                result,
            )
            return result
        except Exception as e:
            logger.error("[AI] Error calculating confidence: %s", str(e))
            return 50.0

    # ------------------------------------------------------------------
    # Opción D — Análisis facial por IA (selfie vs foto del documento)
    # ------------------------------------------------------------------

    def _analyze_face_match(self, selfie_b64: str, doc_front_b64: str) -> dict:
        """
        Usa llava:7b para comparar la selfie con la foto del documento.
        Ollama no puede procesar dos imágenes en el mismo prompt nativo,
        por lo que se usa un truco: enviar las dos imágenes en el array
        'images' y pedirle al modelo que compare la primera con la segunda.
        """
        try:
            prompt = self._build_system_prompt("verify_face_match")
            if not prompt.strip():
                return {"face_match_score": -1, "issues": [], "reasoning": "Prompt no disponible"}

            logger.info("[AI] Calling Ollama for face match (selfie vs doc)...")

            # Limpiar prefijos data URI de ambas imágenes
            def strip_prefix(b64: str) -> str:
                return b64.split(",", 1)[1] if "," in b64 else b64

            selfie_clean   = strip_prefix(selfie_b64)
            doc_front_clean = strip_prefix(doc_front_b64)

            # Pasar ambas imágenes en el array — llava las procesa en orden
            response = self.ollama.analyze_image_pair(selfie_clean, doc_front_clean, prompt)
            logger.info("[AI] Raw face match response: %s", str(response)[:200])

            result = self._extract_json_from_response(response)

            # Normalizar campo face_match_score desde posibles variantes
            if "face_match_score" not in result:
                if "score" in result:
                    result["face_match_score"] = result["score"]
                elif "similarity" in result:
                    result["face_match_score"] = result["similarity"]
                else:
                    result["face_match_score"] = 50

            # Asegurar que issues y reasoning existan
            if "issues" not in result:
                result["issues"] = []
            if "reasoning" not in result:
                result["reasoning"] = result.get("overall_assessment", "")

            return result

        except Exception as e:
            logger.error("[AI] Face match analysis error: %s", str(e))
            return {"face_match_score": -1, "issues": [f"Error: {str(e)}"], "reasoning": ""}

    # ------------------------------------------------------------------
    # Opción D — Verificación del dorso del documento
    # ------------------------------------------------------------------

    def _verify_back_document(self, doc_back_b64: str) -> dict:
        """
        Usa llava:7b para verificar que la imagen provista es el dorso real
        del documento y no la cara frontal subida por error.
        """
        try:
            prompt = self._build_system_prompt("verify_back_document")
            if not prompt.strip():
                return {"back_score": -1, "is_back": True, "issues": []}

            logger.info("[AI] Calling Ollama for back document verification...")
            response = self.ollama.analyze_image(doc_back_b64, prompt)
            logger.info("[AI] Raw back document response: %s", str(response)[:200])

            result = self._extract_json_from_response(response)

            # Normalizar campo back_score
            if "back_score" not in result:
                if "score" in result:
                    result["back_score"] = result["score"]
                elif "tampering_score" in result:
                    result["back_score"] = result["tampering_score"]
                else:
                    result["back_score"] = 50

            if "is_back" not in result:
                result["is_back"] = result.get("back_score", 50) >= 50

            if "issues" not in result:
                result["issues"] = []

            return result

        except Exception as e:
            logger.error("[AI] Back document verification error: %s", str(e))
            return {"back_score": -1, "is_back": True, "issues": [f"Error: {str(e)}"]}

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
