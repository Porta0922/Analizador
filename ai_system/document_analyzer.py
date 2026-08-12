import json
import hashlib
import base64
from pathlib import Path
from typing import Optional, Dict, Any

from .config import (
    COHERENCE_THRESHOLD,
    TAMPERING_THRESHOLD,
    OVERALL_CONFIDENCE_THRESHOLD,
    PROMPTS_DIR
)
from .ollama_client import OllamaClient
from .database import Database
from .models import AIAnalysis, LearnedPattern


class DocumentAnalyzer:
    def __init__(self, ollama_client: OllamaClient, db: Database):
        self.ollama = ollama_client
        self.db = db
        self.patterns = self._load_learned_patterns()
        self._load_prompts()
    
    def _load_prompts(self):
        """Load prompt templates from files."""
        self.prompts = {}
        prompt_files = {
            "analyze_document": "analyze_document.txt",
            "validate_coherence": "validate_coherence.txt",
            "detect_tampering": "detect_tampering.txt"
        }
        
        for key, filename in prompt_files.items():
            prompt_path = PROMPTS_DIR / filename
            if prompt_path.exists():
                self.prompts[key] = prompt_path.read_text(encoding="utf-8")
            else:
                self.prompts[key] = ""
    
    def _load_learned_patterns(self) -> Dict[str, list]:
        """Load learned patterns from database."""
        patterns = {}
        for pattern_type in ["field_regex", "common_error", "tampering_sign"]:
            patterns[pattern_type] = self.db.get_patterns_by_type(pattern_type)
        return patterns
    
    def _calculate_image_hash(self, image_b64: str) -> str:
        """Calculate simple hash for image deduplication."""
        # Remove data URI prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        
        return hashlib.md5(image_b64.encode()).hexdigest()
    
    def _build_coherence_prompt(self, form_data: dict) -> str:
        """Build coherence validation prompt with form data."""
        template = self.prompts.get("validate_coherence", "")
        
        # Fill template with actual data
        filled = template.format(
            nombre=form_data.get("primerNombre", ""),
            apellido=form_data.get("primerApellido", ""),
            numero=form_data.get("numeroDoc", ""),
            tipo=form_data.get("tipoDoc", ""),
            sexo=form_data.get("sexo", ""),
            fecha=form_data.get("fechaNacimiento", "")
        )
        
        # Apply learned patterns
        filled = self._apply_learned_patterns(filled, "coherence")
        
        return filled
    
    def _apply_learned_patterns(self, prompt: str, pattern_type: str) -> str:
        """Inject learned patterns into prompt."""
        patterns = self.patterns.get(pattern_type, [])
        
        if not patterns:
            return prompt
        
        # Get top patterns by confidence
        top_patterns = sorted(patterns, key=lambda p: p.confidence, reverse=True)[:5]
        
        pattern_context = "\n".join([
            f"- {p.pattern_data.get('description', '')}"
            for p in top_patterns
            if p.confidence > 0.5
        ])
        
        if pattern_context:
            prompt += f"\n\nPatrones aprendidos previamente:\n{pattern_context}"
        
        return prompt
    
    def _extract_json_from_response(self, response: dict) -> dict:
        """Extract JSON data from Ollama response."""
        if "error" in response:
            return {"score": 50, "issues": [response["error"]], "reasoning": response.get("error", "")}
        
        # If it's already parsed JSON with expected fields
        if "score" in response or "tampering_score" in response or "coherence_score" in response:
            return response
        
        # Handle different response formats from the model
        # Common variations: {"coherencia": false}, {"valid": true}, etc.
        if "coherencia" in response:
            return {
                "score": 80 if response.get("coherencia", False) else 30,
                "issues": [] if response.get("coherencia", False) else ["Inconsistencia detectada"],
                "is_valid": response.get("coherencia", False)
            }
        
        if "valid" in response or "is_valid" in response:
            is_valid = response.get("valid", response.get("is_valid", False))
            return {
                "score": 80 if is_valid else 30,
                "issues": [] if is_valid else ["Documento no válido"],
                "is_valid": is_valid
            }
        
        # If we have a reasoning field but no structured data, create defaults
        if "reasoning" in response or "raw_response" in response:
            reasoning = response.get("reasoning", response.get("raw_response", ""))
            return {
                "score": 50,
                "tampering_score": 50,
                "issues": [],
                "suspicious_areas": [],
                "overall_assessment": reasoning[:200] if reasoning else "Análisis no disponible",
                "reasoning": reasoning
            }
        
        return response
    
    def analyze(
        self,
        selfie_b64: str,
        doc_front_b64: str,
        doc_back_b64: Optional[str],
        form_data: dict
    ) -> AIAnalysis:
        """Main analysis method that coordinates all AI checks."""
        
        import logging
        logger = logging.getLogger("ai-system")
        
        # 1. Coherence validation (simple checks only)
        logger.info("[AI] Starting coherence analysis...")
        coherence = self._analyze_coherence(form_data)
        logger.info("[AI] Coherence result: %s", coherence)
        
        # 2. Tampering detection
        logger.info("[AI] Starting tampering detection...")
        tampering = self._detect_tampering(doc_front_b64)
        logger.info("[AI] Tampering result: %s", tampering)
        
        # 3. Overall confidence calculation
        overall = self._calculate_overall_confidence(coherence, tampering)
        
        # Safely extract values with defaults
        coherence_score = coherence.get("score", 50) if isinstance(coherence, dict) else 50
        coherence_issues = coherence.get("issues", []) if isinstance(coherence, dict) else []
        tampering_score = tampering.get("tampering_score", 50) if isinstance(tampering, dict) else 50
        tampering_areas = tampering.get("suspicious_areas", []) if isinstance(tampering, dict) else []
        
        extracted_data = tampering.get("extracted_fields", {}) if isinstance(tampering, dict) else {}
        
        coherence_reasoning = coherence.get("reasoning", "") if isinstance(coherence, dict) else ""
        tampering_reasoning = tampering.get("overall_assessment", "") if isinstance(tampering, dict) else ""
        
        return AIAnalysis(
            coherence_score=coherence_score,
            coherence_issues=coherence_issues,
            tampering_score=tampering_score,
            tampering_areas=tampering_areas,
            extracted_data=extracted_data,
            overall_confidence=overall,
            reasoning=f"{coherence_reasoning} {tampering_reasoning}".strip()
        )
    
    def _analyze_coherence(self, form_data: dict) -> dict:
        """Analyze data coherence - simple checks only."""
        import logging
        logger = logging.getLogger("ai-system")
        
        try:
            prompt = self.prompts.get("validate_coherence", "")
            
            # Simple validation: check for duplicates and empty fields
            issues = []
            values = {}
            
            for key, value in form_data.items():
                if value and value.strip():
                    val = value.strip().upper()
                    if val in values:
                        issues.append(f"Valor duplicado '{val}' en {key} y {values[val]}")
                    else:
                        values[val] = key
            
            # Check for empty required fields
            required = ["primerNombre", "primerApellido", "numeroDoc", "sexo", "fechaNacimiento"]
            for field in required:
                if not form_data.get(field, "").strip():
                    issues.append(f"Campo {field} vacio")
            
            score = 80 if not issues else 40
            
            return {
                "score": score,
                "issues": issues[:3],  # Max 3 issues
                "is_valid": len(issues) == 0
            }
        except Exception as e:
            logger.error("[AI] Coherence analysis error: %s", str(e))
            return {"score": 50, "issues": [], "is_valid": True}
    
    def _detect_tampering(self, image_b64: str) -> dict:
        """Detect tampering using vision model."""
        import logging
        logger = logging.getLogger("ai-system")
        
        try:
            prompt = self.prompts.get("detect_tampering", "")
            prompt = self._apply_learned_patterns(prompt, "tampering")
            
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
        """Calculate overall confidence score from individual analyses."""
        import logging
        logger = logging.getLogger("ai-system")
        
        try:
            coherence_score = coherence.get("score", 50) if isinstance(coherence, dict) else 50
            tampering_score = tampering.get("tampering_score", 50) if isinstance(tampering, dict) else 50
            
            # Weighted average (coherence 40%, tampering 60%)
            overall = (coherence_score * 0.4) + (tampering_score * 0.6)
            
            # Apply penalties for issues
            issues = coherence.get("issues", []) if isinstance(coherence, dict) else []
            issues_count = len(issues)
            if issues_count > 0:
                overall *= (1 - (issues_count * 0.1))
            
            result = round(max(0, min(100, overall)), 2)
            logger.info("[AI] Confidence calculation: coherence=%s, tampering=%s, overall=%s", 
                       coherence_score, tampering_score, result)
            return result
        except Exception as e:
            logger.error("[AI] Error calculating confidence: %s", str(e))
            return 50.0
    
    def should_reject(self, analysis: AIAnalysis) -> bool:
        """Determine if analysis should be automatically rejected."""
        return (
            analysis.overall_confidence < OVERALL_CONFIDENCE_THRESHOLD or
            analysis.coherence_score < COHERENCE_THRESHOLD or
            analysis.tampering_score < TAMPERING_THRESHOLD
        )
    
    def get_analysis_summary(self, analysis: AIAnalysis) -> str:
        """Generate human-readable summary of analysis."""
        status = "APROBADO" if not self.should_reject(analysis) else "RECHAZADO"
        
        summary = f"Estado: {status}\n"
        summary += f"Confianza General: {analysis.overall_confidence:.1f}%\n"
        summary += f"Coherencia: {analysis.coherence_score:.1f}%\n"
        summary += f"Integridad: {analysis.tampering_score:.1f}%\n"
        
        if analysis.coherence_issues:
            summary += f"\nProblemas de Coherencia:\n"
            for issue in analysis.coherence_issues:
                summary += f"  - {issue}\n"
        
        if analysis.tampering_areas:
            summary += f"\nÁreas Sospechosas:\n"
            for area in analysis.tampering_areas:
                summary += f"  - {area}\n"
        
        if analysis.reasoning:
            summary += f"\nAnálisis:\n{analysis.reasoning}\n"
        
        return summary
