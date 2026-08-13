import httpx
import json
from typing import Optional, Dict, Any

from .config import OLLAMA_BASE_URL, OLLAMA_VISION_MODEL, OLLAMA_TEXT_MODEL


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or OLLAMA_BASE_URL
        self.client = httpx.Client(timeout=60.0)
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extrae el primer objeto JSON válido del texto de respuesta del modelo.

        Estrategia (en orden):
          1. Parse directo (modelo devolvió JSON limpio)
          2. Extrae de bloque markdown ```json ... ```
          3. Scanner de profundidad: encuentra el primer '{' y avanza hasta
             encontrar el '}' de cierre que lo empareja — robusto a texto
             antes/después del JSON, arrays anidados, y strings con llaves
          4. Búsqueda de score numérico explícito como último recurso
          5. Fallback con raw_response
        """
        import re

        if not response_text or not response_text.strip():
            return {"raw_response": "", "reasoning": "Empty response"}

        text = response_text.strip()

        # ── 1. Parse directo ─────────────────────────────────────────────────
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # ── 2. Bloque markdown ```json``` ─────────────────────────────────────
        code_block = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
        if code_block:
            try:
                result = json.loads(code_block.group(1).strip())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # ── 3. Scanner de profundidad (el más robusto) ────────────────────────
        # Recorre el texto carácter a carácter llevando la cuenta de {, } y
        # respetando strings (ignora llaves dentro de "...").
        # Devuelve el primer objeto JSON completo que encuentre.
        start = text.find('{')
        if start != -1:
            depth    = 0
            in_str   = False
            escape   = False
            for i, ch in enumerate(text[start:], start):
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_str:
                    escape = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            result = json.loads(candidate)
                            if isinstance(result, dict):
                                return result
                        except json.JSONDecodeError:
                            # Intentar reparaciones menores: comillas simples, trailing commas
                            try:
                                fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                                result = json.loads(fixed)
                                if isinstance(result, dict):
                                    return result
                            except json.JSONDecodeError:
                                pass
                        break  # El primer objeto falló — no seguir buscando

        # ── 4. Extracción de score numérico explícito ─────────────────────────
        score_match = re.search(
            r'(?:face_match_score|back_score|tampering_score|score)["\s:]*(\d+)',
            text, re.IGNORECASE
        )
        if score_match:
            score = min(100, max(0, int(score_match.group(1))))
            # Extraer reasoning del texto libre si existe
            reasoning_match = re.search(r'reasoning["\s:]*["\']([^"\']{10,200})', text, re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else text[:200]
            return {
                "score":           score,
                "face_match_score": score,
                "tampering_score": score,
                "back_score":      score,
                "issues":          [],
                "reasoning":       reasoning,
            }

        # ── 5. Fallback total ─────────────────────────────────────────────────
        return {"raw_response": text[:500], "reasoning": text[:500]}
    
    def get_available_models(self) -> list:
        """Get list of available Ollama models."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except Exception:
            return []
    
    def analyze_image(self, image_b64: str, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Analyze an image using a vision model."""
        model = model or OLLAMA_VISION_MODEL
        
        # Remove data URI prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9
            }
        }
        
        try:
            import logging
            logger = logging.getLogger("ai-system")
            
            logger.info("[OLLAMA] Sending request to %s with model %s", self.base_url, model)
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            logger.info("[OLLAMA] Response status: %d", response.status_code)
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "")
                logger.info("[OLLAMA] Raw response (first 200 chars): %s", str(raw_response)[:200])
                parsed = self._parse_json_response(raw_response)
                logger.info("[OLLAMA] Parsed response: %s", parsed)
                return parsed
            else:
                logger.error("[OLLAMA] HTTP error: %d - %s", response.status_code, response.text[:200])
                return {"error": f"HTTP {response.status_code}", "details": response.text[:500]}
        
        except httpx.ConnectError:
            return {"error": "connection_failed", "message": "No se pudo conectar a Ollama. Asegúrese de que esté ejecutándose."}
        except Exception as e:
            import logging
            logger = logging.getLogger("ai-system")
            logger.error("[OLLAMA] Exception: %s", str(e))
            return {"error": str(e)}
    
    def analyze_image_pair(
        self,
        image_a_b64: str,
        image_b_b64: str,
        prompt: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Opción D — Compara dos imágenes enviándolas juntas en el array 'images'.
        llava procesa las imágenes en orden: la primera es la imagen A (selfie),
        la segunda es la imagen B (foto del documento).
        Ambos b64 deben estar sin prefijo data URI.
        """
        import logging
        logger = logging.getLogger("ai-system")
        model = model or OLLAMA_VISION_MODEL

        payload = {
            "model": model,
            "prompt": prompt,
            "images": [image_a_b64, image_b_b64],
            "stream": False,
            "options": {
                "temperature": 0.2,   # más determinista para comparación biométrica
                "top_p": 0.9,
            },
        }

        try:
            logger.info("[OLLAMA] Sending image-pair request to %s (model=%s)", self.base_url, model)
            response = self.client.post(f"{self.base_url}/api/generate", json=payload)
            logger.info("[OLLAMA] Response status: %d", response.status_code)

            if response.status_code == 200:
                result      = response.json()
                raw_response = result.get("response", "")
                logger.info("[OLLAMA] Raw face-pair response (200 chars): %s", str(raw_response)[:200])
                return self._parse_json_response(raw_response)
            else:
                logger.error("[OLLAMA] HTTP error: %d - %s", response.status_code, response.text[:200])
                return {"error": f"HTTP {response.status_code}", "face_match_score": -1}

        except httpx.ConnectError:
            return {"error": "connection_failed", "face_match_score": -1,
                    "message": "No se pudo conectar a Ollama."}
        except Exception as e:
            logger.error("[OLLAMA] Exception in analyze_image_pair: %s", str(e))
            return {"error": str(e), "face_match_score": -1}

    def analyze_text(self, text: str, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Analyze text using a language model."""
        model = model or OLLAMA_TEXT_MODEL
        
        full_prompt = f"{prompt}\n\nDatos para analizar:\n{text}"
        
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9
            }
        }
        
        try:
            import logging
            logger = logging.getLogger("ai-system")
            
            logger.info("[OLLAMA] Sending text request to model %s", model)
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            logger.info("[OLLAMA] Response status: %d", response.status_code)
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get("response", "")
                logger.info("[OLLAMA] Raw response (first 200 chars): %s", str(raw_response)[:200])
                parsed = self._parse_json_response(raw_response)
                logger.info("[OLLAMA] Parsed response: %s", parsed)
                return parsed
            else:
                logger.error("[OLLAMA] HTTP error: %d - %s", response.status_code, response.text[:200])
                return {"error": f"HTTP {response.status_code}", "details": response.text[:500]}
        
        except httpx.ConnectError:
            return {"error": "connection_failed", "message": "No se pudo conectar a Ollama. Asegúrese de que está ejecutándose."}
        except Exception as e:
            import logging
            logger = logging.getLogger("ai-system")
            logger.error("[OLLAMA] Exception: %s", str(e))
            return {"error": str(e)}
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
