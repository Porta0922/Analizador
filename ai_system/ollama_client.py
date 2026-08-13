import httpx
import json
from typing import Optional, Dict, Any

from .config import OLLAMA_BASE_URL, OLLAMA_VISION_MODEL, OLLAMA_TEXT_MODEL


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or OLLAMA_BASE_URL
        self.client = httpx.Client(timeout=60.0)
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from response text that might contain other content."""
        import re
        
        if not response_text or not response_text.strip():
            return {"raw_response": "", "reasoning": "Empty response"}
        
        # Try direct JSON parse
        try:
            result = json.loads(response_text)
            if isinstance(result, dict):
                return result
            return {"raw_response": response_text, "reasoning": response_text}
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        code_block = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', response_text)
        if code_block:
            try:
                result = json.loads(code_block.group(1))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                fixed = json_match.group().replace('""', '"').replace('\\n', '\n')
                try:
                    result = json.loads(fixed)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass
        
        # If response contains key terms, create a default structure
        if any(term in response_text.lower() for term in ["score", "issue", "problem", "error"]):
            # Try to extract score from text
            score_match = re.search(r'(?:score|puntuacion|calificacion)[:\s]*(\d+)', response_text, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 50
            
            return {
                "score": score,
                "tampering_score": score,
                "issues": [],
                "reasoning": response_text[:500]
            }
        
        # Return raw text as reasoning if JSON extraction fails
        return {"raw_response": response_text, "reasoning": response_text[:500]}
    
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
