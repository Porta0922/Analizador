import os
from pathlib import Path

# Base directory for AI system
AI_BASE_DIR = Path(__file__).parent

# Database
DATABASE_PATH = AI_BASE_DIR / "data" / "ai_data.db"

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "llama3.2:3b")

# CORS
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

# Analysis thresholds
COHERENCE_THRESHOLD = float(os.getenv("COHERENCE_THRESHOLD", "70.0"))
TAMPERING_THRESHOLD = float(os.getenv("TAMPERING_THRESHOLD", "80.0"))
OVERALL_CONFIDENCE_THRESHOLD = float(os.getenv("OVERALL_CONFIDENCE_THRESHOLD", "60.0"))

# Motor de decisión (Fase 1) — veredicto binario APROBADO/RECHAZADO
VERDICT_VERSION = os.getenv("VERDICT_VERSION", "1.0.0")
APPROVAL_THRESHOLD = float(os.getenv("APPROVAL_THRESHOLD", "70.0"))
# Si es true, no se puede aprobar cuando Ollama está caído (dato insuficiente)
AI_REQUIRED_FOR_APPROVAL = os.getenv("AI_REQUIRED_FOR_APPROVAL", "true").lower() in ("1", "true", "yes", "on")

# Antifraude entre solicitudes (Fase 3)
ENABLE_CROSS_REQUEST_FRAUD = os.getenv("ENABLE_CROSS_REQUEST_FRAUD", "true").lower() in ("1", "true", "yes", "on")
# Similitud facial mínima entre la selfie actual y las de análisis previos del mismo CI.
# Por debajo de esto → mismo documento con cara distinta (posible fraude).
FRAUD_RING_SIM_THRESHOLD = float(os.getenv("FRAUD_RING_SIM_THRESHOLD", "60.0"))

# Robustez / latencia (Fase 4)
AI_TAMPERING_CACHE = os.getenv("AI_TAMPERING_CACHE", "true").lower() in ("1", "true", "yes", "on")
AI_TAMPERING_CACHE_SIZE = int(os.getenv("AI_TAMPERING_CACHE_SIZE", "100"))
OLLAMA_CONCURRENCY = int(os.getenv("OLLAMA_CONCURRENCY", "1"))

# Learning parameters
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.1"))
MIN_CORRECTIONS_FOR_PATTERN = int(os.getenv("MIN_CORRECTIONS_FOR_PATTERN", "3"))
PATTERN_CONFIDENCE_THRESHOLD = float(os.getenv("PATTERN_CONFIDENCE_THRESHOLD", "0.7"))

# Data directories
DATA_DIR = AI_BASE_DIR / "data"
VERIFIED_DIR = DATA_DIR / "verified"
CORRECTIONS_DIR = DATA_DIR / "corrections"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
PROMPTS_DIR = AI_BASE_DIR / "prompts"

# Ensure directories exist
for dir_path in [DATA_DIR, VERIFIED_DIR, CORRECTIONS_DIR, EMBEDDINGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
