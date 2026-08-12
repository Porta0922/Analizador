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
