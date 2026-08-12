"""
seed_patterns.py — Siembra patrones de dominio Paraguay en la DB.

Ejecutar una sola vez (o idempotentemente) para garantizar que el sistema
ya tiene reglas activas desde el primer arranque, sin necesitar correcciones
previas del usuario.

Uso:
    python -m ai_system.seed_patterns
    # o desde la raíz del proyecto:
    python -c "from ai_system.seed_patterns import seed; seed()"
"""

import logging
from .database import Database
from .models import LearnedPattern

logger = logging.getLogger("ai-system")

# ---------------------------------------------------------------------------
# Patrones semilla — Paraguay / LATAM
# Cada entrada sigue el mismo formato que LearnedPattern.pattern_data
# ---------------------------------------------------------------------------
SEED_PATTERNS = [
    # --- Nombres válidos frecuentemente marcados como error ---
    {
        "pattern_type": "name_format",
        "confidence": 0.95,
        "description": "'BANILDO' es un nombre válido en Paraguay. No marcar como error ortográfico.",
        "field_name": "primerNombre",
        "expected_pattern": "BANILDO",
    },
    {
        "pattern_type": "name_format",
        "confidence": 0.95,
        "description": "'SINDULFO' es un nombre válido en Paraguay. No marcar como error ortográfico.",
        "field_name": "primerNombre",
        "expected_pattern": "SINDULFO",
    },
    {
        "pattern_type": "name_format",
        "confidence": 0.95,
        "description": "'WILFRIDO' es un nombre válido en Paraguay. No marcar como error ortográfico.",
        "field_name": "primerNombre",
        "expected_pattern": "WILFRIDO",
    },
    {
        "pattern_type": "name_format",
        "confidence": 0.95,
        "description": "'NATANAEL' es un nombre válido en Paraguay. No marcar como error ortográfico.",
        "field_name": "primerNombre",
        "expected_pattern": "NATANAEL",
    },
    # --- Apellidos compuestos ---
    {
        "pattern_type": "name_format",
        "confidence": 0.92,
        "description": "'MORALES FERNANDEZ' es un apellido compuesto válido. No tratar como duplicado.",
        "field_name": "primerApellido",
        "expected_pattern": "MORALES FERNANDEZ",
    },
    {
        "pattern_type": "name_format",
        "confidence": 0.92,
        "description": "Los apellidos compuestos de dos palabras son normales en Paraguay. No reportar como inconsistencia.",
        "field_name": "primerApellido",
        "expected_pattern": "__compound__",
    },
    # --- Tipo de documento ---
    {
        "pattern_type": "document_type",
        "confidence": 0.98,
        "description": "'CÉDULA DE IDENTIDAD POLICIAL' es un tipo de documento oficial válido en Paraguay.",
        "field_name": "tipoDoc",
        "expected_pattern": "CÉDULA DE IDENTIDAD POLICIAL",
    },
    {
        "pattern_type": "document_type",
        "confidence": 0.98,
        "description": "'CÉDULA DE IDENTIDAD' es el tipo de documento de identidad estándar en Paraguay.",
        "field_name": "tipoDoc",
        "expected_pattern": "CÉDULA DE IDENTIDAD",
    },
    # --- Número de CI ---
    {
        "pattern_type": "number_format",
        "confidence": 0.90,
        "description": "El número de CI paraguayo puede tener entre 6 y 8 dígitos con o sin puntos (ej. '1.234.567' o '1234567').",
        "field_name": "numeroDoc",
        "expected_pattern": r"^\d{1,3}\.?\d{3}\.?\d{3,4}$",
    },
    # --- Formato de sexo ---
    {
        "pattern_type": "gender_format",
        "confidence": 0.95,
        "description": "El campo sexo acepta: M, F, MASCULINO, FEMENINO. Todos son válidos.",
        "field_name": "sexo",
        "expected_pattern": "M|F|MASCULINO|FEMENINO",
    },
    # --- Formato de fecha ---
    {
        "pattern_type": "date_format",
        "confidence": 0.90,
        "description": "Las fechas en documentos paraguayos pueden venir en formato DD/MM/YYYY o DD-MM-YYYY.",
        "field_name": "fechaNacimiento",
        "expected_pattern": r"^\d{2}[/\-]\d{2}[/\-]\d{4}$",
    },
]


def seed(force: bool = False) -> int:
    """
    Insert seed patterns into the DB.

    Args:
        force: If True, re-insert even if seed patterns already exist.
               If False (default), skip if the DB already has ≥ 5 patterns.

    Returns:
        Number of patterns inserted.
    """
    db = Database()

    if not force:
        stats = db.get_statistics()
        if stats["learned_patterns"] >= 5:
            logger.info("[SEED] DB already has %d patterns — skipping seed.", stats["learned_patterns"])
            return 0

    inserted = 0
    for entry in SEED_PATTERNS:
        pattern_data = {
            "field_name": entry["field_name"],
            "expected_pattern": entry["expected_pattern"],
            "description": entry["description"],
            "source": "seed_paraguay",
        }
        pattern = LearnedPattern(
            pattern_type=entry["pattern_type"],
            pattern_data=pattern_data,
            confidence=entry["confidence"],
            times_applied=0,
            success_rate=1.0,
        )
        db.save_pattern(pattern)
        inserted += 1
        logger.info("[SEED] Inserted: %s", entry["description"][:60])

    logger.info("[SEED] Done. %d patterns inserted.", inserted)
    return inserted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = seed()
    print(f"Seed complete: {count} patterns inserted.")
