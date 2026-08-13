import base64
import logging
import os
import re
import traceback

import cv2
import easyocr
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import InceptionResnetV1, MTCNN
from pydantic import BaseModel, Field, field_validator
from torch.nn.functional import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("id-verifier")

# ---------------------------------------------------------------------------
# Configuración por variables de entorno
# ---------------------------------------------------------------------------
MAX_IMAGE_B64_LENGTH    = int(os.getenv("MAX_IMAGE_B64_LENGTH",   "15_000_000"))
MAX_FORM_FIELDS         = int(os.getenv("MAX_FORM_FIELDS",        "20"))
MAX_FIELD_VALUE_LENGTH  = int(os.getenv("MAX_FIELD_VALUE_LENGTH", "200"))
OCR_MIN_CONFIDENCE      = float(os.getenv("OCR_MIN_CONFIDENCE",   "0.4"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

# ---------------------------------------------------------------------------
# Opciones A — Thresholds faciales dinámicos por calidad de detección
# ---------------------------------------------------------------------------
# "both"   → ambas caras detectadas correctamente — exigente
# "one"    → solo una cara detectada              — tolerante
# "none"   → ninguna cara detectada               → RECHAZO automático

FACE_THRESHOLD_BOTH  = float(os.getenv("FACE_THRESHOLD_BOTH",  "65.0"))
FACE_THRESHOLD_ONE   = float(os.getenv("FACE_THRESHOLD_ONE",   "55.0"))

# Opción C — Si selfie y documento superan este umbral de similitud entre
# ellos → probable fraude (misma imagen subida dos veces)
SELFIE_DOC_FRAUD_THRESHOLD = float(os.getenv("SELFIE_DOC_FRAUD_THRESHOLD", "92.0"))

# Compatibilidad hacia atrás: FACE_SIMILARITY_THRESHOLD se mantiene como alias
# para el umbral "both". Si alguien ya lo tiene configurado en el entorno,
# se usa como base para ambos thresholds.
_legacy = float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0"))
if _legacy > 0:
    FACE_THRESHOLD_BOTH = _legacy
    FACE_THRESHOLD_ONE  = max(45.0, _legacy - 10.0)

# ---------------------------------------------------------------------------
# Opciones B — Estados del dorso del documento
# ---------------------------------------------------------------------------
# "ok"          → dorso provisto y diferente al frente
# "same_as_front" → dorso idéntico o casi idéntico al frente (posible error)
# "duplicate"   → frente y dorso superan umbral genérico de duplicados (≥70%)
BACK_IDENTICAL_THRESHOLD = float(os.getenv("BACK_IDENTICAL_THRESHOLD", "92.0"))

app = FastAPI(title="Verificación de Identidad Local GPU", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Inicialización de modelos (una sola vez al arrancar)
# ---------------------------------------------------------------------------
use_gpu = os.getenv("USE_GPU", "").lower() not in ("0", "false", "no") and torch.cuda.is_available()
device  = torch.device("cuda:0" if use_gpu else "cpu")
logger.info("--> Ejecutando backend en: %s", device)
if use_gpu:
    logger.info("--> GPU detectada: %s", torch.cuda.get_device_name(0))

ocr_reader = easyocr.Reader(["es"], gpu=use_gpu)

mtcnn = MTCNN(
    keep_all=False,
    device=device,
    min_face_size=30,
    thresholds=[0.6, 0.7, 0.7],
    post_process=True,
)

face_net = InceptionResnetV1(pretrained="vggface2").eval().to(device)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------------------------------------------------------
# Modelos de petición
# ---------------------------------------------------------------------------
class VerificationRequest(BaseModel):
    selfie_b64:       str  = Field(min_length=1, max_length=MAX_IMAGE_B64_LENGTH)
    id_card_b64:      str  = Field(min_length=1, max_length=MAX_IMAGE_B64_LENGTH)
    id_card_back_b64: str  = Field(default="",   max_length=MAX_IMAGE_B64_LENGTH)
    form_data:        dict = Field(default_factory=dict, max_length=MAX_FORM_FIELDS)

    @field_validator("selfie_b64", "id_card_b64")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        candidate = value.split(",", 1)[1] if "," in value else value
        if not re.fullmatch(r"[A-Za-z0-9+/=]+", candidate):
            raise ValueError("El payload base64 contiene caracteres inválidos.")
        try:
            base64.b64decode(candidate, validate=True)
        except Exception as exc:
            raise ValueError("El payload base64 está corrupto.") from exc
        return value

    @field_validator("form_data")
    @classmethod
    def validate_form_data(cls, value: dict) -> dict:
        for key, val in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Las claves de form_data deben ser strings no vacíos.")
            if not isinstance(val, str):
                raise ValueError(f"El campo '{key}' debe ser un string.")
            if len(val) > MAX_FIELD_VALUE_LENGTH:
                raise ValueError(f"El campo '{key}' excede la longitud máxima.")
        return value


# ---------------------------------------------------------------------------
# Utilidades de imagen
# ---------------------------------------------------------------------------
def b64_to_cv2(b64_string: str):
    """Decodifica base64 (con o sin prefijo data URI) a imagen OpenCV BGR."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string, validate=True)
    nparr     = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def normalize_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9\s]", " ", text.upper())


# ---------------------------------------------------------------------------
# Detección de rostro y embeddings
# ---------------------------------------------------------------------------
def _cv2_to_rgb(cv2_img):
    return cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)


def _haar_fallback_embedding(cv2_img):
    """Haar cascade fallback para fotos de cédula pequeñas donde MTCNN falla."""
    gray    = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    min_dim = min(gray.shape)
    if min_dim < 200:
        scale   = max(2.0, round(300 / min_dim))
        gray    = cv2.resize(gray,    None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cv2_img = cv2.resize(cv2_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    for g in (gray, cv2.equalizeHist(gray)):
        faces = face_cascade.detectMultiScale(g, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
        if len(faces) > 0:
            x, y, w, h  = max(faces, key=lambda r: r[2] * r[3])
            face_roi     = cv2_img[y:y + h, x:x + w]
            face_roi     = cv2.resize(face_roi, (160, 160), interpolation=cv2.INTER_LINEAR)
            rgb          = _cv2_to_rgb(face_roi)
            tensor       = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor       = (tensor - 0.5) / 0.25
            with torch.no_grad():
                emb = face_net(tensor.unsqueeze(0).to(device))
            return emb, True

    return None, False


def extract_face_embedding(cv2_img):
    """
    Detecta rostro y extrae embedding 512-dim.
    Retorna (embedding, face_detected:bool).
    face_detected=False indica embedding de baja confianza (imagen completa).
    """
    if cv2_img is None:
        return None, False

    rgb = _cv2_to_rgb(cv2_img)

    # Nivel 1: MTCNN (más preciso)
    face_tensor = mtcnn(rgb)
    if face_tensor is not None:
        with torch.no_grad():
            emb = face_net(face_tensor.unsqueeze(0).to(device))
        return emb, True

    # Nivel 2: Haar cascade (fotos pequeñas de cédula)
    emb, ok = _haar_fallback_embedding(cv2_img)
    if emb is not None:
        return emb, ok

    # Nivel 3: imagen completa como último recurso
    small  = cv2.resize(rgb, (160, 160))
    tensor = torch.from_numpy(small).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.25
    with torch.no_grad():
        emb = face_net(tensor.unsqueeze(0).to(device))
    return emb, False


def _cosine_sim_pct(emb_a, emb_b) -> float:
    """Similitud coseno entre dos embeddings, escalada a 0-100 y clipada a ≥0."""
    return round(max(0.0, cosine_similarity(emb_a, emb_b).item()) * 100, 2)


def calculate_similarity(img1, img2):
    """
    Calcula similitud facial (selfie vs frente documento).

    Retorna:
        similarity_pct  : float 0-100 (None si fallo total)
        face_state      : "both" | "selfie" | "id" | "none"
        is_same_person  : bool — usa threshold dinámico según face_state
        face_quality    : "high" | "degraded" | "failed"
    """
    emb1, ok1 = extract_face_embedding(img1)
    emb2, ok2 = extract_face_embedding(img2)

    if emb1 is None or emb2 is None:
        return None, "none", False, "failed"

    if ok1 and ok2:
        state   = "both"
        quality = "high"
    elif ok1 or ok2:
        state   = "selfie" if ok1 else "id"
        quality = "degraded"
    else:
        state   = "none"
        quality = "failed"

    sim_pct = _cosine_sim_pct(emb1, emb2)

    # Opción A — threshold dinámico
    if state == "none":
        # Ninguna cara detectada → rechazar independientemente del score
        is_match = False
    elif state == "both":
        is_match = sim_pct >= FACE_THRESHOLD_BOTH
    else:
        is_match = sim_pct >= FACE_THRESHOLD_ONE

    return sim_pct, state, is_match, quality


# ---------------------------------------------------------------------------
# Detección de documentos duplicados / dorso = frente
# ---------------------------------------------------------------------------
def _average_hash(img, hash_size: int = 16):
    resized = cv2.resize(img, (hash_size, hash_size))
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    bits    = (gray > gray.mean()).flatten()
    return bits


def _hamming_distance(h1, h2) -> int:
    return int(np.sum(h1 != h2))


def _orb_similarity(img1, img2) -> float:
    orb   = cv2.ORB_create(nfeatures=1000)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return 0.0

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < 0.75 * n.distance]
    total   = min(len(kp1), len(kp2))
    return (len(good) / total * 100) if total > 0 else 0.0


def _doc_similarity_score(img1, img2) -> float:
    """
    Devuelve un score 0-100 que representa qué tan similares son dos imágenes
    de documento. Combina aHash (40%) + ORB keypoints (60%).
    """
    size   = (400, 400)
    img1_r = cv2.resize(img1, size)
    img2_r = cv2.resize(img2, size)

    h1, h2   = _average_hash(img1_r), _average_hash(img2_r)
    hash_sim = max(0.0, 100.0 - (_hamming_distance(h1, h2) / len(h1) * 100))
    orb_sim  = _orb_similarity(img1_r, img2_r)

    combined = hash_sim * 0.4 + orb_sim * 0.6
    logger.info("[DOC-SIM] hash=%.1f%% orb=%.1f%% combined=%.1f%%", hash_sim, orb_sim, combined)
    return round(combined, 2)


def classify_back_document(front_img, back_img) -> tuple[float, str]:
    """
    Opción B — Clasifica el estado del dorso del documento.

    Returns:
        (similarity_score, status)
        status puede ser:
          "same_as_front" → dorso = frente (misma foto subida dos veces)
          "duplicate"     → muy similares pero no idénticos (≥70%)
          "ok"            → dorso diferente al frente — correcto
    """
    score = _doc_similarity_score(front_img, back_img)

    if score >= BACK_IDENTICAL_THRESHOLD:
        status = "same_as_front"
    elif score >= 70.0:
        status = "duplicate"
    else:
        status = "ok"

    logger.info("[BACK] score=%.1f%% → status=%s", score, status)
    return score, status


# ---------------------------------------------------------------------------
# Opción C — Detección de fraude: selfie ≈ documento
# ---------------------------------------------------------------------------
def detect_selfie_document_fraud(selfie_img, doc_img) -> tuple[float, bool]:
    """
    Compara selfie vs frente del documento a nivel de imagen (no de rostro).
    Si la similitud de imagen es muy alta → el usuario subió la misma foto
    como selfie y como documento.

    Returns:
        (image_similarity_score, is_fraud)
    """
    score    = _doc_similarity_score(selfie_img, doc_img)
    is_fraud = score >= SELFIE_DOC_FRAUD_THRESHOLD
    logger.info("[FRAUD] selfie-vs-doc score=%.1f%% → is_fraud=%s", score, is_fraud)
    return score, is_fraud


# ---------------------------------------------------------------------------
# OCR y matching de campos
# ---------------------------------------------------------------------------
def field_text_matches(expected_value: str, extracted_text: str) -> bool:
    words = normalize_text(expected_value).split()
    if not words:
        return False
    for word in words:
        if not re.search(r"\b" + re.escape(word) + r"\b", extracted_text):
            compact_expected = "".join(words)
            compact_text     = "".join(extracted_text.split())
            return compact_expected in compact_text
    return True


def _fuzzy_date_matches(expected: str, extracted_text: str) -> bool:
    m = re.search(r"(\d{1,2})\s+(\d{1,2})\s+(\d{4})", extracted_text)
    if not m:
        return False
    ed, em, ey = re.findall(r"\d+", expected)
    od, om, oy = m.groups()
    try:
        return (abs(int(ed) - int(od)) <= 2 and
                abs(int(em) - int(om)) <= 1 and
                abs(int(ey) - int(oy)) <= 1)
    except ValueError:
        return False


def field_text_matches_smart(field: str, expected_value: str, extracted_text: str) -> bool:
    if field == "tipoDoc":
        return True
    return field_text_matches(expected_value, extracted_text)


def extract_document_text(reader, img) -> str:
    results   = reader.readtext(img, detail=1)
    lines     = [normalize_text(text) for _, text, conf in results if conf >= OCR_MIN_CONFIDENCE]
    full_text = " ".join(lines)
    logger.info("[OCR] Texto extraído: %s", full_text)
    return full_text


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu": use_gpu,
        "device": str(device),
        "model": "InceptionResnetV1 (VGGFace2)",
        "thresholds": {
            "face_both":           FACE_THRESHOLD_BOTH,
            "face_one":            FACE_THRESHOLD_ONE,
            "back_identical":      BACK_IDENTICAL_THRESHOLD,
            "selfie_doc_fraud":    SELFIE_DOC_FRAUD_THRESHOLD,
        },
    }


@app.post("/verify")
def verify_identity(data: VerificationRequest):
    """
    Verifica identidad combinando:
      - Comparación facial selfie vs documento (con threshold dinámico)
      - OCR + matching de campos del formulario
      - Detección fraude: selfie = documento (Opción C)
      - Clasificación del dorso: ok / same_as_front / duplicate (Opción B)

    Campos nuevos en la respuesta:
      face_quality          : "high" | "degraded" | "failed"
      face_threshold_used   : float — threshold aplicado según calidad
      is_selfie_fraud       : bool  — selfie y doc son la misma imagen
      selfie_doc_similarity : float — score imagen selfie vs doc
      back_document_status  : "ok" | "same_as_front" | "duplicate"
      back_similarity       : float — score similitud frente vs dorso
    """
    try:
        selfie_img = b64_to_cv2(data.selfie_b64)
        id_img     = b64_to_cv2(data.id_card_b64)

        if selfie_img is None or id_img is None:
            raise HTTPException(
                status_code=422,
                detail="No se pudo decodificar una o ambas imágenes.",
            )

        # ── Similitud facial con threshold dinámico (Opción A) ──────────
        similarity_pct, face_state, is_same_person, face_quality = calculate_similarity(
            selfie_img, id_img
        )

        if face_state == "none":
            threshold_used = FACE_THRESHOLD_BOTH  # referencia, pero se rechazó por calidad
        elif face_state == "both":
            threshold_used = FACE_THRESHOLD_BOTH
        else:
            threshold_used = FACE_THRESHOLD_ONE

        # ── Fraude: selfie idéntica al documento (Opción C) ─────────────
        selfie_doc_sim, is_selfie_fraud = detect_selfie_document_fraud(selfie_img, id_img)

        # ── OCR + matching de campos ─────────────────────────────────────
        logger.info("[VERIFY] form_data: %s", data.form_data)
        extracted_text = extract_document_text(ocr_reader, id_img)

        field_matches = {}
        for field, expected_value in data.form_data.items():
            if not expected_value:
                field_matches[field] = False
                continue
            field_matches[field] = field_text_matches_smart(field, expected_value, extracted_text)

        # ── Clasificación del dorso (Opción B) ──────────────────────────
        back_similarity    = None
        back_document_status = "not_provided"   # no debería ocurrir según el contexto

        if data.id_card_back_b64.strip():
            id_back_img = b64_to_cv2(data.id_card_back_b64)
            if id_back_img is not None:
                back_similarity, back_document_status = classify_back_document(id_img, id_back_img)
            else:
                back_document_status = "decode_error"
        else:
            back_document_status = "not_provided"

        return {
            "status": "success",

            # ── Resultados faciales ──────────────────────────────────────
            "face_detected":        face_state,
            "face_quality":         face_quality,
            "face_threshold_used":  threshold_used,
            "facial_similarity":    similarity_pct,
            "is_same_person":       is_same_person,

            # ── Fraude selfie = documento (Opción C) ─────────────────────
            "is_selfie_fraud":      is_selfie_fraud,
            "selfie_doc_similarity": selfie_doc_sim,

            # ── Campos del formulario ────────────────────────────────────
            "field_matches": field_matches,

            # ── Estado del dorso (Opción B) ──────────────────────────────
            "back_document_status": back_document_status,
            "back_similarity":      back_similarity,

            # ── Compatibilidad hacia atrás ───────────────────────────────
            "document_duplicate_similarity": back_similarity,
            "is_document_duplicate":         back_document_status in ("same_as_front", "duplicate"),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ERROR en /verify:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc)) from exc
