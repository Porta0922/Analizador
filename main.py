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
# Configuración por variables de entorno (con defaults seguros para desarrollo)
# ---------------------------------------------------------------------------
MAX_IMAGE_B64_LENGTH = int(os.getenv("MAX_IMAGE_B64_LENGTH", "15_000_000"))
MAX_FORM_FIELDS = int(os.getenv("MAX_FORM_FIELDS", "20"))
MAX_FIELD_VALUE_LENGTH = int(os.getenv("MAX_FIELD_VALUE_LENGTH", "200"))
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.4"))
FACE_SIMILARITY_THRESHOLD = float(os.getenv("FACE_SIMILARITY_THRESHOLD", "60.0"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app = FastAPI(title="Verificación de Identidad Local GPU", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Inicialización de modelos (una sola vez)
# ---------------------------------------------------------------------------
use_gpu = os.getenv("USE_GPU", "").lower() not in ("0", "false", "no") and torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
logger.info("--> Ejecutando backend en: %s", device)
if use_gpu:
    logger.info("--> GPU detectada: %s", torch.cuda.get_device_name(0))

ocr_reader = easyocr.Reader(["es"], gpu=use_gpu)

# MTCNN: detección + recorte + alineación de rostros
mtcnn = MTCNN(
    keep_all=False,
    device=device,
    min_face_size=30,
    thresholds=[0.6, 0.7, 0.7],
    post_process=True,
)

# InceptionResnetV1: embeddings faciales pre-entrenados en VGGFace2
face_net = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Haar cascade como fallback para fotos de cédula muy pequeñas
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------------------------------------------------------------
# Validación de entrada
# ---------------------------------------------------------------------------
class VerificationRequest(BaseModel):
    selfie_b64: str = Field(min_length=1, max_length=MAX_IMAGE_B64_LENGTH)
    id_card_b64: str = Field(min_length=1, max_length=MAX_IMAGE_B64_LENGTH)
    id_card_back_b64: str = Field(default="", max_length=MAX_IMAGE_B64_LENGTH)
    form_data: dict = Field(default_factory=dict, max_length=MAX_FORM_FIELDS)

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
# Utilidades
# ---------------------------------------------------------------------------
def b64_to_cv2(b64_string: str):
    """Decodifica un base64 (con o sin prefijo data URI) a imagen OpenCV."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string, validate=True)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def normalize_text(text: str) -> str:
    """Normaliza a mayúsculas conservando espacios entre palabras."""
    return re.sub(r"[^A-Z0-9\s]", " ", text.upper())


# ---------------------------------------------------------------------------
# Detección de rostro + embeddings faciales
# ---------------------------------------------------------------------------
def _cv2_to_rgb(cv2_img):
    return cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)


def _haar_fallback_embedding(cv2_img):
    """Fallback: usa Haar cascade para detectar rostro en fotos de cédula
    muy pequeñas donde MTCNN no funciona, y genera un embedding aproximado."""
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    min_dim = min(gray.shape)
    if min_dim < 200:
        scale = max(2.0, round(300 / min_dim))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cv2_img = cv2.resize(cv2_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    for g in (gray, cv2.equalizeHist(gray)):
        faces = face_cascade.detectMultiScale(g, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            face_roi = cv2_img[y:y + h, x:x + w]
            # Resize to minimum 160x160 to avoid conv errors
            face_roi = cv2.resize(face_roi, (160, 160), interpolation=cv2.INTER_LINEAR)
            rgb = _cv2_to_rgb(face_roi)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor = (tensor - 0.5) / 0.25  # normalizar como VGGFace2
            with torch.no_grad():
                emb = face_net(tensor.unsqueeze(0).to(device))
            return emb, True

    return None, False


def extract_face_embedding(cv2_img):
    """Detecta rostro y extrae embedding facial.

    Retorna (embedding_tensor, face_detected).
    - MTCNN es preciso para selfies y fotos frontales.
    - Haar cascade es fallback para fotos de cédula muy pequeñas.
    - Si nada funciona, usa la imagen completa (menos confiable).
    """
    if cv2_img is None:
        return None, False

    rgb = _cv2_to_rgb(cv2_img)

    # 1) Intentar con MTCNN (el más preciso)
    face_tensor = mtcnn(rgb)
    if face_tensor is not None:
        with torch.no_grad():
            emb = face_net(face_tensor.unsqueeze(0).to(device))
        return emb, True

    # 2) Fallback: Haar cascade (fotos de cédula pequeñas)
    emb, ok = _haar_fallback_embedding(cv2_img)
    if emb is not None:
        return emb, ok

    # 3) Último recurso: embedding de la imagen completa (baja calidad)
    small = cv2.resize(rgb, (160, 160))
    tensor = torch.from_numpy(small).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.25
    with torch.no_grad():
        emb = face_net(tensor.unsqueeze(0).to(device))
    return emb, False


def calculate_similarity(img1, img2):
    """Calcula similitud facial usando embeddings de InceptionResnetV1.

    Retorna (similitud_pct, face_state).
    face_state: "both" | "selfie" | "id" | "none".
    La similitud es la distancia coseno (-1 a 1) escalada a 0-100%.
    Misma persona típicamente > 70%, diferente < 40%.
    """
    emb1, ok1 = extract_face_embedding(img1)
    emb2, ok2 = extract_face_embedding(img2)

    if emb1 is None or emb2 is None:
        return None, "none"

    state = "both" if (ok1 and ok2) else ("selfie" if ok1 else ("id" if ok2 else "none"))

    sim = cosine_similarity(emb1, emb2).item()
    similarity = max(0.0, sim) * 100
    return round(similarity, 2), state


def _average_hash(img, hash_size=16):
    """Calcula el hash promedio (aHash) de una imagen."""
    resized = cv2.resize(img, (hash_size, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    mean = gray.mean()
    bits = (gray > mean).flatten()
    return bits


def _hamming_distance(hash1, hash2):
    """Distancia Hamming entre dos hashes."""
    return int(np.sum(hash1 != hash2))


def _orb_similarity(img1, img2):
    """Similitud por features ORB (keypoints matching)."""
    orb = cv2.ORB_create(nfeatures=1000)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    total = min(len(kp1), len(kp2))
    return (len(good) / total * 100) if total > 0 else 0.0


def detect_duplicate_documents(img1, img2):
    """Detecta si dos imágenes del documento son iguales o muy similares.
    Combina aHash + ORB para ser robusto a iluminación/ángulo."""
    if img1 is None or img2 is None:
        return None, False

    size = (400, 400)
    img1_r = cv2.resize(img1, size)
    img2_r = cv2.resize(img2, size)

    # 1) Perceptual hash (robusto a iluminación)
    h1 = _average_hash(img1_r)
    h2 = _average_hash(img2_r)
    dist = _hamming_distance(h1, h2)
    hash_bits = len(h1)
    hash_sim = max(0, 100 - (dist / hash_bits * 100))

    # 2) ORB feature matching (robusto a ángulo/perspectiva)
    orb_sim = _orb_similarity(img1_r, img2_r)

    # Combinar: peso 40% hash + 60% features
    combined = hash_sim * 0.4 + orb_sim * 0.6

    logger.info(
        "[DUPLICATE] hash_sim=%.1f%% orb_sim=%.1f%% combined=%.1f%%",
        hash_sim, orb_sim, combined,
    )
    return round(combined, 2), combined >= 70.0


def field_text_matches(expected_value: str, extracted_text: str) -> bool:
    """Verifica que TODAS las palabras del campo esperado aparezcan como
    tokens independientes en el texto OCR. Si falla, intenta matching
    compacto (sin espacios) para números con formato variable."""
    words = normalize_text(expected_value).split()
    if not words:
        return False

    # 1) Check estándar: cada palabra como token independiente
    for word in words:
        if not re.search(r"\b" + re.escape(word) + r"\b", extracted_text):
            # 2) Fallback compacto: unir todo sin espacios (para "6302723" vs "630 2723")
            compact_expected = "".join(words)
            compact_text = "".join(extracted_text.split())
            return compact_expected in compact_text
    return True


def _fuzzy_date_matches(expected: str, extracted_text: str) -> bool:
    """Matching tolerante a errores OCR en fechas.
    Busca patrón DD MM YYYY en el texto con tolerancia ±2 en día, ±1 en mes."""
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
    """Matching inteligente. tipoDoc se ignora porque no es relevante."""
    if field == "tipoDoc":
        return True  # Siempre matchea - no es relevante
    return field_text_matches(expected_value, extracted_text)


def extract_document_text(ocr_reader, img) -> str:
    """Ejecuta OCR filtrando resultados de baja confianza."""
    results = ocr_reader.readtext(img, detail=1)
    lines = [normalize_text(text) for _, text, conf in results if conf >= OCR_MIN_CONFIDENCE]
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
        "threshold": FACE_SIMILARITY_THRESHOLD,
    }


@app.post("/verify")
def verify_identity(data: VerificationRequest):
    # Endpoint síncrono: FastAPI lo ejecuta en un threadpool, sin bloquear
    # el event loop durante OCR/visión.
    try:
        selfie_img = b64_to_cv2(data.selfie_b64)
        id_img = b64_to_cv2(data.id_card_b64)

        if selfie_img is None or id_img is None:
            raise HTTPException(
                status_code=422,
                detail="No se pudo decodificar una o ambas imágenes a formato OpenCV.",
            )

        similarity_pct, face_state = calculate_similarity(selfie_img, id_img)

        logger.info("[VERIFY] form_data recibido: %s", data.form_data)
        extracted_text = extract_document_text(ocr_reader, id_img)

        field_matches = {}
        for field, expected_value in data.form_data.items():
            if not expected_value:
                field_matches[field] = False
                continue
            match = field_text_matches_smart(field, expected_value, extracted_text)
            logger.info("[MATCH] '%s' -> '%s': %s | regex: %s", field, expected_value, match, r"\b" + re.escape(normalize_text(expected_value)) + r"\b")
            field_matches[field] = match

        # Detección de documentos duplicados (frente vs dorso)
        doc_duplicate_sim = None
        is_doc_duplicate = False
        if data.id_card_back_b64:
            id_back_img = b64_to_cv2(data.id_card_back_b64)
            if id_back_img is not None:
                doc_duplicate_sim, is_doc_duplicate = detect_duplicate_documents(id_img, id_back_img)
                logger.info("[DUPLICATE] Similitud documentos: %s%%, duplicado: %s", doc_duplicate_sim, is_doc_duplicate)

        return {
            "status": "success",
            "face_detected": face_state,
            "facial_similarity": similarity_pct,
            "is_same_person": similarity_pct >= FACE_SIMILARITY_THRESHOLD,
            "field_matches": field_matches,
            "document_duplicate_similarity": doc_duplicate_sim,
            "is_document_duplicate": is_doc_duplicate,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ERROR en /verify:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc)) from exc
