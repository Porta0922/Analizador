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


def _haar_detect_box(cv2_img):
    """
    Intenta detectar un rostro con Haar cascade.
    Retorna (x, y, w, h) del rostro más grande encontrado, o None.
    Hace upscale si la imagen es muy pequeña.
    """
    gray    = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    min_dim = min(gray.shape)
    scale   = 1.0

    if min_dim < 200:
        scale   = max(2.0, round(300 / min_dim))
        gray    = cv2.resize(gray,    None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cv2_img = cv2.resize(cv2_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    for g in (gray, cv2.equalizeHist(gray)):
        faces = face_cascade.detectMultiScale(g, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            # Reescalar coordenadas a la imagen original si hubo upscale
            if scale != 1.0:
                x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            return x, y, w, h

    return None


def _crop_face_from_box(cv2_img, x, y, w, h, padding_ratio: float = 0.20):
    """
    Recorta la región del rostro con un padding proporcional.
    padding_ratio=0.20 agrega un 20% de margen alrededor del bounding box.
    Útil para mejorar el embedding al incluir contexto del rostro.
    """
    H, W = cv2_img.shape[:2]
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)
    return cv2_img[y1:y2, x1:x2]


def _embedding_from_crop(face_crop_cv2):
    """
    Genera embedding FaceNet desde un recorte de cara (imagen OpenCV).
    Intenta MTCNN primero sobre el recorte; si falla, normaliza directo.
    """
    rgb         = _cv2_to_rgb(face_crop_cv2)
    face_tensor = mtcnn(rgb)
    if face_tensor is not None:
        with torch.no_grad():
            return face_net(face_tensor.unsqueeze(0).to(device))

    # Fallback: redimensionar el recorte directamente a 160×160
    small  = cv2.resize(rgb, (160, 160))
    tensor = torch.from_numpy(small).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.25
    with torch.no_grad():
        return face_net(tensor.unsqueeze(0).to(device))


def extract_face_embedding(cv2_img):
    """
    Detecta rostro y extrae embedding 512-dim.

    Retorna:
        embedding     : tensor
        face_detected : bool — True si se detectó un rostro real
        box           : dict {x, y, w, h, x_norm, y_norm, w_norm, h_norm} o None
        landmarks     : lista de 5 puntos [{x,y}, ...] o None

    Orden de intentos:
      1. MTCNN detect() → bounding box + landmarks + recorte mejorado con padding
      2. Haar cascade   → bounding box + recorte mejorado
      3. Imagen completa como último recurso
    """
    if cv2_img is None:
        return None, False, None, None

    H, W  = cv2_img.shape[:2]
    rgb   = _cv2_to_rgb(cv2_img)

    # ── Nivel 1: MTCNN con detección de box y landmarks ─────────────────────
    try:
        boxes, probs, points = mtcnn.detect(rgb, landmarks=True)

        if boxes is not None and len(boxes) > 0 and probs[0] is not None:
            # Tomar la detección de mayor probabilidad
            best_idx = int(np.argmax(probs))
            box_raw  = boxes[best_idx]      # [x1, y1, x2, y2]
            prob     = float(probs[best_idx])

            if prob >= 0.90:
                x1, y1, x2, y2 = [int(v) for v in box_raw]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                box = {
                    "x": x1, "y": y1,
                    "w": x2 - x1, "h": y2 - y1,
                    # Coordenadas normalizadas 0-1 para el frontend
                    "x_norm": round(x1 / W, 4),
                    "y_norm": round(y1 / H, 4),
                    "w_norm": round((x2 - x1) / W, 4),
                    "h_norm": round((y2 - y1) / H, 4),
                    "prob":   round(prob, 3),
                    "method": "mtcnn",
                }

                # Landmarks: array (N, 5, 2) → lista de 5 puntos {x, y, x_norm, y_norm}
                lm_raw   = points[best_idx]   # shape (5, 2): ojos, nariz, comisuras boca
                lm_names = ["eye_left", "eye_right", "nose", "mouth_left", "mouth_right"]
                landmarks = [
                    {
                        "name":   lm_names[i],
                        "x":      int(lm_raw[i][0]),
                        "y":      int(lm_raw[i][1]),
                        "x_norm": round(float(lm_raw[i][0]) / W, 4),
                        "y_norm": round(float(lm_raw[i][1]) / H, 4),
                    }
                    for i in range(5)
                ]

                # Recortar con padding y generar embedding de mayor calidad
                face_crop = _crop_face_from_box(cv2_img, x1, y1, x2 - x1, y2 - y1, padding_ratio=0.15)
                emb       = _embedding_from_crop(face_crop)
                logger.info("[FACE] MTCNN box=[%d,%d,%d,%d] prob=%.2f", x1, y1, x2, y2, prob)
                return emb, True, box, landmarks

    except Exception as e:
        logger.warning("[FACE] MTCNN detect() failed: %s — falling back to Haar", str(e))

    # ── Nivel 2: Haar cascade ────────────────────────────────────────────────
    haar_result = _haar_detect_box(cv2_img)
    if haar_result is not None:
        x, y, w, h = haar_result
        box = {
            "x": x, "y": y, "w": w, "h": h,
            "x_norm": round(x / W, 4),
            "y_norm": round(y / H, 4),
            "w_norm": round(w / W, 4),
            "h_norm": round(h / H, 4),
            "prob":   0.70,
            "method": "haar",
        }
        face_crop = _crop_face_from_box(cv2_img, x, y, w, h, padding_ratio=0.15)
        emb       = _embedding_from_crop(face_crop)
        logger.info("[FACE] Haar box=[%d,%d,%d,%d]", x, y, w, h)
        return emb, True, box, None

    # ── Nivel 3: imagen completa como último recurso ─────────────────────────
    small  = cv2.resize(rgb, (160, 160))
    tensor = torch.from_numpy(small).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.25
    with torch.no_grad():
        emb = face_net(tensor.unsqueeze(0).to(device))
    logger.info("[FACE] Fallback: full-image embedding (no face detected)")
    return emb, False, None, None

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
        selfie_box      : dict con bounding box de la selfie (o None)
        selfie_landmarks: lista de 5 landmarks de la selfie (o None)
        id_box          : dict con bounding box del documento (o None)
        id_landmarks    : lista de 5 landmarks del documento (o None)
    """
    emb1, ok1, box1, lm1 = extract_face_embedding(img1)
    emb2, ok2, box2, lm2 = extract_face_embedding(img2)

    if emb1 is None or emb2 is None:
        return None, "none", False, "failed", None, None, None, None

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
        is_match = False
    elif state == "both":
        is_match = sim_pct >= FACE_THRESHOLD_BOTH
    else:
        is_match = sim_pct >= FACE_THRESHOLD_ONE

    return sim_pct, state, is_match, quality, box1, lm1, box2, lm2


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

# Niveles de fraude — usados en fraud_reason
# "identical_image"   : imagen idéntica a nivel de píxeles (>92%) Y embedding facial idéntico (>95%)
# "photo_of_screen"   : imagen muy similar (>85%) pero embedding facial NO idéntico → foto de pantalla
# "none"              : no se detectó fraude
FRAUD_FACE_IDENTICAL_THRESHOLD  = float(os.getenv("FRAUD_FACE_IDENTICAL_THRESHOLD",  "95.0"))
FRAUD_IMAGE_HIGH_THRESHOLD      = float(os.getenv("FRAUD_IMAGE_HIGH_THRESHOLD",      "85.0"))


def detect_selfie_document_fraud(
    selfie_img,
    doc_img,
    face_similarity_pct: float,
    face_state: str,
) -> tuple[float, bool, str]:
    """
    Opción C mejorada — Detecta fraude combinando dos señales independientes:

      Señal 1 — Similitud de imagen (aHash + ORB):
        Compara la selfie vs el frente del documento a nivel de píxeles/features.
        Alta similitud → las imágenes se parecen mucho visualmente.

      Señal 2 — Similitud facial (FaceNet embedding, ya calculada):
        Compara los rostros extraídos. Si es >95% → los embeddings son casi
        idénticos, lo que indica que es literalmente la misma foto de cara.

    Lógica de decisión:
      • imagen_sim >= 92% AND facial_sim >= 95%  → "identical_image" (fraude real)
        Ambas señales disparan → la selfie es la misma foto que el documento.

      • imagen_sim >= 85% AND facial_sim < 95%   → "photo_of_screen" (sospechoso)
        La imagen se parece mucho pero los embeddings difieren → posiblemente una
        foto tomada con el celular mostrando el documento en pantalla. No es fraude
        confirmado pero es sospechoso y merece revisión manual.

      • imagen_sim < 85%                         → "none" (sin fraude)
        Las imágenes son visualmente diferentes → no hay fraude de imagen.

    Args:
        selfie_img           : imagen OpenCV de la selfie
        doc_img              : imagen OpenCV del frente del documento
        face_similarity_pct  : score 0-100 de la comparación facial FaceNet (ya calculado)
        face_state           : "both"|"selfie"|"id"|"none" — calidad de detección

    Returns:
        (image_similarity_score, is_fraud, fraud_reason)
        fraud_reason: "identical_image" | "photo_of_screen" | "none"
    """
    img_score = _doc_similarity_score(selfie_img, doc_img)

    # Solo podemos afirmar "identical_image" si ambas caras fueron detectadas
    # correctamente (face_state = "both"). Si no hay detección facial fiable,
    # no elevamos a fraude confirmado aunque la imagen sea muy parecida.
    face_detected_both = face_state == "both"

    if img_score >= SELFIE_DOC_FRAUD_THRESHOLD and face_detected_both and face_similarity_pct >= FRAUD_FACE_IDENTICAL_THRESHOLD:
        # Ambas señales altas → misma foto subida dos veces
        reason   = "identical_image"
        is_fraud = True

    elif img_score >= FRAUD_IMAGE_HIGH_THRESHOLD:
        # Imagen muy parecida pero embedding facial difiere (o no hay detección) →
        # foto del documento tomada desde una pantalla de celular u otro artefacto
        reason   = "photo_of_screen"
        is_fraud = False  # sospechoso pero no bloquear automáticamente

    else:
        reason   = "none"
        is_fraud = False

    logger.info(
        "[FRAUD] img_score=%.1f%% face_sim=%.1f%% face_state=%s → reason=%s is_fraud=%s",
        img_score,
        face_similarity_pct if face_similarity_pct is not None else -1,
        face_state,
        reason,
        is_fraud,
    )
    return img_score, is_fraud, reason


# ---------------------------------------------------------------------------
# OCR y matching de campos
# ---------------------------------------------------------------------------
def field_text_matches(expected_value: str, extracted_text: str) -> bool:
    """
    Verifica que TODAS las palabras del campo esperado aparezcan en el texto OCR.
    Estrategias en orden:
      1. Token por token con word-boundary
      2. Compact match (sin espacios) — para números fragmentados por OCR
    """
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
    """
    Matching tolerante a errores OCR en fechas.
    Acepta separadores: espacios, guiones, barras, puntos.
    Tolerancia: ±2 días, ±1 mes, 0 años.
    """
    # Normalizar separadores del valor esperado a extraer d/m/y
    parts = re.findall(r"\d+", expected)
    if len(parts) < 3:
        return False
    ed, em, ey = parts[0], parts[1], parts[2]

    # Buscar patrón de fecha en el texto OCR (cualquier separador o sin separador)
    # Formatos: DD/MM/YYYY, DD-MM-YYYY, DD MM YYYY, DDMMYYYY
    patterns = [
        r"(\d{1,2})[\s/\-\.](\d{1,2})[\s/\-\.](\d{4})",   # con separador
        r"(\d{2})(\d{2})(\d{4})",                            # sin separador (DDMMYYYY)
    ]
    for pat in patterns:
        m = re.search(pat, extracted_text)
        if m:
            od, om, oy = m.group(1), m.group(2), m.group(3)
            try:
                if (abs(int(ed) - int(od)) <= 2 and
                        abs(int(em) - int(om)) <= 1 and
                        int(ey) == int(oy)):
                    return True
            except ValueError:
                continue
    return False


def _normalize_date_for_match(value: str) -> str:
    """Normaliza fechas a formato comparable: extrae solo los dígitos en orden."""
    parts = re.findall(r"\d+", value)
    return " ".join(parts) if parts else value


def field_text_matches_smart(field: str, expected_value: str, extracted_text: str) -> bool:
    """
    Matching inteligente por tipo de campo.
    - tipoDoc     → siempre True (campo no relevante para OCR)
    - fechaNacimiento / fecha* / date* → fuzzy date match con tolerancia
    - numeroDoc / numero* → compact match (ignora puntos, espacios, guiones)
    - resto       → match estándar por tokens
    """
    field_lower = field.lower()

    # Tipo de documento: no relevante para OCR
    if field_lower == "tipodoc":
        return True

    # Fechas: matching tolerante a formato y errores OCR
    if "fecha" in field_lower or "date" in field_lower or "nacimiento" in field_lower:
        return _fuzzy_date_matches(expected_value, extracted_text)

    # Número de documento: ignorar separadores (puntos, guiones, espacios)
    if "numero" in field_lower or "num" in field_lower or field_lower in ("doc", "documento"):
        # Compact match: comparar solo dígitos
        expected_digits = re.sub(r"\D", "", normalize_text(expected_value))
        ocr_digits      = re.sub(r"\D", "", extracted_text)
        if expected_digits and expected_digits in ocr_digits:
            return True
        return field_text_matches(expected_value, extracted_text)

    # Matching estándar
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
        similarity_pct, face_state, is_same_person, face_quality, \
            selfie_box, selfie_landmarks, id_box, id_landmarks = calculate_similarity(
            selfie_img, id_img
        )

        if face_state == "none":
            threshold_used = FACE_THRESHOLD_BOTH
        elif face_state == "both":
            threshold_used = FACE_THRESHOLD_BOTH
        else:
            threshold_used = FACE_THRESHOLD_ONE

        # ── Fraude: selfie idéntica al documento (Opción C mejorada) ────
        # Pasamos face_similarity_pct y face_state para la lógica combinada.
        # detect_selfie_document_fraud ahora distingue entre:
        #   "identical_image"  → misma foto subida dos veces (fraude real)
        #   "photo_of_screen"  → foto tomada de pantalla con el doc (sospechoso)
        #   "none"             → sin fraude detectado
        selfie_doc_sim, is_selfie_fraud, fraud_reason = detect_selfie_document_fraud(
            selfie_img,
            id_img,
            face_similarity_pct=similarity_pct if similarity_pct is not None else 0.0,
            face_state=face_state,
        )

        # ── OCR + matching de campos ─────────────────────────────────────
        logger.info("[VERIFY] form_data: %s", data.form_data)
        extracted_text = extract_document_text(ocr_reader, id_img)
        logger.info("[OCR] Texto completo extraído: %s", extracted_text)

        # Campos que no están en el documento de identidad — no se verifican por OCR
        # (son datos del sistema/formulario que no aparecen en el carnet)
        NON_DOCUMENT_FIELDS = {
            "rg", "estadocivil", "estado_civil", "edad", "nacionalidad",
            "fechafinalta", "fecha_fin_alta", "direccion", "telefono",
            "email", "correo", "ocupacion", "profesion",
        }

        field_matches = {}
        for field, expected_value in data.form_data.items():
            field_key = field.lower().replace(" ", "")

            # Campo vacío → no verificable
            if not expected_value or not expected_value.strip():
                field_matches[field] = None
                continue

            # Campo que no corresponde al documento → no aplicable, no marcar como error
            if field_key in NON_DOCUMENT_FIELDS:
                field_matches[field] = None
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

            # ── Bounding boxes y landmarks (para dibujar en la extensión) ─
            "selfie_face_box":       selfie_box,
            "selfie_face_landmarks": selfie_landmarks,
            "id_face_box":           id_box,
            "id_face_landmarks":     id_landmarks,

            # ── Fraude selfie = documento (Opción C) ─────────────────────
            "is_selfie_fraud":       is_selfie_fraud,
            "selfie_doc_similarity": selfie_doc_sim,
            "fraud_reason":          fraud_reason,

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
