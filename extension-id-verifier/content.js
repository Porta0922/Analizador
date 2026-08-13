// Configuración: ajustar URL del backend según despliegue
const CONFIG = {
  BACKEND_URL: "http://localhost:8000",
  AI_BACKEND_URL: "http://localhost:8001",
  // Palabras que ayudan a identificar los inputs de archivo (selfie vs documento)
  SELFIE_HINTS: ["selfie", "rostro", "retrato"],
  ID_HINTS: ["id", "documento", "dni", "carnet", "cedula", "identificacion", "frente", "cip"],
  BACK_HINTS: ["dorso", "reverso", "back", "trasera"],
  // Mapeo de campos del backend a labels del formulario
  // Campos marcados como [no_doc] no se verifican por OCR (no están en el carnet)
  FIELD_MAP: {
    primerNombre:    "Primer Nombre",
    segundoNombre:   "Segundo Nombre",
    primerApellido:  "Primer Apellido",
    segundoApellido: "Segundo Apellido",
    numeroDoc:       "Número Documento",
    tipoDoc:         "Tipo Documento",
    sexo:            "Sexo",
    fechaNacimiento: "Fecha Nacimiento",
  },
  // Campos del formulario que NO se verifican con OCR del documento
  // (son datos del sistema, no del carnet) — se envían igual para contexto IA
  NON_DOC_FIELDS: new Set([
    "rg", "estadocivil", "estado_civil", "edad", "nacionalidad",
    "fechafinalta", "fecha_fin_alta", "direccion", "telefono",
    "email", "correo", "ocupacion", "profesion",
  ]),
};

const normalize = (s) =>
  (s || "").toUpperCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/\s+/g, " ").trim();

function matchesHints(text, hints) {
  const norm = normalize(text);
  return hints.some((h) => norm.includes(h.toUpperCase()));
}

// ---------------------------------------------------------------------------
// Recolección de datos del formulario
// ---------------------------------------------------------------------------

// Lee un archivo como data URL (base64) sin depender de CORS/taint de <img>.
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("No se pudo leer el archivo."));
    reader.readAsDataURL(file);
  });
}

async function readInputFile(input) {
  const file = input.files && input.files[0];
  if (!file) return null;
  return fileToBase64(file);
}

function selectFileInput(predicate) {
  const inputs = Array.from(document.querySelectorAll("input[type=file]"));
  return inputs.find((input) => {
    const haystack = [input.name, input.id, input.accept, input.getAttribute("data-field") || ""].join(" ");
    return predicate(haystack);
  });
}

async function collectImages() {
  const images = Array.from(document.querySelectorAll("img")).filter((i) => i.src);
  console.info("[id-verifier] Imágenes en el DOM:", images.map((i) => i.src).join(", "));

  const missing = {
    selfie: { b64: null, img: null },
    id:     { b64: null, img: null },
    idBack: { b64: null, img: null },
  };
  const used = new Set();

  // ── Estrategia 0: imágenes etiquetadas por texto visible adyacente ─────────
  // Busca encabezados/labels con texto "Selfie", "Documento frente", "Documento dorso"
  // y luego localiza el <img> más cercano dentro del mismo contenedor.
  const LABEL_MAP = [
    { slot: "selfie", hints: ["selfie", "foto personal", "rostro"] },
    { slot: "id",     hints: ["documento frente", "cip frente", "frente", "documento"] },
    { slot: "idBack", hints: ["documento dorso", "cip dorso", "dorso", "reverso"] },
  ];

  function findImgNearLabel(hintWords) {
    const allText = Array.from(document.querySelectorAll(
      "h1,h2,h3,h4,h5,h6,p,span,label,div,td,th,b,strong"
    ));
    for (const el of allText) {
      const norm = normalize(el.textContent);
      if (hintWords.some((h) => norm === normalize(h) || norm.includes(normalize(h)))) {
        // Buscar <img> dentro del mismo padre o en el siguiente contenedor hermano
        const parent = el.closest("div, td, section, article, figure") || el.parentElement;
        if (!parent) continue;
        // Buscar en el padre y en el padre del padre
        for (const container of [parent, parent.parentElement].filter(Boolean)) {
          const img = container.querySelector("img");
          if (img && img.src && !used.has(img)) return img;
        }
      }
    }
    return null;
  }

  for (const { slot, hints } of LABEL_MAP) {
    if (missing[slot].b64) continue;
    const img = findImgNearLabel(hints);
    if (img) {
      const b64 = await imgToBase64(img);
      if (b64) {
        missing[slot].b64 = b64;
        missing[slot].img = img;
        used.add(img);
        console.info(`[id-verifier] slot '${slot}' desde label: ${img.src.slice(0, 80)}`);
      }
    }
  }

  // ── Estrategia 1: inputs de archivo ───────────────────────────────────────
  const selfieInput = selectFileInput((s) => matchesHints(s, CONFIG.SELFIE_HINTS));
  const idInput     = selectFileInput((s) => matchesHints(s, CONFIG.ID_HINTS));
  const backInput   = selectFileInput((s) => matchesHints(s, CONFIG.BACK_HINTS));

  const setFromInput = async (slot, input) => {
    if (!input || missing[slot].b64) return;
    const b64 = await readInputFile(input);
    if (b64) {
      missing[slot].b64 = b64;
      missing[slot].img = null;
      console.info("[id-verifier] desde input:", input.name || input.id);
    }
  };
  await setFromInput("selfie", selfieInput);
  await setFromInput("id",     idInput);
  await setFromInput("idBack", backInput);

  // Positional fallback: inputs[0]=selfie, [1]=frente, [2]=dorso
  const inputs = Array.from(document.querySelectorAll("input[type=file]"))
    .filter((i) => i.files && i.files[0]);
  await setFromInput("selfie", inputs[0]);
  await setFromInput("id",     inputs[1]);
  await setFromInput("idBack", inputs[2]);

  // ── Estrategia 2: imágenes blob:/data: (previews subidas) ────────────────
  const previews = images.filter(
    (i) => i.src.startsWith("blob:") || i.src.startsWith("data:")
  );
  for (const slot of ["selfie", "id", "idBack"]) {
    if (missing[slot].b64) continue;
    for (const img of previews) {
      if (used.has(img)) continue;
      used.add(img);
      const b64 = await imgToBase64(img);
      if (b64) {
        missing[slot].b64 = b64;
        missing[slot].img = img;
        console.info("[id-verifier] desde preview blob:", img.src.slice(0, 80));
        break;
      }
    }
  }

  // ── Estrategia 3: imágenes del servidor por ruta (EClub pattern) ──────────
  // Detecta imágenes cuyas rutas contienen keywords conocidos del formulario
  const SERVER_PATTERNS = [
    { slot: "selfie", patterns: [/selfie/i, /rostro/i, /SELFIE/] },
    { slot: "id",     patterns: [/CIP_FRENTE/i, /frente/i, /front/i, /documento/i] },
    { slot: "idBack", patterns: [/CIP_DORSO/i, /dorso/i, /back/i, /reverso/i] },
  ];
  for (const { slot, patterns } of SERVER_PATTERNS) {
    if (missing[slot].b64) continue;
    for (const img of images) {
      if (used.has(img)) continue;
      if (patterns.some((p) => p.test(img.src))) {
        const b64 = await imgToBase64(img);
        if (b64) {
          missing[slot].b64 = b64;
          missing[slot].img = img;
          used.add(img);
          console.info(`[id-verifier] slot '${slot}' por ruta de servidor: ${img.src.slice(0, 80)}`);
          break;
        }
      }
    }
  }

  // ── Estrategia 4: cualquier <img> legible — fallback general ──────────────
  for (const slot of ["selfie", "id", "idBack"]) {
    if (missing[slot].b64) continue;
    for (const img of images) {
      if (used.has(img)) continue;
      used.add(img);
      const b64 = await imgToBase64(img);
      if (b64) {
        missing[slot].b64 = b64;
        missing[slot].img = img;
        console.info(`[id-verifier] slot '${slot}' desde fallback general: ${img.src.slice(0, 80)}`);
        break;
      }
    }
  }

  return {
    selfie:    missing.selfie.b64,
    id:        missing.id.b64,
    idBack:    missing.idBack.b64,
    selfieImg: missing.selfie.img,
    idImg:     missing.id.img,
    idBackImg: missing.idBack.img,
  };
}

async function imgToBase64(img) {
  // 1) fetch directo (funciona con blob: y data:)
  try {
    const resp = await fetch(img.src);
    if (resp.ok) {
      const blob = await resp.blob();
      return await fileToBase64(blob);
    }
    console.warn("[id-verifier] fetch -> HTTP", resp.status);
  } catch (err) {
    console.warn("[id-verifier] fetch falló para:", img.src, err.message);
  }

  // 2) fallback con <canvas> (válido para blob:/mismo origen; falla si CORS)
  try {
    const width = img.naturalWidth || 640;
    const height = img.naturalHeight || 480;
    const scale = Math.min(1, 1200 / Math.max(width, height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  } catch (err) {
    console.warn("[id-verifier] canvas falló para:", img.src, err.message);
    return null;
  }
}

function collectFormData() {
  const formData = {};
  for (const [key, label] of Object.entries(CONFIG.FIELD_MAP)) {
    const input = findInputByFieldKey(key);
    const value = input && (input.value || "").trim() ? input.value.trim() : "";
    formData[key] = value;
    console.info("[id-verifier] campo '" + key + "' -> input:", input ? input.tagName + "[name=" + input.name + ",id=" + input.id + "]" : "NO ENCONTRADO", "valor:", value);
  }
  return formData;
}

// Asocia un campo del backend con su input buscando por name, luego por label.
// Usa posición visual (getBoundingClientRect) para encontrar el input correcto
// en formularios tipo grilla donde label e input están en columnas separadas.
function findInputByFieldKey(key) {
  const candidates = Array.from(document.querySelectorAll("input, select, textarea"));

  // 1) Por atributo name/id (más confiable)
  const byName = candidates.find((el) => normalize(el.name) === normalize(key) || normalize(el.id) === normalize(key));
  if (byName) return byName;

  // 2) Por texto de <label>
  const labelText = normalize(CONFIG.FIELD_MAP[key]);
  if (!labelText) return null;

  const labels = Array.from(document.querySelectorAll("label"));
  const target = labels.find((el) => normalize(el.textContent) === labelText);
  if (!target) return null;

  // 3) Si el label tiene "for", usarlo
  if (target.htmlFor) {
    const input = document.getElementById(target.htmlFor);
    if (input) return input;
  }

  // 4) Por posición visual: buscar el input que está a la misma altura
  const targetRect = target.getBoundingClientRect();
  const visibleInputs = candidates.filter((el) => el.offsetParent !== null);

  let bestInput = null;
  let bestDist = Infinity;

  for (const input of visibleInputs) {
    const inputRect = input.getBoundingClientRect();
    // Misma fila (vertical ±40px)
    const yDist = Math.abs(targetRect.top - inputRect.top);
    if (yDist < 40 && inputRect.left > targetRect.left - 20) {
      const xDist = Math.abs(inputRect.left - targetRect.left);
      if (xDist < bestDist) {
        bestDist = xDist;
        bestInput = input;
      }
    }
  }

  return bestInput;
}

// ---------------------------------------------------------------------------
// Llamada al backend y renderizado
// ---------------------------------------------------------------------------

async function verify() {
  const { selfie, id, idBack, selfieImg, idImg, idBackImg } = await collectImages();
  debugSelection = { selfieImg, idImg, idBackImg };
  highlightSelection(selfieImg, idImg, idBackImg);

  if (!selfie || !id) {
    const missing = [];
    if (!selfie) missing.push("selfie");
    if (!id) missing.push("documento");
    showError(
      `No se pudieron leer las imágenes (falta: ${missing.join(" y ")}). ` +
        "Abre la consola (F12) para ver el detalle; si vienen de otro dominio, " +
        "la extensión no puede leerlas y necesitará los inputs de archivo.",
    );
    return;
  }

  // Evitar reenvíos redundantes con las mismas imágenes
  const signature = `${selfie}|${id}|${idBack || ""}`;
  if (lastSignature === signature) return;
  lastSignature = signature;

  const formData = collectFormData();
  showLoading();

  try {
    const resp = await fetch(`${CONFIG.BACKEND_URL}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        selfie_b64: selfie,
        id_card_b64: id,
        id_card_back_b64: idBack || "",
        form_data: formData,
      }),
    });

    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `Error del servidor (${resp.status})`);
    renderResults(data);

    // Call AI backend for additional analysis (once only)
    if (!document.getElementById("ai-analysis-section")) {
      analyzeWithAI(selfie, id, idBack, formData).then((aiResult) => {
        if (aiResult && aiResult.result && !document.getElementById("ai-analysis-section")) {
          renderAIResults(aiResult);
        }
      }).catch(() => {});
    }

  } catch (err) {
    showError(`Verificación falló: ${err.message}`);
  }
}

let lastSignature = null;
let debugSelection = { selfieImg: null, idImg: null, idBackImg: null };

// ---------------------------------------------------------------------------
// Depuración visual: recuadros sobre las imágenes elegidas
// ---------------------------------------------------------------------------

function highlightSelection(selfieImg, idImg, idBackImg) {
  clearHighlight();
  if (selfieImg) drawDebugBox(selfieImg, "selfie-debug-box", "#2196f3", "SELFIE");
  if (idImg) drawDebugBox(idImg, "id-debug-box", "#ff9800", "DOCUMENTO");
  if (idBackImg) drawDebugBox(idBackImg, "idback-debug-box", "#9c27b0", "DORSO");
}

function drawDebugBox(img, boxClass, color, label) {
  const parent = img.offsetParent || img.parentElement;
  if (!parent) return;
  parent.style.position = "relative";

  const box = document.createElement("div");
  box.className = boxClass;
  box.style.cssText =
    `position:absolute;top:0;left:0;right:0;bottom:0;` +
    `border:4px solid ${color};box-sizing:border-box;pointer-events:none;z-index:99999;` +
    `box-shadow:0 0 0 1px rgba(0,0,0,0.4) inset;`;
  const tag = document.createElement("div");
  tag.textContent = label;
  tag.style.cssText =
    `position:absolute;top:0;left:0;background:${color};color:#fff;` +
    `font:bold 11px/18px monospace;padding:0 6px;`;
  box.appendChild(tag);
  parent.appendChild(box);
}

function clearHighlight() {
  document.querySelectorAll(".selfie-debug-box, .id-debug-box, .idback-debug-box").forEach((el) => el.remove());
}

// ---------------------------------------------------------------------------
// Dibujo de bounding box + landmarks de rostro sobre la imagen real del DOM
// (Opción de zoom facial — los datos vienen del backend en la respuesta)
// ---------------------------------------------------------------------------

/**
 * Dibuja un recuadro y landmarks sobre una <img> del DOM usando las
 * coordenadas normalizadas (0-1) devueltas por el backend.
 *
 * @param {HTMLImageElement} imgEl  - el elemento <img> en la página
 * @param {Object}           box    - {x_norm, y_norm, w_norm, h_norm, method}
 * @param {Array}            landmarks - [{name, x_norm, y_norm}, ...]
 * @param {string}           color  - color del recuadro (CSS)
 * @param {string}           boxId  - id único para poder limpiar después
 */
function drawFaceDetection(imgEl, box, landmarks, color, boxId) {
  if (!imgEl || !box) return;

  const parent = imgEl.offsetParent || imgEl.parentElement;
  if (!parent) return;
  parent.style.position = "relative";

  // Limpiar overlay anterior con el mismo id
  document.querySelectorAll(`[data-face-overlay="${boxId}"]`).forEach((e) => e.remove());

  // Dimensiones renderizadas del <img> en el DOM
  const rect    = imgEl.getBoundingClientRect();
  const pRect   = parent.getBoundingClientRect();
  const imgLeft = rect.left - pRect.left;
  const imgTop  = rect.top  - pRect.top;
  const imgW    = rect.width;
  const imgH    = rect.height;

  // Convertir coordenadas normalizadas a píxeles del DOM
  const px = (norm) => norm * imgW;
  const py = (norm) => norm * imgH;

  const container = document.createElement("div");
  container.setAttribute("data-face-overlay", boxId);
  container.style.cssText =
    `position:absolute;` +
    `left:${imgLeft}px;top:${imgTop}px;` +
    `width:${imgW}px;height:${imgH}px;` +
    `pointer-events:none;z-index:99999;`;

  // ── Bounding box ────────────────────────────────────────────────────────
  const boxEl = document.createElement("div");
  const bx = px(box.x_norm), by = py(box.y_norm);
  const bw = px(box.w_norm), bh = py(box.h_norm);

  boxEl.style.cssText =
    `position:absolute;` +
    `left:${bx}px;top:${by}px;width:${bw}px;height:${bh}px;` +
    `border:2px solid ${color};box-sizing:border-box;` +
    `box-shadow:0 0 0 1px rgba(0,0,0,0.5);`;

  // Label con método de detección y probabilidad
  const methodLabel = document.createElement("div");
  const methodText  = box.method === "mtcnn"
    ? `MTCNN ${box.prob != null ? (box.prob * 100).toFixed(0) + "%" : ""}`
    : `HAAR ${box.prob != null ? (box.prob * 100).toFixed(0) + "%" : ""}`;
  methodLabel.style.cssText =
    `position:absolute;top:-17px;left:0;` +
    `background:${color};color:#000;font:bold 9px/16px monospace;` +
    `padding:0 4px;white-space:nowrap;`;
  methodLabel.textContent = methodText;
  boxEl.appendChild(methodLabel);
  container.appendChild(boxEl);

  // ── Landmarks ─────────────────────────────────────────────────────────
  if (landmarks && landmarks.length > 0) {
    landmarks.forEach((lm) => {
      const dot = document.createElement("div");
      const lx  = px(lm.x_norm);
      const ly  = py(lm.y_norm);
      dot.title = lm.name;
      dot.style.cssText =
        `position:absolute;` +
        `left:${lx - 3}px;top:${ly - 3}px;` +
        `width:6px;height:6px;border-radius:50%;` +
        `background:${color};box-shadow:0 0 0 1px rgba(0,0,0,0.6);`;
      container.appendChild(dot);
    });
  }

  parent.appendChild(container);
}

function clearFaceOverlays() {
  document.querySelectorAll("[data-face-overlay]").forEach((e) => e.remove());
}

// ---------------------------------------------------------------------------
// Helpers para construir alertas inline dentro del badge
// ---------------------------------------------------------------------------

function _makeAlert(text, color, icon) {
  const el = document.createElement("div");
  el.style.cssText =
    `display:flex;align-items:flex-start;gap:5px;` +
    `background:rgba(${color},0.15);border-left:3px solid rgb(${color});` +
    `padding:4px 6px;border-radius:0 4px 4px 0;margin-bottom:4px;font-size:10px;color:#fff;`;
  el.innerHTML = `<span style="flex-shrink:0">${icon}</span><span>${text}</span>`;
  return el;
}

function renderResults(res) {
  clearFeedback();

  // ── Determinar color general del header ───────────────────────────────────
  // La verificación es negativa si: personas distintas, fraude detectado,
  // ningún rostro detectado, o dorso = frente.
  const hasCriticalAlert =
    !res.is_same_person ||
    res.is_selfie_fraud ||
    res.face_detected === "none" ||
    res.back_document_status === "same_as_front";

  const badge = document.createElement("div");
  badge.id = "selfie-similarity-badge";
  badge.style.cssText =
    "position:fixed;bottom:12px;left:50%;transform:translateX(-50%);z-index:999999;" +
    "background:rgba(10,10,15,0.96);color:#fff;padding:12px 16px;border-radius:8px;" +
    "font:13px/20px monospace;min-width:340px;max-width:420px;" +
    "box-shadow:0 4px 16px rgba(0,0,0,0.5);";

  // ── Header: similitud + estado general ───────────────────────────────────
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;gap:8px;margin-bottom:8px;";

  const dotColor = hasCriticalAlert ? "#f44336" : (res.is_same_person ? "#4caf50" : "#ff9800");
  const statusDot = document.createElement("span");
  statusDot.style.cssText = `width:10px;height:10px;border-radius:50%;background:${dotColor};flex-shrink:0;`;

  const headerText = document.createElement("span");
  headerText.style.cssText = "font-weight:bold;font-size:14px;";
  const simLabel = res.facial_similarity != null ? `${res.facial_similarity}%` : "N/A";
  headerText.textContent = `Similitud: ${simLabel}`;
  if (res.face_threshold_used != null) {
    const threshSpan = document.createElement("span");
    threshSpan.style.cssText = "font-size:10px;color:#888;margin-left:6px;font-weight:normal;";
    threshSpan.textContent = `(umbral: ${res.face_threshold_used}%)`;
    headerText.appendChild(threshSpan);
  }

  header.appendChild(statusDot);
  header.appendChild(headerText);
  badge.appendChild(header);

  // ── Alertas críticas (Opciones A, B, C) ──────────────────────────────────

  // [Opción C] Fraude: selfie = documento — mensaje según causa
  if (res.is_selfie_fraud && res.fraud_reason === "identical_image") {
    badge.appendChild(_makeAlert(
      `FRAUDE DETECTADO: la selfie y el documento frontal son la misma imagen ` +
      `(similitud imagen: ${res.selfie_doc_similarity}%, similitud facial: ${res.facial_similarity}%). ` +
      "Requiere verificación manual inmediata.",
      "220,53,69", "🚨"
    ));
  } else if (res.fraud_reason === "photo_of_screen") {
    // Sospechoso pero no bloqueante: imagen muy parecida pero embeddings distintos
    badge.appendChild(_makeAlert(
      `Posible foto del documento desde pantalla de celular ` +
      `(similitud imagen: ${res.selfie_doc_similarity}%). ` +
      "La selfie debería ser una foto de la persona, no del carnet. Verificar manualmente.",
      "255,152,0", "�"
    ));
  }

  // [Opción A] Ningún rostro detectado → resultado no confiable
  if (res.face_detected === "none") {
    badge.appendChild(_makeAlert(
      "No se detectó ningún rostro en las imágenes. El resultado de similitud facial NO es confiable. " +
      "Verificar que la selfie y la foto del documento sean correctas.",
      "220,53,69", "👤"
    ));
  } else if (res.face_detected === "selfie") {
    // Solo detectó en selfie → embedding del doc es de baja calidad
    badge.appendChild(_makeAlert(
      `Rostro detectado solo en la selfie. La foto del documento no tiene un rostro claro. ` +
      `Resultado: ${res.facial_similarity}% (umbral reducido: ${res.face_threshold_used}%).`,
      "255,152,0", "⚠️"
    ));
  } else if (res.face_detected === "id") {
    badge.appendChild(_makeAlert(
      `Rostro detectado solo en el documento. La selfie no tiene un rostro claro. ` +
      `Resultado: ${res.facial_similarity}% (umbral reducido: ${res.face_threshold_used}%).`,
      "255,152,0", "⚠️"
    ));
  }

  // [Opción B] Estado del dorso
  if (res.back_document_status === "same_as_front") {
    badge.appendChild(_makeAlert(
      `El dorso del documento es idéntico al frente (similitud ${res.back_similarity}%). ` +
      "Se subió la misma imagen dos veces. Solicitar el dorso correcto.",
      "220,53,69", "📄"
    ));
  } else if (res.back_document_status === "duplicate") {
    badge.appendChild(_makeAlert(
      `Frente y dorso del documento son muy similares (${res.back_similarity}%). ` +
      "Posible error al subir las imágenes.",
      "255,152,0", "📄"
    ));
  } else if (res.back_document_status === "decode_error") {
    badge.appendChild(_makeAlert(
      "No se pudo procesar la imagen del dorso del documento.",
      "255,152,0", "⚠️"
    ));
  }

  // [Opción A] Calidad de detección cuando ambas están presentes — solo info
  if (res.face_quality === "high" && res.is_same_person) {
    // Todo ok — no mostrar nada extra
  }

  // ── Resultado facial ──────────────────────────────────────────────────────
  const faceResult = document.createElement("div");
  faceResult.style.cssText =
    `font-size:11px;padding:4px 0;` +
    `color:${res.is_same_person ? "#4caf50" : "#f44336"};`;
  faceResult.textContent = res.is_same_person
    ? "✓ Coincidencia facial confirmada"
    : "✗ Coincidencia facial negativa";
  badge.appendChild(faceResult);

  // ── Campos del formulario ─────────────────────────────────────────────────
  const fieldSummary = document.createElement("div");
  fieldSummary.style.cssText =
    "font-size:11px;color:#aaa;border-top:1px solid rgba(255,255,255,0.15);" +
    "padding-top:6px;margin-top:4px;";
  const allMatches   = Object.entries(res.field_matches || {});
  const verifiable   = allMatches.filter(([, v]) => v !== null);
  const matchCount   = verifiable.filter(([, v]) => v === true).length;
  const naCount      = allMatches.length - verifiable.length;
  let summaryText    = `Campos: ${matchCount}/${verifiable.length} coinciden en documento`;
  if (naCount > 0) summaryText += ` (${naCount} no verificables por OCR)`;
  fieldSummary.textContent = summaryText;
  badge.appendChild(fieldSummary);

  document.body.appendChild(badge);

  // ── Resaltar inputs del formulario ────────────────────────────────────────
  // null  = campo no verificable por OCR → gris (sin borde)
  // true  = match confirmado             → verde
  // false = no coincide en documento     → rojo
  for (const [key, isMatch] of Object.entries(res.field_matches || {})) {
    const input = findInputByFieldKey(key);
    if (input) {
      input.classList.remove("valid-field", "invalid-field", "unknown-field");
      if (isMatch === true)       input.classList.add("valid-field");
      else if (isMatch === false) input.classList.add("invalid-field");
      // null → sin clase especial (campo no aplica)
    }
  }

  // ── Dibujar bounding boxes y landmarks sobre las imágenes del DOM ─────────
  // Los datos vienen del backend (coordenadas normalizadas 0-1).
  // Se usan las referencias a los <img> guardadas por collectImages().
  clearFaceOverlays();

  const selfieImgEl = debugSelection.selfieImg;
  const idImgEl     = debugSelection.idImg;

  if (selfieImgEl && res.selfie_face_box) {
    drawFaceDetection(
      selfieImgEl,
      res.selfie_face_box,
      res.selfie_face_landmarks || [],
      "#4caf50",   // verde
      "selfie-face"
    );
  }

  if (idImgEl && res.id_face_box) {
    drawFaceDetection(
      idImgEl,
      res.id_face_box,
      res.id_face_landmarks || [],
      "#4caf50",   // verde — naranja si calidad degradada
      "id-face"
    );
  }
}

function renderAIResults(aiResult) {
  const badge = document.getElementById("selfie-similarity-badge");
  if (!badge || !aiResult || !aiResult.result) return;

  const result      = aiResult.result;
  const shouldReject = aiResult.should_reject;

  const aiDiv = document.createElement("div");
  aiDiv.id = "ai-analysis-section";
  aiDiv.style.cssText =
    "margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.3);";

  // ── Header IA ─────────────────────────────────────────────────────────────
  const header = document.createElement("div");
  header.style.cssText =
    "font-weight:bold;margin-bottom:5px;color:" + (shouldReject ? "#f44336" : "#4caf50") + ";";
  header.textContent = shouldReject ? "IA: RECHAZADO" : "IA: APROBADO";
  aiDiv.appendChild(header);

  // ── Scores base ───────────────────────────────────────────────────────────
  const scores = document.createElement("div");
  scores.style.cssText = "font-size:10px;color:#ccc;margin-bottom:4px;";
  scores.textContent =
    `Coherencia: ${result.coherence_score}% | ` +
    `Integridad: ${result.tampering_score}% | ` +
    `Confianza: ${result.overall_confidence}%`;
  aiDiv.appendChild(scores);

  // ── Opción D: Face match por IA ───────────────────────────────────────────
  if (result.face_match_score != null && result.face_match_score >= 0) {
    const fmScore  = result.face_match_score;
    const fmOk     = fmScore >= 60;
    const fmColor  = fmOk ? "#4caf50" : "#f44336";
    const fmBg     = fmOk ? "rgba(76,175,80,0.12)" : "rgba(244,67,54,0.12)";
    const fmBorder = fmOk ? "#4caf50" : "#f44336";

    const fmDiv = document.createElement("div");
    fmDiv.style.cssText =
      `background:${fmBg};border-left:3px solid ${fmBorder};` +
      `padding:4px 6px;border-radius:0 4px 4px 0;margin-bottom:4px;`;

    const fmTitle = document.createElement("div");
    fmTitle.style.cssText = `font-size:10px;font-weight:bold;color:${fmColor};margin-bottom:2px;`;
    fmTitle.textContent = `${fmOk ? "✓" : "✗"} IA Facial: ${fmScore}% de coincidencia`;
    fmDiv.appendChild(fmTitle);

    if (result.face_match_reasoning) {
      const fmReason = document.createElement("div");
      fmReason.style.cssText = "font-size:9px;color:#bbb;line-height:1.3;";
      fmReason.textContent = result.face_match_reasoning;
      fmDiv.appendChild(fmReason);
    }

    if (result.face_match_issues && result.face_match_issues.length > 0) {
      const fmIssues = document.createElement("div");
      fmIssues.style.cssText = "font-size:9px;color:#ff9800;margin-top:2px;";
      fmIssues.textContent = "⚠ " + result.face_match_issues.join(" · ");
      fmDiv.appendChild(fmIssues);
    }

    aiDiv.appendChild(fmDiv);
  }

  // ── Opción D: Verificación del dorso por IA ───────────────────────────────
  if (result.back_analysis_score != null && result.back_analysis_score >= 0) {
    const bScore  = result.back_analysis_score;
    const isBack  = bScore >= 50;
    const bColor  = isBack ? "#4caf50" : "#f44336";
    const bBg     = isBack ? "rgba(76,175,80,0.12)" : "rgba(244,67,54,0.12)";

    const bDiv = document.createElement("div");
    bDiv.style.cssText =
      `background:${bBg};border-left:3px solid ${bColor};` +
      `padding:4px 6px;border-radius:0 4px 4px 0;margin-bottom:4px;`;

    const bTitle = document.createElement("div");
    bTitle.style.cssText = `font-size:10px;font-weight:bold;color:${bColor};margin-bottom:2px;`;
    bTitle.textContent = isBack
      ? `✓ Dorso verificado por IA (${bScore}%)`
      : `✗ IA: la imagen del dorso no parece ser el reverso del documento (${bScore}%)`;
    bDiv.appendChild(bTitle);

    if (!isBack) {
      const bWarn = document.createElement("div");
      bWarn.style.cssText = "font-size:9px;color:#ff9800;margin-top:2px;";
      bWarn.textContent = "Posiblemente se subió el frente del documento dos veces.";
      bDiv.appendChild(bWarn);
    }

    if (result.back_analysis_issues && result.back_analysis_issues.length > 0) {
      const bIssues = document.createElement("div");
      bIssues.style.cssText = "font-size:9px;color:#ff9800;margin-top:2px;";
      bIssues.textContent = "⚠ " + result.back_analysis_issues.join(" · ");
      bDiv.appendChild(bIssues);
    }

    aiDiv.appendChild(bDiv);
  }

  // ── Problemas de coherencia ───────────────────────────────────────────────
  if (result.coherence_issues && result.coherence_issues.length > 0) {
    const issues = document.createElement("div");
    issues.style.cssText = "font-size:10px;color:#ff9800;margin-top:3px;";
    const issueList = result.coherence_issues.map(
      (i) => (typeof i === "object" ? i.reason || i.message || JSON.stringify(i) : i)
    );
    issues.textContent = "Problemas: " + issueList.join(", ");
    aiDiv.appendChild(issues);
  }

  // ── Áreas sospechosas (tampering) ────────────────────────────────────────
  if (result.tampering_areas && result.tampering_areas.length > 0) {
    const areas = document.createElement("div");
    areas.style.cssText = "font-size:10px;color:#ff9800;margin-top:2px;";
    const areaList = result.tampering_areas.map(
      (a) => (typeof a === "object" ? a.area || a.reason || JSON.stringify(a) : a)
    );
    areas.textContent = "Áreas sospechosas: " + areaList.join(", ");
    aiDiv.appendChild(areas);
  }

  // ── Botones de feedback ───────────────────────────────────────────────────
  const btnContainer = document.createElement("div");
  btnContainer.style.cssText = "display:flex;gap:4px;margin-top:7px;";

  const confirmBtn = document.createElement("button");
  confirmBtn.textContent = "✓ IA Correcta";
  confirmBtn.style.cssText =
    "background:#28a745;color:white;border:none;padding:3px 6px;" +
    "border-radius:3px;cursor:pointer;font-size:10px;flex:1;";
  confirmBtn.onclick = async () => {
    confirmBtn.disabled = true;
    confirmBtn.textContent = "Enviando...";
    await submitAIFeedback(aiResult.analysis_id, true);
    showThankYou(badge, "Feedback enviado. ¡Gracias!");
  };

  const rejectBtn = document.createElement("button");
  rejectBtn.textContent = "✗ IA Incorrecta";
  rejectBtn.style.cssText =
    "background:#dc3545;color:white;border:none;padding:3px 6px;" +
    "border-radius:3px;cursor:pointer;font-size:10px;flex:1;";
  rejectBtn.onclick = () => showCorrectionForm(badge, aiResult);

  btnContainer.appendChild(confirmBtn);
  btnContainer.appendChild(rejectBtn);
  aiDiv.appendChild(btnContainer);

  badge.appendChild(aiDiv);
}

function showCorrectionForm(badge, aiResult) {
  // Remove buttons
  const btnContainer = badge.querySelector("div:last-child");
  if (btnContainer) btnContainer.remove();

  const formDiv = document.createElement("div");
  formDiv.style.cssText = "margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 8px;";

  const title = document.createElement("div");
  title.style.cssText = "font-size: 11px; color: #ff9800; margin-bottom: 6px;";
  title.textContent = "¿Qué está mal? Seleccioná los problemas:";
  formDiv.appendChild(title);

  const issues = [
    { id: "wrong_similarity", label: "Similitud facial incorrecta" },
    { id: "wrong_coherence", label: "Coherencia de datos incorrecta" },
    { id: "wrong_tampering", label: "Detección de fraude incorrecta" },
    { id: "wrong_fields", label: "Campos coincidentes incorrectos" },
  ];

  const checkboxes = [];
  issues.forEach((issue) => {
    const label = document.createElement("label");
    label.style.cssText = "display: flex; align-items: center; gap: 4px; font-size: 10px; color: #ccc; margin: 3px 0; cursor: pointer;";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = issue.id;
    checkbox.style.cssText = "cursor: pointer;";
    checkboxes.push(checkbox);

    const span = document.createElement("span");
    span.textContent = issue.label;

    label.appendChild(checkbox);
    label.appendChild(span);
    formDiv.appendChild(label);
  });

  // Additional notes
  const notesLabel = document.createElement("div");
  notesLabel.style.cssText = "font-size: 10px; color: #aaa; margin-top: 6px; margin-bottom: 2px;";
  notesLabel.textContent = "Notas adicionales (opcional):";
  formDiv.appendChild(notesLabel);

  const notesInput = document.createElement("textarea");
  notesInput.style.cssText =
    "width: 100%; height: 40px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.3); " +
    "color: white; border-radius: 3px; padding: 4px; font-size: 10px; resize: none; box-sizing: border-box;";
  notesInput.placeholder = "Ej: El nombre real es Juan, no José...";
  formDiv.appendChild(notesInput);

  // Submit button
  const submitBtn = document.createElement("button");
  submitBtn.textContent = "Enviar Corrección";
  submitBtn.style.cssText =
    "background: #ff9800; color: black; border: none; padding: 5px 10px; " +
    "border-radius: 3px; cursor: pointer; font-size: 10px; width: 100%; margin-top: 8px; font-weight: bold;";
  submitBtn.onclick = async () => {
    const selectedIssues = checkboxes.filter((cb) => cb.checked).map((cb) => cb.value);
    const notes = notesInput.value.trim();

    if (selectedIssues.length === 0 && !notes) {
      alert("Seleccioná al menos un problema o escribí una nota.");
      return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "Enviando...";

    const corrections = selectedIssues.map((issue) => ({
      field_name: issue,
      expected_value: notes || issue,
      extracted_value: "",
      was_correct: false,
    }));

    await submitAIFeedback(aiResult.analysis_id, false, corrections);
    showThankYou(badge, "Corrección enviada. ¡Gracias!");
  };
  formDiv.appendChild(submitBtn);

  badge.appendChild(formDiv);
}

function showThankYou(badge, message) {
  badge.innerHTML = "";
  badge.style.cssText =
    "position:fixed;bottom:12px;left:50%;transform:translateX(-50%);z-index:999999;" +
    "background:rgba(40,167,69,0.95);color:#fff;padding:12px 16px;border-radius:8px;" +
    "font:13px/20px monospace;min-width:200px;text-align:center;";

  const thankYou = document.createElement("div");
  thankYou.textContent = message;
  badge.appendChild(thankYou);

  // Auto-hide after 3 seconds
  setTimeout(() => {
    badge.style.opacity = "0";
    badge.style.transition = "opacity 0.5s";
    setTimeout(() => badge.remove(), 500);
  }, 3000);
}

function showLoading() {
  clearFeedback();
  const badge = document.createElement("div");
  badge.id = "selfie-similarity-badge";
  badge.style.cssText =
    "position:fixed;bottom:12px;left:50%;transform:translateX(-50%);z-index:999999;" +
    "background:rgba(0,0,0,0.9);color:#fff;padding:12px 16px;border-radius:8px;" +
    "font:13px/20px monospace;min-width:200px;text-align:center;";
  badge.textContent = "Verificando...";
  document.body.appendChild(badge);
}

function showError(message) {
  clearFeedback();
  const badge = document.createElement("div");
  badge.id = "selfie-similarity-badge";
  badge.style.cssText =
    "position:fixed;bottom:12px;left:50%;transform:translateX(-50%);z-index:999999;" +
    "background:rgba(220,53,69,0.95);color:#fff;padding:12px 16px;border-radius:8px;" +
    "font:13px/20px monospace;min-width:200px;text-align:center;";
  badge.textContent = message;
  document.body.appendChild(badge);
}

function clearFeedback() {
  document.querySelectorAll("#selfie-similarity-badge").forEach((el) => el.remove());
  document.querySelectorAll(".valid-field, .invalid-field").forEach((el) => {
    el.classList.remove("valid-field", "invalid-field");
  });
  clearFaceOverlays();
}

function currentSelfieImg() {
  if (debugSelection.selfieImg && debugSelection.selfieImg.isConnected) return debugSelection.selfieImg;
  const images = Array.from(document.querySelectorAll("img")).filter((i) => i.src);
  return images.find((i) => i.src.startsWith("blob:") || i.src.startsWith("data:")) || images[0];
}

// ---------------------------------------------------------------------------
// AI Analysis Functions
// ---------------------------------------------------------------------------

async function analyzeWithAI(selfieB64, idFrontB64, idBackB64, formData) {
  try {
    const resp = await fetch(`${CONFIG.AI_BACKEND_URL}/ai/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        selfie_b64: selfieB64,
        doc_front_b64: idFrontB64,
        doc_back_b64: idBackB64 || null,
        form_data: formData,
      }),
    });

    if (!resp.ok) {
      console.warn("[id-verifier] AI analysis failed:", resp.status);
      return null;
    }

    return await resp.json();
  } catch (err) {
    console.warn("[id-verifier] AI analysis error:", err.message);
    return null;
  }
}

async function submitAIFeedback(analysisId, confirmed, corrections = []) {
  try {
    const resp = await fetch(`${CONFIG.AI_BACKEND_URL}/ai/feedback/${analysisId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        confirmed: confirmed,
        corrections: corrections,
      }),
    });

    if (!resp.ok) {
      console.warn("[id-verifier] AI feedback failed:", resp.status);
      return null;
    }

    return await resp.json();
  } catch (err) {
    console.warn("[id-verifier] AI feedback error:", err.message);
    return null;
  }
}

function showAIAnalysisButton(analysisId, summary) {
  const badge = document.getElementById("selfie-similarity-badge");
  if (!badge) return;

  const aiDiv = document.createElement("div");
  aiDiv.id = "ai-analysis-controls";
  aiDiv.style.cssText = "margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.3); padding-top: 8px;";

  const summaryText = document.createElement("div");
  summaryText.style.cssText = "font-size: 10px; margin-bottom: 6px; color: #aaa;";
  summaryText.textContent = summary || "Análisis IA disponible";
  aiDiv.appendChild(summaryText);

  const btnContainer = document.createElement("div");
  btnContainer.style.cssText = "display: flex; gap: 4px;";

  const confirmBtn = document.createElement("button");
  confirmBtn.textContent = "✓ Correcto";
  confirmBtn.style.cssText =
    "background: #28a745; color: white; border: none; padding: 4px 8px; " +
    "border-radius: 3px; cursor: pointer; font-size: 11px; flex: 1;";
  confirmBtn.onclick = async () => {
    confirmBtn.disabled = true;
    confirmBtn.textContent = "Enviando...";
    await submitAIFeedback(analysisId, true);
    confirmBtn.textContent = "✓ Enviado";
    confirmBtn.style.background = "#155724";
  };

  const rejectBtn = document.createElement("button");
  rejectBtn.textContent = "✗ Rechazar";
  rejectBtn.style.cssText =
    "background: #dc3545; color: white; border: none; padding: 4px 8px; " +
    "border-radius: 3px; cursor: pointer; font-size: 11px; flex: 1;";
  rejectBtn.onclick = async () => {
    rejectBtn.disabled = true;
    rejectBtn.textContent = "Enviando...";
    await submitAIFeedback(analysisId, false);
    rejectBtn.textContent = "✗ Rechazado";
    rejectBtn.style.background = "#721c24";
  };

  btnContainer.appendChild(confirmBtn);
  btnContainer.appendChild(rejectBtn);
  aiDiv.appendChild(btnContainer);

  badge.appendChild(aiDiv);
}

// ---------------------------------------------------------------------------
// AI Learning Dashboard (Phase 1)
// Muestra métricas en tiempo real consumiendo GET /ai/stats
// Panel flotante independiente, toggle con botón 📊
// ---------------------------------------------------------------------------

let dashboardVisible = false;
let dashboardPanel = null;
let dashboardRefreshTimer = null;

async function fetchAIStats() {
  try {
    const resp = await fetch(`${CONFIG.AI_BACKEND_URL}/ai/stats`);
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

function buildTrendSparkline(trend) {
  // Render a minimal ASCII-style sparkline from accuracy_trend data
  if (!trend || trend.length < 2) return "";
  const values = trend.map((t) => t.accuracy_pct || 0);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const blocks = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
  return values
    .slice(-12) // last 12 data points
    .map((v) => blocks[Math.round(((v - min) / range) * (blocks.length - 1))])
    .join("");
}

function renderDashboard(stats) {
  if (!dashboardPanel) return;

  const total     = stats.total_analyses     || 0;
  const confirmed = stats.confirmed_analyses || 0;
  const rejected  = stats.rejected_analyses  || 0;
  const corrections = stats.total_corrections || 0;
  const patterns  = stats.learned_patterns   || 0;
  const accPct    = stats.accuracy_pct != null ? stats.accuracy_pct : (stats.accuracy_rate != null ? (stats.accuracy_rate * 100).toFixed(1) : "—");
  const approvedPct = total > 0 ? ((confirmed / total) * 100).toFixed(1) : "0.0";
  const rejectedPct = total > 0 ? ((rejected  / total) * 100).toFixed(1) : "0.0";

  const trend   = stats.accuracy_trend    || [];
  const topErr  = stats.top_error_fields  || [];
  const active  = stats.active_patterns   || [];
  const sparkline = buildTrendSparkline(trend);

  // Calculate historic baseline vs current (first vs last snapshot)
  let trendArrow = "";
  if (trend.length >= 2) {
    const first = trend[0].accuracy_pct || 0;
    const last  = trend[trend.length - 1].accuracy_pct || 0;
    const delta = (last - first).toFixed(1);
    trendArrow = delta >= 0
      ? `<span style="color:#4caf50">➔ +${delta}%</span>`
      : `<span style="color:#f44336">➔ ${delta}%</span>`;
  }

  dashboardPanel.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
      <span style="font-weight:bold;font-size:12px;color:#fff;">📊 AI Learning Dashboard</span>
      <button id="ai-dashboard-close" style="background:none;border:none;color:#aaa;cursor:pointer;font-size:14px;padding:0 2px;line-height:1;">✕</button>
    </div>

    <div style="border-bottom:1px solid rgba(255,255,255,0.15);padding-bottom:6px;margin-bottom:6px;">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:3px 12px;font-size:10px;color:#ccc;">
        <span>Total análisis:</span>   <span style="color:#fff;font-weight:bold;">${total}</span>
        <span>Aprobados:</span>        <span style="color:#4caf50;">${confirmed} (${approvedPct}%)</span>
        <span>Rechazados:</span>       <span style="color:#f44336;">${rejected} (${rejectedPct}%)</span>
        <span>Correcciones:</span>     <span style="color:#ff9800;">${corrections}</span>
        <span>Patrones activos:</span> <span style="color:#2196f3;">${patterns}</span>
        <span>Precisión actual:</span> <span style="color:#fff;font-weight:bold;">${accPct}% ${trendArrow}</span>
      </div>
    </div>

    ${sparkline ? `
    <div style="margin-bottom:6px;">
      <div style="font-size:9px;color:#888;margin-bottom:2px;">TENDENCIA DE PRECISIÓN</div>
      <div style="font-size:13px;letter-spacing:1px;color:#4caf50;font-family:monospace;">${sparkline}</div>
    </div>` : ""}

    ${topErr.length > 0 ? `
    <div style="border-bottom:1px solid rgba(255,255,255,0.15);padding-bottom:6px;margin-bottom:6px;">
      <div style="font-size:9px;color:#888;margin-bottom:3px;">TOP CAMPOS CON ERRORES</div>
      ${topErr.map((f, i) => `
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#ccc;margin:1px 0;">
          <span>${i + 1}. ${f.field_name}</span>
          <span style="color:#ff9800;">${f.error_count} err (${f.error_rate_pct}%)</span>
        </div>`).join("")}
    </div>` : ""}

    ${active.length > 0 ? `
    <div>
      <div style="font-size:9px;color:#888;margin-bottom:3px;">PATRONES INYECTADOS ACTIVOS</div>
      ${active.slice(0, 5).map((p) => `
        <div style="font-size:9px;color:#ccc;margin:2px 0;line-height:1.3;">
          <span style="color:#2196f3;">●</span> ${p.description || p.pattern_type}
          <span style="color:#888;margin-left:4px;">[${(p.confidence * 100).toFixed(0)}%]</span>
        </div>`).join("")}
    </div>` : `
    <div style="font-size:9px;color:#666;font-style:italic;">Sin patrones activos aún. Envía correcciones para generar patrones.</div>`}

    <div style="margin-top:6px;padding-top:4px;border-top:1px solid rgba(255,255,255,0.1);display:flex;justify-content:flex-end;">
      <button id="ai-dashboard-refresh" style="background:rgba(33,150,243,0.2);border:1px solid rgba(33,150,243,0.4);color:#2196f3;cursor:pointer;font-size:9px;padding:2px 8px;border-radius:3px;">↻ Actualizar</button>
    </div>
  `;

  // Wire up close button
  const closeBtn = dashboardPanel.querySelector("#ai-dashboard-close");
  if (closeBtn) closeBtn.onclick = toggleDashboard;

  // Wire up refresh button
  const refreshBtn = dashboardPanel.querySelector("#ai-dashboard-refresh");
  if (refreshBtn) refreshBtn.onclick = () => loadAndRenderDashboard();
}

async function loadAndRenderDashboard() {
  if (!dashboardPanel) return;

  // Show loading state
  const refreshBtn = dashboardPanel.querySelector("#ai-dashboard-refresh");
  if (refreshBtn) { refreshBtn.textContent = "↻ Cargando…"; refreshBtn.disabled = true; }

  const stats = await fetchAIStats();
  if (!stats) {
    dashboardPanel.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
        <span style="font-weight:bold;font-size:12px;color:#fff;">📊 AI Learning Dashboard</span>
        <button id="ai-dashboard-close" style="background:none;border:none;color:#aaa;cursor:pointer;font-size:14px;padding:0 2px;">✕</button>
      </div>
      <div style="font-size:10px;color:#f44336;">Backend AI no disponible (puerto 8001)</div>`;
    const closeBtn = dashboardPanel.querySelector("#ai-dashboard-close");
    if (closeBtn) closeBtn.onclick = toggleDashboard;
    return;
  }

  renderDashboard(stats);
}

function createDashboardPanel() {
  const panel = document.createElement("div");
  panel.id = "ai-learning-dashboard";
  panel.style.cssText =
    "position:fixed;bottom:12px;right:12px;z-index:999998;" +
    "background:rgba(15,15,20,0.97);color:#fff;padding:12px 14px;border-radius:8px;" +
    "font:11px/1.5 monospace;width:280px;" +
    "box-shadow:0 4px 20px rgba(0,0,0,0.6);border:1px solid rgba(255,255,255,0.1);";
  panel.innerHTML = `<div style="font-size:10px;color:#888;">Cargando métricas…</div>`;
  document.body.appendChild(panel);
  return panel;
}

function createDashboardToggleBtn() {
  // Remove existing button if any
  document.querySelectorAll("#ai-dashboard-toggle-btn").forEach((el) => el.remove());

  const btn = document.createElement("button");
  btn.id = "ai-dashboard-toggle-btn";
  btn.title = "AI Learning Dashboard";
  btn.style.cssText =
    "position:fixed;bottom:12px;right:12px;z-index:999997;" +
    "background:rgba(33,150,243,0.9);color:#fff;border:none;" +
    "width:36px;height:36px;border-radius:50%;cursor:pointer;" +
    "font-size:16px;display:flex;align-items:center;justify-content:center;" +
    "box-shadow:0 2px 8px rgba(0,0,0,0.4);transition:background 0.2s;";
  btn.textContent = "📊";
  btn.onclick = toggleDashboard;
  document.body.appendChild(btn);
  return btn;
}

function toggleDashboard() {
  dashboardVisible = !dashboardVisible;

  const toggleBtn = document.getElementById("ai-dashboard-toggle-btn");

  if (dashboardVisible) {
    // Hide the toggle button while panel is open
    if (toggleBtn) toggleBtn.style.display = "none";

    // Remove stale panel
    document.querySelectorAll("#ai-learning-dashboard").forEach((el) => el.remove());
    dashboardPanel = createDashboardPanel();
    loadAndRenderDashboard();

    // Auto-refresh every 30 seconds while open
    clearInterval(dashboardRefreshTimer);
    dashboardRefreshTimer = setInterval(loadAndRenderDashboard, 30_000);
  } else {
    clearInterval(dashboardRefreshTimer);
    document.querySelectorAll("#ai-learning-dashboard").forEach((el) => el.remove());
    dashboardPanel = null;
    if (toggleBtn) toggleBtn.style.display = "flex";
  }
}

// Initialize toggle button after DOM is ready (called from start())
function initDashboardButton() {
  createDashboardToggleBtn();
}

// ---------------------------------------------------------------------------
// Orquestación: verificar cuando la página cargue imágenes, con debounce
// ---------------------------------------------------------------------------

let debounceTimer = null;
function scheduleVerify() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(verify, 800);
}

// Solo reaccionar a nodos relevantes (previsualizaciones de archivos,
// p-image o inputs de archivo), ignorando mapas, iconos, etc.
function isRelevantNode(node) {
  if (node.nodeType !== 1) return false;
  if (node.matches("p-image") || (node.querySelector && node.querySelector("p-image"))) return true;
  if (node.matches("input[type=file]")) return true;
  if (node.matches("img")) return node.src && (node.src.startsWith("blob:") || node.src.startsWith("data:"));
  return !!(node.querySelector && node.querySelector('img[src^="blob:"], img[src^="data:"], input[type=file]'));
}

const observer = new MutationObserver((mutations) => {
  const relevant = mutations.some((m) =>
    Array.from(m.addedNodes).some((n) => isRelevantNode(n)),
  );
  if (relevant) scheduleVerify();
});

function start() {
  scheduleVerify();
  observer.observe(document.body, { childList: true, subtree: true });
  initDashboardButton();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", start);
} else {
  start();
}
