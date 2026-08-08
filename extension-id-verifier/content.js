// Configuración: ajustar URL del backend según despliegue
const CONFIG = {
  BACKEND_URL: "http://localhost:8000",
  // Palabras que ayudan a identificar los inputs de archivo (selfie vs documento)
  SELFIE_HINTS: ["selfie", "rostro", "retrato"],
  ID_HINTS: ["id", "documento", "dni", "carnet", "cedula", "identificacion"],
  // Mapeo de campos del backend a labels/name del formulario
  FIELD_MAP: {
    primerNombre: "Primer Nombre",
    segundoNombre: "Segundo Nombre",
    primerApellido: "Primer Apellido",
    segundoApellido: "Segundo Apellido",
    numeroDoc: "Número Documento",
    tipoDoc: "Tipo Documento",
    sexo: "Sexo",
    fechaNacimiento: "Fecha Nacimiento",
  },
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
    id: { b64: null, img: null },
    idBack: { b64: null, img: null },
  };
  const used = new Set();

  // 1) Inputs de archivo (fuente más confiable: evita CORS)
  const selfieInput = selectFileInput((s) => matchesHints(s, CONFIG.SELFIE_HINTS));
  const idInput = selectFileInput((s) => matchesHints(s, CONFIG.ID_HINTS));
  const setFromInput = async (slot, input) => {
    if (!input || missing[slot].b64) return;
    const b64 = await readInputFile(input);
    if (b64) {
      missing[slot].b64 = b64;
      missing[slot].img = null;
      console.info(`[id-verifier] ${slot} desde input de archivo:`, input.name || input.id);
    }
  };
  await setFromInput("selfie", selfieInput);
  await setFromInput("id", idInput);

  const inputs = Array.from(document.querySelectorAll("input[type=file]")).filter((i) => i.files && i.files[0]);
  await setFromInput("selfie", inputs[0]);
  await setFromInput("id", inputs[1]);
  await setFromInput("idBack", inputs[2]);

  // 2) Previsualizaciones subidas (blob:/data:) EN ORDEN DE DOM:
  //    selfie → documento frente → documento dorso
  const previews = images.filter((i) => i.src.startsWith("blob:") || i.src.startsWith("data:"));
  for (const slot of ["selfie", "id", "idBack"]) {
    if (missing[slot].b64) continue;
    for (const img of previews) {
      if (used.has(img)) continue;
      used.add(img);
      const b64 = await imgToBase64(img);
      if (b64) {
        missing[slot].b64 = b64;
        missing[slot].img = img;
        console.info(`[id-verifier] ${slot} desde <img> (preview):`, img.src);
        break;
      }
    }
  }

  // 3) Fallback: cualquier <img> legible si no hay blobs
  for (const slot of ["selfie", "id", "idBack"]) {
    if (missing[slot].b64) continue;
    for (const img of images) {
      if (used.has(img)) continue;
      used.add(img);
      const b64 = await imgToBase64(img);
      if (b64) {
        missing[slot].b64 = b64;
        missing[slot].img = img;
        console.info(`[id-verifier] ${slot} desde <img> (fallback):`, img.src);
        break;
      }
    }
  }

  return {
    selfie: missing.selfie.b64,
    id: missing.id.b64,
    idBack: missing.idBack.b64,
    selfieImg: missing.selfie.img,
    idImg: missing.id.img,
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
    console.warn(`[id-verifier] fetch ${img.src} -> HTTP ${resp.status}`);
  } catch (err) {
    console.warn(`[id-verifier] fetch falló para ${img.src}:`, err.message);
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
    console.warn(`[id-verifier] canvas falló para ${img.src}:`, err.message);
    return null;
  }
}

function collectFormData() {
  const formData = {};
  for (const [key, label] of Object.entries(CONFIG.FIELD_MAP)) {
    const input = findInputByFieldKey(key);
    const value = input && (input.value || "").trim() ? input.value.trim() : "";
    formData[key] = value;
    console.info(`[id-verifier] campo '${key}' -> input:`, input ? `${input.tagName}[name=${input.name},id=${input.id}]` : "NO ENCONTRADO", `valor: "${value}"`);
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
  drawDebugLegend(selfieImg, idImg, idBackImg, null, false);
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

function drawDebugLegend(selfieImg, idImg, idBackImg, docDupSim, isDocDup) {
  const existing = document.getElementById("id-verifier-legend");
  if (existing) existing.remove();

  const legend = document.createElement("div");
  legend.id = "id-verifier-legend";
  legend.style.cssText =
    "position:fixed;bottom:12px;right:12px;z-index:999999;background:rgba(0,0,0,0.85);" +
    "color:#fff;font:12px/20px monospace;padding:8px 12px;border-radius:6px;";
  legend.innerHTML =
    `<div><span style="display:inline-block;width:10px;height:10px;background:#2196f3;margin-right:6px;"></span>` +
    `Selfie (${selfieImg ? "SI" : "no encontrada"})</div>` +
    `<div><span style="display:inline-block;width:10px;height:10px;background:#ff9800;margin-right:6px;"></span>` +
    `Documento (${idImg ? "SI" : "no encontrado"})</div>` +
    `<div><span style="display:inline-block;width:10px;height:10px;background:#9c27b0;margin-right:6px;"></span>` +
    `Dorso (${idBackImg ? "SI" : "no encontrado"})</div>` +
    (docDupSim !== null
      ? `<div style="color:${isDocDup ? "#f44336" : "#4caf50"};">` +
        `Documentos: ${docDupSim}%${isDocDup ? " (DUPLICADOS)" : ""}</div>`
      : "");
  document.body.appendChild(legend);
}

function clearHighlight() {
  document.querySelectorAll(".selfie-debug-box, .id-debug-box, .idback-debug-box").forEach((el) => el.remove());
  const legend = document.getElementById("id-verifier-legend");
  if (legend) legend.remove();
}

function renderResults(res) {
  clearFeedback();

  const selfieImg = debugSelection.selfieImg || currentSelfieImg();
  const idImg = debugSelection.idImg;
  const idBackImg = debugSelection.idBackImg;

  const warning =
    res.face_detected && res.face_detected !== "both"
      ? ` (rostro detectado solo en ${res.face_detected === "none" ? "ninguna" : res.face_detected})`
      : "";

  if (selfieImg) {
    const parent = selfieImg.offsetParent || selfieImg.parentElement;
    if (parent) {
      parent.style.position = "relative";
      const badge = document.createElement("div");
      badge.id = "selfie-similarity-badge";
      badge.className = `selfie-overlay-badge ${res.is_same_person ? "badge-match" : "badge-mismatch"}`;
      let text = `Similitud: ${res.facial_similarity}%${warning}`;
      if (res.is_document_duplicate) {
        text += ` | DOC DUPLICADOS: ${res.document_duplicate_similarity}%`;
      }
      badge.textContent = text;
      parent.appendChild(badge);
    }
  }

  // Dibujar leyenda actualizada con estado de duplicados
  drawDebugLegend(selfieImg, idImg, idBackImg, res.document_duplicate_similarity, res.is_document_duplicate);

  for (const [key, isMatch] of Object.entries(res.field_matches || {})) {
    const input = findInputByFieldKey(key);
    if (input) {
      input.classList.remove("valid-field", "invalid-field");
      input.classList.add(isMatch ? "valid-field" : "invalid-field");
    }
  }
}

function showLoading() {
  clearFeedback();
  const img = currentSelfieImg();
  if (!img) return;
  const parent = img.offsetParent || img.parentElement;
  if (!parent) return;
  parent.style.position = "relative";
  const badge = document.createElement("div");
  badge.id = "selfie-similarity-badge";
  badge.className = "selfie-overlay-badge badge-loading";
  badge.textContent = "Verificando...";
  parent.appendChild(badge);
}

function showError(message) {
  clearFeedback();
  const img = currentSelfieImg();
  if (!img) return;
  const parent = img.offsetParent || img.parentElement;
  if (!parent) return;
  parent.style.position = "relative";
  const badge = document.createElement("div");
  badge.id = "selfie-similarity-badge";
  badge.className = "selfie-overlay-badge badge-mismatch";
  badge.textContent = message;
  parent.appendChild(badge);
}

function clearFeedback() {
  document.querySelectorAll("#selfie-similarity-badge").forEach((el) => el.remove());
  document.querySelectorAll(".valid-field, .invalid-field").forEach((el) => {
    el.classList.remove("valid-field", "invalid-field");
  });
}

function currentSelfieImg() {
  if (debugSelection.selfieImg && debugSelection.selfieImg.isConnected) return debugSelection.selfieImg;
  const images = Array.from(document.querySelectorAll("img")).filter((i) => i.src);
  return images.find((i) => i.src.startsWith("blob:") || i.src.startsWith("data:")) || images[0];
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
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", start);
} else {
  start();
}
