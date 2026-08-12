// Configuración: ajustar URL del backend según despliegue
const CONFIG = {
  BACKEND_URL: "http://localhost:8000",
  AI_BACKEND_URL: "http://localhost:8001",
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
      console.info("[id-verifier] desde input de archivo:", input.name || input.id);
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
        console.info("[id-verifier] desde <img> (preview):", img.src);
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
        console.info("[id-verifier] desde <img> (fallback):", img.src);
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

function renderResults(res) {
  clearFeedback();

  const selfieImg = debugSelection.selfieImg || currentSelfieImg();
  const idImg = debugSelection.idImg;
  const idBackImg = debugSelection.idBackImg;

  const warning =
    res.face_detected && res.face_detected !== "both"
      ? ` (rostro detectado solo en ${res.face_detected === "none" ? "ninguna" : res.face_detected})`
      : "";

  // Create fixed badge at bottom of screen
  const badge = document.createElement("div");
  badge.id = "selfie-similarity-badge";
  badge.style.cssText =
    "position:fixed;bottom:12px;left:50%;transform:translateX(-50%);z-index:999999;" +
    "background:rgba(0,0,0,0.9);color:#fff;padding:12px 16px;border-radius:8px;" +
    "font:13px/20px monospace;min-width:320px;box-shadow:0 4px 12px rgba(0,0,0,0.4);";

  // Header with status
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;gap:8px;margin-bottom:8px;";
  const statusDot = document.createElement("span");
  statusDot.style.cssText = `width:10px;height:10px;border-radius:50%;background:${res.is_same_person ? "#4caf50" : "#f44336"};flex-shrink:0;`;
  const headerText = document.createElement("span");
  headerText.style.cssText = "font-weight:bold;font-size:14px;";
  headerText.textContent = `Similitud: ${res.facial_similarity}%${warning}`;
  header.appendChild(statusDot);
  header.appendChild(headerText);
  badge.appendChild(header);

  // Document duplicate info
  if (res.is_document_duplicate) {
    const dupInfo = document.createElement("div");
    dupInfo.style.cssText = "color:#ff9800;font-size:11px;margin-bottom:4px;";
    dupInfo.textContent = `DOC DUPLICADOS: ${res.document_duplicate_similarity}%`;
    badge.appendChild(dupInfo);
  }

  // Field matches summary
  const fieldSummary = document.createElement("div");
  fieldSummary.style.cssText = "font-size:11px;color:#aaa;border-top:1px solid rgba(255,255,255,0.2);padding-top:6px;margin-top:4px;";
  const matches = Object.entries(res.field_matches || {});
  const matchCount = matches.filter(([, v]) => v).length;
  fieldSummary.textContent = `Campos: ${matchCount}/${matches.length} coinciden`;
  badge.appendChild(fieldSummary);

  document.body.appendChild(badge);

  // Highlight inputs
  for (const [key, isMatch] of Object.entries(res.field_matches || {})) {
    const input = findInputByFieldKey(key);
    if (input) {
      input.classList.remove("valid-field", "invalid-field");
      input.classList.add(isMatch ? "valid-field" : "invalid-field");
    }
  }
}

function renderAIResults(aiResult) {
  const badge = document.getElementById("selfie-similarity-badge");
  if (!badge || !aiResult || !aiResult.result) return;

  const result = aiResult.result;
  const shouldReject = aiResult.should_reject;

  // Create AI section
  const aiDiv = document.createElement("div");
  aiDiv.id = "ai-analysis-section";
  aiDiv.style.cssText = "margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.3);";

  // AI Header
  const header = document.createElement("div");
  header.style.cssText = "font-weight: bold; margin-bottom: 4px; color: " + (shouldReject ? "#f44336" : "#4caf50") + ";";
  header.textContent = shouldReject ? "IA: RECHAZADO" : "IA: APROBADO";
  aiDiv.appendChild(header);

  // Scores
  const scores = document.createElement("div");
  scores.style.cssText = "font-size: 10px; color: #ccc;";
  scores.innerHTML =
    `Coherencia: ${result.coherence_score}% | Integridad: ${result.tampering_score}% | Confianza: ${result.overall_confidence}%`;
  aiDiv.appendChild(scores);

  // Issues if any
  if (result.coherence_issues && result.coherence_issues.length > 0) {
    const issues = document.createElement("div");
    issues.style.cssText = "font-size: 10px; color: #ff9800; margin-top: 4px;";
    const issueList = result.coherence_issues.map(i => typeof i === "object" ? (i.reason || i.message || JSON.stringify(i)) : i);
    issues.textContent = "Problemas: " + issueList.join(", ");
    aiDiv.appendChild(issues);
  }

  // Tampering areas if any
  if (result.tampering_areas && result.tampering_areas.length > 0) {
    const areas = document.createElement("div");
    areas.style.cssText = "font-size: 10px; color: #ff9800; margin-top: 2px;";
    const areaList = result.tampering_areas.map(a => typeof a === "object" ? (a.area || a.reason || JSON.stringify(a)) : a);
    areas.textContent = "Áreas sospechosas: " + areaList.join(", ");
    aiDiv.appendChild(areas);
  }

  // Feedback buttons
  const btnContainer = document.createElement("div");
  btnContainer.style.cssText = "display: flex; gap: 4px; margin-top: 6px;";

  const confirmBtn = document.createElement("button");
  confirmBtn.textContent = "✓ IA Correcta";
  confirmBtn.style.cssText =
    "background: #28a745; color: white; border: none; padding: 3px 6px; " +
    "border-radius: 3px; cursor: pointer; font-size: 10px; flex: 1;";
  confirmBtn.onclick = async () => {
    confirmBtn.disabled = true;
    confirmBtn.textContent = "Enviando...";
    await submitAIFeedback(aiResult.analysis_id, true);
    showThankYou(badge, "Feedback enviado. Gracias!");
  };

  const rejectBtn = document.createElement("button");
  rejectBtn.textContent = "✗ IA Incorrecta";
  rejectBtn.style.cssText =
    "background: #dc3545; color: white; border: none; padding: 3px 6px; " +
    "border-radius: 3px; cursor: pointer; font-size: 10px; flex: 1;";
  rejectBtn.onclick = () => {
    showCorrectionForm(badge, aiResult);
  };

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
