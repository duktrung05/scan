const state = {
  file: null,
  previewUrl: null,
  schema: null,
  result: null,
  tab: "structured",
};

const $ = (selector) => document.querySelector(selector);
const elements = {
  healthDot: $("#health-dot"),
  healthLabel: $("#health-label"),
  modelLabel: $("#model-label"),
  mode: $("#mode"),
  documentType: $("#document-type"),
  language: $("#language"),
  run: $("#run-button"),
  drop: $("#drop-zone"),
  browse: $("#browse-button"),
  replace: $("#replace-button"),
  fileInput: $("#file-input"),
  fileMeta: $("#file-meta"),
  preview: $("#preview"),
  editor: $("#schema-editor"),
  resetSchema: $("#reset-schema"),
  schemaError: $("#schema-error"),
  schemaState: $("#schema-state"),
  badge: $("#validity-badge"),
  empty: $("#empty-output"),
  output: $("#output-code"),
  artifacts: $("#artifact-links"),
  message: $("#message"),
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return response;
}

async function initialize() {
  try {
    const [healthResponse, schemasResponse] = await Promise.all([
      api("/api/health"),
      api("/api/schemas"),
    ]);
    const health = await healthResponse.json();
    const schemas = await schemasResponse.json();
    elements.healthDot.classList.add("ok");
    elements.healthLabel.textContent = "Pipeline ready";
    elements.modelLabel.textContent = `${health.provider} · ${health.model}`;
    elements.documentType.innerHTML = schemas.schemas
      .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
      .join("");
    await loadSchema();
  } catch (error) {
    elements.healthDot.classList.add("fail");
    elements.healthLabel.textContent = "Pipeline unavailable";
    showMessage(error.message);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadSchema() {
  if (elements.mode.value === "markdown") {
    elements.editor.value = "Markdown extraction does not require a JSON Schema.";
    elements.editor.disabled = true;
    elements.documentType.disabled = true;
    elements.resetSchema.disabled = true;
    elements.schemaError.textContent = "";
    return;
  }
  elements.editor.disabled = false;
  elements.documentType.disabled = false;
  elements.resetSchema.disabled = false;
  const response = await api(`/api/schemas/${encodeURIComponent(elements.documentType.value)}`);
  state.schema = await response.json();
  elements.editor.value = JSON.stringify(state.schema, null, 2);
  elements.schemaState.textContent = "Built-in schema";
  validateEditor();
}

function validateEditor() {
  if (elements.mode.value === "markdown") return true;
  try {
    const parsed = JSON.parse(elements.editor.value);
    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
      throw new Error("Schema must be a JSON object");
    }
    elements.schemaError.textContent = "";
    elements.schemaState.textContent = "Schema syntax valid";
    elements.run.disabled = !state.file;
    return true;
  } catch (error) {
    elements.schemaError.textContent = error.message;
    elements.schemaState.textContent = "Schema needs attention";
    elements.run.disabled = true;
    return false;
  }
}

function chooseFile(file) {
  if (!file) return;
  state.file = file;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  elements.fileMeta.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  elements.drop.classList.add("hidden");
  elements.preview.classList.remove("hidden");
  elements.replace.classList.remove("hidden");
  elements.preview.innerHTML = "";
  if (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")) {
    const frame = document.createElement("iframe");
    frame.src = state.previewUrl;
    frame.title = "PDF preview";
    elements.preview.append(frame);
  } else {
    const image = document.createElement("img");
    image.src = state.previewUrl;
    image.alt = "Uploaded document preview";
    elements.preview.append(image);
  }
  elements.run.disabled = !validateEditor();
  clearMessage();
}

function showMessage(text) {
  elements.message.textContent = text;
  elements.message.classList.remove("hidden");
}

function clearMessage() {
  elements.message.textContent = "";
  elements.message.classList.add("hidden");
}

function setLoading(loading) {
  elements.run.classList.toggle("loading", loading);
  elements.run.disabled = loading || !state.file;
}

function selectTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  renderOutput();
}

function renderOutput() {
  if (!state.result) return;
  let value;
  if (state.tab === "raw") {
    value = state.result.raw_output;
  } else if (state.tab === "validation") {
    value = {
      valid: state.result.valid,
      repaired: state.result.repaired,
      issues: state.result.issues,
      inference: state.result.inference,
    };
  } else {
    value = state.result.mode === "markdown"
      ? state.result.raw_output
      : state.result.normalized_output;
  }
  elements.output.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function showResult(result) {
  state.result = result;
  elements.empty.classList.add("hidden");
  elements.output.classList.remove("hidden");
  if (result.mode === "markdown") {
    elements.badge.className = "badge markdown";
    elements.badge.textContent = "Markdown";
  } else if (result.valid) {
    elements.badge.className = "badge valid";
    elements.badge.textContent = result.repaired ? "Valid · repaired" : "Valid";
  } else {
    elements.badge.className = "badge invalid";
    elements.badge.textContent = "Needs review";
  }
  elements.artifacts.innerHTML = Object.entries(result.artifacts)
    .filter(([name]) => name !== "source")
    .map(([name, filename]) => {
      const href = `/api/runs/${encodeURIComponent(result.run_id)}/artifacts/${encodeURIComponent(filename)}`;
      return `<a href="${href}">${escapeHtml(name)}</a>`;
    })
    .join("");
  selectTab("structured");
}

async function runExtraction() {
  if (!state.file || !validateEditor()) return;
  clearMessage();
  setLoading(true);
  const form = new FormData();
  form.append("file", state.file);
  form.append("mode", elements.mode.value);
  form.append("document_type", elements.documentType.value);
  form.append("language", elements.language.value);
  if (elements.mode.value === "json") form.append("schema_text", elements.editor.value);
  try {
    const response = await api("/api/extract", { method: "POST", body: form });
    showResult(await response.json());
  } catch (error) {
    showMessage(error.message);
  } finally {
    setLoading(false);
  }
}

elements.browse.addEventListener("click", () => elements.fileInput.click());
elements.replace.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", (event) => chooseFile(event.target.files[0]));
elements.drop.addEventListener("click", () => elements.fileInput.click());
elements.drop.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") elements.fileInput.click();
});
elements.drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.drop.classList.add("drag");
});
elements.drop.addEventListener("dragleave", () => elements.drop.classList.remove("drag"));
elements.drop.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.drop.classList.remove("drag");
  chooseFile(event.dataTransfer.files[0]);
});
elements.mode.addEventListener("change", async () => {
  await loadSchema();
  elements.run.disabled = !state.file;
});
elements.documentType.addEventListener("change", loadSchema);
elements.resetSchema.addEventListener("click", loadSchema);
elements.editor.addEventListener("input", validateEditor);
elements.run.addEventListener("click", runExtraction);
document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => selectTab(button.dataset.tab));
});

initialize();

