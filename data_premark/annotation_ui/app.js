"use strict";

const app = {
  state: null,
  filtered: [],
  currentIndex: 0,
  current: null,
  image: new Image(),
  mode: "inspect",
  dirty: false,
  pivot: null,
  tip: null,
};

const $ = (selector) => document.querySelector(selector);
const canvas = $("#annotationCanvas");
const ctx = canvas.getContext("2d");
const form = $("#reviewForm");

function numberOrNull(value) {
  const text = String(value ?? "").trim();
  if (!text) return null;
  const valueNumber = Number(text);
  return Number.isFinite(valueNumber) ? valueNumber : null;
}

function angleFrom(pivot, tip) {
  let angle = Math.atan2(tip.y - pivot.y, tip.x - pivot.x) * 180 / Math.PI;
  if (angle < 0) angle += 360;
  return angle;
}

function currentFormValue(name) {
  return form.elements.namedItem(name).value;
}

function setFormValue(name, value) {
  form.elements.namedItem(name).value = value ?? "";
}

function markDirty(changed = true) {
  app.dirty = changed;
  $("#saveState").textContent = changed ? "有未保存修改" : "已保存";
}

function completed(item) {
  return ["accepted", "corrected"].includes(item.review.review_status);
}

function updateProgress(completedCount = null) {
  const count = completedCount ?? app.state.items.filter(completed).length;
  const total = app.state.total;
  $("#progressText").textContent = `${count} / ${total} 已完成`;
  $("#progressBar").style.width = `${total ? count / total * 100 : 0}%`;
}

function filterItems() {
  const mode = $("#filterSelect").value;
  app.filtered = app.state.items.filter((item) => {
    if (mode === "pending") return !completed(item);
    if (mode === "completed") return completed(item);
    return true;
  });
  if (!app.filtered.some((item) => item.record_id === app.current?.record_id)) app.currentIndex = 0;
  renderSampleList();
}

function renderSampleList() {
  const list = $("#sampleList");
  list.innerHTML = "";
  app.filtered.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `sample-item ${item.record_id === app.current?.record_id ? "active" : ""}`;
    button.innerHTML = `<span class="sample-item-top"><span>${String(index + 1).padStart(2, "0")} · ${item.record_id.slice(0, 8)}</span><i class="status-dot ${completed(item) ? "done" : ""}"></i></span><small>${item.sampling.stratum} · ${item.image.path}</small>`;
    button.addEventListener("click", () => navigateTo(index));
    list.appendChild(button);
  });
  $("#sampleCounter").textContent = `${app.filtered.length} 张`;
}

function defaultPivot(item) {
  const reviewX = numberOrNull(item.review.pivot_x);
  const reviewY = numberOrNull(item.review.pivot_y);
  if (reviewX !== null && reviewY !== null) return {x: reviewX, y: reviewY};
  const point = item.auto_annotation.pivot?.point;
  return point && Number.isFinite(point.x) && Number.isFinite(point.y) ? {x: point.x, y: point.y} : null;
}

function defaultCandidate(item) {
  const requested = item.review.pointer_candidate_id || item.auto_annotation.selected_pointer_candidate_id;
  const candidates = item.auto_annotation.pointer_candidates || [];
  return candidates.find((candidate) => candidate.candidate_id === requested) || candidates[0] || null;
}

function applyCandidate(candidate, mark = true) {
  if (!candidate?.tip || !app.pivot) return;
  app.tip = {x: Number(candidate.tip.x), y: Number(candidate.tip.y)};
  setFormValue("pointer_candidate_id", candidate.candidate_id || "");
  setFormValue("pointer_angle_deg", angleFrom(app.pivot, app.tip).toFixed(3));
  if (mark) markDirty();
  renderCandidates();
  draw();
}

function populateForm(item) {
  const review = item.review;
  const autoShape = item.auto_annotation.shape?.predicted || item.sampling.stratum;
  setFormValue("review_shape", review.review_shape || autoShape);
  app.pivot = defaultPivot(item);
  setFormValue("pivot_x", app.pivot ? app.pivot.x.toFixed(6) : "");
  setFormValue("pivot_y", app.pivot ? app.pivot.y.toFixed(6) : "");
  ["pointer_candidate_id", "pointer_angle_deg", "reading", "unit", "range_min", "range_max", "minor_division", "comment"].forEach((name) => setFormValue(name, review[name]));
  setFormValue("pointer_role", review.pointer_role || "measurement_pointer");
  setFormValue("review_status", review.review_status || "pending");
  app.tip = null;
  const candidate = defaultCandidate(item);
  if (candidate && !review.pointer_angle_deg) applyCandidate(candidate, false);
  else if (candidate) app.tip = {x: Number(candidate.tip.x), y: Number(candidate.tip.y)};
  markDirty(false);
}

async function navigateTo(index, force = false, preserveMessage = false) {
  if (!app.filtered.length) return;
  if (app.dirty && !force && !window.confirm("当前修改尚未保存，确定离开吗？")) return;
  app.currentIndex = Math.max(0, Math.min(index, app.filtered.length - 1));
  app.current = app.filtered[app.currentIndex];
  populateForm(app.current);
  $("#recordId").textContent = app.current.record_id;
  $("#shapeBadge").textContent = app.current.sampling.stratum;
  $("#imageMeta").textContent = `${app.current.image.width} × ${app.current.image.height} · ${app.current.image.path}`;
  $("#loadingOverlay").style.display = "block";
  app.image.onload = () => {
    canvas.width = app.image.naturalWidth;
    canvas.height = app.image.naturalHeight;
    $("#loadingOverlay").style.display = "none";
    draw();
  };
  app.image.onerror = () => showMessage("原图加载失败", true);
  app.image.src = `/api/image/${encodeURIComponent(app.current.record_id)}?v=${Date.now()}`;
  renderSampleList();
  renderCandidates();
  if (!preserveMessage) $("#messageBox").textContent = "";
}

function drawCross(point, color, radius = 9) {
  const x = point.x * canvas.width;
  const y = point.y * canvas.height;
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(2, canvas.width / 500);
  ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x - radius * 1.5, y); ctx.lineTo(x + radius * 1.5, y); ctx.moveTo(x, y - radius * 1.5); ctx.lineTo(x, y + radius * 1.5); ctx.stroke();
}

function drawLine(start, end, color, width = 3, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(start.x * canvas.width, start.y * canvas.height); ctx.lineTo(end.x * canvas.width, end.y * canvas.height); ctx.stroke();
  ctx.restore();
}

function draw() {
  if (!app.image.complete || !canvas.width) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(app.image, 0, 0, canvas.width, canvas.height);
  const auto = app.current.auto_annotation;
  const box = auto.dial_boundary?.detector_box;
  if (box) {
    ctx.strokeStyle = "rgba(239,106,58,.8)";
    ctx.lineWidth = Math.max(2, canvas.width / 600);
    ctx.strokeRect(box.x_min * canvas.width, box.y_min * canvas.height, (box.x_max - box.x_min) * canvas.width, (box.y_max - box.y_min) * canvas.height);
  }
  const autoPivot = auto.pivot?.point;
  if (autoPivot) drawCross(autoPivot, "#f0c45b", 7);
  (auto.pointer_candidates || []).forEach((candidate) => {
    if (candidate.line?.start && candidate.line?.end) drawLine(candidate.line.start, candidate.line.end, "#f0c45b", Math.max(2, canvas.width / 700), .55);
  });
  if (app.pivot) drawCross(app.pivot, "#21d4b4", 10);
  if (app.pivot && app.tip) {
    drawLine(app.pivot, app.tip, "#21d4b4", Math.max(3, canvas.width / 420));
    drawCross(app.tip, "#21d4b4", 6);
  }
}

function renderCandidates() {
  const target = $("#candidateList");
  target.innerHTML = "";
  const selectedId = currentFormValue("pointer_candidate_id");
  (app.current?.auto_annotation.pointer_candidates || []).forEach((candidate) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `candidate ${candidate.candidate_id === selectedId ? "selected" : ""}`;
    const confidence = Number(candidate.confidence);
    button.textContent = `${candidate.candidate_id} · ${Number.isFinite(confidence) ? Math.round(confidence * 100) + "%" : "—"}`;
    button.addEventListener("click", () => applyCandidate(candidate));
    target.appendChild(button);
  });
  if (!target.children.length) target.textContent = "没有自动候选，请手动点击轴心和针尖。";
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
    y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)),
  };
}

function updatePointerTip(point) {
  if (!app.pivot) return showMessage("请先点击轴心", true);
  app.tip = point;
  setFormValue("pointer_candidate_id", "");
  setFormValue("pointer_angle_deg", angleFrom(app.pivot, app.tip).toFixed(3));
  markDirty();
  renderCandidates();
  draw();
}

let pointerDragging = false;
canvas.addEventListener("pointerdown", (event) => {
  if (!app.current || app.mode === "inspect") return;
  const point = canvasPoint(event);
  if (app.mode === "pivot") {
    app.pivot = point;
    setFormValue("pivot_x", point.x.toFixed(6));
    setFormValue("pivot_y", point.y.toFixed(6));
    if (app.tip) setFormValue("pointer_angle_deg", angleFrom(app.pivot, app.tip).toFixed(3));
    markDirty();
    renderCandidates();
    draw();
  } else if (app.mode === "pointer") {
    pointerDragging = true;
    canvas.setPointerCapture(event.pointerId);
    updatePointerTip(point);
  }
});
canvas.addEventListener("pointermove", (event) => {
  if (pointerDragging && app.mode === "pointer") updatePointerTip(canvasPoint(event));
});
canvas.addEventListener("pointerup", (event) => {
  pointerDragging = false;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
});
canvas.addEventListener("pointercancel", () => { pointerDragging = false; });

function setMode(mode) {
  app.mode = mode;
  document.querySelectorAll(".mode").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
}

function collectPayload(status) {
  const payload = {};
  new FormData(form).forEach((value, key) => { payload[key] = String(value).trim(); });
  payload.review_status = status;
  return payload;
}

function showMessage(message, error = false) {
  const box = $("#messageBox");
  box.textContent = message;
  box.className = `message-box ${error ? "error" : "success"}`;
}

async function save(status, moveNext) {
  if (!app.current) return;
  try {
    const previousIndex = app.currentIndex;
    const currentId = app.current.record_id;
    const response = await fetch(`/api/records/${encodeURIComponent(app.current.record_id)}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(collectPayload(status)),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "保存失败");
    app.current.review = result.review;
    markDirty(false);
    updateProgress(result.completed);
    showMessage(status === "pending" ? "已暂存，仍计为待审核" : "保存成功");
    filterItems();
    if (moveNext && app.filtered.length) {
      const stillVisible = app.filtered.some((item) => item.record_id === currentId);
      const next = Math.min(stillVisible ? previousIndex + 1 : previousIndex, app.filtered.length - 1);
      await navigateTo(next, true, true);
    }
  } catch (error) {
    showMessage(error.message, true);
  }
}

async function start() {
  const response = await fetch("/api/state");
  app.state = await response.json();
  const shapeSelect = $("#reviewShape");
  app.state.shape_labels.forEach((shape) => {
    const option = document.createElement("option"); option.value = shape; option.textContent = shape; shapeSelect.appendChild(option);
  });
  updateProgress(app.state.completed);
  filterItems();
  await navigateTo(0, true);
}

document.querySelectorAll(".mode").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
form.addEventListener("input", () => markDirty());
form.elements.namedItem("pivot_x").addEventListener("input", () => { const x = numberOrNull(currentFormValue("pivot_x")); if (x !== null && app.pivot) { app.pivot.x = x; draw(); } });
form.elements.namedItem("pivot_y").addEventListener("input", () => { const y = numberOrNull(currentFormValue("pivot_y")); if (y !== null && app.pivot) { app.pivot.y = y; draw(); } });
$("#filterSelect").addEventListener("change", () => { filterItems(); navigateTo(0, true); });
$("#pendingButton").addEventListener("click", () => save("pending", true));
$("#acceptedButton").addEventListener("click", () => save("accepted", true));
$("#correctedButton").addEventListener("click", () => save("corrected", true));
window.addEventListener("beforeunload", (event) => { if (app.dirty) { event.preventDefault(); event.returnValue = ""; } });
window.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") { event.preventDefault(); save(currentFormValue("review_status") || "pending", false); return; }
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (event.key.toLowerCase() === "a") navigateTo(app.currentIndex - 1);
  if (event.key.toLowerCase() === "d") navigateTo(app.currentIndex + 1);
  if (event.key.toLowerCase() === "p") setMode("pivot");
  if (event.key.toLowerCase() === "t") setMode("pointer");
});

start().catch((error) => showMessage(`初始化失败：${error.message}`, true));
