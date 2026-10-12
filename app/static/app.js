const $ = (id) => document.getElementById(id);
const canvas = $("zone-canvas");
const context = canvas.getContext("2d");
let selected = null, background = null, zones = [], points = [], mode = "row";
let eventSource = null, chartValues = [];

function toast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  setTimeout(() => node.classList.remove("show"), 3500);
}

function errorMessage(result, fallback) {
  return [result?.error, result?.hint].filter(Boolean).join(" ") || fallback;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const result = await response.json();
  if (!response.ok) throw new Error(errorMessage(result, `Request failed (${response.status})`));
  return result;
}

function redraw() {
  if (!background) return;
  context.drawImage(background, 0, 0, canvas.width, canvas.height);
  zones.forEach((zone, index) => {
    const pts = zone.points;
    context.beginPath();
    context.moveTo(...pts[0]);
    pts.slice(1).forEach(point => context.lineTo(...point));
    context.closePath();
    context.strokeStyle = "#4da3ff";
    context.lineWidth = Math.max(2, canvas.width / 700);
    context.stroke();
    context.fillStyle = "#e8eff688";
    context.font = `${Math.max(12, canvas.width / 70)}px Segoe UI`;
    context.fillText(String(index + 1), pts[0][0] + 4, pts[0][1] + 16);
  });
  points.forEach(point => {
    context.beginPath();
    context.arc(point[0], point[1], 4, 0, Math.PI * 2);
    context.fillStyle = "#ffb454";
    context.fill();
  });
}

function interpolate(a, b, t) {
  return [Math.round(a[0] + (b[0] - a[0]) * t), Math.round(a[1] + (b[1] - a[1]) * t)];
}

function commitPolygon() {
  const count = mode === "row" ? Number($("row-count").value) : 1;
  if (!Number.isInteger(count) || count < 1 || count > 40) {
    toast("Choose a row size from 1 to 40.", true);
    points = [];
    redraw();
    return;
  }
  for (let index = 0; index < count; index++) {
    const left = index / count, right = (index + 1) / count;
    zones.push({points: [
      interpolate(points[0], points[1], left),
      interpolate(points[0], points[1], right),
      interpolate(points[3], points[2], right),
      interpolate(points[3], points[2], left)
    ]});
  }
  points = [];
  $("undo").disabled = false;
  $("clear").disabled = false;
  $("save-zones").disabled = false;
  redraw();
}

canvas.addEventListener("click", event => {
  if (!background) return;
  const rect = canvas.getBoundingClientRect();
  points.push([
    Math.round((event.clientX - rect.left) * canvas.width / rect.width),
    Math.round((event.clientY - rect.top) * canvas.height / rect.height)
  ]);
  if (points.length === 4) commitPolygon();
  else redraw();
});

async function setSource(name, label) {
  selected = name;
  $("source-label").textContent = label;
  $("canvas-empty").hidden = true;
  const frameResponse = await fetch(`/api/source/${encodeURIComponent(name)}/frame`);
  if (!frameResponse.ok) {
    throw new Error(errorMessage(await frameResponse.json(), "Could not load source frame."));
  }
  background = new Image();
  background.src = URL.createObjectURL(await frameResponse.blob());
  await background.decode();
  canvas.width = background.naturalWidth;
  canvas.height = background.naturalHeight;
  zones = [];
  points = [];
  const saved = await requestJson(`/api/zones/${encodeURIComponent(name)}`);
  $("saved-zones").disabled = !saved.saved;
  $("start").disabled = true;
  redraw();
  if (saved.saved) toast("Saved zones are available; use them or draw new zones.");
}

$("sample").addEventListener("click", async () => {
  try {
    await requestJson("/api/source", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({kind: "sample"})});
    await setSource("parking1", "Sample video · parking1");
  } catch (error) { toast(error.message, true); }
});

async function upload(file) {
  const form = new FormData();
  form.append("video", file);
  const result = await requestJson("/api/upload", {method: "POST", body: form});
  await setSource(result.name, `Uploaded video · ${result.filename}`);
}

$("upload").addEventListener("change", async event => {
  if (event.target.files[0]) {
    try { await upload(event.target.files[0]); } catch (error) { toast(error.message, true); }
  }
});
$("dropzone").addEventListener("dragover", event => { event.preventDefault(); $("dropzone").classList.add("drag"); });
$("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("drag"));
$("dropzone").addEventListener("drop", async event => {
  event.preventDefault();
  $("dropzone").classList.remove("drag");
  if (event.dataTransfer.files[0]) {
    try { await upload(event.dataTransfer.files[0]); } catch (error) { toast(error.message, true); }
  }
});
$("use-camera").addEventListener("click", async () => {
  try {
    const index = Number($("camera").value);
    const result = await requestJson("/api/source", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({kind: "camera", index})});
    await setSource(result.name, `Camera · ${index}`);
  } catch (error) { toast(error.message, true); }
});

$("saved-zones").addEventListener("click", async () => {
  try {
    const result = await requestJson(`/api/zones/${encodeURIComponent(selected)}`);
    zones = result.zones;
    points = [];
    $("start").disabled = false;
    $("undo").disabled = zones.length === 0;
    $("clear").disabled = zones.length === 0;
    redraw();
  } catch (error) { toast(error.message, true); }
});
$("mode").addEventListener("click", () => {
  mode = mode === "row" ? "single" : "row";
  $("mode").textContent = `Mode: ${mode === "row" ? "Row" : "Single slot"}`;
});
$("undo").addEventListener("click", () => {
  if (points.length) points = [];
  else zones.pop();
  $("undo").disabled = zones.length === 0 && points.length === 0;
  $("clear").disabled = zones.length === 0 && points.length === 0;
  $("save-zones").disabled = zones.length === 0;
  redraw();
});
$("clear").addEventListener("click", () => {
  zones = [];
  points = [];
  $("undo").disabled = $("clear").disabled = $("save-zones").disabled = true;
  redraw();
});
$("save-zones").addEventListener("click", async () => {
  try {
    if (!selected || !zones.length) throw new Error("Draw at least one slot before saving.");
    await requestJson(`/api/zones/${encodeURIComponent(selected)}`, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({zones})
    });
    $("start").disabled = false;
    $("saved-zones").disabled = false;
    toast(`${zones.length} zones saved.`);
  } catch (error) { toast(error.message, true); }
});

function applyEvent(data) {
  $("occupied").textContent = data.occupied ?? "—";
  $("available").textContent = data.available ?? "—";
  $("lot-full").hidden = !data.lot_full;
  $("video-time").textContent = `${Number(data.video_time_s || 0).toFixed(1)} s`;
  $("inference").textContent = `${Number(data.inference_hz || 0).toFixed(1)} Hz · ${Number(data.inference_ms || 0).toFixed(0)} ms`;
  $("display").textContent = `${Number(data.display_fps || 0).toFixed(1)} FPS`;
  $("lag").textContent = `${Number(data.lag_ms || 0).toFixed(0)} ms`;
  $("chips").replaceChildren(...(data.states || []).map((state, index) => {
    const chip = document.createElement("span");
    chip.className = `chip${state ? " occupied" : ""}`;
    chip.textContent = `${index + 1} ${state ? "OCCUPIED" : "FREE"}`;
    return chip;
  }));
  chartValues.push(data.occupied || 0);
  chartValues = chartValues.slice(-120);
  const chart = $("chart"), ctx = chart.getContext("2d");
  ctx.clearRect(0, 0, chart.width, chart.height);
  if (chartValues.length > 1) {
    const maximum = Math.max(1, ...chartValues);
    ctx.strokeStyle = "#4da3ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    chartValues.forEach((value, index) => {
      const x = index * chart.width / (chartValues.length - 1);
      const y = chart.height - 8 - value * (chart.height - 20) / maximum;
      index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
}

$("start").addEventListener("click", async () => {
  try {
    await requestJson("/api/start", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name: selected, settings: {
        model: $("model").value, rule: $("rule").value, roi: $("roi").checked,
        smoothing: $("smooth").checked, cars_only: $("cars-only").checked
      }})
    });
    $("stream").src = `/stream?started=${Date.now()}`;
    $("video-empty").hidden = true;
    $("start").disabled = true;
    $("stop").disabled = false;
    $("status").classList.add("running");
    $("status").lastElementChild.textContent = "Live";
    chartValues = [];
    eventSource?.close();
    eventSource = new EventSource("/events");
    eventSource.onmessage = event => {
      const data = JSON.parse(event.data);
      if (!data.done) applyEvent(data);
      else finishSession();
    };
    eventSource.onerror = () => toast("Live event stream disconnected.", true);
  } catch (error) { toast(error.message, true); }
});

function finishSession() {
  $("stop").disabled = true;
  $("status").classList.remove("running");
  $("status").lastElementChild.textContent = "Finished";
  eventSource?.close();
  if (selected) {
    const names = [["output.mp4", "Annotated video"], ["events.csv", "Events"], ["sessions.csv", "Sessions"], ["summary.json", "Summary"]];
    $("downloads").replaceChildren(...names.map(([file, label]) => {
      const link = document.createElement("a");
      link.href = `/api/download/${encodeURIComponent(selected)}/${file}`;
      link.textContent = `Download ${label}`;
      return link;
    }));
  }
}

$("stop").addEventListener("click", async () => {
  try {
    await requestJson("/api/stop", {method: "POST"});
    finishSession();
  } catch (error) { toast(error.message, true); }
});
