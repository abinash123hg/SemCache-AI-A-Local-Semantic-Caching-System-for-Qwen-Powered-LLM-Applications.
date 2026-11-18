// SemCache AI frontend. Plain JS, no frameworks.
// The API base lives here once; it must match backend/config.py (port 8000).
const API = "http://127.0.0.1:8000";

// ---------- helpers ----------

async function api(path, options = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const msg = data && data.error ? data.error.message : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) {
    node.append(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

function fmtMs(v) { return v == null ? "-" : `${Math.round(v)} ms`; }
function fmtPct(v) { return v == null ? "-" : `${(v * 100).toFixed(1)}%`; }

function backendDownMessage(err) {
  if (err instanceof TypeError) {
    return "Cannot reach the backend at " + API + ". Is the SemCache Backend window open?";
  }
  return err.message;
}

// The loading status cycles through what the cache is actually doing, so the
// wait is readable instead of a static "Thinking". Order matches the real flow:
// lookup first, then generation.
const THINKING_PHRASES = [
  "Thinking",
  "Searching the cache",
  "Comparing similarity",
  "Generating an answer",
];

function startThinking(node) {
  let i = 0;
  node.textContent = THINKING_PHRASES[0];
  const timer = setInterval(() => {
    i = (i + 1) % THINKING_PHRASES.length;
    node.textContent = THINKING_PHRASES[i];
  }, 1100);
  return () => clearInterval(timer);
}

// ---------- tabs ----------

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "cache") loadCache();
    if (btn.dataset.tab === "analytics") loadAnalytics();
    if (btn.dataset.tab === "benchmark") loadBenchResults();
    if (btn.dataset.tab === "settings") loadSettings();
  });
});

// ---------- chat ----------

const messages = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatStatus = document.getElementById("chat-status");
let busy = false;

document.getElementById("new-chat").addEventListener("click", () => {
  messages.innerHTML = "";
  messages.append(el("div", { class: "empty-state" },
    "Ask a question to start. Repeats and paraphrases should come back from the cache."));
});

// Render the "details" inspector plus the flag-as-wrong button once the answer
// (streamed or cached) is complete.
function fillAnswer(botMsg, question, d) {
  botMsg.innerHTML = "";
  botMsg.append(el("div", { class: "role" }, "SemCache"));
  const textLine = el("div", { class: "text" }, d.answer);
  textLine.append(el("span", { class: "badge " + (d.hit ? "hit" : "miss") },
    d.hit ? (d.exact_hit ? "HIT (exact)" : "HIT (semantic)") : "MISS"));
  botMsg.append(textLine);

  const inspector = el("div", { class: "inspector" });
  const lines = [];
  if (d.hit) {
    lines.push(`Matched cached question: "${d.matched_question}"`);
    lines.push(`Similarity: ${d.similarity} | Margin: ${d.margin} | Cache age: ${Math.round(d.cache_age_seconds)}s`);
  } else {
    lines.push(`Miss reason: ${d.miss_reason}`);
  }
  const t = d.timings_ms;
  lines.push(`Timings: embed ${fmtMs(t.embedding)}, retrieve ${fmtMs(t.retrieval)}, generate ${fmtMs(t.generation)}, total ${fmtMs(t.total)}`);
  lines.push(`Model: ${d.chat_model} | Tokens: prompt ${d.tokens.prompt ?? "?"}, completion ${d.tokens.completion ?? "?"}`);
  for (const line of lines) inspector.append(el("div", {}, line));

  const actions = el("div", { class: "msg-actions" });
  const toggle = el("button", {
    class: "link-btn",
    onclick: () => inspector.classList.toggle("open"),
  }, "details");
  actions.append(toggle);
  if (d.hit) {
    actions.append(el("button", {
      class: "link-btn warn",
      onclick: async () => {
        try {
          await api("/feedback", {
            method: "POST",
            body: JSON.stringify({ entry_id: d.entry_id, question, reason: "flagged as wrong from chat" }),
          });
          chatStatus.textContent = "Flagged. The cached entry was removed.";
        } catch (err) { chatStatus.textContent = backendDownMessage(err); }
      },
    }, "flag as wrong"));
  }
  botMsg.append(inspector, actions);
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question || busy) return;
  busy = true;
  chatInput.value = "";

  messages.querySelector(".empty-state")?.remove();
  messages.append(el("div", { class: "msg user" },
    el("div", { class: "role" }, "You"),
    el("div", { class: "text" }, question)));

  const thinkText = el("div", { class: "text thinking" }, THINKING_PHRASES[0]);
  const botMsg = el("div", { class: "msg bot" },
    el("div", { class: "role" }, "SemCache"), thinkText);
  messages.append(botMsg);
  messages.scrollTop = messages.scrollHeight;
  chatStatus.textContent = "";
  const stopThinking = startThinking(thinkText);

  // The answer arrives as Server-Sent Events: a "meta" object first (hit/miss),
  // then "token" pieces while the model writes, then a final "done" object.
  let streamText = "";
  let streamStarted = false;
  let done = false;

  function appendToken(piece) {
    if (!streamStarted) {
      stopThinking();
      botMsg.innerHTML = "";
      botMsg.append(el("div", { class: "role" }, "SemCache"),
        el("div", { class: "text" }));
      streamStarted = true;
    }
    streamText += piece;
    botMsg.querySelector(".text").textContent = streamText;
    messages.scrollTop = messages.scrollHeight;
  }

  try {
    const res = await fetch(API + "/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const j = await res.json(); if (j.error) msg = j.error.message; } catch { /* ignore */ }
      throw new Error(msg);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done: streamDone } = await reader.read();
      if (streamDone) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const evt of events) {
        const line = evt.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const obj = JSON.parse(line.slice(5).trim());
        if (obj.type === "token") appendToken(obj.data);
        else if (obj.type === "done") {
          done = true;
          stopThinking();
          fillAnswer(botMsg, question, obj.data);
        } else if (obj.type === "error") throw new Error(obj.data);
      }
    }
    if (!done) {
      // Stream ended without a final object; show whatever we have.
      if (!streamStarted) throw new Error("No response from the backend.");
    }
  } catch (err) {
    stopThinking();
    if (streamStarted && streamText) {
      chatStatus.textContent = "Stream error: " + backendDownMessage(err);
    } else {
      botMsg.innerHTML = "";
      botMsg.append(el("div", { class: "role" }, "SemCache"),
        el("div", { class: "text error-text" }, "Error: " + backendDownMessage(err)));
    }
  } finally {
    busy = false;
    messages.scrollTop = messages.scrollHeight;
    chatInput.focus();
  }
});

// ---------- cache explorer ----------

const cacheList = document.getElementById("cache-list");

async function loadCache() {
  const search = document.getElementById("cache-search").value.trim();
  cacheList.innerHTML = "";
  try {
    const d = await api("/cache?limit=500&search=" + encodeURIComponent(search));
    if (!d.entries.length) {
      cacheList.append(el("div", { class: "empty-state" }, "No cached entries yet."));
      return;
    }
    for (const e of d.entries) {
      const created = new Date(e.created_at * 1000).toLocaleString();
      const expires = new Date(e.expires_at * 1000).toLocaleString();
      const card = el("div", { class: "entry-card" },
        el("div", { class: "q" }, e.question),
        el("div", { class: "a" }, e.answer),
        el("div", { class: "entry-meta" },
          el("span", {}, `id ${e.id}`),
          el("span", {}, `hits: ${e.hit_count}`),
          el("span", {}, `dim: ${e.dim}`),
          el("span", {}, `created: ${created}`),
          el("span", {}, `expires: ${expires}`),
          el("span", {}, `tokens: ${e.prompt_tokens}+${e.completion_tokens}`)),
        el("div", { class: "msg-actions" },
          el("button", {
            class: "link-btn warn",
            onclick: async () => {
              try { await api("/cache/" + e.id, { method: "DELETE" }); loadCache(); }
              catch (err) { alert(backendDownMessage(err)); }
            },
          }, "delete")));
      cacheList.append(card);
    }
  } catch (err) {
    cacheList.append(el("div", { class: "empty-state" }, backendDownMessage(err)));
  }
}

document.getElementById("cache-refresh").addEventListener("click", loadCache);
document.getElementById("cache-search").addEventListener("input", () => {
  clearTimeout(window._cacheSearchT);
  window._cacheSearchT = setTimeout(loadCache, 300);
});
document.getElementById("cache-clear").addEventListener("click", async () => {
  if (!confirm("Delete ALL cached entries? This cannot be undone.")) return;
  try {
    const d = await api("/cache/clear", { method: "POST", body: JSON.stringify({ confirm: true }) });
    alert(`Removed ${d.removed} entries.`);
    loadCache();
  } catch (err) { alert(backendDownMessage(err)); }
});

document.getElementById("cache-cleanup").addEventListener("click", async () => {
  try {
    const d = await api("/cache/cleanup", { method: "POST" });
    alert(`Removed ${d.removed} expired entries.`);
    loadCache();
  } catch (err) { alert(backendDownMessage(err)); }
});

document.getElementById("cache-export").addEventListener("click", async () => {
  try {
    const data = await api("/cache/export");
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = el("a", { href: url, download: "semcache-export.json" });
    document.body.append(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) { alert(backendDownMessage(err)); }
});

const importFile = document.getElementById("cache-import-file");
document.getElementById("cache-import").addEventListener("click", () => importFile.click());
importFile.addEventListener("change", async () => {
  const file = importFile.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const parsed = JSON.parse(text);
    const entries = Array.isArray(parsed) ? parsed : parsed.entries;
    const replace = confirm("Replace all current entries? OK = replace, Cancel = add to the existing ones.");
    const d = await api("/cache/import", {
      method: "POST",
      body: JSON.stringify({ entries, replace }),
    });
    alert(`Imported ${d.inserted} entries (${d.skipped} skipped).`);
    loadCache();
  } catch (err) { alert(backendDownMessage(err)); }
  finally { importFile.value = ""; }
});

// ---------- analytics ----------

async function loadAnalytics() {
  const cards = document.getElementById("analytics-cards");
  const note = document.getElementById("savings-note");
  cards.innerHTML = "";
  try {
    const d = await api("/analytics");
    if (!d.requests) {
      cards.append(el("div", { class: "empty-state" }, "No requests recorded yet. Use the Chat tab first."));
      note.textContent = "";
      drawHistogram([]);
      return;
    }
    const items = [
      ["Requests", d.requests],
      ["Hit rate", fmtPct(d.hit_rate)],
      ["Miss rate", fmtPct(d.miss_rate)],
      ["Exact hits", d.exact_hits],
      ["Semantic hits", d.semantic_hits],
      ["Mean latency", fmtMs(d.mean_total_ms)],
      ["P95 latency", fmtMs(d.p95_total_ms)],
      ["Mean embed time", fmtMs(d.mean_embedding_ms)],
      ["Mean retrieval time", fmtMs(d.mean_retrieval_ms)],
      ["Mean generation time", fmtMs(d.mean_generation_ms)],
      ["LLM calls avoided", d.llm_calls_avoided],
      ["False-positive reports", d.false_positive_reports],
      ["Cached entries", d.entries],
    ];
    for (const [label, value] of items) {
      cards.append(el("div", { class: "card" },
        el("div", { class: "label" }, label),
        el("div", { class: "value" }, String(value))));
    }
    note.textContent =
      `HYPOTHETICAL external-API savings: $${d.hypothetical_api_savings_usd.toFixed(4)} ` +
      `(at configured per-token prices; tokens saved: ${d.tokens_saved.prompt} prompt + ${d.tokens_saved.completion} completion). ` +
      `ACTUAL local cost: $${d.actual_local_cost_usd.toFixed(2)} - Ollama runs on this machine, so only electricity and hardware are used. The two numbers must not be mixed.`;
    drawHistogram(d.similarity_histogram || []);
  } catch (err) {
    cards.append(el("div", { class: "empty-state" }, backendDownMessage(err)));
  }
}

document.getElementById("analytics-refresh").addEventListener("click", loadAnalytics);

function drawHistogram(bins) {
  const canvas = document.getElementById("hist-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!bins.length) {
    ctx.fillStyle = "#9aa3b2";
    ctx.font = "14px Segoe UI";
    ctx.fillText("No similarity data yet.", 20, 30);
    return;
  }
  const pad = 40;
  const w = canvas.width - pad * 2;
  const h = canvas.height - pad * 1.5;
  const maxCount = Math.max(...bins.map((b) => b.count), 1);
  const barW = w / bins.length;
  ctx.strokeStyle = "#2e333c";
  ctx.beginPath(); ctx.moveTo(pad, pad / 2); ctx.lineTo(pad, pad / 2 + h); ctx.lineTo(pad + w, pad / 2 + h); ctx.stroke();
  bins.forEach((b, i) => {
    const bh = (b.count / maxCount) * h;
    ctx.fillStyle = b.from >= 0.9 ? "#4caf7d" : "#5aa7ff";
    ctx.fillRect(pad + i * barW + 2, pad / 2 + h - bh, barW - 4, bh);
    ctx.fillStyle = "#9aa3b2";
    ctx.font = "10px Segoe UI";
    ctx.fillText(b.from.toFixed(2), pad + i * barW, pad / 2 + h + 14);
    if (b.count) ctx.fillText(String(b.count), pad + i * barW + 4, pad / 2 + h - bh - 4);
  });
}

// ---------- benchmark ----------

const benchStatus = document.getElementById("bench-status");
const benchResults = document.getElementById("bench-results");
let benchPoller = null;

document.getElementById("bench-run").addEventListener("click", async () => {
  const which = document.getElementById("bench-select").value;
  try {
    await api("/benchmark/run", { method: "POST", body: JSON.stringify({ experiment: which }) });
    benchStatus.textContent = "Running...";
    document.getElementById("bench-run").disabled = true;
    clearInterval(benchPoller);
    benchPoller = setInterval(pollBench, 3000);
  } catch (err) { benchStatus.textContent = backendDownMessage(err); }
});

document.getElementById("bench-refresh").addEventListener("click", loadBenchResults);

async function pollBench() {
  try {
    const d = await api("/benchmark/results");
    if (d.state.running) {
      benchStatus.textContent = `Running experiment ${d.state.current}...`;
      return;
    }
    clearInterval(benchPoller);
    document.getElementById("bench-run").disabled = false;
    benchStatus.textContent = d.state.error ? "Failed: " + d.state.error : "Done.";
    renderBench(d.results);
  } catch (err) { benchStatus.textContent = backendDownMessage(err); }
}

async function loadBenchResults() {
  benchResults.innerHTML = "";
  try {
    const d = await api("/benchmark/results");
    if (d.state.running) {
      benchStatus.textContent = `Running experiment ${d.state.current}...`;
      document.getElementById("bench-run").disabled = true;
      clearInterval(benchPoller);
      benchPoller = setInterval(pollBench, 3000);
    } else {
      document.getElementById("bench-run").disabled = false;
      if (d.state.error) benchStatus.textContent = "Last run failed: " + d.state.error;
    }
    renderBench(d.results);
  } catch (err) {
    benchResults.append(el("div", { class: "empty-state" }, backendDownMessage(err)));
  }
}

function metaLine(meta) {
  if (!meta) return "";
  const hw = meta.hardware || {};
  return `Run ${meta.date} | ${meta.chat_model} + ${meta.embed_model} | tau=${meta.tau} delta=${meta.delta} TTL=${meta.ttl_hours}h | dataset v${meta.dataset_version} (${meta.dataset_pairs} pairs) | CPU ${hw.cpu_count} cores, RAM ${hw.ram_gb ?? "?"} GB`;
}

function tableFrom(headers, rows) {
  const t = el("table");
  t.append(el("thead", {}, el("tr", {}, ...headers.map((h) => el("th", {}, h)))));
  const tb = el("tbody");
  for (const r of rows) tb.append(el("tr", {}, ...r.map((c) => el("td", {}, String(c ?? "-")))));
  t.append(tb);
  return t;
}

function renderBench(results) {
  benchResults.innerHTML = "";
  if (!Object.keys(results || {}).length) {
    benchResults.append(el("div", { class: "empty-state" },
      "No benchmark results saved yet. Press Run benchmark."));
    return;
  }
  const order = ["exp1_false_positives", "exp2_threshold_sweep", "exp3_margin_check", "exp4_ttl", "exp5_baseline"];
  for (const key of order) {
    const r = results[key];
    if (!r) continue;
    const block = el("div", { class: "bench-block" });

    if (key === "exp1_false_positives") {
      block.append(el("h3", {}, "1. False-positive test"));
      block.append(el("p", { class: "note" }, metaLine(r.meta)));
      const c = r.counts;
      block.append(el("p", {}, `True positives: ${c.true_positives} | False negatives: ${c.false_negatives} | False positives: ${c.false_positives} | True negatives: ${c.true_negatives}`));
      if (r.false_positives.length) {
        block.append(tableFrom(["a", "b", "label", "similarity", "margin"],
          r.false_positives.map((p) => [p.a, p.b, p.label, p.similarity, p.margin])));
      }
      if (r.false_negatives.length) {
        block.append(el("h4", {}, "Missed pairs (should have hit)"));
        block.append(tableFrom(["a", "b", "label", "similarity", "margin"],
          r.false_negatives.map((p) => [p.a, p.b, p.label, p.similarity, p.margin])));
      }
    }

    if (key === "exp2_threshold_sweep") {
      block.append(el("h3", {}, "2. Threshold sweep (tau 0.80 - 0.95)"));
      block.append(el("p", { class: "note" }, metaLine(r.meta)));
      block.append(tableFrom(["tau", "TP", "FP", "FN", "TN", "precision", "recall", "F1", "FP rate"],
        r.sweep.map((s) => [s.tau, s.tp, s.fp, s.fn, s.tn, s.precision, s.recall, s.f1, s.false_positive_rate])));
      const canvas = el("canvas", { width: "800", height: "240" });
      block.append(canvas);
      requestAnimationFrame(() => drawSweep(canvas, r.sweep));
    }

    if (key === "exp3_margin_check") {
      block.append(el("h3", {}, "3. Margin check (delta = 0 vs chosen)"));
      block.append(el("p", { class: "note" }, metaLine(r.meta)));
      block.append(tableFrom(["mode", "TP", "FP", "FN", "precision", "recall"],
        [["delta=0", r.delta_zero.tp, r.delta_zero.fp, r.delta_zero.fn, r.delta_zero.precision, r.delta_zero.recall],
         [`delta=${r.delta_chosen.delta}`, r.delta_chosen.tp, r.delta_chosen.fp, r.delta_chosen.fn, r.delta_chosen.precision, r.delta_chosen.recall]]));
      if (r.delta_zero.false_positive_examples.length) {
        block.append(el("h4", {}, "False positives with delta=0"));
        block.append(tableFrom(["a", "b", "similarity", "margin"],
          r.delta_zero.false_positive_examples.map((p) => [p.a, p.b, p.similarity, p.margin])));
      }
    }

    if (key === "exp4_ttl") {
      block.append(el("h3", {}, "4. TTL expiry test"));
      block.append(el("p", { class: "note" }, metaLine(r.meta)));
      block.append(el("p", {},
        `Hit before expiry: ${r.hit_before_expiry} | Hit after expiry: ${r.hit_after_expiry} (${r.reason_after}) | PASSED: ${r.passed}`));
    }

    if (key === "exp5_baseline") {
      block.append(el("h3", {}, "5. Baseline comparison (same 10 questions, same model/hardware)"));
      block.append(el("p", { class: "note" }, metaLine(r.meta)));
      const rows = [];
      for (const mode of ["no_cache", "exact_only", "semantic_cache"]) {
        const m = r.results[mode];
        if (!m) continue;
        rows.push([mode, m.llm_calls, m.hits, m.mean_total_ms + " ms", m.p95_total_ms + " ms"]);
      }
      block.append(tableFrom(["mode", "LLM calls", "cache hits", "mean latency", "P95 latency"], rows));
    }

    benchResults.append(block);
  }
}

function drawSweep(canvas, sweep) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const pad = 45, w = canvas.width - pad * 2, h = canvas.height - pad * 1.6;
  ctx.strokeStyle = "#2e333c";
  ctx.beginPath(); ctx.moveTo(pad, 20); ctx.lineTo(pad, 20 + h); ctx.lineTo(pad + w, 20 + h); ctx.stroke();
  const series = [
    { key: "precision", color: "#5aa7ff" },
    { key: "recall", color: "#4caf7d" },
    { key: "false_positive_rate", color: "#d9534f" },
    { key: "f1", color: "#d98e4a" },
  ];
  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.beginPath();
    sweep.forEach((p, i) => {
      const v = p[s.key];
      if (v == null) return;
      const x = pad + (i / (sweep.length - 1)) * w;
      const y = 20 + h - v * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.font = "11px Segoe UI";
  series.forEach((s, i) => {
    ctx.fillStyle = s.color;
    ctx.fillText(s.key, pad + 10 + i * 150, 14);
  });
  ctx.fillStyle = "#9aa3b2";
  sweep.forEach((p, i) => {
    if (i % 2 === 0) {
      const x = pad + (i / (sweep.length - 1)) * w;
      ctx.fillText(p.tau.toFixed(2), x - 10, 20 + h + 16);
    }
  });
}

// ---------- settings ----------

const healthBox = document.getElementById("health-box");

async function loadSettings() {
  healthBox.innerHTML = "";
  try {
    const [h, m, cfg] = await Promise.all([
      api("/health"), api("/models/status"), api("/config"),
    ]);
    healthBox.append(
      el("h3", {}, "Health"),
      el("p", {}, "Backend: ", el("span", { class: "ok" }, "online")),
      el("p", {}, "Ollama: ", el("span", { class: m.ollama_running ? "ok" : "bad" },
        m.ollama_running ? "running" : "NOT running")),
      el("p", {}, "Chat model: " + (cfg.chat_model || "") + " | Embed model: " + (cfg.embed_model || "")));

    const modelList = el("div");
    for (const name of [cfg.chat_model, cfg.embed_model]) {
      const installed = m.installed.some((t) => t === name || t.split(":")[0] === name.split(":")[0]);
      const line = el("p", {}, `${name}: `);
      if (installed) {
        line.append(el("span", { class: "ok" }, "installed"));
      } else {
        line.append(el("span", { class: "bad" }, "missing "));
        const prog = el("span", { class: "progress-line" });
        const btn = el("button", { class: "btn", onclick: () => pullModel(name, prog, btn) }, "Pull");
        line.append(btn, prog);
      }
      modelList.append(line);
    }
    healthBox.append(modelList);

    document.getElementById("cfg-tau").value = cfg.tau;
    document.getElementById("cfg-delta").value = cfg.delta;
    document.getElementById("cfg-ttl").value = cfg.ttl_hours;
    loadNamespaces();
  } catch (err) {
    healthBox.append(el("p", { class: "bad" }, backendDownMessage(err)));
  }
}

async function loadNamespaces() {
  const box = document.getElementById("ns-list");
  box.innerHTML = "";
  try {
    const d = await api("/config/namespaces");
    if (!d.namespaces.length) {
      box.append(el("div", { class: "note" }, "No overrides set; every namespace uses the global values above."));
      return;
    }
    for (const n of d.namespaces) {
      box.append(el("div", { class: "entry-meta" },
        el("span", {}, n.namespace),
        el("span", {}, `tau: ${n.tau}`),
        el("span", {}, `delta: ${n.delta}`),
        el("span", {}, `ttl: ${n.ttl_hours}h`)));
    }
  } catch (err) {
    box.append(el("div", { class: "note" }, backendDownMessage(err)));
  }
}

async function pullModel(model, prog, btn) {
  btn.disabled = true;
  prog.textContent = " starting...";
  try {
    const res = await fetch(API + "/models/pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const chunk = JSON.parse(line);
          if (chunk.status === "error") prog.textContent = " error: " + chunk.error;
          else if (chunk.status === "success") prog.textContent = " done";
          else prog.textContent = " " + (chunk.status || "") + (chunk.completed && chunk.total
            ? ` ${Math.round((chunk.completed / chunk.total) * 100)}%` : "");
        } catch { /* partial line */ }
      }
    }
    loadSettings();
  } catch (err) {
    prog.textContent = " failed: " + err.message;
    btn.disabled = false;
  }
}

document.getElementById("cfg-save").addEventListener("click", async () => {
  const status = document.getElementById("cfg-status");
  try {
    const d = await api("/config", {
      method: "POST",
      body: JSON.stringify({
        tau: parseFloat(document.getElementById("cfg-tau").value),
        delta: parseFloat(document.getElementById("cfg-delta").value),
        ttl_hours: parseFloat(document.getElementById("cfg-ttl").value),
      }),
    });
    status.textContent = `Saved: tau=${d.tau} delta=${d.delta} ttl=${d.ttl_hours}h`;
  } catch (err) { status.textContent = backendDownMessage(err); }
});

document.getElementById("ns-save").addEventListener("click", async () => {
  const status = document.getElementById("ns-status");
  const namespace = document.getElementById("ns-name").value.trim();
  if (!namespace) { status.textContent = "Enter a namespace id first."; return; }
  const body = { namespace };
  const tau = document.getElementById("ns-tau").value.trim();
  const delta = document.getElementById("ns-delta").value.trim();
  const ttl = document.getElementById("ns-ttl").value.trim();
  if (tau) body.tau = parseFloat(tau);
  if (delta) body.delta = parseFloat(delta);
  if (ttl) body.ttl_hours = parseFloat(ttl);
  try {
    const d = await api("/config", { method: "POST", body: JSON.stringify(body) });
    status.textContent = `Override saved for ${d.namespace}: tau=${d.tau} delta=${d.delta} ttl=${d.ttl_hours}h`;
    loadNamespaces();
  } catch (err) { status.textContent = backendDownMessage(err); }
});

// first load: show health status right away so a down backend is obvious
loadSettings();
