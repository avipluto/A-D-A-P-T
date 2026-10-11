// ADAPT frontend: plain ES module, no build step. Talks to the API on the same origin.
// All text from the server (extracted values, web quotes, titles) is inserted with
// textContent, never innerHTML, so page content can't inject markup or scripts.

const $ = (id) => document.getElementById(id);
const PRIORITY_WEIGHT = { high: 3, medium: 2, low: 1 };
const WEIGHT_PRIORITY = { 3: "high", 2: "medium", 1: "low" };

const state = { frame: null, parentId: null, decision: null };

// ---------- small DOM helpers ----------
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) node.append(c);
  return node;
}
function select(options, value, onchange) {
  const s = el("select", { onchange: (e) => onchange(e.target.value) });
  for (const [v, label] of options) {
    const o = el("option", { value: v, text: label });
    if (String(v) === String(value)) o.selected = true;
    s.append(o);
  }
  return s;
}
function safeUrl(url) {
  try { const u = new URL(url); return u.protocol === "https:" || u.protocol === "http:" ? u.href : null; }
  catch { return null; }
}
function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = !msg; }
function fmt(v) { return typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2)) : String(v); }

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `request failed (${res.status})`);
  }
  return data;
}
async function busy(button, fn) {
  button.disabled = true; showError("");
  try { await fn(); } catch (e) { showError(e.message); } finally { button.disabled = false; }
}

// ---------- frame helpers ----------
function attributes() {
  const names = new Set();
  for (const o of state.frame.options) for (const k of Object.keys(o.attributes)) names.add(k);
  for (const c of [...state.frame.criteria, ...state.frame.constraints]) names.add(c.attribute);
  return [...names];
}
function parseValue(text) {
  const t = text.trim();
  if (t === "") return null;
  if (/^(yes|true)$/i.test(t)) return true;
  if (/^(no|false)$/i.test(t)) return false;
  const n = Number(t.replace(/[,₹$\s]/g, ""));
  return Number.isFinite(n) ? n : t;
}
function nextId(i) { return i < 26 ? String.fromCharCode(97 + i) : `o${i}`; }
function snake(s) { return s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, ""); }

// ---------- render: frame editor ----------
function renderFrame() {
  const f = state.frame;
  $("frame-card").hidden = false;
  const inferred = JSON.stringify(f).includes('"inferred"');
  $("confirm-banner").hidden = !inferred;

  const attrs = attributes();
  const table = $("options-table");
  table.replaceChildren();
  const head = el("tr", {}, el("th", { text: "Option" }));
  for (const a of attrs) {
    head.append(el("th", {},
      el("input", { value: a, "aria-label": "attribute name", onchange: (e) => renameAttribute(a, e.target.value) }),
      el("button", { class: "link", text: "remove", onclick: () => removeAttribute(a) })));
  }
  head.append(el("th"));
  table.append(head);

  f.options.forEach((o, i) => {
    const row = el("tr", {}, el("td", {},
      el("input", { value: o.label, "aria-label": "option name", onchange: (e) => { o.label = e.target.value; o.provenance = "stated"; } })));
    for (const a of attrs) {
      const v = o.attributes[a];
      const input = el("input", {
        value: v ? String(v.value) : "", placeholder: "unknown", "aria-label": `${o.label} ${a}`,
        onchange: (e) => {
          const x = parseValue(e.target.value);
          if (x === null) delete o.attributes[a];
          else o.attributes[a] = { value: x, unit: v ? v.unit : null, provenance: "stated", source: null };
          renderFrame();
        },
      });
      const cell = el("div", { class: "cell" }, input);
      if (v) cell.append(el("span", { class: `badge ${v.provenance}`, text: v.provenance + (v.unit ? ` · ${v.unit}` : "") }));
      row.append(el("td", {}, cell));
    }
    row.append(el("td", {}, el("button", { class: "link", text: "remove", onclick: () => { f.options.splice(i, 1); renderFrame(); } })));
    table.append(row);
  });

  const attrChoices = attrs.map((a) => [a, a]);
  const crit = $("criteria");
  crit.replaceChildren();
  f.criteria.forEach((c, i) => {
    crit.append(el("div", { class: "line" },
      select(attrChoices, c.attribute, (v) => { c.attribute = v; c.provenance = "stated"; }),
      select([["maximize", "more is better"], ["minimize", "less is better"]], c.direction, (v) => { c.direction = v; c.provenance = "stated"; }),
      select([["", "importance unknown"], ["high", "matters most"], ["medium", "fairly important"], ["low", "matters a little"]],
        WEIGHT_PRIORITY[c.weight] ?? "", (v) => { c.weight = v ? PRIORITY_WEIGHT[v] : null; c.provenance = "stated"; }),
      el("button", { class: "link", text: "remove", onclick: () => { f.criteria.splice(i, 1); renderFrame(); } })));
  });

  const cons = $("constraints");
  cons.replaceChildren();
  f.constraints.forEach((c, i) => {
    cons.append(el("div", { class: "line" },
      select(attrChoices, c.attribute, (v) => { c.attribute = v; c.provenance = "stated"; }),
      select([["<=", "at most"], [">=", "at least"], ["==", "must be"], ["!=", "must not be"]], c.operator,
        (v) => { c.operator = v; c.provenance = "stated"; }),
      el("input", { value: String(c.value), size: 8, "aria-label": "limit value",
        onchange: (e) => { const x = parseValue(e.target.value); if (x !== null) { c.value = x; c.provenance = "stated"; } } }),
      el("button", { class: "link", text: "remove", onclick: () => { f.constraints.splice(i, 1); renderFrame(); } })));
  });
}

function renameAttribute(from, toRaw) {
  const to = snake(toRaw);
  if (!to || to === from) return;
  for (const o of state.frame.options) if (from in o.attributes) { o.attributes[to] = o.attributes[from]; delete o.attributes[from]; }
  for (const c of [...state.frame.criteria, ...state.frame.constraints]) if (c.attribute === from) c.attribute = to;
  renderFrame();
}
function removeAttribute(a) {
  for (const o of state.frame.options) delete o.attributes[a];
  state.frame.criteria = state.frame.criteria.filter((c) => c.attribute !== a);
  state.frame.constraints = state.frame.constraints.filter((c) => c.attribute !== a);
  renderFrame();
}

// ---------- render: result ----------
function renderResult(resp) {
  state.decision = resp;
  const r = resp.result;
  const labels = Object.fromEntries(resp.frame.options.map((o) => [o.id, o.label]));
  $("result-card").hidden = false;

  const head = $("headline");
  head.className = `headline ${r.outcome}`;
  head.textContent = r.outcome === "recommend" ? `Recommended: ${labels[r.recommended]}`
    : r.outcome === "clarify" ? "ADAPT needs more information" : "No option meets all your limits";
  $("reason").textContent = r.reason;

  const fragile = $("fragile");
  fragile.hidden = !(r.fragile || r.near_tie);
  fragile.textContent = r.fragile ? "Close call: a small change in what matters to you would change this recommendation (see below)."
    : "Near tie: the top options score almost the same.";

  $("questions").replaceChildren(...r.questions.map((q) => el("li", { text: q.prompt })));
  if (r.questions.length) $("questions").append(el("li", { class: "hint", text: "Add the answers in the table above, then press Decide again." }));
  $("notes").replaceChildren(...[...r.notes, ...r.assumptions.map((a) => `Assumed value: ${a}`)].map((n) => el("li", { text: n })));

  $("ranking-wrap").hidden = !r.ranking.length;
  $("ranking").replaceChildren(...r.ranking.map((x) => {
    const mid = (x.score_low + x.score_high) / 2;
    const why = Object.entries(x.contributions).sort((a, b) => b[1][1] - a[1][1])
      .map(([k, [lo, hi]]) => `${k} ${lo === hi ? lo.toFixed(2) : `${lo.toFixed(2)}–${hi.toFixed(2)}`}`).join(" · ");
    return el("div", { class: `rank ${x.option_id === r.recommended ? "top" : ""}` },
      el("span", { class: "name", text: x.label }),
      el("div", { class: "track", role: "img", "aria-label": `score ${mid.toFixed(2)}` },
        el("div", { class: "range", style: `left:${x.score_low * 100}%;width:${(x.score_high - x.score_low) * 100}%` }),
        el("div", { class: "fill", style: `width:${x.score_low * 100}%` })),
      el("span", { class: "score", text: x.score_low === x.score_high ? x.score_low.toFixed(2) : `${x.score_low.toFixed(2)}–${x.score_high.toFixed(2)}` }),
      el("span", { class: "why", text: why }));
  }));

  $("excluded-wrap").hidden = !r.excluded.length;
  $("excluded").replaceChildren(...r.excluded.map((e) => el("li", { text: `${e.label}: breaks ${e.failed.join(", ")}` })));

  const pending = resp.lookups.filter((p) => p.status !== "apply");
  $("lookups-wrap").hidden = !pending.length;
  $("lookups").replaceChildren(...pending.map((p) => {
    const box = el("div", { class: "proposal" },
      el("div", {}, el("strong", { text: `${labels[p.option_id]} · ${p.attribute}: ` }),
        el("span", { text: p.value === null ? "not found" : `${fmt(p.value)} (needs your OK)` }),
        p.value === null ? null : el("button", { class: "link", text: "use this value", onclick: () => useValue(p) })),
      el("div", { class: "hint", text: p.reasons.join("; ") }));
    for (const s of p.sources.slice(0, 3)) {
      const href = safeUrl(s.url);
      box.append(el("div", { class: "src" },
        href ? el("a", { href, target: "_blank", rel: "noopener noreferrer", text: s.domain }) : el("span", { text: s.domain }),
        el("span", { text: ` · ${s.tier} · ${s.published_at ? s.published_at.slice(0, 10) : "undated"}` })),
        el("blockquote", { text: s.quote }));
    }
    return box;
  }));

  $("citations-wrap").hidden = !r.citations.length;
  $("citations").replaceChildren(...r.citations.map((c) => {
    const href = safeUrl(c.url);
    return el("li", {},
      el("span", { text: `${labels[c.option_id]} · ${c.attribute} = ${fmt(c.value)} — ` }),
      href ? el("a", { href, target: "_blank", rel: "noopener noreferrer", text: c.title || new URL(href).hostname }) : el("span", { text: c.url }),
      el("span", { class: "hint", text: ` (published ${c.published_at || "date unknown"}, retrieved ${c.retrieved_at})` }));
  }));

  const choice = $("fb-choice");
  choice.replaceChildren(el("option", { value: "", text: "not chosen yet" }),
    ...resp.frame.options.map((o) => el("option", { value: o.id, text: o.label })));
  renderStars();
  $("fb-status").textContent = "";
  $("result-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

function useValue(p) {
  const o = state.frame.options.find((x) => x.id === p.option_id);
  if (o) o.attributes[p.attribute] = { value: p.value, unit: null, provenance: "stated", source: null };
  renderFrame();
  showError("");
  $("frame-card").scrollIntoView({ behavior: "smooth" });
}

// ---------- feedback ----------
function renderStars() {
  for (const box of document.querySelectorAll(".stars")) {
    box.replaceChildren(...[1, 2, 3, 4, 5].map((n) => el("button", {
      text: "★", "aria-label": `${n} of 5`, onclick: () => sendFeedback(box.dataset.kind, n, box),
    })));
  }
}
async function sendFeedback(kind, rating, box) {
  if (!state.decision) return;
  try {
    await api("/feedback", {
      decision_id: state.decision.decision_id, kind, rating,
      chosen_option_id: $("fb-choice").value || null,
    });
    [...box.children].forEach((b, i) => b.classList.toggle("on", i < rating));
    $("fb-status").textContent = `Saved (${kind}: ${rating}/5), pending review.`;
  } catch (e) { showError(e.message); }
}

// ---------- actions ----------
$("extract").addEventListener("click", (e) => busy(e.target, async () => {
  const text = $("situation").value.trim() || $("situation").placeholder;
  $("situation").value = text;
  const resp = await api("/decision", { situation: text });
  state.frame = resp.frame;
  state.parentId = resp.decision_id;
  $("result-card").hidden = true;
  renderFrame();
  $("frame-card").scrollIntoView({ behavior: "smooth" });
}));

$("blank").addEventListener("click", () => {
  state.frame = {
    situation: "decision", options: [
      { id: "a", label: "Option A", attributes: {}, provenance: "stated" },
      { id: "b", label: "Option B", attributes: {}, provenance: "stated" }],
    criteria: [], constraints: [], goals: [], context: {}, unknowns: [],
  };
  state.parentId = null;
  $("result-card").hidden = true;
  renderFrame();
});

$("add-option").addEventListener("click", () => {
  const f = state.frame;
  const used = new Set(f.options.map((o) => o.id));
  let i = 0; while (used.has(nextId(i))) i++;
  f.options.push({ id: nextId(i), label: `Option ${String.fromCharCode(65 + i)}`, attributes: {}, provenance: "stated" });
  renderFrame();
});
$("add-attribute").addEventListener("click", () => {
  const name = snake(prompt("Attribute name (e.g. price, battery_hours):") || "");
  if (!name || attributes().includes(name)) return;
  // the new column appears through its criterion; values start empty ("unknown")
  state.frame.criteria.push({ attribute: name, direction: "maximize", weight: null, provenance: "stated" });
  renderFrame();
});
$("add-criterion").addEventListener("click", () => {
  const a = attributes()[0]; if (!a) return showError("Add an attribute first.");
  state.frame.criteria.push({ attribute: a, direction: "maximize", weight: null, provenance: "stated" });
  renderFrame();
});
$("add-constraint").addEventListener("click", () => {
  const a = attributes()[0]; if (!a) return showError("Add an attribute first.");
  state.frame.constraints.push({ attribute: a, operator: "<=", value: 0, provenance: "stated" });
  renderFrame();
});

$("decide").addEventListener("click", (e) => busy(e.target, async () => {
  const resp = await api("/decision", {
    frame: state.frame, confirmed: true, parent_id: state.parentId, lookup: $("lookup").checked,
  });
  state.frame = resp.frame;  // includes any values filled from the web
  renderFrame();
  renderResult(resp);
}));

$("fb-correction").addEventListener("click", (e) => busy(e.target, async () => {
  if (!state.decision) return;
  const notes = $("fb-notes").value.trim();
  if (!notes) return showError("Write what was wrong first.");
  await api("/feedback", { decision_id: state.decision.decision_id, kind: "correction", notes,
    chosen_option_id: $("fb-choice").value || null });
  $("fb-notes").value = "";
  $("fb-status").textContent = "Correction saved, pending review.";
}));

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    for (const t of document.querySelectorAll(".tab")) {
      const on = t === tab; t.classList.toggle("active", on); t.setAttribute("aria-selected", String(on));
      $(`tab-${t.dataset.tab}`).hidden = !on;
    }
  });
}

api("/model-info").then((m) => {
  const v = m.versions;
  $("versions").textContent = `engine ${v.engine} · scorer ${v.scorer} · reader ${v.extractor}` +
    (v.retrieval ? ` · search ${v.retrieval}` : " · search off") + (m.fallbacks.length ? " · fallback in use" : "");
  if (m.fallbacks.length) $("versions").title = m.fallbacks.join("\n");
}).catch(() => { $("versions").textContent = "API not reachable"; });
