// Research Paper Assistant - front end.
//
// No framework and no build step on purpose: it's one file you can read top to
// bottom. The server streams Server-Sent Events from /api/ask:
//   sources -> the passages retrieval found (shown in the margin right away)
//   token   -> a piece of the answer as the model writes it
//   repair  -> the model forgot source numbers; a second pass is adding them
//   final   -> the checked answer (citations validated, refusal rules applied)
//   error   -> something went wrong, with a message we can show as-is

const $ = (sel, root = document) => root.querySelector(sel);

const els = {
  model: $("#model"),
  modelStatus: $("#model-status"),
  refresh: $("#refresh-models"),
  mode: $("#mode"),
  paper: $("#paper"),
  k: $("#k"),
  thread: $("#thread"),
  intro: $("#intro"),
  examples: $("#examples"),
  form: $("#composer"),
  question: $("#question"),
  ask: $("#ask"),
  tagline: $("#tagline"),
  dialog: $("#benchmark"),
  dialogBody: $("#benchmark-body"),
};

const state = {
  providers: [],
  papers: [],
  busy: false,
};

// Remember the reader's picks between visits. Storage can be blocked (private
// windows, strict settings), so every access is wrapped.
const saved = (() => {
  try { return JSON.parse(localStorage.getItem("rpa.settings") || "{}"); } catch { return {}; }
})();
function save(patch) {
  Object.assign(saved, patch);
  try { localStorage.setItem("rpa.settings", JSON.stringify(saved)); } catch { /* not important */ }
}

// Questions that show off different parts of the system. `paper` (a title
// fragment) turns on the single-paper filter for that example.
const EXAMPLES = [
  { q: "Which datasets are used for Armenian named entity recognition?" },
  { q: "What metric correlates best with human judgements for table-to-text generation?" },
  { q: "How many participants took part in the user study?", paper: "Learning Autocomplete Systems" },
  { q: "What languages are used for zero-shot neural machine translation?" },
  { q: "What is the capital of France?", note: "off-topic, should decline" },
];

const REFUSAL_WHY = {
  low_relevance: "None of the passages it found looked related to the question, so no model was asked.",
  model_declined: "The model read the passages in the margin and didn't find an answer in them.",
  no_citations: "The model answered without pointing to any source, so the answer was held back.",
};

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} returned ${res.status}`);
  return res.json();
}

async function loadInfo() {
  const info = await getJSON("/api/info");
  state.papers = info.papers;
  els.tagline.textContent = `Cited answers from ${info.papers.length} NLP research papers`;

  els.paper.innerHTML = `<option value="">All ${info.papers.length} papers</option>` +
    info.papers.map(p => `<option value="${esc(p.id)}">${esc(p.title)}</option>`).join("");
  if (saved.paper && info.papers.some(p => p.id === saved.paper)) els.paper.value = saved.paper;

  const mode = saved.mode || info.default_mode;
  els.mode.innerHTML = info.modes.map(m => `
    <input type="radio" name="mode" id="mode-${m.id}" value="${m.id}" ${m.id === mode ? "checked" : ""}>
    <label for="mode-${m.id}">${esc(m.label)}</label>`).join("");

  els.k.value = saved.k || info.top_k;
}

async function loadModels() {
  els.refresh.classList.add("spinning");
  try {
    const data = await getJSON("/api/models");
    state.providers = data.providers;

    els.model.innerHTML = data.providers.map(p => {
      if (!p.available) {
        return `<optgroup label="${esc(p.label)}"><option disabled>${esc(p.note)}</option></optgroup>`;
      }
      const opts = p.models.map(m => {
        const value = `${p.id}::${m.name}`;
        const extra = p.id === "ollama" ? [m.params, m.size_gb ? `${m.size_gb} GB` : ""].filter(Boolean).join(", ") : "";
        const label = p.id === "extractive" ? "Quote the best sentences" : m.name;
        return `<option value="${esc(value)}">${esc(label)}${extra ? `  (${esc(extra)})` : ""}</option>`;
      }).join("");
      return `<optgroup label="${esc(p.label)}">${opts}</optgroup>`;
    }).join("");
    els.model.disabled = false;

    // Keep the reader's last pick if it's still installed, otherwise use the server's default.
    const values = [...els.model.options].filter(o => !o.disabled).map(o => o.value);
    const fallback = `${data.default.provider}::${data.default.model}`;
    els.model.value = values.includes(saved.model) ? saved.model : fallback;
  } catch (err) {
    els.model.innerHTML = `<option>Couldn't load models</option>`;
    setStatus(`Couldn't ask the server for models: ${err.message}`, "warn");
    return;
  } finally {
    els.refresh.classList.remove("spinning");
  }
  updateStatus();
}

function selected() {
  const [provider, ...rest] = els.model.value.split("::");
  return { provider, model: rest.join("::") };
}

function setStatus(html, tone = "") {
  els.modelStatus.innerHTML = html;
  els.modelStatus.dataset.tone = tone;
}

function updateStatus() {
  const { provider } = selected();
  const ollama = state.providers.find(p => p.id === "ollama");
  if (provider === "ollama") {
    const n = ollama.models.length;
    setStatus(`Running locally through Ollama, ${n} model${n === 1 ? "" : "s"} installed`, "ok");
  } else if (provider === "extractive") {
    setStatus("No model: answers are sentences quoted from the passages", "");
  } else {
    const p = state.providers.find(x => x.id === provider);
    setStatus(`Answers come from ${esc(p ? p.label.replace(/^API \((.*)\)$/, "$1") : provider)}`, "ok");
  }
  // If Ollama isn't usable, say how to fix it even while another model is selected.
  if (ollama && !ollama.available && provider !== "ollama") {
    setStatus(`${els.modelStatus.innerHTML}. ${formatNote(ollama.note)}`, "warn");
  }
}

// "Start it with: ollama serve" -> show the command in bold so it's easy to spot.
function formatNote(note) {
  return esc(note).replace(/: (ollama .+)$/, ": <code>$1</code>");
}

function renderExamples() {
  els.examples.innerHTML = EXAMPLES.map((ex, i) => {
    const paper = ex.paper ? state.papers.find(p => p.title.startsWith(ex.paper)) : null;
    const scope = ex.note || (paper ? "one paper" : "all papers");
    return `<li><button type="button" data-i="${i}"><span class="ex-text">${esc(ex.q)}</span>` +
           `<span class="scope">${esc(scope)}</span></button></li>`;
  }).join("");
  els.examples.addEventListener("click", e => {
    const btn = e.target.closest("button[data-i]");
    if (!btn) return;
    const ex = EXAMPLES[btn.dataset.i];
    const paper = ex.paper ? state.papers.find(p => p.title.startsWith(ex.paper)) : null;
    els.paper.value = paper ? paper.id : "";
    save({ paper: els.paper.value });
    els.question.value = ex.q;
    ask();
  });
}

// ---------------------------------------------------------------------------
// Asking
// ---------------------------------------------------------------------------

async function ask() {
  const question = els.question.value.trim();
  if (!question || state.busy) return;
  const { provider, model } = selected();
  if (!provider) return;

  state.busy = true;
  els.ask.disabled = true;
  els.question.value = "";
  autosize();

  const turn = newTurn(question);
  const body = {
    question, provider, model,
    mode: $("input[name=mode]:checked", els.mode)?.value || "hybrid_rerank",
    k: Number(els.k.value) || 5,
    paper_id: els.paper.value || null,
  };

  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (!res.ok) {
      let detail = `The server returned ${res.status}.`;
      try {
        const j = await res.json();
        detail = typeof j.detail === "string" ? j.detail : (j.detail?.[0]?.msg || detail);
      } catch { /* keep the generic message */ }
      throw new Error(detail);
    }
    await readEvents(res, event => turn.handle(event, model));
  } catch (err) {
    turn.handle({ type: "error", message: err.message });
  } finally {
    state.busy = false;
    els.ask.disabled = false;
    els.question.focus();
  }
}

// Parse a text/event-stream body: events are separated by a blank line,
// each one is a "data: {json}" line.
async function readEvents(res, onEvent) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      const data = block.split("\n").filter(l => l.startsWith("data: ")).map(l => l.slice(6)).join("");
      if (data) onEvent(JSON.parse(data));
    }
  }
}

function newTurn(question) {
  els.intro?.remove();
  const node = $("#turn-template").content.firstElementChild.cloneNode(true);
  $(".question", node).textContent = question;
  els.thread.append(node);
  node.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });

  const answer = $(".answer", node);
  const status = $(".answer-status", node);
  const bodyEl = $(".answer-body", node);
  const meta = $(".answer-meta", node);
  const margin = $(".margin", node);
  const list = $(".sources", node);
  const toggle = $(".margin-toggle", node);

  let raw = "";
  let chunks = [];
  let retrievalS = 0;

  toggle.addEventListener("click", () => {
    const open = margin.classList.toggle("collapsed") === false;
    toggle.setAttribute("aria-expanded", String(open));
  });
  wireFootnotes(node);

  return {
    handle(event, modelName) {
      switch (event.type) {
        case "sources":
          chunks = event.chunks;
          retrievalS = event.retrieval_s;
          renderSources(list, chunks);
          toggle.hidden = false;
          margin.classList.add("collapsed");
          toggle.textContent = `${chunks.length} passages`;
          answer.dataset.state = "thinking";
          status.textContent = selected().provider === "extractive"
            ? "Picking the best sentences" : `Reading ${chunks.length} passages with ${modelName}`;
          break;
        case "token":
          raw += event.text;
          answer.dataset.state = "streaming";
          bodyEl.innerHTML = renderAnswer(raw, chunks.length);
          break;
        case "repair":
          answer.dataset.state = "thinking";
          status.textContent = "The answer had no source numbers. Asking the model to add them";
          break;
        case "final":
          finish(event.answer);
          break;
        case "error":
          answer.dataset.state = "error";
          bodyEl.innerHTML = `<p>${esc(event.message)}</p>`;
          break;
      }
    },
  };

  function finish(ans) {
    if (ans.refused) {
      answer.dataset.state = "refused";
      bodyEl.innerHTML = `<p>${esc(ans.text)}</p><p class="refusal-why">${esc(REFUSAL_WHY[ans.reason] || "")}</p>`;
      markCited(list, [], "", chunks, question);     // nothing was used, so nothing looks cited
    } else {
      answer.dataset.state = "done";
      bodyEl.innerHTML = renderAnswer(ans.raw, chunks.length);
      markCited(list, ans.cited, ans.raw, chunks, question);
    }
    const cited = new Set(ans.cited);
    toggle.textContent = ans.refused
      ? `${chunks.length} passages checked`
      : `${chunks.length} passages, ${cited.size} cited`;
    if (!ans.refused && window.matchMedia("(min-width: 1061px)").matches === false) {
      margin.classList.remove("collapsed");
      toggle.setAttribute("aria-expanded", "true");
    }

    const model = `<span class="meta-model">${esc(ans.model)}</span>`;
    const secs = `${ans.generation_s.toFixed(1)} s`;
    const who = ans.reason === "low_relevance" ? "No model was asked"
      : ans.provider === "extractive" ? "Quoted from the passages"
      : ans.refused ? `${model} read the passages in ${secs}`
      : `${model} wrote this in ${secs}`;
    const parts = [
      `<span>${who}</span>`,
      `<span>Found ${chunks.length} passages in ${retrievalS.toFixed(1)} s</span>`,
    ];
    if (ans.cite_repair) parts.push(`<span>Source numbers added on a second pass</span>`);
    if (!ans.refused) parts.push(`<button type="button" class="copy">Copy with citations</button>`);
    meta.innerHTML = parts.join("");
    $(".copy", meta)?.addEventListener("click", async e => {
      try {
        await navigator.clipboard.writeText(ans.text);
        e.target.textContent = "Copied";
      } catch {
        e.target.textContent = "Couldn't copy";
      }
      setTimeout(() => { e.target.textContent = "Copy with citations"; }, 1800);
    });
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const CITE_RE = /\[(\d+(?:\s*[,;]\s*\d+)*)\]/g;

// QASPER text keeps the papers' inline LaTeX ("probability $\delta$"). Typeset the
// common bits so "$\delta =0.5$" reads as "δ = 0.5"; anything unknown is left as-is.
const TEX = { alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", theta: "θ", lambda: "λ", mu: "μ",
  sigma: "σ", tau: "τ", phi: "φ", pi: "π", rho: "ρ", omega: "ω", times: "×", cdot: "·", leq: "≤", geq: "≥",
  le: "≤", ge: "≥", neq: "≠", approx: "≈", rightarrow: "→", leftarrow: "←", leftrightarrow: "↔", infty: "∞",
  sum: "Σ", in: "∈", mid: "|", pm: "±", ldots: "…", lbrace: "{", rbrace: "}" };
function tex(html) {
  return html.replace(/\$([^$]{1,60})\$/g, (whole, body) => {
    let out = body
      .replace(/\\(?:mathrm|mathit|textbf|textit|text|mathbf|textsc|rm)\s*\{([^}]*)\}/g, "$1")
      .replace(/\\([a-zA-Z]+)\s*/g, (m, name) => (name in TEX ? TEX[name] : m))
      .replace(/_\{([^}]*)\}|_(\w)/g, (m, a, b) => `<sub>${a ?? b}</sub>`)
      .replace(/\^\{([^}]*)\}|\^(\w)/g, (m, a, b) => `<sup>${a ?? b}</sup>`)
      .replace(/\s*=\s*/g, " = ").replace(/[{}]/g, "").trim();
    return /\\/.test(out) ? whole : `<i class="tex">${out}</i>`;
  });
}

// Model text -> HTML: paragraphs, simple lists, **bold**, and [n] markers turned
// into footnote buttons. The text is escaped first, so nothing the model writes
// can inject markup.
function renderAnswer(raw, nSources) {
  // Footnote numbers hug the word they belong to: "100 [2]." -> "100²."
  const tight = raw.replace(/\s+(?=\[\d+(?:\s*[,;]\s*\d+)*\])/g, "");
  const withCites = esc(tight).replace(CITE_RE, (_, nums) =>
    nums.split(/[,;]/).map(s => {
      const n = Number(s.trim());
      const ok = n >= 1 && n <= nSources;
      return `<button type="button" class="fn${ok ? "" : " invalid"}" data-n="${n}"` +
             ` aria-label="${ok ? `Source ${n}` : `Source ${n} doesn't exist`}">${n}</button>`;
    }).join(""));
  const bolded = tex(withCites).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  return bolded.split(/\n{2,}/).map(block => {
    const lines = block.split("\n").filter(l => l.trim());
    if (lines.length && lines.every(l => /^\s*([-*•]|\d+\.)\s+/.test(l))) {
      return `<ul>${lines.map(l => `<li>${l.replace(/^\s*([-*•]|\d+\.)\s+/, "")}</li>`).join("")}</ul>`;
    }
    return `<p>${lines.join("<br>")}</p>`;
  }).join("");
}

function renderSources(list, chunks) {
  // Scores are only comparable within one answer, so the bar is relative to this set.
  const scores = chunks.map(c => c.score);
  const lo = Math.min(...scores), hi = Math.max(...scores);
  list.innerHTML = chunks.map((c, i) => {
    const rel = hi > lo ? (c.score - lo) / (hi - lo) : 1;
    return `
      <li class="source" id="${uid(list)}-src-${i + 1}" data-n="${i + 1}">
        <span class="source-num" aria-hidden="true">${i + 1}</span>
        <div>
          <p class="source-title">${esc(c.title)}</p>
          <p class="source-section">${esc(c.section)}</p>
          <p class="source-text">${tex(esc(c.text))}</p>
          <div class="source-foot">
            <span class="score" title="${esc(c.score_type)} score ${c.score}">
              <span class="score-bar"><i style="width:${Math.round(15 + 85 * rel)}%"></i></span>
              ${c.score_type === "rerank" ? "relevance" : "match"}
            </span>
            <button type="button" class="expand" aria-expanded="false">Show full passage</button>
            <span class="uncited-label" hidden>Not cited</span>
          </div>
        </div>
      </li>`;
  }).join("");
  list.querySelectorAll(".expand").forEach(btn => btn.addEventListener("click", () => {
    const li = btn.closest(".source");
    const open = li.classList.toggle("open");
    btn.textContent = open ? "Show less" : "Show full passage";
    btn.setAttribute("aria-expanded", String(open));
  }));
}

let uidCounter = 0;
function uid(el) {
  if (!el.dataset.uid) el.dataset.uid = `t${++uidCounter}`;
  return el.dataset.uid;
}

// After the answer is final: dim the sources nobody cited, and in each cited
// source highlight the sentence that shares the most words with the answer
// sentences that cite it. It's a heuristic, but a useful one: it shows the
// reader where to look instead of making them read 450 tokens.
function markCited(list, cited, raw, chunks, question) {
  const citedSet = new Set(cited);
  const questionWords = words(question);
  const answerSentences = splitSentences(raw);
  list.querySelectorAll(".source").forEach(li => {
    const n = Number(li.dataset.n);
    const isCited = citedSet.has(n);
    li.classList.toggle("uncited", !isCited);
    const label = li.querySelector(".uncited-label");
    label.hidden = isCited;
    label.textContent = cited.length ? "Not cited" : "Not used";
    if (!isCited) return;

    const citing = answerSentences.filter(s => citesNumber(s, n));
    const target = words((citing.length ? citing : answerSentences).join(" "));
    // Words from the answer count double; the question's words help when the
    // answer itself is tiny ("100").
    const sentences = splitSentences(chunks[n - 1].text);
    let best = -1, bestScore = 1;              // need a little more than one shared word
    sentences.forEach((s, i) => {
      let score = 0;
      words(s).forEach(t => { score += target.has(t) ? 2 : questionWords.has(t) ? 1 : 0; });
      if (score > bestScore) { best = i; bestScore = score; }
    });
    if (best < 0) return;
    // Show the evidence with a sentence of context either side; the full passage
    // (still highlighted) is one click away, so the highlight is never clipped.
    const mark = (s, i) => i === best ? `<span class="evidence swipe">${tex(esc(s))}</span>` : tex(esc(s));
    const from = Math.max(0, best - 1), to = Math.min(sentences.length, best + 2);
    const excerpt = (from > 0 ? "… " : "") + sentences.slice(from, to).map((s, j) => mark(s, from + j)).join(" ") +
                    (to < sentences.length ? " …" : "");
    const text = li.querySelector(".source-text");
    text.classList.add("has-excerpt");
    text.innerHTML = `<span class="excerpt">${excerpt}</span><span class="full">${sentences.map(mark).join(" ")}</span>`;
  });
}

function citesNumber(sentence, n) {
  for (const m of sentence.matchAll(CITE_RE)) {
    if (m[1].split(/[,;]/).some(x => Number(x.trim()) === n)) return true;
  }
  return false;
}

function splitSentences(text) {
  return text.replace(/\s+/g, " ").split(/(?<=[.!?])\s+(?=[A-Z0-9"(\[])/).filter(Boolean);
}

const STOP = new Set("the and for are was were with that this from into their they them than then also which what when where who how does did have has had been being its our out use used using can not but all any more most other such only over per".split(" "));
function words(text) {
  return new Set((text.toLowerCase().replace(CITE_RE, " ").match(/[a-z0-9][a-z0-9+\-.]*[a-z0-9]|[a-z0-9]/g) || [])
    .filter(w => w.length > 2 && !STOP.has(w)));
}

// Hovering or focusing a footnote lights up its source in the margin; clicking
// scrolls to it (and opens the margin on small screens).
function wireFootnotes(turn) {
  const setActive = n => {
    turn.querySelectorAll(".fn").forEach(b => b.classList.toggle("active", n !== null && Number(b.dataset.n) === n));
    turn.querySelectorAll(".source").forEach(li => li.classList.toggle("active", n !== null && Number(li.dataset.n) === n));
  };
  const fnFrom = e => e.target.closest?.(".fn:not(.invalid)");
  turn.addEventListener("mouseover", e => { const b = fnFrom(e); if (b) setActive(Number(b.dataset.n)); });
  turn.addEventListener("mouseout", e => { if (fnFrom(e)) setActive(null); });
  turn.addEventListener("focusin", e => { const b = fnFrom(e); if (b) setActive(Number(b.dataset.n)); });
  turn.addEventListener("focusout", e => { if (fnFrom(e)) setActive(null); });
  turn.addEventListener("click", e => {
    const b = fnFrom(e);
    if (!b) return;
    const margin = turn.querySelector(".margin");
    margin.classList.remove("collapsed");
    turn.querySelector(".margin-toggle").setAttribute("aria-expanded", "true");
    const li = turn.querySelector(`.source[data-n="${b.dataset.n}"]`);
    li?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    setActive(Number(b.dataset.n));
  });
}

// ---------------------------------------------------------------------------
// "How well does it work?" - numbers straight from results/
// ---------------------------------------------------------------------------

const pct = x => `${(100 * x).toFixed(1)}%`;

async function openBenchmark() {
  els.dialog.showModal();
  try {
    const data = await getJSON("/api/benchmark");
    els.dialogBody.innerHTML = benchmarkHTML(data);
  } catch (err) {
    els.dialogBody.innerHTML = `<p>Couldn't load the results: ${esc(err.message)}</p>`;
  }
}

function benchmarkHTML({ retrieval, answers }) {
  const label = { dense: "Dense", hybrid: "Hybrid (BM25 + dense)", hybrid_rerank: "Hybrid + rerank" };
  const scopeTable = scope => {
    const rows = retrieval.filter(r => r.scope === scope);
    if (!rows.length) return "";
    const best = Math.max(...rows.map(r => r["hit@5"]));
    return `<div class="table-wrap"><table>
      <thead><tr><th>Search</th><th>Found evidence in top 5</th><th>Top 1</th><th>MRR</th><th>Time per question</th></tr></thead>
      <tbody>${rows.map(r => `<tr class="${r["hit@5"] === best ? "best" : ""}">
        <td>${esc(label[r.mode] || r.mode)}</td>
        <td class="num"><span class="bar-cell"><span class="bar"><i style="width:${100 * r["hit@5"]}%"></i></span>${pct(r["hit@5"])}</span></td>
        <td class="num">${pct(r["hit@1"])}</td>
        <td class="num">${Number(r.mrr).toFixed(3)}</td>
        <td class="num">${Number(r.median_latency_s ?? r.latency_s ?? 0).toFixed(2)} s</td>
      </tr>`).join("")}</tbody></table></div>`;
  };

  const name = a => a.provider === "extractive" ? "No LLM (quotes sentences)"
    : `${a.model}${a.prompt_version ? `, prompt ${a.prompt_version}` : ""}`;
  const answerTable = answers.length ? `<div class="table-wrap"><table>
      <thead><tr><th>Model</th><th>Token F1</th><th>Valid citations</th><th>Declined answerable</th><th>Declined off-topic</th></tr></thead>
      <tbody>${answers.map(a => `<tr>
        <td>${esc(name(a))}</td>
        <td class="num">${Number(a.token_f1).toFixed(3)}</td>
        <td class="num">${a.citation_accuracy === "" ? "n/a" : pct(a.citation_accuracy)}</td>
        <td class="num">${pct(a.refusal_rate)}</td>
        <td class="num">${pct(a.off_topic_refusal_rate)}</td>
      </tr>`).join("")}</tbody></table></div>` : "";

  return `
    <p>Retrieval is scored on 144 questions from the QASPER dataset, where annotators marked the
    paragraphs that answer each question. A question counts as found when one of the top 5
    passages contains such a paragraph.</p>
    <h3>Searching one paper</h3>
    <p>The question is asked about a known paper, which is how QASPER was built.</p>
    ${scopeTable("paper")}
    <h3>Searching all papers</h3>
    <p>Harder: many questions, like "what baselines do they use?", never say which paper they mean.</p>
    ${scopeTable("corpus")}
    <h3>Answers</h3>
    <p>30 sampled questions plus 10 off-topic ones. Token F1 counts word overlap with the
    annotators' short answers (people agreeing with each other score about 0.61 on QASPER), and a citation is
    valid when it points to a passage the model was actually given.</p>
    ${answerTable}`;
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function autosize() {
  els.question.style.height = "auto";
  els.question.style.height = `${Math.min(els.question.scrollHeight + 2, 180)}px`;
}

els.form.addEventListener("submit", e => { e.preventDefault(); ask(); });
els.question.addEventListener("input", autosize);
els.question.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(); }
});
document.addEventListener("keydown", e => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName);
  if (e.key === "/" && !typing && !els.dialog.open) { e.preventDefault(); els.question.focus(); }
});

els.model.addEventListener("change", () => { save({ model: els.model.value }); updateStatus(); });
els.refresh.addEventListener("click", loadModels);
els.paper.addEventListener("change", () => save({ paper: els.paper.value }));
els.mode.addEventListener("change", e => save({ mode: e.target.value }));
els.k.addEventListener("change", () => {
  els.k.value = Math.max(1, Math.min(10, Number(els.k.value) || 5));
  save({ k: Number(els.k.value) });
});
document.querySelectorAll(".stepper button").forEach(b => b.addEventListener("click", () => {
  els.k.value = Math.max(1, Math.min(10, Number(els.k.value) + Number(b.dataset.step)));
  save({ k: Number(els.k.value) });
}));

$("#open-benchmark").addEventListener("click", openBenchmark);
$("#close-benchmark").addEventListener("click", () => els.dialog.close());
els.dialog.addEventListener("click", e => { if (e.target === els.dialog) els.dialog.close(); });

(async () => {
  try {
    await loadInfo();
    renderExamples();
  } catch (err) {
    els.examples.innerHTML = `<li><p>Couldn't reach the server: ${esc(err.message)}</p></li>`;
  }
  await loadModels();
})();
