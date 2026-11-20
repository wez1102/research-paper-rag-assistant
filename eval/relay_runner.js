// Browser-side runner for answer_eval.py's relay path.
//
// Why this exists: in the round-2 evaluation session the Python code ran in a
// sandbox that could not reach Ollama on the Mac, but a browser tab on the Mac
// could. This script runs inside a tab opened at http://localhost:11434 (same
// origin as Ollama, so no CORS), rebuilds each prompt from a job bundle, checks
// its SHA-1 against the one Python computed, calls /api/chat with the same options
// generate.py uses, and keeps every result in localStorage so a reload loses nothing.
//
// Usage from the console (or a browser-automation tool):
//   RELAY.load(bundle)            // {systems, chunks, jobs} from relay_*_jobs.json
//   RELAY.start("qwen3:8b")       // runs every job that has no result yet
//   RELAY.status()                // {done, total, running, last_error}
//   RELAY.results()               // {job_id: {content, sha1, ttft_s, total_s, ...}}
(() => {
  const KEY = "relay.v1";
  const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
  const S = { systems: saved.systems || {}, chunks: saved.chunks || {}, jobs: saved.jobs || [],
              results: saved.results || {}, running: false, last_error: null };
  const persist = () => localStorage.setItem(KEY, JSON.stringify(
    { systems: S.systems, chunks: S.chunks, jobs: S.jobs, results: S.results }));

  async function sha1(text) {
    const buf = await crypto.subtle.digest("SHA-1", new TextEncoder().encode(text));
    return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, "0")).join("");
  }

  // Put the shared pieces back: {{SYS:v2}} -> system prompt, {{C:id}} -> source block.
  function expand(messages) {
    return messages.map(m => ({
      role: m.role,
      content: m.content
        .replace(/\{\{SYS:([^}]+)\}\}/g, (_, v) => S.systems[v])
        .replace(/\{\{C:([^}]+)\}\}/g, (_, id) => {
          if (!(id in S.chunks)) throw new Error("missing chunk " + id);
          return S.chunks[id];
        }),
    }));
  }

  async function runJob(job, model, options) {
    const messages = expand(job.messages);
    const digest = await sha1(JSON.stringify(messages));
    if (digest !== job.sha1) throw new Error(`sha1 mismatch for ${job.id}`);
    const t0 = performance.now();
    let ttft = null, content = "", last = null;
    const resp = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      // Same request generate.py sends: streaming, no thinking, temperature 0, fixed seed.
      body: JSON.stringify({ model, stream: true, think: false, messages, options }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const msg = JSON.parse(line);
        if (msg.error) throw new Error(msg.error);
        const piece = (msg.message && msg.message.content) || "";
        if (piece && ttft === null) ttft = (performance.now() - t0) / 1000;
        content += piece;
        if (msg.done) last = msg;
      }
    }
    if (!last) throw new Error("stream ended before done=true");
    content = content.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
    return {
      content, sha1: digest, model, ttft_s: ttft, total_s: (performance.now() - t0) / 1000,
      eval_count: last.eval_count, prompt_eval_count: last.prompt_eval_count,
      total_duration_s: last.total_duration / 1e9, load_duration_s: last.load_duration / 1e9,
    };
  }

  window.RELAY = {
    load(bundle) {
      Object.assign(S.systems, bundle.systems || {});
      Object.assign(S.chunks, bundle.chunks || {});
      const known = new Set(S.jobs.map(j => j.id));
      for (const j of bundle.jobs || []) if (!known.has(j.id)) S.jobs.push(j);
      persist();
      return { jobs: S.jobs.length, chunks: Object.keys(S.chunks).length };
    },
    addChunks(obj) { Object.assign(S.chunks, obj); persist(); return Object.keys(S.chunks).length; },
    async start(model, options = { temperature: 0, seed: 42, num_predict: 400 }, filter = null) {
      if (S.running) return "already running";
      S.running = true; S.last_error = null;
      (async () => {
        for (const job of S.jobs) {
          if (S.results[job.id] || (filter && !job.id.match(filter))) continue;
          const m = job.model || model;
          try {
            S.results[job.id] = await runJob(job, m, job.options || options);
            persist();
          } catch (e) {
            S.last_error = `${job.id}: ${e.message}`;
          }
        }
        S.running = false;
      })();
      return "started";
    },
    status(filter = null) {
      const jobs = filter ? S.jobs.filter(j => j.id.match(filter)) : S.jobs;
      return { done: jobs.filter(j => S.results[j.id]).length, total: jobs.length,
               running: S.running, last_error: S.last_error };
    },
    results(filter = null) {
      const out = {};
      for (const [k, v] of Object.entries(S.results)) if (!filter || k.match(filter)) out[k] = v;
      return out;
    },
    reset() { localStorage.removeItem(KEY); location.reload(); },
  };
  return "RELAY ready";
})();
