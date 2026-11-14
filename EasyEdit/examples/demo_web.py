"""demo_web.py -- local web UI for the CALM editor (white + blue).

Why a web UI instead of Java: this talks to the *same* Python editor object and
only replaces the pixels. A browser gives full control over the look (no OS
widget theming), and it needs nothing beyond the stdlib -- no JDK, no extra pip
packages.

Run:
    .venv\\Scripts\\python.exe EasyEdit\\examples\\demo_web.py

It prints a local URL (default http://127.0.0.1:8765) and opens your browser.
Add --no-open if you do not want that.

Endpoints (all JSON, POST unless noted):
    GET  /            the page
    POST /api/ask     {prompt}
    POST /api/edit    {prompt, subject, relation, target_new, ground_truth?}
    POST /api/reset   clear side memory
    GET  /api/state   side memory stats + whether the model is loaded

The model is built once at startup and guarded by a lock, so one request at a
time -- same behaviour as the terminal version.
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


def resolve_easyedit_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


class Engine:
    def __init__(self, root_dir, model, read_template):
        self.root_dir = root_dir
        self.model = model
        self.read_template = read_template
        self.editor = None
        self.lock = threading.RLock()
        self.ready = False

    def load(self):
        os.chdir(self.root_dir)
        sys.path.insert(0, self.root_dir)
        sys.path.insert(0, os.path.join(self.root_dir, "easyeditor"))
        from easyeditor.editors.calm_editor import CALMEditor
        from easyeditor.models.calm.calm_hparams import CALMHyperParams
        yaml_path = os.path.join(self.root_dir, "hparams", "CALM", f"{self.model}.yaml")
        hp = CALMHyperParams.from_hparams(yaml_path)
        hp.memory_read = True
        hp.memory_read_template = self.read_template
        self.editor = CALMEditor.from_hparams(hp)
        self.ready = True

    def ask(self, prompt):
        with self.lock:
            ans, conf = self.editor._current_answer_and_confidence(prompt)
        return {"answer": ans.strip(), "conf": float(conf)}

    def edit(self, prompt, subject, relation, target_new, ground_truth=None):
        with self.lock:
            before = self.ask(prompt)
            gt = ground_truth or before["answer"]
            metrics, _, _ = self.editor.edit(
                prompts=[prompt], ground_truth=[gt], target_new=[target_new],
                subject=[subject], relation=[relation],
                sequential_edit=True, verbose=False)
            calm = metrics[0]["calm"]
            log = list(getattr(self.editor.gate, "_decision_log", []))
            reason = log[-1].get("reason") if log else None
            ans, info = self.editor.answer_with_memory(
                prompt, {"prompt": prompt, "target_new": target_new,
                         "subject": subject, "relation": relation,
                         "ground_truth": gt})
        return {"before": before, "decision": calm.get("decision"),
                "route": calm.get("route"), "reason": reason,
                "note": info.get("note"), "channel": info.get("channel"),
                "after": (ans or "").strip(), "final_ok": bool(info.get("final_ok")),
                "subject": subject, "relation": relation, "target_new": target_new}

    def reset(self):
        with self.lock:
            try:
                self.editor.memory.clear()
            except Exception:
                pass
        return {"ok": True, "stats": self.stats()}

    def stats(self):
        try:
            return self.editor.memory.stats()
        except Exception:
            return {}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CALM editor</title>
<style>
  :root {
    --bg: #f4f7fd;
    --card: #ffffff;
    --blue: #2563eb;
    --blue-dark: #1d4ed8;
    --blue-soft: #eaf1ff;
    --ink: #0f172a;
    --muted: #64748b;
    --line: #e3e9f4;
    --green: #15803d;
    --green-soft: #e8f6ee;
    --amber: #b45309;
    --amber-soft: #fdf3e3;
    --red: #b91c1c;
    --red-soft: #fdeced;
    --mono: ui-monospace, "Cascadia Mono", Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.55 "Segoe UI", system-ui, -apple-system, "Helvetica Neue", sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  header {
    display: flex; align-items: center; gap: 14px;
    padding: 16px 26px; background: #fff; border-bottom: 1px solid var(--line);
    box-shadow: 0 1px 3px rgba(15, 23, 42, .04);
    position: sticky; top: 0; z-index: 5;
  }
  .logo {
    width: 30px; height: 30px; border-radius: 9px; flex: none;
    background: linear-gradient(140deg, #3b82f6, #1d4ed8);
    box-shadow: 0 4px 10px rgba(37, 99, 235, .28);
  }
  h1 { font-size: 15px; font-weight: 650; margin: 0; letter-spacing: .1px; }
  .sub { font-size: 12px; color: var(--muted); margin-top: 1px; }
  .pill {
    margin-left: auto; font-size: 12px; font-weight: 600;
    padding: 5px 12px; border-radius: 999px;
    background: var(--amber-soft); color: var(--amber);
  }
  .pill.ready { background: var(--green-soft); color: var(--green); }
  .pill.busy { background: var(--blue-soft); color: var(--blue-dark); }
  .pill.err { background: var(--red-soft); color: var(--red); }

  main {
    display: grid; grid-template-columns: 392px minmax(0, 1fr);
    gap: 20px; padding: 20px 26px 30px; align-items: start;
  }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, .04), 0 8px 22px -18px rgba(15, 23, 42, .25);
  }
  .panel { padding: 16px 18px; display: grid; gap: 18px; }
  .group { display: grid; gap: 8px; }
  .label {
    font-size: 11px; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
    color: var(--blue);
  }
  .hintline { font-size: 12px; color: var(--muted); }
  textarea, input {
    width: 100%; font-family: var(--mono); font-size: 13px; color: var(--ink);
    background: #fbfdff; border: 1px solid var(--line); border-radius: 9px;
    padding: 9px 11px; outline: none; transition: border-color .12s, box-shadow .12s;
  }
  textarea { min-height: 92px; resize: vertical; line-height: 1.5; }
  textarea:focus, input:focus {
    border-color: #93b4f7; box-shadow: 0 0 0 3px rgba(37, 99, 235, .12); background: #fff;
  }
  .grid2 { display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 6px 10px; align-items: center; }
  .grid2 span { font-size: 12.5px; color: #334155; }
  button {
    font: inherit; font-weight: 600; font-size: 13px; cursor: pointer;
    border-radius: 9px; padding: 9px 14px; border: 1px solid transparent;
    transition: background .12s, border-color .12s, transform .04s;
  }
  button:active { transform: translateY(1px); }
  .primary { background: var(--blue); color: #fff; box-shadow: 0 4px 12px -6px rgba(37,99,235,.6); }
  .primary:hover { background: var(--blue-dark); }
  .ghost { background: #fff; color: var(--blue-dark); border-color: #c7d8fb; }
  .ghost:hover { background: var(--blue-soft); }
  .quiet { background: #fff; color: #475569; border-color: var(--line); }
  .quiet:hover { background: #f6f8fc; }
  .row { display: flex; gap: 8px; }
  .row > button { flex: 1; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip {
    font-size: 12px; font-weight: 600; padding: 6px 11px; border-radius: 999px;
    background: var(--blue-soft); color: var(--blue-dark); border: 1px solid #d3e1fd;
  }
  .chip:hover { background: #dde9ff; }
  kbd {
    font-family: var(--mono); font-size: 10.5px; padding: 1px 5px; border-radius: 4px;
    background: rgba(255,255,255,.24); border: 1px solid rgba(255,255,255,.35);
  }
  .quiet kbd, .chip kbd { background: #eef2f9; border-color: #dde5f1; color: #64748b; }

  /* transcript */
  .feed { padding: 18px 20px; display: grid; gap: 14px; min-height: 300px; }
  .empty {
    border: 1.5px dashed #cddcf7; border-radius: 12px; padding: 34px 24px;
    text-align: center; color: var(--muted); background: #fcfdff;
  }
  .empty strong { display: block; color: var(--ink); font-size: 13.5px; margin-bottom: 4px; }
  .entry { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: #fff; }
  .entry.query { border-color: #d9e5fb; }
  .entry.edit { border-color: #dfe7f6; }
  .head {
    display: flex; align-items: center; gap: 8px; padding: 9px 14px;
    background: #fafcff; border-bottom: 1px solid var(--line);
  }
  .badge {
    font-size: 10.5px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase;
    padding: 2px 8px; border-radius: 6px;
  }
  .b-prompt { background: var(--blue-soft); color: var(--blue-dark); }
  .b-before { background: #f1f5f9; color: #475569; }
  .b-edit { background: #eef2ff; color: #3730a3; }
  .b-write { background: var(--amber-soft); color: var(--amber); }
  .b-route { background: var(--green-soft); color: var(--green); }
  .b-reject { background: var(--red-soft); color: var(--red); }
  .b-after { background: var(--green-soft); color: var(--green); }
  .body { padding: 11px 14px 13px; display: grid; gap: 8px; }
  .k { font-size: 11px; font-weight: 700; letter-spacing: .06em; color: var(--muted); text-transform: uppercase; width: 66px; flex: none; }
  .line { display: flex; gap: 10px; align-items: baseline; }
  .val { font-family: var(--mono); font-size: 13px; word-break: break-word; }
  .prompt-text { font-family: var(--mono); font-size: 13px; color: #1e293b; line-height: 1.55; }
  .answer { font-family: var(--mono); font-size: 13.5px; font-weight: 600; }
  .answer.bad { color: #334155; }
  .answer.good { color: var(--green); }
  .muted { color: var(--muted); font-size: 12px; }
  .verdict { font-size: 12px; font-weight: 700; padding: 3px 9px; border-radius: 999px; display: inline-block; }
  .v-flip { background: var(--green-soft); color: var(--green); }
  .v-same { background: #f1f5f9; color: #475569; }
  .v-bad { background: var(--red-soft); color: var(--red); }
  .blocked { border-color: #f4d7a4; background: #fffdf7; }
  .blocked .body { color: var(--amber); font-size: 13px; }
  .spinner {
    display: inline-block; width: 11px; height: 11px; border-radius: 50%;
    border: 2px solid #c7d8fb; border-top-color: var(--blue); animation: spin .7s linear infinite;
    vertical-align: -1px; margin-right: 6px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  footer { padding: 0 26px 26px; color: var(--muted); font-size: 12px; }
  @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <div class="logo"></div>
  <div>
    <h1>CALM editor</h1>
    <div class="sub">Macau / University of Macau demo -- type a prompt, edit one fact, every answer is decoded live by the model</div>
  </div>
  <div class="pill" id="status">loading model</div>
</header>

<main>
  <section class="card panel">
    <div class="group">
      <div class="label">1 &middot; ask the model</div>
      <textarea id="prompt" spellcheck="false">The capital of China is Beijing. Hong Kong is a SAR of China. Macau is a SAR of</textarea>
      <button class="primary" id="askBtn">Ask the model <kbd>Ctrl+Enter</kbd></button>
    </div>

    <div class="group">
      <div class="label">2 &middot; edit one fact</div>
      <div class="grid2">
        <span>Subject</span><input id="subject" value="Macau" spellcheck="false">
        <span>Relation</span><input id="relation" value="is a SAR of" spellcheck="false">
        <span>New answer</span><input id="newans" value="China" spellcheck="false">
        <span>Ground truth</span><input id="gt" placeholder="leave empty = current answer" spellcheck="false">
      </div>
      <button class="ghost" id="editBtn">Apply edit <kbd>Enter</kbd></button>
    </div>

    <div class="group">
      <div class="label">presets</div>
      <div class="chips">
        <button class="chip" data-p="macau">Macau &rarr; China (flips)</button>
        <button class="chip" data-p="lang">Macau language &rarr; Portuguese (REJECT)</button>
      </div>
    </div>

    <div class="group">
      <div class="label">session</div>
      <div class="row">
        <button class="quiet" id="scriptBtn">Run scripted demo</button>
        <button class="quiet" id="clearBtn">Clear</button>
        <button class="quiet" id="resetBtn">Reset memory</button>
      </div>
      <div class="hintline" id="hint">Clear wipes the transcript; Reset memory wipes the 64 shards.</div>
    </div>
  </section>

  <section class="card">
    <div class="feed" id="feed">
      <div class="empty" id="empty">
        <strong>Nothing yet</strong>
        Pick a preset or type your own prompt, then press Ask. The model&rsquo;s own answer appears first;
        after an edit the same prompt is asked again.
      </div>
    </div>
  </section>
</main>
<footer id="mirror"></footer>

<script>
const $ = (id) => document.getElementById(id);
const feed = $("feed"), statusPill = $("status"), hint = $("hint");

const PRESETS = {
  macau: { prompt: "The capital of China is Beijing. Hong Kong is a SAR of China. Macau is a SAR of",
           subject: "Macau", relation: "is a SAR of", newans: "China" },
  lang: { prompt: "The capital of Portugal is Lisbon. The official language of Macau is",
          subject: "Macau", relation: "language", newans: "Portuguese" }
};

function setStatus(text, cls) {
  statusPill.textContent = text;
  statusPill.className = "pill" + (cls ? " " + cls : "");
}

async function api(path, body) {
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify(body) } : {};
  const r = await fetch(path, opt);
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function clearEmpty() {
  const e = $("empty");
  if (e) e.remove();
}

function entry(cls, badge, badgeCls, rows) {
  clearEmpty();
  const d = document.createElement("div");
  d.className = "entry " + cls;
  d.innerHTML =
    '<div class="head"><span class="badge ' + badgeCls + '">' + badge + "</span></div>" +
    '<div class="body">' + rows + "</div>";
  d.style.opacity = "0";
  d.style.transform = "translateY(4px)";
  feed.appendChild(d);
  requestAnimationFrame(() => {
    d.style.transition = "opacity .18s, transform .18s";
    d.style.opacity = "1"; d.style.transform = "none";
  });
  d.scrollIntoView({ block: "nearest", behavior: "smooth" });
  return d;
}

function line(k, v) {
  return '<div class="line"><span class="k">' + k + '</span><span class="val">' + v + "</span></div>";
}

function renderAsked(prompt, res, cls) {
  entry("query", "Prompt", "b-prompt",
        '<div class="prompt-text">' + esc(prompt) + "</div>");
  const conf = res.conf.toFixed(2);
  entry(cls || "query", "Before", "b-before",
        line("answer", '<span class="answer bad">' + esc(res.answer) + "</span>") +
        line("conf", '<span class="muted">' + conf + "</span>"));
}

function renderEdited(p) {
  const gateCls = { WRITE: "b-write", ROUTE: "b-route", REJECT: "b-reject" }[p.decision] || "b-before";
  let rows = line("edit", esc("(" + p.subject + ", " + p.relation + ")  ") +
                  '<span class="muted">' + esc(p.before.answer) + "</span> &rarr; <b>" +
                  esc(p.target_new) + "</b>");
  rows += line("gate", '<span class="badge ' + gateCls + '">' + esc(p.decision) + "</span>" +
                       ' <span class="muted">route=' + esc(p.route) + " &middot; reason=" +
                       esc(p.reason) + "</span>");
  if (p.channel === "memory" || p.channel === "memory_direct" || p.channel === "memory_arb") {
    rows += line("memory", "stored note = <b>" + esc(p.note) + "</b>");
  } else if (p.decision === "REJECT") {
    rows += line("memory", '<span class="muted">nothing written, the edit carried no new fact</span>');
  }
  const flip = p.decision === "ROUTE" && p.final_ok;
  const verdict = flip ? '<span class="verdict v-flip">flipped by side memory</span>'
    : (p.decision === "REJECT" ? '<span class="verdict v-same">unchanged, already correct</span>'
      : (p.final_ok ? '<span class="verdict v-flip">flipped</span>'
        : '<span class="verdict v-bad">still wrong</span>'));
  rows += line("after", '<span class="answer ' + (p.final_ok ? "good" : "bad") + '">' +
                        esc(p.after) + "</span>");
  rows += line("", verdict + ' <span class="muted">channel ' + esc(p.channel) + "</span>");
  entry("edit", "Result", "b-after", rows);
}

async function doAsk() {
  const prompt = $("prompt").value.trim();
  if (!prompt) { hint.textContent = "Type a prompt first."; return; }
  setStatus("asking", "busy");
  try {
    const res = await api("/api/ask", { prompt });
    renderAsked(prompt, res);
    setStatus("ready", "ready");
  } catch (e) { setStatus("error", "err"); hint.textContent = e.message; }
}

async function doEdit() {
  const prompt = $("prompt").value.trim();
  const body = {
    prompt,
    subject: $("subject").value.trim(),
    relation: $("relation").value.trim(),
    target_new: $("newans").value.trim(),
    ground_truth: $("gt").value.trim() || null
  };
  if (!body.prompt) { hint.textContent = "Type a prompt first."; return; }
  if (!body.subject || !body.relation || !body.target_new) {
    hint.textContent = "Subject, relation and new answer are all required."; return;
  }
  setStatus("editing", "busy");
  try {
    const res = await api("/api/edit", body);
    if (res.blocked) {
      entry("query blocked", "Blocked", "b-write", esc(res.blocked));
      hint.textContent = res.blocked;
      setStatus("ready", "ready");
      return;
    }
    renderAsked(prompt, res.before, "query");
    renderEdited(res);
    setStatus("ready", "ready");
  } catch (e) { setStatus("error", "err"); hint.textContent = e.message; }
}

function fill(p) {
  $("prompt").value = p.prompt;
  $("subject").value = p.subject;
  $("relation").value = p.relation;
  $("newans").value = p.newans;
  $("gt").value = "";
  hint.textContent = "Fields filled. Press Apply edit.";
}

async function runScript() {
  setStatus("scripted demo", "busy");
  for (const key of ["macau", "lang"]) {
    fill(PRESETS[key]);
    await new Promise((r) => setTimeout(r, 260));
    await doEdit();
    await new Promise((r) => setTimeout(r, 420));
  }
  hint.textContent = "end of scripted demo";
  setStatus("ready", "ready");
}

$("askBtn").onclick = doAsk;
$("editBtn").onclick = doEdit;
$("scriptBtn").onclick = runScript;
$("clearBtn").onclick = () => {
  feed.innerHTML = '<div class="empty" id="empty"><strong>Nothing yet</strong>' +
    "Transcript cleared. Side memory is untouched -- press Reset memory to wipe it.</div>";
};
$("resetBtn").onclick = async () => {
  setStatus("resetting", "busy");
  try {
    const res = await api("/api/reset", {});
    hint.textContent = "side memory cleared (total_stored = " +
      (res.stats && res.stats.total_stored !== undefined ? res.stats.total_stored : "?") + ")";
    setStatus("ready", "ready");
  } catch (e) { setStatus("error", "err"); }
};
document.querySelectorAll(".chip").forEach((b) => {
  b.onclick = () => fill(PRESETS[b.dataset.p]);
});
$("prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doAsk(); }
});
["subject", "relation", "newans", "gt"].forEach((id) => {
  $(id).addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); doEdit(); }
  });
});

(async () => {
  try {
    const st = await api("/api/state");
    setStatus(st.ready ? "ready" : "loading model", st.ready ? "ready" : "");
    if (new URLSearchParams(location.search).get("demo") === "1") {
      await new Promise((r) => setTimeout(r, 400));
      runScript();
    }
  } catch (e) { setStatus("offline", "err"); }
})();
</script>
</body>
</html>
"""


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            raw = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

        def do_GET(self):
            route = urlsplit(self.path).path
            if route in ("/", "/index.html"):
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif route.startswith("/api/state"):
                self._send(200, json.dumps({"ready": engine.ready,
                                            "stats": engine.stats(),
                                            "model": engine.model}))
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            route = urlsplit(self.path).path
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                payload = {}
            try:
                if route == "/api/ask":
                    self._send(200, json.dumps(engine.ask(payload["prompt"].strip())))
                elif route == "/api/edit":
                    prompt = payload["prompt"].strip()
                    subject = payload["subject"].strip()
                    relation = payload["relation"].strip()
                    target = payload["target_new"].strip()
                    blocked = guard(prompt, subject)
                    if blocked:
                        self._send(200, json.dumps({"blocked": blocked}))
                        return
                    self._send(200, json.dumps(engine.edit(
                        prompt, subject, relation, target,
                        payload.get("ground_truth") or None)))
                elif route == "/api/reset":
                    self._send(200, json.dumps(engine.reset()))
                else:
                    self._send(404, json.dumps({"error": "not found"}))
            except Exception as exc:
                self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))

    return Handler


def guard(prompt, subject):
    import difflib
    import re
    if subject.lower() in prompt.lower():
        return None
    words = re.findall(r"[A-Za-z]{3,}", prompt)
    near = difflib.get_close_matches(subject, words, n=1, cutoff=0.5)
    if near:
        return (f"'{subject}' does not appear in the prompt "
                f"(close match: '{near[0]}'). Fix the spelling first.")
    return f"'{subject}' does not appear in the prompt. Check the subject."


def main():
    ap = argparse.ArgumentParser(description="local web UI for the CALM editor")
    ap.add_argument("--model", choices=["gpt2-small", "gpt2-xl"], default="gpt2-small")
    ap.add_argument("--easyedit-root", default=None)
    ap.add_argument("--read-template", default="prefix",
                    choices=["prefix", "instruct", "qa", "fewshot", "repeat"])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.easyedit_root or resolve_easyedit_root())
    if not os.path.isdir(os.path.join(root, "easyeditor")):
        raise SystemExit(f"not an EasyEdit directory: {root}")

    print("building editor ...", flush=True)
    t0 = time.time()
    engine = Engine(root, args.model, args.read_template)
    engine.load()
    print(f"ready in {time.time() - t0:.1f}s", flush=True)

    url = f"http://{args.host}:{args.port}/"
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"serving {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)


if __name__ == "__main__":
    main()
