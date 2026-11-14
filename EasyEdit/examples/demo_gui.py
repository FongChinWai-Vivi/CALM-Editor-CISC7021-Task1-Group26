"""demo_gui.py -- point-and-click version of demo_repl.py (no terminal needed).

Two inputs only, no command syntax:

  1. Prompt box   -> press "Ask"        -> see what the model says right now
  2. Subject / Relation / New answer    -> press "Apply edit" -> see it flip

Preset buttons fill the fields for you, so a recorded take cannot be ruined by
a typo (a typo in the prompt is the one failure mode that silently kills the
demo: "Itlay" is a different token, the subject check then finds nothing).

The transcript on the right is the same evidence as the terminal version:
BEFORE, the gate decision, the memory note, AFTER.

Usage
-----
  .venv\\Scripts\\python.exe EasyEdit\\examples\\demo_gui.py
  .venv\\Scripts\\python.exe EasyEdit\\examples\\demo_gui.py --selftest   # auto-run, print, exit
"""

import argparse
import difflib
import os
import queue
import re
import sys
import threading
import time


def resolve_easyedit_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


PRESETS = [
    dict(name="Italy  (flip via memory)",
         prompt="The capital of France is Paris. The capital of Germany is Berlin. "
                "The capital of Italy is",
         subject="Italy", relation="capital", new="Milan"),
    dict(name="Spain  (gate says REJECT)",
         prompt="The capital of France is Paris. The capital of Germany is Berlin. "
                "The capital of Spain is",
         subject="Spain", relation="capital", new="Madrid"),
]


# ---------------------------------------------------------------- engine side
class Engine:
    def __init__(self, easyedit_root, model, read_template):
        self.root_dir = easyedit_root
        self.model = model
        self.read_template = read_template
        self.editor = None

    def load(self):
        os.chdir(self.root_dir)
        sys.path.insert(0, self.root_dir)
        sys.path.insert(0, os.path.join(self.root_dir, "easyeditor"))
        from easyeditor.editors.calm_editor import CALMEditor
        from easyeditor.models.calm.calm_hparams import CALMHyperParams
        yaml_path = os.path.join(self.root_dir, "hparams", "CALM", f"{self.model}.yaml")
        hparams = CALMHyperParams.from_hparams(yaml_path)
        hparams.memory_read = True
        hparams.memory_read_template = self.read_template
        self.editor = CALMEditor.from_hparams(hparams)
        return self.editor

    def ask(self, prompt):
        ans, conf = self.editor._current_answer_and_confidence(prompt)
        return {"answer": ans.strip(), "conf": conf}

    def edit(self, prompt, subject, relation, target_new, ground_truth=None):
        before = self.ask(prompt)
        gt = ground_truth or before["answer"]
        metrics, _, _ = self.editor.edit(
            prompts=[prompt], ground_truth=[gt], target_new=[target_new],
            subject=[subject], relation=[relation], sequential_edit=True, verbose=False)
        calm = metrics[0]["calm"]
        log = list(getattr(self.editor.gate, "_decision_log", []))
        reason = log[-1].get("reason") if log else None
        ans, info = self.editor.answer_with_memory(
            prompt, {"prompt": prompt, "target_new": target_new, "subject": subject,
                     "relation": relation, "ground_truth": gt})
        return {"before": before, "decision": calm.get("decision"),
                "route": calm.get("route"), "reason": reason,
                "note": info.get("note"), "channel": info.get("channel"),
                "after": (ans or "").strip(), "final_ok": bool(info.get("final_ok"))}

    def reset(self):
        try:
            self.editor.memory.clear()
            return True
        except Exception:
            return False

    def stats(self):
        try:
            return self.editor.memory.stats()
        except Exception:
            return {}


# ---------------------------------------------------------------------- GUI
def run_gui(args):
    import tkinter as tk
    from tkinter import font as tkfont

    BG = "#f4f7fd"
    PANEL = "#ffffff"
    FG = "#0f172a"
    DIM = "#64748b"
    CYAN = "#2563eb"
    GREEN = "#15803d"
    RED = "#b91c1c"
    YELLOW = "#b45309"
    BLUE = "#2563eb"

    root = tk.Tk()
    root.title("CALM editor - live demo")
    root.configure(bg=BG)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{max(1100, sw - 40)}x{max(700, sh - 70)}")
    root.minsize(1000, 660)

    mono = tkfont.Font(family="Consolas", size=12)
    mono_big = tkfont.Font(family="Consolas", size=16, weight="bold")
    ui = tkfont.Font(family="Segoe UI", size=10)
    ui_bold = tkfont.Font(family="Segoe UI", size=10, weight="bold")

    # ---- header
    head = tk.Frame(root, bg=PANEL, height=54)
    head.pack(side="top", fill="x")
    head.pack_propagate(False)
    tk.Label(head, text="CALM editor", bg=PANEL, fg=FG, font=("Segoe UI", 14, "bold")
             ).pack(side="left", padx=(18, 8), pady=12)
    tk.Label(head, text="type a prompt, then edit one fact", bg=PANEL, fg=DIM, font=ui
             ).pack(side="left", pady=12)
    status = tk.Label(head, text="loading model ...", bg=PANEL, fg=YELLOW, font=ui_bold)
    status.pack(side="right", padx=18)

    body = tk.Frame(root, bg=BG)
    body.pack(side="top", fill="both", expand=True)

    # ---- left form
    left = tk.Frame(body, bg=PANEL, width=400)
    left.pack(side="left", fill="y")
    left.pack_propagate(False)

    def section(text):
        tk.Label(left, text=text, bg=PANEL, fg=CYAN, font=ui_bold,
                 anchor="w").pack(fill="x", padx=14, pady=(11, 4))

    section("1 . ASK THE MODEL")
    prompt_box = tk.Text(left, height=5, width=40, wrap="word", font=mono,
                         bg="#fbfdff", fg=FG, insertbackground=FG,
                         relief="flat", padx=8, pady=6)
    prompt_box.pack(padx=14, fill="x")
    ask_btn = tk.Button(left, text="Ask  (Ctrl+Enter)", font=ui_bold, relief="flat",
                        bg=BLUE, fg="white", activebackground="#1d4ed8",
                        activeforeground="white", cursor="hand2")
    ask_btn.pack(padx=14, pady=6, fill="x")

    section("2 . EDIT ONE FACT  (applies to the prompt above)")
    form = tk.Frame(left, bg=PANEL)
    form.pack(padx=14, fill="x")
    entries = {}
    for i, (key, label, hint) in enumerate([
            ("subject", "Subject", "Italy"),
            ("relation", "Relation", "capital"),
            ("new", "New answer", "Milan"),
            ("gt", "Ground truth (optional)", "leave empty = current answer")]):
        tk.Label(form, text=label, bg=PANEL, fg=FG, font=ui, anchor="w"
                 ).grid(row=i, column=0, sticky="w", pady=2)
        e = tk.Entry(form, font=mono, bg="#fbfdff", fg=FG, insertbackground=FG,
                     relief="flat", width=18)
        e.grid(row=i, column=1, sticky="ew", padx=(8, 0), pady=2)
        entries[key] = e
    form.columnconfigure(1, weight=1)

    edit_btn = tk.Button(left, text="Apply edit  (Enter)", font=ui_bold, relief="flat",
                         bg=GREEN, fg="#ffffff", activebackground="#166534",
                         cursor="hand2")
    edit_btn.pack(padx=14, pady=7, fill="x")

    section("PRESETS  (fills the fields, no typing needed)")
    presets = tk.Frame(left, bg=PANEL)
    presets.pack(padx=14, fill="x")
    for p in PRESETS:
        b = tk.Button(presets, text=p["name"], font=ui, relief="flat", anchor="w",
                      bg="#eef2f9", fg=FG, activebackground="#e0e8f6",
                      cursor="hand2")
        b.pack(fill="x", pady=1)

    section("SESSION")
    sess = tk.Frame(left, bg=PANEL)
    sess.pack(padx=14, fill="x")
    script_btn = tk.Button(sess, text="Run scripted demo", font=ui, relief="flat",
                           bg="#eef2f9", fg=FG, activebackground="#e0e8f6",
                           cursor="hand2")
    script_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
    clear_btn = tk.Button(sess, text="Clear", font=ui, relief="flat", bg="#eef2f9",
                          fg=FG, activebackground="#e0e8f6", cursor="hand2")
    clear_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))
    reset_btn = tk.Button(sess, text="Reset memory", font=ui, relief="flat",
                          bg="#eef2f9", fg=FG, activebackground="#e0e8f6",
                          cursor="hand2")
    reset_btn.pack(side="left", fill="x", expand=True, padx=(8, 0))

    hint = tk.Label(left, text="", bg=PANEL, fg=DIM, font=ui, wraplength=356,
                    justify="left", anchor="w")
    hint.pack(fill="x", padx=14, pady=(10, 12))

    # ---- transcript
    right = tk.Frame(body, bg=BG)
    right.pack(side="left", fill="both", expand=True)
    out = tk.Text(right, wrap="word", font=mono, bg=BG, fg=FG, relief="flat",
                  padx=22, pady=16, insertbackground=BG, spacing1=2, spacing3=4)
    out.pack(side="left", fill="both", expand=True)
    sb = tk.Scrollbar(right, command=out.yview, width=12)
    sb.pack(side="right", fill="y")
    out.configure(yscrollcommand=sb.set, state="disabled")
    out.tag_configure("quiet", foreground=DIM, font=ui)
    out.tag_configure("hdr", foreground=CYAN, font=ui_bold)
    out.tag_configure("prompt", foreground="#475569")
    out.tag_configure("before", foreground=RED, font=mono_big)
    out.tag_configure("after_ok", foreground=GREEN, font=mono_big)
    out.tag_configure("after_bad", foreground=RED, font=mono_big)
    out.tag_configure("after_flat", foreground=YELLOW, font=mono_big)
    out.tag_configure("write", foreground=YELLOW, font=ui_bold)
    out.tag_configure("route", foreground=GREEN, font=ui_bold)
    out.tag_configure("reject", foreground=RED, font=ui_bold)
    out.tag_configure("detail", foreground=DIM, font=ui)
    out.tag_configure("verdict_ok", foreground=GREEN, font=ui_bold)
    out.tag_configure("verdict_bad", foreground=RED, font=ui_bold)
    out.tag_configure("warn", foreground=YELLOW, font=ui_bold)

    def emit(text, *tags):
        out.configure(state="normal")
        out.insert("end", text, tags or ("prompt",))
        out.see("end")
        out.configure(state="disabled")

    def emit_line(text="", *tags):
        emit(text + "\n", *tags)

    # ---- worker plumbing
    jobs, results = queue.Queue(), queue.Queue()
    engine = Engine(os.path.abspath(args.easyedit_root or resolve_easyedit_root()),
                    args.model, args.read_template)

    def worker():
        while True:
            kind, payload = jobs.get()
            try:
                if kind == "load":
                    engine.load()
                    results.put(("loaded", engine.stats()))
                elif kind == "ask":
                    results.put(("asked", (payload, engine.ask(payload))))
                elif kind == "edit":
                    results.put(("edited", (payload, engine.edit(**payload))))
                elif kind == "reset":
                    results.put(("reset", engine.reset()))
                elif kind == "script":
                    for step in payload:
                        if step[0] == "ask":
                            results.put(("asked", (step[1], engine.ask(step[1]))))
                        else:
                            results.put(("edited", (step[1], engine.edit(**step[1]))))
                        time.sleep(args.pause)
                    results.put(("script_done", None))
            except Exception as exc:      # keep the window alive, report honestly
                results.put(("error", f"{type(exc).__name__}: {exc}"))

    threading.Thread(target=worker, daemon=True).start()

    # ---- actions
    def fill(preset):
        prompt_box.delete("1.0", "end")
        prompt_box.insert("1.0", preset["prompt"])
        for k in ("subject", "relation", "new", "gt"):
            entries[k].delete(0, "end")
        entries["subject"].insert(0, preset["subject"])
        entries["relation"].insert(0, preset["relation"])
        entries["new"].insert(0, preset["new"])
        hint.configure(text="Fields filled. Press Apply edit.")
        return preset

    def guard(prompt, subject):
        """The one silent failure mode: subject not in the prompt (typos)."""
        if subject.lower() in prompt.lower():
            return None
        words = re.findall(r"[A-Za-z]{3,}", prompt)
        near = difflib.get_close_matches(subject, words, n=1, cutoff=0.5)
        if near:
            return (f"'{subject}' does not appear in the prompt "
                    f"(close match: '{near[0]}'). Fix the spelling first.")
        return f"'{subject}' does not appear in the prompt. Check the subject."

    def on_ask():
        prompt = prompt_box.get("1.0", "end").strip()
        if not prompt:
            hint.configure(text="Type a prompt first.")
            return
        status.configure(text="asking ...", fg=YELLOW)
        jobs.put(("ask", prompt))

    def on_edit():
        prompt = prompt_box.get("1.0", "end").strip()
        if not prompt:
            hint.configure(text="Type a prompt first.")
            return
        subject = entries["subject"].get().strip()
        relation = entries["relation"].get().strip()
        new = entries["new"].get().strip()
        gt = entries["gt"].get().strip() or None
        if not (subject and relation and new):
            hint.configure(text="Subject, relation and new answer are all required.")
            return
        problem = guard(prompt, subject)
        if problem:
            hint.configure(text=problem)
            emit_line("")
            emit_line("BLOCKED  " + problem, "warn")
            return
        status.configure(text="editing ...", fg=YELLOW)
        jobs.put(("edit", dict(prompt=prompt, subject=subject, relation=relation,
                               target_new=new, ground_truth=gt)))

    def on_script():
        seq = []
        for p in PRESETS:
            seq.append(("ask", p["prompt"]))
            seq.append(("edit", dict(prompt=p["prompt"], subject=p["subject"],
                                     relation=p["relation"], target_new=p["new"],
                                     ground_truth=None)))
        status.configure(text="scripted demo running ...", fg=YELLOW)
        jobs.put(("script", seq))

    def on_reset_memory():
        jobs.put(("reset", None))

    ask_btn.configure(command=on_ask)
    edit_btn.configure(command=on_edit)
    for b, p in zip(presets.winfo_children(), PRESETS):
        b.configure(command=lambda pp=p: fill(pp))
    script_btn.configure(command=on_script)
    clear_btn.configure(command=lambda: (out.configure(state="normal"),
                                         out.delete("1.0", "end"),
                                         out.configure(state="disabled")))
    reset_btn.configure(command=on_reset_memory)
    prompt_box.focus_set()
    root.bind("<Control-Return>", lambda e: on_ask())
    root.bind("<Return>", lambda e: on_edit())

    # ---- result rendering
    def render_asked(prompt, res):
        emit_line("")
        emit_line("PROMPT", "hdr")
        for line in prompt.split("\n"):
            emit_line(line, "prompt")
        emit_line(f"BEFORE   '{res['answer']}'", "before")
        emit_line(f"         confidence {res['conf']:.2f}", "detail")

    def render_edited(payload, res):
        b = res["before"]
        emit_line("")
        emit_line(f"EDIT     ({payload['subject']}, {payload['relation']})  "
                  f"'{b['answer']}'  ->  {payload['target_new']}", "hdr")
        decision = res["decision"]
        tag = {"WRITE": "write", "ROUTE": "route", "REJECT": "reject"}.get(decision, "detail")
        emit(f"{'GATE':9}{decision or '?'}", tag)
        emit_line(f"   route={res['route']}   reason={res['reason']}", "detail")
        if res["channel"] in ("memory", "memory_direct", "memory_arb"):
            emit_line(f"{'MEMORY':9}stored note = '{res['note']}'", "detail")
        elif decision == "REJECT":
            emit_line(f"{'MEMORY':9}nothing written, the edit carried no new fact", "detail")
        atag = "after_ok" if res["final_ok"] else ("after_flat" if decision == "REJECT" else "after_bad")
        emit_line(f"AFTER    '{res['after']}'", atag)
        verdict = ("flipped by side memory" if res["final_ok"] and decision == "ROUTE"
                   else "unchanged, already correct" if decision == "REJECT"
                   else "flipped" if res["final_ok"] else "still wrong")
        emit_line(f"         channel {res['channel']}   {verdict}",
                  "verdict_ok" if res["final_ok"] else "verdict_bad")

    def poll():
        try:
            while True:
                kind, data = results.get_nowait()
                if kind == "loaded":
                    status.configure(text="ready", fg=GREEN)
                    hint.configure(text="Pick a preset, or type your own prompt. "
                                        "Ctrl+Enter = Ask, Enter = Apply edit.")
                    emit_line("model loaded, side memory empty "
                              f"(shards={data.get('n_shards')}, "
                              f"max_per_shard={data.get('max_per_shard')})", "quiet")
                    emit_line("", "quiet")
                    emit_line("Session 2 of the demo: the model answers as it is, "
                              "then one fact is edited.", "quiet")
                elif kind == "asked":
                    prompt, res = data
                    render_asked(prompt, res)
                    status.configure(text="ready", fg=GREEN)
                elif kind == "edited":
                    payload, res = data
                    render_edited(payload, res)
                    status.configure(text="ready", fg=GREEN)
                elif kind == "reset":
                    emit_line("")
                    emit_line("memory cleared" if data else "memory clear failed", "quiet")
                    status.configure(text="ready", fg=GREEN)
                elif kind == "script_done":
                    status.configure(text="ready", fg=GREEN)
                    emit_line("")
                    emit_line("end of scripted demo", "quiet")
                elif kind == "error":
                    emit_line("")
                    emit_line("ERROR    " + str(data), "warn")
                    status.configure(text="error", fg=RED)
        except queue.Empty:
            pass
        root.after(60, poll)

    jobs.put(("load", None))
    root.after(60, poll)

    if args.selftest:
        def selftest():
            on_script()
            root.after(int(args.pause * 1000 * 8 + 6000), lambda: (
                print(out.get("1.0", "end")), root.destroy()))
        root.after(9000, selftest)

    if args.screenshot:
        def shoot():
            try:
                from PIL import ImageGrab
                root.update_idletasks()
                root.lift()
                root.attributes("-topmost", True)
                root.update()
                time.sleep(0.6)
                root.update_idletasks()
                gx, gy = root.winfo_rootx(), root.winfo_rooty()
                gw, gh = root.winfo_width(), root.winfo_height()
                full_probe = ImageGrab.grab(all_screens=True)
                scale = max(1.0, full_probe.size[0] / float(root.winfo_screenwidth()))
                print(f"window at {gx},{gy} {gw}x{gh} dpi_scale={scale:.3f}")
                if abs(scale - 1.0) < 0.02:
                    ImageGrab.grab(bbox=(gx, gy, gx + gw, gy + gh)).save(args.screenshot)
                else:
                    full = full_probe
                    box = (int(gx * scale), int(gy * scale),
                           int((gx + gw) * scale), int((gy + gh) * scale))
                    full.crop(box).save(args.screenshot)
                print("screenshot:", args.screenshot)
            except Exception as exc:
                print("screenshot failed:", exc)
            root.destroy()
        root.after(int(args.screenshot_delay * 1000), shoot)

    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="point-and-click CALM demo")
    ap.add_argument("--model", choices=["gpt2-small", "gpt2-xl"], default="gpt2-small")
    ap.add_argument("--easyedit-root", default=None)
    ap.add_argument("--read-template", default="prefix",
                    choices=["prefix", "instruct", "qa", "fewshot", "repeat"])
    ap.add_argument("--pause", type=float, default=1.2, help="seconds between demo steps")
    ap.add_argument("--selftest", action="store_true", help="run the scripted demo, print, exit")
    ap.add_argument("--screenshot", default=None, help="save a screenshot, then exit")
    ap.add_argument("--screenshot-delay", type=float, default=22.0)
    args = ap.parse_args()
    run_gui(args)


if __name__ == "__main__":
    main()
