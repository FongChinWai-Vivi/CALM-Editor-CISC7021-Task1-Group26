"""demo_repl.py -- interactive CALM demo: you type, the editor answers.

Why this exists
---------------
The batch script (demo_showcase.py) prints a whole log, which reads like a
report rather than a demo. This script turns the same pipeline into a REPL:
the model is built once, then every line you type is answered live, and every
edit you issue is rendered as a single card. On camera the operator types the
prompt and the edit, so the audience sees the behaviour, not a log dump.

Commands
--------
  <free text>                      send a prompt, see what the model says now
  /edit <subject> <relation> <new> [ground_truth]
                                   run one edit on the last prompt
  /state                           what is in side memory right now
  /clear                           clear the screen (declutter before a take)
  /help  /quit

Example session (this is what the recording looks like)
-------------------------------------------------------
  prompt> The capital of France is Paris. The capital of Germany is Berlin.
          The capital of Italy is
  prompt> /edit Italy capital Milan

Nothing printed here is canned: the BEFORE and AFTER answers are decoded by
the model in front of you.

Usage
-----
  .venv\\Scripts\\python.exe EasyEdit\\examples\\demo_repl.py

  # hands-free rehearsal / fallback take: types the script for you
  .venv\\Scripts\\python.exe EasyEdit\\examples\\demo_repl.py --autotype

Recording tip: font size 20pt, window maximised at 1920x1080, then run with
--pause 0.4 so each card stays on screen long enough to read.
"""

import argparse
import difflib
import os
import re
import sys
import textwrap
import time


def resolve_easyedit_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


# ------------------------------------------------------------------ colours
class C:
    on = True
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    GREY = "\033[90m"

    @classmethod
    def w(cls, text, *codes):
        if not cls.on or not codes:
            return text
        return "".join(codes) + str(text) + cls.RESET


def rule(width, char="-"):
    print(C.w(char * width, C.GREY))


def card(title, lines, width, colour=None):
    """Draw one bordered card. Keeps a take visually to a single object."""
    inner = width - 2
    head = f" {title} "
    pad = inner - len(head)
    print(C.w("+" + head + "-" * max(0, pad) + "+", colour or C.GREY, C.BOLD))
    for ln in lines:
        for chunk in textwrap.wrap(str(ln), inner - 2) or [""]:
            print(C.w("| ", C.GREY) + chunk.ljust(inner - 2) + C.w(" |", C.GREY))
    print(C.w("+" + "-" * inner + "+", colour or C.GREY))


# ------------------------------------------------------------------ helpers
def strip_ws(x):
    return (x or "").strip()


def main():
    ap = argparse.ArgumentParser(description="interactive CALM demo")
    ap.add_argument("--model", choices=["gpt2-small", "gpt2-xl"], default="gpt2-small")
    ap.add_argument("--easyedit-root", default=None)
    ap.add_argument("--width", type=int, default=84)
    ap.add_argument("--pause", type=float, default=0.4)
    ap.add_argument("--read-template", default="prefix",
                    choices=["prefix", "instruct", "qa", "fewshot", "repeat"])
    ap.add_argument("--autotype", action="store_true",
                    help="replay a built-in session by typing it out, for rehearsal "
                         "or as a hands-free fallback take")
    ap.add_argument("--no-color", dest="color", action="store_false", default=True)
    args = ap.parse_args()
    C.on = args.color and sys.stdout.isatty() or args.color
    W = max(60, args.width)

    easyedit_root = os.path.abspath(args.easyedit_root or resolve_easyedit_root())
    if not os.path.isdir(os.path.join(easyedit_root, "easyeditor")):
        raise SystemExit(f"not an EasyEdit directory: {easyedit_root} "
                         f"(pass --easyedit-root)")
    os.chdir(easyedit_root)
    sys.path.insert(0, easyedit_root)
    sys.path.insert(0, os.path.join(easyedit_root, "easyeditor"))

    from easyeditor.editors.calm_editor import CALMEditor
    from easyeditor.models.calm.calm_hparams import CALMHyperParams

    yaml_path = os.path.join(easyedit_root, "hparams", "CALM", f"{args.model}.yaml")

    print(C.w("CALM editor -- live demo", C.BOLD, C.CYAN))
    print(C.w("type a prompt to ask, /edit to change a fact, /help for commands", C.GREY))
    print(C.w("loading model ...", C.GREY))
    t0 = time.time()
    hp = CALMHyperParams.from_hparams(yaml_path)
    hp.memory_read = True
    hp.memory_read_template = args.read_template
    editor = CALMEditor.from_hparams(hp)
    print(C.w(f"ready in {time.time() - t0:.1f}s  ({args.model}, device={editor.device}, "
              f"read template={args.read_template})", C.GREY))
    print("")

    last = {"prompt": None, "before": None, "conf": None}

    def ask(prompt):
        ans, conf = editor._current_answer_and_confidence(prompt)
        last.update(prompt=prompt, before=ans, conf=conf)
        print("")
        card("PROMPT", [C.w(prompt, C.CYAN)], W, C.CYAN)
        print(f"  {C.w('BEFORE', C.BOLD, C.RED)}  {ans.strip()!r}"
              f"   {C.w(f'confidence {conf:.2f}', C.GREY)}")
        return ans, conf

    def do_edit(subject, relation, target_new, ground_truth=None):
        if not last["prompt"]:
            print(C.w("  no prompt yet -- type a prompt first", C.YELLOW))
            return
        if subject.lower() not in last["prompt"].lower():
            words = re.findall(r"[A-Za-z]{3,}", last["prompt"])
            near = difflib.get_close_matches(subject, words, n=1, cutoff=0.5)
            tip = f" (close match: {near[0]!r})" if near else ""
            print(C.w(f"  '{subject}' is not in the prompt{tip} -- fix the spelling first",
                      C.YELLOW))
            return
        gt = ground_truth or strip_ws(last["before"])
        metrics, _, extra = editor.edit(
            prompts=[last["prompt"]], ground_truth=[gt], target_new=[target_new],
            subject=[subject], relation=[relation], sequential_edit=True, verbose=False)
        calm = metrics[0]["calm"]
        log = list(getattr(editor.gate, "_decision_log", []))
        reason = log[-1].get("reason") if log else None
        ans, info = editor.answer_with_memory(
            last["prompt"], {"prompt": last["prompt"], "target_new": target_new,
                             "subject": subject, "relation": relation,
                             "ground_truth": gt})
        decision = calm.get("decision")
        decision_col = {"WRITE": C.YELLOW, "ROUTE": C.GREEN,
                        "REJECT": C.RED}.get(decision, C.GREY)
        lines = [f"{'edit':8} ({subject}, {relation})  {strip_ws(last['before']) or '?'}"
                 f"  ->  {target_new}",
                 f"{'gate':8} " + C.w(str(decision), decision_col, C.BOLD)
                 + f"   route={calm.get('route')}   reason={reason}"]
        if info.get("channel") in ("memory", "memory_direct", "memory_arb"):
            lines.append(f"{'memory':8} stored note = {info.get('note')!r}")
        elif decision == "REJECT":
            lines.append(f"{'memory':8} nothing written (edit carried no new fact)")
        print("")
        card("EDIT", lines, W, decision_col)
        time.sleep(args.pause)
        ok = bool(info.get("final_ok"))
        if decision == "REJECT":
            mark = C.w("unchanged, nothing to fix", C.YELLOW)
        elif ok:
            mark = C.w("flipped to the edited fact", C.GREEN, C.BOLD)
        else:
            mark = C.w("still wrong", C.RED, C.BOLD)
        channel = info.get("channel")
        print(f"  {C.w('AFTER', C.BOLD, C.GREEN if ok else C.RED)}   "
              f"{ans.strip()!r}   {C.w('(channel ' + str(channel) + ')', C.GREY)}"
              f"  -> {mark}")

    def state():
        stats = editor.memory.stats() if hasattr(editor.memory, "stats") else {}
        print("")
        card("SIDE MEMORY", [f"{k}: {v}" for k, v in stats.items()] or ["(no rows)"], W)

    def help_text():
        print("")
        card("COMMANDS", [
            "<free text>                     ask the model, see what it says now",
            "/edit subj relation new [truth] change one fact on the last prompt",
            "/state                          inspect side memory",
            "/clear                          clear the screen before a take",
            "/quit                           exit",
        ], W)

    script = [
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Italy is",
        "/edit Italy capital Milan",
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Spain is",
        "/edit Spain capital Madrid",
        "/quit",
    ]

    def type_out(text):
        if not args.autotype:
            return text
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(0.012)
        sys.stdout.write("\n")
        return text

    idx = 0
    while True:
        if args.autotype:
            if idx >= len(script):
                break
            line = script[idx]
            idx += 1
            print(C.w("prompt> ", C.BOLD, C.GREEN), end="")
            line = type_out(line)
        else:
            try:
                line = input(C.w("prompt> ", C.BOLD, C.GREEN))
            except (EOFError, KeyboardInterrupt):
                print("")
                break

        line = line.strip()
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/help":
            help_text()
            continue
        if line == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue
        if line == "/state":
            state()
            continue
        if line.startswith("/edit"):
            parts = line.split()
            if len(parts) < 4:
                print(C.w("  usage: /edit <subject> <relation> <new_target> [ground_truth]",
                          C.YELLOW))
                continue
            do_edit(parts[1], parts[2], parts[3],
                    parts[4] if len(parts) > 4 else None)
            time.sleep(args.pause)
            continue
        if line.startswith("/"):
            print(C.w(f"  unknown command {line!r} -- /help", C.YELLOW))
            continue
        ask(line)
        time.sleep(args.pause)

    print("")
    print(C.w("bye", C.GREY))


if __name__ == "__main__":
    main()
