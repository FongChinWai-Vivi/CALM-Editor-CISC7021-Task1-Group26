# -*- coding: utf-8 -*-
"""R13 -- held-out confirmation of the window effect.

R12 found that the W=100 arm (one write-out per slice) lifts slice 0-99 to e2e
0.82 / dNLL 0.14 against 0.59 / 0.53 at the frozen W=50 operating point.  That
number was produced on the same eight slices that have been used for every
previous decision in this project, so on its own it is a post-hoc selection on
the evaluation set, not a result.

This driver re-runs BOTH arms on offsets 1600-1900, which no earlier arm has
ever touched, with the R6 argument vector verbatim for the baseline arm and the
same vector with --defer-window 100 for the probe arm.  Nothing else changes:
same edits, same layers, same budget, same guard scale, same memory settings,
same paraphrase evaluation.

  R13_hold_o{o}_W50_b025   -- control, should reproduce ~0.62 if the eight
                              original slices were not lucky
  R13_hold_o{o}_W100_b025  -- treatment

Idempotent: a slice whose results json exists is skipped.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

REPO = r"C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
EE = os.path.join(REPO, "EasyEdit")

OFFSETS = [1600, 1700, 1800, 1900]
EDITS = 100
ARMS = [("W50_b025", 50), ("W100_b025", 100)]


def build(offset: int, W: int, out_dir: str):
    return [
        "-u", r"examples\w6_defer_stream.py",
        "--edits", str(EDITS), "--data-offset", str(offset),
        "--defer-window", str(W), "--layers", "4,5,6,7,8",
        "--refine-passes", "1", "--delta-budget", "0.25", "--guard-scale", "0.5",
        "--memory-read", "--memory-read-template", "instruct",
        "--memory-serve", "inject",
        "--memory-arb-mode", "ctx",
        "--survival-every", "0", "--eval-paraphrase",
        "--out-dir", out_dir,
    ]


def main() -> int:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    t00 = time.time()
    for offset in OFFSETS:
        for tag, W in ARMS:
            out_dir = r"w6run\R13_hold_o%d_%s" % (offset, tag)
            res = os.path.join(EE, out_dir, "results_defer_W%d.json" % W)
            if os.path.exists(res):
                print("[skip] %s" % out_dir, flush=True)
                continue
            log = os.path.join(REPO, "w6_rev3", "r13_%s_o%d.log" % (tag, offset))
            err = os.path.join(REPO, "w6_rev3", "r13_%s_o%d.err.log" % (tag, offset))
            print("[run ] %s" % out_dir, flush=True)
            t0 = time.time()
            with open(log, "w", encoding="utf-8", errors="replace") as fh, \
                 open(err, "w", encoding="utf-8", errors="replace") as eh:
                p = subprocess.run([PY] + build(offset, W, out_dir), cwd=EE,
                                   stdout=fh, stderr=eh, env=env)
            print("[done] %s exit=%d %.1f min" % (out_dir, p.returncode, (time.time() - t0) / 60),
                  flush=True)
    print("R13 COMPLETE %.1f min" % ((time.time() - t00) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
