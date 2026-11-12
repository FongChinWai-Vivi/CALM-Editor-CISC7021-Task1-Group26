# Rev4 param-channel probe driver.
#
# Two levers on the 0.369 parameter-channel number:
#   A  b015    delta-budget 0.15, W=50  (fewer, cleaner writes)
#   B  b035    delta-budget 0.35, W=50  (more writes, more collateral damage)
#   C  W25b025 delta-budget 0.25, W=25  (shorter window -> less interference)
#
# Everything else identical to R6 (frozen Rev4 headline run) so arms stay
# slice-by-slice comparable. Idempotent: skips runs whose results json exists.
import os, subprocess, sys, time, json

REPO = r"C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
EE = os.path.join(REPO, "EasyEdit")
LOGDIR = os.path.join(REPO, "w6_rev3")
EDITS = os.environ.get("PROBE_EDITS", "100")

OFFSETS = [0, 100, 200, 300, 400, 500, 600, 700]
ARMS = [
    ("b015", "0.15", "50"),
    ("b035", "0.35", "50"),
    ("W25b025", "0.25", "25"),
]

def log(msg):
    line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg
    print(line, flush=True)
    with open(os.path.join(LOGDIR, "param_probe_driver.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    only = os.environ.get("PROBE_ARM", "").strip()
    for tag, budget, W in ARMS:
        if only and tag != only:
            continue
        for o in OFFSETS:
            out = r"w6run\R11_o%d_W%s_%s_inject" % (o, W, tag)
            res = os.path.join(EE, "w6run", "R11_o%d_W%s_%s_inject" % (o, W, tag),
                               "results_defer_W%s.json" % W)
            if os.path.exists(res):
                log("skip %s o%d (done)" % (tag, o))
                continue
            logf = os.path.join(LOGDIR, "r11_%s_o%d.log" % (tag, o))
            args = [PY, "-u", r"examples\w6_defer_stream.py",
                    "--edits", EDITS, "--data-offset", str(o),
                    "--defer-window", W, "--layers", "4,5,6,7,8",
                    "--refine-passes", "1", "--delta-budget", budget,
                    "--guard-scale", "0.5",
                    "--memory-read", "--memory-read-template", "instruct",
                    "--eval-paraphrase", "--out-dir", out]
            t0 = time.time()
            log("run  %s o%d -> %s (edits=%s)" % (tag, o, out, EDITS))
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            with open(logf, "w", encoding="utf-8", errors="replace") as fh:
                rc = subprocess.call(args, cwd=EE, stdout=fh, stderr=subprocess.STDOUT, env=env)
            log("done %s o%d exit=%d %.1f min" % (tag, o, rc, (time.time() - t0) / 60.0))
    log("PARAM PROBE COMPLETE")

if __name__ == "__main__":
    main()
