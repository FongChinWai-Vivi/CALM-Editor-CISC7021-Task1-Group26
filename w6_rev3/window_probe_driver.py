# Rev4 W12 probe driver -- the window curve, sideways.
#
# R11 established the delta-budget curve at W=50 (2 flushes over 100 edits).
# W25 showed that halving the window is strictly worse.  This driver walks the
# other direction, and the design has one property worth stating up front:
#
#     with n_edits = 100, every W >= 100 produces exactly ONE tail flush.
#
# So W=100 and W=200 are behaviourally the same arm at this stream length --
# only W=100 is run, and the report says why rather than pretending a second
# data point exists.  The single lever under test is flush count 2 -> 1 at a
# fixed budget of 0.25.
#
# Everything else identical to R6 (frozen Rev4 headline) at slice level.
# Idempotent: skips runs whose results json already exists.
import os, subprocess, sys, time

REPO = r"C:\Users\07viv\Desktop\CALM-Editor-CISC7021-Task1-Group26"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
EE = os.path.join(REPO, "EasyEdit")
LOGDIR = os.path.join(REPO, "w6_rev3")
EDITS = os.environ.get("PROBE_EDITS", "100")
OFFSETS = [int(x) for x in os.environ.get("PROBE_OFFSETS", "0,100,200,300,400,500,600,700").split(",")]

ARMS = [
    ("W100b025", "0.25", "100"),
]


def log(msg):
    line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg
    print(line, flush=True)
    with open(os.path.join(LOGDIR, "window_probe_driver.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    only = os.environ.get("PROBE_ARM", "").strip()
    for tag, budget, W in ARMS:
        if only and tag != only:
            continue
        for o in OFFSETS:
            out = r"w6run\R12_o%d_W%s_%s_inject" % (o, W, tag)
            res = os.path.join(EE, "w6run", "R12_o%d_W%s_%s_inject" % (o, W, tag),
                               "results_defer_W%s.json" % W)
            if os.path.exists(res):
                log("skip %s o%d (done)" % (tag, o))
                continue
            logf = os.path.join(LOGDIR, "r12_%s_o%d.log" % (tag, o))
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
    log("WINDOW PROBE COMPLETE")


if __name__ == "__main__":
    main()
