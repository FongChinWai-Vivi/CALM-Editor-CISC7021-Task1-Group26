"""Quick gate-decision diagnostic: first 30 edits, ns-proj on/off."""
import os, sys, json
def _find_repo_root(start):  # walk upwards until a directory containing .git is found
    path = start
    while True:
        if os.path.isdir(os.path.join(path, ".git")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return os.path.dirname(start)
        path = parent


ROOT = _find_repo_root(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.insert(0, os.path.join(ROOT, "EasyEdit"))
os.chdir(os.path.join(ROOT, "EasyEdit"))  # hparams use EasyEdit-relative paths

from easyeditor.editors.calm_editor import CALMEditor
from easyeditor.models.calm.calm_hparams import CALMHyperParams

NS = os.environ.get("NS", "0") == "1"
CKPT = os.path.join(ROOT, "FONG CHIN WAI", "w4_model_level_improve", "ckpt",
                    "hypernet_accum16_warm48_steps1000_gpt2-small.pt")

hparams = CALMHyperParams.from_hparams(
    os.path.join(ROOT, "EasyEdit", "hparams", "CALM", "gpt2-small.yaml"))
hparams.hypernet_ckpt = CKPT

editor = CALMEditor.from_hparams(hparams)
hn = editor.hypernetwork
if NS:
    hn.ns_proj = True
    hn.ns_keep = 8
    print("[debug] ns_proj ON (keep=8, cos=%.2f, min_keep=%.2f)"
          % (hn.ns_cos, hn.ns_min_keep))
else:
    print("[debug] ns_proj OFF")

data_path = os.path.join(ROOT, "EasyEdit", "data", "counterfact.json")
with open(data_path, "r", encoding="utf-8") as f:
    records = json.load(f)
reqs = []
for rec in records[:30]:
    rr = rec["requested_rewrite"]
    prompt = rr["prompt"].replace("{}", rr["subject"])
    target_new = rr["target_new"]["str"] if isinstance(rr["target_new"], dict) else rr["target_new"]
    target_true = rr["target_true"]["str"] if isinstance(rr["target_true"], dict) else rr["target_true"]
    reqs.append({"prompt": prompt, "subject": rr["subject"],
                 "relation": rr.get("relation_id", "fact"),
                 "target_new": target_new, "ground_truth": target_true})

metrics, _, extra = editor.edit_requests(reqs, sequential_edit=True, verbose=False)

print("\n#  subject            decision reason                    conf  conf_score pre_ok post_ok")
for i, m in enumerate(metrics):
    calm = m["calm"]
    log = editor.gate._decision_log[i]
    print("%2d  %-18s %-7s %-24s %.2f  %.2f      %s    %s"
          % (i, (reqs[i]["subject"] or "")[:18], calm["decision"],
             log.get("reason", ""), log.get("current_confidence", 0.0),
             calm.get("gate_score", 0.0),
             m["pre"]["rewrite_ok"], m["post"]["rewrite_ok"]))

print("\ndecisions:", editor.gate.get_decision_summary())
