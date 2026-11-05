# Audit: dump per-edit decision/route/rewrite_ok for a hypernet ckpt under
# the W4 b128 protocol, separating REAL WRITE hits from ROUTE/REJECT hits.
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # EasyEdit/
_CALL_CWD = os.getcwd()
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from easyeditor.editors.calm_editor import CALMEditor
from easyeditor.models.calm.calm_hparams import CALMHyperParams


def load_requests(data_path, max_edits):
    with open(data_path, encoding="utf-8") as fh:
        records = json.load(fh)[:max_edits]
    reqs = []
    for rec in records:
        rr = rec["requested_rewrite"]
        prompt = rr["prompt"].replace("{}", rr["subject"])
        tnew = rr["target_new"]["str"] if isinstance(rr["target_new"], dict) else rr["target_new"]
        ttrue = rr["target_true"]["str"] if isinstance(rr["target_true"], dict) else rr["target_true"]
        reqs.append({"prompt": prompt, "subject": rr["subject"],
                     "relation": rr.get("relation_id", "fact"),
                     "target_new": tnew, "ground_truth": ttrue})
    return reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-edits", type=int, default=500)
    ap.add_argument("--hparams", default=os.path.join(ROOT, "hparams", "CALM", "gpt2-small.yaml"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "counterfact.json"))
    args = ap.parse_args()

    if not os.path.isabs(args.ckpt):
        args.ckpt = os.path.join(_CALL_CWD, args.ckpt)
    reqs = load_requests(args.data, args.max_edits)
    hparams = CALMHyperParams.from_hparams(args.hparams)
    hparams.hypernet_ckpt = os.path.abspath(args.ckpt)
    hparams.write_budget = 128
    hparams.dual_write = True
    editor = CALMEditor.from_hparams(hparams)

    t0 = time.time()
    all_metrics, _model, extra = editor.edit_requests(reqs, sequential_edit=True, verbose=False)
    dt = time.time() - t0

    per_edit = []
    for i, m in enumerate(all_metrics):
        calm = m["calm"]
        per_edit.append({
            "idx": i,
            "decision": calm["decision"],
            "route": calm["route"],
            "rewrite_ok": bool(m["post"]["rewrite_ok"]),
        })

    write_ok = sum(1 for e in per_edit if e["decision"] == "WRITE" and e["rewrite_ok"])
    write_n = sum(1 for e in per_edit if e["decision"] == "WRITE")
    route_ok = sum(1 for e in per_edit if e["decision"] == "ROUTE" and e["rewrite_ok"])
    route_n = sum(1 for e in per_edit if e["decision"] == "ROUTE")
    rej_ok = sum(1 for e in per_edit if e["decision"] == "REJECT" and e["rewrite_ok"])
    rej_n = sum(1 for e in per_edit if e["decision"] == "REJECT")
    total_ok = sum(1 for e in per_edit if e["rewrite_ok"])

    summary = {
        "ckpt": args.ckpt,
        "total_seconds": round(dt, 2),
        "WRITE": {"n": write_n, "rewrite_ok": write_ok, "real_write_hit_rate": round(write_ok / write_n, 4) if write_n else None},
        "ROUTE": {"n": route_n, "rewrite_ok": route_ok},
        "REJECT": {"n": rej_n, "rewrite_ok": rej_ok},
        "model_level_total_ok": total_ok,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_edit": per_edit}, f, indent=2, ensure_ascii=False)
    print(f"written: {args.out}", flush=True)


if __name__ == "__main__":
    main()
