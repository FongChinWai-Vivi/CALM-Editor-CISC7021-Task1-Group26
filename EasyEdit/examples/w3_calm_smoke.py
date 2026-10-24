"""
w3_calm_smoke.py -- CALM W3 end-to-end smoke test (gpt2-small / gpt2-xl).

Runs CALMEditor on a small stream of factual edits (single + small batch)
through the full pipeline: conflict gate -> hypernetwork WRITE / side-memory
ROUTE / gate REJECT.  Writes a JSON result file and prints a human-readable
summary.

Usage (from the repository root, with the project .venv active):

    .venv\\Scripts\\python EasyEdit\\examples\\w3_calm_smoke.py --model gpt2-small
    .venv\\Scripts\\python EasyEdit\\examples\\w3_calm_smoke.py --model gpt2-xl
    .venv\\Scripts\\python EasyEdit\\examples\\w3_calm_smoke.py --model gpt2-small --mend-backend
"""

import argparse
import json
import os
import sys
import time


def resolve_easyedit_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def main():
    parser = argparse.ArgumentParser(description="CALM W3 smoke test")
    parser.add_argument("--model", choices=["gpt2-small", "gpt2-xl"],
                        default="gpt2-small")
    parser.add_argument("--output-dir", default=None,
                        help="absolute path where the JSON result is written")
    parser.add_argument("--mend-backend", action="store_true",
                        help="additionally exercise the MEND archive backend "
                             "(gpt2-small only)")
    args = parser.parse_args()

    easyedit_root = resolve_easyedit_root()
    os.chdir(easyedit_root)
    sys.path.insert(0, easyedit_root)
    sys.path.insert(0, os.path.join(easyedit_root, "easyeditor"))

    from easyeditor.editors.calm_editor import CALMEditor
    from easyeditor.models.calm.calm_hparams import CALMHyperParams

    model_key = args.model
    yaml_path = os.path.join(easyedit_root, "hparams", "CALM", f"{model_key}.yaml")
    print(f"[CALM] smoke test | model={model_key} hparams={yaml_path}")

    # ---- edit stream (covers single + small-batch + a same-subject conflict) ----
    cases = [
        {"prompt": "The capital of France is", "ground_truth": "Paris",
         "target_new": "Lyon", "subject": "France", "relation": "capital"},
        {"prompt": "The capital of Spain is", "ground_truth": "Madrid",
         "target_new": "Seville", "subject": "Spain", "relation": "capital"},
        {"prompt": "The capital of France is", "ground_truth": "Paris",
         "target_new": "Marseille", "subject": "France", "relation": "capital"},
        {"prompt": "The currency of Japan is", "ground_truth": "yen",
         "target_new": "yen", "subject": "Japan", "relation": "currency"},
    ]

    t0 = time.time()
    editor = CALMEditor.from_hparams(CALMHyperParams.from_hparams(yaml_path))
    build_time = time.time() - t0

    t0 = time.time()
    metrics, edited_model, extra = editor.edit(
        prompts=[c["prompt"] for c in cases],
        ground_truth=[c["ground_truth"] for c in cases],
        target_new=[c["target_new"] for c in cases],
        subject=[c["subject"] for c in cases],
        relation=[c["relation"] for c in cases],
        sequential_edit=True,
        verbose=True,
    )
    edit_time = time.time() - t0

    routes = [m["calm"]["route"] for m in metrics]
    route_counts = {r: routes.count(r) for r in sorted(set(routes))}

    result = {
        "model": model_key,
        "hparams_yaml": yaml_path,
        "n_edits": len(metrics),
        "build_seconds": round(build_time, 2),
        "edit_seconds": round(edit_time, 2),
        "route_counts": route_counts,
        "gate_summary": extra["gate_summary"],
        "memory_stats": extra["memory_stats"],
        "per_edit": metrics,
        "mend_backend_check": None,
    }

    # ---- optional: exercise the MEND archive backend (gpt2-small) ----
    if args.mend_backend and model_key == "gpt2-small":
        try:
            import dataclasses
            import yaml
            from easyeditor.models.mend.mend_hparams import MENDHyperParams
            train_yaml = os.path.join(easyedit_root, "hparams", "TRAINING", "MEND", "gpt2-small.yaml")
            archive = os.path.join(easyedit_root, "results", "models", "MEND", "gpt2-small")
            with open(train_yaml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            cfg.setdefault("lr_scale", 1.0)
            cfg["alg_name"] = "MEND"
            cfg["archive"] = archive
            cfg["model_name"] = os.path.join(easyedit_root, "hugging_cache", "gpt2-small")
            cfg["tokenizer_name"] = cfg["model_name"]
            # training yaml carries extra keys (alg, dataset, train, ...) that
            # MENDHyperParams does not declare -- keep only declared fields.
            valid = {fld.name for fld in dataclasses.fields(MENDHyperParams)}
            cfg = {k: v for k, v in cfg.items() if k in valid}
            # YAML 1.1 parses "1e-4" as a string; BaseTrainer normally
            # numeric-converts before training, so do the same here.
            def _num(v):
                if isinstance(v, str):
                    try:
                        return float(v)
                    except ValueError:
                        return v
                return v
            for _f in ("lr", "edit_lr", "lr_lr", "grad_clip", "cedit", "cloc", "cbase"):
                if _f in cfg:
                    cfg[_f] = _num(cfg[_f])
            mend_hp = MENDHyperParams(**cfg)

            editor2 = CALMEditor.from_hparams(
                CALMHyperParams.from_hparams(yaml_path)
            )
            editor2.hypernetwork = None
            from easyeditor.models.mend.mend_main import MendRewriteExecutor
            editor2.mend_executor = MendRewriteExecutor()
            editor2.mend_params = mend_hp
            _, model2, _ = editor2.edit(
                prompts=[cases[0]["prompt"]], ground_truth=[cases[0]["ground_truth"]],
                target_new=[cases[0]["target_new"]], subject=[cases[0]["subject"]],
                relation=[cases[0]["relation"]], sequential_edit=True, verbose=False,
            )
            result["mend_backend_check"] = {
                "status": "ok",
                "archive": archive,
                "note": "MEND archive backend ran without error on one edit.",
            }
        except Exception as exc:  # noqa: BLE001
            result["mend_backend_check"] = {
                "status": "skipped_or_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    # ---- write JSON ----
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"w3_calm_smoke_{model_key}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[CALM] JSON result written to: {out_path}")

    # ---- summary ----
    print("\n=== CALM W3 smoke summary ===")
    print(f"model={model_key} edits={len(metrics)} "
          f"build={result['build_seconds']}s edit={result['edit_seconds']}s")
    print(f"route_counts={route_counts}")
    print(f"gate_summary={extra['gate_summary']}")
    for i, m in enumerate(metrics, 1):
        c = m["calm"]
        print(f"  {i}. route={c['route']:<12} decision={c['decision']:<8} "
              f"pre={m['pre']['answer']!r:<20} post={m['post']['answer']!r:<20} "
              f"rewrite_ok={m['post']['rewrite_ok']} conf={c['current_confidence']:.3f}")
    print(f"mend_backend_check={result['mend_backend_check']}")

    ok = len(metrics) == len(cases) and all("calm" in m for m in metrics)
    print(f"\n[CALM] pipeline status: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
