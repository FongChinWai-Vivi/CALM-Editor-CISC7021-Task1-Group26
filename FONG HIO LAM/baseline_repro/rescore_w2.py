"""Re-score the free-generation rewrite metric of W2 under three strictness levels.

The saved per-case generations are reused, so this runs in seconds and needs no GPU.
Levels:
  exact        : normalised generation equals the target exactly
  early_window : the target appears within the first WINDOW tokens of the generation
  containment  : the target appears anywhere in the generation (previous definition)
"""
import json
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\lenovo\AppData\Roaming\Tencent\Marvis\User\oAN1i2aaR5ovTznkq0tCzWzHV7LU\workspace\conv_68079777edfb45928ab3278e8145ae51\temp")
from transformers import AutoTokenizer  # noqa: E402

W2 = Path(r"C:\UM - master\year 1\Sem 1\Tue - CISC7021\project\w2\FONG HIO LAM")
WINDOW = 10
tok = AutoTokenizer.from_pretrained("gpt2-xl")

PUNCT = str.maketrans("", "", string.punctuation)


def norm(text: str) -> str:
    text = text.lower().translate(PUNCT)
    return re.sub(r"\s+", " ", text).strip()


def score(gen: str, target: str):
    ng, nt = norm(gen), norm(target)
    exact = ng == nt
    ids = tok(gen, add_special_tokens=False)["input_ids"][:WINDOW]
    head = tok.decode(ids)
    early = norm(target) in norm(head)
    cont = norm(target) in norm(gen)
    return exact, early, cont


def rescore(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    key = "post_edit" if "post_edit" in d["per_case_generation"] else None
    cases = d["per_case_generation"]["post_edit"]
    n = len(cases)
    exact = early = cont = 0
    for c in cases:
        e, w, cc = score(c["generation"], c["target_new"])
        c["rewrite_hit_exact"] = e
        c["rewrite_hit_early_window"] = w
        c["rewrite_hit_containment"] = cc
        c.pop("rewrite_hit_strict", None)
        exact += e
        early += w
        cont += cc
    d["metrics_generation"]["rewrite_acc_exact"] = round(exact / n, 4)
    d["metrics_generation"]["rewrite_acc_early_window"] = round(early / n, 4)
    d["metrics_generation"]["rewrite_acc_containment"] = round(cont / n, 4)
    for stale in ("rewrite_acc_strict", "paraphrase_acc_strict", "locality_neighborhood_acc_strict",
                  "locality_attribute_acc_strict", "portability_acc_strict"):
        d["metrics_generation"].pop(stale, None)
        d["metrics_pre_edit_baseline"].pop(stale, None)
    d["metric_definitions"]["rewrite_strictness"] = (
        f"exact = normalised generation equals target; early_window = target appears in the first "
        f"{WINDOW} generated tokens; containment = target appears anywhere in the generation "
        f"(standard MQuAKE/ZSRE free-generation protocol)")
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{path.name}: n={n} exact={exact} early={early} containment={cont}")


for f in ("w2_xl_memit.json", "w2_xl_alphaedit.json"):
    p = W2 / f
    if p.exists():
        rescore(p)
    else:
        print("missing", p)
