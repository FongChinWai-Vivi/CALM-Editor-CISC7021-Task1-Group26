"""Rev3 diagnostics: why did the in-loop memory note fail to answer?

Loads gpt2-small through CALMEditor with the read path on, stores one note,
and prints, for the same case:

  * the exact note prefix and its token length
  * tokenizer truncation / padding sides and the configured max_length
  * what the editor actually feeds the model (truncated ids -> decoded text)
  * generation with the editor's own helper vs an untruncated prompt

No edits are applied, so this is a few seconds of GPU on a clean model.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "EasyEdit"))
sys.path.insert(0, os.path.join(ROOT, "FONG HIO LAM", "side_memory_router"))

import torch  # noqa: E402

from easyeditor.editors.calm_editor import CALMEditor  # noqa: E402
from easyeditor.models.calm.calm_hparams import CALMHyperParams  # noqa: E402

HP = os.path.join(ROOT, "EasyEdit", "hparams", "CALM", "gpt2-small.yaml")

hp = CALMHyperParams.from_hparams(HP)
hp.hypernet_alg = "closedform"
hp.cf_layers = [4, 5, 6, 7, 8]
hp.cf_l2 = 1.0
hp.cf_l2_mode = "rel"
hp.cf_z_clamp = 1.0
hp.cf_max_contexts = 10
hp.cf_v_num_grad_steps = 20
hp.defer_window = 50
hp.memory_read = True
hp.memory_read_guard = True
hp.memory_read_max_new = 5

ed = CALMEditor.from_hparams(hp)
tok = ed.tok
print("tok.truncation_side =", tok.truncation_side)
print("tok.padding_side    =", tok.padding_side)
print("hparams.max_length  =", getattr(hp, "max_length", None))

req = {"subject": "Edwin of Northumbria", "relation": "religion",
       "target_new": "Islam",
       "prompt": "Edwin of Northumbria was a king who converted to"}
print("build_memory_note   =", repr(ed._build_memory_note(
    req["subject"], req["relation"], req["target_new"])))

stored = ed._mem_store(req["subject"], req["relation"], req["target_new"])
probe = ed.memory_probe(req)
print("stored =", stored, "| probe =", probe)

pref = ed._build_memory_note(req["subject"], req["relation"], probe["note"])
composed = pref + req["prompt"]
ids = tok(composed, return_tensors="pt", truncation=True,
          max_length=getattr(hp, "max_length", 64)).input_ids
print("composed tokens (untruncated) =", tok(composed)["input_ids"].__len__())
print("fed tokens (truncated)        =", ids.size(1))
print("fed text (decoded)            =", repr(tok.decode(ids[0], skip_special_tokens=True)))
print("note survives truncation?     =", "[Islam]" in tok.decode(ids[0], skip_special_tokens=True))

ans_editor, conf = ed._current_answer_and_confidence(composed, max_new_tokens=5)
print("editor helper answer          =", repr(ans_editor), "conf", round(conf, 3))

ids2 = tok(composed, return_tensors="pt").input_ids.to(ed.device)
with torch.no_grad():
    out = ed.model.generate(ids2, max_new_tokens=5, do_sample=False,
                            pad_token_id=tok.eos_token_id)
print("untruncated answer           =",
      repr(tok.decode(out[0, ids2.size(1):], skip_special_tokens=True)))

full = ed.answer_with_memory(req["prompt"], req)
print("answer_with_memory           =", full[0], "|", {k: full[1][k] for k in
      ("channel", "base_answer", "base_ok", "injected", "reason", "memory_answer")
      if k in full[1]})
print("read_stats =", ed.read_stats)
